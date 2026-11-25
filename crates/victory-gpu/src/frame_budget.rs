//! The frame budget: at 30 frames a second, a frame has 33.3 ms for the
//! title's work, ours and the GPU's. Two things here:
//!
//! - A report of each frame past [`REPORT`] (at most one line a second,
//!   counting the others): how long it took, what of it was ours (copies
//!   and uploads before submissions), the GPU's, and pipeline waits, and
//!   with the guard on the slowest timed commands and what it skipped.
//!   It needs no GPU timing and takes no lock: a few relaxed atomic adds a
//!   submission, and swaps at a present.
//! - The slow-pipeline guard: compute dispatches and mesh draws are timed
//!   on the GPU (two timestamps each, `HostList::time_start`). A frame
//!   (present to present) in which one of a pipeline's commands took
//!   longer than the limit (`VICTORY_SLOW_PIPELINE_MS`, 100 ms) is one
//!   strike for it, however many of its commands did; a frame in which
//!   all of its timed commands kept to the limit clears its strikes. The
//!   first [`WARMUP`] frames after a pipeline is first timed are no
//!   strikes (first-use hitches). After [`STRIKES`] over-limit frames in a
//!   row the pipeline is banned, said once: draws and dispatches of it
//!   recorded from then on are dropped (`State::bind`). Lists recorded
//!   before still hold its commands and run them each time they are
//!   submitted again; submission does not bind again. After [`COOLDOWN`]
//!   it is tried again (said once); still over the limit [`STRIKES`]
//!   frames in a row, it is banned again for twice as long (up to
//!   [`COOLDOWN_MAX`]); a frame within the limit keeps it. A submission
//!   cannot be stopped once it runs, and one the host takes 2 s for resets
//!   the device; a frame of a second stalls the title's world. The work
//!   the guard drops is the title's, so what it computes is missing (a
//!   pass's effect, or what later passes read from it): visible, logged,
//!   and better than a world stopped or lost. `VICTORY_SLOW_PIPELINES=0`
//!   turns it off: nothing is timed (no query heaps, no read-backs) and
//!   its lock is never taken.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Frames longer than this are reported (a frame at 30 a second is
/// 33.3 ms; past this one has slipped).
pub(crate) const REPORT: Duration = Duration::from_millis(50);
/// Over-limit frames in a row before a pipeline is banned.
pub(crate) const STRIKES: u32 = 3;
/// Frames after a pipeline is first timed that are no strikes for it. Its
/// first runs may pay for work the driver defers to first use (shader
/// compilation and caches) and for memory first made resident; that is
/// over within a few frames, while a pipeline slow for its own work stays
/// slow after. Ten frames are a third of a second at 30 a second, ten
/// seconds at the one a second a world takes today.
pub(crate) const WARMUP: u64 = 10;
/// How long a first ban lasts before the pipeline is tried again: long
/// enough that a pipeline slow for good costs [`STRIKES`] slow frames a
/// minute at most, short enough that one slow for a while (a heavy scene,
/// a busy host) comes back within the minute.
pub(crate) const COOLDOWN: Duration = Duration::from_secs(60);
/// The longest ban: each ban after a retry doubles the last, up to this
/// (60 s, 2, 4, 8, then 16 min): a pipeline slow for good then costs
/// [`STRIKES`] slow frames each 16 minutes, and one that becomes quick
/// (another scene) runs again within them.
pub(crate) const COOLDOWN_MAX: Duration = Duration::from_secs(16 * 60);
/// The limit for one command without `VICTORY_SLOW_PIPELINE_MS`.
const LIMIT_MS: f64 = 100.0;
/// Slowest commands kept per frame for its report.
const SLOWEST: usize = 3;

/// The guard's settings, read once: on, and its limit for one command in
/// ms.
fn settings() -> (bool, f64) {
    static SETTINGS: OnceLock<(bool, f64)> = OnceLock::new();
    *SETTINGS.get_or_init(|| {
        let on = std::env::var_os("VICTORY_SLOW_PIPELINES").is_none_or(|v| v != "0");
        let limit = std::env::var("VICTORY_SLOW_PIPELINE_MS")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|&ms| ms > 0.0)
            .unwrap_or(LIMIT_MS);
        if on {
            tracing::info!(limit_ms = limit, strikes = STRIKES, "slow-pipeline guard");
        }
        (on, limit)
    })
}

/// Whether the guard is on: 0 not read yet, 1 off, 2 on.
static ON: AtomicU8 = AtomicU8::new(0);

/// Whether commands are timed (the guard is on): one relaxed load once
/// the setting was read.
pub(crate) fn timing() -> bool {
    match ON.load(Ordering::Relaxed) {
        0 => {
            let on = settings().0;
            let _ = ON.compare_exchange(0, 1 + u8::from(on), Ordering::Relaxed, Ordering::Relaxed);
            ON.load(Ordering::Relaxed) == 2
        }
        v => v == 2,
    }
}

/// The guard's limit for one command, in ms.
pub(crate) fn limit() -> f64 {
    settings().1
}

/// What submissions and waits added to the frame so far, and the report's
/// clock: atomics, so the report takes no lock.
struct Report {
    /// Ours before submissions (copies, uploads), and the GPU's, in µs.
    prepare_us: AtomicU64,
    gpu_us: AtomicU64,
    submissions: AtomicU32,
    /// Waits for pipelines being made, in µs.
    waited_us: AtomicU64,
    /// Draws and dispatches the guard dropped.
    skipped: AtomicU32,
    /// The end of the last frame, and of the last report: µs from
    /// [`epoch`], plus one (0: none yet).
    last: AtomicU64,
    reported: AtomicU64,
    /// Slow frames not reported since the last report.
    unreported: AtomicU32,
}

/// A slow frame to report.
#[derive(Debug, PartialEq)]
struct Slow {
    took: Duration,
    prepare: Duration,
    gpu: Duration,
    submissions: u32,
    waited: Duration,
    skipped: u32,
    /// Slow frames not reported since the last report.
    others: u32,
}

/// The report's clock starts at the first frame.
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn us(d: Duration) -> u64 {
    d.as_micros().try_into().unwrap_or(u64::MAX)
}

impl Report {
    const fn new() -> Self {
        Self {
            prepare_us: AtomicU64::new(0),
            gpu_us: AtomicU64::new(0),
            submissions: AtomicU32::new(0),
            waited_us: AtomicU64::new(0),
            skipped: AtomicU32::new(0),
            last: AtomicU64::new(0),
            reported: AtomicU64::new(0),
            unreported: AtomicU32::new(0),
        }
    }

    fn submitted(&self, prepare: Duration, gpu: Duration) {
        self.prepare_us.fetch_add(us(prepare), Ordering::Relaxed);
        self.gpu_us.fetch_add(us(gpu), Ordering::Relaxed);
        self.submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// A frame ended at `now`; a new one begins. The frame if it was slow
    /// and none was reported in the last second (else it is counted). One
    /// thread presents; another racing it only blurs a report.
    fn presented(&self, now: Instant) -> Option<Slow> {
        let at = us(now.saturating_duration_since(epoch())) + 1;
        let last = self.last.swap(at, Ordering::Relaxed);
        let micros = |a: &AtomicU64| Duration::from_micros(a.swap(0, Ordering::Relaxed));
        let (prepare, gpu, waited) = (
            micros(&self.prepare_us),
            micros(&self.gpu_us),
            micros(&self.waited_us),
        );
        let submissions = self.submissions.swap(0, Ordering::Relaxed);
        let skipped = self.skipped.swap(0, Ordering::Relaxed);
        if last == 0 {
            return None;
        }
        let took = Duration::from_micros(at.saturating_sub(last));
        if took <= REPORT {
            return None;
        }
        let reported = self.reported.load(Ordering::Relaxed);
        if reported != 0 && at.saturating_sub(reported) < 1_000_000 {
            self.unreported.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.reported.store(at, Ordering::Relaxed);
        let others = self.unreported.swap(0, Ordering::Relaxed);
        Some(Slow {
            took,
            prepare,
            gpu,
            submissions,
            waited,
            skipped,
            others,
        })
    }

    #[cfg(test)]
    fn clear(&self) {
        for a in [&self.prepare_us, &self.gpu_us, &self.waited_us, &self.last, &self.reported] {
            a.store(0, Ordering::Relaxed);
        }
        for a in [&self.submissions, &self.skipped, &self.unreported] {
            a.store(0, Ordering::Relaxed);
        }
    }
}

static FRAME: Report = Report::new();

/// The slow-pipeline guard's state; only used while it is on.
#[derive(Default)]
struct Guard {
    /// Frames presented while the guard is on: the warm-up's clock.
    frames: u64,
    /// The frame each pipeline was first timed in.
    seen: HashMap<String, u64>,
    /// Over-limit frames in a row per pipeline (none: no strikes).
    strikes: HashMap<String, u32>,
    /// Pipelines whose draws and dispatches are dropped when recorded.
    banned: HashMap<String, Ban>,
    /// Pipelines tried again after a ban, with that ban's length: banned
    /// again for twice as long if still slow, kept (and forgotten) after a
    /// frame within the limit.
    retried: HashMap<String, Duration>,
    /// The pipelines timed this frame (not banned ones), each with its
    /// slowest command's ms.
    timed: HashMap<String, f64>,
    /// The frame's slowest timed commands: (ms, pipeline).
    slowest: Vec<(f64, String)>,
}

/// A ban: until when, and how long it is.
#[derive(Debug, PartialEq)]
struct Ban {
    until: Instant,
    cooldown: Duration,
}

impl Guard {
    /// A timed command of `pipeline` took `ms` on the GPU. Strikes are
    /// counted when the frame ends.
    fn ran(&mut self, pipeline: &str, ms: f64) {
        keep_slowest(&mut self.slowest, ms, pipeline);
        if self.banned.contains_key(pipeline) {
            return;
        }
        match self.timed.get_mut(pipeline) {
            Some(slowest) => *slowest = slowest.max(ms),
            None => {
                self.timed.insert(pipeline.to_owned(), ms);
            }
        }
    }

    /// A frame ended at `now`, with commands over `limit` ms or not: one
    /// strike for each pipeline with a command over it (not in its
    /// warm-up), none left for those whose timed commands all kept to it.
    /// A pipeline at [`STRIKES`] is banned; bans that ended are retried.
    /// The frame's slowest commands.
    fn presented(&mut self, now: Instant, limit: f64) -> Vec<(f64, String)> {
        let frame = self.frames;
        self.frames += 1;
        let mut timed = std::mem::take(&mut self.timed);
        for (pipeline, ms) in timed.drain() {
            let first = match self.seen.get(&pipeline) {
                Some(&first) => first,
                None => {
                    self.seen.insert(pipeline.clone(), frame);
                    frame
                }
            };
            if frame - first < WARMUP {
                continue;
            }
            if ms <= limit {
                self.strikes.remove(&pipeline);
                self.retried.remove(&pipeline);
                continue;
            }
            let strikes = self.strikes.entry(pipeline.clone()).or_default();
            *strikes += 1;
            if *strikes < STRIKES {
                continue;
            }
            self.strikes.remove(&pipeline);
            let cooldown = match self.retried.remove(&pipeline) {
                Some(last) => {
                    let cooldown = (2 * last).min(COOLDOWN_MAX);
                    tracing::warn!(
                        pipeline,
                        ms = format_args!("{ms:.0}"),
                        limit_ms = limit,
                        retry_s = cooldown.as_secs(),
                        "a pipeline tried again is still too slow on this host: dropped again, \
                         for longer"
                    );
                    cooldown
                }
                None => {
                    tracing::warn!(
                        pipeline,
                        ms = format_args!("{ms:.0}"),
                        limit_ms = limit,
                        frames = STRIKES,
                        retry_s = COOLDOWN.as_secs(),
                        "a pipeline too slow on this host is dropped from draws and dispatches \
                         recorded from now on, until it is tried again; lists recorded before \
                         still run it (VICTORY_SLOW_PIPELINES=0 keeps it): what it computes is \
                         missing"
                    );
                    COOLDOWN
                }
            };
            let until = now + cooldown;
            self.banned.insert(pipeline, Ban { until, cooldown });
        }
        // The next frame keeps the map's room.
        self.timed = timed;
        let retried = &mut self.retried;
        self.banned.retain(|pipeline, ban| {
            if ban.until > now {
                return true;
            }
            tracing::info!(
                pipeline,
                banned_s = ban.cooldown.as_secs(),
                "a pipeline dropped as too slow is tried again"
            );
            retried.insert(pipeline.clone(), ban.cooldown);
            false
        });
        std::mem::take(&mut self.slowest)
    }
}

static GUARD: Mutex<Option<Guard>> = Mutex::new(None);
/// Whether any pipeline is banned (the check on every bind is then free).
static ANY_BANNED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
thread_local! {
    /// Times this thread took the guard's lock.
    static LOCKED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

fn guard<R>(f: impl FnOnce(&mut Guard) -> R) -> R {
    #[cfg(test)]
    LOCKED.with(|n| n.set(n.get() + 1));
    let mut held = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let g = held.get_or_insert_with(Guard::default);
    let r = f(g);
    ANY_BANNED.store(!g.banned.is_empty(), Ordering::Relaxed);
    r
}

/// Whether the guard dropped `pipeline` (and counts the draw or dispatch
/// not run). Asked while recording: commands already in a closed list
/// run whenever it is submitted.
pub(crate) fn skips(pipeline: &str) -> bool {
    if !ANY_BANNED.load(Ordering::Relaxed) {
        return false;
    }
    let banned = guard(|g| g.banned.contains_key(pipeline));
    if banned {
        FRAME.skipped.fetch_add(1, Ordering::Relaxed);
    }
    banned
}

/// A list's timed commands ran: each one's pipeline and GPU time in ms.
/// Strikes are counted when the frame ends ([`presented`]).
pub(crate) fn ran(timed: Vec<(String, f64)>) {
    if timed.is_empty() || !timing() {
        return;
    }
    guard(|g| {
        for (pipeline, ms) in timed {
            g.ran(&pipeline, ms);
        }
    });
}

/// Keeps the [`SLOWEST`] slowest commands of a frame, slowest first.
fn keep_slowest(slowest: &mut Vec<(f64, String)>, ms: f64, pipeline: &str) {
    if slowest.len() < SLOWEST || slowest.last().is_some_and(|l| l.0 < ms) {
        slowest.push((ms, pipeline.to_owned()));
        slowest.sort_by(|a, b| b.0.total_cmp(&a.0));
        slowest.truncate(SLOWEST);
    }
}

/// A submission: our work before it, and the GPU's.
pub(crate) fn submitted(prepare: Duration, gpu: Duration) {
    FRAME.submitted(prepare, gpu);
}

/// A wait for a pipeline being made.
pub(crate) fn waited(time: Duration) {
    FRAME.waited_us.fetch_add(us(time), Ordering::Relaxed);
}

/// A frame was presented: reports it if it was slow; a new one begins.
pub(crate) fn presented() {
    let now = Instant::now();
    let slowest = if timing() {
        guard(|g| g.presented(now, limit()))
    } else {
        Vec::new()
    };
    let Some(slow) = FRAME.presented(now) else {
        return;
    };
    let ms = |d: Duration| format_args!("{:.0}", d.as_secs_f64() * 1e3).to_string();
    let slowest: Vec<String> = slowest
        .iter()
        .map(|(t, p)| format!("{t:.0} ms {p}"))
        .collect();
    tracing::warn!(
        frame_ms = ms(slow.took),
        ours_ms = ms(slow.prepare),
        gpu_ms = ms(slow.gpu),
        pipeline_waits_ms = ms(slow.waited),
        submissions = slow.submissions,
        skipped = slow.skipped,
        ?slowest,
        other_slow_frames = slow.others,
        "slow frame"
    );
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::{ID3D12PipelineState, ID3D12RootSignature};
    use windows::core::Interface;

    /// The tests share the guard: one at a time (`crate::draw`'s too).
    pub(crate) static ONE: Mutex<()> = Mutex::new(());

    /// Clears the guard and the frame (tests share the process).
    pub(crate) fn reset() {
        *GUARD.lock().unwrap_or_else(|e| e.into_inner()) = None;
        ANY_BANNED.store(false, Ordering::Relaxed);
        FRAME.clear();
    }

    /// The guard on or off until dropped (then as before); hold [`ONE`].
    pub(crate) struct Switched(u8);

    pub(crate) fn switch(on: bool) -> Switched {
        timing();
        Switched(ON.swap(1 + u8::from(on), Ordering::Relaxed))
    }

    impl Drop for Switched {
        fn drop(&mut self) {
            ON.store(self.0, Ordering::Relaxed);
        }
    }

    /// Bans `pipeline` through the guard's calls: its warm-up, then
    /// [`STRIKES`] frames over the limit.
    pub(crate) fn ban(pipeline: &str) {
        for _ in 0..WARMUP + u64::from(STRIKES) {
            ran(vec![(pipeline.to_owned(), limit() + 1.0)]);
            presented();
        }
    }

    const LIMIT: f64 = 100.0;
    const SLOW: f64 = LIMIT + 1.0;
    const QUICK: f64 = LIMIT / 2.0;

    /// A frame of `g` ending at `now` in which `pipeline` ran commands of
    /// `ms` each.
    fn frame_at(g: &mut Guard, now: Instant, pipeline: &str, ms: &[f64]) {
        for &ms in ms {
            g.ran(pipeline, ms);
        }
        g.presented(now, LIMIT);
    }

    /// A guard past `pipeline`'s warm-up, its frames quick.
    fn warm(pipeline: &str, now: Instant) -> Guard {
        let mut g = Guard::default();
        for _ in 0..WARMUP {
            frame_at(&mut g, now, pipeline, &[QUICK]);
        }
        g
    }

    /// The first [`WARMUP`] frames after a pipeline is first timed are no
    /// strikes, however slow; the frames after are.
    #[test]
    fn no_strikes_during_the_warm_up() {
        let now = Instant::now();
        let mut g = Guard::default();
        frame_at(&mut g, now, "[compute other]", &[QUICK]);
        for _ in 0..WARMUP {
            frame_at(&mut g, now, "[compute first]", &[SLOW, SLOW]);
        }
        assert!(g.strikes.is_empty() && g.banned.is_empty(), "warming up");
        for _ in 0..STRIKES - 1 {
            frame_at(&mut g, now, "[compute first]", &[SLOW]);
        }
        assert!(!g.banned.contains_key("[compute first]"));
        frame_at(&mut g, now, "[compute first]", &[SLOW]);
        assert!(g.banned.contains_key("[compute first]"), "after the warm-up");
    }

    /// Commands over the limit in one frame are one strike, however many:
    /// a pipeline is not banned in the frame its commands first ran slow.
    #[test]
    fn slow_commands_in_one_frame_are_one_strike() {
        let now = Instant::now();
        let mut g = warm("[compute slow]", now);
        frame_at(&mut g, now, "[compute slow]", &[SLOW; STRIKES as usize]);
        assert!(!g.banned.contains_key("[compute slow]"), "one frame, one strike");
        assert_eq!(g.strikes["[compute slow]"], 1);
    }

    /// A frame in which all of a pipeline's commands kept to the limit
    /// clears its strikes (a one-off hitch does not add up), and frames in
    /// which it did not run leave them.
    #[test]
    fn a_frame_within_the_limit_clears_the_strikes() {
        let now = Instant::now();
        let mut g = warm("[compute hitch]", now);
        frame_at(&mut g, now, "[compute hitch]", &[SLOW]);
        frame_at(&mut g, now, "[compute hitch]", &[QUICK, QUICK]);
        for _ in 0..STRIKES - 1 {
            frame_at(&mut g, now, "[compute hitch]", &[SLOW]);
            frame_at(&mut g, now, "[compute other]", &[QUICK]);
        }
        assert!(!g.banned.contains_key("[compute hitch]"), "slow, quick, slow, slow");
    }

    /// [`STRIKES`] over-limit frames in a row ban a pipeline at the end of
    /// the last, not before, and only it, for [`COOLDOWN`]; a frame counts
    /// as over the limit if any of its commands was.
    #[test]
    fn slow_frames_in_a_row_ban_a_pipeline() {
        let now = Instant::now();
        let mut g = warm("[compute slow]", now);
        for _ in 0..STRIKES - 1 {
            frame_at(&mut g, now, "[compute slow]", &[QUICK, SLOW, QUICK]);
            frame_at(&mut g, now, "[compute quick]", &[QUICK]);
        }
        g.ran("[compute slow]", SLOW);
        assert!(!g.banned.contains_key("[compute slow]"), "not before the frame ends");
        g.presented(now, LIMIT);
        let until = now + COOLDOWN;
        let ban = Ban {
            until,
            cooldown: COOLDOWN,
        };
        assert_eq!(g.banned.get("[compute slow]"), Some(&ban));
        assert!(!g.banned.contains_key("[compute quick]"));
    }

    /// A banned pipeline is tried again after its cooldown; still slow
    /// [`STRIKES`] frames in a row, it is banned again for twice as long,
    /// up to [`COOLDOWN_MAX`].
    #[test]
    fn a_ban_ends_and_a_pipeline_still_slow_is_banned_longer() {
        let p = "[compute slow]";
        let mut now = Instant::now();
        let mut g = warm(p, now);
        let mut cooldown = COOLDOWN;
        for _ in 0..8 {
            for _ in 0..STRIKES {
                frame_at(&mut g, now, p, &[SLOW]);
            }
            assert_eq!(g.banned[p].cooldown, cooldown);
            // Lists recorded before the ban still run it: no strikes.
            frame_at(&mut g, now + cooldown / 2, p, &[SLOW]);
            frame_at(&mut g, now + cooldown - Duration::from_millis(1), p, &[SLOW]);
            assert!(g.banned.contains_key(p), "within the cooldown");
            assert!(g.strikes.is_empty());
            now += cooldown;
            g.presented(now, LIMIT);
            assert!(!g.banned.contains_key(p), "tried again after {cooldown:?}");
            assert_eq!(g.retried[p], cooldown);
            cooldown = (2 * cooldown).min(COOLDOWN_MAX);
        }
        assert_eq!(cooldown, COOLDOWN_MAX);
    }

    /// A pipeline tried again that keeps to the limit stays allowed, and a
    /// later ban of it starts again at [`COOLDOWN`].
    #[test]
    fn a_pipeline_quick_when_tried_again_stays_allowed() {
        let p = "[compute better]";
        let mut now = Instant::now();
        let mut g = warm(p, now);
        for _ in 0..STRIKES {
            frame_at(&mut g, now, p, &[SLOW]);
        }
        now += COOLDOWN;
        g.presented(now, LIMIT);
        assert!(!g.banned.contains_key(p));
        frame_at(&mut g, now, p, &[SLOW]);
        frame_at(&mut g, now, p, &[QUICK]);
        assert!(g.retried.is_empty() && g.strikes.is_empty(), "kept");
        for _ in 0..100 {
            now += Duration::from_secs(1);
            frame_at(&mut g, now, p, &[QUICK]);
        }
        assert!(!g.banned.contains_key(p));
        for _ in 0..STRIKES {
            frame_at(&mut g, now, p, &[SLOW]);
        }
        assert_eq!(g.banned[p].cooldown, COOLDOWN, "a new ban");
    }

    /// The guard through its calls: a ban is seen by `skips`.
    #[test]
    fn a_banned_pipeline_is_skipped() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let _on = switch(true);
        reset();
        ban("[compute global]");
        assert!(skips("[compute global]"));
        assert!(!skips("[compute other]"));
        reset();
    }

    /// With the guard off nothing is timed and its lock is never taken:
    /// not on submissions, waits, presents or binds, and a list asked to
    /// time a command makes no query heap.
    #[test]
    fn the_guard_off_times_nothing_and_takes_no_lock() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let _off = switch(false);
        reset();
        let locked = LOCKED.with(|n| n.get());
        for _ in 0..WARMUP + u64::from(STRIKES) {
            submitted(Duration::from_millis(1), Duration::from_millis(1));
            waited(Duration::from_millis(1));
            ran(vec![("[compute off]".into(), 1e6)]);
            assert!(!skips("[compute off]"));
            presented();
        }
        assert_eq!(LOCKED.with(|n| n.get()), locked, "the guard's lock taken");
        assert!(GUARD.lock().unwrap().is_none(), "the guard's state made");
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let list = host.new_list().unwrap();
        assert_eq!(list.time_start(), None);
        list.close();
        host.execute(&[&list]);
        assert!(list.timestamp_heap().is_none(), "a query heap made");
        assert_eq!(LOCKED.with(|n| n.get()), locked, "the guard's lock taken");
    }

    /// With the guard on, a list makes its query heap at its first timed
    /// command, not before (a list with none never does), and keeps it
    /// across resets.
    #[test]
    fn lists_make_their_query_heap_once_when_they_time_a_command() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let _on = switch(true);
        let list = host.new_list().unwrap();
        list.close();
        host.execute(&[&list]);
        assert!(list.timestamp_heap().is_none(), "no timed command, no heap");
        let mut heaps = Vec::new();
        for _ in 0..3 {
            list.reset();
            let k = list.time_start().expect("timed");
            list.time_end(k, "[compute heap]");
            list.close();
            host.execute(&[&list]);
            heaps.push(list.timestamp_heap().expect("a heap").as_raw());
        }
        assert!(heaps.iter().all(|&h| h == heaps[0]), "the same heap after resets");
        reset();
    }

    /// A frame keeps its slowest commands, slowest first.
    #[test]
    fn a_frame_keeps_its_slowest_commands() {
        let mut slowest = Vec::new();
        for (ms, p) in [(5.0, "a"), (1.0, "b"), (9.0, "c"), (3.0, "d"), (0.5, "e")] {
            keep_slowest(&mut slowest, ms, p);
        }
        let order: Vec<&str> = slowest.iter().map(|(_, p)| p.as_str()).collect();
        assert_eq!(order, ["c", "a", "d"]);
    }

    /// A compute pipeline that loops `n` (a root constant) times a thread
    /// and writes a UAV (root parameter 0), 64 threads a group.
    fn looping(host: &crate::host::Host) -> (ID3D12RootSignature, ID3D12PipelineState) {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;
        const SOURCE: &str = r#"
#define RS "UAV(u0), RootConstants(num32BitConstants=1, b0)"
RWByteAddressBuffer o : register(u0);
cbuffer c : register(b0) { uint n; }
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    for (uint i = 0; i < n; i++) {
        x = x * 1664525 + 1013904223;
    }
    o.Store(id.x * 4, x);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names outlive the call.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("budget"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes, a shader with its root signature.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        }
    }

    /// Timed commands measure the GPU's work: after the list ran, a
    /// dispatch of a long loop took far longer than one of a short loop,
    /// each under its pipeline.
    #[test]
    fn timed_dispatches_measure_their_gpu_time() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let _on = switch(true);
        let (root, pso) = looping(host);
        let groups = 1024u32;
        let (out, _) = host.shared_buffer(u64::from(groups) * 64 * 4).unwrap();
        let list = host.new_list().unwrap();
        for (loops, pipeline) in [(1u32, "[compute short]"), (1 << 16, "[compute long]")] {
            let k = list.time_start().expect("timed");
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list
                    .SetComputeRootUnorderedAccessView(0, out.GetGPUVirtualAddress());
                list.list.SetComputeRoot32BitConstant(1, loops, 0);
                list.list.Dispatch(groups, 1, 1);
            }
            list.time_end(k, pipeline);
        }
        list.close();
        host.execute(&[&list]);
        let times: HashMap<String, f64> =
            list.timed_ms(host.timestamp_frequency()).into_iter().collect();
        let (short, long) = (times["[compute short]"], times["[compute long]"]);
        assert!(
            long > 10.0 * short.max(0.001),
            "short {short} ms, long {long} ms"
        );
        reset();
    }

    /// A frame's report starts over at each present; the first present
    /// only starts the clock; a slow frame is reported, and the slow ones
    /// within a second of it only counted.
    #[test]
    fn frames_start_over_at_each_present() {
        let r = Report::new();
        let t = epoch() + Duration::from_secs(1);
        let ms = Duration::from_millis;
        assert_eq!(r.presented(t), None);
        r.submitted(ms(5), ms(7));
        r.waited_us.fetch_add(3000, Ordering::Relaxed);
        let slow = Slow {
            took: ms(60),
            prepare: ms(5),
            gpu: ms(7),
            submissions: 1,
            waited: ms(3),
            skipped: 0,
            others: 0,
        };
        assert_eq!(r.presented(t + ms(60)), Some(slow));
        assert_eq!(r.presented(t + ms(80)), None, "quick");
        assert_eq!(r.presented(t + ms(200)), None, "slow, within the second");
        let slow = r.presented(t + ms(1100)).expect("slow");
        assert_eq!((slow.submissions, slow.others), (0, 1));
    }
}
