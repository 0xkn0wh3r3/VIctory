//! The frame budget: at 30 frames a second, a frame has 33.3 ms for the
//! title's work, ours and the GPU's. Two things here:
//!
//! - A report of each frame past [`REPORT`] (at most one line a second,
//!   counting the others): how long it took, what of it was ours (copies
//!   and uploads before submissions), the GPU's, and pipeline waits, the
//!   slowest timed commands and what the guard skipped.
//! - The slow-pipeline guard: compute dispatches and mesh draws are timed
//!   on the GPU (two timestamps each, `HostList::time_start`). A frame
//!   (present to present) in which one of a pipeline's commands took
//!   longer than the limit (`VICTORY_SLOW_PIPELINE_MS`, 100 ms) is one
//!   strike for it, however many of its commands did; a frame in which
//!   all of its timed commands kept to the limit clears its strikes. After
//!   [`STRIKES`] over-limit frames in a row the pipeline is banned, said
//!   once: draws and dispatches of it recorded from then on are dropped
//!   (`State::bind`). Lists recorded before still hold its commands and
//!   run them each time they are submitted again; submission does not
//!   bind again. A submission cannot be stopped once it runs, and one the
//!   host takes 2 s for resets the device; a frame of a second stalls the
//!   title's world. The work the guard drops is the title's, so what it
//!   computes is missing (a pass's effect, or what later passes read from
//!   it): visible, logged, and better than a world stopped or lost.
//!   `VICTORY_SLOW_PIPELINES=0` turns it off.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Frames longer than this are reported (a frame at 30 a second is
/// 33.3 ms; past this one has slipped).
pub(crate) const REPORT: Duration = Duration::from_millis(50);
/// Over-limit frames in a row before a pipeline is banned.
pub(crate) const STRIKES: u32 = 3;
/// The limit for one command without `VICTORY_SLOW_PIPELINE_MS`.
const LIMIT_MS: f64 = 100.0;
/// Slowest commands kept per frame for its report.
const SLOWEST: usize = 3;

/// The guard's settings: on, and its limit for one command in ms.
pub(crate) fn guard() -> (bool, f64) {
    static SETTINGS: std::sync::OnceLock<(bool, f64)> = std::sync::OnceLock::new();
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

/// Whether commands are timed (the guard is on).
pub(crate) fn timing() -> bool {
    guard().0
}

#[derive(Default)]
struct State {
    /// Over-limit frames in a row per pipeline (none: no strikes).
    strikes: HashMap<String, u32>,
    /// Pipelines whose draws and dispatches are dropped when recorded.
    banned: HashSet<String>,
    frame: Frame,
    /// The end of the last frame, and of the last report.
    last: Option<Instant>,
    reported: Option<Instant>,
    /// Slow frames not reported since the last report.
    unreported: u32,
}

/// What a frame did so far.
#[derive(Default)]
struct Frame {
    /// Ours before submissions (copies, uploads), and the GPU's.
    prepare: Duration,
    gpu: Duration,
    submissions: u32,
    /// Waits for pipelines being made.
    waited: Duration,
    /// The slowest timed commands: (ms, pipeline).
    slowest: Vec<(f64, String)>,
    /// Draws and dispatches the guard dropped.
    skipped: u32,
    /// The pipelines timed this frame while the guard is on (not banned
    /// ones), each with its slowest command's ms.
    timed: HashMap<String, f64>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// Whether any pipeline is banned (the check on every bind is then free).
static ANY_BANNED: AtomicBool = AtomicBool::new(false);

fn state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(State::default))
}

/// Whether the guard dropped `pipeline` (and counts the draw or dispatch
/// not run). Asked while recording: commands already in a closed list
/// run whenever it is submitted.
pub(crate) fn skips(pipeline: &str) -> bool {
    if !ANY_BANNED.load(Ordering::Relaxed) {
        return false;
    }
    state(|s| {
        let banned = s.banned.contains(pipeline);
        s.frame.skipped += u32::from(banned);
        banned
    })
}

/// A timed command of `pipeline` took `ms` on the GPU. Strikes are
/// counted when the frame ends ([`presented`]).
pub(crate) fn ran(pipeline: &str, ms: f64) {
    let (on, _) = guard();
    state(|s| {
        keep_slowest(&mut s.frame.slowest, ms, pipeline);
        if !on || s.banned.contains(pipeline) {
            return;
        }
        match s.frame.timed.get_mut(pipeline) {
            Some(slowest) => *slowest = slowest.max(ms),
            None => {
                s.frame.timed.insert(pipeline.to_owned(), ms);
            }
        }
    });
}

/// A frame's strikes: one for each pipeline with a command over the limit,
/// and none left for those whose timed commands all kept to it. A
/// pipeline at [`STRIKES`] is banned. Leaves `timed` empty.
fn strike(s: &mut State, timed: &mut HashMap<String, f64>) {
    let (_, limit) = guard();
    for (pipeline, ms) in timed.drain() {
        if ms <= limit {
            s.strikes.remove(&pipeline);
            continue;
        }
        let strikes = s.strikes.entry(pipeline.clone()).or_default();
        *strikes += 1;
        if *strikes < STRIKES {
            continue;
        }
        tracing::warn!(
            pipeline,
            ms = format_args!("{ms:.0}"),
            limit_ms = limit,
            frames = STRIKES,
            "a pipeline too slow on this host is dropped from draws and dispatches \
             recorded from now on; lists recorded before still run it \
             (VICTORY_SLOW_PIPELINES=0 keeps it): what it computes is missing"
        );
        s.strikes.remove(&pipeline);
        s.banned.insert(pipeline);
        ANY_BANNED.store(true, Ordering::Relaxed);
    }
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
    state(|s| {
        s.frame.prepare += prepare;
        s.frame.gpu += gpu;
        s.frame.submissions += 1;
    });
}

/// A wait for a pipeline being made.
pub(crate) fn waited(time: Duration) {
    state(|s| s.frame.waited += time);
}

/// A frame was presented: reports it if it was slow; a new one begins.
pub(crate) fn presented() {
    let now = Instant::now();
    state(|s| {
        let mut frame = std::mem::take(&mut s.frame);
        strike(s, &mut frame.timed);
        // The next frame keeps the map's room.
        s.frame.timed = std::mem::take(&mut frame.timed);
        let Some(last) = s.last.replace(now) else {
            return;
        };
        let took = now - last;
        if took <= REPORT {
            return;
        }
        if s.reported.is_some_and(|r| now - r < Duration::from_secs(1)) {
            s.unreported += 1;
            return;
        }
        s.reported = Some(now);
        let others = std::mem::take(&mut s.unreported);
        let ms = |d: Duration| format_args!("{:.0}", d.as_secs_f64() * 1e3).to_string();
        let slowest: Vec<String> = frame
            .slowest
            .iter()
            .map(|(t, p)| format!("{t:.0} ms {p}"))
            .collect();
        tracing::warn!(
            frame_ms = ms(took),
            ours_ms = ms(frame.prepare),
            gpu_ms = ms(frame.gpu),
            pipeline_waits_ms = ms(frame.waited),
            submissions = frame.submissions,
            skipped = frame.skipped,
            ?slowest,
            other_slow_frames = others,
            "slow frame"
        );
    });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The tests share the guard: one at a time (`crate::draw`'s too).
    pub(crate) static ONE: Mutex<()> = Mutex::new(());

    /// Clears the guard and the frame (tests share the process).
    pub(crate) fn reset() {
        *STATE.lock().unwrap() = None;
        ANY_BANNED.store(false, Ordering::Relaxed);
    }

    /// A frame in which `pipeline` ran commands of `ms` each.
    pub(crate) fn frame(pipeline: &str, ms: &[f64]) {
        for &ms in ms {
            ran(pipeline, ms);
        }
        presented();
    }

    /// The guard's limit, after a reset; None where the guard is off.
    fn start() -> Option<f64> {
        reset();
        let (on, limit) = guard();
        on.then_some(limit) // else VICTORY_SLOW_PIPELINES=0 here
    }

    /// Commands over the limit in one frame are one strike, however many:
    /// a pipeline is not banned in the frame its commands first ran slow.
    #[test]
    fn slow_commands_in_one_frame_are_one_strike() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(limit) = start() else {
            return;
        };
        frame("[compute slow]", &[limit + 1.0; STRIKES as usize]);
        assert!(!skips("[compute slow]"), "one frame, one strike");
        reset();
    }

    /// A frame in which all of a pipeline's commands kept to the limit
    /// clears its strikes (a one-off hitch does not add up), and frames in
    /// which it did not run leave them.
    #[test]
    fn a_frame_within_the_limit_clears_the_strikes() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(limit) = start() else {
            return;
        };
        let (slow, quick) = (limit + 1.0, limit / 2.0);
        frame("[compute hitch]", &[slow]);
        frame("[compute hitch]", &[quick, quick]);
        for _ in 0..STRIKES - 1 {
            frame("[compute hitch]", &[slow]);
            frame("[compute other]", &[quick]);
        }
        assert!(!skips("[compute hitch]"), "slow, quick, slow, slow");
        reset();
    }

    /// [`STRIKES`] over-limit frames in a row ban a pipeline at the end of
    /// the last, not before, and only it; a frame counts as over the limit
    /// if any of its commands was.
    #[test]
    fn slow_frames_in_a_row_ban_a_pipeline() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(limit) = start() else {
            return;
        };
        let (slow, quick) = (limit + 1.0, limit / 2.0);
        for _ in 0..STRIKES - 1 {
            frame("[compute slow]", &[quick, slow, quick]);
            frame("[compute quick]", &[quick]);
        }
        ran("[compute slow]", slow);
        assert!(!skips("[compute slow]"), "not before the frame ends");
        presented();
        assert!(skips("[compute slow]"));
        assert!(!skips("[compute quick]"));
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

    /// Timed commands measure the GPU's work: after the list ran, a
    /// dispatch of a long loop took far longer than one of a short loop,
    /// each under its pipeline.
    #[test]
    fn timed_dispatches_measure_their_gpu_time() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        if !timing() {
            return; // VICTORY_SLOW_PIPELINES=0 here
        }
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
        let (root, pso) = unsafe {
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
        };
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
        // SAFETY: a plain query.
        let frequency = unsafe { host.queue.GetTimestampFrequency() }.unwrap();
        let times: HashMap<String, f64> = list.timed_ms(frequency).into_iter().collect();
        let (short, long) = (times["[compute short]"], times["[compute long]"]);
        assert!(
            long > 10.0 * short.max(0.001),
            "short {short} ms, long {long} ms"
        );
    }

    /// A frame's report starts over at each present; the first present
    /// only starts the clock.
    #[test]
    fn frames_start_over_at_each_present() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        presented();
        submitted(Duration::from_millis(5), Duration::from_millis(7));
        waited(Duration::from_millis(3));
        state(|s| {
            assert_eq!(
                (s.frame.prepare, s.frame.gpu, s.frame.waited),
                (
                    Duration::from_millis(5),
                    Duration::from_millis(7),
                    Duration::from_millis(3)
                )
            );
            assert_eq!(s.frame.submissions, 1);
        });
        presented();
        state(|s| assert_eq!(s.frame.submissions, 0, "a new frame"));
        reset();
    }
}
