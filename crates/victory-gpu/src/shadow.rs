//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (in video memory the CPU writes through the PCI BAR where
//! the host has GPU upload heaps, else in system memory; see
//! `docs/gpu-memory.md`), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).
//!
//! With `VICTORY_GPU_UPLOAD=1`, copies of buffers made without unordered
//! access are in GPU upload heaps (video memory the CPU writes, not reads).
//! With `VICTORY_GPU_UPLOAD_WRITES=1` too, so are copies of the others, each
//! with a mirror in system memory that the CPU reads and writes ([`Vram`]):
//! refreshes write video memory and mark its pages stale in the mirror, and
//! pages come back from video memory before the CPU reads them ([`pull`],
//! [`Shadow::freshen`]).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

fn merge_writeback(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let mut changed = 0;
    for ((title_byte, host_byte), old_byte) in title.iter_mut().zip(host).zip(old) {
        let mask = 0u8.wrapping_sub(u8::from(*host_byte != *old_byte));
        let delta = (*title_byte ^ *host_byte) & mask;
        *title_byte ^= delta;
        *host_byte = *title_byte;
        changed |= delta;
    }
    changed != 0
}

#[test]
fn writeback_merge_matches_bytewise_oracle() {
    for len in [0, 1, 63, 64, 65, 193, 65_536] {
        let old = vec![0x11; len];
        for stride in [1, 5, 67, len + 1] {
            let mut title = old.clone();
            let mut host = old.clone();
            for index in (0..len).step_by(3) {
                title[index] = 0x22;
            }
            for index in (0..len).step_by(stride) {
                host[index] = if index % 2 == 0 { 0x22 } else { 0x33 };
            }
            let expected: Vec<_> = host
                .iter()
                .zip(&old)
                .zip(&title)
                .map(|((&gpu, &previous), &cpu)| if gpu == previous { cpu } else { gpu })
                .collect();
            let changed = expected != title;
            assert_eq!(merge_writeback(&mut title, &mut host, &old), changed);
            assert_eq!(title, expected);
            assert_eq!(host, expected);
        }
    }
}
/// The most marked pages that go back with one read of title memory
/// ([`Shadow::copy_back_marked_run`]): 4 MiB.
const RUN: usize = 64;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<crate::fx::Set<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
    /// Title ranges `[from, to)` whose copies [`Shadow::copy_queued`]
    /// queued, for [`Shadow::queued_made`] (tracked memory only).
    queued: Mutex<Vec<(usize, usize)>>,
    /// Its first bit in the buffer views' written-pages bitmap, one per
    /// 64 KiB title page it touches (tracked memory only): writes through
    /// views of it are marked there ([`Shadow::mark_entry`]).
    marks: Option<u32>,
    /// The host buffer is the title memory itself (`VICTORY_IN_PLACE`): a
    /// buffer placed in a heap opened over the title's allocation, so there
    /// is nothing to copy either way.
    in_place: bool,
    /// In place: the heap it is placed in ([`HeapId`]), which tells it
    /// apart from buffers placed in a heap opened later at the same address.
    heap: Option<HeapId>,
    /// Where it starts in `buffer` (0 but for copies sharing a buffer), and
    /// the run of the shared buffer it holds ([`Arena`]), given back when it
    /// goes.
    base: u64,
    slice: Option<Slice>,
    /// In a GPU upload heap ([`register_read_only`]): the CPU must not read
    /// it back.
    pub(crate) upload: std::sync::atomic::AtomicBool,
    /// A copy shaders may write, in a GPU upload heap (`buffer`, `gpu`), with
    /// `data` its mirror in system memory, where the CPU reads and writes:
    /// what the CPU writes goes on into video memory ([`Shadow::push`]), what
    /// the GPU wrote comes back before the CPU reads it ([`pull`]).
    vram: Option<Vram>,
}

/// The video memory side of a mirrored copy.
struct Vram {
    /// The mirror's buffer (system memory), where [`pull`] copies to.
    mirror: ID3D12Resource,
    /// The copy in video memory, mapped (written, never read).
    data: *mut u8,
    /// A bit per [`PAGE`] of the copy (by offset) whose mirror is behind
    /// video memory: refreshes write video memory only, and the mirror
    /// catches up when the CPU reads it ([`Shadow::freshen`]).
    stale: Mutex<Vec<u64>>,
    /// Offsets (whole pages, sorted, apart) a submission takes snapshots of
    /// ([`snapshots_ahead`]): refreshes there write the mirror too, so that
    /// the snapshots need no copy back from video memory.
    ahead: Mutex<Vec<(usize, usize)>>,
}

/// The references `buffer` has besides the caller's. (`Release` returns the
/// count left; COM documents it for tests because another thread may take
/// or drop references meanwhile. Here no other one can take a reference
/// when none is held: 0 stays 0.)
pub(crate) fn others(buffer: &ID3D12Resource) -> u32 {
    use windows::core::Interface;
    // SAFETY: a live COM object; one reference taken and given back.
    unsafe {
        let raw = buffer.as_raw();
        let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
        ((**unknown).AddRef)(raw);
        ((**unknown).Release)(raw).saturating_sub(1)
    }
}

/// Snapshot segments (offset, bytes) and the pieces compared.
type Snapshot = (Vec<(usize, Vec<u8>)>, crate::fx::Set<usize>);

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool, which keeps the largest up to
/// [`POOL_BYTES`] in all: a world submission snapshots several copies, and
/// fresh buffers cost a page fault per 4 KiB touched (millions per frame).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let mut total = 0;
    pool.retain(|v| {
        total += v.capacity();
        total <= POOL_BYTES
    });
}

/// Snapshot memory kept for reuse.
const POOL_BYTES: usize = 512 << 20;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        self.freshen(at, at + 4 * n);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
        // As written on the GPU and pulled back.
        self.push(at, bytes.len());
    }

    /// Whether it is in video memory with a mirror ([`Shadow::push`]).
    #[cfg(test)]
    pub(crate) fn mirrored(&self) -> bool {
        self.vram.is_some()
    }

    /// The pages (by index in the copy) its mirror is behind on.
    #[cfg(test)]
    fn stale_pages(&self) -> Vec<usize> {
        self.stale_runs(0, self.len as usize)
            .into_iter()
            .flat_map(|(_, at, n)| at / PAGE..(at + n).div_ceil(PAGE))
            .collect()
    }

    /// The CPU wrote `[at, at + len)` (offsets) of the mirror: the same
    /// bytes into the copy in video memory.
    fn push(&self, at: usize, len: usize) {
        if let Some(v) = &self.vram {
            let len = len.min((self.len as usize).saturating_sub(at));
            // SAFETY: inside both mappings (cut to the copy); the mirror is
            // system memory, the video memory is only written.
            unsafe { std::ptr::copy_nonoverlapping(self.data.add(at), v.data.add(at), len) };
        }
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        self.freshen(at, at + 4 * n);
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.base
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        if self.in_place {
            self.landed(page, PAGE);
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        self.freshen(from - start, to - start);
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) || title == host {
            return 0;
        }
        {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, host.as_ptr(), n);
        }
        // Our write is not the title's: the copy holds the page as it is.
        if let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        n
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.in_place {
            self.keep_in_place(list);
            return;
        }
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    /// An in-place buffer has nothing to bring up to date, but `list`'s
    /// commands name its GPU address: the list keeps the placed buffer (and
    /// so its heap, which a placed resource holds) until it is reset, for
    /// the title may release the buffer, or the heap be let go, before the
    /// list runs.
    fn keep_in_place(&self, list: &HostList) {
        // By the buffer: kept, its address is not given out again meanwhile.
        list.keep_once(windows::core::Interface::as_raw(&self.buffer) as usize, &self.buffer);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Arc<Shadow> {
        Shadow::new_at(buffer, data, gpu, start, len, None, None)
    }

    /// [`Shadow::new`] of a copy in a run of a shared buffer (`data` and
    /// `gpu` are the run's), or of one in video memory with `data` its
    /// mirror (`vram`).
    fn new_at(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        slice: Option<Slice>,
        vram: Option<Vram>,
    ) -> Arc<Shadow> {
        Shadow::make(buffer, data, gpu, start, len, slice, vram, None)
    }

    /// [`Shadow::new`], in place (`heap` given: `buffer` is placed in that
    /// heap, opened over the title's memory) or a copy of its own.
    fn new_with(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64, heap: Option<HeapId>) -> Arc<Shadow> {
        Shadow::make(buffer, data, gpu, start, len, None, None, heap)
    }

    #[allow(clippy::too_many_arguments)]
    fn make(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        slice: Option<Slice>,
        vram: Option<Vram>,
        heap: Option<HeapId>,
    ) -> Arc<Shadow> {
        let in_place = heap.is_some();
        let tracked = if in_place { None } else { crate::watch::track_copy(start as usize, len as usize, true) };
        let marks = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        let shadow = Arc::new(Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked,
            pending: Mutex::default(),
            before: Mutex::default(),
            queued: Mutex::default(),
            marks,
            in_place,
            heap,
            base: slice.map_or(0, |s| s.offset),
            slice,
            upload: std::sync::atomic::AtomicBool::new(false),
            vram,
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        COPY_BYTES.fetch_add(len, std::sync::atomic::Ordering::Relaxed);
        shadow
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    ///
    /// `marked`: the shaders mark the pages they write through this copy's
    /// views (`victory_shader::emit`, "Write marks"): no snapshot then, only
    /// the pages they marked are compared ([`Shadow::copy_back_marked`]).
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        if self.in_place {
            self.keep_in_place(list);
            return;
        }
        if self.upload.load(std::sync::atomic::Ordering::Relaxed) {
            // A view of the memory that writes (through another resource,
            // or a placed view) found this read-only buffer's copy: right,
            // but its copy-back reads video memory at ~50 MB/s.
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    start = format_args!("{:#x}", self.start),
                    "shaders write the copy of a buffer made without unordered access, in upload memory: slow copy-back"
                );
            });
        }
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
    }

    /// The host writes `[address, address + len)` of the copy in `list`
    /// without a shader (an unordered access clear): as
    /// [`Shadow::gpu_writes`], but its pages are marked as written at
    /// submission, as a shader's view writes mark them, so that they go
    /// back as marked pages ([`copy_back_marks`]): no snapshot of the range
    /// before the submission and no comparison with it after, which for a
    /// big buffer cleared in every submission cost tens of milliseconds
    /// each (`benchmark_blank_frame`). Whole pages only: a refresh does not
    /// take a page the range holds in part, which a title write before the
    /// submission leaves changed, and a changed page that was not armed
    /// keeps the title's bytes ([`Shadow::copy_back_marked`]); the parts of
    /// pages at the range's ends are declared, byte for byte, as are the
    /// writes of copies whose writes are not marked.
    pub(crate) fn host_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        let end = address + len;
        let page = PAGE as u64;
        let (first, last) = (address.next_multiple_of(page), end / page * page);
        if !self.marked() || first >= last {
            self.gpu_writes(list, address, len, false);
            return;
        }
        if address < first {
            self.gpu_writes(list, address, first - address, false);
        }
        self.gpu_writes(list, first, last - first, true);
        if last < end {
            self.gpu_writes(list, last, end - last, false);
        }
        let copy = self.clone();
        list.defer_host(move || copy.mark_pages(first, last - first));
    }

    /// Sets the view-mark bits of the pages `[address, address + len)`
    /// touches, as a shader writing them through a view would (between
    /// submissions: the bitmap is read and cleared after each).
    fn mark_pages(&self, address: u64, len: u64) {
        let (Some(base), Some(b)) = (self.marks, bitmap(false)) else {
            return;
        };
        let (from, to) = (
            address.max(self.start),
            (address + len).min(self.start + self.len),
        );
        if from >= to {
            return;
        }
        let words = b.data as *mut u32;
        for k in (from >> 16) - (self.start >> 16)..=((to - 1) >> 16) - (self.start >> 16) {
            let bit = base + k as u32;
            // SAFETY: a dword of the mapped bitmap (the copy's bits are
            // inside it); the GPU is not using it between submissions.
            unsafe { *words.add(bit as usize / 32) |= 1 << (bit % 32) };
        }
    }

    /// Shaders of `list` may also write `[address, address + len)` of the
    /// copy, past the range they declare (a root view has no size): nothing
    /// of it goes back to title memory, and after the submission
    /// [`Shadow::written_unseen`] marks it.
    pub(crate) fn may_write(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.vram.is_none() || len == 0 {
            return;
        }
        let copy = self.clone();
        list.defer_after(move || copy.written_unseen(address as usize, len as usize));
    }

    /// After a submission whose shaders may have written `[address, address
    /// + len)` of the copy where no [`pull`] brings it: for a copy in video
    /// memory, the mirror's pages there count as stale, so the CPU reads
    /// video memory's bytes there first ([`Shadow::freshen`]).
    pub(crate) fn written_unseen(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        if from < to {
            self.stale(from, to);
        }
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, segments covering the ranges the submission
    /// declares (the world declares overlapping views of its 174 MB heap
    /// dozens of times; one segment over all of them copied most of the
    /// heap), in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(&self, address: usize, len: usize, marked: bool) {
        if self.in_place {
            return;
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
        if marked && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        self.freshen(from, to);
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let segments = &mut before.0;
        // The segments the range overlaps or touches.
        let first = segments.partition_point(|(at, bytes)| at + bytes.len() < from);
        let last = first + segments[first..].iter().take_while(|(at, _)| *at <= to).count();
        if let [(at, bytes)] = &segments[first..last]
            && *at <= from
            && to <= at + bytes.len()
        {
            return;
        }
        let start = segments.get(first).filter(|_| first < last).map_or(from, |s| s.0.min(from));
        let end = segments[first..last].last().map_or(to, |(at, bytes)| (at + bytes.len()).max(to));
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotBytes, (end - start) as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::snapshot(self.start, self.len, (end - start) as u64);
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        let mut at = start;
        for (taken, bytes) in segments.drain(first..last) {
            grown.extend_from_slice(host(at, taken));
            grown.extend_from_slice(&bytes);
            at = taken + bytes.len();
            give_back(bytes);
        }
        grown.extend_from_slice(host(at, end));
        segments.insert(first, (start, grown));
    }

    /// After the submission ran, for a buffer in place: its shaders may have
    /// written `[address, address + len)` of title memory directly, which
    /// no guard or write watch sees. Other copies holding those pages (a
    /// title buffer under the in-place floor, a view's own copy, a texture)
    /// would keep their older bytes: the pages are marked changed, as a
    /// copy-back marks the pages it writes.
    pub(crate) fn landed(&self, address: usize, len: usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if self.in_place && from < to {
            drop(crate::watch::writing(from, to - from));
        }
    }

    /// `[address, address + len)` expanded outward to 64 KiB page
    /// boundaries and cut to the copy (`[start, start + len)`): whole pages
    /// so [`crate::watch::Tracked::take`] guards them for
    /// [`Shadow::copy_back_marked`].
    pub(crate) fn page_range(&self, address: u64, len: u64) -> (u64, u64) {
        let end = self.start.saturating_add(self.len);
        let from = address.clamp(self.start, end);
        let to = address.saturating_add(len).clamp(from, end);
        if from >= to {
            return (from, 0);
        }
        let page = PAGE as u64;
        let from = (from & !(page - 1)).max(self.start);
        let to = to.checked_next_multiple_of(page).unwrap_or(end).min(end);
        (from, to - from)
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        if self.in_place {
            self.landed(address, len);
            return 0;
        }
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        if self.in_place {
            // Written into title memory already: nothing can keep them out.
            self.landed(address, len);
            return;
        }
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = segments.iter().find(|(at, bytes)| *at <= from && to <= at + bytes.len())
            && back
            && from < to
        {
            let before = &before[from - taken..to - taken];
            // The mirror is read below (with snapshots ahead, its pages are
            // current already).
            self.freshen(from, to);
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let title = &mut title[..n];
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    if merge_writeback(title, host, old) {
                        let _writing = crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                    // The copy took the title's bytes: so does video memory.
                    self.push(at, n);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
        written
    }

    /// After a submission whose shaders marked the 64 KiB title page at
    /// `page` as written through views of this copy: the bytes they changed
    /// go into title memory, and the copy takes the title's others. What
    /// they changed is what differs from the copy's bytes before the
    /// submission: title memory where the title has not written the page
    /// since the copy took it, else what the page held when the title first
    /// wrote it ([`crate::watch::Tracked::saved`], if the page was armed).
    /// Returns the bytes written.
    ///
    /// `only`: the title address ranges of the page that go back (sorted,
    /// disjoint); None for all of it. Bytes outside them are left as they
    /// are in both the copy and title memory, as on a page not copied back.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    pub(crate) fn copy_back_marked(&self, page: usize, only: Option<&[(usize, usize)]>) -> usize {
        if self.in_place {
            return 0;
        }
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        self.freshen(from - start, to - start);
        // Our own write below must not save the page for this copy again.
        t.unarm(page / PAGE);
        let saved = t.saved(page / PAGE);
        if saved.is_none() && t.changed(page / PAGE) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedKept, 1);
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) {
            return 0;
        }
        if let Some(only) = only {
            // As below, inside the ranges only.
            let mut changed = false;
            for &(a, e) in only {
                let (a, e) = (a.max(from) - from, e.min(to).saturating_sub(from));
                if a >= e {
                    continue;
                }
                let old: std::borrow::Cow<[u8]> = match &saved {
                    Some(before) => (&before[from - page + a..from - page + e]).into(),
                    None => title[a..e].to_vec().into(),
                };
                changed |= merge_writeback(&mut title[a..e], &mut host[a..e], &old);
            }
            if changed {
                let _writing = crate::watch::writing(from, n);
                crate::host::write_mapped(from as *mut u8, title.as_ptr(), n);
            }
            // The copy differs from title memory outside the ranges: not
            // taken.
            return if changed { n } else { 0 };
        }
        let written = match &saved {
            // The title's bytes are the copy's from before: what differs is
            // the shaders'.
            None => (*host != title[..]).then_some(host.as_ptr()),
            Some(before) => {
                let before = &before[from - page..to - page];
                merge_writeback(&mut title, host, before).then_some(title.as_ptr())
            }
        };
        if let Some(bytes) = written {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, bytes, n);
        }
        self.push(from - start, n);
        // Equal to title memory now: taken, unless it holds the page in part.
        if from == page && to == page + PAGE {
            t.taken(&[page / PAGE]);
        }
        written.map_or(0, |_| n)
    }

    /// Whether the title wrote none of the copy's memory since the copy took
    /// it (tracked memory only): [`Shadow::update`] would copy nothing.
    pub(crate) fn clean(&self) -> bool {
        self.tracked.as_ref().is_some_and(crate::watch::Tracked::clean)
    }

    /// Its tracker's bit: `crate::watch::clean_bit` of it is
    /// [`Shadow::clean`] (none for memory that is not tracked: never clean).
    pub(crate) fn dirty_bit(&self) -> Option<u32> {
        self.tracked.as_ref().and_then(crate::watch::Tracked::bit)
    }

    /// [`Shadow::copy_back_marked`] of whole marked pages in a row (`pages`,
    /// ascending, each the next): those the title has not written since the
    /// copy took them go back [`RUN`] pages at a time, each time with one
    /// read of title memory, one write per stretch of pages the GPU changed
    /// and one take (page by page, a clear of 256 MiB cost 90 ms in these
    /// calls: `benchmark_blank_frame`); the others page by page. Returns the
    /// bytes written.
    pub(crate) fn copy_back_marked_run(&self, pages: &[usize]) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let mut written = 0;
        let mut plain: Vec<usize> = Vec::with_capacity(RUN.min(pages.len()));
        for &page in pages {
            if page < start || page + PAGE > end || t.changed(page / PAGE) {
                written += self.copy_back_plain(t, &mut plain);
                written += self.copy_back_marked(page, None);
            } else {
                plain.push(page);
                if plain.len() == RUN {
                    written += self.copy_back_plain(t, &mut plain);
                }
            }
        }
        written + self.copy_back_plain(t, &mut plain)
    }

    /// The copy-back of [`Shadow::copy_back_marked_run`]'s whole pages in a
    /// row that the title had not written (`pages`, emptied): the title's
    /// bytes are the copy's from before, so what differs is the GPU's. A
    /// page the title writes meanwhile goes back on its own, by what it
    /// saved.
    fn copy_back_plain(&self, t: &crate::watch::Tracked, pages: &mut Vec<usize>) -> usize {
        let (Some(&first), Some(&last)) = (pages.first(), pages.last()) else {
            return 0;
        };
        let start = self.start as usize;
        let (from, to) = (first, last + PAGE);
        let n = to - from;
        self.freshen(from - start, to - start);
        // Our own writes below must not save the pages for this copy again.
        for &page in pages.iter() {
            t.unarm(page / PAGE);
        }
        // Reused memory: a fresh buffer faults at each 4 KiB it takes.
        let mut title = pooled(n);
        title.resize(n, 0);
        if !crate::host::read_into(from as *const u8, &mut title) {
            give_back(title);
            // A page not mapped: page by page, which leaves it alone.
            let written = pages.iter().map(|&p| self.copy_back_marked(p, None)).sum();
            pages.clear();
            return written;
        }
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut written = 0;
        let mut taken = Vec::with_capacity(pages.len());
        let mut alone = Vec::new();
        // The stretch of pages that differ, not yet written: its first byte.
        let mut stretch: Option<usize> = None;
        let mut write = |stretch: &mut Option<usize>, upto: usize| {
            if let Some(at) = stretch.take() {
                let _writing = crate::watch::writing(at, upto - at);
                crate::host::write_mapped(at as *mut u8, host[at - from..].as_ptr(), upto - at);
                written += upto - at;
            }
        };
        for &page in pages.iter() {
            let i = page - from;
            if t.changed(page / PAGE) {
                // Written by the title since the check: by what it saved.
                write(&mut stretch, page);
                alone.push(page);
                continue;
            }
            if host[i..i + PAGE] != title[i..i + PAGE] {
                stretch.get_or_insert(page);
            } else {
                write(&mut stretch, page);
            }
            taken.push(page / PAGE);
        }
        write(&mut stretch, to);
        give_back(title);
        self.push(from - start, n);
        // Equal to title memory now: taken.
        t.taken(&taken);
        pages.clear();
        written
            + alone
                .into_iter()
                .map(|p| self.copy_back_marked(p, None))
                .sum::<usize>()
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        if self.in_place || self.clean() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if self.tracked.is_some() {
            return self.update_tracked(from, to, !busy.is_empty());
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them nothing: what the
            // title wrote there is not known.
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_untracked(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_untracked(at, to);
            }
            return bytes;
        }
        self.update_untracked(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where the
    /// memory is tracked: the pages the title wrote since they were taken,
    /// in one [`crate::watch::Tracked::take`], so that a page wholly in the
    /// range is guarded even where shaders write part of it (split at those
    /// ranges it stayed changed, and [`Shadow::copy_back_marked`] dropped
    /// what they wrote).
    ///
    /// `busy`: a list submitted earlier declared that shaders write in the
    /// range. A submission waits for the one before it, so none of this
    /// submission's shaders has run yet: the pages the title wrote are
    /// copied as anywhere else. Skipped, the title's constants there stayed
    /// stale for the whole submission (a dispatch read its sizes as zeros
    /// and looped until the device was removed). The same bytes go into the
    /// snapshot `writing` took, so copy-back neither takes them for the
    /// GPU's nor puts them back over a later title write.
    fn update_tracked(&self, from: usize, to: usize, busy: bool) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (runs, unmapped) = t.take_split(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
        let start = self.start as usize;
        let mut before = busy.then(|| self.before.lock().unwrap_or_else(|e| e.into_inner()));
        // `copied`: by [`Shadow::copy`] (else written into the mirror).
        let sync_before = |before: Option<&mut Snapshot>, a: usize, b: usize, copied: bool| {
            let Some(before) = before else { return };
            let (a, b) = (a - start, b - start);
            for (at, bytes) in before.0.iter_mut() {
                let (s, e) = ((*at).max(a), (*at + bytes.len()).min(b));
                if s < e {
                    // The mirror too (with no snapshot ahead, a refresh
                    // wrote video memory only).
                    if copied && self.vram.is_some() {
                        self.copy_both(s, e);
                    }
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    bytes[s - *at..e - *at].copy_from_slice(host);
                }
            }
        };
        // A snapshot over the run takes the copied bytes at once; with none
        // there (marked ranges keep none) the copy may be queued.
        let refresh = |before: Option<&mut Snapshot>, a: usize, b: usize| -> usize {
            let (ra, rb) = (a - start, b - start);
            match before {
                Some(before)
                    if before.0.iter().any(|(at, bytes)| *at < rb && ra < *at + bytes.len()) =>
                {
                    let copied = self.copy(a, b);
                    sync_before(Some(before), a, b, true);
                    copied
                }
                _ => self.copy_queued(a, b),
            }
        };
        let mut copied = 0;
        for (a, b) in runs {
            copied += refresh(before.as_deref_mut(), a, b);
        }
        for (a, b) in unmapped {
            // SAFETY: inside the mapping (`take_split` cuts to the buffer).
            unsafe { self.data.add(a - start).write_bytes(0, b - a) };
            self.push(a - start, b - a);
            sync_before(before.as_deref_mut(), a, b, false);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, (b - a) as u64);
        }
        // Parts on memory that is not the runtime's: whole, as below.
        for (a, b) in t.untracked(from, to) {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            copied += refresh(before.as_deref_mut(), a, b);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::WholeBytes, (b - a) as u64);
                add(C::WholeUs, started.elapsed().as_micros() as u64);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
        }
        copied
    }

    /// [`Shadow::update`] of untracked `[from, to)` (inside the buffer).
    fn update_untracked(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy_queued(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// As [`Shadow::copy`], queued where the caller collects a
    /// submission's refreshes ([`crate::copy_pool::queued`]): for copies
    /// nothing reads back before they are made. What a queued copy took
    /// goes into the saved bytes of armed pages once it is made
    /// ([`Shadow::queued_made`]).
    fn copy_queued(&self, from: usize, to: usize) -> usize {
        // A copy in video memory takes it there ([`Shadow::copy`]), which a
        // queued copy into the mirror would not do: at once.
        if self.vram.is_some() {
            return self.copy(from, to);
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        if !crate::copy_pool::copy(at, from as *const u8, to - from) {
            if let Some(t) = &self.tracked {
                // SAFETY: `to - from` bytes inside the mapping, just copied.
                t.refreshed(from, unsafe { std::slice::from_raw_parts(at, to - from) });
            }
        } else if self.tracked.is_some() {
            self.queued.lock().unwrap_or_else(|e| e.into_inner()).push((from, to));
        }
        to - from
    }

    /// After [`crate::copy_pool::run`] made the copies
    /// [`Shadow::copy_queued`] queued: what they took goes into the saved
    /// bytes of the armed pages they cover, as [`Shadow::copy`]'s does.
    pub(crate) fn queued_made(&self) {
        let Some(t) = &self.tracked else {
            return;
        };
        let made = std::mem::take(&mut *self.queued.lock().unwrap_or_else(|e| e.into_inner()));
        let start = self.start as usize;
        for (from, to) in made {
            // SAFETY: `to - from` bytes inside the mapping, copied.
            let bytes =
                unsafe { std::slice::from_raw_parts(self.data.add(from - start), to - from) };
            t.refreshed(from, bytes);
        }
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy:
    /// into video memory alone for a mirrored copy, whose mirror catches up
    /// when the CPU reads it ([`Shadow::freshen`]; writing it too cost a
    /// second copy of everything refreshed).
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        // Saved bytes of armed pages take what the copy took, read from the
        // mirror ([`crate::watch::Tracked::refreshed`]): written too there.
        if let Some(v) = &self.vram
            && !self.tracked.as_ref().is_some_and(|t| t.saves(from, to))
        {
            let (a, b) = (from - start, to - start);
            let ahead = v.ahead.lock().unwrap_or_else(|e| e.into_inner());
            let mut at = a;
            for &(s, e) in ahead.iter() {
                if e <= at {
                    continue;
                }
                if s >= b {
                    break;
                }
                if at < s {
                    self.copy_video(v, at, s);
                }
                let e = e.min(b);
                self.copy_both(at.max(s), e);
                at = e;
            }
            if at < b {
                self.copy_video(v, at, b);
            }
            return to - from;
        }
        self.copy_both(from - start, to - start);
        if let Some(t) = &self.tracked {
            // SAFETY: `to - from` bytes inside the mapping, just copied.
            let bytes =
                unsafe { std::slice::from_raw_parts(self.data.add(from - start), to - from) };
            t.refreshed(from, bytes);
        }
        to - from
    }

    /// Title memory at offsets `[from, to)` into the copy in video memory
    /// only: the mirror is behind there until [`Shadow::freshen`].
    fn copy_video(&self, v: &Vram, from: usize, to: usize) {
        // SAFETY: inside the video memory mapping (offsets within the copy).
        let at = unsafe { v.data.add(from) };
        crate::host::copy_mapped(
            at,
            (self.start as usize + from) as *const u8,
            to - from,
            false,
        );
        self.stale(from, to);
    }

    /// Title memory at offsets `[from, to)` into the copy (the mirror, and
    /// video memory after it).
    fn copy_both(&self, from: usize, to: usize) {
        // SAFETY: inside the mapping (offsets within the copy).
        let at = unsafe { self.data.add(from) };
        crate::host::copy_mapped(
            at,
            (self.start as usize + from) as *const u8,
            to - from,
            false,
        );
        self.push(from, to - from);
    }

    /// The mirror is behind video memory over offsets `[from, to)`.
    fn stale(&self, from: usize, to: usize) {
        let Some(v) = &self.vram else { return };
        let mut bits = v.stale.lock().unwrap_or_else(|e| e.into_inner());
        for k in from / PAGE..to.div_ceil(PAGE) {
            bits[k / 64] |= 1 << (k % 64);
        }
    }

    /// Before the CPU reads offsets `[from, to)` of a mirrored copy: the
    /// pages there the mirror is behind on come from video memory (one GPU
    /// copy; nothing when none is).
    fn freshen(&self, from: usize, to: usize) {
        let mut runs = self.stale_runs(from, to);
        bring_back(&mut runs);
    }

    /// The pages over offsets `[from, to)` the mirror is behind on, as runs
    /// `(copy, offset, len)` (whole pages, or to the copy's end).
    fn stale_runs(&self, from: usize, to: usize) -> Vec<(&Shadow, usize, usize)> {
        let Some(v) = &self.vram else {
            return Vec::new();
        };
        let len = self.len as usize;
        let mut runs: Vec<(&Shadow, usize, usize)> = Vec::new();
        let bits = v.stale.lock().unwrap_or_else(|e| e.into_inner());
        for k in from / PAGE..to.div_ceil(PAGE).min(len.div_ceil(PAGE)) {
            if bits[k / 64] >> (k % 64) & 1 == 0 {
                continue;
            }
            let (a, b) = (k * PAGE, ((k + 1) * PAGE).min(len));
            match runs.last_mut() {
                Some(r) if r.1 + r.2 == a => r.2 = b - r.1,
                _ => runs.push((self, a, b - a)),
            }
        }
        runs
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(slice) = self.slice {
            arena().give_back(slice);
        }
        COPY_BYTES.fetch_sub(self.len, std::sync::atomic::Ordering::Relaxed);
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// Bytes of title memory with a host copy now (copies alive).
static COPY_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`COPY_BYTES`], for the budget's report of what fills memory.
pub(crate) fn copy_bytes() -> u64 {
    COPY_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address.
struct Bitmap {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

/// The bitmap of other shaders, or (`memory`) of shaders that use title
/// memory, whose marks go back only while it is not withheld
/// (`crate::memory::withholding`).
fn bitmap(memory: bool) -> Option<&'static Bitmap> {
    static BITMAPS: [std::sync::OnceLock<Option<Bitmap>>; 2] = [const { std::sync::OnceLock::new() }; 2];
    BITMAPS[usize::from(memory)]
        .get_or_init(|| {
            // Written pages, one read mark per heap index, then one per
            // view heap page.
            let bits = victory_shader::emit::VIEW_PAGE_READS + PAGE_MARKS;
            let bytes = u64::from(bits / 8);
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages and read slots".into());
            crate::host::keep_resident(&buffer, crate::host::Residency::Frame);
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes (`memory`: shaders that use
/// title memory, see [`bitmap`]).
pub(crate) fn view_writes(memory: bool) -> Option<u64> {
    bitmap(memory).map(|b| b.gpu)
}

/// Heap slots whose texture views shaders read: the marks seen so far (a
/// slot's stays set in the bitmap until [`forget_read`]), and the slots to
/// clear before the next submission ([`forget_reads_now`]) or at the next
/// [`take_reads`].
static READS: Mutex<(Vec<u32>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));

/// Both bitmaps' read marks (shaders that use title memory mark their reads
/// in theirs).
///
/// # Safety
/// The GPU is done with them (one submission runs at a time), and no other
/// caller holds them.
unsafe fn read_maps<'a>() -> Vec<&'a mut [u32]> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    [false, true]
        .into_iter()
        .filter_map(bitmap)
        // SAFETY: the mapped bitmap's read marks, as the caller says.
        .map(|b| unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) })
        .collect()
}

/// Read marks cleared by [`clear_forgotten`] before [`reads_since`] took
/// them, a bit per slot: still reads since its last call.
static CLEARED: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// Clears the read marks of the slots forgotten ([`forget_read`]) since
/// the last call. Marks are by slot, whatever the heap: one set is kept for
/// [`reads_since`] (another heap's shader may have read the slot).
fn clear_forgotten(maps: &mut [&mut [u32]], seen: &mut Vec<u32>, forgotten: &mut Vec<u32>) {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    seen.resize(words, 0);
    if forgotten.is_empty() {
        return;
    }
    let mut cleared = CLEARED.lock().unwrap_or_else(|e| e.into_inner());
    cleared.resize(words, 0);
    for slot in forgotten.drain(..) {
        let (w, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        if w < words {
            for bits in maps.iter_mut() {
                cleared[w] |= bits[w] & bit;
                bits[w] &= !bit;
            }
            seen[w] &= !bit;
        }
    }
}

/// Before a submission runs (the last one has): the read marks of slots
/// forgotten since are cleared now, so that the submission's reads of them
/// are taken after it ([`take_reads`]).
pub(crate) fn forget_reads_now() {
    // SAFETY: no submission runs.
    let mut maps = unsafe { read_maps() };
    if maps.is_empty() {
        return;
    }
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    let (seen, forgotten) = &mut *reads;
    clear_forgotten(&mut maps, seen, forgotten);
}

/// After a submission: the heap slots its shaders first read texture views
/// at (`victory_shader::emit`, "Read marks"), each once until
/// [`forget_read`]. Clears the slots forgotten since first.
pub(crate) fn take_reads() -> Vec<u32> {
    // SAFETY: the submission has run.
    let mut maps = unsafe { read_maps() };
    if maps.is_empty() {
        return Vec::new();
    }
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    let (seen, forgotten) = &mut *reads;
    clear_forgotten(&mut maps, seen, forgotten);
    // Slots past every view heap's are no heap's (`descriptor::view_slots`):
    // their marks are left for a larger heap to find.
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let used = (crate::descriptor::view_slots().div_ceil(32) as usize).min(words);
    let mut new = Vec::new();
    for (w, s) in seen[..used].iter_mut().enumerate() {
        let mut fresh = maps.iter().fold(0, |x, bits| x | bits[w]) & !*s;
        *s |= fresh;
        while fresh != 0 {
            new.push(w as u32 * 32 + fresh.trailing_zeros());
            fresh &= fresh - 1;
        }
    }
    new
}

/// The heap slots shaders read texture views at since the last call (both
/// bitmaps, and marks forgotten meanwhile), whose marks are cleared: what
/// the title used (eviction stamps,
/// `crate::resource::evict`). Between submissions, after [`take_reads`]
/// has seen every mark: its own record of slots seen keeps a slot read
/// again from being reported as new.
pub(crate) fn reads_since() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    let mut slots = Vec::new();
    let cleared = std::mem::take(&mut *CLEARED.lock().unwrap_or_else(|e| e.into_inner()));
    for b in [false, true].into_iter().filter_map(bitmap) {
        // SAFETY: the mapped bitmap's read marks; no submission runs.
        let bits = unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
        // As in `take_reads`: no further than any view heap's slots.
        let used = (crate::descriptor::view_slots().div_ceil(32) as usize).min(words);
        for (w, word) in bits[..used].iter_mut().enumerate() {
            let mut x = std::mem::take(word) | cleared.get(w).copied().unwrap_or(0);
            while x != 0 {
                slots.push(w as u32 * 32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Heap slot `slot` now holds a texture view left null: its next read is
/// marked again.
pub(crate) fn forget_read(slot: u32) {
    READS.lock().unwrap_or_else(|e| e.into_inner()).1.push(slot);
}

/// Read marks of view heap pages: one per page of heap indices.
const PAGE_MARKS: u32 = victory_shader::emit::HEAP_VIEWS / victory_shader::emit::VIEW_PAGE_SLOTS;

/// After a submission: the view heap pages its shaders read through views
/// (`victory_shader::emit`, "Read marks"; paged heaps only), from both
/// bitmaps, whose marks are cleared.
pub(crate) fn take_page_reads() -> Vec<u32> {
    let words = (PAGE_MARKS / 32) as usize;
    let first = (victory_shader::emit::VIEW_PAGE_READS / 32) as usize;
    let mut pages = Vec::new();
    let mut read = vec![0u32; words];
    for b in [false, true].into_iter().filter_map(bitmap) {
        // SAFETY: the mapped bitmap's page read marks; the GPU is done with
        // them (one submission runs at a time).
        let bits =
            unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
        for (r, word) in read.iter_mut().zip(bits) {
            *r |= std::mem::take(word);
        }
    }
    for (w, mut x) in read.into_iter().enumerate() {
        while x != 0 {
            pages.push(w as u32 * 32 + x.trailing_zeros());
            x &= x - 1;
        }
    }
    pages
}

/// After a submission: every page its shaders marked as written through
/// views goes back ([`Shadow::copy_back_marked`]), but, with
/// `memory_withheld`, pages that shaders using title memory marked; the
/// bitmaps are cleared. `declared`: the ranges of copies the submission's
/// lists declared, and whether they go back (false for ranges whose writes
/// stay out of title memory: withheld). On a page a withheld range touches,
/// only what other ranges declared there goes back, less the withheld ones:
/// a valid write is not lost for sharing a page with a withheld one, and no
/// byte a withheld range declared, nor any no list declared, is released.
/// Before the copy-backs of declared ranges, which disarm the copies.
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
pub(crate) fn copy_back_marks(
    declared: &[(Arc<Shadow>, usize, usize, bool)],
    memory_withheld: bool,
) {
    let (mut marked, mut held) = (Vec::new(), Vec::new());
    for memory in [false, true] {
        let Some(b) = bitmap(memory) else {
            continue;
        };
        let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
        // SAFETY: the mapped bitmap; the GPU is done with it (one submission
        // runs at a time).
        let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
        // Marks of offsets past the bitmap land in its last bit, no copy's.
        bits[words - 1] = 0;
        let used = (marks().end as usize).div_ceil(32).min(words);
        for (w, word) in bits[..used].iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            let into = if memory && memory_withheld { &mut held } else { &mut marked };
            while x != 0 {
                into.push((w * 32) as u32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    marked.sort_unstable();
    marked.dedup();
    let page_of = |m: &Marks, n: u32| {
        let (&base, copy) = m.owners.range(..=n).next_back()?;
        let copy = copy.upgrade()?;
        let k = n - base;
        (k < pages(copy.start, copy.len)).then(|| {
            let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
            (copy, page)
        })
    };
    if marked.is_empty() && held.is_empty() {
        return;
    }
    let m = marks();
    let pages: Vec<(Arc<Shadow>, usize)> = marked.iter().filter_map(|&n| page_of(&m, n)).collect();
    let held: Vec<(Arc<Shadow>, usize)> = held.into_iter().filter_map(|n| page_of(&m, n)).collect();
    drop(m);
    // Into the mirrors, held back or not: the copies keep the GPU's bytes
    // either way.
    pull(
        &pages
            .iter()
            .chain(&held)
            .map(|(copy, page)| (copy.clone(), *page, PAGE))
            .collect::<Vec<_>>(),
    );
    // Held back: dropped from the copies (`crate::memory::drop_withheld`).
    for (_, page) in held {
        crate::memory::drop_withheld(page, PAGE);
    }
    // Whole pages of a copy in a row go back together (a run of them a
    // clear or a big dispatch wrote: a kernel copy and a guard change per
    // run, not per page).
    let mut run: Option<(Arc<Shadow>, Vec<usize>)> = None;
    let flush = |run: &mut Option<(Arc<Shadow>, Vec<usize>)>| {
        if let Some((copy, pages)) = run.take() {
            let written = copy.copy_back_marked_run(&pages);
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::MarkedPages, pages.len() as u64);
                crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
            }
            #[cfg(not(feature = "diagnostics"))]
            let _ = written;
        }
    };
    for (copy, page) in pages {
        // The declared ranges of this copy on the page, cut to it.
        let on_page = |back: bool| -> Vec<(usize, usize)> {
            declared
                .iter()
                .filter(|(c, a, n, b)| {
                    *b == back && Arc::ptr_eq(c, &copy) && *a < page + PAGE && page < a + n
                })
                .map(|(_, a, n, _)| ((*a).max(page), (a + n).min(page + PAGE)))
                .collect()
        };
        let withheld = on_page(false);
        let only = (!withheld.is_empty()).then(|| subtract(on_page(true), &withheld));
        if only.as_ref().is_some_and(Vec::is_empty) {
            continue;
        }
        if only.is_none() {
            match &mut run {
                Some((c, pages))
                    if Arc::ptr_eq(c, &copy) && pages.last().map(|p| p + PAGE) == Some(page) =>
                {
                    pages.push(page);
                }
                _ => {
                    flush(&mut run);
                    run = Some((copy, vec![page]));
                }
            }
            continue;
        }
        flush(&mut run);
        let written = copy.copy_back_marked(page, only.as_deref());
        #[cfg(feature = "diagnostics")]
        {
            crate::heat::copied_back(copy.range(), written);
            crate::copystat::add(crate::copystat::C::MarkedPages, 1);
            crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = written;
    }
    flush(&mut run);
}

/// `ranges` (half-open) less `minus`: sorted, disjoint.
fn subtract(mut ranges: Vec<(usize, usize)>, minus: &[(usize, usize)]) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut joined: Vec<(usize, usize)> = Vec::new();
    for (a, e) in ranges {
        match joined.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(e),
            _ => joined.push((a, e)),
        }
    }
    for &(ma, me) in minus {
        joined = joined
            .into_iter()
            .flat_map(|(a, e)| [(a, e.min(ma)), (a.max(me), e)])
            .filter(|(a, e)| a < e)
            .collect();
    }
    joined
}

/// Before the CPU reads what shaders may have written: the title address
/// ranges `(copy, address, len)` of copies in video memory are copied into
/// their mirrors, by the GPU, at once (one list for all; ranges of other
/// copies are skipped). Where shaders may write with nothing pulled after
/// (past a root view's declared range, through the heap's views of copies
/// without marks), the mirror's pages count as stale instead
/// ([`Shadow::written_unseen`]).
pub(crate) fn pull(ranges: &[(Arc<Shadow>, usize, usize)]) {
    // Whole pages, so that the pages pulled are current in the mirror.
    let mut runs: Vec<(&Shadow, usize, usize)> = Vec::new();
    for (copy, address, len) in ranges {
        if copy.vram.is_none() {
            continue;
        }
        let (from, to) = copy.inside(*address, *len);
        if from >= to {
            continue;
        }
        let (from, to) = (
            from / PAGE * PAGE,
            (to.div_ceil(PAGE) * PAGE).min(copy.len as usize),
        );
        runs.push((copy, from, to - from));
    }
    bring_back(&mut runs);
}

/// Before a submission whose lists declared that shaders write title
/// address ranges `(copy, address, len)` (not marked): their snapshots
/// ([`Shadow::writing`]) read the copies' mirrors. The pages there the
/// mirrors are behind on come back now, all in one GPU copy, and until
/// [`snapshots_done`] refreshes there write the mirrors too: else each
/// snapshot waited for a GPU copy of its own.
pub(crate) fn snapshots_ahead(ranges: &[(Arc<Shadow>, usize, usize)]) {
    let mut runs: Vec<(&Shadow, usize, usize)> = Vec::new();
    for (copy, address, len) in ranges {
        let Some(v) = &copy.vram else { continue };
        let (from, to) = copy.inside(*address, *len);
        if from >= to {
            continue;
        }
        let (from, to) = (
            from / PAGE * PAGE,
            (to.div_ceil(PAGE) * PAGE).min(copy.len as usize),
        );
        {
            let mut ahead = v.ahead.lock().unwrap_or_else(|e| e.into_inner());
            ahead.push((from, to));
            ahead.sort_unstable();
            let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ahead.len());
            for &(a, b) in ahead.iter() {
                match merged.last_mut() {
                    Some(m) if a <= m.1 => m.1 = m.1.max(b),
                    _ => merged.push((a, b)),
                }
            }
            *ahead = merged;
        }
        runs.extend(copy.stale_runs(from, to));
    }
    bring_back(&mut runs);
}

/// After the submission's copy-backs: refreshes write video memory alone
/// again ([`snapshots_ahead`]).
pub(crate) fn snapshots_done(ranges: &[(Arc<Shadow>, usize, usize)]) {
    for (copy, _, _) in ranges {
        if let Some(v) = &copy.vram {
            v.ahead.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
    }
}

/// Copies runs `(copy, offset, len)` of mirrored copies (whole pages, or
/// to the copy's end) from video memory into their mirrors, by the GPU,
/// in one list; the pages are current in the mirrors after. Each copy's
/// stale pages stay locked over the GPU copy: a refresh meanwhile marks
/// its page again once the lock is released, so the bit is not lost.
fn bring_back(runs: &mut [(&Shadow, usize, usize)]) {
    if runs.is_empty() {
        return;
    }
    runs.sort_unstable_by_key(|r| (std::ptr::from_ref(r.0) as usize, r.1));
    // Overlapping or adjacent runs of a copy as one region.
    let mut merged: Vec<(&Shadow, usize, usize)> = Vec::with_capacity(runs.len());
    for &(copy, at, n) in runs.iter() {
        if let Some(last) = merged.last_mut()
            && std::ptr::eq(last.0, copy)
            && at <= last.1 + last.2
        {
            last.2 = last.2.max(at + n - last.1);
            continue;
        }
        merged.push((copy, at, n));
    }
    // Locked in address order (the runs are sorted by copy).
    let mut locked = Vec::new();
    let mut regions = Vec::with_capacity(merged.len());
    for &(copy, at, n) in &merged {
        let Some(v) = &copy.vram else { continue };
        if !locked.last().is_some_and(|(c, _)| std::ptr::eq(*c, copy)) {
            locked.push((copy, v.stale.lock().unwrap_or_else(|e| e.into_inner())));
        }
        let bits = &mut locked.last_mut().expect("just pushed").1;
        for k in at / PAGE..(at + n).div_ceil(PAGE) {
            bits[k / 64] &= !(1 << (k % 64));
        }
        regions.push((&v.mirror, at as u64, &copy.buffer, at as u64, n as u64));
    }
    #[cfg(test)]
    if !regions.is_empty() {
        BROUGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(host) = host() {
        host.copy_now(&regions);
    }
}

/// GPU copies [`bring_back`] made (tests).
#[cfg(test)]
static BROUGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
    /// The title made it without unordered access: shaders only read it,
    /// so its copy may be in a GPU upload heap ([`register_read_only`]).
    read_only: bool,
}

/// Title buffers by (start address, resource), with their copy once made.
/// Read by every draw's views (`find`) on the title's threads, at once.
static BUFFERS: RwLock<BTreeMap<(u64, usize), Entry>> = RwLock::new(BTreeMap::new());

fn buffers() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

fn buffers_read() -> std::sync::RwLockReadGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

/// The key of the innermost title buffer holding `[address, end)`
/// ([`enclosing`], among buffers up to the longest one's length below).
fn innermost(
    map: &BTreeMap<(u64, usize), Entry>,
    address: u64,
    end: u64,
) -> Option<(u64, usize)> {
    enclosing(map, MAX_SPAN.load(Ordering::Relaxed), address, end)
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// The longest title buffer registered so far (it stays after the buffer is
/// released): a buffer starting further below an address than this cannot
/// hold it. Changed under `BUFFERS`' lock.
static MAX_SPAN: AtomicU64 = AtomicU64::new(0);

/// Counts title buffers made and released: which buffer is the innermost
/// one holding some bytes ([`find`]) may differ when it changed.
static OWNERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The count of [`OWNERS`].
pub(crate) fn owners() -> u64 {
    OWNERS.load(std::sync::atomic::Ordering::Acquire)
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    register_as(resource, start, len, false);
}

/// [`register`] for a buffer the title made without unordered access:
/// shaders only read it, and the CPU only writes its copy (refreshes), so
/// the copy can be in video memory the CPU writes through the PCI BAR
/// ([`Host::gpu_upload_buffer`](crate::host::Host::gpu_upload_buffer)).
pub(crate) fn register_read_only(resource: usize, start: u64, len: u64) {
    register_as(resource, start, len, true);
}

fn register_as(resource: usize, start: u64, len: u64, read_only: bool) {
    let mut map = buffers();
    MAX_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert(
        (start, resource),
        Entry {
            end: start.saturating_add(len),
            shadow: None,
            read_only,
        },
    );
    // After the buffers changed: a count read before is looked at again.
    OWNERS.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let removed = buffers().remove(&(start, resource));
    OWNERS.fetch_add(1, std::sync::atomic::Ordering::Release);
    if removed.is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The innermost entry holding `[address, end)`: the one starting closest
/// below `address` that reaches `end`, among those starting at most `span`
/// (the longest entry's length) below `end`. Big buffers hold many smaller
/// ones (a heap's placed buffers), so the one holding an address past
/// those can be any number of entries back.
fn enclosing(map: &BTreeMap<(u64, usize), Entry>, span: u64, address: u64, end: u64) -> Option<(u64, usize)> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    map.range((lowest, 0)..=(address, usize::MAX))
        .rev()
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k)
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    // A copy made already, as it nearly always is, is found under the
    // shared lock; the copy is made under the exclusive one.
    {
        let map = buffers_read();
        let Some(key) = innermost(&map, address, end) else {
            drop(map);
            return loose(address, end);
        };
        if let Some(shadow) = map.get(&key).and_then(|e| e.shadow.clone()) {
            return Some(shadow);
        }
    }
    let mut map = buffers();
    let Some(key) = innermost(&map, address, end) else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() && in_place_enabled() && entry.end - key.0 >= in_place_min() {
        // In place: from the 64 KiB boundary at or below the buffer (views
        // find their bytes by address, `Shadow::offset`), past its end by the
        // 256 bytes a view ending in its last bytes reads where the
        // allocation has them, else to its end.
        const G: u64 = 64 << 10;
        let lo = key.0 / G * G;
        let part = [entry.end + 256, entry.end].into_iter().find_map(|end| {
            let hi = end.next_multiple_of(G);
            committed_part(lo, hi - lo, key.0, end)
        });
        if let Some((s0, l0)) = part
            && let Some((buffer, gpu, heap)) = in_place_buffer(s0, l0)
        {
            crate::host::name(&buffer, || format!("title buffer in place {:#x}+{:#x}", key.0, entry.end - key.0));
            entry.shadow = Some(Shadow::new_with(buffer, s0 as *mut u8, gpu, s0, l0, Some(heap)));
            changed();
        }
    }
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access): only the others'
        // copies may be in upload memory.
        let size = len.next_multiple_of(1 << 16);
        let host = host()?;
        let upload = entry
            .read_only
            .then(|| host.gpu_upload_buffer(size))
            .flatten();
        let in_upload = upload.is_some();
        // A buffer shaders may write: in video memory too, with a mirror.
        let mut vram = None;
        let (buffer, data) = match upload {
            Some(made) => made,
            None => {
                let mirror = host.shared_buffer(size)?;
                match (!entry.read_only && host.gpu_upload_writes())
                    .then(|| host.gpu_upload_buffer(size))
                    .flatten()
                {
                    Some((video, video_data)) => {
                        vram = Some(Vram {
                            mirror: mirror.0,
                            data: video_data,
                            stale: Mutex::new(vec![0; (size as usize).div_ceil(PAGE).div_ceil(64)]),
                            ahead: Mutex::new(Vec::new()),
                        });
                        (video, mirror.1)
                    }
                    None => mirror,
                }
            }
        };
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let shadow = Shadow::new_at(buffer, data, gpu, key.0, len, None, vram);
        shadow
            .upload
            .store(in_upload, std::sync::atomic::Ordering::Relaxed);
        entry.shadow = Some(shadow);
        changed();
    }
    entry.shadow.clone()
}

/// Title buffers and loose windows are the title's memory itself, opened as
/// a host heap, instead of copies (#880): `VICTORY_IN_PLACE=1` on any host,
/// another value never. Unset, on a host whose GPU shares memory with the
/// CPU, where title memory opened in place passed the self-test
/// (`crate::unified`): there shaders reach every placement at the same
/// speed (an AMD APU measured all within noise of each other), so copies
/// only cost their refreshes and copy-backs. Elsewhere, unset is off: on a
/// discrete GPU in place reads cross the bus as the copies do, and that it
/// gains in a title is not shown yet.
fn in_place_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_IN_PLACE") {
        Some(v) => v == "1",
        None => {
            let Some(host) = host().filter(|h| h.memory.uma) else {
                return false;
            };
            match crate::unified::self_test(&host.device) {
                Ok(()) => {
                    tracing::info!(
                        "title memory in place: the host GPU shares memory with the CPU \
                         (VICTORY_IN_PLACE=0 keeps copies)"
                    );
                    true
                }
                Err(why) => {
                    tracing::warn!(
                        %why,
                        "the host GPU shares memory with the CPU, but title memory does not \
                         open in place there: copies kept"
                    );
                    false
                }
            }
        }
    })
}

/// Registers title memory `[at, at + len)`, which `heap` holds in place,
/// with `crate::pinned`: the title's frees, protections and remaps there,
/// which fail while the heap lives, first let go of it (`release`: what is
/// placed in it stops being handed out; lists still holding it keep it
/// until they are reset). The registration goes with the heap.
fn pin_heap(
    heap: &windows::Win32::Graphics::Direct3D12::ID3D12Heap,
    at: u64,
    len: u64,
    release: impl FnOnce() + Send + 'static,
) {
    use windows::Win32::Graphics::Direct3D::ID3DDestructionNotifier;
    use windows::core::Interface;
    extern "system" fn forget(data: *mut std::ffi::c_void) {
        // SAFETY: the box registered below, given back once, when the heap
        // is destroyed.
        unsafe { Box::from_raw(data.cast::<crate::pinned::Pin>()) }.forget();
    }
    let pin = crate::pinned::pin(at as usize, len as usize, Box::new(release));
    let Ok(notifier) = heap.cast::<ID3DDestructionNotifier>() else {
        // Registered until the title changes the memory.
        return;
    };
    let data = Box::into_raw(Box::new(pin));
    // SAFETY: `forget` takes the box back when the heap goes.
    if unsafe { notifier.RegisterDestructionCallback(Some(forget), data.cast()) }.is_err() {
        // SAFETY: not registered: still ours. Dropped, it stays registered
        // until the title changes the memory.
        drop(unsafe { Box::from_raw(data) });
    }
}


/// A heap opened over a title allocation, its size and its id; `None` if
/// it could not be opened.
type OpenedHeap = Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64, HeapId)>;

/// Tells apart heaps opened over title memory, also two opened over one
/// allocation base (the title released the allocation and made another
/// there): what is let go of a stale heap is only what was placed in it.
type HeapId = u64;

fn next_heap_id() -> HeapId {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Host heaps opened over title allocations, by allocation base.
static IN_PLACE_HEAPS: Mutex<Option<std::collections::HashMap<usize, OpenedHeap>>> = Mutex::new(None);

/// The size of the allocation starting at `base` if all of it is still
/// committed memory of that allocation, else `None`.
fn allocation_extent(base: usize) -> Option<u64> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut at = base;
    loop {
        // SAFETY: plain data, filled by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0
            || info.AllocationBase as usize != base
        {
            return (at > base).then(|| (at - base) as u64);
        }
        if info.State != MEM_COMMIT {
            return None;
        }
        at = info.BaseAddress as usize + info.RegionSize;
    }
}

/// The smallest title buffer that goes in place (`VICTORY_IN_PLACE_MIN_KB`,
/// 1024 by default). Smaller ones keep their copies: shaders' atomics and
/// small reads of host memory cross the bus each time, and a once-a-frame
/// compute pass over a tiny buffer in host memory took 19–67 ms of GPU time
/// instead of 1–6 ms (measured by sirrus98 on an RTX 4090 for the in-place
/// work of fxceless/victory-fork#772, which keeps the same floor).
fn in_place_min() -> u64 {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| std::env::var("VICTORY_IN_PLACE_MIN_KB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(1024) << 10)
}

/// The share of an allocation's pages that must already be in memory
/// before it is opened as a heap (percent; `VICTORY_IN_PLACE_DENSITY`).
fn density_wanted() -> u64 {
    static WANTED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *WANTED.get_or_init(|| {
        std::env::var("VICTORY_IN_PLACE_DENSITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(90)
            .min(100)
    })
}

/// Density decisions by allocation (base and size: an allocation made
/// again at the same address is a new one), and when they were made.
type Decided = std::collections::HashMap<(usize, u64), (bool, std::time::Instant)>;
static DECIDED: Mutex<Option<Decided>> = Mutex::new(None);

/// Whether most of the allocation at `base` (`size` bytes) is already in
/// memory. Opening a heap over an allocation makes all of it resident:
/// importing it as host memory pins every page, also committed pages the
/// title never touched, which cost nothing until then (Grand Theft Auto VI
/// on Linux: 1.9 GB more anonymous memory against 1.4 GB fewer copies, so
/// in place cost more than the copies it replaced). A sparse allocation
/// keeps its copies. Asked again after 10 s, as the title fills its
/// allocations.
fn dense(base: usize, size: Option<u64>) -> bool {
    use windows_sys::Win32::System::ProcessStatus::{
        K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let Some(size) = size else { return false };
    if let Some(&(dense, when)) = DECIDED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|d| d.get(&(base, size)))
        && when.elapsed() < std::time::Duration::from_secs(10)
    {
        return dense;
    }
    // At most 64Ki pages asked, evenly spread over the allocation.
    let pages = (size / 4096).max(1);
    let step = pages.div_ceil(1 << 16);
    let mut asked: Vec<PSAPI_WORKING_SET_EX_INFORMATION> = (0..pages.div_ceil(step))
        .map(|i| PSAPI_WORKING_SET_EX_INFORMATION {
            VirtualAddress: (base as u64 + i * step * 4096) as *mut _,
            ..Default::default()
        })
        .collect();
    // SAFETY: our own process and an array of the size given.
    let ok = unsafe {
        K32QueryWorkingSetEx(
            GetCurrentProcess(),
            asked.as_mut_ptr().cast(),
            (asked.len() * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
        )
    } != 0;
    // SAFETY: the union's flags word; bit 0 is "valid" (in memory).
    let resident = asked
        .iter()
        .filter(|a| unsafe { a.VirtualAttributes.Flags } & 1 != 0)
        .count() as u64;
    let dense = ok && resident * 100 >= asked.len() as u64 * density_wanted();
    tracing::debug!(
        base = format_args!("{base:#x}"),
        size,
        resident,
        asked = asked.len(),
        dense,
        "title allocation density"
    );
    DECIDED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .insert((base, size), (dense, std::time::Instant::now()));
    dense
}

/// Once a frame: heaps opened over allocations the title has released or
/// changed since (another size, or not all committed) are let go with the
/// in-place windows and title buffers placed in them (by heap id). The title
/// frees some of its memory itself, unseen by us; what this cannot tell is
/// an allocation released and made again at the same address and size.
pub(crate) fn check_in_place() {
    let pending = std::mem::take(&mut *STALE_IN_PLACE.lock().unwrap_or_else(|e| e.into_inner()));
    for id in pending {
        let_go_heap(id);
    }
    if !in_place_enabled() {
        return;
    }
    let stale: Vec<HeapId> = {
        let heaps = IN_PLACE_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps
            .iter()
            .flat_map(|h| h.iter())
            .filter_map(|(&base, v)| v.as_ref().map(|(_, size, id)| (base, *size, *id)))
            .filter(|&(base, size, _)| allocation_extent(base) != Some(size))
            .map(|(_, _, id)| id)
            .collect()
    };
    for id in stale {
        let_go_heap(id);
    }
}

/// Heaps found stale while a caller held `BUFFERS` or `LOOSE`: what is placed
/// in them is let go at the next frame's check. By id: by then a new heap
/// may be open at the same base, with buffers placed in it.
static STALE_IN_PLACE: Mutex<Vec<HeapId>> = Mutex::new(Vec::new());

/// Lets go of heap `id` (if still cached) and the in-place windows and
/// title buffers placed in it, and of nothing else: not of a heap opened
/// later over the same memory, nor of what is placed in that one.
fn let_go_heap(id: HeapId) {
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let before = map.len();
    map.retain(|_, sh| sh.heap != Some(id));
    let mut gone = before - map.len();
    drop(map);
    for e in buffers().values_mut() {
        if e.shadow.as_ref().is_some_and(|sh| sh.heap == Some(id)) {
            e.shadow = None;
            gone += 1;
        }
    }
    if let Some(h) = IN_PLACE_HEAPS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        h.retain(|_, v| v.as_ref().is_none_or(|(_, _, i)| *i != id));
    }
    if gone > 0 {
        changed();
    }
}

/// Title memory `[at, at + len)` was mapped or unmapped: in-place windows
/// and title buffers over it, and heaps opened over allocations it touches show the old pages
/// (pinned while a list still holds them), so lookups get new ones.
pub(crate) fn in_place_remapped(at: u64, len: u64) {
    if in_place_enabled() {
        let_go_in_place(at, len);
    }
}

/// Lets go of in-place windows and title buffers over `[at, at + len)` and
/// heaps opened over allocations it touches.
fn let_go_in_place(at: u64, len: u64) {
    let end = at.saturating_add(len);
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let before = map.len();
    map.retain(|&(s, l), sh| !(sh.in_place && s < end && at < s + l));
    let mut gone = before - map.len();
    drop(map);
    for e in buffers().values_mut() {
        if e.shadow.as_ref().is_some_and(|sh| sh.in_place && sh.start < end && at < sh.start + sh.len) {
            e.shadow = None;
            gone += 1;
        }
    }
    let mut heaps = IN_PLACE_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps.as_mut() {
        h.retain(|&base, v| match v {
            Some((_, size, _)) => !((base as u64) < end && at < base as u64 + *size),
            None => true,
        });
    }
    drop(heaps);
    if gone > 0 {
        changed();
    }
}

/// The part of window `[start, start + len)` around `[address, end)` that
/// is committed memory of one allocation, cut to 64 KiB: `None` unless it
/// holds `[address, end)`.
fn committed_part(start: u64, len: u64, address: u64, end: u64) -> Option<(u64, u64)> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    const G: u64 = 64 << 10;
    let query = |at: u64| -> Option<MEMORY_BASIC_INFORMATION> {
        // SAFETY: plain data, filled by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        (unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } != 0).then_some(info)
    };
    let first = query(address)?;
    if first.State != MEM_COMMIT {
        return None;
    }
    let base = first.AllocationBase as u64;
    // Down from the region holding `address`, while committed and the same allocation.
    let mut lo = first.BaseAddress as u64;
    while lo > start {
        match query(lo - 1) {
            Some(i) if i.State == MEM_COMMIT && i.AllocationBase as u64 == base => lo = i.BaseAddress as u64,
            _ => break,
        }
    }
    let mut hi = first.BaseAddress as u64 + first.RegionSize as u64;
    while hi < start + len {
        match query(hi) {
            Some(i) if i.State == MEM_COMMIT && i.AllocationBase as u64 == base => hi = i.BaseAddress as u64 + i.RegionSize as u64,
            _ => break,
        }
    }
    // A placed buffer starts at a multiple of 64 KiB; its end only needs to
    // be committed pages.
    let lo = lo.max(start).next_multiple_of(G);
    let hi = (hi.min(start + len)) / 4096 * 4096;
    (lo <= address && end <= hi).then(|| (lo, hi - lo))
}

/// A host buffer that is the title memory `[start, start + len)` itself:
/// placed at its offset in a heap opened over the allocation holding it.
/// `None` where that cannot be (not one committed allocation, the heap
/// refused, a placement off 64 KiB). With the id of the heap it is placed in.
fn in_place_buffer(start: u64, len: u64) -> Option<(ID3D12Resource, u64, HeapId)> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::Interface;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let refuse = |why: &'static str| {
        #[cfg(feature = "diagnostics")]
        {
            let mut r = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
            *r.entry(why).or_insert(0) += 1;
        }
        let _ = why;
        None
    };
    // The whole range committed, in one allocation.
    let mut base = 0usize;
    let mut at = start as usize;
    while at < (start + len) as usize {
        // SAFETY: plain data, filled by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 || info.State != MEM_COMMIT {
            return refuse("not committed");
        }
        let b = info.AllocationBase as usize;
        if base == 0 {
            base = b;
        } else if b != base {
            return refuse("several allocations");
        }
        at = info.BaseAddress as usize + info.RegionSize;
    }
    let offset = start - base as u64;
    if !offset.is_multiple_of(64 << 10) {
        return refuse("offset");
    }
    let host = host()?;
    // Where the host opens a sub-range, only the buffer's own pages become
    // a heap: nothing else of the allocation is pinned, and the placed
    // buffer holds its heap (`place_in`), so the heap goes with the buffer.
    if !whole_allocations_only()
        && let Ok(dev13) = host.device.cast::<ID3D12Device13>()
    {
        let size = len.next_multiple_of(4096) as usize;
        // SAFETY: committed pages of one allocation of the title's, from
        // a 64 KiB boundary of it.
        let opened = unsafe { dev13.OpenExistingHeapFromAddress1::<ID3D12Heap>(start as *const _, size) };
        if let Ok(heap) = opened {
            let id = next_heap_id();
            pin_heap(&heap, start, size as u64, move || let_go_in_place(start, len));
            return place_in(&host.device, &heap, 0, len)
                .map(|(buffer, gpu)| (buffer, gpu, id))
                .or_else(|| refuse("placement refused"));
        }
    }
    // A heap opened earlier over an allocation the title has released or
    // changed since goes first, with what is placed in it.
    let extent = allocation_extent(base);
    let stale = {
        let heaps = IN_PLACE_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps
            .as_ref()
            .and_then(|h| h.get(&base))
            .and_then(|v| v.as_ref().map(|(_, size, id)| (*size, *id)))
            .filter(|&(size, _)| extent != Some(size))
    };
    // Only the heap goes here: the callers hold `BUFFERS` or `LOOSE`, which
    // letting go of what is placed in it takes, so that waits for the next
    // frame's check ([`check_in_place`]), outside those locks.
    let mut heaps = IN_PLACE_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, id)) = stale {
        if let Some(h) = heaps.as_mut() {
            h.remove(&base);
        }
        STALE_IN_PLACE.lock().unwrap_or_else(|e| e.into_inner()).push(id);
    }
    if !heaps.as_ref().is_some_and(|h| h.contains_key(&base)) && !dense(base, extent) {
        return refuse("sparse");
    }
    let heap = heaps.get_or_insert_with(Default::default).entry(base).or_insert_with(|| {
        let dev3: ID3D12Device3 = host.device.cast().ok()?;
        // SAFETY: the base of a committed allocation of the title's.
        let heap: ID3D12Heap = unsafe { dev3.OpenExistingHeapFromAddress(base as *const _) }.ok()?;
        // SAFETY: a plain query.
        let size = unsafe { heap.GetDesc() }.SizeInBytes;
        tracing::debug!(base = format_args!("{base:#x}"), size, "title allocation opened as a host heap");
        let id = next_heap_id();
        pin_heap(&heap, base as u64, size, move || let_go_heap(id));
        Some((heap, size, id))
    });
    let Some((heap, size, id)) = heap.clone() else {
        return refuse("heap refused");
    };
    drop(heaps);
    if offset + len > size {
        return refuse("past the heap");
    }
    place_in(&host.device, &heap, offset, len)
        .map(|(buffer, gpu)| (buffer, gpu, id))
        .or_else(|| refuse("placement refused"))
}

/// Tests of the whole-allocation path (the heap cache, stale heaps) turn
/// off sub-range heaps, which a host with `ID3D12Device13` would take.
#[cfg(test)]
static WHOLE_ALLOCATIONS_ONLY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn whole_allocations_only() -> bool {
    #[cfg(test)]
    return WHOLE_ALLOCATIONS_ONLY.load(Ordering::Relaxed);
    #[cfg(not(test))]
    false
}

/// A buffer of `len` bytes placed at `offset` in `heap`, opened over title
/// memory, and its GPU address.
fn place_in(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    heap: &windows::Win32::Graphics::Direct3D12::ID3D12Heap,
    offset: u64,
    len: u64,
) -> Option<(ID3D12Resource, u64)> {
    use windows::Win32::Graphics::Direct3D12::*;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a buffer inside the heap.
    unsafe {
        device.CreatePlacedResource(
            heap,
            offset,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    }
    .ok()?;
    let buffer = buffer?;
    // The buffer holds its heap: its pin (`pin_heap`) goes when the heap is
    // destroyed, which must not be while anything is placed in it. Native
    // D3D12 keeps the heap for its placed resources, but vkd3d-proton
    // destroys it, for its destruction callbacks, once its own references
    // go, and the sub-range path keeps none.
    // SAFETY: a live buffer and heap; the buffer's private data holds a
    // reference to the heap until the buffer goes.
    unsafe { buffer.SetPrivateDataInterface(&HOLDS_ITS_HEAP, heap) }.ok()?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    Some((buffer, gpu))
}

/// The private data by which a buffer placed over title memory holds its
/// heap.
const HOLDS_ITS_HEAP: windows::core::GUID =
    windows::core::GUID::from_u128(0x6f1d_2c3a_9b84_4e57_a0c2_51e8_7d93_b41f);

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches. A window is 256 KiB:
/// lookups while Grand Theft Auto VI loads its world touched about 30% of
/// the 4 MiB windows this used to copy, so most of their bytes (GBs of host
/// memory) were never asked for; smaller windows cost more of them, each a
/// host buffer and a page-table entry.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 256 << 10;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

/// The longest window in `LOOSE`. Changed under its lock.
static LOOSE_SPAN: AtomicU64 = AtomicU64::new(0);

/// Below this nothing is ever mapped (the null page and its guard area).
const LOWEST_MAPPED: u64 = 0x10000;

/// The window copy holding `[address, end)` among `windows` (by start and
/// length, none longer than `span`), the innermost if several do.
fn enclosing_window<V>(windows: &BTreeMap<(u64, u64), V>, span: u64, address: u64, end: u64) -> Option<&V> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    windows
        .range((lowest, 0)..=(address, u64::MAX))
        .rev()
        .find(|&(&(s, l), _)| s.saturating_add(l) >= end)
        .map(|(_, v)| v)
}

/// The window copy for `[address, end)` outside buffers: one already made
/// that holds it (a range spanning two windows has one of its own, which
/// later ranges in it share), else a new one. None for memory that is not
/// committed (a stale or null address): 4 MiB of zeros would not help.
fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if address < LOWEST_MAPPED {
        return None;
    }
    let start = address & !(WINDOW - 1);
    let len = end.checked_next_multiple_of(WINDOW)? - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = enclosing_window(&map, LOOSE_SPAN.load(Ordering::Relaxed), address, end) {
        return Some(shadow.clone());
    }
    if !crate::texture::committed(address as *const u8, 1) {
        return None;
    }
    if in_place_enabled()
        && let Some((s0, l0)) = committed_part(start, len, address, end)
        && let Some((buffer, gpu, heap)) = in_place_buffer(s0, l0)
    {
        crate::host::name(&buffer, || format!("title memory in place {s0:#x}+{l0:#x}"));
        let shadow = Shadow::new_with(buffer, s0 as *mut u8, gpu, s0, l0, Some(heap));
        LOOSE_SPAN.fetch_max(l0, Ordering::Relaxed);
        map.insert((s0, l0), shadow.clone());
        changed();
        return Some(shadow);
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Shadow::new(buffer, data, gpu, start, len);
    LOOSE_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert((start, len), shadow.clone());
    changed();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len, element).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Why title memory stayed copied instead of going in place, by reason
/// (diagnostics).
#[cfg(feature = "diagnostics")]
static REFUSED: Mutex<std::collections::BTreeMap<&'static str, u64>> = Mutex::new(std::collections::BTreeMap::new());

/// Logs [`VIEWS`] per frame over `frames`, and why ranges stayed copied.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let refused = std::mem::take(&mut *REFUSED.lock().unwrap_or_else(|e| e.into_inner()));
    if !refused.is_empty() {
        tracing::info!(?refused, "title memory kept as a copy, not in place (since the line before)");
    }
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    let (chunks, held) = {
        let a = arena();
        // SAFETY: live buffers.
        let size = |c: &(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)| unsafe { c.0.GetDesc() }.Width;
        let held: u64 = a.chunks.iter().map(|c| size(c) - c.3.values().sum::<u64>()).sum();
        (a.chunks.len(), held)
    };
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live in {chunks} shared buffers ({:.1} MB held)",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3,
            held as f64 / 1e6
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64, element: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len, element)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    // A run of a shared buffer, its start a whole number of elements in.
    let (buffer, data, gpu, slice) = arena().take(len, element)?;
    let shadow = Shadow::new_at(buffer, data, gpu, address, len, Some(slice), None);
    map.insert((address, len, element), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// A run of an [`Arena`] buffer: the buffer, where the run starts, its
/// bytes.
#[derive(Clone, Copy)]
struct Slice {
    chunk: usize,
    offset: u64,
    len: u64,
}

/// Shared buffers for the copies of single views ([`exact`]): they are
/// small (~2 KB on average) and come and go by the hundred a frame, and a
/// buffer of their own each cost a kernel call to make and one to free on
/// the title's threads and the submitting one (300-820 a frame), and 64 KiB
/// of memory each (~5,000 live).
#[derive(Default)]
struct Arena {
    /// Each buffer, its mapping and host address, and its free runs
    /// (offset → bytes).
    chunks: Vec<(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)>,
}

/// Bytes of each arena buffer (a larger copy gets one of its size).
const CHUNK: u64 = 16 << 20;

static ARENA: Mutex<Arena> = Mutex::new(Arena { chunks: Vec::new() });

fn arena() -> std::sync::MutexGuard<'static, Arena> {
    ARENA.lock().unwrap_or_else(|e| e.into_inner())
}

impl Arena {
    /// A run of `len` bytes starting at a multiple of `element` (and of 256)
    /// in a shared buffer: the buffer, the run's mapping and host address.
    fn take(&mut self, len: u64, element: u64) -> Option<(ID3D12Resource, *mut u8, u64, Slice)> {
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let align = 256 / gcd(256, element.max(1)) * element.max(1);
        let len = len.max(1).next_multiple_of(4);
        let fits = |free: &BTreeMap<u64, u64>| {
            free.iter().find_map(|(&at, &n)| {
                let start = at.next_multiple_of(align);
                (start + len <= at + n).then_some((at, n, start))
            })
        };
        let found = self.chunks.iter().enumerate().find_map(|(c, chunk)| fits(&chunk.3).map(|f| (c, f)));
        let (chunk, (at, n, start)) = match found {
            Some(found) => found,
            None => {
                let size = CHUNK.max((len + align).next_multiple_of(1 << 16));
                let (buffer, data) = host()?.shared_buffer(size)?;
                crate::host::name(&buffer, || format!("title views {size:#x}"));
                // SAFETY: a live buffer.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                let free = BTreeMap::from([(0, size)]);
                self.chunks.push((buffer, data as usize, gpu, free));
                let c = self.chunks.len() - 1;
                (c, fits(&self.chunks[c].3)?)
            }
        };
        let (buffer, data, gpu, free) = &mut self.chunks[chunk];
        free.remove(&at);
        if start > at {
            free.insert(at, start - at);
        }
        if at + n > start + len {
            free.insert(start + len, at + n - start - len);
        }
        let slice = Slice { chunk, offset: start, len };
        Some((buffer.clone(), (*data + start as usize) as *mut u8, *gpu + start, slice))
    }

    /// A run goes back, joined to its free neighbours.
    fn give_back(&mut self, slice: Slice) {
        let Some((_, _, _, free)) = self.chunks.get_mut(slice.chunk) else { return };
        let (mut at, mut n) = (slice.offset, slice.len);
        if let Some((&before, &m)) = free.range(..at).next_back()
            && before + m == at
        {
            free.remove(&before);
            (at, n) = (before, n + m);
        }
        if let Some(m) = free.remove(&(at + n)) {
            n += m;
        }
        free.insert(at, n);
    }
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> =
        buffers_read().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers_read()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod bench;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::*;

    /// The copy holding a view's first byte holds it to that copy's end:
    /// a lookup of that whole range finds the same copy, in an inner buffer
    /// inside an outer one, in the outer one, and in loose memory in no
    /// buffer. Constant views are looked up once so (`draw`'s `readable`).
    #[test]
    fn the_copy_holding_a_views_first_byte_holds_it_to_its_end() {
        if host().is_none() {
            return;
        }
        let same = |address: u64, size: u64| {
            let first = find(address, 1).expect("a copy");
            let len = size.min(first.left(address));
            let whole = find(address, len).expect("a copy");
            assert!(Arc::ptr_eq(&first, &whole), "{address:#x}+{size}");
        };
        let memory = vec![0u8; 4 * PAGE];
        let base = memory.as_ptr() as u64;
        let page = PAGE as u64;
        // Resource ids of their own (the addresses are the memory's).
        let (outer, inner) = (base as usize | 1, base as usize | 2);
        register(outer, base, 4 * page);
        register(inner, base + page, page);
        // In the outer buffer; in the inner one; past the inner one's end.
        for (address, size) in [(base + 16, 10_000), (base + page + 256, 10_000), (base + 2 * page - 100, 10_000)] {
            same(address, size);
        }
        unregister(inner, base + page);
        unregister(outer, base);
        drop(memory);
        // Loose memory: copies of the windows it touches.
        let loose = vec![0u8; 2 * PAGE];
        let address = loose.as_ptr() as u64 + 64;
        same(address, 1 << 20);
        same(address, 64);
    }

    /// The views' written-pages bitmaps, which shaders that mark their
    /// writes set every frame, are kept resident.
    #[test]
    fn the_written_pages_bitmap_is_kept_resident() {
        for memory in [false, true] {
            let Some(b) = bitmap(memory) else {
                return; // no host GPU here
            };
            assert_eq!(
                crate::host::kept(&b._buffer),
                Some(crate::host::Residency::Frame)
            );
        }
    }

    /// Runs this binary's test `name` again in a child process with
    /// `VICTORY_GPU_UPLOAD=1`, and `VICTORY_GPU_UPLOAD_WRITES=1` if `writes`
    /// (the switches are read once per process): true
    /// in that child, where the test does its work; false in the parent,
    /// once the child passed.
    fn rerun_with_gpu_upload(name: &str, writes: bool) -> bool {
        const CHILD: &str = "VICTORY_TEST_GPU_UPLOAD_CHILD";
        if std::env::var_os(CHILD).is_some() {
            return true;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                name,
                "--exact",
                "--test-threads=1",
                "--nocapture",
                "--include-ignored",
            ])
            .env(CHILD, "1")
            .env("VICTORY_GPU_UPLOAD", "1")
            .env("VICTORY_GPU_UPLOAD_WRITES", if writes { "1" } else { "0" })
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "{name} with VICTORY_GPU_UPLOAD=1:\n{text}"
        );
        assert!(text.contains("1 passed"), "{name} did not run:\n{text}");
        eprintln!("{text}");
        false
    }

    /// Read-only buffers come and go, more bytes of them in all than the
    /// GPU upload budget, a few at a time: each one's copy still goes in a
    /// GPU upload heap, and its bytes leave the budget once the last
    /// reference to its buffer is gone (a clone, as descriptor tables hold,
    /// keeps them charged). Counted for the whole run, copies went to
    /// system memory for good after the budget's worth (#2120).
    #[test]
    fn upload_budget_is_given_back_when_copies_go() {
        use windows::core::Interface;
        if !rerun_with_gpu_upload(
            "shadow::tests::upload_budget_is_given_back_when_copies_go",
            false,
        ) {
            return;
        }
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !host.gpu_upload() {
            return; // no GPU upload heaps (resizable BAR) here
        }
        const LEN: usize = 128 << 20;
        let p = crate::watch::tests::pool(LEN / PAGE);
        let base = crate::host::upload_bytes();
        // 2.5 GiB in all, past the 2 GiB budget.
        for round in 0..20 {
            register_read_only(p, p as u64, LEN as u64);
            let copy = find(p as u64, LEN as u64).expect("a copy");
            assert_eq!(
                heap_type(&copy),
                D3D12_HEAP_TYPE_GPU_UPLOAD,
                "round {round}: in a GPU upload heap"
            );
            assert_eq!(
                crate::host::kept(&copy.buffer),
                Some(crate::host::Residency::Frame),
                "round {round}: kept resident"
            );
            assert_eq!(
                crate::host::upload_bytes(),
                base + LEN as u64,
                "round {round}: charged"
            );
            // A clone of the buffer, as a descriptor table holds, outlives
            // the copy: still charged until it goes.
            let held = copy.buffer.clone();
            let weak = Arc::downgrade(&copy);
            unregister(p, p as u64);
            drop(copy);
            assert!(weak.upgrade().is_none(), "round {round}: the copy is gone");
            assert_eq!(
                crate::host::upload_bytes(),
                base + LEN as u64,
                "round {round}: held"
            );
            assert!(!held.as_raw().is_null());
            drop(held);
            assert_eq!(
                crate::host::upload_bytes(),
                base,
                "round {round}: given back"
            );
        }
        crate::watch::tests::free(p);
    }

    /// A buffer shaders may write, its copy made (mirrored in video memory
    /// when the switches are on), and title memory filled with a pattern
    /// the copy then took. Returns the pool, its length and the copy.
    fn mirrored_buffer(pages: usize) -> (usize, usize, Arc<Shadow>) {
        let p = crate::watch::tests::pool(pages);
        let len = pages * PAGE;
        register(p, p as u64, len as u64);
        let copy = find(p as u64, len as u64).expect("a copy");
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, len) };
        for (i, b) in title.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        copy.update(p, len);
        (p, len, copy)
    }

    /// Records into `list` the GPU writing `bytes` into the copy at title
    /// address `at` (a copy from a buffer of ours, as a shader would store).
    fn gpu_store(list: &HostList, copy: &Shadow, at: usize, bytes: &[u8]) {
        let host = host().unwrap();
        let (src, data) = host.shared_buffer(bytes.len() as u64).unwrap();
        // SAFETY: our mapped buffer of `bytes.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
        list.keep(&src);
        // SAFETY: an open list; both buffers live and common.
        unsafe {
            list.list.CopyBufferRegion(
                &copy.buffer,
                copy.offset(at as u64),
                &src,
                0,
                bytes.len() as u64,
            )
        };
    }

    /// The bytes of the copy as the GPU sees them (video memory, or the
    /// system memory copy), copied out by the GPU.
    fn gpu_bytes(copy: &Shadow, at: usize, n: usize) -> Vec<u8> {
        let host = host().unwrap();
        let (out, data) = host.shared_buffer(n as u64).unwrap();
        host.copy_now(&[(&out, 0, &copy.buffer, copy.offset(at as u64), n as u64)]);
        // SAFETY: our mapped buffer, the copy done.
        unsafe { std::slice::from_raw_parts(data, n) }.to_vec()
    }

    /// With `VICTORY_GPU_UPLOAD_WRITES=1` too, the copy of a buffer shaders
    /// may write is in a GPU upload heap, mirrored in system memory: the
    /// GPU sees what a refresh wrote, and the mirror holds the same.
    #[test]
    fn written_buffers_copies_are_in_video_memory_with_a_mirror() {
        if !rerun_with_gpu_upload(
            "shadow::tests::written_buffers_copies_are_in_video_memory_with_a_mirror",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return; // no GPU upload heaps here
        }
        let (p, len, copy) = mirrored_buffer(2);
        assert!(copy.mirrored());
        assert_eq!(heap_type(&copy), D3D12_HEAP_TYPE_GPU_UPLOAD);
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert!(
            gpu_bytes(&copy, p, len) == title,
            "the GPU sees the refreshed bytes"
        );
        // The refresh wrote video memory only; the mirror catches up when
        // the CPU reads it.
        // SAFETY: the mirror's bytes (read again after `freshen` below).
        let mirror = unsafe { std::slice::from_raw_parts(copy.data, len) };
        assert!(mirror != title, "the refresh did not write the mirror");
        copy.freshen(0, len);
        // SAFETY: as above.
        let mirror = unsafe { std::slice::from_raw_parts(copy.data, len) };
        assert!(mirror == title, "the mirror has caught up");
        // The title writes again; the next refresh reaches video memory.
        // SAFETY: our pool's bytes.
        unsafe { ((p + PAGE + 100) as *mut u8).write_volatile(0xA5) };
        copy.update(p, len);
        assert_eq!(gpu_bytes(&copy, p + PAGE + 100, 1), [0xA5]);
        unregister(p, p as u64);
    }

    /// End to end through a submission: the GPU writes the copy in video
    /// memory, in a range a list declared (`marked`: and in a page shaders
    /// marked as written), while the title writes another byte of the same
    /// page; afterwards title memory holds both writes. Without the mirror
    /// being brought up to date (`pull`) the GPU's bytes would never reach
    /// it, nor title memory.
    fn gpu_writes_reach_title_memory(marked: bool) {
        let host = host().unwrap();
        let (p, len, copy) = mirrored_buffer(2);
        assert!(copy.mirrored());
        let list = host.new_list().unwrap();
        let (at, gpu) = (p + PAGE + 64, [0xC3u8; 16]);
        copy.refresh(&list, p as u64, len as u64);
        copy.gpu_writes(&list, p as u64, len as u64, marked);
        gpu_store(&list, &copy, at, &gpu);
        list.close();
        // The title writes the same page meanwhile, past the GPU's bytes.
        // SAFETY: our pool's bytes.
        unsafe { ((p + PAGE + 200) as *mut u8).write_volatile(0x3C) };
        if marked {
            // The shaders' mark for the page (the GPU copy sets none).
            let (bit, _) = copy.mark_entry(at as u64).expect("marked memory");
            let words = bitmap(false).unwrap().data as *mut u32;
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *words.add(bit as usize / 32) |= 1 << (bit % 32) };
        }
        host.execute(&[&list]);
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert_eq!(
            title[PAGE + 64..PAGE + 80],
            gpu,
            "the GPU's bytes (marked {marked})"
        );
        assert_eq!(
            title[PAGE + 200],
            0x3C,
            "the title's byte (marked {marked})"
        );
        assert_eq!(
            title[PAGE + 63],
            ((PAGE + 63) % 251) as u8,
            "the rest as it was"
        );
        assert_eq!(
            gpu_bytes(&copy, p + PAGE + 64, 16),
            gpu,
            "video memory keeps them"
        );
        unregister(p, p as u64);
    }

    #[test]
    fn gpu_writes_to_mirrored_copies_reach_title_memory() {
        if !rerun_with_gpu_upload(
            "shadow::tests::gpu_writes_to_mirrored_copies_reach_title_memory",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return;
        }
        gpu_writes_reach_title_memory(false);
        if bitmap(false).is_some() {
            gpu_writes_reach_title_memory(true);
        }
    }

    /// A list whose writes stay out of title memory (a program using title
    /// memory read memory with no copy): the copy keeps the GPU's bytes
    /// until its next refresh takes title memory's again
    /// (`crate::memory::drop_withheld`), and so does its mirror (the CPU's
    /// next look at the range sees them), while title memory keeps its own.
    #[test]
    fn withheld_writes_stay_in_the_copy_and_its_mirror() {
        if !rerun_with_gpu_upload(
            "shadow::tests::withheld_writes_stay_in_the_copy_and_its_mirror",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(1);
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, len as u64);
        {
            // A program using title memory declares the write: only such
            // programs' writes are withheld when their list misses.
            let _writer = list.writer_uses_memory(true);
            copy.gpu_writes(&list, p as u64, len as u64, false);
        }
        gpu_store(&list, &copy, p + 8, &[0x99; 8]);
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        // SAFETY: our pool's bytes, and the mirror's.
        let (title, mirror) = unsafe {
            (
                std::slice::from_raw_parts(p as *const u8, len),
                std::slice::from_raw_parts(copy.data, len),
            )
        };
        assert_eq!(title[8], 8, "title memory keeps its byte");
        assert_eq!(mirror[8..16], [0x99; 8], "the mirror has the GPU's");
        unregister(p, p as u64);
    }

    /// The mirror falls behind where a refresh wrote video memory only, and
    /// catches up on what the CPU reads, and only there: a read of a page
    /// brings that page back; pages GPU writes are pulled for come back
    /// whole; a refresh where a snapshot is ahead writes both.
    #[test]
    fn mirrors_catch_up_on_the_pages_read() {
        if !rerun_with_gpu_upload("shadow::tests::mirrors_catch_up_on_the_pages_read", true) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(4);
        assert_eq!(
            copy.stale_pages(),
            [0, 1, 2, 3],
            "the refresh wrote video memory only"
        );
        // SAFETY: our pool's bytes, and the mirror's.
        let (title, mirror) = unsafe {
            (
                std::slice::from_raw_parts(p as *const u8, len),
                std::slice::from_raw_parts(copy.data, len),
            )
        };
        // A read of a few bytes of page 1.
        assert_eq!(
            copy.read_host((p + PAGE + 8) as u64, 1)[0].to_le_bytes(),
            title[PAGE + 8..PAGE + 12]
        );
        assert_eq!(copy.stale_pages(), [0, 2, 3]);
        assert!(
            mirror[PAGE..2 * PAGE] == title[PAGE..2 * PAGE],
            "page 1 came back whole"
        );
        assert!(
            mirror[2 * PAGE..3 * PAGE] != title[2 * PAGE..3 * PAGE],
            "page 2 did not"
        );
        // Bytes the GPU wrote in page 3 are pulled: the whole page comes.
        pull(&[(copy.clone(), p + 3 * PAGE + 100, 4)]);
        assert_eq!(copy.stale_pages(), [0, 2]);
        assert!(mirror[3 * PAGE..] == title[3 * PAGE..]);
        // A snapshot ahead over page 2: the refresh there writes both.
        let ahead = [(copy.clone(), p + 2 * PAGE + 10, 10)];
        snapshots_ahead(&ahead);
        assert_eq!(copy.stale_pages(), [0], "page 2 brought back up front");
        // SAFETY: our pool's bytes.
        unsafe {
            ((p + 2 * PAGE + 500) as *mut u8).write_volatile(0x11);
            ((p + 40) as *mut u8).write_volatile(0x22);
        }
        copy.update(p, len);
        assert_eq!(copy.stale_pages(), [0], "page 2 stays current");
        assert_eq!(mirror[2 * PAGE + 500], 0x11);
        snapshots_done(&ahead);
        // SAFETY: our pool's bytes.
        unsafe { ((p + 2 * PAGE + 600) as *mut u8).write_volatile(0x33) };
        copy.update(p, len);
        assert_eq!(copy.stale_pages(), [0, 2], "video memory only again");
        assert_eq!(gpu_bytes(&copy, p + 2 * PAGE + 600, 1), [0x33]);
        copy.freshen(0, len);
        assert!(copy.stale_pages().is_empty());
        assert!(mirror == title, "the mirror has all of it");
        unregister(p, p as u64);
    }

    /// A submission whose list declares shaders writing several ranges,
    /// apart, of a copy the refresh left behind in its mirror: their
    /// snapshots come back in one GPU copy, and what shaders write in
    /// another in one more (not one per range); title memory then has the
    /// GPU's bytes and the title's.
    #[test]
    fn snapshots_of_mirrored_copies_come_back_at_once() {
        if !rerun_with_gpu_upload(
            "shadow::tests::snapshots_of_mirrored_copies_come_back_at_once",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(4);
        assert_eq!(copy.stale_pages().len(), 4);
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, len as u64);
        for k in 0..4 {
            copy.gpu_writes(&list, (p + k * PAGE + 64) as u64, 256, false);
        }
        gpu_store(&list, &copy, p + 2 * PAGE + 100, &[0xE7; 8]);
        list.close();
        // The title writes page 3, inside its range, before the submission.
        // SAFETY: our pool's bytes.
        unsafe { ((p + 3 * PAGE + 70) as *mut u8).write_volatile(0x5A) };
        let before = BROUGHT.load(std::sync::atomic::Ordering::Relaxed);
        host.execute(&[&list]);
        let copies = BROUGHT.load(std::sync::atomic::Ordering::Relaxed) - before;
        assert_eq!(copies, 2, "the snapshots' pages, then what the GPU wrote");
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert_eq!(
            title[2 * PAGE + 100..2 * PAGE + 108],
            [0xE7; 8],
            "the GPU's bytes"
        );
        assert_eq!(title[3 * PAGE + 70], 0x5A, "the title's byte");
        assert_eq!(title[64], 64, "the rest as it was");
        unregister(p, p as u64);
    }

    /// A range declared with marks, over a copy the refresh left behind in
    /// its mirror: only the page shaders marked comes back (one GPU copy),
    /// not the range, and its bytes reach title memory.
    #[test]
    fn marked_ranges_of_mirrored_copies_bring_back_their_marked_pages_only() {
        if !rerun_with_gpu_upload(
            "shadow::tests::marked_ranges_of_mirrored_copies_bring_back_their_marked_pages_only",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() || bitmap(false).is_none() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(4);
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, len as u64);
        copy.gpu_writes(&list, p as u64, len as u64, true);
        let at = p + 2 * PAGE + 100;
        gpu_store(&list, &copy, at, &[0x4D; 8]);
        list.close();
        let (bit, _) = copy.mark_entry(at as u64).expect("marked memory");
        let words = bitmap(false).unwrap().data as *mut u32;
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *words.add(bit as usize / 32) |= 1 << (bit % 32) };
        let before = BROUGHT.load(std::sync::atomic::Ordering::Relaxed);
        host.execute(&[&list]);
        let copies = BROUGHT.load(std::sync::atomic::Ordering::Relaxed) - before;
        assert_eq!(copies, 1, "the marked page alone");
        assert_eq!(copy.stale_pages(), [0, 1, 3], "the other pages stay behind");
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert_eq!(title[2 * PAGE + 100..2 * PAGE + 108], [0x4D; 8]);
        unregister(p, p as u64);
    }

    /// Shaders write past the range their list declares (as through a root
    /// view): the bytes stay out of title memory, and the mirror is stale
    /// there. A later list declaring them snapshots the GPU's bytes, so its
    /// copy-back neither puts them back nor over a byte the title wrote
    /// during the submission.
    #[test]
    fn writes_past_declared_ranges_leave_the_mirror_stale() {
        if !rerun_with_gpu_upload(
            "shadow::tests::writes_past_declared_ranges_leave_the_mirror_stale",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(2);
        copy.freshen(0, len);
        // Page 0 declared, page 1 written past it.
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, len as u64);
        copy.gpu_writes(&list, p as u64, PAGE as u64, false);
        copy.may_write(&list, (p + PAGE) as u64, PAGE as u64);
        gpu_store(&list, &copy, p + PAGE + 8, &[0x77; 4]);
        list.close();
        host.execute(&[&list]);
        // SAFETY: our pool's bytes.
        let title = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        let before = |at: usize| (at % 251) as u8;
        assert_eq!(title(PAGE + 8), before(PAGE + 8), "not declared: not in title memory");
        assert_eq!(copy.stale_pages(), [1], "the mirror is behind there");
        // Page 1 declared; the title writes one of the GPU's bytes meanwhile.
        let list = host.new_list().unwrap();
        copy.refresh(&list, (p + PAGE) as u64, PAGE as u64);
        copy.gpu_writes(&list, (p + PAGE) as u64, PAGE as u64, false);
        // SAFETY: our pool's bytes.
        list.defer_host(move || unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(0x55) });
        list.close();
        host.execute(&[&list]);
        assert_eq!(title(PAGE + 8), 0x55, "the title's byte");
        assert_eq!(title(PAGE + 9), before(PAGE + 9), "the earlier GPU bytes stay out");
        unregister(p, p as u64);
    }

    /// Pages shaders using title memory marked while their results are held
    /// back stay out of title memory, and the mirror has the GPU's bytes
    /// there, as video memory does.
    #[test]
    fn held_back_marked_pages_come_back_to_the_mirror() {
        if !rerun_with_gpu_upload(
            "shadow::tests::held_back_marked_pages_come_back_to_the_mirror",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        let (true, Some(memory)) = (host.gpu_upload_writes(), bitmap(true)) else {
            return;
        };
        let (p, len, copy) = mirrored_buffer(2);
        copy.freshen(0, len);
        let list = host.new_list().unwrap();
        gpu_store(&list, &copy, p + PAGE + 8, &[0x66; 4]);
        list.close();
        host.execute(&[&list]);
        let (bit, _) = copy.mark_entry((p + PAGE) as u64).expect("marked memory");
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *(memory.data as *mut u32).add(bit as usize / 32) |= 1 << (bit % 32) };
        copy_back_marks(&[], true);
        // SAFETY: our pool's bytes, and the mirror's.
        let (title, mirror) = unsafe {
            (
                ((p + PAGE + 8) as *const u8).read_volatile(),
                std::slice::from_raw_parts(copy.data.add(PAGE + 8), 4),
            )
        };
        assert_eq!(title, ((PAGE + 8) % 251) as u8, "held back");
        assert_eq!(mirror, [0x66; 4], "the mirror has the GPU's bytes");
        unregister(p, p as u64);
    }

    /// The copy's tests against system memory copies, again with copies in
    /// video memory and their mirrors behind (snapshots, refreshes under a
    /// busy range, copy-back by bytes and by marked pages).
    #[test]
    fn copy_tests_pass_with_mirrored_copies() {
        if std::env::var_os("VICTORY_TEST_GPU_UPLOAD_CHILD").is_some() {
            return;
        }
        for name in [
            "shadow::tests::title_writes_under_a_busy_range_are_copied",
            "shadow::tests::snapshots_cover_the_declared_ranges_and_writes_go_back_once",
            "shadow::tests::marked_pages_go_back_by_the_bytes_the_gpu_changed",
        ] {
            rerun_with_gpu_upload(name, true);
        }
    }

    /// A shader reading and writing 256 MiB of a written buffer's copy, in
    /// system memory and (`VICTORY_GPU_UPLOAD_WRITES=1`) in video memory, and
    /// what bringing 64 marked pages back to the mirror costs.
    /// `cargo test --release -p victory-gpu --lib written_copies -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark: run in release with --ignored --nocapture"]
    fn written_copies_in_video_and_system_memory() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        if !rerun_with_gpu_upload(
            "shadow::tests::written_copies_in_video_and_system_memory",
            true,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload_writes() {
            eprintln!("no GPU upload heaps here");
            return;
        }
        const SRC: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer b : register(u0);
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // Read 16 dwords, write them back incremented: 64 bytes read and
    // written per thread.
    for (uint k = 0; k < 16; k++) {
        uint at = (id.x * 16 + k) * 4;
        b.Store(at, b.Load(at) + 1);
    }
}"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source outlives the call.
        unsafe {
            D3DCompile(
                SRC.as_ptr().cast(),
                SRC.len(),
                s!("b"),
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
        // SAFETY: the blob's bytes; plain creation.
        let (root, pso) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        const BYTES: usize = 256 << 20;
        // The system memory copy: a mirrored copy's mirror, used by the GPU
        // as the copies are without the switch.
        let p = crate::watch::tests::pool(BYTES / PAGE);
        register(p, p as u64, BYTES as u64);
        let copy = find(p as u64, BYTES as u64).unwrap();
        assert!(copy.mirrored());
        copy.update(p, BYTES);
        let mirror = &copy.vram.as_ref().unwrap().mirror;
        for (label, buffer) in [
            ("system memory (as without the switch)", mirror),
            ("video memory (GPU upload heap)", &copy.buffer),
        ] {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let list = host.new_list().unwrap();
                // SAFETY: an open list, live objects.
                unsafe {
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetPipelineState(&pso);
                    list.list
                        .SetComputeRootUnorderedAccessView(0, buffer.GetGPUVirtualAddress());
                    list.list.Dispatch((BYTES / 4 / 16 / 256) as u32, 1, 1);
                }
                list.close();
                let t = std::time::Instant::now();
                host.execute(&[&list]);
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "a shader reading and writing 256 MiB of the copy in {label}: {:.2} ms ({:.1} GB/s each way)",
                best * 1e3,
                BYTES as f64 / best / 1e9
            );
        }
        // Bringing written pages back: 64 scattered 64 KiB pages (4 MiB).
        let pages: Vec<_> = (0..64)
            .map(|k| (copy.clone(), p + k * 61 * PAGE % BYTES, PAGE))
            .collect();
        let mut best = f64::MAX;
        for _ in 0..10 {
            let t = std::time::Instant::now();
            pull(&pages);
            best = best.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "bringing 64 written 64 KiB pages back to the mirror: {:.3} ms",
            best * 1e3
        );
        let one = [(copy.clone(), p, PAGE)];
        let mut best = f64::MAX;
        for _ in 0..10 {
            let t = std::time::Instant::now();
            pull(&one);
            best = best.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "bringing 1 written page back (the fixed cost of a pull): {:.3} ms",
            best * 1e3
        );
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// The heap type a copy's buffer is in.
    fn heap_type(copy: &Shadow) -> D3D12_HEAP_TYPE {
        let mut props = D3D12_HEAP_PROPERTIES::default();
        // SAFETY: a live committed buffer.
        unsafe {
            copy.buffer
                .GetHeapProperties(Some(&mut props), None)
                .unwrap()
        };
        props.Type
    }

    /// With `VICTORY_GPU_UPLOAD=1`, the copy of a buffer made without
    /// unordered access is in a GPU upload heap: a refresh writes the
    /// title's bytes there, and the GPU reads them (copied out into system
    /// memory here). With `VICTORY_GPU_UPLOAD_WRITES=0`, a buffer shaders
    /// may write keeps its copy in system memory.
    #[test]
    fn read_only_buffers_get_their_copies_in_video_memory() {
        if !rerun_with_gpu_upload(
            "shadow::tests::read_only_buffers_get_their_copies_in_video_memory",
            false,
        ) {
            return;
        }
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !host.gpu_upload() {
            return; // no GPU upload heaps (resizable BAR) here
        }
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register_read_only(p, p as u64, len as u64);
        let copy = find(p as u64, len as u64).expect("a copy");
        assert!(copy.upload.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(heap_type(&copy), D3D12_HEAP_TYPE_GPU_UPLOAD);
        // The title writes; the copy is refreshed; the GPU reads it.
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, len) };
        for (i, b) in title.iter_mut().enumerate() {
            *b = (i % 253) as u8;
        }
        copy.update(p, len);
        let (out, data) = host.shared_buffer(len as u64).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: an open list; both buffers are live and start common.
        unsafe {
            list.list
                .CopyBufferRegion(&out, 0, &copy.buffer, 0, len as u64)
        };
        list.close();
        host.execute(&[&list]);
        // SAFETY: the mapped output, `len` bytes, the copy done.
        let read = unsafe { std::slice::from_raw_parts(data, len) };
        assert!(
            read == &title[..],
            "what the GPU read is what the title wrote"
        );
        unregister(p, p as u64);
        // A buffer shaders may write.
        let q = crate::watch::tests::pool(1);
        register(q, q as u64, PAGE as u64);
        let writable = find(q as u64, PAGE as u64).expect("a copy");
        assert!(!writable.upload.load(std::sync::atomic::Ordering::Relaxed));
        assert_ne!(heap_type(&writable), D3D12_HEAP_TYPE_GPU_UPLOAD);
        unregister(q, q as u64);
    }

    /// A shader reading 256 MiB of a read-only buffer's copy (GPU upload
    /// heap) and of a writable one's (system memory), with
    /// `VICTORY_GPU_UPLOAD=1`.
    /// `cargo test --release -p victory-gpu --lib reading_copies -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark: run in release with --ignored --nocapture"]
    fn reading_copies_in_video_and_system_memory() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::{Interface, s};
        if !rerun_with_gpu_upload(
            "shadow::tests::reading_copies_in_video_and_system_memory",
            false,
        ) {
            return;
        }
        let Some(host) = host() else {
            return;
        };
        if !host.gpu_upload() {
            eprintln!("no GPU upload heaps here");
            return;
        }
        const SRC: &str = r#"
#define RS "SRV(t0), UAV(u0)"
ByteAddressBuffer b : register(t0);
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint s = 0;
    for (uint k = 0; k < 16; k++) s += b.Load((id.x * 16 + k) * 4);
    if (s == 0x12345678) o.Store(0, s);
}"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source outlives the call.
        unsafe {
            D3DCompile(
                SRC.as_ptr().cast(),
                SRC.len(),
                s!("b"),
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
        // SAFETY: the blob's bytes; plain creation.
        let (root, pso) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        const BYTES: usize = 256 << 20;
        let (out, _) = host.shared_buffer(256).unwrap();
        for read_only in [false, true] {
            let p = crate::watch::tests::pool(BYTES / PAGE);
            if read_only {
                register_read_only(p, p as u64, BYTES as u64);
            } else {
                register(p, p as u64, BYTES as u64);
            }
            let copy = find(p as u64, BYTES as u64).unwrap();
            // The first refresh copies all of it (every page is new).
            let t = std::time::Instant::now();
            copy.update(p, BYTES);
            let refresh = t.elapsed().as_secs_f64();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let list = host.new_list().unwrap();
                // SAFETY: an open list, live objects.
                unsafe {
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetPipelineState(&pso);
                    list.list.SetComputeRootShaderResourceView(0, copy.gpu);
                    list.list
                        .SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
                    list.list.Dispatch((BYTES / 4 / 16 / 256) as u32, 1, 1);
                }
                list.close();
                let t = std::time::Instant::now();
                host.execute(&[&list]);
                best = best.min(t.elapsed().as_secs_f64());
            }
            let _ = copy.buffer.as_raw();
            eprintln!(
                "{}: shader reads 256 MiB in {:.2} ms ({:.1} GB/s); refresh writes it in {:.1} ms ({:.1} GB/s)",
                if read_only {
                    "read-only buffer's copy (GPU upload heap)"
                } else {
                    "writable buffer's copy (system memory)"
                },
                best * 1e3,
                BYTES as f64 / best / 1e9,
                refresh * 1e3,
                BYTES as f64 / refresh / 1e9
            );
            unregister(p, p as u64);
            crate::watch::tests::free(p);
        }
    }

    /// A shader's read mark of heap slot `slot` in the bitmap of shaders
    /// that use title memory (`memory`) or of the others.
    pub(crate) fn mark_read(memory: bool, slot: u32) {
        let b = bitmap(memory).expect("a host GPU");
        let bit = victory_shader::emit::VIEW_READS + slot;
        // SAFETY: a bit of the mapped bitmap; no submission runs (the caller
        // holds the submission lock).
        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
    }

    /// A shader's read mark of view heap page `page`, as [`mark_read`].
    pub(crate) fn mark_page_read(memory: bool, page: u32) {
        let b = bitmap(memory).expect("a host GPU");
        let bit = victory_shader::emit::VIEW_PAGE_READS + page;
        // SAFETY: as in `mark_read`.
        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    /// A title buffer over a pool and, after it, a plain allocation (memory
    /// that is not the runtime's): the pool is tracked and guarded; the
    /// plain allocation's bytes are copied at each use and never guarded,
    /// so a value the title keeps there reaches the copy, and again after
    /// the title changes it.
    #[test]
    fn bytes_off_the_runtimes_memory_are_copied_at_each_use() {
        use crate::watch::tests::{MEM_WRITE_WATCH, free};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualQuery,
        };
        // A pool page and a plain one right after it, in two pages found free
        // (other tests allocate meanwhile: tried again if taken).
        let (pool, plain) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations where it was, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let at = |q: usize, kind: u32| {
                        VirtualAlloc(q as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (pool, plain) = (at(p, MEM_WRITE_WATCH), at(p + PAGE, 0));
                    if p != 0 && pool == p && plain == p + PAGE {
                        return Some((pool, plain));
                    }
                    [pool, plain].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("two free pages");
        let len = 2 * PAGE;
        register(pool, pool as u64, len as u64);
        let Some(copy) = find(pool as u64, len as u64) else {
            unregister(pool, pool as u64);
            [pool, plain].into_iter().for_each(free);
            return; // no host GPU here
        };
        let word = (plain + 64) as *mut u32;
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        // SAFETY: a word of our plain allocation.
        unsafe { word.write_volatile(0x1234) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x1234]);
        assert!(!copy.clean(), "never clean: part of it is not tracked");
        // SAFETY: as above.
        unsafe { word.write_volatile(0x5678) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x5678], "the title's change reaches the copy");
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(plain), PAGE_READWRITE, "the plain allocation is not");
        unregister(pool, pool as u64);
        // The pages stay allocated (as the other tests' pools): a copy some
        // page table still holds must not name memory reused meanwhile.
    }

    /// Runs of the shared buffers start a whole number of elements (and of
    /// 256 bytes) in, do not overlap, and join again when given back.
    #[test]
    fn arena_runs_are_aligned_to_their_elements_and_join_again() {
        if host().is_none() {
            return; // no host GPU here
        }
        let mut arena = Arena::default();
        let (_, _, _, a) = arena.take(100, 456).unwrap();
        let (_, _, _, b) = arena.take(5000, 16).unwrap();
        let (_, _, _, c) = arena.take(4, 4).unwrap();
        assert!(a.offset % 456 == 0 && a.offset % 256 == 0 && b.offset % 256 == 0);
        let mut runs = [a, b, c].map(|s| (s.offset, s.offset + s.len));
        runs.sort_unstable();
        assert!(runs.windows(2).all(|w| w[0].1 <= w[1].0), "{runs:?}");
        for s in [b, a, c] {
            arena.give_back(s);
        }
        assert_eq!(arena.chunks.len(), 1);
        assert_eq!(arena.chunks[0].3.len(), 1, "one free run again");
        // Larger than a buffer: one of its own.
        let (_, _, _, d) = arena.take(CHUNK + 1, 4).unwrap();
        assert_eq!((arena.chunks.len(), d.chunk), (2, 1));
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64, false);
        copy.writing(p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE, false);
        copy.writing(p + PAGE, PAGE, false);
        copy.writing(p + 80, 8, false);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "merged where they overlap or touch");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        // No submission takes the marks set below meanwhile.
        let _quiet = crate::host::tests::no_submissions();
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(p, len);
        copy.writing(p, len, true);
        assert!(copy.before.lock().unwrap().0.is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1.
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], false);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(copy.copy_back(p, len), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// Marks of shaders using title memory stay out of title memory while
    /// their results are held back, and are cleared; the other shaders'
    /// marks of the same submission go back.
    #[test]
    fn marks_of_shaders_using_title_memory_are_held_back() {
        // No submission takes the marks set below meanwhile.
        let _quiet = crate::host::tests::no_submissions();
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(others), Some(memory)) = (find(p as u64, len as u64), bitmap(false), bitmap(true)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        // Page 0 marked by another shader, page 1 by one using title memory.
        for (b, n) in [(others, base), (memory, base + 1)] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], true);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(PAGE + 10)), (0xaa, 0), "page 1 held back");
        // SAFETY: a dword of the mapped bitmap.
        assert_eq!(unsafe { *(memory.data as *const u32).add((base + 1) / 32) } & 1 << ((base + 1) % 32), 0, "cleared");
        unregister(p, p as u64);
    }

    /// What a frame withheld leaves every host copy of its memory at the
    /// frame's end (run textures9's shape): in frame N a program using title
    /// memory writes a record into the copies and the frame misses, so it
    /// stays out of title memory but in the copies for the GPU; in frame
    /// N+1, which misses nothing, a program reading the record must see
    /// title memory's bytes, not frame N's, in a title buffer's copy and in
    /// an inner buffer's copy of the same pages alike.
    #[test]
    fn what_a_frame_withholds_leaves_the_copies_at_its_end() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        // SAFETY: a byte of our pool.
        unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
        register(p, p as u64, len as u64);
        register(p + 1, p as u64, PAGE as u64);
        let (Some(outer), Some(inner)) = (find(p as u64, len as u64), find(p as u64 + 16, 16)) else {
            unregister(p + 1, p as u64);
            unregister(p, p as u64);
            return; // no host GPU here
        };
        assert!(!Arc::ptr_eq(&outer, &inner));
        let byte = |c: &Shadow| c.read_host(p as u64 + 16, 1)[0] & 0xff;
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
            c.writing(start as usize, n as usize, false);
            c.write_host(p as u64 + 16, &[0x22]);
            c.keep_writes(start as usize, n as usize);
        }
        crate::memory::drop_withheld(p + 16, 4);
        assert_eq!((byte(&outer), byte(&inner)), (0x22, 0x22), "frame N: the GPU's copies keep it");
        crate::memory::log_frame();
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
        }
        assert_eq!((byte(&outer), byte(&inner)), (0x11, 0x11), "frame N+1: title memory's bytes");
        // SAFETY: as above.
        assert_eq!(unsafe { ((p + 16) as *const u8).read_volatile() }, 0x11, "title memory untouched");
        unregister(p + 1, p as u64);
        unregister(p, p as u64);
    }

    /// A partial refresh of an armed page updates its saved bytes too (both
    /// under and beside a pending marked range): copy-back neither puts the
    /// refreshed byte back over a later title write nor drops a shader write
    /// back to the pre-fault value.
    #[test]
    fn partial_refresh_under_marked_range_updates_saved_bytes() {
        // No submission works through the copies meanwhile.
        let _quiet = crate::host::tests::no_submissions();
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let write = |at: usize, v: u8| {
            // SAFETY: a byte of our pool.
            unsafe { ((p + at) as *mut u8).write_volatile(v) };
        };
        copy.update(p, len);
        // Arms both pages: page 0 is wholly busy; page 1 past +16 is free.
        copy.writing(p, PAGE + 16, true);
        for (at, v) in [(20, 0x55), (24, 0x44), (40, 0x11), (PAGE + 20, 0x55)] {
            write(at, v);
        }
        assert!(copy.update(p + 16, 16) > 0 && copy.update(p + PAGE + 16, 16) > 0);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + 24) as u64, &[0x00]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        write(20, 0x66);
        write(PAGE + 20, 0x77);
        copy.copy_back_marked(p, None);
        copy.copy_back_marked(p + PAGE, None);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [read(10), read(20), read(24), read(40)],
            [0xaa, 0x66, 0x00, 0x11]
        );
        assert_eq!([read(PAGE + 10), read(PAGE + 20)], [0xbb, 0x77]);
        copy.copy_back(p, PAGE + 16);
        unregister(p, p as u64);
    }

    /// A marked page that a withheld list's range touches (issue 1976): what
    /// another list declared there goes back; the withheld list's bytes and
    /// bytes no list declared stay out of title memory, and the copy keeps
    /// them. A page only a valid list declared goes back whole.
    #[test]
    fn a_withheld_range_does_not_hold_back_a_valid_write_on_its_page() {
        // No submission takes the marks set below meanwhile.
        let _quiet = crate::host::tests::no_submissions();
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        // List A (goes back) declares [64, 68) and [PAGE + 64, PAGE + 68);
        // list B (withheld) declares [256, 260).
        let declared = [
            (copy.clone(), p + 64, 4, true),
            (copy.clone(), p + PAGE + 64, 4, true),
            (copy.clone(), p + 256, 4, false),
        ];
        for (_, a, n, _) in &declared {
            copy.writing(*a, *n, true);
        }
        copy.write_host((p + 64) as u64, &[0x5a; 4]);
        copy.write_host((p + 256) as u64, &[0xa5; 4]);
        // A write through a view no list declared, on the withheld page.
        copy.write_host((p + 1000) as u64, &[0x77]);
        copy.write_host((p + PAGE + 64) as u64, &[0x3c; 4]);
        // An undeclared write on the page only A declared goes back with it.
        copy.write_host((p + PAGE + 1000) as u64, &[0x66]);
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&declared, true);
        // SAFETY: bytes of our pool.
        let title = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [title(64), title(67)],
            [0x5a; 2],
            "A's write, sharing B's page"
        );
        assert_eq!(
            [title(256), title(259), title(1000)],
            [0; 3],
            "B's and undeclared bytes withheld"
        );
        assert_eq!(
            copy.read_host((p + 256) as u64, 1),
            vec![0xa5a5_a5a5],
            "the copy keeps B's bytes"
        );
        assert_eq!(
            copy.read_host((p + 1000) as u64, 1),
            vec![0x77],
            "and the undeclared one"
        );
        assert_eq!(
            [title(PAGE + 64), title(PAGE + 1000)],
            [0x3c, 0x66],
            "a page B did not touch"
        );
        for (_, a, n, back) in &declared {
            if *back {
                copy.copy_back(*a, *n);
            } else {
                copy.keep_writes(*a, *n);
            }
        }
        unregister(p, p as u64);
    }

    /// A marked page that went back in part (a withheld range touched it)
    /// stays changed, with no saved bytes. The next submission's refresh of
    /// a marked range takes the page whole, so its shader writes, and a
    /// title write during it, go back.
    #[test]
    fn a_page_that_went_back_in_part_takes_the_next_marked_writes_back() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let (Some(copy), Some(b)) = (find(p as u64, PAGE as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        let mark = || {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(base / 32) |= 1 << (base % 32) };
        };
        // SAFETY: bytes of our pool.
        let title = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        copy.update(p, PAGE);
        let declared = [(copy.clone(), p + 64, 4, true), (copy.clone(), p + 256, 4, false)];
        for (_, a, n, _) in &declared {
            copy.writing(*a, *n, true);
        }
        copy.write_host((p + 64) as u64, &[0x5a; 4]);
        copy.write_host((p + 256) as u64, &[0xa5; 4]);
        mark();
        copy_back_marks(&declared, true);
        copy.copy_back(p + 64, 4);
        copy.keep_writes(p + 256, 4);
        assert_eq!([title(64), title(256)], [0x5a, 0], "went back in part");
        assert!(!copy.clean(), "the page stays changed");
        // The next submission, as `crate::draw` refreshes a marked view.
        let (at, n) = copy.page_range(p as u64 + 64, 4);
        copy.update(at as usize, n as usize);
        copy.writing(p + 64, 4, true);
        copy.write_host((p + 64) as u64, &[0xc3; 4]);
        // SAFETY: a byte of our pool.
        unsafe { ((p + 2000) as *mut u8).write_volatile(0x11) };
        mark();
        copy_back_marks(&[(copy.clone(), p + 64, 4, true)], true);
        copy.copy_back(p + 64, 4);
        assert_eq!(
            [title(64), title(67), title(2000)],
            [0xc3, 0xc3, 0x11],
            "the second submission's writes and the title's"
        );
        unregister(p, p as u64);
    }

    #[test]
    fn subtract_cuts_ranges_and_joins_overlaps() {
        assert_eq!(
            subtract(vec![(10, 20), (0, 5), (15, 30)], &[(12, 14), (25, 40)]),
            [(0, 5), (10, 12), (14, 25)]
        );
        assert_eq!(subtract(vec![(0, 10)], &[(0, 10)]), []);
        assert_eq!(subtract(vec![(0, 10)], &[]), [(0, 10)]);
    }

    /// As above with the refreshes queued ([`crate::copy_pool::queued`], as
    /// `Batch::flush` does): once made, what they took goes into the saved
    /// bytes too ([`Shadow::queued_made`]).
    #[test]
    fn queued_partial_refresh_under_marked_range_updates_saved_bytes() {
        // No submission works through the copies meanwhile.
        let _quiet = crate::host::tests::no_submissions();
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let write = |at: usize, v: u8| {
            // SAFETY: a byte of our pool.
            unsafe { ((p + at) as *mut u8).write_volatile(v) };
        };
        copy.update(p, len);
        copy.writing(p, PAGE + 16, true);
        for (at, v) in [(20, 0x55), (24, 0x44), (40, 0x11), (PAGE + 20, 0x55)] {
            write(at, v);
        }
        let (copied, pieces) = crate::copy_pool::queued(|| {
            copy.update(p + 16, 16) + copy.update(p + PAGE + 16, 16)
        });
        // A copy in video memory (the default where the host has GPU upload
        // heaps, #2870) copies at once instead (#2130).
        assert!(copied > 0 && (copy.mirrored() || !pieces.is_empty()), "queued");
        crate::copy_pool::run(pieces);
        copy.queued_made();
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + 24) as u64, &[0x00]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        write(20, 0x66);
        write(PAGE + 20, 0x77);
        copy.copy_back_marked(p, None);
        copy.copy_back_marked(p + PAGE, None);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [read(10), read(20), read(24), read(40)],
            [0xaa, 0x66, 0x00, 0x11]
        );
        assert_eq!([read(PAGE + 10), read(PAGE + 20)], [0xbb, 0x77]);
        copy.copy_back(p, PAGE + 16);
        unregister(p, p as u64);
    }

    /// `partial_refresh_under_marked_range_updates_saved_bytes` with the
    /// copy in video memory and a mirror: a refresh over saved bytes writes
    /// the mirror too, and the saved bytes take it.
    #[test]
    fn partial_refresh_of_a_mirrored_copy_updates_saved_bytes() {
        rerun_with_gpu_upload(
            "shadow::tests::partial_refresh_under_marked_range_updates_saved_bytes",
            true,
        );
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }
    /// A clear (the host's write, which marks no page) of part of a buffer
    /// goes back as marked pages, the two whole pages in a row together:
    /// its bytes reach title memory, the title's byte written before the
    /// submission next to it on the same page stays, and pages it does not
    /// touch keep theirs.
    #[test]
    fn host_writes_go_back_as_marked_pages() {
        let Some(host) = host() else {
            return;
        };
        if bitmap(false).is_none() {
            return;
        }
        let (p, len, copy) = mirrored_buffer(4);
        if !copy.marked() {
            unregister(p, p as u64);
            return; // no write marks for this memory here
        }
        let (from, to) = (p + PAGE + 100, p + 3 * PAGE);
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, len as u64);
        copy.host_writes(&list, from as u64, (to - from) as u64);
        let (src, data) = host.shared_buffer((to - from) as u64).unwrap();
        // SAFETY: our mapped buffer of that many bytes.
        unsafe { std::ptr::write_bytes(data, 0x5A, to - from) };
        list.keep(&src);
        // SAFETY: an open list; both buffers live and common.
        unsafe {
            list.list.CopyBufferRegion(
                &copy.buffer,
                copy.offset(from as u64),
                &src,
                0,
                (to - from) as u64,
            )
        };
        list.close();
        // The title writes the first cleared page, before the clear's range.
        // SAFETY: our pool's bytes.
        unsafe { ((p + PAGE + 50) as *mut u8).write_volatile(0xE1) };
        host.execute(&[&list]);
        // SAFETY: our pool's bytes.
        let title = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert!(
            title[from - p..to - p].iter().all(|&b| b == 0x5A),
            "the clear's bytes"
        );
        assert_eq!(title[PAGE + 50], 0xE1, "the title's byte");
        let pattern = |i: usize| (i % 251) as u8;
        assert_eq!(title[PAGE + 99], pattern(PAGE + 99), "before the clear");
        assert!(
            (0..PAGE)
                .chain(3 * PAGE..len)
                .all(|i| title[i] == pattern(i)),
            "the pages it does not touch"
        );
        unregister(p, p as u64);
    }

    /// A heavy frame with nothing on screen, as a title makes one: many
    /// submissions, each reading and writing a big buffer of title memory
    /// that the title also writes between them. Times each part of the
    /// copy machinery per frame (synthetic; run with
    /// `--release --ignored --nocapture blank_frame`).
    #[test]
    #[ignore = "synthetic benchmark; run with --release --ignored --nocapture"]
    fn benchmark_blank_frame() {
        let Some(host) = host() else {
            return;
        };
        let setting = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let (mb, subs, dirty) = (
            setting("BENCH_MB", 256),
            setting("BENCH_SUBS", 60),
            setting("BENCH_DIRTY", 8),
        );
        let pages = mb * (1 << 20) / PAGE;
        let len = pages * PAGE;
        let p = crate::watch::tests::pool(pages);
        register(p, p as u64, len as u64);
        let copy = find(p as u64, len as u64).expect("a copy");
        copy.update(p, len);
        let lists: Vec<HostList> = (0..subs).map(|_| host.new_list().unwrap()).collect();
        // What one submission does: whether it reads the buffer, how it
        // writes it (0 not, 1 shaders that declare the range, 2 shaders that
        // mark their pages, 3 a clear of all of it, 4 the same declared as
        // shaders' writes, as clears were), and whether the title writes it
        // between submissions.
        let cases: [(&str, bool, u8, bool); 7] = [
            ("empty submissions", false, 0, false),
            ("reads, title idle", true, 0, false),
            ("reads + title writes", true, 0, true),
            ("reads + shader writes (declared)", true, 1, true),
            ("reads + shader writes (marked)", true, 2, true),
            ("reads + clears (declared, as before)", true, 4, true),
            ("reads + clears (marked)", true, 3, true),
        ];
        // What the clears write: all zeros or all ones, by turns.
        let fills: Vec<ID3D12Resource> = [0u8, 0xFF]
            .iter()
            .map(|&v| {
                let (b, data) = host.shared_buffer(len as u64).unwrap();
                // SAFETY: our mapped buffer of `len` bytes.
                unsafe { std::ptr::write_bytes(data, v, len) };
                b
            })
            .collect();
        eprintln!(
            "{mb} MiB buffer, {subs} submissions a frame, {dirty} pages written by the title between them"
        );
        for (name, read, write, title_writes) in cases {
            if write >= 2 && bitmap(false).is_none() {
                eprintln!("{name}: no write-mark bitmap here");
                continue;
            }
            let mut best = std::time::Duration::MAX;
            for _ in 0..3 {
                let started = std::time::Instant::now();
                for (k, list) in lists.iter().enumerate() {
                    if read {
                        copy.refresh(list, p as u64, len as u64);
                    }
                    match write {
                        1 | 2 => copy.gpu_writes(list, p as u64, len as u64, write == 2),
                        3 => copy.host_writes(list, p as u64, len as u64),
                        4 => copy.gpu_writes(list, p as u64, len as u64, false),
                        _ => {}
                    }
                    if write >= 3 {
                        let fill = &fills[k % 2];
                        list.keep(fill);
                        // SAFETY: an open list; both buffers live and common.
                        unsafe {
                            list.list.CopyBufferRegion(
                                &copy.buffer,
                                copy.offset(p as u64),
                                fill,
                                0,
                                len as u64,
                            )
                        };
                    } else if write != 0 {
                        gpu_store(list, &copy, p + (k % pages) * PAGE + 64, &[k as u8; 16]);
                    }
                    list.close();
                    if title_writes {
                        for d in 0..dirty {
                            let page = (k * 7919 + d * 104_729) % pages;
                            // SAFETY: our pool's bytes.
                            unsafe { ((p + page * PAGE + 8) as *mut u8).write_volatile(k as u8) };
                        }
                    }
                    host.execute(&[list]);
                    list.reset();
                }
                best = best.min(started.elapsed());
            }
            eprintln!(
                "{name}: {:.1} ms a frame ({:.2} ms a submission)",
                best.as_secs_f64() * 1e3,
                best.as_secs_f64() * 1e3 / subs as f64
            );
        }
        unregister(p, p as u64);
    }

    const MIB: u64 = 1 << 20;

    fn entry(start: u64, len: u64) -> Entry {
        Entry {
            end: start + len,
            shadow: None,
            read_only: false,
        }
    }

    /// A 256 MiB parent with `children` 64 KiB buffers placed in its first
    /// part, as a heap's placed buffers.
    fn heap(base: u64, children: u64) -> BTreeMap<(u64, usize), Entry> {
        let mut map = BTreeMap::new();
        map.insert((base, 1), entry(base, 256 * MIB));
        for i in 0..children {
            map.insert((base + i * 0x10000, 100 + i as usize), entry(base + i * 0x10000, 0x10000));
        }
        map
    }

    #[test]
    fn a_parent_is_found_past_many_children() {
        let base = 0x10_0000_0000;
        let map = heap(base, 200);
        let span = 256 * MIB;
        // Past all 200 children (and the 64 entries looked at before).
        let at = base + 200 * 0x10000 + 0x100;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        let first_64 = map.range(..=(at, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= at + 16);
        assert!(first_64.is_none(), "the parent is more than 64 entries back");
        // Near the parent's end.
        let at = base + 256 * MIB - 16;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        // Inside a child: the child, the innermost.
        let at = base + 150 * 0x10000 + 8;
        assert_eq!(enclosing(&map, span, at, at + 8), Some((base + 150 * 0x10000, 250)));
        // Across two children: only the parent holds it.
        let at = base + 151 * 0x10000 - 8;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        // Past the parent, and before it.
        let at = base + 256 * MIB;
        assert_eq!(enclosing(&map, span, at, at + 1), None);
        assert_eq!(enclosing(&map, span, base - 16, base + 16), None);
        // Longer than any buffer: none, without looking.
        assert_eq!(enclosing(&map, span, base, base + span + 1), None);
        assert_eq!(enclosing(&map, 0, base, base + 1), None);
    }

    #[test]
    fn registered_buffers_widen_the_search() {
        // High above anything a test maps; resource numbers no live
        // resource has.
        let base = 0x7f00_0000_0000;
        let len = 512 * MIB;
        register(1, base, len);
        for i in 0..100u64 {
            register(2 + i as usize, base + i * 0x1000, 0x1000);
        }
        assert!(MAX_SPAN.load(Ordering::Relaxed) >= len);
        let at = base + len - 64;
        let found = enclosing(&buffers(), MAX_SPAN.load(Ordering::Relaxed), at, at + 64);
        for i in 0..100u64 {
            unregister(2 + i as usize, base + i * 0x1000);
        }
        unregister(1, base);
        assert_eq!(found, Some((base, 1)));
        assert!(buffers().range((base, 0)..=(base + len, usize::MAX)).next().is_none());
    }

    #[test]
    fn loose_windows_are_reused_for_ranges_inside_them() {
        let w = WINDOW;
        let base = 0x20_0000_0000;
        let mut windows = BTreeMap::new();
        windows.insert((base, w), "first");
        // A range across the next two windows got one of its own.
        windows.insert((base + w, 2 * w), "spanning");
        let span = 2 * w;
        let find = |address: u64, len: u64| enclosing_window(&windows, span, address, address + len).copied();
        assert_eq!(find(base, 1), Some("first"));
        assert_eq!(find(base + w - 16, 16), Some("first"));
        // Inside the spanning window, also in its second half.
        assert_eq!(find(base + w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 2 * w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 3 * w - 8, 8), Some("spanning"));
        // Across the first window's end: none holds it whole.
        assert_eq!(find(base + w - 8, 16), None);
        // Past every window, before them, longer than any.
        assert_eq!(find(base + 3 * w, 1), None);
        assert_eq!(find(base - 8, 4), None);
        assert_eq!(find(base, 3 * w), None);
    }

    #[test]
    fn loose_memory_must_be_committed() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // The null page and its guard area.
        assert!(loose(0, 16).is_none());
        assert!(loose(0x1000, 0x1010).is_none());
        assert!(loose(LOWEST_MAPPED - 1, LOWEST_MAPPED + 1).is_none());
        // Reserved, not committed.
        // SAFETY: a fresh reservation, released below.
        let p = unsafe { VirtualAlloc(std::ptr::null(), WINDOW as usize, MEM_RESERVE, PAGE_READWRITE) } as u64;
        assert_ne!(p, 0);
        assert!(loose(p + 64, p + 128).is_none());
        // SAFETY: our reservation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        // Too close to the top to round up to a window.
        assert!(loose(u64::MAX - 8, u64::MAX).is_none());
    }

    /// The committed part around a range, of one allocation, cut to 64 KiB.
    #[test]
    fn the_committed_part_of_a_window_holds_the_range_or_is_none() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const K: u64 = 64 << 10;
        // SAFETY: our own reservation; the middle 4 x 64 KiB committed.
        let base = unsafe { VirtualAlloc(std::ptr::null(), (8 * K) as usize, MEM_RESERVE, PAGE_READWRITE) } as u64;
        assert_ne!(base, 0);
        // SAFETY: committing pages of our own reservation.
        unsafe { VirtualAlloc((base + 2 * K) as *const _, (4 * K) as usize, MEM_COMMIT, PAGE_READWRITE) };
        // A range in the committed middle: the committed part of the window.
        assert_eq!(committed_part(base, 8 * K, base + 3 * K + 8, base + 3 * K + 16), Some((base + 2 * K, 4 * K)));
        // Cut to the window.
        assert_eq!(committed_part(base + 3 * K, K, base + 3 * K + 8, base + 3 * K + 16), Some((base + 3 * K, K)));
        // A range reaching into reserved memory: none.
        assert_eq!(committed_part(base, 8 * K, base + 6 * K - 8, base + 6 * K + 8), None);
        assert_eq!(committed_part(base, 8 * K, base + 8, base + 16), None);
        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    /// Memory of one committed allocation becomes a host buffer in place:
    /// what the CPU wrote there is what the buffer holds.
    #[test]
    fn committed_memory_opens_as_a_buffer_in_place() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        if host().is_none() {
            return;
        }
        const LEN: u64 = 1 << 20;
        // SAFETY: our own allocation.
        let base = unsafe { VirtualAlloc(std::ptr::null(), LEN as usize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as u64;
        assert_ne!(base, 0);
        touch(base as usize, LEN as usize);
        // SAFETY: our memory.
        unsafe { *((base + (64 << 10)) as *mut u32) = 0x1234_5678 };
        let Some((buffer, gpu, _)) = in_place_buffer(base + (64 << 10), 128 << 10) else {
            // A host that cannot open process memory as a heap: copies stay.
            // SAFETY: our allocation.
            unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
            return;
        };
        assert_ne!(gpu, 0);
        // SAFETY: a plain query on a live buffer.
        assert_eq!(unsafe { buffer.GetDesc() }.Width, 128 << 10);
        // Off 64 KiB: refused.
        assert!(in_place_buffer(base + 4096, 64 << 10).is_none());
        drop(buffer);
        let_go_in_place(base, LEN);
        // SAFETY: our allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }


    /// A heap opened over an allocation that is released and made again
    /// with another size is not handed out again.
    #[test]
    fn a_heap_over_a_released_allocation_is_let_go() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let _whole = WholeAllocations::on();
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        if host().is_none() {
            return;
        }
        // SAFETY: our own allocation.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 1 << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        assert_eq!(allocation_extent(base), Some(1 << 20));
        touch(base, 1 << 20);
        if in_place_buffer(base as u64, 64 << 10).is_none() {
            // SAFETY: our allocation.
            unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
            return; // no heaps from process memory on this host
        }
        // SAFETY: our allocation.
        let freed = unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) } != 0;
        if !freed || allocation_extent(base).is_some() {
            // Windows keeps memory a heap was opened over until the heap
            // goes (Wine releases it at once): let go of it, then release.
            let_go_in_place(base as u64, 1 << 20);
            // SAFETY: our allocation.
            assert_ne!(unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) }, 0, "released once the heap is let go");
        }
        assert_eq!(allocation_extent(base), None);
        // SAFETY: a new allocation, at that address if it is free.
        let again = unsafe { VirtualAlloc(base as *const _, 2 << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        if again == base {
            touch(again, 2 << 20);
            let (buffer, _, _) = in_place_buffer(base as u64, 2 << 20).expect("a heap over the new allocation");
            // SAFETY: a plain query on a live buffer.
            assert_eq!(unsafe { buffer.GetDesc() }.Width, 2 << 20, "placed in the new, larger heap");
            drop(buffer);
            // SAFETY: our allocation.
            unsafe { VirtualFree(again as *mut _, 0, MEM_RELEASE) };
        }
        let_go_in_place(base as u64, 2 << 20);
    }


    /// A stale heap found while a caller holds `LOOSE` and `BUFFERS` (as
    /// `loose` and `find` do) does not take those locks again.
    #[test]
    fn a_stale_heap_is_replaced_under_the_callers_locks() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let _whole = WholeAllocations::on();
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        if host().is_none() {
            return;
        }
        // SAFETY: our own allocation.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 1 << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        touch(base, 1 << 20);
        if in_place_buffer(base as u64, 64 << 10).is_none() {
            // SAFETY: our allocation.
            unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
            return; // no heaps from process memory on this host
        }
        // The cached heap no longer matches its allocation (as after the
        // title released it and allocated again with another size). Made so
        // in the cache itself: Windows does not change memory a heap was
        // opened over while the heap lives.
        let mut stale = 0;
        if let Some(Some((_, size, id))) = IN_PLACE_HEAPS.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|h| h.get_mut(&base)) {
            *size += 64 << 10;
            stale = *id;
        }
        {
            let _loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
            let _buffers = buffers();
            // Would wait on the locks above forever if it let go of what is
            // placed in the heap here.
            let _ = in_place_buffer(base as u64, 64 << 10);
        }
        assert!(STALE_IN_PLACE.lock().unwrap_or_else(|e| e.into_inner()).contains(&stale));
        check_in_place();
        assert!(STALE_IN_PLACE.lock().unwrap_or_else(|e| e.into_inner()).is_empty(), "let go at the frame's check");
        // The heap opened in place of the stale one stays cached (only the
        // stale one is let go): let go of it before the memory, or a later
        // test given this address finds it.
        let_go_in_place(base as u64, 1 << 20);
        // SAFETY: our allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }


    /// A stale heap replaced by a new one at the same base is let go at the
    /// frame's check with what was placed in it, and nothing of the new
    /// heap: not the heap, nor the windows placed in it.
    #[test]
    fn stale_queue_does_not_evict_replacement_heap_at_same_base() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let _whole = WholeAllocations::on();
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        if host().is_none() {
            return;
        }
        // SAFETY: our own allocation.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 1 << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        touch(base, 1 << 20);
        let place = |len: u64| {
            let (buffer, gpu, heap) = in_place_buffer(base as u64, len)?;
            let shadow = Shadow::new_with(buffer, base as *mut u8, gpu, base as u64, len, Some(heap));
            LOOSE.lock().unwrap_or_else(|e| e.into_inner()).insert((base as u64, len), shadow);
            Some(heap)
        };
        let Some(old) = place(64 << 10) else {
            // SAFETY: our allocation.
            unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
            return; // no heaps from process memory on this host
        };
        // The cached heap no longer matches its allocation (as after the
        // title released it and made another of a new size there).
        if let Some(Some((_, size, _))) = IN_PLACE_HEAPS.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|h| h.get_mut(&base)) {
            *size += 64 << 10;
        }
        let new = place(128 << 10).expect("a new heap at the same base");
        assert_ne!(new, old);
        assert!(STALE_IN_PLACE.lock().unwrap_or_else(|e| e.into_inner()).contains(&old));
        check_in_place();
        let (old_gone, survived) = {
            let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
            (
                !loose.contains_key(&(base as u64, 64 << 10)),
                loose.get(&(base as u64, 128 << 10)).is_some_and(|s| s.heap == Some(new)),
            )
        };
        let cached = IN_PLACE_HEAPS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|h| h.get(&base).cloned())
            .flatten()
            .map(|(_, _, id)| id);
        let_go_in_place(base as u64, 1 << 20);
        // SAFETY: our allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
        assert!(old_gone, "the window in the stale heap is let go");
        assert!(survived, "the window in the new heap stays");
        assert_eq!(cached, Some(new), "the new heap stays cached");
    }


    /// A list that uses an in-place buffer keeps it (and so its heap) until
    /// it is reset, though nothing is refreshed: the title may release the
    /// buffer, or its heap be let go, before the list runs.
    #[test]
    fn a_list_keeps_the_in_place_buffers_it_uses() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        use windows::core::{IUnknown, Interface};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        let Some(host) = host() else { return };
        // SAFETY: our own allocation.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 1 << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        touch(base, 1 << 20);
        let Some((buffer, gpu, heap)) = in_place_buffer(base as u64, 64 << 10) else {
            // SAFETY: our allocation.
            unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
            return; // no heaps from process memory on this host
        };
        let seen = buffer.clone();
        // References to the buffer (besides the one asking).
        let refs = || {
            let u: IUnknown = seen.cast().unwrap();
            let vtable = u.vtable();
            // SAFETY: a live object; one reference added and taken back.
            unsafe {
                (vtable.AddRef)(u.as_raw());
                (vtable.Release)(u.as_raw()) - 1
            }
        };
        let shadow = Shadow::new_with(buffer, base as *mut u8, gpu, base as u64, 64 << 10, Some(heap));
        let list = host.new_list().unwrap();
        let before = refs();
        shadow.refresh(&list, base as u64, 256);
        shadow.refresh(&list, base as u64 + 256, 256);
        shadow.gpu_writes(&list, base as u64, 4096, false);
        assert_eq!(refs(), before + 1, "kept once by the list");
        drop(shadow);
        assert_eq!(refs(), before, "the list holds it after the copy is gone");
        list.close();
        list.reset();
        assert_eq!(refs(), before - 1, "let go when the list is reset");
        drop(seen);
        let_go_in_place(base as u64, 1 << 20);
        // SAFETY: our allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }


    /// What shaders wrote through a buffer in place went into title memory
    /// unseen by the guards: other copies holding the page see it changed,
    /// as after a copy-back.
    #[test]
    fn writes_in_place_mark_other_copies_of_the_page_changed() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        use crate::watch::tests::{free, pool};
        if host().is_none() {
            return;
        }
        let p = pool(2);
        touch(p, 2 * PAGE);
        let Some((buffer, gpu, heap)) = in_place_buffer(p as u64, PAGE as u64) else {
            free(p);
            return; // no heaps from process memory on this host
        };
        let in_place = Shadow::new_with(buffer, p as *mut u8, gpu, p as u64, PAGE as u64, Some(heap));
        let other = crate::watch::track(p, 2 * PAGE, true).expect("pool memory is tracked");
        other.take(p, p + 2 * PAGE);
        // Windows does not guard memory a heap was opened over: such a page
        // stays changed, so copies of it are always copied again.
        let guarded = !other.changed(p / PAGE);
        in_place.copy_back(p + 8, 4);
        let marked = other.changed(p / PAGE);
        let beside = other.changed(p / PAGE + 1);
        drop((other, in_place));
        let_go_in_place(p as u64, 2 * PAGE as u64);
        free(p);
        if guarded {
            assert!(marked, "the page written in place is changed for other copies");
            assert!(!beside, "and only that page");
        }
    }


    /// The in-place tests share the heap cache and the queue of stale heaps
    /// (and may get each other's addresses back): one at a time.
    static IN_PLACE_TESTS: Mutex<()> = Mutex::new(());

    /// Writes every page of `[base, base + len)`, so the allocation counts as
    /// dense ([`dense`]).
    fn touch(base: usize, len: usize) {
        for page in (0..len).step_by(4096) {
            // SAFETY: committed memory of the test's own allocation.
            unsafe { ((base + page) as *mut u8).write_volatile(1) };
        }
    }

    /// Releases the test's allocation at `base` once the heaps over it are
    /// gone. Windows may let go of the pages a little after the last heap
    /// (WARP on CI after the buffer holding it went, NVIDIA up to 6 ms), as
    /// `crate::pinned::title_memory_change` allows for: retried for up to 1 s.
    fn free_once_let_go(base: usize) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        let start = std::time::Instant::now();
        // SAFETY: the test's own allocation, released once.
        while unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) } == 0 {
            let error = std::io::Error::last_os_error();
            assert!(
                start.elapsed() < std::time::Duration::from_secs(1),
                "still held after 1 s: {error}"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Takes the whole-allocation path for the life of the guard.
    struct WholeAllocations;
    impl WholeAllocations {
        fn on() -> Self {
            // Decisions of earlier tests may name an address given out again.
            *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = None;
            WHOLE_ALLOCATIONS_ONLY.store(true, Ordering::Relaxed);
            WholeAllocations
        }
    }
    impl Drop for WholeAllocations {
        fn drop(&mut self) {
            WHOLE_ALLOCATIONS_ONLY.store(false, Ordering::Relaxed);
        }
    }

    /// A sparse allocation (committed, mostly never touched) is not opened
    /// whole: that would make all of it resident.
    #[test]
    fn a_sparse_allocation_is_not_opened_whole() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let _whole = WholeAllocations::on();
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        if host().is_none() {
            return;
        }
        const LEN: usize = 16 << 20;
        // SAFETY: our own allocation.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        touch(base, 64 << 10);
        assert!(
            in_place_buffer(base as u64, 64 << 10).is_none(),
            "1 of 256 pieces touched"
        );
        assert!(
            IN_PLACE_HEAPS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_none_or(|h| !h.contains_key(&base))
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    /// Unset, `VICTORY_IN_PLACE` is on exactly where the host GPU shares
    /// memory with the CPU and title memory passed the in-place self-test.
    #[test]
    fn title_memory_is_in_place_by_default_on_a_unified_host() {
        let Some(host) = host() else { return };
        if std::env::var_os("VICTORY_IN_PLACE").is_some() {
            return;
        }
        let unified = host.memory.uma && crate::unified::self_test(&host.device).is_ok();
        assert_eq!(in_place_enabled(), unified, "{}", host.memory);
    }

    /// The title freeing memory a window is in place over lets go of the
    /// window first (title memory a heap holds refuses to be freed), and a
    /// heap that goes by itself takes its registration along.
    #[test]
    fn the_title_frees_memory_a_window_is_in_place_over() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        if host().is_none() {
            return;
        }
        const LEN: u64 = 1 << 20;
        let alloc = || {
            // SAFETY: our own allocation.
            let at = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    LEN as usize,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            } as u64;
            assert_ne!(at, 0);
            touch(at as usize, LEN as usize);
            at
        };
        let base = alloc();
        let Some((buffer, gpu, heap)) = in_place_buffer(base, LEN) else {
            // SAFETY: our allocation.
            unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
            return; // the host refuses process memory as a heap
        };
        assert_eq!(crate::pinned::held_over(base as usize, LEN as usize), 1);
        let window = Shadow::new_with(buffer, base as *mut u8, gpu, base, LEN, Some(heap));
        LOOSE.lock().unwrap_or_else(|e| e.into_inner()).insert((base, LEN), window);
        // The title frees it: the window goes first, with its heap.
        // SAFETY: our allocation.
        let free = || unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
        assert_ne!(crate::pinned::title_memory_change(base as usize, LEN as usize, free), 0);
        assert!(!LOOSE.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&(base, LEN)));
        assert_eq!(crate::pinned::held_over(base as usize, LEN as usize), 0);
        // A heap that goes by itself takes its registration along.
        let base = alloc();
        let (buffer, _, _) = in_place_buffer(base, 64 << 10).expect("opened above");
        assert_eq!(crate::pinned::held_over(base as usize, LEN as usize), 1);
        drop(buffer);
        assert_eq!(crate::pinned::held_over(base as usize, LEN as usize), 0);
        // SAFETY: our allocation, no heap on it.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    /// A buffer placed over title memory keeps its heap, and so the heap's
    /// pin, for as long as the buffer lives, also once nothing else holds the
    /// heap (vkd3d-proton destroyed it when `in_place_buffer` returned), and
    /// the pin goes with the buffer. On both paths: a sub-range heap per
    /// buffer, and a whole-allocation heap the cache has let go of.
    #[test]
    fn a_buffer_in_place_holds_its_heap_and_pin() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        if host().is_none() {
            return;
        }
        const LEN: usize = 1 << 20;
        for whole in [false, true] {
            let _whole = whole.then(WholeAllocations::on);
            // SAFETY: our own allocation.
            let base = unsafe {
                VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
            } as usize;
            assert_ne!(base, 0);
            touch(base, LEN);
            let Some((buffer, _, id)) = in_place_buffer(base as u64, LEN as u64) else {
                // SAFETY: our allocation.
                unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
                return; // the host refuses process memory as a heap
            };
            // Nothing but the buffer holds the heap now.
            let_go_heap(id);
            for _ in 0..3 {
                assert_eq!(crate::pinned::held_over(base, LEN), 1, "whole allocation: {whole}");
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            drop(buffer);
            assert_eq!(crate::pinned::held_over(base, LEN), 0, "whole allocation: {whole}");
            free_once_let_go(base);
        }
    }

    /// Where the host opens sub-ranges, a buffer in place is a heap over its
    /// own pages only, also in a sparse allocation, and no whole-allocation
    /// heap is kept.
    #[test]
    fn a_sub_range_heap_covers_only_the_buffer() {
        let _one = IN_PLACE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D12::ID3D12Device13;
        use windows::core::Interface;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        let Some(host) = host() else { return };
        if host.device.cast::<ID3D12Device13>().is_err() {
            return; // no sub-range heaps on this host (vkd3d-proton)
        }
        const LEN: usize = 16 << 20;
        // SAFETY: our own allocation.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        // What the cache holds at this base (left by a test that had the
        // address before, if anything): the sub-range path adds nothing.
        let cached = || {
            IN_PLACE_HEAPS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .and_then(|h| h.get(&base).map(|v| v.as_ref().map(|(_, _, id)| *id)))
        };
        let before = cached();
        let Some((buffer, gpu, _)) = in_place_buffer((base + (1 << 20)) as u64, 128 << 10) else {
            // SAFETY: our allocation.
            unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
            return; // the host refuses process memory as a heap
        };
        assert_ne!(gpu, 0);
        // SAFETY: a plain query on a live buffer.
        assert_eq!(unsafe { buffer.GetDesc() }.Width, 128 << 10);
        assert_eq!(cached(), before, "no whole-allocation heap opened");
        drop(buffer);
        // The heap went with the buffer.
        free_once_let_go(base);
    }
}
