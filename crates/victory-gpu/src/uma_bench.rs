//! Where title memory could live for the GPU, on this host (synthetic).
//!
//! Host copies of title memory are in system memory (`Host::shared_buffer`:
//! a custom heap, write-back, L0) and are refreshed from title memory and
//! copied back each submission (`crate::shadow`, `crate::watch`). This
//! measures the alternatives a translated program could read and write
//! instead, with the access patterns translated programs make:
//!
//! - system memory, committed (today's copies);
//! - title memory itself, opened as a heap (`OpenExistingHeapFromAddress`):
//!   a write-watched private allocation (the title's pools) and a view of a
//!   pagefile section in a placeholder (its physical pages);
//! - an upload heap (write-combined system memory, read only);
//! - a GPU upload heap (video memory the CPU maps, through Resizable BAR);
//! - video memory (a default heap).
//!
//! `uma_probe` reports what each adapter says about its memory (UMA,
//! cache-coherent UMA, GPU upload heaps, budgets) and checks, per kind of
//! title memory, whether it opens as a heap, whether the GPU sees CPU writes
//! and the CPU the GPU's, whether write watching and read-only guards see
//! GPU writes, and what the GPU reads after the title remaps a view.
//! `uma_bench` times one dispatch per placement and pattern with GPU
//! timestamps on a queue of its own, and CPU copies in and out.
//!
//! `cargo test -p victory-gpu --release --lib uma_ -- --ignored --nocapture
//! --test-threads 1`

use std::ffi::c_void;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCSTR, s};
use windows_sys::Win32::System::Memory as mem;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer c : register(b0) { uint pattern; uint count; uint mask; uint add; }
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer rw : register(u0);
RWByteAddressBuffer sink : register(u1);

uint hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return x;
}

[RootSignature(RS)]
[numthreads(256, 1, 1)]
void bench(uint3 id : SV_DispatchThreadID) {
    uint t = id.x;
    uint s = 0;
    for (uint i = 0; i < count; i++) {
        uint lin = i * THREADS + t;
        uint h = hash(t * 0x9E3779B9u + i);
        if (pattern == 0) {
            s += src.Load((lin * 4) & mask);
        } else if (pattern == 1) {
            uint4 v = src.Load4((lin * 16) & mask);
            s += v.x + v.y + v.z + v.w;
        } else if (pattern == 2) {
            s += src.Load((h * 4) & mask);
        } else if (pattern == 3) {
            uint a = (h * 64) & mask;
            uint4 v = src.Load4(a) + src.Load4(a + 16) + src.Load4(a + 32) + src.Load4(a + 48);
            s += v.x + v.y + v.z + v.w;
        } else if (pattern == 4) {
            s += src.Load((h * 4) & 0xFFFFF);
        } else if (pattern == 5) {
            rw.Store((lin * 4) & mask, lin);
        } else if (pattern == 6) {
            rw.Store((h * 4) & mask, h);
        } else if (pattern == 7) {
            rw.InterlockedAdd((h * 4) & mask, 1);
        } else if (pattern == 8) {
            uint a = (lin * 4) & mask;
            rw.Store(a, rw.Load(a) + 1);
        } else {
            s += rw.Load((h * 4) & mask);
        }
    }
    if (s == 0x9E3779B9u) {
        sink.Store(0, s);
    }
}

// Copies `count` dwords of rw to sink and adds `add` to each in rw.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void probe(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) {
        uint v = rw.Load(id.x * 4);
        sink.Store(id.x * 4, v);
        rw.Store(id.x * 4, v + add);
    }
}
"#;

/// Threads a bench dispatch (groups of 256).
const THREADS: u32 = 256 * 1024;
/// Bytes of each placement's buffer.
const SIZE: u64 = 256 << 20;
const PAGE: usize = 1 << 16;

/// Patterns: name, accesses per thread, bytes per access, whether it
/// writes (needs a UAV of the placement).
const PATTERNS: [(&str, u32, u32, bool); 10] = [
    ("linear read 4 B", 256, 4, false),
    ("linear read 16 B", 64, 16, false),
    ("scattered read 4 B", 64, 4, false),
    ("scattered read 64 B line", 16, 64, false),
    ("hot 1 MiB read 4 B", 64, 4, false),
    ("linear write 4 B", 256, 4, true),
    ("scattered write 4 B", 64, 4, true),
    ("scattered atomic add", 64, 4, true),
    ("linear read+write 4 B", 128, 8, true),
    ("scattered read 4 B (UAV)", 64, 4, true),
];

fn compile(entry: PCSTR) -> Vec<u8> {
    let source = SOURCE.replace("THREADS", &format!("{THREADS}u"));
    let (mut code, mut errs): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the source outlives the call.
    let r = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("uma"),
            None,
            None,
            entry,
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errs),
        )
    };
    if r.is_err() {
        // SAFETY: the error blob's bytes.
        let msg = errs.map(|e| unsafe {
            String::from_utf8_lossy(std::slice::from_raw_parts(
                e.GetBufferPointer().cast::<u8>(),
                e.GetBufferSize(),
            ))
            .into_owned()
        });
        panic!("compile: {msg:?}");
    }
    let code = code.unwrap();
    // SAFETY: the blob's bytes.
    unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
            .to_vec()
    }
}

fn desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn committed(
    device: &ID3D12Device,
    heap: D3D12_HEAP_PROPERTIES,
    size: u64,
    uav: bool,
) -> windows::core::Result<ID3D12Resource> {
    let flags = if uav {
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    } else {
        D3D12_RESOURCE_FLAG_NONE
    };
    let state = match heap.Type {
        D3D12_HEAP_TYPE_UPLOAD => D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_HEAP_TYPE_READBACK => D3D12_RESOURCE_STATE_COPY_DEST,
        _ => D3D12_RESOURCE_STATE_COMMON,
    };
    let mut out: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc(size, flags),
            state,
            None,
            &mut out,
        )?;
    }
    Ok(out.unwrap())
}

fn heap(kind: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: kind,
        ..Default::default()
    }
}

/// `Host::shared_buffer`'s memory.
fn system() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    }
}

fn map(r: &ID3D12Resource) -> *mut u8 {
    let mut p = std::ptr::null_mut();
    // SAFETY: a CPU-visible resource.
    unsafe { r.Map(0, None, Some(&mut p)).unwrap() };
    p.cast()
}

/// A buffer of `size` bytes over all of an opened heap.
fn placed(
    device: &ID3D12Device,
    heap: &ID3D12Heap,
    size: u64,
) -> windows::core::Result<ID3D12Resource> {
    let flags =
        D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    let mut out: Option<ID3D12Resource> = None;
    // SAFETY: plain creation in a live heap.
    unsafe {
        device.CreatePlacedResource(
            heap,
            0,
            &desc(size, flags),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut out,
        )?;
    }
    Ok(out.unwrap())
}

fn open(device: &ID3D12Device, at: *const u8) -> windows::core::Result<ID3D12Heap> {
    let d3: ID3D12Device3 = device.cast()?;
    // SAFETY: memory of ours that outlives the heap.
    unsafe { d3.OpenExistingHeapFromAddress(at.cast()) }
}

fn open1(device: &ID3D12Device, at: *const u8, len: usize) -> windows::core::Result<ID3D12Heap> {
    let d13: ID3D12Device13 = device.cast()?;
    // SAFETY: memory of ours that outlives the heap.
    unsafe { d13.OpenExistingHeapFromAddress1(at.cast(), len) }
}

/// Title memory as the runtime allocates it (`victory_hle` xmem).
enum Title {
    /// A private allocation (the title's pools: write-watched).
    Private { at: *mut u8, len: usize },
    /// A view of a pagefile section in a placeholder (physical pages).
    View {
        section: windows_sys::Win32::Foundation::HANDLE,
        at: *mut u8,
        len: usize,
    },
}

impl Title {
    fn private(len: usize, watch: bool) -> Title {
        let kind = mem::MEM_RESERVE | mem::MEM_COMMIT | if watch { 0x20_0000 } else { 0 };
        // SAFETY: plain allocation.
        let at = unsafe {
            mem::VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                len,
                kind,
                mem::PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert!(!at.is_null());
        Title::Private { at: at.cast(), len }
    }

    fn section(len: usize, commit: bool) -> windows_sys::Win32::Foundation::HANDLE {
        let kind = if commit {
            mem::SEC_COMMIT
        } else {
            mem::SEC_RESERVE
        };
        // SAFETY: plain section creation.
        let h = unsafe {
            mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | kind,
                (len as u64 >> 32) as u32,
                len as u32,
                std::ptr::null(),
            )
        };
        assert!(!h.is_null());
        h
    }

    /// A view of a new section (reserved and committed as xmem does, or
    /// committed whole) in a placeholder of its size.
    fn view(len: usize, commit: bool) -> Title {
        let section = Title::section(len, commit);
        // SAFETY: plain placeholder reservation and view mapping.
        let at = unsafe {
            let base = mem::VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                len,
                mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
                mem::PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            );
            assert!(!base.is_null());
            let view = mem::MapViewOfFile3(
                section,
                GetCurrentProcess(),
                base,
                0,
                len,
                mem::MEM_REPLACE_PLACEHOLDER,
                mem::PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value, base);
            if !commit {
                let c = mem::VirtualAlloc(base, len, mem::MEM_COMMIT, mem::PAGE_READWRITE);
                assert!(!c.is_null());
            }
            base
        };
        Title::View {
            section,
            at: at.cast(),
            len,
        }
    }

    fn at(&self) -> *mut u8 {
        match *self {
            Title::Private { at, .. } | Title::View { at, .. } => at,
        }
    }

    fn len(&self) -> usize {
        match *self {
            Title::Private { len, .. } | Title::View { len, .. } => len,
        }
    }

    // The bench's memory, written by one processor at a time.
    #[allow(clippy::mut_from_ref)]
    fn dwords(&self) -> &mut [u32] {
        // SAFETY: our committed read-write memory.
        unsafe { std::slice::from_raw_parts_mut(self.at().cast(), self.len() / 4) }
    }
}

impl Drop for Title {
    fn drop(&mut self) {
        // SAFETY: what `private`/`view` made.
        unsafe {
            match *self {
                Title::Private { at, .. } => {
                    mem::VirtualFree(at.cast(), 0, mem::MEM_RELEASE);
                }
                Title::View { section, at, .. } => {
                    let a = mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: at.cast() };
                    mem::UnmapViewOfFile2(GetCurrentProcess(), a, 0);
                    mem::VirtualFree(at.cast(), 0, mem::MEM_RELEASE);
                    windows_sys::Win32::Foundation::CloseHandle(section);
                }
            }
        }
    }
}

/// A queue, a list and a fence of the bench's own, and the pipelines.
struct Gpu {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    done: u64,
    stamps: ID3D12QueryHeap,
    readback: ID3D12Resource,
    stamp_back: *mut u64,
    frequency: u64,
    root: ID3D12RootSignature,
    bench: ID3D12PipelineState,
    probe: ID3D12PipelineState,
}

impl Gpu {
    fn new(device: ID3D12Device) -> Gpu {
        let (bench, probe) = (compile(s!("bench")), compile(s!("probe")));
        // SAFETY: plain creation.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .unwrap();
            list.Close().unwrap();
            let fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            let mut stamps: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 2,
                        NodeMask: 0,
                    },
                    &mut stamps,
                )
                .unwrap();
            let readback = committed(&device, heap(D3D12_HEAP_TYPE_READBACK), 16, false).unwrap();
            let stamp_back = map(&readback).cast();
            let frequency = queue.GetTimestampFrequency().unwrap();
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &bench).unwrap();
            let pso = |code: &[u8]| -> ID3D12PipelineState {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                };
                device.CreateComputePipelineState(&desc).unwrap()
            };
            let (bench, probe) = (pso(&bench), pso(&probe));
            Gpu {
                device,
                queue,
                allocator,
                list,
                fence,
                event,
                done: 0,
                stamps: stamps.unwrap(),
                readback,
                stamp_back,
                frequency,
                root,
                bench,
                probe,
            }
        }
    }

    /// Copies `pieces` runs of `piece` bytes (every other one, as pages a
    /// title wrote) from `src` to `dst` on the GPU; GPU time in ms.
    fn copy(&mut self, dst: &ID3D12Resource, src: &ID3D12Resource, piece: u64, pieces: u64) -> f64 {
        // SAFETY: the list is idle; live buffers in the common state.
        unsafe {
            self.allocator.Reset().unwrap();
            self.list.Reset(&self.allocator, None).unwrap();
            let l = &self.list;
            l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            for k in 0..pieces {
                let at = 2 * k * piece;
                l.CopyBufferRegion(dst, at, src, at, piece);
            }
            l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            l.ResolveQueryData(
                &self.stamps,
                D3D12_QUERY_TYPE_TIMESTAMP,
                0,
                2,
                &self.readback,
                0,
            );
            l.Close().unwrap();
            self.queue
                .ExecuteCommandLists(&[Some(l.cast::<ID3D12CommandList>().unwrap())]);
            self.done += 1;
            self.queue.Signal(&self.fence, self.done).unwrap();
            self.fence
                .SetEventOnCompletion(self.done, self.event)
                .unwrap();
            WaitForSingleObject(self.event, INFINITE);
            let t = std::slice::from_raw_parts(self.stamp_back, 2);
            t[1].saturating_sub(t[0]) as f64 * 1e3 / self.frequency as f64
        }
    }

    /// Records one dispatch, runs it, waits; its GPU time in ms.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &mut self,
        pso: bool,
        constants: [u32; 4],
        src: u64,
        rw: u64,
        sink: u64,
        groups: u32,
    ) -> f64 {
        let pso = if pso {
            self.bench.clone()
        } else {
            self.probe.clone()
        };
        // SAFETY: the list is idle (every run is waited for); live objects.
        unsafe {
            self.allocator.Reset().unwrap();
            self.list.Reset(&self.allocator, None).unwrap();
            let l = &self.list;
            l.SetComputeRootSignature(&self.root);
            l.SetPipelineState(&pso);
            l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src);
            l.SetComputeRootUnorderedAccessView(2, rw);
            l.SetComputeRootUnorderedAccessView(3, sink);
            l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            l.Dispatch(groups, 1, 1);
            l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            l.ResolveQueryData(
                &self.stamps,
                D3D12_QUERY_TYPE_TIMESTAMP,
                0,
                2,
                &self.readback,
                0,
            );
            l.Close().unwrap();
            self.queue
                .ExecuteCommandLists(&[Some(l.cast::<ID3D12CommandList>().unwrap())]);
            self.done += 1;
            self.queue.Signal(&self.fence, self.done).unwrap();
            self.fence
                .SetEventOnCompletion(self.done, self.event)
                .unwrap();
            WaitForSingleObject(self.event, INFINITE);
            let t = std::slice::from_raw_parts(self.stamp_back, 2);
            t[1].saturating_sub(t[0]) as f64 * 1e3 / self.frequency as f64
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: our event.
        let _ = unsafe { CloseHandle(self.event) };
    }
}

/// Each adapter (and WARP): its device, name, whether software.
fn adapters() -> Vec<(ID3D12Device, String)> {
    let mut out = Vec::new();
    // SAFETY: plain DXGI and device creation.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return out;
        };
        let mut all: Vec<IDXGIAdapter1> = (0..8)
            .map_while(|i| factory.EnumAdapters1(i).ok())
            .collect();
        let warp: Option<IDXGIAdapter1> = factory.EnumWarpAdapter().ok();
        all.extend(warp);
        let mut seen = std::collections::HashSet::new();
        for a in all {
            let Ok(d) = a.GetDesc1() else { continue };
            if !seen.insert((d.AdapterLuid.HighPart, d.AdapterLuid.LowPart)) {
                continue;
            }
            let name = String::from_utf16_lossy(&d.Description)
                .trim_end_matches('\0')
                .to_string();
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&a, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                eprintln!("{name}: no D3D12 device");
                continue;
            }
            let name = format!(
                "{name} (dedicated {} MB, shared {} MB)",
                d.DedicatedVideoMemory >> 20,
                d.SharedSystemMemory >> 20
            );
            out.push((dev.unwrap(), name));
        }
    }
    out
}

/// What `device` reports and whether title memory opens in place there
/// (`crate::unified`, as the host logs it).
fn report(device: &ID3D12Device) {
    let t = std::time::Instant::now();
    let memory = crate::unified::detect(device);
    eprintln!("  {memory} (asked and tested in {:.2} ms)", t.elapsed().as_secs_f64() * 1e3);
}

/// Opens `title` as a heap and checks what the GPU and CPU see of each
/// other's writes, write watching, guards, and (views) remapping.
fn check(gpu: &mut Gpu, label: &str, title: &Title, watch: bool) {
    let dev = gpu.device.clone();
    let at = title.at();
    let t = std::time::Instant::now();
    let heap = match open(&dev, at) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("  {label}: OpenExistingHeapFromAddress failed: {e}");
            return;
        }
    };
    let opened = t.elapsed();
    // SAFETY: a live heap.
    let hd = unsafe { heap.GetDesc() };
    let buf = match placed(&dev, &heap, hd.SizeInBytes) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "  {label}: opened ({} bytes) but no placed buffer: {e}",
                hd.SizeInBytes
            );
            return;
        }
    };
    eprintln!(
        "  {label}: opened in {:.1} us: {} bytes, heap flags {:#x}, pool {} page {}",
        opened.as_secs_f64() * 1e6,
        hd.SizeInBytes,
        hd.Flags.0,
        hd.Properties.MemoryPoolPreference.0,
        hd.Properties.CPUPageProperty.0
    );
    let out = committed(&dev, system(), 1 << 20, true).unwrap();
    let outp = map(&out).cast::<u32>();
    // SAFETY: a live buffer.
    let (gva, ova) = unsafe { (buf.GetGPUVirtualAddress(), out.GetGPUVirtualAddress()) };
    let n = 2 * PAGE as u32 / 4;
    let words = title.dwords();
    for (i, w) in words[..n as usize].iter_mut().enumerate() {
        *w = i as u32 * 3 + 7;
    }
    if watch {
        // SAFETY: our memory.
        unsafe { mem::ResetWriteWatch(at.cast(), title.len()) };
    }
    gpu.dispatch(false, [0, n, 0, 1000], ova, gva, ova, n / 64);
    // SAFETY: the sink, written by the finished dispatch.
    let seen = unsafe { std::slice::from_raw_parts(outp, n as usize) };
    let gpu_saw_cpu = (0..n as usize).all(|i| seen[i] == i as u32 * 3 + 7);
    let words = title.dwords();
    let cpu_saw_gpu = (0..n as usize).all(|i| words[i] == i as u32 * 3 + 1007);
    let mut watched = String::from("-");
    if watch {
        let mut addrs = vec![std::ptr::null_mut::<c_void>(); 64];
        let (mut count, mut gran) = (64usize, 0u32);
        // SAFETY: buffers of the given sizes.
        let r = unsafe {
            mem::GetWriteWatch(
                0,
                at.cast(),
                title.len(),
                addrs.as_mut_ptr(),
                &mut count,
                &mut gran,
            )
        };
        // And CPU writes under the heap: still watched?
        // SAFETY: our memory.
        unsafe { mem::ResetWriteWatch(at.cast(), title.len()) };
        title.dwords()[5000] = 1;
        title.dwords()[100_000] = 1;
        let (mut count2, mut gran2) = (64usize, 0u32);
        // SAFETY: buffers of the given sizes.
        unsafe {
            mem::GetWriteWatch(
                0,
                at.cast(),
                title.len(),
                addrs.as_mut_ptr(),
                &mut count2,
                &mut gran2,
            )
        };
        watched = format!("{count} pages (r {r}); 2 CPU writes under the heap: {count2} pages");
    }
    // A read-only guard on the first page, then a GPU write there.
    let mut old = 0;
    // SAFETY: our memory.
    let protected =
        unsafe { mem::VirtualProtect(at.cast(), PAGE, mem::PAGE_READONLY, &mut old) } != 0;
    let before = title.dwords()[0];
    gpu.dispatch(false, [0, 64, 0, 1], ova, gva, ova, 1);
    let through_guard = title.dwords()[0] == before + 1;
    // SAFETY: our memory.
    unsafe { mem::VirtualProtect(at.cast(), PAGE, mem::PAGE_READWRITE, &mut old) };
    eprintln!(
        "    GPU sees CPU writes {gpu_saw_cpu}, CPU sees GPU writes {cpu_saw_gpu}, \
         write watch after GPU writes: {watched}; guard set {protected}, GPU write lands \
         through it {through_guard}"
    );
    // A second heap on the same memory.
    match open(&dev, at) {
        Ok(_) => eprintln!("    a second heap on the same memory: opens"),
        Err(e) => eprintln!("    a second heap on the same memory: {e}"),
    }
    if let Title::View { at, len, .. } = *title {
        // The title maps other pages there: what does the GPU read?
        let other = Title::section(len, true);
        // SAFETY: our view and placeholder; a new section of the same size.
        unsafe {
            let a = mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: at.cast() };
            let unmapped =
                mem::UnmapViewOfFile2(GetCurrentProcess(), a, mem::MEM_PRESERVE_PLACEHOLDER) != 0;
            let err = std::io::Error::last_os_error();
            let v = mem::MapViewOfFile3(
                other,
                GetCurrentProcess(),
                at.cast(),
                0,
                len,
                mem::MEM_REPLACE_PLACEHOLDER,
                mem::PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            let remapped = v.Value == at.cast();
            if remapped {
                std::ptr::write_bytes(at, 0xB0, 4 * 64);
            }
            gpu.dispatch(false, [0, 64, 0, 0], ova, gva, ova, 1);
            let got = std::slice::from_raw_parts(outp, 1)[0];
            eprintln!(
                "    remap: unmapped {unmapped} ({err}), new view mapped {remapped}; CPU reads \
                 {:#x}, GPU reads {got:#x} (old pages {:#x})",
                if remapped { *at.cast::<u32>() } else { 0 },
                before + 1
            );
            // Put the original back so Drop unmaps a view of its own section.
            if remapped {
                mem::UnmapViewOfFile2(GetCurrentProcess(), v, mem::MEM_PRESERVE_PLACEHOLDER);
            }
            windows_sys::Win32::Foundation::CloseHandle(other);
        }
    }
    drop(buf);
    drop(heap);
    // With the heap gone, guards and unmapping work again?
    let mut old = 0;
    // SAFETY: our memory.
    let again = unsafe { mem::VirtualProtect(at.cast(), PAGE, mem::PAGE_READONLY, &mut old) } != 0;
    // SAFETY: our memory.
    unsafe { mem::VirtualProtect(at.cast(), PAGE, mem::PAGE_READWRITE, &mut old) };
    eprintln!("    after the heap is released: guard set {again}");
}

/// Without any heap: guards, unmapping a view; a guard set before opening;
/// freeing memory under a heap.
fn controls(dev: &ID3D12Device) {
    let view = Title::view(4 << 20, false);
    let at = view.at();
    let mut old = 0;
    // SAFETY: our memory and view.
    unsafe {
        let guard = mem::VirtualProtect(at.cast(), PAGE, mem::PAGE_READONLY, &mut old) != 0;
        mem::VirtualProtect(at.cast(), PAGE, mem::PAGE_READWRITE, &mut old);
        let a = mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: at.cast() };
        let unmapped =
            mem::UnmapViewOfFile2(GetCurrentProcess(), a, mem::MEM_PRESERVE_PLACEHOLDER) != 0;
        eprintln!("  control, no heap: guard set {guard}, view unmapped {unmapped}");
        if unmapped {
            // Map it back for Drop.
            let Title::View { section, len, .. } = view else {
                unreachable!()
            };
            mem::MapViewOfFile3(
                section,
                GetCurrentProcess(),
                at.cast(),
                0,
                len,
                mem::MEM_REPLACE_PLACEHOLDER,
                mem::PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
        }
    }
    let pool = Title::private(4 << 20, true);
    // SAFETY: our memory.
    unsafe { mem::VirtualProtect(pool.at().cast(), PAGE, mem::PAGE_READONLY, &mut old) };
    let whole = open(dev, pool.at()).map(|_| ()).map_err(|e| e.to_string());
    let part = open1(dev, pool.at(), 4 << 20)
        .map(|_| ())
        .map_err(|e| e.to_string());
    // SAFETY: inside `pool`.
    let past = open1(dev, unsafe { pool.at().add(PAGE) }, 3 << 20)
        .map(|_| ())
        .map_err(|e| e.to_string());
    eprintln!(
        "  a guard on page 0 before opening: open {whole:?}, open1 whole {part:?}, \
         open1 past it {past:?}"
    );
    // SAFETY: our memory.
    unsafe { mem::VirtualProtect(pool.at().cast(), PAGE, mem::PAGE_READWRITE, &mut old) };
    // Decommitting and freeing under a live heap.
    let doomed = Title::private(4 << 20, true);
    let heap = open1(dev, doomed.at(), 4 << 20).ok();
    // SAFETY: our memory.
    let decommit = unsafe { mem::VirtualFree(doomed.at().cast(), PAGE, mem::MEM_DECOMMIT) } != 0;
    let err = std::io::Error::last_os_error();
    eprintln!(
        "  under a heap (opened {}): decommit a page {decommit} ({err})",
        heap.is_some()
    );
    drop(heap);
}

#[test]
#[ignore]
fn uma_probe() {
    for (dev, name) in adapters() {
        eprintln!("{name}");
        report(&dev);
        let mut gpu = Gpu::new(dev.clone());
        controls(&dev);
        check(
            &mut gpu,
            "private 4 MiB",
            &Title::private(4 << 20, false),
            false,
        );
        check(
            &mut gpu,
            "private write-watched 4 MiB",
            &Title::private(4 << 20, true),
            true,
        );
        check(
            &mut gpu,
            "section view (committed) 4 MiB",
            &Title::view(4 << 20, true),
            false,
        );
        check(
            &mut gpu,
            "section view (reserved, committed later) 4 MiB",
            &Title::view(4 << 20, false),
            false,
        );
        // Parts of an allocation, and sections themselves.
        let big = Title::private(8 << 20, true);
        for (off, len) in [(PAGE, 1 << 20), (4096, PAGE), (0, 1 << 20)] {
            // SAFETY: inside `big`.
            let at = unsafe { big.at().add(off) };
            match open(&dev, at) {
                Ok(h) => eprintln!(
                    "  open at +{off:#x} of an 8 MiB allocation: {} bytes",
                    // SAFETY: a live heap.
                    unsafe { h.GetDesc() }.SizeInBytes
                ),
                Err(e) => eprintln!("  open at +{off:#x} of an 8 MiB allocation: {e}"),
            }
            match open1(&dev, at, len) {
                Ok(h) => eprintln!(
                    "  open1 at +{off:#x}, {len:#x} bytes: {} bytes",
                    // SAFETY: a live heap.
                    unsafe { h.GetDesc() }.SizeInBytes
                ),
                Err(e) => eprintln!("  open1 at +{off:#x}, {len:#x} bytes: {e}"),
            }
        }
        for commit in [true, false] {
            let section = Title::section(4 << 20, commit);
            let d3: ID3D12Device3 = dev.cast().unwrap();
            // SAFETY: our section handle.
            let r = unsafe { d3.OpenExistingHeapFromFileMapping::<ID3D12Heap>(HANDLE(section)) };
            eprintln!(
                "  OpenExistingHeapFromFileMapping ({}): {}",
                if commit { "committed" } else { "reserved" },
                match r {
                    // SAFETY: a live heap.
                    Ok(h) => format!("{} bytes", unsafe { h.GetDesc() }.SizeInBytes),
                    Err(e) => e.to_string(),
                }
            );
            // SAFETY: ours.
            unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
        }
        // What opening costs by size (fresh write-watched allocations).
        for len in [PAGE, 4 << 20, 64 << 20, 256 << 20] {
            let title = Title::private(len, true);
            title.dwords()[0] = 1;
            let t = std::time::Instant::now();
            let h = open(&dev, title.at());
            let o = t.elapsed();
            let t = std::time::Instant::now();
            let b = h.as_ref().ok().map(|h| placed(&dev, h, len as u64));
            let p = t.elapsed();
            let t = std::time::Instant::now();
            drop(b);
            drop(h);
            let f = t.elapsed();
            eprintln!(
                "  open {} KiB: {:.1} us, placed buffer {:.1} us, release {:.1} us",
                len >> 10,
                o.as_secs_f64() * 1e6,
                p.as_secs_f64() * 1e6,
                f.as_secs_f64() * 1e6
            );
        }
    }
}

/// One placement's buffer: its GPU address, CPU mapping if any, whether
/// shaders may write it, and what keeps it alive.
struct Placement {
    label: &'static str,
    gpu: u64,
    cpu: Option<*mut u8>,
    uav: bool,
    _keep: (Option<ID3D12Resource>, Option<ID3D12Heap>, Option<Title>),
}

fn placements(dev: &ID3D12Device) -> Vec<Placement> {
    let mut out = Vec::new();
    let mut add = |label, r: windows::core::Result<ID3D12Resource>, uav, mapped: bool| match r {
        Ok(r) => {
            let cpu = mapped.then(|| map(&r));
            // SAFETY: a live buffer.
            let gpu = unsafe { r.GetGPUVirtualAddress() };
            out.push(Placement {
                label,
                gpu,
                cpu,
                uav,
                _keep: (Some(r), None, None),
            });
        }
        Err(e) => eprintln!("{label}: {e}"),
    };
    add(
        "system memory (today's copies)",
        committed(dev, system(), SIZE, true),
        true,
        true,
    );
    add(
        "upload heap (read only)",
        committed(dev, heap(D3D12_HEAP_TYPE_UPLOAD), SIZE, false),
        false,
        true,
    );
    add(
        "GPU upload heap (VRAM via BAR)",
        committed(dev, heap(D3D12_HEAP_TYPE_GPU_UPLOAD), SIZE, true),
        true,
        true,
    );
    add(
        "video memory (default heap)",
        committed(dev, heap(D3D12_HEAP_TYPE_DEFAULT), SIZE, true),
        true,
        false,
    );
    for (label, title) in [
        (
            "title pool opened in place",
            Title::private(SIZE as usize, true),
        ),
        (
            "title section view opened in place",
            Title::view(SIZE as usize, false),
        ),
    ] {
        let heap = open(dev, title.at()).and_then(|h| placed(dev, &h, SIZE).map(|b| (h, b)));
        match heap {
            Ok((h, b)) => {
                // SAFETY: a live buffer.
                let gpu = unsafe { b.GetGPUVirtualAddress() };
                let cpu = Some(title.at());
                out.push(Placement {
                    label,
                    gpu,
                    cpu,
                    uav: true,
                    _keep: (Some(b), Some(h), Some(title)),
                });
            }
            Err(e) => eprintln!("{label}: {e}"),
        }
    }
    out
}

#[test]
#[ignore]
fn uma_bench() {
    let hardware = adapters()
        .into_iter()
        .filter(|a| !a.1.starts_with("Microsoft Basic"));
    for (dev, name) in hardware {
        eprintln!("{name}");
        report(&dev);
        let mut gpu = Gpu::new(dev.clone());
        let sink = committed(&dev, heap(D3D12_HEAP_TYPE_DEFAULT), 1 << 16, true).unwrap();
        let dummy = committed(&dev, heap(D3D12_HEAP_TYPE_DEFAULT), SIZE, true).unwrap();
        // SAFETY: live buffers.
        let (sink, dummy) = unsafe { (sink.GetGPUVirtualAddress(), dummy.GetGPUVirtualAddress()) };
        let places = placements(&dev);
        // Fill what the CPU maps (touches the pages too).
        for p in &places {
            if let Some(c) = p.cpu {
                // SAFETY: SIZE bytes mapped.
                unsafe { std::ptr::write_bytes(c, 1, SIZE as usize) };
            }
        }
        let mut header = format!("{:<28}", "GB/s (ms)");
        for p in &places {
            header += &format!(" | {:>22}", &p.label[..p.label.len().min(22)]);
        }
        eprintln!("{header}");
        for (k, &(pname, count, bytes, writes)) in PATTERNS.iter().enumerate() {
            let mut row = format!("{pname:<28}");
            for p in &places {
                if writes && !p.uav {
                    row += &format!(" | {:>22}", "-");
                    continue;
                }
                let (src, rw) = if writes {
                    (dummy, p.gpu)
                } else {
                    (p.gpu, sink)
                };
                let c = [k as u32, count, SIZE as u32 - 1, 0];
                let mut best = f64::MAX;
                for _ in 0..4 {
                    best = best.min(gpu.dispatch(true, c, src, rw, sink, THREADS / 256));
                }
                let total = THREADS as f64 * count as f64 * bytes as f64;
                row += &format!(" | {:>12.1} ({:>6.2})", total / best / 1e6, best);
            }
            eprintln!("{row}");
        }
        // GPU copies between video memory and the others (refresh and
        // copy-back done by the GPU): 64 MiB whole, and 512 64 KiB pages of
        // every other one (32 MiB).
        let vram = places.iter().find(|p| p.label.starts_with("video memory"));
        if let Some(v) = vram.and_then(|v| v._keep.0.clone()) {
            eprintln!("GPU copies to and from video memory (GB/s):");
            for p in &places {
                let Some(r) = p._keep.0.clone() else { continue };
                if p.label.starts_with("video memory") {
                    continue;
                }
                let mut best = [f64::MAX; 4];
                for _ in 0..3 {
                    best[0] = best[0].min(gpu.copy(&v, &r, 64 << 20, 1));
                    best[1] = best[1].min(gpu.copy(&v, &r, PAGE as u64, 512));
                    if p.uav {
                        best[2] = best[2].min(gpu.copy(&r, &v, 64 << 20, 1));
                        best[3] = best[3].min(gpu.copy(&r, &v, PAGE as u64, 512));
                    }
                }
                let gbs = |ms: f64, n: f64| if ms == f64::MAX { 0.0 } else { n / ms / 1e6 };
                eprintln!(
                    "  {:<36} in: {:>6.1} whole, {:>6.1} pages; out: {:>6.1} whole, {:>6.1} pages",
                    p.label,
                    gbs(best[0], 64.0 * 1048576.0),
                    gbs(best[1], 32.0 * 1048576.0),
                    gbs(best[2], 64.0 * 1048576.0),
                    gbs(best[3], 32.0 * 1048576.0)
                );
            }
        }
        // CPU copies in and out of what it maps (64 MiB, one thread).
        let n = 64 << 20;
        let mut host = vec![3u8; n];
        eprintln!("CPU, one thread, 64 MiB:");
        for p in &places {
            let Some(c) = p.cpu else { continue };
            let mut w = f64::MAX;
            let mut r = f64::MAX;
            let tries = if p.label.starts_with("GPU upload") {
                1
            } else {
                3
            };
            for _ in 0..tries {
                let t = std::time::Instant::now();
                // SAFETY: SIZE bytes mapped.
                unsafe { std::ptr::copy_nonoverlapping(host.as_ptr(), c, n) };
                w = w.min(t.elapsed().as_secs_f64());
                let t = std::time::Instant::now();
                // SAFETY: SIZE bytes mapped.
                unsafe { std::ptr::copy_nonoverlapping(c, host.as_mut_ptr(), n) };
                r = r.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "  {:<36} write {:>6.1} GB/s, read {:>6.2} GB/s",
                p.label,
                n as f64 / w / 1e9,
                n as f64 / r / 1e9
            );
        }
        std::hint::black_box(&host);
    }
}
