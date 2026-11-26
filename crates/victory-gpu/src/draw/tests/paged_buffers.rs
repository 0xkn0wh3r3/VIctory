//! Supply a page-read mark to the normal remake pass, then read the remade physical slots
//! through a D3D12 descriptor table. The shader checks their contents after the fence.

use super::{compile_cs, view_heap};
use crate::descriptor::{self, host::HostHeap};
use crate::host::{Host, host};
use crate::object::{This, slot};
use crate::resource::{buffer_bytes, test_buffer};
use std::ptr::null;
use std::time::{Duration, Instant};
use windows::Win32::Graphics::Direct3D12::{ID3D12PipelineState, ID3D12RootSignature};

const PAGE: u32 = victory_shader::emit::VIEW_PAGE_SLOTS;
const A: u32 = 13 * PAGE + 7;
const B: u32 = 14 * PAGE + 7;
const GUARD: u32 = 0xdead_beef;
const AGE_STEPS: u64 = 400;

// Title heap CBVs are raw SRVs on the host, just as they are for translated shaders.
const SHADER: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2), UAV(u0)), RootConstants(num32BitConstants=1, b0)"
ByteAddressBuffer constantView : register(t0);
ByteAddressBuffer shaderView : register(t1);
RWByteAddressBuffer unorderedView : register(u0);
cbuffer roundTag : register(b0) { uint tag; }
[RootSignature(RS)]
[numthreads(1,1,1)]
void main() {
    unorderedView.Store2(0, uint2(constantView.Load(0), shaderView.Load(0)) ^ tag);
}
"#;

struct Object(This);

impl Drop for Object {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

struct Buffers([Object; 3]);

impl Buffers {
    fn new(inputs: [u32; 2]) -> Self {
        let buffers = Self(std::array::from_fn(|_| Object(test_buffer(256))));
        for (buffer, words) in buffers
            .0
            .iter()
            .zip([[inputs[0]; 3], [inputs[1]; 3], [GUARD; 3]])
        {
            let (data, size) = buffer_bytes(buffer.0).unwrap();
            assert_eq!(size, 256);
            // SAFETY: the first three words of this buffer's own title memory.
            unsafe { data.cast::<[u32; 3]>().write_unaligned(words) };
        }
        buffers
    }

    fn write_views(&self, first: u64) {
        let addr = buffer_bytes(self.0[0].0).unwrap().0 as u64;
        let cbv = [addr as u32, (addr >> 32) as u32, 256, 0];
        let srv = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        let uav = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        descriptor::create_cbv(null(), &cbv, first);
        descriptor::create_srv(null(), self.0[1].0, &srv, first + 32);
        descriptor::create_uav(null(), self.0[2].0, null(), &uav, first + 64);
    }

    fn output(&self) -> Vec<u32> {
        let addr = buffer_bytes(self.0[2].0).unwrap().0 as u64;
        crate::shadow::find(addr, 256).unwrap().read_host(addr, 3)
    }
}

/// Run the age pass only on our heap, after the submission fence while the queue is locked.
fn take_back(heap: &HostHeap, first: u32, start: Instant) {
    heap.set_page_water(u32::MAX, u32::MAX);
    for i in 1..=AGE_STEPS {
        heap.take_back_pages(start + Duration::from_secs(i));
    }
    heap.set_page_water(0, 0);
    assert_eq!(heap.host_page_of(first), Some(0));
    assert!((first..first + 3).all(|slot| !heap.kept(slot)));
}

fn dispatch(
    host: &Host,
    heap: &HostHeap,
    (signature, pipeline): &(ID3D12RootSignature, ID3D12PipelineState),
    first: u32,
    tag: u32,
    after: impl Fn() + Send + 'static,
) {
    assert!(
        (first..first + 3).all(|slot| heap.kept(slot)),
        "page {} must hold all three live buffer views before dispatch {tag}",
        first / PAGE
    );
    let list = host.new_list().unwrap();
    // SAFETY: the three adjacent slots of this physical page hold two raw SRVs and a raw UAV.
    unsafe {
        list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
        list.list.SetComputeRootSignature(signature);
        list.list.SetComputeRootDescriptorTable(0, heap.gpu(first));
        list.list.SetComputeRoot32BitConstant(1, tag, 0);
        list.list.SetPipelineState(pipeline);
        list.list.Dispatch(1, 1, 1);
    }
    list.defer_heap(&heap.viewed);
    list.defer_after(after);
    list.close();
    host.execute(&[&list]);
}

#[test]
fn buffer_views_are_remade_after_their_physical_page_is_reused() {
    let Some(host) = host() else { return };
    if !descriptor::host::view_pages() {
        eprintln!("unpaged adapter; VICTORY_VIEW_PAGES=1 enables this test");
        return;
    }
    let a = Buffers::new([0x1357, 0x2468]);
    let b = Buffers::new([0xabcd, 0x7890]);
    let (object, title, heap) = view_heap(15 * PAGE);
    let _owner = Object(object);
    let handle = |slot: u32| title.start + 32 * u64::from(slot);
    let pipeline = compile_cs(SHADER);
    a.write_views(handle(A));
    let physical = heap.page_table_entry(A / PAGE);
    assert_ne!(physical, 0);
    let gpu = heap.gpu(A);
    let start = Instant::now();
    dispatch(host, heap, &pipeline, A, 1, move || {
        take_back(heap, A, start)
    });
    assert_eq!(a.output(), [0x1357 ^ 1, 0x2468 ^ 1, GUARD]);
    assert!(
        (A..A + 3).all(|i| heap.made_from(i, handle(i))),
        "take-back retained A's keys"
    );

    b.write_views(handle(B));
    assert_eq!(heap.page_table_entry(B / PAGE), physical);
    assert_eq!(heap.gpu(B), gpu);
    dispatch(host, heap, &pipeline, B, 2, move || {
        take_back(heap, B, start + Duration::from_secs(AGE_STEPS));
        // A's records have not been rewritten. A page read places and remakes them all.
        descriptor::turn_pages(&[A / PAGE], Instant::now());
    });
    assert_eq!(b.output(), [0xabcd ^ 2, 0x7890 ^ 2, GUARD]);
    assert_eq!(heap.page_table_entry(A / PAGE), physical);
    assert_eq!(heap.gpu(A), gpu);
    dispatch(host, heap, &pipeline, A, 3, || {});
    assert_eq!(a.output(), [0x1357 ^ 3, 0x2468 ^ 3, GUARD]);
    assert_eq!(b.output(), [0xabcd ^ 2, 0x7890 ^ 2, GUARD]);
}
