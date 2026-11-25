//! End-to-end tests of wave64 compute programs on hosts with and without
//! 64-lane waves (`lower::wave64`): synthetic GFX10 programs of 64 to 128
//! threads with ballots, lane reads, EXEC changes over both halves of the
//! wave, whole-wave reductions and prefix sums, lowered for a host of
//! 32-lane waves (each console wave split in two) and for one of 64,
//! validated by `dxil.dll` when present, and run on every hardware adapter
//! and WARP whose wave sizes allow the program; every lane's results are
//! checked against a model of the console's wave.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

// ---- an assembler for the instructions the programs use ----

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC: u8 = 106;
const EXEC: u8 = 126;
const LITERAL: u16 = 255;

/// Inline integer constant `n` (0-64) as a source operand.
const fn k(n: u16) -> u16 {
    128 + n
}

const fn v(n: u8) -> u16 {
    256 + n as u16
}

const fn s(n: u8) -> u16 {
    n as u16
}

/// SOP1 `op sdst, src0`.
const fn sop1(op: u32, sdst: u8, src0: u16) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | op << 8 | src0 as u32
}

const fn s_mov_b32(sdst: u8, src0: u16) -> u32 {
    sop1(0x03, sdst, src0)
}

const fn s_mov_b64(sdst: u8, src0: u16) -> u32 {
    sop1(0x04, sdst, src0)
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u32, sdst: u8, src0: u16, src1: u16) -> u32 {
    0x8000_0000 | op << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// SOPP `op simm16`.
const fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | op << 16 | imm as u16 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0`.
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0`.
const fn vop1(op: u32, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | op << 9 | src0 as u32
}

const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    vop1(0x01, vdst, src0)
}

/// `v_readfirstlane_b32 sdst, src0`.
const fn v_readfirstlane_b32(sdst: u8, src0: u16) -> u32 {
    vop1(0x02, sdst, src0)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
}

/// VOP2 with DPP: `op vdst, vsrc0, vsrc1` with `ctrl`, the row and bank
/// write masks and bound_ctrl.
const fn vop2_dpp(op: u32, vdst: u8, vsrc0: u8, vsrc1: u8, ctrl: u32, row_mask: u32) -> [u32; 2] {
    [
        vop2(op, vdst, 0xFA, vsrc1),
        vsrc0 as u32 | ctrl << 8 | 1 << 19 | 0xF << 24 | row_mask << 28,
    ]
}

/// VOP1 `v_mov_b32` with DPP.
const fn v_mov_dpp(vdst: u8, vsrc0: u8, ctrl: u32) -> [u32; 2] {
    [
        vop1(0x01, vdst, 0xFA),
        vsrc0 as u32 | ctrl << 8 | 1 << 19 | 0xF << 24 | 0xF << 28,
    ]
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers); compares write the
/// SGPR pair `vdst`.
const fn vop3(op: u32, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | op << 16 | vdst as u32,
        src0 as u32 | (src1 as u32) << 9 | (src2 as u32) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen
/// offset:byte_off`.
const fn buffer_store_dword(vdata: u8, vaddr: u8, srsrc: u8, byte_off: u32) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12 | byte_off,
        0x80 << 24 | (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

// VOP2 opcodes.
const V_CNDMASK: u32 = 0x01;
const V_MAX_U32: u32 = 0x14;
const V_LSHRREV: u32 = 0x16;
const V_LSHLREV: u32 = 0x1A;
const V_AND: u32 = 0x1B;
const V_ADD: u32 = 0x25;
const V_SUB: u32 = 0x26;
// VOP3-only and VOPC opcodes.
const V_CMP_EQ_U32: u32 = 0xC2;
const V_CMP_GT_U32: u32 = 0xC4;
const V_CMP_NE_U32: u32 = 0xC5;
const V_MUL_LO_U32: u32 = 0x169;
const V_READLANE: u32 = 0x360;
const V_WRITELANE: u32 = 0x361;
const V_MBCNT_LO: u32 = 0x365;
const V_MBCNT_HI: u32 = 0x366;
const V_PERMLANEX16: u32 = 0x378;
// SOP1 and SOP2 opcodes.
const S_BCNT1_B64: u32 = 0x10;
const S_FF1_B64: u32 = 0x14;
const S_AND_SAVEEXEC_B64: u32 = 0x24;
const S_OR_SAVEEXEC_B64: u32 = 0x25;
const S_OR_B64: u32 = 0x11;
// DPP controls.
const QUAD_ID: u32 = 0xE4;
const fn row_shr(n: u32) -> u32 {
    0x110 | n
}
const fn row_xmask(n: u32) -> u32 {
    0x160 | n
}
const ROW_SHARE_15: u32 = 0x15F;

/// The prologue of every program: the UAV descriptor in s[4:7] (the table
/// at s0), the thread id x in v0, its byte address (×4) in v10, and the
/// lane's data `x` in v1 ([`data`]) and its bit 0 in v2.
fn prologue() -> Vec<u32> {
    let mut w = vec![s_mov_b32(1, k(0))];
    w.extend(s_load_dwordx4(4, 0));
    w.push(vop2(V_LSHLREV, 10, k(2), 0));
    w.extend([s_mov_b32(2, LITERAL), 0x9E37_79B1]);
    w.extend(vop3(V_MUL_LO_U32, 1, v(0), s(2), 0));
    w.push(vop2(V_LSHRREV, 1, k(9), 1));
    w.push(vop2(V_AND, 2, k(1), 1));
    w
}

/// The data the prologue gives thread `t` (v1).
fn data(t: u32) -> u32 {
    t.wrapping_mul(0x9E37_79B1) >> 9
}

/// Stores `vdata` as result `n`: dword `n * threads + t` of the UAV.
fn store(vdata: u8, n: u32, threads: u32) -> [u32; 2] {
    buffer_store_dword(vdata, 10, 4, 4 * n * threads)
}

// ---- building and running ----

/// Lowers `words` as a wave64 compute program of `threads` threads (x
/// only, the UAV table at s0 and `lds` bytes of LDS) for a host with
/// 64-lane waves or without.
fn lower_cs(
    words: &[u32],
    threads: u32,
    lds: u32,
    host_wave64: bool,
) -> Result<HostShader, String> {
    lower_counting(words, threads, lds, host_wave64).map(|(s, _)| s)
}

/// [`lower_cs`], and the group syncs the split made.
fn lower_counting(
    words: &[u32],
    threads: u32,
    lds: u32,
    host_wave64: bool,
) -> Result<(HostShader, u32), String> {
    lower_waves(words, threads, lds, true, host_wave64, false)
}

/// [`lower_counting`] of a console wave64 program, or (not `wave64`) of a
/// wave32 one; with `group_id`, the group id x in s1.
fn lower_waves(
    words: &[u32],
    threads: u32,
    lds: u32,
    wave64: bool,
    host_wave64: bool,
    group_id: bool,
) -> Result<(HostShader, u32), String> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | u32::from(group_id) << 7 | (lds / 512) << 15,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower_for(
        &lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
            native16: false,
        },
        host_wave64,
    )
    .map_err(|e| e.to_string())?;
    let syncs = lowered.group_syncs;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok((shader, syncs))
}

/// Runs `words` (a wave64 program of `threads` threads storing `results`
/// results per thread) on every hardware adapter and WARP: lowered for
/// 32-lane host waves where the device runs them, and for 64-lane ones
/// where it runs those. Checks each run's results against `want` (result
/// n of thread t at `want[n * threads + t]`). Returns how many runs had
/// host waves of 32 lanes.
fn run_everywhere(words: &[u32], threads: u32, results: u32, want: &[u32]) -> usize {
    assert_eq!(want.len() as u32, results * threads);
    let split = lower_cs(words, threads, 0, false).expect("lowered for 32-lane waves");
    let native = lower_cs(words, threads, 0, true).expect("lowered for 64-lane waves");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut hosts = Host::hardware();
    match Host::new(true) {
        Ok(h) => hosts.push(("WARP".into(), h)),
        Err(e) => eprintln!("no SM 6.6 WARP device ({e}); skipped"),
    }
    let (mut ran, mut ran32) = (0, 0);
    for (name, host) in &hosts {
        let (min, max) = host.wave_lanes();
        for (shader, lanes) in [(&split, 32), (&native, 64)] {
            if !(min..=max).contains(&lanes) {
                eprintln!("{name}: waves of {min}-{max} lanes; {lanes}-lane program skipped");
                continue;
            }
            eprintln!("{name}: waves of {min}-{max} lanes; running the {lanes}-lane program");
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{name} ({lanes} lanes): pipeline: {e}"));
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(results * threads)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{name} ({lanes} lanes): dispatch: {e}"));
            for (i, (&got, &w)) in out[0].iter().zip(want).enumerate() {
                let (n, t) = (i as u32 / threads, i as u32 % threads);
                assert_eq!(
                    got,
                    w,
                    "{name}, {lanes}-lane waves: result {n} of thread {t} (wave {}, lane {}) \
                     is {got:#x}, want {w:#x}",
                    t / 64,
                    t % 64
                );
            }
            ran += 1;
            if lanes == 32 {
                ran32 += 1;
            }
        }
    }
    assert!(ran > 0, "no SM 6.6 device ran the program");
    ran32
}

/// The console's view of a group: per wave, the threads of its lanes.
fn waves(threads: u32) -> impl Iterator<Item = std::ops::Range<u32>> {
    (0..threads.div_ceil(64)).map(move |w| 64 * w..(64 * w + 64).min(threads))
}

/// The lane mask of the lanes of `wave` where `f` holds.
fn mask(wave: &std::ops::Range<u32>, f: impl Fn(u32) -> bool) -> u64 {
    wave.clone()
        .filter(|&t| f(t))
        .fold(0, |m, t| m | 1 << (t % 64))
}

// ---- programs ----

/// Ballots of a per-lane bit over both halves: the mask's dwords, its
/// population count and first lane (`s_bcnt1`, `s_ff1`) and each lane's
/// rank among the lanes with the bit (`v_mbcnt_lo`, `v_mbcnt_hi`).
fn ballot_program(threads: u32) -> Vec<u32> {
    let mut w = prologue();
    w.extend(vop3(V_CMP_NE_U32, 10, k(0), v(2), 0)); // s[10:11] = x & 1
    w.push(sop1(S_BCNT1_B64, 12, s(10)));
    w.push(sop1(S_FF1_B64, 13, s(10)));
    w.extend(vop3(V_MBCNT_LO, 3, s(10), k(0), 0));
    w.extend(vop3(V_MBCNT_HI, 3, s(11), v(3), 0));
    w.push(v_mov_b32(4, s(10)));
    w.push(v_mov_b32(5, s(11)));
    w.push(v_mov_b32(6, s(12)));
    w.push(v_mov_b32(7, s(13)));
    for (n, r) in [4u8, 5, 6, 7, 3].into_iter().enumerate() {
        w.extend(store(r, n as u32, threads));
    }
    w.push(S_ENDPGM);
    w
}

fn ballot_model(threads: u32) -> Vec<u32> {
    let mut want = vec![0; 5 * threads as usize];
    for wave in waves(threads) {
        let m = mask(&wave, |t| data(t) & 1 != 0);
        for t in wave.clone() {
            let below = m & ((1u64 << (t % 64)) - 1);
            let r = [
                m as u32,
                (m >> 32) as u32,
                m.count_ones(),
                if m == 0 { u32::MAX } else { m.trailing_zeros() },
                below.count_ones(),
            ];
            for (n, x) in r.into_iter().enumerate() {
                want[n * threads as usize + t as usize] = x;
            }
        }
    }
    want
}

/// Lane reads: `v_readlane_b32` of lanes 5 and 40 and of lane 45 named by
/// an SGPR, `v_readfirstlane_b32` with every lane in EXEC and with only
/// the lanes of a compare, a ballot under that EXEC, and `v_writelane_b32`
/// of lane 50.
fn lanes_program(threads: u32) -> Vec<u32> {
    let mut w = prologue();
    w.extend(vop3(V_READLANE, 14, v(1), k(5), 0));
    w.extend(vop3(V_READLANE, 15, v(1), k(40), 0));
    w.push(s_mov_b32(16, k(45)));
    w.extend(vop3(V_READLANE, 17, v(1), s(16), 0));
    w.push(v_readfirstlane_b32(18, v(1)));
    w.push(v_mov_b32(5, v(0)));
    w.extend(vop3(V_WRITELANE, 5, s(15), k(50), 0));
    // EXEC = lanes whose x has bit 0 and whose thread id is above 36 in
    // its wave, then the first of them and a ballot under that EXEC.
    w.push(s_mov_b64(40, EXEC as u16));
    w.extend(vop3(V_CMP_NE_U32, 10, k(0), v(2), 0));
    w.push(vop2(V_AND, 6, k(63), 0));
    w.extend(vop3(V_CMP_GT_U32, 12, v(6), k(36), 0));
    w.push(sop2(0x0F, 20, s(10), s(12))); // s_and_b64 s[20:21]
    w.push(sop1(S_AND_SAVEEXEC_B64, 22, s(20)));
    w.push(v_readfirstlane_b32(19, v(1)));
    w.extend([s_mov_b32(3, LITERAL), 0x0040_0000]);
    w.extend(vop3(V_CMP_GT_U32, 24, v(1), s(3), 0)); // x > 2^22
    w.push(sop1(S_BCNT1_B64, 23, s(24)));
    w.push(s_mov_b64(EXEC, s(40)));
    w.push(v_mov_b32(6, s(14)));
    w.push(v_mov_b32(7, s(15)));
    w.push(v_mov_b32(8, s(17)));
    w.push(v_mov_b32(9, s(18)));
    w.push(v_mov_b32(11, s(19)));
    w.push(v_mov_b32(12, s(23)));
    for (n, r) in [6u8, 7, 8, 9, 11, 12, 5].into_iter().enumerate() {
        w.extend(store(r, n as u32, threads));
    }
    w.push(S_ENDPGM);
    w
}

fn lanes_model(threads: u32) -> Vec<u32> {
    let mut want = vec![0; 7 * threads as usize];
    for wave in waves(threads) {
        let x = |l: u32| data(wave.start + l);
        let exec = mask(&wave, |t| data(t) & 1 != 0 && t % 64 > 36);
        let first = if exec == 0 { 0 } else { exec.trailing_zeros() };
        let gt = mask(&wave, |t| data(t) > 0x0040_0000) & exec;
        for t in wave.clone() {
            let l = t % 64;
            let r = [
                x(5),
                x(40),
                x(45),
                x(0),
                x(first),
                gt.count_ones(),
                if l == 50 { x(40) } else { t },
            ];
            for (n, v) in r.into_iter().enumerate() {
                want[n * threads as usize + t as usize] = v;
            }
        }
    }
    want
}

/// Whole-wave sections as the console compiler expands wave operations
/// (`s_or_saveexec -1` ... restore), over the lanes of a compare: an
/// inclusive prefix sum (row prefixes by `row_shr`, rows joined by
/// `row_share`, `v_permlanex16` and a row-masked add, halves by
/// `v_readlane` of lane 31), the exclusive one (less the lane's own term),
/// the total (`v_readlane` of lane 63) and a maximum (`row_xmask`
/// butterflies, `v_permlanex16`, both halves' `v_readlane`).
fn reduce_program(threads: u32) -> Vec<u32> {
    let mut w = prologue();
    w.push(s_mov_b64(40, EXEC as u16));
    w.push(vop2(V_AND, 15, k(3), 1));
    w.extend(vop3(V_CMP_NE_U32, VCC, k(0), v(15), 0)); // (x & 3) != 0
    w.push(sop1(S_AND_SAVEEXEC_B64, 42, VCC as u16));
    w.push(sop1(S_OR_SAVEEXEC_B64, 30, k(65))); // -1: every lane
    // Sum: v7 = the lane's term, then its inclusive prefix.
    w.extend(vop3(0x100 | V_CNDMASK, 7, k(0), v(1), s(30))); // v_cndmask_b32 v7, 0, v1
    w.push(v_mov_b32(13, v(7)));
    for n in [1, 2, 4, 8] {
        w.extend(vop2_dpp(V_ADD, 7, 7, 7, row_shr(n), 0xF));
    }
    w.extend(v_mov_dpp(9, 7, ROW_SHARE_15)); // the row's total
    // Each lane reads lane 15 of the other row of its half.
    w.extend(vop3(V_PERMLANEX16, 9, v(9), k(65), k(65)));
    w.extend(vop2_dpp(V_ADD, 7, 9, 7, QUAD_ID, 0xA)); // rows 1 and 3
    w.extend(vop3(V_READLANE, 32, v(7), k(31), 0));
    w.push(v_mov_b32(11, s(32)));
    w.extend(vop2_dpp(V_ADD, 7, 11, 7, QUAD_ID, 0xC)); // rows 2 and 3
    w.extend(vop3(V_READLANE, 33, v(7), k(63), 0));
    w.push(vop2(V_SUB, 12, v(7), 13));
    // Maximum: v8 = the lane's term, then the wave's maximum.
    w.extend(vop3(0x100 | V_CNDMASK, 8, k(0), v(1), s(30)));
    for n in [1, 2, 4, 8] {
        w.extend(vop2_dpp(V_MAX_U32, 8, 8, 8, row_xmask(n), 0xF));
    }
    w.extend(vop3(V_PERMLANEX16, 14, v(8), k(65), k(65)));
    w.push(vop2(V_MAX_U32, 8, v(14), 8));
    w.extend(vop3(V_READLANE, 34, v(8), k(0), 0));
    w.extend(vop3(V_READLANE, 35, v(8), k(63), 0));
    w.push(vop2(V_MAX_U32, 8, s(34), 8));
    w.push(vop2(V_MAX_U32, 8, s(35), 8));
    w.push(s_mov_b64(EXEC, s(30)));
    w.push(s_mov_b64(EXEC, s(40)));
    w.push(v_mov_b32(14, s(33)));
    for (n, r) in [7u8, 12, 14, 8].into_iter().enumerate() {
        w.extend(store(r, n as u32, threads));
    }
    w.push(S_ENDPGM);
    w
}

fn reduce_model(threads: u32) -> Vec<u32> {
    let mut want = vec![0; 4 * threads as usize];
    for wave in waves(threads) {
        let term = |t: u32| if data(t) & 3 != 0 { data(t) } else { 0 };
        let total = wave.clone().fold(0u32, |a, t| a.wrapping_add(term(t)));
        let max = wave.clone().map(term).max().unwrap_or(0);
        for t in wave.clone() {
            let incl = (wave.start..=t).fold(0u32, |a, u| a.wrapping_add(term(u)));
            let r = [incl, incl.wrapping_sub(term(t)), total, max];
            for (n, v) in r.into_iter().enumerate() {
                want[n * threads as usize + t as usize] = v;
            }
        }
    }
    want
}

/// A ballot after a branch some lanes of each half take (`if (x & 1) x
/// += 5`): the join block is every thread's, so the halves exchange there.
fn joined_program(threads: u32) -> Vec<u32> {
    let mut w = prologue();
    w.extend(vop3(V_CMP_NE_U32, VCC, k(0), v(2), 0));
    w.push(sop1(S_AND_SAVEEXEC_B64, 20, VCC as u16));
    w.push(sopp(0x08, 1)); // s_cbranch_execz past the add
    w.push(vop2(V_ADD, 1, k(5), 1));
    w.push(sop2(S_OR_B64, EXEC, EXEC as u16, s(20)));
    w.push(vop2(V_AND, 3, k(4), 1));
    w.extend(vop3(V_CMP_NE_U32, 10, k(0), v(3), 0)); // (x & 4) != 0
    w.push(sop1(S_BCNT1_B64, 12, s(10)));
    w.extend(vop3(V_MBCNT_LO, 5, s(10), k(0), 0));
    w.extend(vop3(V_MBCNT_HI, 5, s(11), v(5), 0));
    w.push(v_mov_b32(6, s(12)));
    w.extend(store(5, 0, threads));
    w.extend(store(6, 1, threads));
    w.push(S_ENDPGM);
    w
}

fn joined_model(threads: u32) -> Vec<u32> {
    let mut want = vec![0; 2 * threads as usize];
    let x = |t: u32| data(t) + if data(t) & 1 != 0 { 5 } else { 0 };
    for wave in waves(threads) {
        let m = mask(&wave, |t| x(t) & 4 != 0);
        for t in wave.clone() {
            let below = m & ((1u64 << (t % 64)) - 1);
            want[t as usize] = below.count_ones();
            want[(threads + t) as usize] = m.count_ones();
        }
    }
    want
}

/// DS `ds_bpermute_b32 vdst, addr, data0`.
const fn ds_bpermute_b32(vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [
        0x36 << 26 | 0xB3 << 18,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

/// What the permutes start from in the lanes outside their EXEC.
const PERMUTE_OLD: u32 = 0xDEAD_0000;

/// `ds_bpermute_b32` of lane `(l + 37) & 63` (half the lanes read the
/// other half), with every lane in EXEC, then under the lanes whose x has
/// bit 0 (lanes outside EXEC read 0).
fn permute_program(threads: u32) -> Vec<u32> {
    let mut w = prologue();
    w.push(vop2(V_ADD, 3, k(37), 0));
    w.push(vop2(V_AND, 3, k(63), 3));
    w.push(vop2(V_LSHLREV, 3, k(2), 3));
    w.extend(ds_bpermute_b32(4, 3, 1));
    w.extend([v_mov_b32(5, LITERAL), PERMUTE_OLD]);
    w.push(s_mov_b64(40, EXEC as u16));
    w.extend(vop3(V_CMP_NE_U32, VCC, k(0), v(2), 0));
    w.push(sop1(S_AND_SAVEEXEC_B64, 20, VCC as u16));
    w.extend(ds_bpermute_b32(5, 3, 1));
    w.push(s_mov_b64(EXEC, s(40)));
    w.extend(store(4, 0, threads));
    w.extend(store(5, 1, threads));
    w.push(S_ENDPGM);
    w
}

fn permute_model(threads: u32) -> Vec<u32> {
    let mut want = vec![0; 2 * threads as usize];
    for wave in waves(threads) {
        let odd = mask(&wave, |t| data(t) & 1 != 0);
        for t in wave.clone() {
            let from = (t % 64 + 37) & 63;
            let src = wave.start + from;
            let all = if src < wave.end { data(src) } else { 0 };
            want[t as usize] = all;
            want[(threads + t) as usize] = if data(t) & 1 == 0 {
                PERMUTE_OLD
            } else if odd >> from & 1 != 0 {
                data(src)
            } else {
                0
            };
        }
    }
    want
}

// ---- tests ----

#[test]
fn the_assembler_encodes_what_the_decoder_reads() {
    let text = |words: &[u32]| -> Vec<String> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .unwrap()
            .iter()
            .map(|i| i.text(true))
            .collect()
    };
    let t = text(&reduce_program(128)).join("\n");
    for want in [
        "s_or_saveexec_b64 s[30:31], -1",
        "v_cndmask_b32 v7, 0, v1, s[30:31]",
        "row_shr:8",
        "row_share:15",
        "v_permlanex16_b32 v9, v9, -1, -1",
        "row_mask:0xa",
        "v_readlane_b32 s33, v7, 63",
        "buffer_store_dword v8, v10, s[4:7], 0 offen offset:1536",
    ] {
        assert!(t.contains(want), "no `{want}` in\n{t}");
    }
    let t = text(&lanes_program(128)).join("\n");
    for want in [
        "v_readlane_b32 s17, v1, s16",
        "v_writelane_b32 v5, s15, 50",
        "v_readfirstlane_b32 s19, v1",
        "s_and_saveexec_b64 s[22:23], s[20:21]",
    ] {
        assert!(t.contains(want), "no `{want}` in\n{t}");
    }
    let t = text(&ballot_program(128)).join("\n");
    for want in [
        "v_cmp_ne_u32 s[10:11], 0, v2",
        "s_bcnt1_i32_b64 s12, s[10:11]",
        "s_ff1_i32_b64 s13, s[10:11]",
        "v_mbcnt_hi_u32_b32 v3, s11, v3",
    ] {
        assert!(t.contains(want), "no `{want}` in\n{t}");
    }
}

/// Two console waves of 64 lanes (four host waves of 32), and one and a
/// half (the second wave's high half missing).
const GROUPS: [u32; 3] = [128, 96, 64];

#[test]
fn ballots_span_both_halves_of_the_wave() {
    let mut ran32 = 0;
    for threads in GROUPS {
        ran32 += run_everywhere(&ballot_program(threads), threads, 5, &ballot_model(threads));
    }
    assert!(ran32 > 0, "no device ran 32-lane waves");
}

#[test]
fn lane_reads_reach_the_other_half() {
    let mut ran32 = 0;
    // (Not 96: lanes 40 and 45 of its second wave do not exist.)
    for threads in [128, 64] {
        ran32 += run_everywhere(&lanes_program(threads), threads, 7, &lanes_model(threads));
    }
    assert!(ran32 > 0, "no device ran 32-lane waves");
}

#[test]
fn whole_wave_reductions_and_prefix_sums_join_the_halves() {
    let mut ran32 = 0;
    for threads in GROUPS {
        ran32 += run_everywhere(&reduce_program(threads), threads, 4, &reduce_model(threads));
    }
    assert!(ran32 > 0, "no device ran 32-lane waves");
}

#[test]
fn a_ballot_after_a_branch_joins_the_halves() {
    let mut ran32 = 0;
    for threads in GROUPS {
        ran32 += run_everywhere(&joined_program(threads), threads, 2, &joined_model(threads));
    }
    assert!(ran32 > 0, "no device ran 32-lane waves");
}

#[test]
fn permutes_read_lanes_of_the_other_half() {
    let mut ran32 = 0;
    for threads in GROUPS {
        ran32 += run_everywhere(
            &permute_program(threads),
            threads,
            2,
            &permute_model(threads),
        );
    }
    assert!(ran32 > 0, "no device ran 32-lane waves");
}

/// A group of at most 32 threads is one host wave of 32: nothing to
/// exchange, the high dword of every ballot 0.
#[test]
fn a_group_of_32_threads_needs_no_exchange() {
    let ran32 = run_everywhere(&ballot_program(32), 32, 5, &ballot_model(32));
    assert!(ran32 > 0, "no device ran 32-lane waves");
    // No exchange, so no group sync; 64 threads exchange each ballot's
    // dword (store, sync, load, sync).
    assert_eq!(
        lower_counting(&ballot_program(32), 32, 0, false).unwrap().1,
        0
    );
    assert_eq!(
        lower_counting(&ballot_program(64), 64, 0, false).unwrap().1,
        2
    );
    assert_eq!(
        lower_counting(&ballot_program(64), 64, 0, true).unwrap().1,
        0
    );
}

/// A ballot inside a branch: the group's other threads may not get there,
/// so the halves cannot sync to exchange; refused on 32-lane hosts only.
#[test]
fn a_ballot_in_divergent_code_is_refused_without_64_lane_waves() {
    let mut w = prologue();
    w.extend(vop3(V_CMP_NE_U32, VCC, k(0), v(2), 0));
    w.push(sop1(S_AND_SAVEEXEC_B64, 20, VCC as u16));
    w.push(sopp(0x08, 6)); // s_cbranch_execz past the ballot and store
    w.extend(vop3(V_CMP_GT_U32, 12, v(1), s(2), 0));
    w.push(sop1(S_BCNT1_B64, 14, s(12)));
    w.push(v_mov_b32(5, s(14)));
    w.extend(store(5, 0, 128));
    w.push(sop2(S_OR_B64, EXEC, EXEC as u16, s(20)));
    w.push(S_ENDPGM);
    let Err(e) = lower_cs(&w, 128, 0, false) else {
        panic!("not refused")
    };
    assert!(e.contains("not every thread of the group arrives"), "{e}");
    lower_cs(&w, 128, 0, true).expect("lowered for 64-lane waves");
    // One host wave holds a group of 32: nothing to exchange.
    lower_cs(&w, 32, 0, false).expect("one host wave");
}

/// A readlane in a loop: each pass would need every thread in step.
#[test]
fn a_lane_read_in_a_loop_is_refused_without_64_lane_waves() {
    let mut w = prologue();
    w.push(s_mov_b32(16, k(0)));
    // loop: s17 = v1 of lane s16; v1 += s17; s16 += 1; while s16 != 3
    w.extend(vop3(V_READLANE, 17, v(1), s(16), 0));
    w.push(vop2(V_ADD, 1, s(17), 1));
    w.push(sop2(0x00, 16, s(16), k(1))); // s_add_u32
    w.push(0xBF07_0000 | (k(3) as u32) << 8 | 16); // s_cmp_lg_u32 s16, 3
    w.push(sopp(0x05, -6)); // s_cbranch_scc1 loop
    w.extend(store(1, 0, 128));
    w.push(S_ENDPGM);
    let Err(e) = lower_cs(&w, 128, 0, false) else {
        panic!("not refused")
    };
    assert!(e.contains("not every thread of the group arrives"), "{e}");
    lower_cs(&w, 128, 0, true).expect("lowered for 64-lane waves");
}

/// LDS in a group that is one console wave: the console compiler leaves
/// out the barriers that two host waves would need.
#[test]
fn lds_in_one_wave64_wave_of_more_than_32_threads_is_refused() {
    let mut w = prologue();
    w.extend(vop3(V_READLANE, 14, v(1), k(40), 0));
    w.push(v_mov_b32(5, s(14)));
    w.extend(store(5, 0, 64));
    w.push(S_ENDPGM);
    let Err(e) = lower_cs(&w, 64, 512, false) else {
        panic!("not refused")
    };
    assert!(e.contains("LDS shared without barriers"), "{e}");
    lower_cs(&w, 64, 512, true).expect("lowered for 64-lane waves");
    lower_cs(&w, 32, 512, false).expect("one host wave");
    lower_cs(&w, 128, 512, false).expect("two console waves");
}

// ---- what the split costs ----

/// The cross-lane operation of each block of [`timing_program`].
#[derive(Clone, Copy, PartialEq)]
enum Cross {
    /// None: the program reads only its own lane's console lane.
    None,
    /// `v_readlane_b32` of lane 5, added to every lane.
    ReadLane,
    /// `v_readfirstlane_b32` (every lane in EXEC), added to every lane.
    ReadFirstLane,
}

/// The prologue, each lane's console lane (`v_mbcnt`) added to its data,
/// then `blocks` blocks of `alu` multiply-xorshifts and one `cross` operation,
/// and the result stored by group 0 (the group id x in s1), and by lanes of
/// other groups whose result is 0x12345678 (the same value in the same
/// dword: every group computes what group 0 does). Other groups' stores to
/// the same dwords made a floor of their own, and a store by group 0 alone
/// let the compiler skip the work everywhere else. The same words as a
/// wave64 and as a wave32 program (`wave64`), but for the masks at the end.
fn timing_program(threads: u32, blocks: u32, alu: u32, cross: Cross, wave64: bool) -> Vec<u32> {
    let mut w = vec![s_mov_b32(20, s(1))];
    w.extend(prologue());
    w.extend(vop3(V_MBCNT_LO, 3, 0xC1, k(0), 0)); // -1
    w.extend(vop3(V_MBCNT_HI, 3, 0xC1, v(3), 0));
    w.push(vop2(V_ADD, 1, v(3), 1));
    for _ in 0..blocks {
        // Multiply and xor-shift: not affine, so no compiler folds a chain.
        for _ in 0..alu {
            w.extend(vop3(V_MUL_LO_U32, 1, v(1), s(2), 0));
            w.push(vop2(V_LSHRREV, 4, k(9), 1));
            w.push(vop2(0x1D, 1, v(4), 1)); // v_xor_b32
        }
        match cross {
            Cross::None => {}
            Cross::ReadLane => w.extend(vop3(V_READLANE, 14, v(1), k(5), 0)),
            Cross::ReadFirstLane => w.push(v_readfirstlane_b32(14, v(1))),
        }
        if cross != Cross::None {
            w.push(vop2(V_ADD, 1, s(14), 1));
        }
    }
    w.extend([s_mov_b32(3, LITERAL), 0x1234_5678]);
    w.extend(vop3(V_CMP_EQ_U32, 22, s(3), v(1), 0));
    w.push(0xBF06_8014); // s_cmp_eq_u32 s20, 0
    if wave64 {
        w.push(sop2(0x0B, 22, 0xC1, s(22))); // s_cselect_b64 s[22:23], -1, s[22:23]
        w.push(sop1(S_AND_SAVEEXEC_B64, 26, s(22)));
    } else {
        w.push(sop2(0x0A, 22, 0xC1, s(22))); // s_cselect_b32 s22, -1, s22
        w.push(sop1(0x3C, 26, s(22))); // s_and_saveexec_b32 s26, s22
    }
    w.extend(store(1, 0, threads));
    w.push(S_ENDPGM);
    w
}

/// [`timing_program`]'s results in console waves of `lanes` lanes.
fn timing_model(threads: u32, blocks: u32, alu: u32, cross: Cross, lanes: u32) -> Vec<u32> {
    let mut vals: Vec<u32> = (0..threads)
        .map(|t| data(t).wrapping_add(t % lanes))
        .collect();
    for _ in 0..blocks {
        for x in vals.iter_mut() {
            for _ in 0..alu {
                *x = x.wrapping_mul(0x9E37_79B1);
                *x ^= *x >> 9;
            }
        }
        let from = match cross {
            Cross::None => continue,
            Cross::ReadLane => 5,
            Cross::ReadFirstLane => 0,
        };
        let before = vals.clone();
        for (t, x) in vals.iter_mut().enumerate() {
            *x = x.wrapping_add(before[t / lanes as usize * lanes as usize + from]);
        }
    }
    vals
}

/// The GPU's time per dispatch of compute-heavy programs (up to 65536
/// groups of 256 threads) on each hardware adapter: a console wave64 program split
/// into two 32-lane host waves (`lower::wave64`), the same words as a
/// wave32 program (the work without the split: its cross-lane reads stay
/// in 32 lanes, so its results differ), and the wave64 program on 64-lane
/// host waves where the adapter runs them. Each run's results are checked
/// against the console's (for its wave size).
/// `cargo test --release -p victory-shader --lib wave64_split_cost -- --ignored --nocapture`
#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
fn wave64_split_cost() {
    const THREADS: u32 = 256;
    const GROUPS: u32 = 65536;
    const REPEAT: u32 = 50;
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = Host::hardware();
    if hosts.is_empty() {
        eprintln!("no hardware adapter; skipped");
        return;
    }
    let cases = [
        ("lane ids only, 128 ALU steps", 16, 8, Cross::None),
        ("1 v_readlane per 64 ALU steps (x2)", 2, 64, Cross::ReadLane),
        ("1 v_readlane per 32 ALU steps (x4)", 4, 32, Cross::ReadLane),
        ("1 v_readlane per 8 ALU steps (x16)", 16, 8, Cross::ReadLane),
        (
            "1 v_readfirstlane per 8 ALU steps (x16)",
            16,
            8,
            Cross::ReadFirstLane,
        ),
        ("16 v_readlane back to back", 16, 0, Cross::ReadLane),
        ("64 v_readlane back to back", 64, 0, Cross::ReadLane),
    ];
    'adapters: for (name, host) in &hosts {
        let (min, max) = host.wave_lanes();
        // Groups for about 0.4 ms per dispatch of the first program, at most
        // GROUPS: an integrated GPU's list of 51 dispatches of GROUPS groups
        // ran past the timeout that removes the device.
        let groups = {
            let (blocks, alu, cross) = (cases[0].1, cases[0].2, cases[0].3);
            let words = timing_program(THREADS, blocks, alu, cross, false);
            let (shader, _) = lower_waves(&words, THREADS, 0, false, false, true).unwrap();
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let d = Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(THREADS)],
                samplers: &[],
                groups: [256, 1, 1],
            };
            match host.dispatch_timed(&root, &pso, &d, 4) {
                // The host's timing is seconds for all four: ms per dispatch.
                Ok((_, secs)) => {
                    let ms = secs * 1e3 / 4.0;
                    ((256.0 * 0.4 / ms.max(1e-6)) as u32).clamp(64, GROUPS)
                }
                Err(e) => {
                    eprintln!("{name}: {e}; adapter skipped");
                    continue;
                }
            }
        };
        eprintln!(
            "{name}: waves of {min}-{max} lanes; {groups} groups of {THREADS}, GPU ms per dispatch (median of 3 x {REPEAT})"
        );
        eprintln!(
            "| program | group syncs | wave32 program | wave64 split | split / wave32 | wave64 native |"
        );
        eprintln!("|---|---|---|---|---|---|");
        for (label, blocks, alu, cross) in cases {
            let mut times = Vec::new();
            let mut syncs = 0;
            for (wave64, host64) in [(false, false), (true, false), (true, true)] {
                let words = timing_program(THREADS, blocks, alu, cross, wave64);
                let lanes = if host64 { 64 } else { 32 };
                if !(min..=max).contains(&lanes) {
                    times.push(None);
                    continue;
                }
                let (shader, n) = lower_waves(&words, THREADS, 0, wave64, host64, true)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                if wave64 && !host64 {
                    syncs = n;
                }
                let (root, pso) = host
                    .compute_pipeline(&shader)
                    .unwrap_or_else(|e| panic!("{name}, {label}: pipeline: {e}"));
                let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                let d = Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(THREADS)],
                    samplers: &[],
                    groups: [groups, 1, 1],
                };
                let mut ms = Vec::new();
                for _ in 0..3 {
                    let (out, t) = match host.dispatch_timed(&root, &pso, &d, REPEAT) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("{name}, {label}: {e}; adapter skipped");
                            continue 'adapters;
                        }
                    };
                    let want =
                        timing_model(THREADS, blocks, alu, cross, if wave64 { 64 } else { 32 });
                    assert_eq!(
                        out[0], want,
                        "{name}, {label}, wave64 {wave64}, host {lanes} lanes"
                    );
                    // Seconds for all REPEAT dispatches: ms per dispatch.
                    ms.push(t * 1e3 / f64::from(REPEAT));
                }
                ms.sort_by(f64::total_cmp);
                times.push(Some(ms[1]));
            }
            let cell = |t: Option<f64>| t.map_or("n/a".into(), |t| format!("{t:.3}"));
            let ratio = match (times[0], times[1]) {
                (Some(a), Some(b)) => format!("{:.2}x", b / a),
                _ => "n/a".into(),
            };
            eprintln!(
                "| {label} | {syncs} | {} | {} | {ratio} | {} |",
                cell(times[0]),
                cell(times[1]),
                cell(times[2])
            );
        }
    }
}
