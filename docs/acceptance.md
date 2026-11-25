# Acceptance and benchmark suite (plan and inventory)

VIctory is checked today by unit tests inside each crate and by runs of the
title. This document plans a third layer that needs neither the title nor any
game data: a **conformance title** we write ourselves, run through VIctory
like a real one, that exercises each platform area the way a large
open-world GDK title does, checks the behaviour public documentation states,
and measures the workloads such a title puts on the platform.

## Rules

- **Public sources only.** Microsoft Learn's public GDK reference, public
  Win32/D3D12/DirectStorage documentation, AMD's public RDNA2 ISA guide,
  public conference talks and hardware disclosures. Nothing from game
  binaries, leaked SDKs or NDA documentation.
- **No Microsoft GDK in the suite.** The GDK's license (Microsoft Public Game
  Development Kit License Agreement, Section 3(a)) allows its use only to
  develop and test Windows titles for Windows, and excludes software for
  Xbox Series consoles; Section 4(d) forbids reverse engineering it. The
  conformance title therefore does not link the GDK's libraries or include
  its headers. It calls the runtime the way VIctory already documents
  (`crates/victory-hle/src/modules/xgameruntime.rs`: `QueryApiImpl`, area
  objects by provider and interface id), with declarations written from the
  public reference pages.
- **Asset policy as for the rest of the tree** (`docs/asset-policy.md`):
  nothing proprietary is committed; workloads are generated.

## Oracles

What tells a test the right answer, strongest first:

| Oracle | Covers | Source |
|---|---|---|
| Public reference text (return codes, limits, ordering) | every runtime area | learn.microsoft.com/gaming/gdk/docs/reference |
| Open-source implementations of the same API (MIT) | XTaskQueue / XAsync: `Source/Task` of libHttpClient; GDeflate: DirectStorage's reference codec | github.com/microsoft/libHttpClient, github.com/microsoft/DirectStorage |
| Reference decoders | Opus (libopus), zlib (RFC 1950/1951) | public |
| Host reference renderer | GPU output (WARP, the D3D12 software adapter) | Windows |
| Hardware numbers | throughput and latency targets | Hot Chips 32, Xbox Wire, GDK overview pages |

Areas whose reference pages are restricted (XMem, XAPU, D3D12.X) have no
public oracle beyond their published error codes and the Win32
equivalents; their tests check only properties VIctory itself states and
the cross-area invariants below (memory written by one path reads back
through another).

## Platform areas

`hle` = the VIctory module (`crates/victory-hle/src/modules/`); `unit` =
its `#[test]` count on main at 940d32f.

| Area | hle | unit | Documented behaviour to check (public reference) |
|---|---|---|---|
| Runtime init | xgameruntime | 1 | Initialize S_OK and repeatable per module; E_GAMERUNTIME_* codes (0x89240100…) |
| Task queues, async | xasync | 7 | dispatch modes (Manual, ThreadPool, SerializedThreadPool, Immediate) per port; `XTaskQueueDispatch` true only when it ran a call; Terminate cancels pending callbacks (`canceled=true`), later submits E_ABORT; `XAsyncGetStatus` E_PENDING until done; cancel then E_ABORT; provider op order Begin, DoWork (E_PENDING reschedules), GetResult, Cancel, Cleanup |
| User | xuser | 2 | AddDefaultUserSilently returns the same user each call; gamertag component buffer maxima (16/97/15/101 bytes), E_INSUFFICIENT_BUFFER; handles closed once; `XUserCompare`, local id equality |
| Game saves | xgamesave | 7 | 16 MB per update (E_GS_UPDATE_TOO_BIG 0x80830005), 256 MB quota (E_GS_QUOTA_EXCEEDED 0x80830006), atomic updates, one change per blob per update, container name charset (E_GS_INVALID_CONTAINER_NAME), submit consumes the update |
| Package | xpackage | 1 | identifier length bound; invalid chunk selector E_GAMEPACKAGE_INVALID_SELECTOR; mount, mount path size and path, close |
| Store | xstore | 1 | context, game license query, product query paging (`HasMorePages`) |
| System | xsystem | 4 | device type values (Series S 6, Series X 7); callable before init; console id format and E_INSUFFICIENT_BUFFER below `XSystemConsoleIdBytes`; analytics family "Xbox" |
| Threads | kernel | 3 | `XThreadSetTimeSensitive`/`XThreadIsTimeSensitive`; APIs unsafe on such threads return E_TIME_CRITICAL_THREAD (game saves: E_GS_ASYNC_FUNCTION_REQUIRED) |
| System UI | xgameui | 1 | text entry: length 0 means 1024, console cap 1024; result size then result |
| Networking | xnetworking | 1 | connectivity hint (offline by design: `networkInitialized` set, no connectivity), safe on time-sensitive threads |
| Display | display | 1 | `XDisplayTryEnableHdrMode` result values and defaults (0.01/1000/1000 nits) |
| Errors | xgameruntime | 1 | `XErrorSetOptions` values; callback set and called |
| Speech | speech | 1 | synthesizer create, voices, stream size then data |
| Input | gameinput | 7 | kinds (gamepad 0x40000, keyboard 0x4, mouse 0x8); readings are reference-counted (same pointer: no new input); history walkable for about 0.5 s; disconnect fails current reading |
| DirectStorage (console) | dstorage | 13 | FIFO completion per queue; auto-submit past half capacity; capacity bounds; 16-byte file offsets for zlib and BCPack; 32 MiB destination limit (1 GiB with 64 KiB/2 MiB pages); request reusable after enqueue; cancel by tag completes as success; error record |
| Files | files | 5 | Win32/CRT file semantics against the host itself (native run of the same test as the oracle) |
| Memory | xmem | 10 | (restricted reference) alignment attributes; physical 64 KiB pages mapped more than once read back the same bytes through every alias; 32-bit allocations below 4 GiB |
| Audio | audio, apu, dsp | 7 | (restricted reference) XAPU/XDSP error codes; decoded Opus equals libopus |
| Media | media | 2 | video decode of a generated clip |
| Upscaling | fidelityfx | 3 | FSR passes run and match the PC FSR output within tolerance |

## Workloads (benchmarks)

Sized from public figures ([S]) where they exist, otherwise estimates ([E]).
Each runs through VIctory and, where the API exists on Windows, natively,
and reports the metrics listed.

| Workload | Shape | Pass | Metrics | Basis |
|---|---|---|---|---|
| Streaming | generated 8-16 GiB package of 64-512 KiB chunks, zlib, zlib+BCPack and raw; 4 priority queues; open-loop 50K requests/s; cancels by tag | every byte equals the reference decode; FIFO; cancelled requests succeed | raw GB/s per 250 ms window (console minimum 2.0), decoded GB/s (4.8 at 2:1), submit-thread CPU, p99 latency per priority, bytes/s into title memory | [S] GDK DirectStorage overview, Xbox Wire Velocity Architecture, Hot Chips 32 |
| Title-memory churn | 2,000 map/unmap of 64 KiB physical pages per frame; CPU writes 50-200 MB/frame across a 6 GiB GPU-visible heap at 30 fps | reads through every alias, CPU and GPU, equal the writes | copy/scan ms per frame (target 3) | [S] 64 KiB pages; [E] rates |
| Job system | 14 workers on 7 cores x 2 SMT, 160 fibers, 20K-50K jobs/frame, graphs 6-10 deep | affinity and CPU sets honoured; fiber-local state survives switches; no lost wake-ups | wake latency, 1 ms timer jitter, throughput vs native | [S] 8 Zen 2 cores (7 for titles, weakly sourced); [E] job counts |
| GPU-driven geometry | meshlets of 128 vertices / 256 primitives, amplification culling 50-100M to 5-10M triangles, ExecuteIndirect with GPU counts, 10K-50K records | image equals WARP | GPU ms at 1440p (frame budget 33 ms) | [S] AMD mesh shader guidance; [E] counts |
| Ray tracing | 10K instances, 3-5M triangles; 200-500 BLAS refits and ~50 rebuilds per frame; 1 GI + 1 reflection ray per pixel at half 1440p | hits equal a CPU reference | BVH ms, trace ms | [S] RT GI and reflections on Series X; [E] sizes |
| Wave64 on 32-lane hosts | ballots, readlane, prefix sums, quad ops | bit-exact against the RDNA2 ISA definitions | translation success rate | [S] AMD RDNA2 ISA guide |
| Texture streaming | MinMip feedback, 1,000-4,000 tile maps per frame from BCPack loads | detiled texels equal the reference for each swizzle mode | ms per frame | [S] 64 KiB tiles |
| Audio | 300 Opus streams at 48 kHz, 128-256 XMA voices, 8-12 convolution channels for 10 minutes | no underruns; PCM equals libopus | host CPU % | [S] Series X audio block: 300+ channels |
| Frame pacing | 30 fps presents at 60 Hz and VRR | frame-time deviation under 1 ms, no frame over 50 ms | pad-to-photon latency | [S] 33 ms frames at launch |
| Saves | 16 MB updates with the process killed at random points | all-or-nothing; exact error codes past the limits | save time | [S] GDK game-save limits |
| Pipelines | ~27,000 pipelines created on workers under a 30 fps loop | no frame over 50 ms | creation throughput | README counts |

## Status

`cargo test -p victory-hle --test conformance` builds the conformance title
(`tests/fixtures/conformance`), loads and binds it through VIctory as
`victory run` does, and runs its checks; the TAP report is printed with
`-- --nocapture`. Checks VIctory is known to answer differently from the
reference are listed in the test (`KNOWN`), each with why, and reported
rather than failed; one of them passing fails the test, so the list stays
true. `-- --ignored` runs the workloads.

Covered so far: runtime initialisation, task queues and async calls,
system information, physical-page aliasing, DirectStorage (memory sources)
and game saves: 82 checks, 80 pass. The two divergences found:

| Check | Reference | VIctory |
|---|---|---|
| `XSystemGetConsoleId` on retail | `00000000.00000000.00000000.00000000.00` | 38 digits ending in 1 |
| DirectStorage automatic submission | when an Enqueue fills more than half the capacity | at half the capacity |

Gaps seen while writing them (documented features VIctory's tables leave
unimplemented, not yet tested): DirectStorage cancel by tag and the error
event and record; the documented request limits (32 MiB destination, 16-byte
offsets for compressed sources), whose behaviour when broken the reference
does not state.

The streaming workload so far reads from memory only (no file source),
with synthetic data that compresses about 3.5:1; on an RTX 5070 Ti machine
(Ryzen 9 7900X), release build: raw 18-25 GB/s, zlib 12 GB/s out (3.5 GB/s
in), far above the console's 2.0 GB/s raw minimum and 4.8 GB/s decompressed,
because no storage is read. File sources and BCPack come next.

## Build order

1. The conformance title and its harness: a test registry, TAP output, a
   `victory run` invocation per area, and the same tests compiled natively
   where an oracle runs on Windows. First areas: system, task queues and
   async (with libHttpClient's implementation as the second oracle), game
   saves (limits and atomicity), DirectStorage ordering and limits.
2. Streaming and title-memory workloads, the stated load bottleneck.
3. The GPU frame workloads against WARP, and wave64 correctness.
4. Audio, frame pacing, input.

## Sources

- GDK reference and error codes: https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/
- DirectStorage on Xbox (overview, limits): https://learn.microsoft.com/gaming/gdk/_content/gc/system/overviews/directstorage/directstorage-overview
- Game saves: https://learn.microsoft.com/en-us/gaming/gdk/_content/gc/system/overviews/game-save/game-saves
- libHttpClient (XTaskQueue/XAsync implementation, MIT): https://github.com/microsoft/libHttpClient
- DirectStorage reference codecs (MIT): https://github.com/microsoft/DirectStorage
- AMD RDNA2 ISA: https://www.amd.com/system/files/TechDocs/rdna2-shader-instruction-set-architecture.pdf
- Xbox Series X SoC, Hot Chips 32: https://hc32.hotchips.org/assets/program/conference/day1/HotChips2020_GPU_Microsoft_Jeff_Andrews_v2.pdf
- Xbox Velocity Architecture: https://news.xbox.com/en-us/2020/07/14/a-closer-look-at-xbox-velocity-architecture/
- Mesh shaders (AMD guidance): https://gpuopen.com/learn/mesh_shaders/mesh_shaders-optimization_and_best_practices/
