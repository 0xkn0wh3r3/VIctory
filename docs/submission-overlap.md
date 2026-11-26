# Plan: letting the CPU and the GPU overlap

Status: proposed. Nothing here is implemented yet.

Today every submission is waited for: `Host::execute` records, submits,
then blocks until the GPU has finished, and only then copies back what
shaders wrote and returns to the title (`crates/victory-gpu/src/host.rs`,
the shortcut at the top: "every submission is waited for, so CPU and GPU
never overlap; overlap them with per-submission fences once more than
clears run"). A frame's time is therefore the sum of the title's own CPU
work, VIctory's work around each submission, and the GPU's work, where a
console overlaps the first and the last.

This page says what depends on that wait, the change proposed, in what
order, how it is checked, and when it is not worth doing.

## What it costs today

Measured with `cargo demo` (the conformance title's GPU frame; CrossOver
on an M5 Max, 720p, no overlay), at `--scale 128`: 129 submissions a
frame, ~50 ms a frame, of which ~32 ms are spent waiting for the GPU after
each submission (0.25 ms each; mostly the round trip, the work itself is
small), against ~10 ms recording 8,000 draws and ~4 ms of refresh and
copy-back. Under Wine and D3DMetal the round trip is longer than on
Windows, and the demo waits for every submission itself too (below), so
these numbers say where the time goes in this setup, not in the game.

**First, measure the game.** If a world frame is GPU-bound (the slow-frame
report's `gpu_ms` near the frame time, `crate::frame_budget`), overlapping
the CPU with it saves at most the CPU's share; the GPU's own time (the
snapshot path, PCIe traffic; see `docs/gpu-memory.md`) comes first. The
report's `refresh_ms`, `copy_back_ms` and `outside_ms` against `gpu_ms`
decide it. This plan is worth doing when the CPU parts are a large share.

## What depends on the wait

Everything after the wait in `Host::execute` assumes the GPU is done, and
much of what runs before the next submission assumes the last one is
done. Grouped by what breaks if a submission returns early:

1. **What the title can observe.** A title fence's value advances only
   after `host.execute` returns (`queue.rs` `pump`: `host.execute`, then
   `fence::set`/`let_go`), so after copy-back. Queries are faked as
   bottom-of-pipe CPU writes after the run (`list.rs` ResolveQueryData,
   `host.rs` bottom-of-pipe writes); memory-value waits in `hold.rs` read
   them. Copy-back of shader writes into title memory happens after the
   wait. Any of these done before the GPU finishes shows the title wrong
   data.
2. **Lists reused while running.** After `execute`, `list::recycle` resets
   every host list no title list holds (`queue.rs` `pump`, `list.rs`
   `recycle`); a title list's Reset replaces only a host list a queue
   still holds (`restart`, `strong_count > 1`). Returning early would reset
   lists the GPU is still running.
3. **Shared buffers rewritten each submission.** The written-pages and
   read-mark bitmaps (`shadow.rs` `bitmap`, two for all submissions), the
   title-memory page tables (`memory.rs`, rewritten in `prepare`, "only
   while no submission runs"), texture staging chunks (`texture.rs`,
   rewound each submission), the bindless upload list (`bindless.rs`, one),
   the prelude allocator (`host.rs`), the present copy list, the overlay
   timer, and the `copy_now` copier.
4. **Host copies of title memory.** Refresh writes a copy with the CPU
   before the submission that reads it (`Shadow::update`); copy-back
   compares a copy with its snapshot (`copy_back_or_not`) and frees the
   snapshot when nothing is pending ("one submission runs at a time");
   busy ranges are refreshed because "a submission waits for the one
   before it" (`shadow.rs`, `update_tracked`). Mirrored copies in video
   memory are read back with `copy_now`, which waits for everything
   queued before it.
5. **Lifetimes.** Resources retired from descriptor slots are released
   after each `execute` (`release_retired`); placements reuse ranges once
   nothing holds them; deferred steps keep their `Arc`s until the list is
   reset.
6. **Present.** It pumps held work, then submits its own copy list with
   `execute` and waits (`host.rs` present path); `ResizeBuffers` relies on
   every submission having been waited for.
7. **Diagnostics.** Per-list timing submits each list on its own and reads
   its queries at once; capture buffers are shared per present.

A blocking fence wait from the title (SetEventOnCompletion with no event)
is not implemented: today the value is always reached by then, so it
fails instead of hanging (`fence.rs`).

## The change: one submission in flight

Not "many submissions in flight": nearly every item above holds as long
as a submission is **finished before the next one is prepared**. So:

- `Host::execute` splits into **submit** (everything up to
  ExecuteCommandLists and the host Signal) and **complete** (the wait and
  everything after it), with the state between them (the runs and their
  `Arc`s, snapshots, ranges, the batch, retired resources) kept in one
  `InFlight` record under the `submitted` lock.
- **Submit returns** once the host Signal is queued. The title's
  ExecuteCommandLists returns with it, so the title's threads record and
  simulate the next lists while the GPU runs these.
- **Complete runs** before anything that needs the GPU done: at the start
  of the next submission (before its prepare, so items 3 and 4 hold as
  today), before present, before `copy_now` from outside a submission, and
  on a **completion thread** that waits on the host fence for the record
  in flight, so a title that waits on a fence without submitting again
  still sees it reached.
- **Title fences** signalled after a piece (`hold.rs` Signal items) are set
  by complete, not by the pump: the pump turns a Signal behind a piece in
  flight into a signal owed by that record. Waiters, memory-value waits and
  held queue Waits then behave as today, only later.
- **Lists** are recycled by complete (item 2): the record holds the
  segments' `Arc`s until then, so `restart` already sees them as held.
- **Retired resources** are released by complete (item 5).
- **A blocking fence wait** (null event) waits for the completion thread
  instead of failing.
- **Fallbacks:** `VICTORY_GPU_OVERLAP=0` keeps today's behaviour (submit
  and complete back to back), for A/B runs and as an escape hatch;
  diagnostics timing modes (`VICTORY_GPU_TIMING`, frame timing) keep it
  too, since they read their queries right after the run.

What this buys: the title's CPU time between its submissions (recording,
game logic, other threads) overlaps the GPU's work on the last one. What
it does not: VIctory's own prepare (refresh) and copy-back still wait for
the previous submission, and present still waits for the frame. Those are
phase 3.

### Lock order

The deadlock risk is the main one. Today `HELD` is taken, then
`submitted` inside `host.execute`, then fence waiters inside the pump's
signal. With complete on its own thread: complete takes `submitted`,
finishes the record, drops `submitted`, then sets the owed fences and
pumps `HELD` if free (`pump_held_if_free`), never the other way round.
The present path takes the presenter lock after completing, as it now
takes it around its own `execute`.

## Phases

1. **Make the demo a fair test** (no VIctory change beyond the fence wait):
   the conformance title submits its stress-scale chunks without waiting
   after each (a ring of allocators and lists, one fence value each, waits
   only before reusing one and before reading back), as a game does; and a
   blocking fence wait in VIctory that waits instead of failing (it can
   keep returning at once while every submission is still waited for).
   Measure again: this is the baseline.
2. **One submission in flight** (the change above), behind
   `VICTORY_GPU_OVERLAP`, default off until it has run on Windows, then on.
3. **Only if phase 2's numbers ask for it:** prepare the next submission
   while one runs (per-submission write and read-mark bitmaps, page tables
   and staging per submission or in rings, refresh of copies a running
   submission writes through GPU copies queued after it instead of CPU
   writes, snapshots per submission), and present without waiting for the
   frame. Each of items 3 and 4 above is a change of its own there; this
   phase is a separate plan.

## How it is checked

- **Tests that fail without it**, with a test switch that holds the
  completion thread so a submission stays in flight on purpose:
  - a title fence signalled after a list is not reached, and the list's
    shader writes are not in title memory, until the submission completes;
    both are once it does;
  - a query's result (a bottom-of-pipe write) is not visible before;
  - a title list reset while in flight does not reset its host list until
    complete (the GPU still runs it);
  - a title write to title memory between two submissions is what the
    second one reads (refresh after complete);
  - a blocking fence wait returns once the value is reached, and fails
    only on device removal;
  - `VICTORY_GPU_OVERLAP=0` behaves as today.
- **The acceptance suite**: `cargo conformance` (and its GPU frame against
  WARP on Windows) with overlap on and off.
- **victory-gpu's tests** on Windows, and under Wine with results compared
  to the base, with overlap on and off.
- **The demo** at scales 1, 16, 64 and 128 before and after each phase:
  frame time, VIctory's time a frame by call, and the submission phases
  (`victory_gpu::submission_totals`); add the time `complete` waits, so the
  overlap gained is measured, not inferred.
- **The game on Windows**: the slow-frame report before and after, with
  overlap on and off, over the same scene.

## Risks

- **Deadlocks** between `HELD`, `submitted`, fence waiters and the
  presenter (lock order above); a test of a title waiting on a fence from
  another thread while a submission is held in flight.
- **Device removal and GPU faults** found at complete instead of at once:
  the report moves with it; a fault in a submission must still be
  attributed to it (`tdr.rs`, the device-removed check).
- **Lifetimes**: anything a submission's GPU work touches that is not held
  by its record (resources freed by the title right after
  ExecuteCommandLists, placements reused). The record must hold what the
  deferred steps hold today until complete.
- **Diagnostics** that read results at once must force complete first.
- **Gains smaller than hoped**: if the game is GPU-bound, or its render
  threads already record while the GPU runs, phase 2 saves little; the
  game measurement above decides before phase 3.
