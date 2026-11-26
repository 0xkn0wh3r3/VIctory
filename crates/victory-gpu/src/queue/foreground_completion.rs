//! Observe which completion path releases a real graphics → compute dependency.
//! Only the timer and completion worker take this read lock. The test holds
//! the write lock while the submitting thread drains the graphics record.
use super::{Why, account, held};
use crate::host::Host;
use crate::object::This;
use std::sync::{Arc, OnceLock, RwLock, RwLockReadGuard, mpsc};
use std::thread::{self, ThreadId};
use std::time::Duration;

static BACKGROUND: RwLock<()> = RwLock::new(());

pub(crate) fn background_turn() -> RwLockReadGuard<'static, ()> {
    BACKGROUND.read().unwrap_or_else(|e| e.into_inner())
}

pub(crate) struct Probe {
    finished: Arc<OnceLock<ThreadId>>,
    released: mpsc::Receiver<ThreadId>,
}

impl Probe {
    pub(crate) fn attach(graphics: This, compute: This) -> Self {
        let finished = Arc::new(OnceLock::new());
        let completion = finished.clone();
        crate::list::segments(graphics)
            .last()
            .unwrap()
            .defer_after(move || {
                completion.set(thread::current().id()).unwrap();
            });
        let (send, released) = mpsc::channel();
        // The second compute segment is the one waiting for graphics. A
        // host-only observation does not give this list any title writes.
        let segments = crate::list::segments(compute);
        assert_eq!(segments.len(), 2);
        segments[1].defer_host(move || {
            send.send(thread::current().id()).unwrap();
        });
        Self { finished, released }
    }

    pub(crate) fn complete(self, host: &Host, submit_graphics: impl FnOnce()) -> bool {
        let background = BACKGROUND.write().unwrap_or_else(|e| e.into_inner());
        let first = account().released.len();
        submit_graphics();
        assert!(self.finished.get().is_none(), "graphics is still in flight");
        assert!(self.released.try_recv().is_err(), "compute is still held");

        host.complete_all();
        let foreground = thread::current().id();
        assert_eq!(self.finished.get(), Some(&foreground));
        let observed = self.released.try_recv().ok();
        drop(background);

        // A parent that misses the foreground pump still finishes through
        // the normal timer. This channel waits without polling a title fence.
        let submitter = observed.unwrap_or_else(|| {
            self.released
                .recv_timeout(Duration::from_secs(5))
                .expect("compute released")
        });
        // If the timer won, its deferred work sent the message before pump
        // recorded the reason. Taking its lock waits for accounting to finish.
        let _idle = held();
        let releases = account();
        let [(2, _, why)] = &releases.released[first..] else {
            panic!(
                "expected one released compute segment, saw {:?}",
                &releases.released[first..]
            );
        };
        // SAFETY: read the LUID of the device that owns these command lists.
        let luid = unsafe { host.device.GetAdapterLuid() };
        eprintln!(
            "adapter LUID={:08x}:{:08x}, graphics completed on {foreground:?}, compute submitted on {submitter:?}, release={why:?}, before workers resumed={}",
            luid.HighPart,
            luid.LowPart,
            observed.is_some()
        );
        observed == Some(foreground) && matches!(why, Why::Completed)
    }
}
