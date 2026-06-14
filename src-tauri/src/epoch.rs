//! Epoch-based operator lock (Phase 3, Bullet 3.2).
//!
//! Operator manual commands must always win over voice detections. When the
//! operator acts (picks/projects/navigates a verse), the epoch is bumped and a
//! short lock window opens; any voice detection stamped with an older epoch, or
//! arriving inside the window, is discarded before it can override the
//! operator's choice.
//!
//! Managed as its own Tauri state (not inside `AppState`'s mutex) so both the
//! command handlers and the detection consumer get lock-free reads — matching
//! the companion's "shared via Arc clone, lock-free" intent.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long an operator manual action suppresses voice detections.
pub const OPERATOR_LOCK_MS: u64 = 500;

/// Shared operator-vs-voice arbiter.
#[derive(Default)]
pub struct EpochLock {
    counter: AtomicU64,
    lock_until: Mutex<Option<Instant>>,
}

impl EpochLock {
    /// Current epoch (lock-free read).
    pub fn current(&self) -> u64 {
        self.counter.load(Ordering::SeqCst)
    }

    /// Register an operator manual action: bump the epoch (invalidating any
    /// in-flight voice detection) and open the lock window. Returns the new epoch.
    pub fn acquire(&self) -> u64 {
        let new_epoch = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        if let Ok(mut until) = self.lock_until.lock() {
            *until = Some(Instant::now() + Duration::from_millis(OPERATOR_LOCK_MS));
        }
        new_epoch
    }

    /// Whether a detection stamped at `epoch_at_detection` must be discarded —
    /// its epoch is stale, OR we are still inside the operator lock window.
    pub fn is_locked_out(&self, epoch_at_detection: u64) -> bool {
        let lock_until = self.lock_until.lock().ok().and_then(|u| *u);
        locked_out(
            epoch_at_detection,
            self.current(),
            lock_until,
            Instant::now(),
        )
    }
}

/// Pure decision behind [`EpochLock::is_locked_out`], split out for testing.
fn locked_out(
    epoch_at_detection: u64,
    current_epoch: u64,
    lock_until: Option<Instant>,
    now: Instant,
) -> bool {
    epoch_at_detection < current_epoch || lock_until.map_or(false, |t| now < t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_epoch_is_locked_out() {
        assert!(locked_out(5, 6, None, Instant::now()));
    }

    #[test]
    fn current_epoch_without_window_passes() {
        assert!(!locked_out(6, 6, None, Instant::now()));
    }

    #[test]
    fn inside_lock_window_is_locked_out() {
        let now = Instant::now();
        assert!(locked_out(6, 6, Some(now + Duration::from_millis(100)), now));
    }

    #[test]
    fn after_lock_window_passes() {
        let now = Instant::now();
        // lock_until equals `now`, checked 1ms later → window has elapsed.
        assert!(!locked_out(6, 6, Some(now), now + Duration::from_millis(1)));
    }

    #[test]
    fn acquire_bumps_epoch_and_locks_out_prior_stamp() {
        let lock = EpochLock::default();
        assert_eq!(lock.current(), 0);
        let before = lock.current();
        let new = lock.acquire();
        assert_eq!(new, before + 1);
        // The pre-acquire stamp is stale → locked out.
        assert!(lock.is_locked_out(before));
        // Even the new epoch is inside the 500ms window → still locked out.
        assert!(lock.is_locked_out(new));
    }
}
