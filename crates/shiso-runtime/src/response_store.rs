//! The transcripts a stored response is resumed from, and the bodies a client
//! can fetch again.
//!
//! # Why it holds the conversation and not only the body
//!
//! The Responses surface lets a caller continue a conversation by naming the
//! response it is continuing rather than by resending the history. That only
//! works if the server kept the history, and the history it has to keep is the
//! neutral one: a stored wire body would have to be parsed back into the IR by
//! the surface that wrote it, and the parse would be a second, weaker copy of
//! the mapping that produced it. So an entry carries both - the IR transcript,
//! which is what a resume prepends, and the encoded body, which is what a
//! re-fetch returns verbatim.
//!
//! # Why memory only
//!
//! A conversation that survives a restart is a different product: it needs a
//! retention policy, a privacy story, an on-disk format with a version, and a
//! size budget an operator can see. None of those are decided here, and a
//! filesystem tier added without them is a server that quietly keeps everyone's
//! prompts forever. The bound and the expiry below are what make the in-memory
//! answer honest instead of a leak.
//!
//! # Why the kind is part of every lookup
//!
//! Two surfaces mint identifiers into this one store. Asking for a chat
//! completion by a `resp_` identifier is a client mistake, and answering it with
//! the other surface's body would hand a Responses transcript to a chat client.
//! A mismatch is therefore a miss, and - deliberately - not an eviction: the
//! entry the other surface owns is still perfectly valid.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::Value;
use shiso_ir::Message;

/// Entries the store holds before the oldest is dropped.
pub const DEFAULT_MAX_ENTRIES: usize = 10_000;

/// How long an entry stays fetchable.
// `Duration::from_days`, which this lint asks for, is unstable on the pinned
// 1.98.1 toolchain and is not a `const fn` there at all. The expectation goes
// when the constructor stabilises, and the lint will say so by then failing to
// fire.
#[expect(
    clippy::duration_suboptimal_units,
    reason = "Duration::from_days is unstable on the pinned toolchain"
)]
pub const DEFAULT_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// The environment knob for the entry bound.
const MAX_ENTRIES_ENV: &str = "SHISO_STORE_MAX_ENTRIES";

/// The environment knob for the expiry, in seconds.
const TTL_SECONDS_ENV: &str = "SHISO_STORE_TTL_SECONDS";

/// Which surface minted an identifier.
///
/// The prefix is derived from the kind rather than stored beside it, so an
/// entry whose identifier says one thing and whose kind says another cannot be
/// built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoredKind {
    /// The Responses surface. Identifiers begin `resp_`.
    Response,
    /// The chat-completions surface. Identifiers begin `chatcmpl-`.
    ChatCompletion,
}

impl StoredKind {
    /// The prefix this surface's identifiers carry.
    #[must_use]
    pub const fn id_prefix(self) -> &'static str {
        match self {
            Self::Response => "resp_",
            Self::ChatCompletion => "chatcmpl-",
        }
    }

    /// Mints an identifier for this surface.
    #[must_use]
    pub fn mint(self) -> String {
        format!("{}{}", self.id_prefix(), crate::ids::uuid_v4())
    }
}

/// One stored turn.
#[derive(Debug, Clone)]
pub struct StoredEntry {
    /// The identifier the client will quote back, prefix included.
    pub id: String,
    /// Which surface minted it.
    pub kind: StoredKind,
    /// The model that answered.
    pub model: String,
    /// Seconds since the epoch when the turn completed.
    pub created_at: u64,
    /// The whole conversation, generated turn included, ready to be prepended
    /// to whatever the caller sends next.
    pub messages: Vec<Message>,
    /// The body this response was answered with, to be returned verbatim on a
    /// re-fetch.
    pub body: Value,
}

/// What a lookup found.
///
/// Deliberately not the entry: the identifier and the kind are what the caller
/// already had, and returning them again invites code that reads them from here
/// rather than from the request it is answering.
#[derive(Debug, Clone)]
pub struct GetResult {
    /// The model that answered.
    pub model: String,
    /// Seconds since the epoch when the turn completed.
    pub created_at: u64,
    /// The conversation, generated turn included.
    pub messages: Vec<Message>,
    /// The body the response was answered with.
    pub body: Value,
}

/// An entry with the moment it stops being fetchable.
#[derive(Debug)]
struct Held {
    entry: StoredEntry,
    expires_at: Instant,
}

/// The map and the insertion order that bounds it.
///
/// Order is a queue of identifiers rather than a timestamp sort: insertion is
/// the only thing that changes it, a read never does, and so the bound cannot be
/// defeated by a client that keeps one entry warm by fetching it.
#[derive(Debug, Default)]
struct Inner {
    map: HashMap<String, Held>,
    order: VecDeque<String>,
}

/// The stored responses of one process.
#[derive(Debug)]
pub struct ResponseStore {
    inner: Mutex<Inner>,
    max_entries: usize,
    ttl: Duration,
}

impl ResponseStore {
    /// A store with the given bound and expiry.
    ///
    /// A zero bound is raised to one: a store that can hold nothing accepts
    /// every insert and answers every lookup with a miss, which reads exactly
    /// like an expiry bug.
    #[must_use]
    pub fn new(max_entries: usize, ttl: Duration) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner::default()),
            max_entries: max_entries.max(1),
            ttl,
        })
    }

    /// A store configured from the environment, falling back to the defaults.
    ///
    /// The knobs are read once, here, rather than at each insert: a bound that
    /// could change between two inserts of the same conversation would make
    /// which turn survives a property of when it happened.
    #[must_use]
    pub fn from_env() -> Arc<Self> {
        let entries = read_env(MAX_ENTRIES_ENV).unwrap_or(DEFAULT_MAX_ENTRIES);
        let seconds = read_env(TTL_SECONDS_ENV).map_or(DEFAULT_TTL, |secs| {
            Duration::from_secs(u64::try_from(secs).unwrap_or(u64::MAX))
        });
        Self::new(entries, seconds)
    }

    /// Stores one turn, evicting the oldest if the bound is reached.
    pub fn insert(&self, entry: StoredEntry) {
        let held = Held {
            expires_at: Instant::now() + self.ttl,
            entry,
        };
        let mut inner = self.inner.lock();
        let id = held.entry.id.clone();
        if inner.map.insert(id.clone(), held).is_none() {
            inner.order.push_back(id);
        }
        while inner.order.len() > self.max_entries {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            inner.map.remove(&oldest);
        }
    }

    /// The entry for `id`, when one is stored under that surface.
    ///
    /// An expired entry is removed on the way past. Sweeping on read rather than
    /// on a timer is what keeps this type free of a background task, and the
    /// bound above is what stops an entry nobody ever asks for again from living
    /// forever.
    #[must_use]
    pub fn get(&self, id: &str, kind: StoredKind) -> Option<GetResult> {
        let mut inner = self.inner.lock();
        let expired = inner.map.get(id)?.expires_at <= Instant::now();
        if expired {
            inner.map.remove(id);
            inner.order.retain(|queued| queued.as_str() != id);
            return None;
        }
        let held = inner.map.get(id)?;
        // A kind mismatch is a miss and nothing else: the entry belongs to the
        // other surface and is still perfectly good for it.
        if held.entry.kind != kind {
            return None;
        }
        Some(GetResult {
            model: held.entry.model.clone(),
            created_at: held.entry.created_at,
            messages: held.entry.messages.clone(),
            body: held.entry.body.clone(),
        })
    }

    /// Removes one entry, reporting whether it was there.
    pub fn delete(&self, id: &str, kind: StoredKind) -> bool {
        let mut inner = self.inner.lock();
        match inner.map.get(id) {
            Some(held) if held.entry.kind == kind => {}
            _ => return false,
        }
        inner.map.remove(id);
        inner.order.retain(|queued| queued.as_str() != id);
        true
    }

    /// How many entries are held, expired ones included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How long an entry stays fetchable.
    #[must_use]
    pub const fn ttl(&self) -> Duration {
        self.ttl
    }

    /// How many entries this store will hold.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.max_entries
    }
}

/// Reads one numeric knob, ignoring a value that does not parse.
///
/// Ignoring rather than failing: a malformed knob should not stop a server that
/// has a perfectly good default, and the warning is what an operator sees.
fn read_env(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    match raw.parse() {
        Ok(value) => Some(value),
        Err(_unparsable) => {
            tracing::warn!(knob = name, value = raw, "ignoring an unreadable setting");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ResponseStore, StoredEntry, StoredKind};
    use serde_json::json;
    use shiso_ir::Message;
    use std::time::Duration;

    fn entry(id: &str, kind: StoredKind) -> StoredEntry {
        StoredEntry {
            id: id.to_owned(),
            kind,
            model: "a-model".to_owned(),
            created_at: 7,
            messages: vec![Message::user("hi"), Message::assistant("hello")],
            body: json!({"id": id}),
        }
    }

    #[test]
    fn a_stored_turn_comes_back_with_its_transcript_and_its_body() {
        let store = ResponseStore::new(4, Duration::from_secs(60));
        store.insert(entry("resp_1", StoredKind::Response));
        let found = store
            .get("resp_1", StoredKind::Response)
            .expect("the entry is stored");
        assert_eq!(found.messages.len(), 2);
        assert_eq!(found.body["id"], "resp_1");
        assert_eq!(found.created_at, 7);
    }

    /// The two surfaces share one store, and an identifier one of them minted
    /// must not answer the other's question.
    #[test]
    fn an_identifier_from_the_other_surface_is_a_miss_and_survives_the_miss() {
        let store = ResponseStore::new(4, Duration::from_secs(60));
        store.insert(entry("resp_1", StoredKind::Response));
        assert!(store.get("resp_1", StoredKind::ChatCompletion).is_none());
        assert!(
            store.get("resp_1", StoredKind::Response).is_some(),
            "a mismatch is a miss, not an eviction"
        );
    }

    #[test]
    fn an_expired_entry_is_a_miss_and_is_swept_on_the_way_past() {
        let store = ResponseStore::new(4, Duration::from_millis(0));
        store.insert(entry("resp_1", StoredKind::Response));
        assert!(store.get("resp_1", StoredKind::Response).is_none());
        assert_eq!(store.len(), 0, "the sweep is what keeps the map bounded");
    }

    /// The bound is on insertion order, so the oldest turn is the one that goes.
    #[test]
    fn reaching_the_bound_drops_the_oldest_turn_and_keeps_the_newest() {
        let store = ResponseStore::new(2, Duration::from_secs(60));
        store.insert(entry("resp_1", StoredKind::Response));
        store.insert(entry("resp_2", StoredKind::Response));
        store.insert(entry("resp_3", StoredKind::Response));
        assert!(store.get("resp_1", StoredKind::Response).is_none());
        assert!(store.get("resp_2", StoredKind::Response).is_some());
        assert!(store.get("resp_3", StoredKind::Response).is_some());
        assert_eq!(store.len(), 2);
    }

    /// Re-storing an identifier replaces the turn rather than queueing it twice,
    /// which would let one conversation consume the whole bound.
    #[test]
    fn storing_the_same_identifier_twice_holds_one_entry() {
        let store = ResponseStore::new(4, Duration::from_secs(60));
        store.insert(entry("resp_1", StoredKind::Response));
        store.insert(entry("resp_1", StoredKind::Response));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn deleting_names_the_surface_too_and_reports_whether_anything_went() {
        let store = ResponseStore::new(4, Duration::from_secs(60));
        store.insert(entry("resp_1", StoredKind::Response));
        assert!(!store.delete("resp_1", StoredKind::ChatCompletion));
        assert!(store.delete("resp_1", StoredKind::Response));
        assert!(!store.delete("resp_1", StoredKind::Response));
        assert!(store.is_empty());
    }

    /// A store that can hold nothing would accept every insert and answer every
    /// lookup with a miss, which is indistinguishable from an expiry defect.
    #[test]
    fn a_bound_of_zero_still_holds_one_turn() {
        let store = ResponseStore::new(0, Duration::from_secs(60));
        store.insert(entry("resp_1", StoredKind::Response));
        assert!(store.get("resp_1", StoredKind::Response).is_some());
    }

    #[test]
    fn a_minted_identifier_carries_its_surfaces_prefix() {
        assert!(StoredKind::Response.mint().starts_with("resp_"));
        assert!(StoredKind::ChatCompletion.mint().starts_with("chatcmpl-"));
    }
}
