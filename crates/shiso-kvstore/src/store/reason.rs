//! Why an entry was written.

/// Why an entry was written, stored in its header.
///
/// The reason is a fact about the moment of the write, and it is what an
/// operator reading the store's counters needs to tell "turns are being
/// finished and filed" from "memory had to give this up".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Reason {
    /// Read from a file whose reason byte was outside the known range.
    Unknown = 0,
    /// A prompt this process prefilled from position zero.
    Cold = 1,
    /// A conversation that grew: a resumed prompt's end, or a finished turn.
    Continued = 2,
    /// Page pressure was about to drop the span.
    Evict = 3,
    /// The process is shutting down.
    Shutdown = 4,
    /// A boundary shared by many sessions: a message end, or the point where a
    /// prompt branched off a path the cache already held.
    AgentSystem = 5,
    /// A session's own prefix, which continues a shared one.
    AgentSession = 6,
}

impl Reason {
    /// Every reason, in byte order, for a counter table indexed by reason.
    pub const ALL: [Self; 7] = [
        Self::Unknown,
        Self::Cold,
        Self::Continued,
        Self::Evict,
        Self::Shutdown,
        Self::AgentSystem,
        Self::AgentSession,
    ];

    /// Maps a header byte back to a reason, folding anything unknown to
    /// [`Reason::Unknown`].
    #[must_use]
    pub const fn from_byte(byte: u8) -> Self {
        match byte {
            1 => Self::Cold,
            2 => Self::Continued,
            3 => Self::Evict,
            4 => Self::Shutdown,
            5 => Self::AgentSystem,
            6 => Self::AgentSession,
            _ => Self::Unknown,
        }
    }

    /// The header byte for this reason.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// The lowercase name a metrics label or a log line uses.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Cold => "cold",
            Self::Continued => "continued",
            Self::Evict => "evict",
            Self::Shutdown => "shutdown",
            Self::AgentSystem => "agent_system",
            Self::AgentSession => "agent_session",
        }
    }
}
