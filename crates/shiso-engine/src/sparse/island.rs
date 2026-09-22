//! Islands: the two halves a layer is captured and replayed in.
//!
//! A layer launches on the order of thirty kernels, and at one token those cost
//! more in launch overhead than they do on the device. So each layer is split
//! into two runs that can be captured as a device graph and replayed: the first
//! from the token mixer's stream mix to its combine, the second the same for the
//! routed feed-forward. Everything about that split is bookkeeping, and all of
//! the bookkeeping is here.
//!
//! # What identifies a captured island
//!
//! Four things, and each of them is a way two captures can differ while looking
//! alike:
//!
//! - the layer and which of its two islands it is;
//! - the row count, because the launch sequence bakes it in - a one-token decode
//!   and a two-row speculative verify are different captures;
//! - whether the indexer's selection was live, because that changes which
//!   attention kernel the island launches;
//! - which execution context's own buffers the island writes. The activation
//!   scratch is shared between contexts, so the only thing that tells two of
//!   them apart is the state they own alone - and a captured launch bakes that
//!   in like any other address. Without it, one session would replay launches
//!   that write another session's recurrent state. An island over a decode
//!   batch writes several sessions' buffers and is therefore not captured at
//!   all; [`IslandContext`] says why.
//!
//! # The chain, and where it breaks
//!
//! The combine that closes an island also normalizes the residual for whatever
//! runs next, so an island usually reads an input its predecessor already
//! normed. That chain breaks in exactly three places, and [`chain`] is the one
//! place those three are written down.

use shiso_core::backend::{GraphBackend, GraphKey};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::LayerIdx;

use crate::sparse::hyper::{ChainLink, MixInput};
use crate::sparse::shape::SparseShape;

/// Which half of a layer an island is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Island {
    /// The token mixer's half: its stream mix, the mixer, and the combine.
    Mixer,
    /// The routed feed-forward's half: its stream mix, the experts, and the
    /// combine.
    Experts,
}

impl Island {
    /// The two halves in the order a layer runs them.
    pub const ORDER: [Self; 2] = [Self::Mixer, Self::Experts];

    /// Which half this is, as the number a key packs.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::Mixer => 0,
            Self::Experts => 1,
        }
    }
}

/// The amount a variant is raised by when the indexer's selection is live.
///
/// A flag rather than a field, folded into the row count, because the two
/// together are what a capture's launch sequence depends on and neither alone
/// identifies it. The offset is far above any row count a graph is captured at,
/// so the two halves stay readable.
pub const SPARSE_VARIANT_OFFSET: u32 = 64;

/// Whose own buffers an island writes.
///
/// # Why a batch has no slot rather than a slot of its own
///
/// A capture bakes in the addresses its launches were encoded with. For a pass
/// over one session those are that session's, and the handle names them. For a
/// pass over a decode batch they are the page tables of *every* session in the
/// batch, in the exact row order - and the key has twenty-four bits for a
/// context, which is not enough to name an ordered set of eight
/// sixty-four-bit handles. Hashing one into it would alias two different
/// batches onto one slot, and the symptom of that is the failure the batched
/// kernels exist to close: a replay that reads another sequence's cache and
/// decodes it to fluent text.
///
/// The per-row pointer arrays do not rescue it. They sit at one address and are
/// refilled each step, so a launch that reads *only* them would replay
/// correctly - but the pooled key pass has no batched sibling and binds each
/// row's page table directly, so a batched island holds real per-session
/// addresses either way.
///
/// So [`IslandContext::Batch`] carries nothing, and [`IslandKey::pack`] answers
/// `None` for it. A batched island is encoded every step.
///
/// Deliberately NOT `#[non_exhaustive]`: the planner builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IslandContext {
    /// One session's pass, filed under its handle.
    Session(u32),
    /// A pass over more than one session, which is never captured.
    Batch,
}

/// What tells two captured islands apart.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IslandKey {
    /// The layer.
    pub layer: LayerIdx,
    /// Which of its two islands.
    pub island: Island,
    /// Rows the island was captured at.
    pub rows: u32,
    /// Whether the indexer's selection was live.
    pub selected: bool,
    /// The execution context whose own state the island writes.
    pub context: IslandContext,
}

/// Bits the layer index occupies in a packed key.
const LAYER_BITS: u32 = 16;
/// Bits the island ordinal occupies.
const ISLAND_BITS: u32 = 8;
/// Bits the variant occupies.
const VARIANT_BITS: u32 = 16;
/// Bits the execution context occupies.
const CONTEXT_BITS: u32 = 24;

const _: () = assert!(LAYER_BITS + ISLAND_BITS + VARIANT_BITS + CONTEXT_BITS == 64);

impl IslandKey {
    /// The row count and the selection mode as one number.
    #[must_use]
    pub const fn variant(&self) -> u32 {
        if self.selected {
            self.rows.saturating_add(SPARSE_VARIANT_OFFSET)
        } else {
            self.rows
        }
    }

    /// Packs the four fields into the slot number a device graph is filed
    /// under.
    ///
    /// `None` is an answer rather than a failure: it means this island has no
    /// slot at all, which is what [`IslandContext::Batch`] means, and the
    /// caller encodes the island instead of looking for a capture of it.
    ///
    /// Every field is range-checked rather than masked. A masked field would
    /// alias two different islands onto one slot, and the symptom of that is a
    /// replay that runs the right kernels over the wrong session's state -
    /// which produces fluent output from numbers nothing computed.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] naming the field that does not fit.
    pub fn pack(&self) -> CoreResult<Option<GraphKey>> {
        let IslandContext::Session(context) = self.context else {
            return Ok(None);
        };
        let fields = [
            ("layer", u64::from(self.layer.get()), LAYER_BITS),
            ("island", u64::from(self.island.ordinal()), ISLAND_BITS),
            ("variant", u64::from(self.variant()), VARIANT_BITS),
            ("context", u64::from(context), CONTEXT_BITS),
        ];
        let mut packed = 0u64;
        let mut shift = 0u32;
        for (name, value, bits) in fields {
            if value >= 1u64 << bits {
                return Err(CoreError::InvalidValue {
                    key: "graph key",
                    value: format!("{name} {value} does not fit {bits} bits"),
                });
            }
            packed |= value << shift;
            shift += bits;
        }
        Ok(Some(GraphKey::new(packed)))
    }
}

/// Where the normalization chain is whole and where it is broken, for one
/// layer.
///
/// Deliberately NOT `#[non_exhaustive]`: a test builds one to compare against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerChain {
    /// Whether the token mixer's stream mix has to normalize its own input.
    pub mixer_input: MixInput,
    /// Whether the routed feed-forward's stream mix has to.
    pub experts_input: MixInput,
    /// Whether the token mixer's combine leaves the next mix's norm behind.
    pub mixer_next: ChainLink,
    /// Whether the routed feed-forward's combine does.
    pub experts_next: ChainLink,
}

/// The chain for one layer.
///
/// Three breaks, and they are the whole rule. The first island of a pass has
/// nothing before it. An island whose n-gram block reads and rewrites the
/// residual between the previous combine and this mix cannot use what that
/// combine left, because the residual it left is not the residual the mix now
/// reads. And the last layer's second island feeds either the collapsing mix -
/// which normalizes its own input - or, for a chunk that produces no logits,
/// nothing at all.
#[must_use]
pub fn chain(shape: &SparseShape, il: LayerIdx) -> LayerChain {
    let first = il.get() == 0;
    let own_ngram = shape.has_ple(il);
    let next = il.next();
    let next_exists = next.is_some_and(|next| next.get() < shape.n_layers());
    let next_ngram = next.is_some_and(|next| shape.has_ple(next));
    LayerChain {
        mixer_input: if first || own_ngram {
            MixInput::Raw
        } else {
            MixInput::AlreadyNormed
        },
        // The token mixer's combine always feeds this layer's own feed-forward
        // mix, which is the one place in the pass where nothing can come
        // between them.
        experts_input: MixInput::AlreadyNormed,
        mixer_next: ChainLink::Chained,
        experts_next: if next_exists && !next_ngram {
            ChainLink::Chained
        } else {
            ChainLink::Broken
        },
    }
}

/// The chain for the draft head's only layer.
///
/// The same answer [`chain`] gives a one-layer model, and written out rather
/// than derived from a fabricated shape: the head's layer is both the first -
/// its residual comes from the init-add rather than from a previous combine, so
/// its token mixer normalizes its own input - and the last, because what
/// follows it is the collapsing mix, which normalizes its own.
#[must_use]
pub const fn draft_chain() -> LayerChain {
    LayerChain {
        mixer_input: MixInput::Raw,
        experts_input: MixInput::AlreadyNormed,
        mixer_next: ChainLink::Chained,
        experts_next: ChainLink::Broken,
    }
}

/// When a pass may be captured and replayed at all.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphPolicy {
    /// Speculative draft depth, or zero for plain decode.
    pub spec_depth: u32,
    /// Whether the verify row counts are captured too.
    ///
    /// Off by default: a verify runs one row plus whatever the drafter
    /// produced, so each depth is a capture of its own and the set only pays
    /// once it has run long enough to trust.
    pub capture_verify: bool,
}

impl GraphPolicy {
    /// Whether a pass of `n_tok` rows is capturable.
    ///
    /// Capture needs a fixed launch sequence, which a prefill's varying row
    /// count breaks. One row always qualifies; a speculative verify qualifies
    /// only when the caller asked for it.
    #[must_use]
    pub const fn capturable(&self, n_tok: u32) -> bool {
        n_tok == 1 || (self.spec_depth != 0 && n_tok <= 1 + self.spec_depth && self.capture_verify)
    }
}

/// What actually happened when an island ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IslandRun {
    /// A captured graph was replayed; nothing was encoded.
    Replayed,
    /// The island was encoded into a capture, which was committed and then
    /// launched once.
    Captured,
    /// The island was encoded directly.
    Eager,
}

/// Runs one island, replaying its captured graph when there is one.
///
/// A capture records launches rather than running them, so the capturing step
/// ends with a replay of what it just recorded: an island that was captured and
/// not launched has not happened, and nothing downstream can tell.
///
/// The protocol is three-state and the third state is the one that matters: a
/// capture that fails for any reason - the device refuses to enter capture mode,
/// the encoding fails inside it, the capture cannot be instantiated - falls back
/// to encoding the island eagerly. An island that was skipped because its
/// capture failed would leave a layer half-run, and the symptom of that is not a
/// failure but a wrong answer.
///
/// # Errors
///
/// Whatever `encode` reports when the eager fallback fails too, and whatever the
/// device reports for a replay that was found but could not run.
pub fn run(
    graph: Option<&dyn GraphBackend>,
    key: Option<GraphKey>,
    mut encode: impl FnMut() -> CoreResult<()>,
) -> CoreResult<IslandRun> {
    // An island with no slot is one nothing may be filed under - a batched
    // island, whose captured pointers would belong to the exact ordered set of
    // sessions it was encoded for. There is nothing to replay and nothing to
    // capture into, so it is encoded, every step.
    let (Some(graph), Some(key)) = (graph, key) else {
        encode()?;
        return Ok(IslandRun::Eager);
    };
    if graph.replay(key)? {
        return Ok(IslandRun::Replayed);
    }
    let Ok(capture) = graph.capture(key) else {
        encode()?;
        return Ok(IslandRun::Eager);
    };
    if encode().is_err() {
        // Dropping the guard leaves capture mode; the island still has to run,
        // so it is encoded again outside it. A second failure is the caller's
        // to see.
        drop(capture);
        encode()?;
        return Ok(IslandRun::Eager);
    }
    if capture.finish().is_err() {
        // The capture was discarded, and with it everything encoded into it.
        encode()?;
        return Ok(IslandRun::Eager);
    }
    // A capture *records* launches; it does not run them. So the step that
    // captured an island has, at this point, done none of its work, and the
    // fresh graph is launched once here to do it.
    //
    // Leaving it out is the quietest bug this file can have. The pass still
    // ends, every buffer still holds a number, and the only step that is wrong
    // is the first one at each row count - so a prompt answers correctly and
    // its continuation does not.
    if !graph.replay(key)? {
        // The slot was committed a moment ago and is gone: something evicted
        // it between the two calls. Encoding is the same fallback every other
        // failure here takes.
        encode()?;
        return Ok(IslandRun::Eager);
    }
    Ok(IslandRun::Captured)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{GdnShape, LayerKind};
    use crate::sparse::shape::{GqaShape, IndexerShape, MoeShape, MropeSections, PleShape};

    fn shape(n_layers: u32, ple_layer: Option<u32>) -> SparseShape {
        SparseShape {
            layers: (0..n_layers)
                .map(|il| {
                    if (il + 1).is_multiple_of(4) {
                        LayerKind::Attn
                    } else {
                        LayerKind::Gdn
                    }
                })
                .collect(),
            hidden: 2_560,
            vocab: 1_024,
            streams: 4,
            stream_rank: 320,
            gqa: GqaShape::new(24, 2, 256, 64),
            gdn: GdnShape::new(48, 16, 128, 4),
            moe: MoeShape::new(512, 10, 1, 640),
            indexer: IndexerShape::new(4, 128, 2_048, 4),
            ple: ple_layer.map(|il| PleShape::new(LayerIdx::new(il), 3, 8, 160, 4)),
            mrope: MropeSections::new(11, 11, 10),
            rms_eps: 1.0e-6,
            rope_theta: 1.0e7,
        }
    }

    #[test]
    fn the_chain_breaks_at_the_first_island_of_a_pass() {
        let shape = shape(8, None);
        assert_eq!(chain(&shape, LayerIdx::new(0)).mixer_input, MixInput::Raw);
        assert_eq!(
            chain(&shape, LayerIdx::new(1)).mixer_input,
            MixInput::AlreadyNormed
        );
    }

    #[test]
    fn the_chain_breaks_around_the_layer_that_holds_the_n_gram_block() {
        let shape = shape(8, Some(1));
        // The n-gram block rewrites the residual between the previous combine
        // and this mix, so what that combine left behind is not what this mix
        // reads.
        assert_eq!(chain(&shape, LayerIdx::new(1)).mixer_input, MixInput::Raw);
        // And the combine before it has nothing worth leaving.
        assert_eq!(
            chain(&shape, LayerIdx::new(0)).experts_next,
            ChainLink::Broken
        );
        // A layer nowhere near it is chained on both sides.
        let mid = chain(&shape, LayerIdx::new(4));
        assert_eq!(mid.mixer_input, MixInput::AlreadyNormed);
        assert_eq!(mid.experts_next, ChainLink::Chained);
    }

    #[test]
    fn the_chain_breaks_at_the_last_layer_of_a_pass() {
        let shape = shape(8, None);
        assert_eq!(
            chain(&shape, LayerIdx::new(7)).experts_next,
            ChainLink::Broken
        );
        assert_eq!(
            chain(&shape, LayerIdx::new(6)).experts_next,
            ChainLink::Chained
        );
    }

    #[test]
    fn the_feed_forward_island_is_always_handed_a_normed_input() {
        // Nothing can come between a layer's token mixer and its own
        // feed-forward, so this half of the chain never breaks.
        for ple in [None, Some(0), Some(3), Some(7)] {
            let shape = shape(8, ple);
            for il in 0..8 {
                assert_eq!(
                    chain(&shape, LayerIdx::new(il)).experts_input,
                    MixInput::AlreadyNormed,
                    "layer {il} with the n-gram block at {ple:?}"
                );
            }
        }
    }

    #[test]
    fn a_selected_pass_and_a_dense_one_are_different_captures_at_the_same_row_count() {
        let dense = IslandKey {
            layer: LayerIdx::new(3),
            island: Island::Mixer,
            rows: 1,
            selected: false,
            context: IslandContext::Session(7),
        };
        let selected = IslandKey {
            selected: true,
            ..dense
        };
        assert_eq!(dense.variant(), 1);
        assert_eq!(selected.variant(), 65);
        assert_ne!(
            dense.pack().expect("it fits"),
            selected.pack().expect("it fits")
        );
    }

    #[test]
    fn two_execution_contexts_never_share_a_slot() {
        let a = IslandKey {
            layer: LayerIdx::new(3),
            island: Island::Experts,
            rows: 1,
            selected: false,
            context: IslandContext::Session(1),
        };
        let b = IslandKey {
            context: IslandContext::Session(2),
            ..a
        };
        assert_ne!(a.pack().expect("it fits"), b.pack().expect("it fits"));
    }

    #[test]
    fn every_field_of_a_key_is_recoverable_rather_than_aliased() {
        // Four fields, four positions. The test is that changing any one of
        // them changes the slot, which is what a masked field would not do.
        let base = IslandKey {
            layer: LayerIdx::new(47),
            island: Island::Mixer,
            rows: 9,
            selected: true,
            context: IslandContext::Session(0x00ff_ffff),
        };
        let packed = base.pack().expect("every field fits");
        for other in [
            IslandKey {
                layer: LayerIdx::new(46),
                ..base
            },
            IslandKey {
                island: Island::Experts,
                ..base
            },
            IslandKey { rows: 8, ..base },
            IslandKey {
                selected: false,
                ..base
            },
            IslandKey {
                context: IslandContext::Session(0),
                ..base
            },
        ] {
            assert_ne!(packed, other.pack().expect("every field fits"));
        }
    }

    #[test]
    fn a_context_wider_than_its_field_is_refused_rather_than_masked() {
        let key = IslandKey {
            layer: LayerIdx::new(0),
            island: Island::Mixer,
            rows: 1,
            selected: false,
            context: IslandContext::Session(0x0100_0000),
        };
        let err = key.pack().expect_err("twenty-four bits is the field");
        assert!(
            matches!(&err, CoreError::InvalidValue { value, .. } if value.contains("context")),
            "{err:?}"
        );
    }

    #[test]
    fn a_batched_island_has_no_slot_and_is_encoded_rather_than_replayed() {
        // Not "a slot nothing replays from": no slot at all. A key that named
        // one of the batch's sessions would be a slot another batch containing
        // that session could replay, and what it would replay is this batch's
        // row order over that batch's page tables.
        let batched = IslandKey {
            layer: LayerIdx::new(3),
            island: Island::Mixer,
            rows: 4,
            selected: false,
            context: IslandContext::Batch,
        };
        assert_eq!(batched.pack().expect("a batch has no slot"), None);
        // And every session key still has one, at the same layer and width.
        assert!(
            IslandKey {
                context: IslandContext::Session(1),
                ..batched
            }
            .pack()
            .expect("a session's key fits")
            .is_some()
        );
    }

    #[test]
    fn a_prefill_chunk_is_never_captured_and_a_decode_step_always_is() {
        let plain = GraphPolicy::default();
        assert!(plain.capturable(1));
        assert!(!plain.capturable(2));
        assert!(!plain.capturable(512));
    }

    #[test]
    fn a_speculative_verify_is_captured_only_when_the_caller_asked_for_it() {
        let opt_out = GraphPolicy {
            spec_depth: 4,
            capture_verify: false,
        };
        assert!(opt_out.capturable(1));
        assert!(!opt_out.capturable(5));
        let opt_in = GraphPolicy {
            spec_depth: 4,
            capture_verify: true,
        };
        assert!(opt_in.capturable(5));
        assert!(!opt_in.capturable(6), "one row past the draft depth");
    }
}
