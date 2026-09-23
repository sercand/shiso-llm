//! A chat turn's prose, without the whitespace that frames its structure.
//!
//! # Why the newlines around a call are not content
//!
//! A reasoning template writes an assistant turn as `<think>\n…\n</think>\n\n`,
//! then the content, then `\n\n<tool_call>…` with one `\n` between calls, and it
//! reads a replayed turn's content back through `| trim`. The model reproduces
//! that framing, so a parser that hands every byte outside a tag to the client
//! sends the framing as content: a turn of four calls came back as
//! `"\n\n\n\n\n"`, which a client shows as a blank message. The official serving
//! path answers the same turn with no content at all, and trims the prose of
//! every other turn at both ends.
//!
//! So prose is trimmed at every edge the structure gives it: the start of the
//! turn, either side of a trace, either side of a call, and the end of the
//! turn. Whitespace inside prose is kept. Trimming here changes nothing on
//! replay, because the template trims the same content again.
//!
//! # Why only a rendered chat turn
//!
//! A raw completion has no template behind it. Its text is the caller's
//! verbatim - the leading space of ` Paris` is the answer - so that surface
//! gets the parser without this stage.

use crate::schemes::StreamParser;
use crate::segment::Segment;
use crate::tags::TagParser;

/// A [`TagParser`] whose [`Segment::Text`] is trimmed at structural edges.
#[derive(Debug)]
pub(crate) struct TrimmedProse {
    inner: TagParser,
    /// Nothing has been released since the turn began or since the last
    /// structural segment, so leading whitespace is framing.
    at_edge: bool,
    /// Whitespace at the end of the prose released so far. It is held until
    /// more prose shows it was interior, and dropped at an edge.
    held: String,
}

impl TrimmedProse {
    /// Wraps `inner`, for a turn that starts at an edge.
    pub(crate) const fn new(inner: TagParser) -> Self {
        Self {
            inner,
            at_edge: true,
            held: String::new(),
        }
    }

    /// Trims the text among `segments`; everything else passes through.
    ///
    /// A text segment that is only whitespace produces nothing. Its token ids
    /// are not lost: the caller carries ids over to the next delta that is sent.
    fn trim(&mut self, segments: Vec<Segment>) -> Vec<Segment> {
        let mut out = Vec::with_capacity(segments.len());
        for segment in segments {
            match segment {
                Segment::Text(raw) => {
                    let text = if self.at_edge {
                        raw.trim_start()
                    } else {
                        raw.as_str()
                    };
                    let body = text.trim_end();
                    let tail = &text[body.len()..];
                    if body.is_empty() {
                        // Only whitespace: at an edge it is already gone, and
                        // otherwise it waits to learn which it is.
                        if !self.at_edge {
                            self.held.push_str(tail);
                        }
                        continue;
                    }
                    self.at_edge = false;
                    let mut released = std::mem::take(&mut self.held);
                    released.push_str(body);
                    self.held.push_str(tail);
                    out.push(Segment::Text(released));
                }
                thinking @ Segment::Thinking(_) => out.push(thinking),
                // Every other segment is structure: the prose before it ended
                // here, and the prose after it starts at an edge.
                structure => {
                    self.held.clear();
                    self.at_edge = true;
                    out.push(structure);
                }
            }
        }
        out
    }
}

impl StreamParser for TrimmedProse {
    fn push(&mut self, text: &str) -> Vec<Segment> {
        let segments = self.inner.push(text);
        self.trim(segments)
    }

    fn finish(&mut self) -> Vec<Segment> {
        let segments = self.inner.finish();
        let out = self.trim(segments);
        // The end of the turn is an edge: trailing whitespace was framing.
        self.held.clear();
        out
    }

    fn wants_greedy(&self) -> bool {
        self.inner.wants_greedy()
    }

    fn in_reasoning(&self) -> bool {
        self.inner.in_reasoning()
    }
}
