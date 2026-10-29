//! The tower's word timer end to end.
//!
//! The cases need a Metal device,
//! `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER` and, for the word times,
//! the captures' alignment dumps; each says on stderr why it skipped.

#[cfg(target_os = "macos")]
mod device {
    use std::path::Path;
    use std::sync::Arc;

    use shiso_audio::{AudioClip, AudioTower, MelFrontend, Pcm, WordTiming};

    use super::super::NemotronTranscribeTower;
    use crate::nemotron_transcribe::align::tests::reference_words;
    use crate::nemotron_transcribe::encoder_tests::{clip_pcm, floats};
    use crate::nemotron_transcribe::spm::tests::aligned_refs;

    /// The tower, opened lazily, or `None` (said) when this box has no
    /// device or export.
    #[expect(clippy::print_stderr, reason = "a skipped case says why")]
    fn tower() -> Option<(Arc<dyn shiso_core::Backend>, NemotronTranscribeTower)> {
        let Some(path) = std::env::var_os("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER") else {
            eprintln!("SKIP: SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER is unset");
            return None;
        };
        let Some(backend) = crate::audio_ops::testing::device() else {
            eprintln!("SKIP: no Metal device");
            return None;
        };
        let tower =
            NemotronTranscribeTower::open(&backend, Path::new(&path), false).expect("opens");
        Some((backend, tower))
    }

    fn clip(tower: &NemotronTranscribeTower, pcm: &Pcm) -> AudioClip {
        let mel = MelFrontend::new(tower.mel_spec().clone())
            .expect("spec")
            .compute(pcm.samples());
        let tokens = tower.tokens_for(mel.valid()).expect("tokens");
        AudioClip::new(mel, tokens).expect("clip")
    }

    #[test]
    #[ignore = "needs a Metal device and the speech encoder export"]
    fn silence_encodes_and_yields_no_words() {
        let Some((backend, tower)) = tower() else {
            return;
        };
        let pcm = Pcm::new(vec![0.0; 32_000], 16_000).expect("pcm");
        let clip = clip(&tower, &pcm);
        let timer = tower.word_timer().expect("the tower times words");
        let ticket = timer.want(&clip);
        let rows = tower
            .encode(&backend, std::slice::from_ref(&clip))
            .expect("silence encodes");
        assert_eq!(rows.grids.len(), 1);
        assert_eq!(rows.grids[0].pad_count(), 26, "201 mel frames are 26 rows");
        let outcome = timer.words(&ticket, "", 2.0).expect("no device error");
        assert!(
            matches!(&outcome, WordTiming::Untimed { .. }),
            "an empty transcript has no words: {outcome:?}"
        );
    }

    /// How close the whole tower's times must be.
    #[derive(Clone, Copy)]
    struct Budget {
        /// Words per thousand timed to the reference's millisecond.
        exact_permille: usize,
        /// The worst start or end difference, if any is set.
        seconds_off: Option<f64>,
    }

    /// Every word exact.
    const EXACT: Budget = Budget {
        exact_permille: 1000,
        seconds_off: Some(0.0),
    };

    /// The reference transcript of `rel`'s clip timed by the whole tower:
    /// mel, both branches, the head and the alignment, all ours.
    #[expect(clippy::print_stderr, reason = "the agreement is the test's report")]
    fn words_end_to_end(rel: &str, clip_name: &str, budget: Budget) {
        let Some(dir) = aligned_refs(rel) else {
            return;
        };
        let Some((backend, tower)) = tower() else {
            return;
        };
        let pcm = clip_pcm(clip_name);
        let clip = clip(&tower, &pcm);
        let timer = tower.word_timer().expect("the tower times words");
        let ticket = timer.want(&clip);
        tower
            .encode(&backend, std::slice::from_ref(&clip))
            .expect("encodes");
        let transcript =
            std::fs::read_to_string(dir.join("ctc_transcript.txt")).expect("the transcript");
        let WordTiming::Timed(got) = timer
            .words(&ticket, &transcript, pcm.duration_secs())
            .expect("no device error")
        else {
            panic!("{rel}: the reference transcript did not align");
        };
        let want = reference_words(&dir);
        assert_eq!(got.len(), want.len(), "{rel}: word count");
        // Words are ordered by time, so a word whose time moved past its
        // neighbour's is out of place: bring each back to the reference's
        // position before comparing.
        let mut got = got;
        for i in 0..got.len() {
            if got[i].text != want[i].0
                && let Some(j) = (i + 1..got.len().min(i + 8)).find(|&j| got[j].text == want[i].0)
            {
                let word = got.remove(j);
                got.insert(i, word);
            }
        }
        let rows = u32::try_from(floats(&dir.join("diar_probs.f32")).len() / 8).expect("rows");
        let frame = pcm.duration_secs() / f64::from(2 * rows);
        let mut exact = 0;
        let mut worst = 0.0f64;
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.text, w.0, "{rel}: the words in order");
            let d = (g.start - w.1).abs().max((g.end - w.2).abs());
            if d == 0.0 {
                exact += 1;
            } else {
                eprintln!(
                    "  {:?}: ours {:.3}-{:.3}, reference {:.3}-{:.3}",
                    g.text, g.start, g.end, w.1, w.2
                );
            }
            worst = worst.max(d);
        }
        eprintln!(
            "{rel}: {exact} of {} words timed exactly; worst start or end off by {worst:.3} s \
             (one frame is {frame:.4} s)",
            got.len()
        );
        assert!(
            exact * 1000 >= got.len() * budget.exact_permille,
            "{rel}: only {exact} of {} words timed exactly",
            got.len()
        );
        assert!(
            budget.seconds_off.is_none_or(|limit| worst <= limit),
            "{rel}: a word time is {worst} s off"
        );
        drop(ticket);
        let again = timer.words(&timer.want(&clip), &transcript, pcm.duration_secs());
        assert!(
            matches!(again, Ok(WordTiming::Untimed { .. })),
            "a dropped ticket releases what the encode kept"
        );
    }

    #[test]
    #[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
    fn short_en_words_end_to_end() {
        words_end_to_end("short_en/verbatim", "short_en", EXACT);
    }

    #[test]
    #[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
    fn meeting_60s_words_end_to_end() {
        words_end_to_end("meeting_60s/verbatim", "meeting_60s", EXACT);
    }

    #[test]
    #[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
    fn meeting_60s_speakers_words_end_to_end() {
        words_end_to_end("meeting_60s/speakers", "meeting_60s", EXACT);
    }

    #[test]
    #[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
    fn meeting_10m_words_end_to_end() {
        words_end_to_end(
            "meeting_10m/verbatim",
            "meeting_10m",
            // The speaker branch flips 24 of the clip's 60,008 activity
            // bins against the reference (`meeting_10m_stages_match_reference`),
            // and activity decides which frames a word may sit on: measured on
            // the M4, 1788 of 1815 words exact, most of the rest one 10 ms
            // diarization frame off, one word placed across an 11 s gap.
            Budget {
                exact_permille: 980,
                seconds_off: None,
            },
        );
    }
}
