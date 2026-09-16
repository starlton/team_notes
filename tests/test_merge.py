"""Aligning diarization turns with transcript timings."""

from __future__ import annotations

import pytest

from app.diarize.merge import (SpeakerTurn, apply_speaker_names, assign_speakers,
                               best_speaker_for_interval, merge_adjacent, overlap,
                               relabel_speakers, speaker_statistics)
from app.transcribe.models import TranscriptSegment, Word


def words(*items: tuple[float, float, str]) -> list[Word]:
    return [Word(start, end, text) for start, end, text in items]


def test_overlap_of_disjoint_intervals_is_zero():
    assert overlap(0, 1, 2, 3) == 0
    assert overlap(0, 2, 1, 3) == 1


def test_best_speaker_picks_the_largest_overlap():
    turns = [SpeakerTurn(0, 10, "A"), SpeakerTurn(10, 20, "B")]
    assert best_speaker_for_interval(0, 8, turns) == "A"
    assert best_speaker_for_interval(12, 19, turns) == "B"


def test_best_speaker_falls_back_to_the_nearest_turn():
    """Whisper and pyannote boundaries differ by tens of milliseconds."""
    turns = [SpeakerTurn(1.0, 5.0, "A")]
    assert best_speaker_for_interval(0.9, 0.98, turns) == "A"


def test_best_speaker_gives_up_when_nothing_is_close():
    turns = [SpeakerTurn(100.0, 120.0, "A")]
    assert best_speaker_for_interval(0.0, 1.0, turns) is None


def test_best_speaker_with_no_turns_is_none():
    assert best_speaker_for_interval(0, 1, []) is None


def test_segment_is_split_where_the_speaker_changes():
    segment = TranscriptSegment(0, 4, "one two three four",
                                words=words((0, 1, "one"), (1, 2, "two"),
                                            (2, 3, "three"), (3, 4, "four")))
    turns = [SpeakerTurn(0, 2, "A"), SpeakerTurn(2, 4, "B")]
    result = assign_speakers([segment], turns)
    assert [(piece.speaker, piece.text) for piece in result] == [
        ("A", "one two"), ("B", "three four")]


def test_a_single_stray_word_is_smoothed_away():
    """A one-word flip between two runs of one speaker is a boundary artefact."""
    segment = TranscriptSegment(0, 5, "we should ship it yes now",
                                words=words((0, .6, "we"), (.6, 1.2, "should"),
                                            (1.2, 1.8, "ship"), (1.8, 2.2, "it"),
                                            (2.4, 2.7, "yes"), (3.0, 3.6, "now")))
    turns = [SpeakerTurn(0, 2.3, "A"), SpeakerTurn(2.35, 2.75, "B"),
             SpeakerTurn(2.9, 5.0, "A")]
    result = assign_speakers([segment], turns)
    assert len(result) == 1
    assert result[0].speaker == "A"


def test_a_long_interjection_is_not_smoothed_away():
    segment = TranscriptSegment(0, 6, "a b c d e f",
                                words=words((0, 1, "a"), (1, 2, "b"), (2, 3, "c"),
                                            (3, 4, "d"), (4, 5, "e"), (5, 6, "f")))
    turns = [SpeakerTurn(0, 2, "A"), SpeakerTurn(2, 4, "B"), SpeakerTurn(4, 6, "A")]
    result = assign_speakers([segment], turns)
    assert [piece.speaker for piece in result] == ["A", "B", "A"]


def test_segments_without_word_timings_still_get_a_speaker():
    result = assign_speakers([TranscriptSegment(0, 3, "hello")],
                             [SpeakerTurn(0, 5, "A")])
    assert result[0].speaker == "A"


def test_no_diarization_turns_leaves_segments_untouched():
    segments = [TranscriptSegment(0, 3, "hello")]
    assert assign_speakers(segments, []) == segments


def test_merge_adjacent_joins_the_same_speaker():
    merged = merge_adjacent([
        TranscriptSegment(0, 2, "Hello there.", speaker="A"),
        TranscriptSegment(2.2, 4, "How are you?", speaker="A"),
        TranscriptSegment(4.5, 6, "Fine.", speaker="B"),
    ])
    assert len(merged) == 2
    assert merged[0].text == "Hello there. How are you?"
    assert merged[0].end == 4


def test_merge_adjacent_respects_a_long_gap():
    merged = merge_adjacent([
        TranscriptSegment(0, 2, "One.", speaker="A"),
        TranscriptSegment(30, 32, "Two.", speaker="A"),
    ])
    assert len(merged) == 2


def test_relabel_uses_order_of_first_speech():
    segments = [TranscriptSegment(0, 1, "x", speaker="SPEAKER_07"),
                TranscriptSegment(1, 2, "y", speaker="SPEAKER_02"),
                TranscriptSegment(2, 3, "z", speaker="SPEAKER_07")]
    relabelled, mapping = relabel_speakers(segments)
    assert mapping == {"SPEAKER_07": "Speaker 1", "SPEAKER_02": "Speaker 2"}
    assert [s.speaker for s in relabelled] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_speaker_statistics_sum_to_one():
    stats = speaker_statistics([
        TranscriptSegment(0, 6, "a b c", speaker="Speaker 1"),
        TranscriptSegment(6, 8, "d", speaker="Speaker 2"),
    ])
    assert stats[0]["speaker"] == "Speaker 1"     # ordered by talk time
    assert pytest.approx(sum(row["share"] for row in stats), abs=1e-6) == 1.0
    assert stats[0]["words"] == 3


def test_speaker_statistics_label_unattributed_speech():
    stats = speaker_statistics([TranscriptSegment(0, 1, "hi")])
    assert stats[0]["speaker"] == "Unknown"


def test_apply_speaker_names_replaces_labels():
    renamed = apply_speaker_names(
        [TranscriptSegment(0, 1, "hi", speaker="Speaker 1")], {"Speaker 1": "Ada"})
    assert renamed[0].speaker == "Ada"


def test_apply_speaker_names_leaves_unknown_labels_alone():
    renamed = apply_speaker_names(
        [TranscriptSegment(0, 1, "hi", speaker="Speaker 9")], {"Speaker 1": "Ada"})
    assert renamed[0].speaker == "Speaker 9"


def test_the_whole_merge_chain_produces_a_readable_transcript():
    """End to end: raw Whisper segments plus pyannote turns -> named turns."""
    segments = [
        TranscriptSegment(0, 3, "Shall we start",
                          words=words((0, 1, "Shall"), (1, 2, "we"), (2, 3, "start"))),
        TranscriptSegment(3.1, 6, "Yes lets go",
                          words=words((3.1, 4, "Yes"), (4, 5, "lets"), (5, 6, "go"))),
    ]
    turns = [SpeakerTurn(0, 3.05, "SPEAKER_01"), SpeakerTurn(3.05, 6, "SPEAKER_00")]
    merged = merge_adjacent(assign_speakers(segments, turns))
    relabelled, mapping = relabel_speakers(merged)
    named = apply_speaker_names(relabelled, {"Speaker 1": "Ada", "Speaker 2": "Grace"})
    assert len(mapping) == 2
    assert [(s.speaker, s.text) for s in named] == [
        ("Ada", "Shall we start"), ("Grace", "Yes lets go")]


def test_a_single_speaker_segment_keeps_its_original_text():
    """Whisper can omit a word from its timings; the text must not lose it."""
    segment = TranscriptSegment(0, 4, "Shall we ship the release on Friday?",
                                words=words((0, 1, "Shall"), (1, 2, "we"),
                                            (2, 3, "ship"), (3, 4, "Friday?")))
    result = assign_speakers([segment], [SpeakerTurn(0, 5, "A")])
    assert len(result) == 1
    assert result[0].text == "Shall we ship the release on Friday?"
    assert result[0].speaker == "A"


def test_a_segment_with_no_words_at_all_is_preserved():
    segment = TranscriptSegment(0, 4, "Some text", words=[])
    result = assign_speakers([segment], [SpeakerTurn(0, 5, "A")])
    assert result[0].text == "Some text"
