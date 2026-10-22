from __future__ import annotations

import numpy as np

from tests.audio_gen import silence, tone
from transkrib.diarize import Turn
from transkrib.segment import drop_short, merge_turns, number_speakers, plan_chunks, split_long

RATE = 16_000


def test_merge_same_speaker_short_pause() -> None:
    merged = merge_turns([Turn(0.0, 5.0, "A"), Turn(5.3, 9.0, "A")])
    assert merged == [Turn(0.0, 9.0, "A")]


def test_no_merge_long_pause() -> None:
    turns = [Turn(0.0, 5.0, "A"), Turn(5.8, 9.0, "A")]
    assert merge_turns(turns) == turns


def test_no_merge_speaker_change() -> None:
    turns = [Turn(0.0, 5.0, "A"), Turn(5.3, 9.0, "B")]
    assert merge_turns(turns) == turns


def test_merge_does_not_jump_over_other_speaker() -> None:
    turns = [Turn(0.0, 2.0, "A"), Turn(2.1, 2.3, "B"), Turn(2.4, 4.0, "A")]
    assert merge_turns(turns) == turns


def test_merge_sorts_and_chains() -> None:
    turns = [Turn(4.2, 6.0, "A"), Turn(0.0, 2.0, "A"), Turn(2.2, 4.0, "A")]
    assert merge_turns(turns) == [Turn(0.0, 6.0, "A")]


def _signal_with_pauses(total: float, pauses: list[tuple[float, float]]) -> np.ndarray:
    rng = np.random.default_rng(0)
    signal = tone(total, RATE) + rng.normal(0, 0.05, round(total * RATE)).astype(np.float32)
    for a, b in pauses:
        signal[round(a * RATE) : round(b * RATE)] = silence(b - a, RATE)
    return signal


def test_split_long_cuts_at_planted_silence() -> None:
    pauses = [(18.0, 18.6), (37.0, 37.6), (55.0, 55.6)]
    samples = _signal_with_pauses(70.0, pauses)
    pieces = split_long(Turn(0.0, 70.0, "S"), samples)
    assert pieces[0].start == 0.0 and pieces[-1].end == 70.0
    assert all(p.end - p.start <= 22.0 + 1e-9 for p in pieces)
    assert all(a.end == b.start for a, b in zip(pieces, pieces[1:], strict=False))
    cuts = [p.end for p in pieces[:-1]]
    assert len(cuts) == 3
    for cut, (a, b) in zip(cuts, pauses, strict=True):
        assert a <= cut <= b, (cut, a, b)


def test_split_long_never_exceeds_limits_without_silence() -> None:
    rng = np.random.default_rng(1)
    samples = rng.normal(0, 0.3, 70 * RATE).astype(np.float32)
    for max_sec in (22.0, 24.0):
        pieces = split_long(Turn(0.0, 70.0, "S"), samples, max_sec=max_sec)
        assert all(p.end - p.start <= min(max_sec, 24.0) + 1e-9 for p in pieces)
        assert all(p.end - p.start >= 15.0 - 1e-9 for p in pieces[:-1])
        assert pieces[-1].end == 70.0


def test_split_long_beyond_signal_end() -> None:
    pieces = split_long(Turn(0.0, 70.0, "S"), np.zeros(0, np.float32))
    assert all(p.end - p.start <= 22.0 for p in pieces)
    assert pieces[-1].end == 70.0


def test_split_long_short_region_untouched() -> None:
    region = Turn(3.0, 20.0, "S")
    assert split_long(region, np.zeros(30 * RATE, np.float32)) == [region]


def test_drop_short() -> None:
    turns = [Turn(0.0, 0.2, "A"), Turn(1.0, 1.3, "A"), Turn(2.0, 5.0, "B")]
    assert drop_short(turns) == turns[1:]


def test_number_speakers_by_first_appearance() -> None:
    turns = [Turn(0, 1, "SPEAKER_03"), Turn(1, 2, "SPEAKER_00"), Turn(2, 3, "SPEAKER_03"), Turn(3, 4, "SPEAKER_01")]
    assert number_speakers(turns) == [1, 2, 1, 3]


def test_plan_chunks_pipeline_order() -> None:
    samples = _signal_with_pauses(60.0, [(18.0, 18.6)])
    turns = [Turn(0.0, 10.0, "B"), Turn(10.2, 30.0, "B"), Turn(30.5, 30.7, "C"), Turn(31.0, 40.0, "A")]
    chunks = plan_chunks(turns, samples)
    assert [c.speaker for c in chunks] == ["B", "B", "A"]
    assert chunks[0].start == 0.0 and 18.0 <= chunks[0].end <= 18.6
    assert chunks[1].end == 30.0
    assert chunks[2] == Turn(31.0, 40.0, "A")
