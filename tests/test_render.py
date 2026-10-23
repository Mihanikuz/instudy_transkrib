from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from tests.instudy_reference import extract_subtitles
from transkrib.render import build_meta, parse_vtt, remove_fillers, render_txt, render_vtt, validate_vtt
from transkrib.types import Segment

FIXTURES = Path(__file__).parent / "fixtures" / "render"
ENGINE = "GigaAM v3 e2e RNNT, pyannote community-1"

# 3 спикера, 7 сегментов: пауза > 60 с (сегмент 5) и абзац > 180 с (сегмент 6) у спикера 1.
SCENARIO = [
    Segment(3.12, 9.86, 1, "Добрый день, коллеги. Сегодня продолжаем тему внимания."),
    Segment(10.2, 14.05, 2, "А можно уточнить про прошлое задание?"),
    Segment(14.5, 22.0, 1, "Да,  конечно .  Задание было про « селективное внимание » ."),
    Segment(23.0, 45.0, 1, "Начнём с модели фильтра Бродбента."),
    Segment(120.0, 140.0, 1, "После перерыва продолжим."),
    Segment(141.0, 301.5, 1, "Модель ослабления Трейсман отличается тем, что канал не отсекается полностью."),
    Segment(302.0, 306.5, 3, "Спасибо, всё понятно."),
]


def golden(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def txt(segments: list[Segment]) -> str:
    return render_txt(segments, discipline="Общая психология", date=date(2026, 9, 29), engine_line=ENGINE)


def test_vtt_matches_golden_file() -> None:
    assert render_vtt(SCENARIO).encode("utf-8") == golden("lecture.vtt")


def test_txt_matches_golden_file() -> None:
    assert txt(SCENARIO).encode("utf-8") == golden("lecture.txt")


@pytest.mark.parametrize("render", [render_vtt, txt])
def test_utf8_without_bom_lf_and_final_newline(render) -> None:
    out = render(SCENARIO)
    raw = out.encode("utf-8")
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert "\r" not in out
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_txt_paragraphs() -> None:
    lines = [ln for ln in txt(SCENARIO).split("\n") if ln.startswith("[")]
    assert [ln.split(":", 3)[2].split("]")[1].strip() for ln in lines] == [
        "Спикер 1",
        "Спикер 2",
        "Спикер 1",  # сегменты 3–4 склеены
        "Спикер 1",  # пауза 75 с
        "Спикер 1",  # абзац с 120 с длиннее 180 с
        "Спикер 3",
    ]
    assert lines[2].startswith("[00:00:14] Спикер 1: Да, конечно. Задание было про «селективное внимание». Начнём")


def test_five_hour_timecodes() -> None:
    seg = [Segment(5 * 3600 + 0.5, 5 * 3600 + 3.25, 1, "Конец записи.")]
    assert "05:00:00.500 --> 05:00:03.250" in render_vtt(seg)
    assert "[05:00:00] Спикер 1: Конец записи." in txt(seg)


def test_escaping_in_vtt_but_not_in_txt() -> None:
    seg = [Segment(1.0, 2.0, 1, "<script>alert(1)</script> & co --> x")]
    vtt = render_vtt(seg)
    assert "<v Спикер 1>&lt;script&gt;alert(1)&lt;/script&gt; &amp; co --&gt; x" in vtt
    assert "<script>" not in vtt.split("<v Спикер 1>", 1)[1]
    assert "Спикер 1: <script>alert(1)</script> & co --> x" in txt(seg)


def test_empty_result_is_an_error() -> None:
    for render in (render_vtt, txt):
        with pytest.raises(ValueError):
            render([])
        with pytest.raises(ValueError):
            render([Segment(1.0, 2.0, 1, "   ")])


def test_empty_segments_are_skipped() -> None:
    segs = [Segment(1.0, 2.0, 1, ""), Segment(2.0, 3.0, 1, "Текст.")]
    assert parse_vtt(render_vtt(segs))[0].text == "Текст."


def test_speakers_are_renumbered_without_gaps() -> None:
    # Спикер 2 сказал только «э-э» — его реплика исчезла, Спикер 3 становится Спикером 2.
    segs = [Segment(1.0, 2.0, 1, "Начнём."), Segment(2.0, 3.0, 2, "Э-э..."), Segment(3.0, 4.0, 3, "Вопрос.")]
    assert [c.speaker for c in parse_vtt(render_vtt(segs))] == [1, 2]
    assert "Спикер 3" not in txt(segs)
    # Номера — по первому появлению среди оставшихся реплик, даже если на входе они в другом порядке.
    assert [c.speaker for c in parse_vtt(render_vtt([Segment(1.0, 2.0, 5, "а б"), Segment(2.0, 3.0, 2, "в")]))] == [
        1,
        2,
    ]
    with pytest.raises(ValueError):
        render_vtt([Segment(1.0, 2.0, 0, "а")])


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        # Реальная фраза из проверки на сервере (22.10): заполнители в начале и в середине.
        ("А-а, его, э-э, на него обрушивается критика.", "Его, на него обрушивается критика."),
        ("А-а критических точек, э-э которые, значит, возникают.", "Критических точек, которые, значит, возникают."),
        ("Э-э, в этом случае появляются отчёты.", "В этом случае появляются отчёты."),
        ("Итак. Э-э, следующий момент.", "Итак. Следующий момент."),
        ("Это, м-м-м, важно, эээ, понимать.", "Это, важно, понимать."),
        ("Хм, интересно. Ммм... да.", "Интересно. Да."),
        ("Тут эм вот так.", "Тут вот так."),
        # Союзы, предлоги и обычные слова остаются.
        ("А можно уточнить? О чём речь, у кого и к чему?", "А можно уточнить? О чём речь, у кого и к чему?"),
        ("Ну вот, значит, эмоции и мммодель.", "Ну вот, значит, эмоции и мммодель."),
        ("Э-это аа-группа.", "Э-это аа-группа."),
        ("Э-э.", ""),
    ],
)
def test_remove_fillers(raw: str, clean: str) -> None:
    assert remove_fillers(raw) == clean


def test_fillers_removed_in_both_files() -> None:
    segs = [Segment(1.0, 4.0, 1, "А-а, его, э-э, на него обрушивается критика."), Segment(5.0, 6.0, 2, "Э-э...")]
    vtt, text = render_vtt(segs), txt(segs)
    for out in (vtt, text):
        assert "а-а" not in out.lower() and "э-э" not in out.lower()
    assert "<v Спикер 1>Его, на него обрушивается критика." in vtt
    assert "[00:00:01] Спикер 1: Его, на него обрушивается критика." in text
    assert "Спикер 2" not in vtt + text


def test_only_fillers_is_empty_result() -> None:
    with pytest.raises(ValueError):
        render_vtt([Segment(1.0, 2.0, 1, "Э-э, м-м...")])


def test_voices_are_only_speaker_n() -> None:
    for cue in parse_vtt(render_vtt(SCENARIO)):
        assert cue.speaker >= 1
    assert {line.split(">")[0] for line in render_vtt(SCENARIO).split("\n") if line.startswith("<v ")} == {
        "<v Спикер 1",
        "<v Спикер 2",
        "<v Спикер 3",
    }


def test_overlap_is_trimmed() -> None:
    segs = [Segment(1.0, 5.0, 1, "раз"), Segment(4.0, 6.0, 2, "два")]
    cues = parse_vtt(render_vtt(segs))
    assert (cues[0].end_ms, cues[1].start_ms) == (4000, 4000)


def test_unsorted_input_is_sorted() -> None:
    cues = parse_vtt(render_vtt(list(reversed(SCENARIO))))
    assert [c.start_ms for c in cues] == sorted(c.start_ms for c in cues)


def test_bad_times_rejected() -> None:
    with pytest.raises(ValueError):
        render_vtt([Segment(2.0, 2.0, 1, "а")])
    with pytest.raises(ValueError):
        render_vtt([Segment(-1.0, 2.0, 1, "а")])


def test_validate_vtt_roundtrip() -> None:
    validate_vtt(render_vtt(SCENARIO), SCENARIO)
    with pytest.raises(ValueError):
        validate_vtt(render_vtt(SCENARIO[:-1]), SCENARIO)


@pytest.mark.parametrize(
    "broken",
    [
        "﻿WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\n<v Спикер 1>а\n",
        "WEBVTT\r\n\r\n1\r\n00:00:01.000 --> 00:00:02.000\r\n<v Спикер 1>а\r\n",
        "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\n<v Анна>а\n",
        "WEBVTT\n\n1\n00:00:02.000 --> 00:00:01.000\n<v Спикер 1>а\n",
        "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\n<v Спикер 1><b>а</b>\n",
        "WEBVTT\n",
    ],
)
def test_parse_vtt_rejects_broken_files(broken: str) -> None:
    with pytest.raises(ValueError):
        parse_vtt(broken)


def test_instudy_understands_our_vtt() -> None:
    # Как instudy (BE-7) покажет текст для «Открыть», если возьмёт его из .vtt: голоса и текст без тегов.
    text = extract_subtitles(render_vtt(SCENARIO))
    assert text.startswith("Спикер 1: Добрый день, коллеги.")
    assert "\nСпикер 2: А можно уточнить про прошлое задание?" in text
    assert "\nСпикер 3: Спасибо, всё понятно." in text
    assert "-->" not in text and "<v" not in text


def test_instudy_unescapes_vtt_entities() -> None:
    text = extract_subtitles(render_vtt([Segment(1.0, 2.0, 1, "a < b & c")]))
    assert text == "Спикер 1: a < b & c"


def test_build_meta() -> None:
    meta = build_meta(
        server="instudy_transkrib 0.1.0",
        asr="gigaam v3_e2e_rnnt @abc",
        diarization="pyannote/speaker-diarization-community-1",
        audio_duration_sec=17940.4,
        processing_sec=911.6,
        speakers=4,
        segments=1873,
    )
    assert meta == {
        "server": "instudy_transkrib 0.1.0",
        "asr": "gigaam v3_e2e_rnnt @abc",
        "diarization": "pyannote/speaker-diarization-community-1",
        "audioDurationSec": 17940,
        "processingSec": 912,
        "speakers": 4,
        "segments": 1873,
    }


def test_date_as_iso_string() -> None:
    out = render_txt(SCENARIO, discipline="  Общая  психология ", date="2026-09-29", engine_line=ENGINE)
    assert out.startswith("Общая психология — 29.09.2026\nАвтотранскрипт: GigaAM")


# ------------------------------------------------------------------ правила instudy для результата (T-9)


def _many_speakers(n: int) -> list[Segment]:
    return [Segment(i * 2.0, i * 2.0 + 1.5, i + 1, f"Реплика номер {i + 1}.") for i in range(n)]


def test_more_than_99_speakers_merge_into_99() -> None:
    from tests.instudy_reference import check_vtt

    segments = _many_speakers(120)
    vtt = render_vtt(segments)
    voices = {int(v) for v in re.findall(r"<v Спикер (\d+)>", vtt)}
    assert voices == set(range(1, 100))
    assert vtt.count("<v Спикер 99>") == 22  # 99-й и все, кто после него
    assert check_vtt(vtt) == 120  # instudy принимает
    txt = render_txt(segments, discipline="Тест", date=date(2026, 9, 29), engine_line=ENGINE)
    assert "Спикер 100" not in txt and "Спикер 99: Реплика номер 99. Реплика номер 100." in txt
    assert (
        build_meta(
            server="s", asr="a", diarization="d", audio_duration_sec=1, processing_sec=1, speakers=120, segments=120
        )["speakers"]
        == 99
    )


def test_rendered_vtt_passes_instudy_result_check() -> None:
    from tests.instudy_reference import check_vtt

    segments = [*SCENARIO, Segment(310.0, 315.0, 1, "Формула: a < b && c > d.")]
    vtt = render_vtt(segments)
    assert check_vtt(vtt) == len(segments)
    lines = vtt.split("\n")
    assert lines[0] == "WEBVTT" and lines[1] == ""
    timing = [ln for ln in lines if "-->" in ln]
    assert all(
        re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", ln) for ln in timing
    )  # без настроек
    # между репликами ровно одна пустая строка: номер реплики всегда после пустой строки
    for i, ln in enumerate(lines):
        if "-->" in ln:
            assert lines[i - 2] == "" and lines[i - 1].isdigit()
    assert "a &lt; b &amp;&amp; c &gt; d." in vtt


def test_timecode_over_12_hours_is_rejected() -> None:
    from tests.instudy_reference import ResultError, check_vtt

    ok = render_vtt([Segment(11 * 3600 + 3599.0, 12 * 3600.0, 1, "Последняя реплика.")])
    assert "12:00:00.000" in ok and check_vtt(ok) == 1
    with pytest.raises(ValueError, match="12 часов"):
        render_vtt([Segment(12 * 3600.0, 12 * 3600 + 2.0, 1, "Слишком поздно.")])
    with pytest.raises(ResultError):  # и сам instudy такое не принял бы
        check_vtt("WEBVTT\n\n1\n12:00:00.000 --> 12:00:02.000\n<v Спикер 1>Слишком поздно.\n")
