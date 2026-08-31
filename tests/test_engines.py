from __future__ import annotations

from pathlib import Path

from podtx.engines.registry import available_engines, get_engine
from podtx.models import Segment, Transcript


class FakeEngine:
    name = "fake"
    default_model = "fake-model"

    def transcribe(self, audio_path: Path, *, model: str | None = None, language: str = "en") -> Transcript:
        return Transcript(
            text="hello",
            segments=[Segment(0.0, 1.0, "hello")],
            language=language,
            model=model or self.default_model,
            engine=self.name,
        )


def test_registry_known_engines() -> None:
    assert "parakeet" in available_engines()
    assert "whisper" in available_engines()
    engine = get_engine("parakeet")
    assert engine.name == "parakeet"
    assert engine.default_model.startswith("mlx-community/")


def test_get_engine_unknown_raises() -> None:
    try:
        get_engine("bogus")
        assert False
    except ValueError as exc:
        assert "Unknown engine" in str(exc)


def test_parakeet_load_caches_and_error(monkeypatch, tmp_path: Path) -> None:
    from podtx.engines.parakeet import ParakeetEngine

    eng = ParakeetEngine()
    calls = []

    class FakeEncoder:
        def set_attention_model(self, *a, **k):
            calls.append(a)

    class FakeAsr:
        def __init__(self):
            self.encoder = FakeEncoder()

        def transcribe(self, path):
            class S:
                text = "seg text"
                start = 1.0
                end = 2.0
            return type("R", (), {"text": "full", "segments": [S()]})

    import sys
    import types
    fake_mod = types.ModuleType("parakeet_mlx")
    fake_mod.from_pretrained = lambda m: FakeAsr()
    monkeypatch.setitem(sys.modules, "parakeet_mlx", fake_mod)

    a = eng._load("m1", local_attention=True, local_attention_context_size=256)
    b = eng._load("m1", local_attention=True, local_attention_context_size=256)
    assert a is b
    assert calls == [("rel_pos_local_attn", (256, 256))]

    eng2 = ParakeetEngine()
    c = eng2._load("m2", local_attention=False, local_attention_context_size=256)
    assert c is not None

    def boom(m):
        raise ImportError("no parakeet")
    eng3 = ParakeetEngine()
    monkeypatch.setattr(fake_mod, "from_pretrained", boom)
    try:
        eng3._load("m", local_attention=True, local_attention_context_size=256)
        assert False
    except ImportError:
        pass


def test_parakeet_transcribe_dict_none_timestamps(tmp_path: Path) -> None:
    from podtx.engines.parakeet import ParakeetEngine

    eng = ParakeetEngine()

    class FakeAsr:
        def transcribe(self, path):
            segs = [
                {"text": "s1", "start": None, "end": None},
                {"text": "s2", "start": None, "end": None},
                {"text": "", "start": None, "end": None},
            ]
            return type("R", (), {"text": "s1 s2", "segments": segs})()

    eng._model_cache = {("mlx-community/parakeet-tdt-0.6b-v3", True, 256): FakeAsr()}
    tr = eng.transcribe(Path("/tmp/fake.wav"))
    assert tr.segments[0].start == 0.0
    assert tr.segments[0].end == 0.0
    assert tr.segments[1].start == 0.0


def test_parakeet_no_segments_uses_full_text(tmp_path: Path) -> None:
    from podtx.engines.parakeet import ParakeetEngine

    eng = ParakeetEngine()

    class FakeAsr:
        def transcribe(self, path):
            return type("R", (), {"text": "just words", "segments": []})()

    eng._model_cache = {("mlx-community/parakeet-tdt-0.6b-v3", True, 256): FakeAsr()}
    tr = eng.transcribe(Path("/tmp/fake.wav"))
    assert tr.text == "just words"
    assert tr.segments[0].text == "just words"


def test_whisper_transcribe_various(monkeypatch, tmp_path: Path) -> None:
    from podtx.engines.whisper import WhisperEngine, _is_suspicious_segment

    assert _is_suspicious_segment("") is True
    assert _is_suspicious_segment("aaaaaaaaaaaaaaaa") is True
    assert _is_suspicious_segment(" ".join(["hi"] * 10)) is True
    assert _is_suspicious_segment("lllllllllllllllllllllll") is True
    assert _is_suspicious_segment("normal words here are fine okay") is False

    class FakeResult:
        def get(self, key, default=None):
            if key == "segments":
                return [
                    {"text": "valid first", "start": 0.0, "end": 1.0},
                    {"text": "", "start": 2.0, "end": 3.0},
                    {"text": " ".join(["hi"] * 12), "start": 4.0, "end": 5.0},
                    {"text": "valid last", "start": 6.0, "end": 7.0},
                ]
            if key == "text":
                return "fallback text words"
            if key == "language":
                return "en"
            return default

    def fake_transcribe(*a, **k):
        return FakeResult()

    monkeypatch.setattr("sys.modules", {"mlx_whisper": type("M", (), {"transcribe": staticmethod(fake_transcribe)})()}, raising=False)
    import sys
    import types
    m = types.ModuleType("mlx_whisper")
    m.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", m)

    eng = WhisperEngine()
    tr = eng.transcribe(tmp_path / "a.wav")
    assert "valid first" in tr.text
    assert "valid last" in tr.text
    assert tr.model == WhisperEngine.default_model
    assert tr.language == "en"


def test_whisper_transcribe_all_suspicious_low_unique(monkeypatch, tmp_path: Path) -> None:
    from podtx.engines.whisper import WhisperEngine
    import sys
    import types

    class FakeResult:
        def get(self, key, default=None):
            if key == "segments":
                return [{"text": "a a a a a a a a a a a a a a a a a a a a a a a a a a", "start": 0.0, "end": 1.0}]
            if key == "text":
                return "   "
            if key == "language":
                return "en"
            return default

    def fake_transcribe(*a, **k):
        return FakeResult()

    m = types.ModuleType("mlx_whisper")
    m.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", m)

    eng = WhisperEngine()
    tr = eng.transcribe(tmp_path / "a.wav")
    assert tr.text == ""
    assert tr.segments == []


def test_whisper_import_error(monkeypatch, tmp_path: Path) -> None:
    from podtx.engines.whisper import WhisperEngine
    import sys
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    eng = WhisperEngine()
    try:
        eng.transcribe(tmp_path / "a.wav")
        assert False
    except ImportError:
        pass


def test_whisper_low_unique_diversity(monkeypatch, tmp_path: Path) -> None:
    from podtx.engines.whisper import WhisperEngine
    import sys
    import types

    class FakeResult:
        def get(self, key, default=None):
            if key == "segments":
                return [{"text": "bacdbcadcb" * 3, "start": 0.0, "end": 1.0}]
            return default

    def fake_transcribe(*a, **k):
        return FakeResult()

    m = types.ModuleType("mlx_whisper")
    m.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", m)

    eng = WhisperEngine()
    tr = eng.transcribe(tmp_path / "a.wav")
    # The 30-char single segment is dropped as suspicious (low unique diversity)
    assert tr.segments == []


def test_whisper_no_segments_but_text_fallback(monkeypatch, tmp_path: Path) -> None:
    from podtx.engines.whisper import WhisperEngine
    import sys
    import types

    class FakeResult:
        def get(self, key, default=None):
            if key == "segments":
                return [{"text": " ", "start": 0.0, "end": 1.0}]
            if key == "text":
                return "Some real spoken words are here"
            if key == "language":
                return "en"
            return default

    def fake_transcribe(*a, **k):
        return FakeResult()

    m = types.ModuleType("mlx_whisper")
    m.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", m)

    eng = WhisperEngine()
    tr = eng.transcribe(tmp_path / "a.wav")
    assert tr.text == "Some real spoken words are here"
    assert tr.segments[0].text == tr.text


def test_parakeet_import_error(tmp_path: Path) -> None:
    from podtx.engines.parakeet import ParakeetEngine
    import sys

    saved = sys.modules.pop("parakeet_mlx", None)
    try:
        eng = ParakeetEngine()
        try:
            eng.transcribe(tmp_path / "a.wav")
            assert False
        except ImportError:
            pass
    finally:
        if saved is not None:
            sys.modules["parakeet_mlx"] = saved
