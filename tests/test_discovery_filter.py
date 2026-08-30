from pathlib import Path
from podtx.format_cmd import discover_transcript_jsons

def test_discover_excludes_summary(tmp_path: Path):
    root = tmp_path / "transcripts" / "feed-a"
    root.mkdir(parents=True)
    (root / "ep.json").write_text('{"title":"t"}', encoding="utf-8")
    (root / "ep.summary.json").write_text('{"overview":"o"}', encoding="utf-8")
    (root / "ep.summary.md").write_text('# hi', encoding="utf-8")
    res = discover_transcript_jsons(tmp_path / "transcripts", feed="feed-a")
    assert len(res) == 1
    assert res[0].name == "ep.json"
    res2 = discover_transcript_jsons(tmp_path / "transcripts")
    assert len(res2) == 1
    assert res2[0].name == "ep.json"
