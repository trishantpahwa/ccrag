import json

from click.testing import CliRunner

from ccrag.cli import main


def _lines(n: int) -> str:
    return "\n".join(f"line {i} with sufficient content here" for i in range(n))


def test_status_not_indexed_reports_chunking_mode(tmp_path):
    r = CliRunner().invoke(main, ["status", str(tmp_path)])
    assert r.exit_code == 0
    assert "Not indexed" in r.output
    assert "Chunking:" in r.output


def test_mcp_config_outputs_valid_json(tmp_path):
    r = CliRunner().invoke(main, ["mcp-config", str(tmp_path)])
    assert r.exit_code == 0
    cfg = json.loads(r.output)
    server = cfg["mcpServers"]["ccrag"]
    assert server["args"][0] == "serve"
    assert server["args"][1] == str(tmp_path.resolve())


def test_index_then_status(tmp_path, monkeypatch, fake_embedder_cls):
    monkeypatch.setattr("ccrag.embedder.Embedder", fake_embedder_cls)
    (tmp_path / "a.txt").write_text(_lines(80))

    r = CliRunner().invoke(main, ["index", str(tmp_path)])
    assert r.exit_code == 0, r.output

    r2 = CliRunner().invoke(main, ["status", str(tmp_path)])
    assert r2.exit_code == 0
    assert "files," in r2.output
