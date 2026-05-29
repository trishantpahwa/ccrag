from pathlib import Path

import pytest

from ccrag import chunker
from ccrag.chunker import _fallback_chunks, ast_available, chunk_file, iter_files

PY_SAMPLE = '''def alpha(x):
    y = x + 1
    z = y * 2
    return z


class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return "hi " + self.name
'''


def _lines(n: int, word: str = "line") -> str:
    return "\n".join(f"{word} {i} has sufficient content here" for i in range(n))


def test_ast_available_returns_bool():
    assert isinstance(ast_available(), bool)


def test_fallback_chunks_windows():
    chunks = _fallback_chunks(Path("x.txt"), _lines(100), "text")
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (41, 100), (81, 100)]
    assert all(c.language == "text" for c in chunks)


def test_fallback_skips_tiny_snippets():
    assert _fallback_chunks(Path("x.txt"), "a\n\nb", "text") == []


def test_chunk_file_empty_returns_empty(tmp_path):
    p = tmp_path / "empty.py"
    p.write_text("   \n  \n")
    assert chunk_file(p) == []


def test_chunk_file_unknown_extension_uses_fallback(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text(_lines(100, "note"))
    chunks = chunk_file(p)
    assert chunks
    assert chunks[0].language == "text"
    assert chunks[0].start_line == 1


def test_chunk_file_fallback_when_ts_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(chunker, "_TS_AVAILABLE", False)
    p = tmp_path / "mod.py"
    p.write_text(_lines(100))
    chunks = chunk_file(p)
    # Line-window fallback, but language is still derived from the extension.
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (41, 100), (81, 100)]
    assert all(c.language == "python" for c in chunks)


@pytest.mark.skipif(not ast_available(), reason="tree-sitter not installed")
def test_ast_chunks_python_at_boundaries(tmp_path):
    p = tmp_path / "sample.py"
    p.write_text(PY_SAMPLE)
    chunks = chunk_file(p)
    assert chunks
    assert all(c.language == "python" for c in chunks)
    first_lines = {c.content.splitlines()[0] for c in chunks}
    assert any(s.startswith("def alpha") for s in first_lines)
    assert any(s.startswith("class Greeter") for s in first_lines)
    alpha = next(c for c in chunks if c.content.startswith("def alpha"))
    assert alpha.start_line == 1


@pytest.mark.skipif(not ast_available(), reason="tree-sitter not installed")
def test_ast_skips_tiny_functions(tmp_path):
    src = (
        "def small(): return 1\n\n"
        "def big(a):\n    b = a + 1\n    c = b + 2\n    d = c + 3\n    return d\n"
    )
    p = tmp_path / "s.py"
    p.write_text(src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].content.startswith("def big")


def test_iter_files_filters_extensions_and_ignored_dirs(tmp_path):
    (tmp_path / "keep.py").write_text("x = 1\n")
    (tmp_path / "keep.md").write_text("# hi\n")
    (tmp_path / "skip.bin").write_text("nope\n")
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "dep.js").write_text("y = 2\n")

    found = {p.name for p in iter_files(tmp_path)}
    assert {"keep.py", "keep.md"} <= found
    assert "skip.bin" not in found
    assert "dep.js" not in found


def test_iter_files_respects_gitignore(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "secret.py").write_text("s = 1\n")
    gi = tmp_path / ".gitignore"
    gi.write_text("secret.py\n")

    found = {p.name for p in iter_files(tmp_path, gi)}
    assert "a.py" in found
    assert "secret.py" not in found
