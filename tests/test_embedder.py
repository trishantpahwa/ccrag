import sentence_transformers

from ccrag.embedder import Embedder, models_cache_dir


def test_models_cache_dir(tmp_path):
    assert models_cache_dir(tmp_path) == str(tmp_path / ".ccrag" / "models")


def test_embedder_loads_from_local_cache_when_present(monkeypatch):
    calls = []

    class FakeST:
        def __init__(self, name, local_files_only=False, cache_folder=None):
            calls.append({"local": local_files_only, "cache": cache_folder})

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    _ = Embedder("m", cache_folder="/cache")._model
    # Cached model loads in one shot, offline, from the given folder.
    assert calls == [{"local": True, "cache": "/cache"}]


def test_embedder_downloads_when_not_cached(monkeypatch):
    calls = []

    class FakeST:
        def __init__(self, name, local_files_only=False, cache_folder=None):
            calls.append({"local": local_files_only, "cache": cache_folder})
            if local_files_only:
                raise OSError("not in local cache")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    _ = Embedder("m", cache_folder="/cache")._model
    # First a local-only attempt, then a download — both into .ccrag/models.
    assert [c["local"] for c in calls] == [True, False]
    assert all(c["cache"] == "/cache" for c in calls)
