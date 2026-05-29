from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

try:
    # Preferred: actively maintained, ships wheels for current Python versions.
    from tree_sitter_language_pack import get_language as _get_language
    _TS_AVAILABLE = True
except Exception:
    try:
        # Legacy fallback for environments that still have the old package.
        from tree_sitter_languages import get_language as _get_language
        _TS_AVAILABLE = True
    except Exception:
        _TS_AVAILABLE = False

# Cache one official tree_sitter.Parser per language. We build parsers from the
# Language object ourselves rather than using the pack's get_parser(), whose
# returned objects expose an incompatible Node API on some platforms.
_PARSER_CACHE: dict = {}


def _get_parser(language: str):
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    from tree_sitter import Parser
    lang = _get_language(language)
    try:
        parser = Parser(lang)
    except TypeError:
        # Older tree_sitter API: construct empty, then set the language.
        parser = Parser()
        parser.set_language(lang)
    _PARSER_CACHE[language] = parser
    return parser


def ast_available() -> bool:
    """Whether tree-sitter AST chunking is available in this environment.

    When False, ccrag falls back to line-window chunking for all files.
    """
    return _TS_AVAILABLE

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
}

# Node types that represent top-level logical units worth chunking at
_CHUNK_NODE_TYPES: dict[str, list[str]] = {
    "python": ["function_definition", "class_definition", "decorated_definition"],
    "javascript": ["function_declaration", "class_declaration", "method_definition",
                   "arrow_function", "export_statement"],
    "typescript": ["function_declaration", "class_declaration", "method_definition",
                   "arrow_function", "export_statement", "interface_declaration",
                   "type_alias_declaration"],
    "tsx": ["function_declaration", "class_declaration", "method_definition",
            "arrow_function", "export_statement"],
    "go": ["function_declaration", "method_declaration", "type_declaration"],
    "rust": ["function_item", "impl_item", "struct_item", "enum_item", "trait_item"],
    "java": ["class_declaration", "method_declaration", "interface_declaration"],
    "cpp": ["function_definition", "class_specifier"],
    "c": ["function_definition"],
    "ruby": ["method", "class", "module"],
}

IGNORE_DIRS = {".git", ".ccrag", "node_modules", "__pycache__", ".venv", "venv",
               "dist", "build", ".next", ".nuxt", "target", "vendor"}

TEXT_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys()) | {
    ".txt", ".rst", ".ini", ".cfg", ".env", ".xml", ".html", ".css", ".scss",
}

MAX_CHUNK_BYTES = 6000
MIN_CHUNK_LINES = 3


@dataclass
class Chunk:
    file_path: str
    start_line: int      # 1-indexed
    end_line: int        # 1-indexed, inclusive
    content: str
    language: str


def _fallback_chunks(path: Path, source: str, language: str) -> list[Chunk]:
    """Line-window fallback for unsupported languages."""
    lines = source.splitlines()
    window = 60
    step = 40
    chunks = []
    for start in range(0, len(lines), step):
        end = min(start + window, len(lines))
        snippet = "\n".join(lines[start:end])
        if len(snippet.strip()) < 20:
            continue
        chunks.append(Chunk(
            file_path=str(path),
            start_line=start + 1,
            end_line=end,
            content=snippet,
            language=language,
        ))
    return chunks


def _ast_chunks(path: Path, source: str, language: str) -> list[Chunk]:
    try:
        parser = _get_parser(language)
    except Exception:
        return _fallback_chunks(path, source, language)

    tree = parser.parse(source.encode())
    target_types = set(_CHUNK_NODE_TYPES.get(language, []))
    if not target_types:
        return _fallback_chunks(path, source, language)

    lines = source.splitlines()
    chunks: list[Chunk] = []

    def walk(node):
        if node.type in target_types:
            start = node.start_point[0]
            end = node.end_point[0]
            if end - start < MIN_CHUNK_LINES:
                return
            snippet = "\n".join(lines[start : end + 1])
            if len(snippet.encode()) > MAX_CHUNK_BYTES:
                # recurse into children to get smaller chunks
                for child in node.children:
                    walk(child)
                return
            chunks.append(Chunk(
                file_path=str(path),
                start_line=start + 1,
                end_line=end + 1,
                content=snippet,
                language=language,
            ))
        else:
            for child in node.children:
                walk(child)

    walk(tree.root_node)

    # if AST found nothing useful, fall back
    if not chunks:
        return _fallback_chunks(path, source, language)
    return chunks


def chunk_file(path: Path) -> list[Chunk]:
    ext = path.suffix.lower()
    language = EXTENSION_TO_LANGUAGE.get(ext, "text")

    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not source.strip():
        return []

    if _TS_AVAILABLE and language != "text":
        return _ast_chunks(path, source, language)
    return _fallback_chunks(path, source, language)


def iter_files(root: Path, gitignore_path: Path | None = None):
    """Yield all indexable files under root, respecting .gitignore if present."""
    matcher = None
    if gitignore_path and gitignore_path.exists():
        try:
            from gitignore_parser import parse_gitignore
            matcher = parse_gitignore(gitignore_path)
        except Exception:
            pass

    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if matcher and matcher(str(p)):
            continue
        if p.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        yield p
