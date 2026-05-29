# ccrag

A RAG pipeline for [Claude Code](https://claude.ai/code). Indexes your codebase locally and exposes it as an MCP server so Claude Code can semantically search your code during sessions.

## How it works

```
ccrag index .        →  AST-chunks your code + embeds with sentence-transformers
                         stores vectors in .ccrag/ (LanceDB, stays in your repo)

ccrag serve .        →  MCP server (stdio) that Claude Code connects to
                         exposes search_codebase("how does auth work?") as a tool

Claude Code          →  automatically calls search_codebase when it needs context
                         gets back file paths, line ranges, and code snippets
```

## Install

```bash
pip install ccrag
```

## Usage

**1. Index your codebase**

```bash
cd /your/project
ccrag index .
```

**2. Get the MCP config snippet**

```bash
ccrag mcp-config .
```

This prints a JSON block to paste into `.claude/settings.json`:

```json
{
  "mcpServers": {
    "ccrag": {
      "command": "/path/to/ccrag",
      "args": ["serve", "/your/project"]
    }
  }
}
```

**3. Start a Claude Code session**

Claude Code will now automatically call `search_codebase` whenever it needs to understand the codebase. No changes to your prompts needed.

## Commands

| Command | Description |
|---|---|
| `ccrag index [PATH]` | Index or re-index the codebase |
| `ccrag index --force [PATH]` | Drop and rebuild the index |
| `ccrag serve [PATH]` | Start the MCP server (used by Claude Code) |
| `ccrag watch [PATH]` | Watch for file changes and re-index incrementally |
| `ccrag status [PATH]` | Show index stats (files, chunks) |
| `ccrag mcp-config [PATH]` | Print the settings.json snippet |

## How the index works

- **Chunking**: Uses [tree-sitter](https://tree-sitter.github.io) to split code at function/class/method boundaries — not arbitrary line windows. Falls back to line-window chunking for unsupported languages.
- **Embeddings**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via `sentence-transformers`. Runs entirely locally, no API keys required.
- **Storage**: [LanceDB](https://lancedb.com) in `.ccrag/` inside your project. Add `.ccrag/` to `.gitignore`.
- **Search**: Cosine similarity over dense embeddings. Returns top-K chunks with file path, line range, language, and source.

## Supported languages

Python, JavaScript, TypeScript, TSX, Go, Rust, Java, C, C++, Ruby, PHP, C#, Swift, Kotlin, Scala, Lua, Elixir, Haskell, OCaml, Bash, YAML, JSON, TOML, Markdown, and more.

## MCP tools exposed

| Tool | Description |
|---|---|
| `search_codebase(query, n_results=8)` | Semantic search over indexed code |
| `codebase_stats()` | Number of indexed files and chunks |

## Incremental updates

```bash
ccrag watch .   # re-indexes any file saved, in the background
```

Or just re-run `ccrag index .` — it upserts, so unchanged chunks are overwritten in place (fast for small diffs).

## License

MIT
