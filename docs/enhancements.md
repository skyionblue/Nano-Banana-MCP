# Enhancement Proposals — Implemented in v1.1.0

All changes below were implemented on branch `feature/enhancements-v1.1.0`.

---

## Architecture Change: 2-Tool Design

The server now exposes exactly **2 MCP tools** instead of 7. This keeps the tool list loaded into the model's context minimal.

| Tool | Purpose |
|------|---------|
| `search` | Discover available operations, filtered by optional keyword |
| `execute` | Run any operation by name with typed arguments |

All 7 original operations (generate, edit, continue, list, info, config, status) are now implemented as internal handlers registered in an `OperationDef` registry. The model calls `search` to find what's available and `execute` to run it — the full parameter schemas are only loaded on demand via `search`.

### Usage pattern
```
// Discover operations
search({query: "generate"})

// Run one
execute({operation: "generate_image", arguments: {prompt: "a sunset"}})

// Iterate
execute({operation: "continue_editing", arguments: {prompt: "add birds"}})
```

---

## Token Usage in Responses

Every `generate_image` and `edit_image` response now appends token counts from `response.usageMetadata`:

```
📊 Tokens used: 1,234 (prompt: 980, response: 254)
```

---

## Configurable Model, Format, Output Directory, and Timeout

The Zod `ConfigSchema` was extended with optional fields (all have defaults — existing config files continue to parse cleanly):

| Field | Default | Notes |
|-------|---------|-------|
| `model` | `gemini-2.5-flash-preview-05-20` | Gemini model name |
| `outputDir` | Platform auto-detect | Directory where images are saved |
| `outputFormat` | `png` | `png`, `jpeg`, or `webp` |
| `timeoutMs` | `120000` | Per-call API timeout in ms |

### Environment variable overrides

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini API key (existing) |
| `NANO_BANANA_MODEL` | Override model |
| `NANO_BANANA_OUTPUT_DIR` | Override output directory |
| `NANO_BANANA_OUTPUT_FORMAT` | Override output format |
| `NANO_BANANA_TIMEOUT_MS` | Override timeout |

Env vars take precedence over the config file.

---

## Other Reliability Improvements

- **API timeout** — `Promise.race` with a 120s default, timer cleared in `.finally()`
- **Retry with backoff** — 3 attempts at 1s / 2s / 4s delays, each logged to stderr
- **Path traversal guard** — `imagePath` and `referenceImages` entries are validated against `homedir`, `tmpdir`, `cwd`, and the output directory before any file read
- **Reference image warnings** — failed loads are logged to stderr with the failing path instead of being silently swallowed
- **Config file permissions** — written with `0o600` so only the owning user can read the API key
- **TypeScript interfaces** — `ContentPart`, `McpContent`, and `OperationDef` replace all `any[]` usage

---

## Files Modified

| File | Change |
|------|--------|
| `src/index.ts` | All code changes |
| `package.json` | Version bumped to `1.1.0` |
| `README.md` | New tools, env vars, and architecture documented |
| `docs/enhancements.md` | This file |
