# Enhancement Proposals — v1.2

All items below were implemented in v1.2.0 on branch `feature/enhancements-v1.1.0`.

---

## What Was Implemented

### New Operations

#### `delete_image` ✅
Remove a specific generated image (and its metadata sidecar) from the output directory. Guards against deleting files outside the output directory. Clears the session's `lastImagePath` pointer if the deleted file was the most recent.

```json
execute({"operation": "delete_image", "arguments": {"imagePath": "/path/to/image.png"}})
```

---

#### `enhance_prompt` ✅
Use the Gemini text model (`gemini-2.5-flash`) to expand and improve a rough prompt before generating. Returns a detailed, high-quality prompt you can paste directly into `generate_image`.

```json
execute({
  "operation": "enhance_prompt",
  "arguments": {"prompt": "a cat", "style": "photorealistic"}
})
// Returns: "A photorealistic close-up portrait of a domestic tabby cat with vivid amber eyes..."
```

---

### Session & Data Persistence

#### Image metadata sidecar files ✅
Alongside each saved image (`generated-2025-01-24-abc123.png`), a companion JSON file is written (`generated-2025-01-24-abc123.json`) containing:

```json
{
  "operation": "generate",
  "prompt": "a sunset over mountains",
  "model": "gemini-2.5-flash-preview-05-20",
  "timestamp": "2025-01-24T12:00:00.000Z",
  "tokenUsage": {"total": 1234, "prompt": 980, "response": 254},
  "sourceImage": null
}
```

For edits, `sourceImage` points to the original file. `list_generated_images` shows the prompt from the sidecar alongside each file path.

---

#### Session persistence ✅
`lastImagePath` is now written to `.nano-banana-session.json` on every update and restored on server startup. `continue_editing` works across separate MCP server sessions without needing to re-specify the file path.

---

### UX Improvements

#### `search` verbose mode ✅
The `search` tool now accepts `verbose: true` to include full parameter descriptions instead of just name and type.

```json
search({"verbose": true})
search({"query": "edit", "verbose": true})
```

---

#### Cost estimation ✅
Token usage lines now include an estimated USD cost based on Gemini 2.5 Flash pricing constants:

```
📊 Tokens used: 1,234 (prompt: 980, response: 254) — est. ~$0.0001
```

Very small amounts show as `<$0.001`.

---

#### Configurable prompt prefix/suffix ✅
Users can configure a style string that gets automatically prepended/appended to every generate or edit prompt. Set via environment variable or config field.

```
NANO_BANANA_PROMPT_PREFIX="photorealistic, 8k resolution, cinematic lighting, "
NANO_BANANA_PROMPT_SUFFIX=", masterpiece quality"
```

Also configurable in the MCP env block:

```json
{
  "env": {
    "GEMINI_API_KEY": "your-key",
    "NANO_BANANA_PROMPT_PREFIX": "photorealistic, 8k resolution, "
  }
}
```

---

### Architecture

#### Entry point split ✅
`src/index.ts` now exports `NanoBananaMCP`, `Logger`, `classifyApiError`, `ConfigSchema`, and `CONSTANTS` for testability. The server entry point moved to `src/server.ts`. The `bin` field in `package.json` points at `dist/server.js`.

---

### Testing ✅
Full vitest test suite added in `src/test/index.test.ts` with 114 tests covering:

- `Logger` — level filtering, stderr output, metadata serialization
- `classifyApiError` — all error pattern classifications (timeout, rate limit, auth, permissions, model not found, unavailable, server error, generic fallback)
- `ConfigSchema` — Zod validation for all fields and defaults
- `CONSTANTS` — key value assertions
- `handleSearch` — keyword filter, tag filter, verbose mode, no-match message
- `handleExecute` — unknown operation, missing required params, dispatch
- `configure_gemini_token` — config + genAI setup, file save, missing key
- `generate_image` — unconfigured guard, success path (file save + sidecar + session), prompt affixes, cost estimate, no-image response, API error classification
- `edit_image` — unconfigured guard, path traversal block, success path, sidecar with sourceImage, bad reference images (skipped gracefully), lastImagePath update
- `continue_editing` — no session, missing file, delegates to opEditImage
- `delete_image` — outside dir, file not found, deletes image + sidecar, clears session pointer
- `enhance_prompt` — unconfigured guard, TEXT_MODEL usage, style hint, empty response error, API error classification
- `get_configuration_status` — unconfigured, configured (env / file source, prefix/suffix display)
- `get_last_image_info` — no image, file stats + sidecar prompt, deleted file
- `list_generated_images` — no dir, empty dir, sorted list, limit param, sidecar prompt snippet, last-image marker
- Session persistence — saveSession, loadSession (restore, skip missing file, no-op on missing session)
- Sidecar files — sidecarPath, writeImageSidecar (JSON + permissions + failure tolerance), readImageSidecar
- `applyPromptAffix` — prefix, suffix, both, null config
- `buildSuccessResponse` — generate/edit format, token usage + cost, no-image note, description text, reference images

---

## Environment Variables Added in v1.2

| Variable | Default | Description |
|----------|---------|-------------|
| `NANO_BANANA_PROMPT_PREFIX` | `""` | Text prepended to every generate/edit prompt |
| `NANO_BANANA_PROMPT_SUFFIX` | `""` | Text appended to every generate/edit prompt |

These join the existing `GEMINI_API_KEY`, `NANO_BANANA_MODEL`, `NANO_BANANA_OUTPUT_DIR`, `NANO_BANANA_OUTPUT_FORMAT`, `NANO_BANANA_TIMEOUT_MS`, and `NANO_BANANA_LOG_LEVEL`.

---

## Items Implemented in v1.3.0

| Enhancement | Notes |
|-------------|-------|
| `generate_image_batch` | Hard cap at 5, parallel execution, partial successes, aggregated cost |
| `get_image_history` | Walks sidecar chain oldest-first, handles missing files and cycles |

## Items Implemented in v1.2.1

| Enhancement | Notes |
|-------------|-------|
| API key format validation on startup | Warns on wrong prefix or length at every load point; surfaced in configure_gemini_token response |
