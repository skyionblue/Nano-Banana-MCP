import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import path from "path";
import os from "os";

// Hoist mock fn references so vi.mock factories can close over them
const { mockGenerateContent } = vi.hoisted(() => ({
  mockGenerateContent: vi.fn(),
}));

// Use regular functions (not arrow functions) so they work as constructors with `new`
vi.mock("@modelcontextprotocol/sdk/server/index.js", () => ({
  Server: vi.fn(function(this: any) {
    this.setRequestHandler = vi.fn();
    this.connect = vi.fn().mockResolvedValue(undefined);
  }),
}));

vi.mock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
  StdioServerTransport: vi.fn(function(this: any) {}),
}));

vi.mock("@google/genai", () => ({
  GoogleGenAI: vi.fn(function(this: any) {
    this.models = { generateContent: mockGenerateContent };
  }),
}));

vi.mock("fs/promises");

import * as fs from "fs/promises";
import {
  Logger,
  classifyApiError,
  validateApiKeyFormat,
  ConfigSchema,
  CONSTANTS,
  NanoBananaMCP,
} from "../index.js";
import { ErrorCode, McpError } from "@modelcontextprotocol/sdk/types.js";

// Helper: build a fake CallToolRequest
function fakeRequest(toolName: string, args: Record<string, unknown> = {}) {
  return { params: { name: toolName, arguments: args } } as any;
}

// Helper: directly call handleSearch on the server (bypasses MCP dispatch)
function callSearch(server: NanoBananaMCP, args: Record<string, unknown> = {}) {
  return (server as any).handleSearch(fakeRequest("search", args));
}

// Helper: directly call handleExecute
async function callExecute(server: NanoBananaMCP, operation: string, args: Record<string, unknown> = {}) {
  return (server as any).handleExecute(
    fakeRequest("execute", { operation, arguments: args })
  );
}

// Helper: set private config + genAI on server
function configureServer(server: NanoBananaMCP, overrides: Record<string, unknown> = {}) {
  const config = {
    geminiApiKey: "test-key",
    model: CONSTANTS.MODEL,
    outputFormat: "png",
    timeoutMs: CONSTANTS.TIMEOUT_MS,
    promptPrefix: "",
    promptSuffix: "",
    ...overrides,
  };
  (server as any).config = config;
  (server as any).genAI = { models: { generateContent: mockGenerateContent } };
  (server as any).configSource = "environment";
}

// Stub fs methods with sensible defaults (also clears all mock call history)
function stubFsDefaults() {
  vi.clearAllMocks();
  vi.mocked(fs.mkdir).mockResolvedValue(undefined as any);
  vi.mocked(fs.writeFile).mockResolvedValue(undefined as any);
  vi.mocked(fs.readFile).mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" }));
  vi.mocked(fs.access).mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" }));
  vi.mocked(fs.unlink).mockResolvedValue(undefined as any);
  vi.mocked(fs.stat).mockResolvedValue({ size: 102400, mtime: new Date("2025-01-01T00:00:00Z") } as any);
  vi.mocked(fs.readdir).mockResolvedValue([] as any);
}

// -------------------------------------------------------------------
// Logger
// -------------------------------------------------------------------

describe("Logger", () => {
  let stderrSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    stderrSpy.mockRestore();
  });

  it("defaults to WARN level when no override given and env not set", () => {
    const logger = new Logger(undefined);
    expect(logger.levelName).toBe("WARN");
  });

  it("accepts explicit level override", () => {
    expect(new Logger("DEBUG").levelName).toBe("DEBUG");
    expect(new Logger("INFO").levelName).toBe("INFO");
    expect(new Logger("ERROR").levelName).toBe("ERROR");
    expect(new Logger("SILENT").levelName).toBe("SILENT");
  });

  it("falls back to WARN for invalid level names", () => {
    expect(new Logger("VERBOSE").levelName).toBe("WARN");
    expect(new Logger("ALL").levelName).toBe("WARN");
  });

  it("writes to stderr for warn messages when level is WARN", () => {
    const logger = new Logger("WARN");
    logger.warn("test warning");
    expect(stderrSpy).toHaveBeenCalledOnce();
    expect(String(stderrSpy.mock.calls[0][0])).toContain("WARN");
    expect(String(stderrSpy.mock.calls[0][0])).toContain("test warning");
    expect(String(stderrSpy.mock.calls[0][0])).toContain("[nano-banana]");
  });

  it("includes metadata JSON when provided", () => {
    const logger = new Logger("WARN");
    logger.warn("msg", { key: "value" });
    expect(String(stderrSpy.mock.calls[0][0])).toContain('{"key":"value"}');
  });

  it("suppresses debug messages when level is WARN", () => {
    const logger = new Logger("WARN");
    logger.debug("should not appear");
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it("suppresses info messages when level is WARN", () => {
    const logger = new Logger("WARN");
    logger.info("should not appear");
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it("writes debug, info, warn, error when level is DEBUG", () => {
    const logger = new Logger("DEBUG");
    logger.debug("d");
    logger.info("i");
    logger.warn("w");
    logger.error("e");
    expect(stderrSpy).toHaveBeenCalledTimes(4);
  });

  it("suppresses all output when level is SILENT", () => {
    const logger = new Logger("SILENT");
    logger.debug("d");
    logger.info("i");
    logger.warn("w");
    logger.error("e");
    expect(stderrSpy).not.toHaveBeenCalled();
  });

  it("writes error messages when level is ERROR", () => {
    const logger = new Logger("ERROR");
    logger.error("critical failure", { code: 500 });
    expect(stderrSpy).toHaveBeenCalledOnce();
    expect(String(stderrSpy.mock.calls[0][0])).toContain("ERROR");
  });

  it("does not write warn when level is ERROR", () => {
    const logger = new Logger("ERROR");
    logger.warn("not shown");
    expect(stderrSpy).not.toHaveBeenCalled();
  });
});

// -------------------------------------------------------------------
// classifyApiError
// -------------------------------------------------------------------

describe("classifyApiError", () => {
  it("passes through an existing McpError unchanged", () => {
    const original = new McpError(ErrorCode.InvalidParams, "original");
    expect(classifyApiError(original, "op")).toBe(original);
  });

  it("classifies timeout errors", () => {
    const err = classifyApiError(new Error("Request timed out after 120000ms"), "generate_image");
    expect(err.code).toBe(ErrorCode.InternalError);
    expect(err.message).toContain("timed out");
    expect(err.message).toContain("NANO_BANANA_TIMEOUT_MS");
  });

  it("classifies 429 / rate limit errors", () => {
    const err = classifyApiError(new Error("429 Too Many Requests"), "generate_image");
    expect(err.code).toBe(ErrorCode.InternalError);
    expect(err.message).toContain("Rate limit");
  });

  it("classifies resource_exhausted errors", () => {
    const err = classifyApiError(new Error("RESOURCE_EXHAUSTED quota exceeded"), "generate_image");
    expect(err.message).toContain("Rate limit");
  });

  it("classifies 401 / invalid API key errors", () => {
    const err = classifyApiError(new Error("API_KEY_INVALID: invalid api key"), "generate_image");
    expect(err.code).toBe(ErrorCode.InvalidRequest);
    expect(err.message).toContain("Invalid Gemini API key");
    expect(err.message).toContain("configure_gemini_token");
  });

  it("classifies 403 / permission denied errors", () => {
    const err = classifyApiError(new Error("PERMISSION_DENIED: billing not enabled"), "edit_image");
    expect(err.code).toBe(ErrorCode.InvalidRequest);
    expect(err.message).toContain("access denied");
  });

  it("classifies 404 / model not found errors", () => {
    const err = classifyApiError(new Error("NOT_FOUND: model not found"), "generate_image");
    expect(err.code).toBe(ErrorCode.InvalidRequest);
    expect(err.message).toContain("model not found");
    expect(err.message).toContain("NANO_BANANA_MODEL");
  });

  it("classifies 503 / unavailable errors", () => {
    const err = classifyApiError(new Error("503 service unavailable"), "generate_image");
    expect(err.code).toBe(ErrorCode.InternalError);
    expect(err.message).toContain("temporarily unavailable");
  });

  it("classifies 500 / internal server errors", () => {
    const err = classifyApiError(new Error("500 internal_error"), "generate_image");
    expect(err.code).toBe(ErrorCode.InternalError);
    expect(err.message).toContain("server error");
  });

  it("wraps unrecognised errors as generic InternalError", () => {
    const err = classifyApiError(new Error("something weird happened"), "my_op");
    expect(err.code).toBe(ErrorCode.InternalError);
    expect(err.message).toContain("my_op failed");
    expect(err.message).toContain("something weird happened");
  });

  it("handles non-Error objects", () => {
    const err = classifyApiError("plain string error", "op");
    expect(err.message).toContain("plain string error");
  });

  it("handles null", () => {
    const err = classifyApiError(null, "op");
    expect(err).toBeInstanceOf(McpError);
  });
});

// -------------------------------------------------------------------
// ConfigSchema
// -------------------------------------------------------------------

describe("ConfigSchema", () => {
  it("parses a minimal config with defaults applied", () => {
    const result = ConfigSchema.parse({ geminiApiKey: "my-key" });
    expect(result.geminiApiKey).toBe("my-key");
    expect(result.model).toBe(CONSTANTS.MODEL);
    expect(result.outputFormat).toBe("png");
    expect(result.timeoutMs).toBe(CONSTANTS.TIMEOUT_MS);
    expect(result.promptPrefix).toBe("");
    expect(result.promptSuffix).toBe("");
  });

  it("accepts all fields when provided", () => {
    const result = ConfigSchema.parse({
      geminiApiKey: "key",
      model: "gemini-custom",
      outputDir: "/tmp/imgs",
      outputFormat: "webp",
      timeoutMs: 60000,
      promptPrefix: "ultra HD, ",
      promptSuffix: ", masterpiece",
    });
    expect(result.model).toBe("gemini-custom");
    expect(result.outputDir).toBe("/tmp/imgs");
    expect(result.outputFormat).toBe("webp");
    expect(result.timeoutMs).toBe(60000);
    expect(result.promptPrefix).toBe("ultra HD, ");
    expect(result.promptSuffix).toBe(", masterpiece");
  });

  it("rejects empty geminiApiKey", () => {
    expect(() => ConfigSchema.parse({ geminiApiKey: "" })).toThrow();
  });

  it("rejects missing geminiApiKey", () => {
    expect(() => ConfigSchema.parse({})).toThrow();
  });

  it("rejects invalid outputFormat", () => {
    expect(() => ConfigSchema.parse({ geminiApiKey: "k", outputFormat: "bmp" })).toThrow();
  });

  it("rejects non-positive timeoutMs", () => {
    expect(() => ConfigSchema.parse({ geminiApiKey: "k", timeoutMs: -1 })).toThrow();
    expect(() => ConfigSchema.parse({ geminiApiKey: "k", timeoutMs: 0 })).toThrow();
  });
});

// -------------------------------------------------------------------
// CONSTANTS
// -------------------------------------------------------------------

describe("CONSTANTS", () => {
  it("has expected key values", () => {
    expect(CONSTANTS.OUTPUT_FORMAT).toBe("png");
    expect(CONSTANTS.TIMEOUT_MS).toBe(120_000);
    expect(CONSTANTS.RETRY_ATTEMPTS).toBe(3);
    expect(CONSTANTS.RETRY_BASE_DELAY_MS).toBe(1_000);
    expect(CONSTANTS.CONFIG_FILENAME).toBe(".nano-banana-config.json");
    expect(CONSTANTS.SESSION_FILENAME).toBe(".nano-banana-session.json");
    expect(CONSTANTS.PRICING_INPUT_PER_M).toBeGreaterThan(0);
    expect(CONSTANTS.PRICING_OUTPUT_PER_M).toBeGreaterThan(0);
  });
});

// -------------------------------------------------------------------
// NanaBananaMCP — search
// -------------------------------------------------------------------

describe("NanoBananaMCP.handleSearch", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
  });

  it("returns all operations when no query provided", () => {
    const result = callSearch(server);
    const text = result.content[0].text as string;
    expect(text).toContain("generate_image");
    expect(text).toContain("edit_image");
    expect(text).toContain("continue_editing");
    expect(text).toContain("delete_image");
    expect(text).toContain("enhance_prompt");
    expect(text).toContain("configure_gemini_token");
    expect(text).toContain("get_configuration_status");
    expect(text).toContain("get_last_image_info");
    expect(text).toContain("list_generated_images");
  });

  it("filters by operation name keyword", () => {
    const result = callSearch(server, { query: "delete" });
    const text = result.content[0].text as string;
    expect(text).toContain("delete_image");
    expect(text).not.toContain("generate_image");
    expect(text).not.toContain("edit_image");
  });

  it("filters by description keyword", () => {
    const result = callSearch(server, { query: "reference" });
    const text = result.content[0].text as string;
    expect(text).toContain("edit_image");
  });

  it("filters by tag keyword", () => {
    const result = callSearch(server, { query: "config" });
    const text = result.content[0].text as string;
    expect(text).toContain("configure_gemini_token");
    expect(text).toContain("get_configuration_status");
  });

  it("returns no-match message for unknown query", () => {
    const result = callSearch(server, { query: "xyznonexistentabc" });
    const text = result.content[0].text as string;
    expect(text).toContain("No operations matched");
    expect(text).toContain("xyznonexistentabc");
  });

  it("compact mode shows param name and type, not full description", () => {
    const result = callSearch(server, { query: "generate_image", verbose: false });
    const text = result.content[0].text as string;
    expect(text).toContain("prompt (string)");
    expect(text).not.toContain("Text prompt describing");
  });

  it("verbose mode includes full param descriptions", () => {
    const result = callSearch(server, { query: "generate_image", verbose: true });
    const text = result.content[0].text as string;
    expect(text).toContain("Text prompt describing the image to create");
  });

  it("shows required and optional param labels correctly for edit_image", () => {
    const result = callSearch(server, { query: "edit_image" });
    const text = result.content[0].text as string;
    expect(text).toContain("Required:");
    expect(text).toContain("Optional:");
  });

  it("includes execute usage hint at the bottom", () => {
    const result = callSearch(server);
    const text = result.content[0].text as string;
    expect(text).toContain("execute");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — execute dispatch
// -------------------------------------------------------------------

describe("NanoBananaMCP.handleExecute", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    stubFsDefaults();
    vi.clearAllMocks();
  });

  it("throws InvalidParams for unknown operation", async () => {
    await expect(callExecute(server, "nonexistent_op")).rejects.toMatchObject({
      code: ErrorCode.InvalidParams,
    });
  });

  it("lists available operations in the error message", async () => {
    await expect(callExecute(server, "bad_op")).rejects.toMatchObject({
      message: expect.stringContaining("generate_image"),
    });
  });

  it("throws InvalidParams when required params are missing", async () => {
    configureServer(server);
    await expect(callExecute(server, "generate_image", {})).rejects.toMatchObject({
      code: ErrorCode.InvalidParams,
    });
  });

  it("names the missing param in the error message", async () => {
    configureServer(server);
    await expect(callExecute(server, "generate_image", {})).rejects.toMatchObject({
      message: expect.stringContaining("prompt"),
    });
  });

  it("dispatches to the correct operation handler", async () => {
    configureServer(server);
    const spy = vi.spyOn(server as any, "opGetConfigurationStatus");
    spy.mockResolvedValue({ content: [{ type: "text", text: "ok" }] });
    const result = await callExecute(server, "get_configuration_status");
    expect(spy).toHaveBeenCalledOnce();
    expect(result.content[0].text).toBe("ok");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — configure_gemini_token
// -------------------------------------------------------------------

describe("NanoBananaMCP configure_gemini_token", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    stubFsDefaults();
  });

  it("sets config and genAI and returns success message", async () => {
    const validKey = "AIza" + "A".repeat(35);
    const result = await callExecute(server, "configure_gemini_token", { apiKey: validKey });
    expect(result.content[0].text).toContain("configured successfully");
    expect((server as any).config).not.toBeNull();
    expect((server as any).genAI).not.toBeNull();
  });

  it("saves config to file with restricted permissions", async () => {
    await callExecute(server, "configure_gemini_token", { apiKey: "AIza" + "A".repeat(35) });
    expect(fs.writeFile).toHaveBeenCalledWith(
      expect.stringContaining(".nano-banana-config.json"),
      expect.any(String),
      expect.objectContaining({ mode: 0o600 })
    );
  });

  it("throws InvalidParams when apiKey is missing", async () => {
    await expect(callExecute(server, "configure_gemini_token", {})).rejects.toMatchObject({
      code: ErrorCode.InvalidParams,
    });
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — generate_image
// -------------------------------------------------------------------

describe("NanoBananaMCP generate_image", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
  });

  it("throws InvalidRequest when not configured", async () => {
    const unconfigured = new NanoBananaMCP();
    await expect(callExecute(unconfigured, "generate_image", { prompt: "a cat" })).rejects.toMatchObject({
      code: ErrorCode.InvalidRequest,
    });
  });

  it("returns success with image content and saved file path", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{
        content: {
          parts: [
            { text: "Here is your image." },
            { inlineData: { data: "base64imagedata", mimeType: "image/png" } },
          ],
        },
      }],
      usageMetadata: { totalTokenCount: 500, promptTokenCount: 400, candidatesTokenCount: 100 },
    });

    const result = await callExecute(server, "generate_image", { prompt: "a cat" });
    const texts = result.content.filter((c: any) => c.type === "text").map((c: any) => c.text).join("\n");
    expect(texts).toContain("Image generated with nano-banana");
    expect(texts).toContain("a cat");
    expect(texts).toContain("saved to");
    expect(texts).toContain("Tokens used: 500");
  });

  it("writes the image file to disk", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc123", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 100, promptTokenCount: 80, candidatesTokenCount: 20 },
    });

    await callExecute(server, "generate_image", { prompt: "a dog" });
    expect(fs.writeFile).toHaveBeenCalledWith(
      expect.stringContaining("generated-"),
      expect.any(Buffer)
    );
  });

  it("writes a sidecar JSON file alongside the image", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 100, promptTokenCount: 80, candidatesTokenCount: 20 },
    });

    await callExecute(server, "generate_image", { prompt: "sunset" });
    const writeFileCalls = vi.mocked(fs.writeFile).mock.calls;
    const jsonCall = writeFileCalls.find(([p]) => String(p).endsWith(".json") && !String(p).endsWith("session.json"));
    expect(jsonCall).toBeDefined();
    const sidecarContent = JSON.parse(String(jsonCall![1]));
    expect(sidecarContent.operation).toBe("generate");
    expect(sidecarContent.prompt).toBe("sunset");
    expect(sidecarContent.model).toBe(CONSTANTS.MODEL);
  });

  it("saves session file after generating image", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    await callExecute(server, "generate_image", { prompt: "trees" });
    const sessionCall = vi.mocked(fs.writeFile).mock.calls.find(([p]) =>
      String(p).endsWith(".nano-banana-session.json")
    );
    expect(sessionCall).toBeDefined();
  });

  it("handles response with no image parts gracefully", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ text: "I cannot generate that." }] } }],
      usageMetadata: { totalTokenCount: 10, promptTokenCount: 9, candidatesTokenCount: 1 },
    });

    const result = await callExecute(server, "generate_image", { prompt: "something" });
    expect(result.content[0].text).toContain("No image was returned");
  });

  it("applies prompt prefix and suffix before calling the API", async () => {
    configureServer(server, { promptPrefix: "HD, ", promptSuffix: ", cinematic" });
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [] } }],
      usageMetadata: { totalTokenCount: 0, promptTokenCount: 0, candidatesTokenCount: 0 },
    });

    await callExecute(server, "generate_image", { prompt: "mountains" });
    const calledWith = mockGenerateContent.mock.calls[0][0];
    expect(calledWith.contents).toContain("HD, mountains, cinematic");
  });

  it("includes cost estimate in response for large token usage", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 1_000_000, promptTokenCount: 800_000, candidatesTokenCount: 200_000 },
    });

    const result = await callExecute(server, "generate_image", { prompt: "big prompt" });
    const text = result.content[0].text as string;
    expect(text).toContain("est. ~$");
  });

  it("throws classified error on API failure", async () => {
    mockGenerateContent.mockRejectedValue(new Error("429 Too Many Requests"));
    await expect(callExecute(server, "generate_image", { prompt: "a cat" })).rejects.toMatchObject({
      code: ErrorCode.InternalError,
    });
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — edit_image
// -------------------------------------------------------------------

describe("NanoBananaMCP edit_image", () => {
  let server: NanoBananaMCP;
  const fakeImagePath = path.join(os.homedir(), "test-image.png");

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith(".json")) throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      return Buffer.from("fake image bytes");
    });
  });

  it("throws InvalidRequest when not configured", async () => {
    const unconfigured = new NanoBananaMCP();
    await expect(
      callExecute(unconfigured, "edit_image", { imagePath: fakeImagePath, prompt: "edit" })
    ).rejects.toMatchObject({ code: ErrorCode.InvalidRequest });
  });

  it("throws InvalidParams for paths outside allowed directories", async () => {
    await expect(
      callExecute(server, "edit_image", { imagePath: "/etc/passwd", prompt: "edit" })
    ).rejects.toMatchObject({ code: ErrorCode.InvalidParams });
  });

  it("returns success with edited image content", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "editeddata", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 300, promptTokenCount: 250, candidatesTokenCount: 50 },
    });

    const result = await callExecute(server, "edit_image", { imagePath: fakeImagePath, prompt: "add clouds" });
    const text = result.content.find((c: any) => c.type === "text")?.text as string;
    expect(text).toContain("Image edited with nano-banana");
    expect(text).toContain("add clouds");
  });

  it("writes sidecar with sourceImage reference for edits", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "editeddata", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 100, promptTokenCount: 80, candidatesTokenCount: 20 },
    });

    await callExecute(server, "edit_image", { imagePath: fakeImagePath, prompt: "edit" });

    const writeFileCalls = vi.mocked(fs.writeFile).mock.calls;
    const jsonCall = writeFileCalls.find(([p]) => String(p).endsWith(".json") && !String(p).endsWith("session.json"));
    const sidecar = JSON.parse(String(jsonCall![1]));
    expect(sidecar.operation).toBe("edit");
    expect(sidecar.sourceImage).toBe(fakeImagePath);
  });

  it("skips unreadable reference images without throwing", async () => {
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).includes("bad-ref")) throw new Error("ENOENT");
      return Buffer.from("image bytes");
    });
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "d", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    const badRef = path.join(os.homedir(), "bad-ref.png");
    await expect(
      callExecute(server, "edit_image", {
        imagePath: fakeImagePath,
        prompt: "style it",
        referenceImages: [badRef],
      })
    ).resolves.toBeDefined();
  });

  it("updates lastImagePath after successful edit", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "d", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    await callExecute(server, "edit_image", { imagePath: fakeImagePath, prompt: "edit" });
    expect((server as any).lastImagePath).toMatch(/edited-.*\.png$/);
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — continue_editing
// -------------------------------------------------------------------

describe("NanoBananaMCP continue_editing", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
  });

  it("throws InvalidRequest when no previous image exists in session", async () => {
    await expect(callExecute(server, "continue_editing", { prompt: "adjust" })).rejects.toMatchObject({
      code: ErrorCode.InvalidRequest,
    });
  });

  it("throws InvalidRequest when the last image file no longer exists", async () => {
    (server as any).lastImagePath = path.join(os.homedir(), "missing.png");
    vi.mocked(fs.access).mockRejectedValue(new Error("ENOENT"));

    await expect(callExecute(server, "continue_editing", { prompt: "adjust" })).rejects.toMatchObject({
      code: ErrorCode.InvalidRequest,
    });
  });

  it("delegates to opEditImage when last image is accessible", async () => {
    const imgPath = path.join(os.homedir(), "last.png");
    (server as any).lastImagePath = imgPath;
    vi.mocked(fs.access).mockResolvedValue(undefined as any);

    const editSpy = vi.spyOn(server as any, "opEditImage").mockResolvedValue({
      content: [{ type: "text", text: "edited" }],
    });

    await callExecute(server, "continue_editing", { prompt: "more contrast" });
    expect(editSpy).toHaveBeenCalledWith({
      imagePath: imgPath,
      prompt: "more contrast",
      referenceImages: undefined,
    });
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — delete_image
// -------------------------------------------------------------------

describe("NanoBananaMCP delete_image", () => {
  let server: NanoBananaMCP;
  let imagesDir: string;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
    imagesDir = (server as any).getImagesDirectory();
  });

  it("throws InvalidParams when image path is outside the output directory", async () => {
    await expect(
      callExecute(server, "delete_image", { imagePath: "/etc/passwd" })
    ).rejects.toMatchObject({ code: ErrorCode.InvalidParams });
  });

  it("throws InvalidParams when the file does not exist", async () => {
    const imgPath = path.join(imagesDir, "missing.png");
    vi.mocked(fs.access).mockRejectedValue(new Error("ENOENT"));

    await expect(
      callExecute(server, "delete_image", { imagePath: imgPath })
    ).rejects.toMatchObject({ code: ErrorCode.InvalidParams });
  });

  it("deletes the image file", async () => {
    const imgPath = path.join(imagesDir, "to-delete.png");
    vi.mocked(fs.access).mockResolvedValue(undefined as any);

    await callExecute(server, "delete_image", { imagePath: imgPath });
    expect(fs.unlink).toHaveBeenCalledWith(path.resolve(imgPath));
  });

  it("also deletes the sidecar JSON when it exists", async () => {
    const imgPath = path.join(imagesDir, "to-delete.png");
    vi.mocked(fs.access).mockResolvedValue(undefined as any);

    await callExecute(server, "delete_image", { imagePath: imgPath });
    const deletedPaths = vi.mocked(fs.unlink).mock.calls.map(([p]) => String(p));
    expect(deletedPaths.some(p => p.endsWith(".json"))).toBe(true);
  });

  it("clears lastImagePath and saves session when deleting the current last image", async () => {
    const imgPath = path.join(imagesDir, "last.png");
    (server as any).lastImagePath = path.resolve(imgPath);
    vi.mocked(fs.access).mockResolvedValue(undefined as any);

    const result = await callExecute(server, "delete_image", { imagePath: imgPath });
    expect((server as any).lastImagePath).toBeNull();
    expect(result.content[0].text).toContain("session pointer has been cleared");
    const sessionCall = vi.mocked(fs.writeFile).mock.calls.find(([p]) =>
      String(p).endsWith(".nano-banana-session.json")
    );
    expect(sessionCall).toBeDefined();
  });

  it("returns success message with the deleted path", async () => {
    const imgPath = path.join(imagesDir, "test.png");
    vi.mocked(fs.access).mockResolvedValue(undefined as any);

    const result = await callExecute(server, "delete_image", { imagePath: imgPath });
    expect(result.content[0].text).toContain("Deleted:");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — enhance_prompt
// -------------------------------------------------------------------

describe("NanoBananaMCP enhance_prompt", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
  });

  it("throws InvalidRequest when not configured", async () => {
    const unconfigured = new NanoBananaMCP();
    await expect(
      callExecute(unconfigured, "enhance_prompt", { prompt: "a cat" })
    ).rejects.toMatchObject({ code: ErrorCode.InvalidRequest });
  });

  it("returns the enhanced prompt text from Gemini", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{
        content: { parts: [{ text: "A photorealistic close-up of a tabby cat with amber eyes, soft natural lighting, 8k." }] },
      }],
    });

    const result = await callExecute(server, "enhance_prompt", { prompt: "a cat" });
    expect(result.content[0].text).toContain("Enhanced prompt:");
    expect(result.content[0].text).toContain("photorealistic");
    expect(result.content[0].text).toContain("generate_image");
  });

  it("calls the TEXT_MODEL, not the image model", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ text: "Enhanced: detailed cat portrait" }] } }],
    });

    await callExecute(server, "enhance_prompt", { prompt: "a cat" });
    expect(mockGenerateContent).toHaveBeenCalledWith(
      expect.objectContaining({ model: CONSTANTS.TEXT_MODEL })
    );
  });

  it("includes style hint in the instruction when provided", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ text: "An oil painting of..." }] } }],
    });

    await callExecute(server, "enhance_prompt", { prompt: "a landscape", style: "oil painting" });
    const calledContents = mockGenerateContent.mock.calls[0][0].contents as string;
    expect(calledContents).toContain("oil painting");
  });

  it("throws InternalError when Gemini returns no text", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [] } }],
    });

    await expect(
      callExecute(server, "enhance_prompt", { prompt: "empty" })
    ).rejects.toMatchObject({ code: ErrorCode.InternalError });
  });

  it("classifies API errors (e.g. auth failure)", async () => {
    mockGenerateContent.mockRejectedValue(new Error("401 unauthenticated"));
    await expect(
      callExecute(server, "enhance_prompt", { prompt: "test" })
    ).rejects.toMatchObject({ code: ErrorCode.InvalidRequest });
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — get_configuration_status
// -------------------------------------------------------------------

describe("NanoBananaMCP get_configuration_status", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
  });

  it("returns not-configured message when no key set", async () => {
    const result = await callExecute(server, "get_configuration_status", {});
    expect(result.content[0].text).toContain("not configured");
    expect(result.content[0].text).toContain("configure_gemini_token");
  });

  it("returns configured status with active settings when key is set", async () => {
    configureServer(server);
    const result = await callExecute(server, "get_configuration_status", {});
    const text = result.content[0].text as string;
    expect(text).toContain("configured and ready");
    expect(text).toContain(CONSTANTS.MODEL);
    expect(text).toContain("120s");
  });

  it("shows environment source when configured from env", async () => {
    configureServer(server);
    (server as any).configSource = "environment";
    const result = await callExecute(server, "get_configuration_status", {});
    expect(result.content[0].text).toContain("Environment variable");
  });

  it("shows file source when configured from config file", async () => {
    configureServer(server);
    (server as any).configSource = "config_file";
    const result = await callExecute(server, "get_configuration_status", {});
    expect(result.content[0].text).toContain("Local config file");
  });

  it("shows prompt prefix and suffix when configured", async () => {
    configureServer(server, { promptPrefix: "HD, ", promptSuffix: ", masterpiece" });
    const result = await callExecute(server, "get_configuration_status", {});
    const text = result.content[0].text as string;
    expect(text).toContain("HD, ");
    expect(text).toContain(", masterpiece");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — get_last_image_info
// -------------------------------------------------------------------

describe("NanoBananaMCP get_last_image_info", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
  });

  it("returns a no-image message when session has no last image", async () => {
    const result = await callExecute(server, "get_last_image_info", {});
    expect(result.content[0].text).toContain("No image has been generated");
  });

  it("returns file path, size, and mtime when last image exists", async () => {
    const imgPath = path.join(os.homedir(), "last.png");
    (server as any).lastImagePath = imgPath;
    vi.mocked(fs.stat).mockResolvedValue({ size: 204800, mtime: new Date("2025-06-01T10:00:00Z") } as any);
    vi.mocked(fs.readFile).mockRejectedValue(new Error("no sidecar"));

    const result = await callExecute(server, "get_last_image_info", {});
    const text = result.content[0].text as string;
    expect(text).toContain("200 KB");
    expect(text).toContain(imgPath);
  });

  it("shows prompt from sidecar when sidecar is available", async () => {
    const imgPath = path.join(os.homedir(), "last.png");
    (server as any).lastImagePath = imgPath;
    vi.mocked(fs.stat).mockResolvedValue({ size: 1024, mtime: new Date() } as any);
    vi.mocked(fs.readFile).mockResolvedValue(
      JSON.stringify({ operation: "generate", prompt: "a sunset", model: CONSTANTS.MODEL, timestamp: new Date().toISOString() })
    );

    const result = await callExecute(server, "get_last_image_info", {});
    expect(result.content[0].text).toContain("a sunset");
  });

  it("returns a graceful message when the file has been deleted", async () => {
    const imgPath = path.join(os.homedir(), "gone.png");
    (server as any).lastImagePath = imgPath;
    vi.mocked(fs.stat).mockRejectedValue(new Error("ENOENT"));

    const result = await callExecute(server, "get_last_image_info", {});
    expect(result.content[0].text).toContain("file not found");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — list_generated_images
// -------------------------------------------------------------------

describe("NanoBananaMCP list_generated_images", () => {
  let server: NanoBananaMCP;
  let imagesDir: string;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
    imagesDir = (server as any).getImagesDirectory();
  });

  it("returns no-directory message when output dir doesn't exist", async () => {
    vi.mocked(fs.access).mockRejectedValue(new Error("ENOENT"));
    const result = await callExecute(server, "list_generated_images", {});
    expect(result.content[0].text).toContain("No output directory found");
  });

  it("returns no-images message when directory is empty", async () => {
    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readdir).mockResolvedValue([] as any);

    const result = await callExecute(server, "list_generated_images", {});
    expect(result.content[0].text).toContain("No images found");
  });

  it("lists PNG/JPEG/WEBP images sorted newest first, excludes other files", async () => {
    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readdir).mockResolvedValue([
      { name: "generated-2025-01-01.png", isFile: () => true } as any,
      { name: "generated-2025-06-01.png", isFile: () => true } as any,
      { name: "README.txt", isFile: () => true } as any,
    ]);
    vi.mocked(fs.stat).mockResolvedValue({ size: 51200, mtime: new Date("2025-06-01") } as any);
    vi.mocked(fs.readFile).mockRejectedValue(new Error("no sidecar"));

    const result = await callExecute(server, "list_generated_images", {});
    const text = result.content[0].text as string;
    expect(text).toContain("generated-2025-06-01.png");
    expect(text).not.toContain("README.txt");
    const idx06 = text.indexOf("2025-06");
    const idx01 = text.indexOf("2025-01");
    expect(idx06).toBeLessThan(idx01);
  });

  it("respects the limit parameter", async () => {
    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readdir).mockResolvedValue(
      Array.from({ length: 10 }, (_, i) => ({
        name: `generated-2025-0${String(i + 1).padStart(2, "0")}-01.png`,
        isFile: () => true,
      })) as any
    );
    vi.mocked(fs.stat).mockResolvedValue({ size: 1024, mtime: new Date() } as any);
    vi.mocked(fs.readFile).mockRejectedValue(new Error("no sidecar"));

    const result = await callExecute(server, "list_generated_images", { limit: 3 });
    expect(result.content[0].text).toContain("Showing 3 of 10");
  });

  it("shows prompt snippet from sidecar when available", async () => {
    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readdir).mockResolvedValue([
      { name: "generated-img.png", isFile: () => true } as any,
    ]);
    vi.mocked(fs.stat).mockResolvedValue({ size: 1024, mtime: new Date() } as any);
    vi.mocked(fs.readFile).mockResolvedValue(
      JSON.stringify({ operation: "generate", prompt: "a sunset over mountains", model: CONSTANTS.MODEL, timestamp: new Date().toISOString() })
    );

    const result = await callExecute(server, "list_generated_images", {});
    expect(result.content[0].text).toContain("a sunset over mountains");
  });

  it("marks the last session image with an indicator", async () => {
    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readdir).mockResolvedValue([
      { name: "last.png", isFile: () => true } as any,
    ]);
    vi.mocked(fs.stat).mockResolvedValue({ size: 1024, mtime: new Date() } as any);
    vi.mocked(fs.readFile).mockRejectedValue(new Error("no sidecar"));
    (server as any).lastImagePath = path.join(imagesDir, "last.png");

    const result = await callExecute(server, "list_generated_images", {});
    expect(result.content[0].text).toContain("← last");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — session persistence
// -------------------------------------------------------------------

describe("NanoBananaMCP session persistence", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    stubFsDefaults();
  });

  it("saveSession writes lastImagePath to the session file with restricted permissions", async () => {
    (server as any).lastImagePath = "/home/user/image.png";
    await (server as any).saveSession();
    expect(fs.writeFile).toHaveBeenCalledWith(
      expect.stringContaining(".nano-banana-session.json"),
      expect.stringContaining("/home/user/image.png"),
      expect.objectContaining({ mode: 0o600 })
    );
  });

  it("loadSession restores lastImagePath when the session file and image both exist", async () => {
    vi.mocked(fs.readFile).mockResolvedValueOnce(
      JSON.stringify({ lastImagePath: "/home/user/image.png" })
    );
    vi.mocked(fs.access).mockResolvedValue(undefined as any);

    await (server as any).loadSession();
    expect((server as any).lastImagePath).toBe("/home/user/image.png");
  });

  it("loadSession skips restore when the saved image path no longer exists on disk", async () => {
    vi.mocked(fs.readFile).mockResolvedValueOnce(
      JSON.stringify({ lastImagePath: "/home/user/gone.png" })
    );
    vi.mocked(fs.access).mockRejectedValue(new Error("ENOENT"));

    await (server as any).loadSession();
    expect((server as any).lastImagePath).toBeNull();
  });

  it("loadSession is a no-op when no session file exists", async () => {
    vi.mocked(fs.readFile).mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" }));
    await expect((server as any).loadSession()).resolves.not.toThrow();
    expect((server as any).lastImagePath).toBeNull();
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — sidecar files
// -------------------------------------------------------------------

describe("NanoBananaMCP sidecar files", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    stubFsDefaults();
  });

  it("sidecarPath replaces image extension with .json", () => {
    expect((server as any).sidecarPath("/images/generated-123.png")).toBe("/images/generated-123.json");
    expect((server as any).sidecarPath("/images/edited-abc.jpg")).toBe("/images/edited-abc.json");
  });

  it("writeImageSidecar writes valid JSON with restricted permissions", async () => {
    const sidecar = { operation: "generate" as const, prompt: "test", model: "m", timestamp: "t" };
    await (server as any).writeImageSidecar("/images/img.png", sidecar);
    expect(fs.writeFile).toHaveBeenCalledWith(
      "/images/img.json",
      expect.stringContaining('"prompt": "test"'),
      expect.objectContaining({ mode: 0o600 })
    );
  });

  it("writeImageSidecar does not throw when the write fails", async () => {
    vi.mocked(fs.writeFile).mockRejectedValue(new Error("EPERM"));
    await expect(
      (server as any).writeImageSidecar("/images/img.png", {
        operation: "generate", prompt: "p", model: "m", timestamp: ""
      })
    ).resolves.not.toThrow();
  });

  it("readImageSidecar returns parsed object when file exists", async () => {
    const data = { operation: "generate", prompt: "sunset", model: "m", timestamp: "" };
    vi.mocked(fs.readFile).mockResolvedValueOnce(JSON.stringify(data));
    const result = await (server as any).readImageSidecar("/images/img.png");
    expect(result).toEqual(data);
  });

  it("readImageSidecar returns null when file does not exist", async () => {
    vi.mocked(fs.readFile).mockRejectedValue(new Error("ENOENT"));
    const result = await (server as any).readImageSidecar("/images/img.png");
    expect(result).toBeNull();
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — applyPromptAffix
// -------------------------------------------------------------------

describe("NanoBananaMCP applyPromptAffix", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
  });

  it("returns original prompt unchanged when no prefix or suffix configured", () => {
    configureServer(server, { promptPrefix: "", promptSuffix: "" });
    expect((server as any).applyPromptAffix("mountains")).toBe("mountains");
  });

  it("prepends prefix to prompt", () => {
    configureServer(server, { promptPrefix: "HD, ", promptSuffix: "" });
    expect((server as any).applyPromptAffix("mountains")).toBe("HD, mountains");
  });

  it("appends suffix to prompt", () => {
    configureServer(server, { promptPrefix: "", promptSuffix: ", cinematic" });
    expect((server as any).applyPromptAffix("mountains")).toBe("mountains, cinematic");
  });

  it("applies both prefix and suffix", () => {
    configureServer(server, { promptPrefix: "ultra HD, ", promptSuffix: ", 8k resolution" });
    expect((server as any).applyPromptAffix("sunset")).toBe("ultra HD, sunset, 8k resolution");
  });

  it("handles null config safely by returning the original prompt", () => {
    (server as any).config = null;
    expect((server as any).applyPromptAffix("test")).toBe("test");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — buildSuccessResponse
// -------------------------------------------------------------------

describe("NanoBananaMCP buildSuccessResponse", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
  });

  it("formats a generate response with file path and continue-editing hint", () => {
    const text = (server as any).buildSuccessResponse({
      operation: "generated",
      prompt: "a mountain",
      savedFiles: ["/images/generated-123.png"],
      textContent: "",
    });
    expect(text).toContain("Image generated with nano-banana");
    expect(text).toContain("a mountain");
    expect(text).toContain("/images/generated-123.png");
    expect(text).toContain("continue_editing");
  });

  it("formats an edit response with original image path", () => {
    const text = (server as any).buildSuccessResponse({
      operation: "edited",
      prompt: "add clouds",
      savedFiles: ["/images/edited-456.png"],
      textContent: "",
      originalPath: "/images/original.png",
    });
    expect(text).toContain("Image edited with nano-banana");
    expect(text).toContain("/images/original.png");
    expect(text).toContain("add clouds");
  });

  it("includes token counts and cost estimate", () => {
    const text = (server as any).buildSuccessResponse({
      operation: "generated",
      prompt: "test",
      savedFiles: [],
      textContent: "",
      tokenUsage: { total: 1000, prompt: 800, response: 200 },
    });
    expect(text).toContain("Tokens used: 1,000");
    expect(text).toContain("prompt: 800");
    expect(text).toContain("response: 200");
    expect(text).toContain("est.");
  });

  it("omits the token line entirely when no usage data is provided", () => {
    const text = (server as any).buildSuccessResponse({
      operation: "generated",
      prompt: "test",
      savedFiles: [],
      textContent: "",
    });
    expect(text).not.toContain("Tokens used");
  });

  it("shows '<$0.001' for very small estimated costs", () => {
    const text = (server as any).buildSuccessResponse({
      operation: "generated",
      prompt: "test",
      savedFiles: [],
      textContent: "",
      tokenUsage: { total: 10, prompt: 9, response: 1 },
    });
    expect(text).toContain("<$0.001");
  });

  it("shows no-image note when savedFiles is empty", () => {
    const text = (server as any).buildSuccessResponse({
      operation: "generated",
      prompt: "test",
      savedFiles: [],
      textContent: "",
    });
    expect(text).toContain("No image was returned");
  });

  it("includes the description text when provided", () => {
    const text = (server as any).buildSuccessResponse({
      operation: "generated",
      prompt: "test",
      savedFiles: ["/images/img.png"],
      textContent: "This image shows a mountain.",
    });
    expect(text).toContain("This image shows a mountain.");
  });

  it("lists reference images when provided", () => {
    const text = (server as any).buildSuccessResponse({
      operation: "edited",
      prompt: "style it",
      savedFiles: ["/images/out.png"],
      textContent: "",
      referenceImages: ["/ref/style.jpg"],
    });
    expect(text).toContain("Reference images used");
    expect(text).toContain("/ref/style.jpg");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — generateImageCore (internal)
// -------------------------------------------------------------------

describe("NanoBananaMCP generateImageCore", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
  });

  it("returns savedPaths and imageContent on success", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 100, promptTokenCount: 80, candidatesTokenCount: 20 },
    });

    const result = await (server as any).generateImageCore("a cat");
    expect(result.savedPaths).toHaveLength(1);
    expect(result.savedPaths[0]).toMatch(/generated-.*\.png$/);
    expect(result.imageContent).toHaveLength(1);
    expect(result.imageContent[0].type).toBe("image");
    expect(result.tokenUsage?.total).toBe(100);
  });

  it("returns empty arrays when no image is returned", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ text: "I cannot generate that." }] } }],
      usageMetadata: { totalTokenCount: 10, promptTokenCount: 9, candidatesTokenCount: 1 },
    });

    const result = await (server as any).generateImageCore("something");
    expect(result.savedPaths).toHaveLength(0);
    expect(result.imageContent).toHaveLength(0);
    expect(result.textContent).toContain("cannot generate");
  });

  it("does NOT update lastImagePath (that is the caller's responsibility)", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    (server as any).lastImagePath = null;
    await (server as any).generateImageCore("test");
    expect((server as any).lastImagePath).toBeNull();
  });

  it("writes a sidecar for each saved image", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    await (server as any).generateImageCore("sunset");
    const sidecarCall = vi.mocked(fs.writeFile).mock.calls.find(([p]) =>
      String(p).endsWith(".json") && !String(p).endsWith("session.json")
    );
    expect(sidecarCall).toBeDefined();
    const sidecar = JSON.parse(String(sidecarCall![1]));
    expect(sidecar.operation).toBe("generate");
    expect(sidecar.prompt).toBe("sunset");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — generate_image_batch
// -------------------------------------------------------------------

describe("NanoBananaMCP generate_image_batch", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
  });

  it("throws when not configured", async () => {
    const unconfigured = new NanoBananaMCP();
    await expect(
      callExecute(unconfigured, "generate_image_batch", { prompt: "test" })
    ).rejects.toMatchObject({ code: ErrorCode.InvalidRequest });
  });

  it("generates the requested number of images in parallel", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 100, promptTokenCount: 80, candidatesTokenCount: 20 },
    });

    const result = await callExecute(server, "generate_image_batch", { prompt: "sunset", count: 3 });
    expect(mockGenerateContent).toHaveBeenCalledTimes(3);
    expect(result.content[0].text).toContain("3/3");
  });

  it("defaults count to 2 when not specified", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    await callExecute(server, "generate_image_batch", { prompt: "test" });
    expect(mockGenerateContent).toHaveBeenCalledTimes(2);
  });

  it(`caps count at BATCH_MAX_COUNT (${CONSTANTS.BATCH_MAX_COUNT})`, async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    await callExecute(server, "generate_image_batch", { prompt: "test", count: 99 });
    expect(mockGenerateContent).toHaveBeenCalledTimes(CONSTANTS.BATCH_MAX_COUNT);
  });

  it("sets lastImagePath to the first successful result", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    await callExecute(server, "generate_image_batch", { prompt: "test", count: 3 });
    expect((server as any).lastImagePath).toMatch(/generated-.*\.png$/);
  });

  it("saves session after batch completes", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    await callExecute(server, "generate_image_batch", { prompt: "test", count: 2 });
    const sessionCall = vi.mocked(fs.writeFile).mock.calls.find(([p]) =>
      String(p).endsWith(".nano-banana-session.json")
    );
    expect(sessionCall).toBeDefined();
  });

  it("returns partial successes when some images fail", async () => {
    // Spy on generateImageCore directly to avoid retry complexity
    let callCount = 0;
    vi.spyOn(server as any, "generateImageCore").mockImplementation(async () => {
      callCount++;
      if (callCount === 2) throw new Error("503 service unavailable");
      return {
        savedPaths: [path.join(os.homedir(), `img${callCount}.png`)],
        imageContent: [],
        textContent: "",
        tokenUsage: { total: 50, prompt: 40, response: 10 },
      };
    });

    const result = await callExecute(server, "generate_image_batch", { prompt: "test", count: 3 });
    const text = result.content[0].text as string;
    expect(text).toContain("2/3");
    expect(text).toContain("failed");
    expect(text).toContain("⚠️");
  });

  it("throws when ALL images fail", async () => {
    mockGenerateContent.mockRejectedValue(new Error("401 unauthenticated"));
    await expect(
      callExecute(server, "generate_image_batch", { prompt: "test", count: 2 })
    ).rejects.toMatchObject({ code: ErrorCode.InternalError });
  });

  it("includes aggregated token usage and cost in response", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 500_000, promptTokenCount: 400_000, candidatesTokenCount: 100_000 },
    });

    const result = await callExecute(server, "generate_image_batch", { prompt: "test", count: 2 });
    const text = result.content[0].text as string;
    expect(text).toContain("Total tokens:");
    expect(text).toContain("est.");
  });

  it("includes all image content parts in the response", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    const result = await callExecute(server, "generate_image_batch", { prompt: "test", count: 3 });
    const images = result.content.filter((c: any) => c.type === "image");
    expect(images).toHaveLength(3);
  });

  it("lists all saved file paths in the response", async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ inlineData: { data: "abc", mimeType: "image/png" } }] } }],
      usageMetadata: { totalTokenCount: 50, promptTokenCount: 40, candidatesTokenCount: 10 },
    });

    const result = await callExecute(server, "generate_image_batch", { prompt: "waves", count: 2 });
    const text = result.content[0].text as string;
    const savedPaths = text.match(/- .*generated-.*\.png/g);
    expect(savedPaths).toHaveLength(2);
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — get_image_history
// -------------------------------------------------------------------

describe("NanoBananaMCP get_image_history", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
  });

  it("returns a no-sidecar message for an image with no metadata", async () => {
    const imgPath = path.join(os.homedir(), "img.png");
    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readFile).mockRejectedValue(new Error("ENOENT"));

    const result = await callExecute(server, "get_image_history", { imagePath: imgPath });
    expect(result.content[0].text).toContain("No history found");
    expect(result.content[0].text).toContain("no metadata sidecar");
  });

  it("returns a single-entry chain for an original image with no sourceImage", async () => {
    const imgPath = path.join(os.homedir(), "generated.png");
    const sidecar = { operation: "generate", prompt: "a sunset", model: CONSTANTS.MODEL, timestamp: "2025-01-01T00:00:00.000Z" };
    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readFile).mockResolvedValue(JSON.stringify(sidecar));

    const result = await callExecute(server, "get_image_history", { imagePath: imgPath });
    const text = result.content[0].text as string;
    expect(text).toContain("1 image(s) in chain");
    expect(text).toContain("[Original]");
    expect(text).toContain("← current");
    expect(text).toContain("a sunset");
  });

  it("walks a multi-step edit chain and shows oldest first", async () => {
    const original = path.join(os.homedir(), "generated.png");
    const edit1 = path.join(os.homedir(), "edited-1.png");
    const edit2 = path.join(os.homedir(), "edited-2.png");

    const sidecarFor = (p: string): object => {
      if (p.endsWith("edited-2.json")) return { operation: "edit", prompt: "add clouds", model: CONSTANTS.MODEL, timestamp: "2025-01-03T00:00:00.000Z", sourceImage: edit1 };
      if (p.endsWith("edited-1.json")) return { operation: "edit", prompt: "add birds", model: CONSTANTS.MODEL, timestamp: "2025-01-02T00:00:00.000Z", sourceImage: original };
      if (p.endsWith("generated.json")) return { operation: "generate", prompt: "a sunset", model: CONSTANTS.MODEL, timestamp: "2025-01-01T00:00:00.000Z" };
      throw new Error("ENOENT");
    };

    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => JSON.stringify(sidecarFor(String(p))));

    const result = await callExecute(server, "get_image_history", { imagePath: edit2 });
    const text = result.content[0].text as string;
    expect(text).toContain("3 image(s) in chain");
    expect(text).toContain("[Original]");
    expect(text).toContain("[Edit 1]");
    expect(text).toContain("[Edit 2]");
    expect(text).toContain("← current");
    // Oldest first: original before edits
    expect(text.indexOf("[Original]")).toBeLessThan(text.indexOf("[Edit 1]"));
    expect(text.indexOf("[Edit 1]")).toBeLessThan(text.indexOf("[Edit 2]"));
    expect(text).toContain("a sunset");
    expect(text).toContain("add birds");
    expect(text).toContain("add clouds");
  });

  it("marks missing ancestor files without throwing", async () => {
    const original = path.join(os.homedir(), "generated.png");
    const edit1 = path.join(os.homedir(), "edited-1.png");

    vi.mocked(fs.access).mockImplementation(async (p: any) => {
      if (String(p) === original) throw new Error("ENOENT");
    });
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith("edited-1.json")) return JSON.stringify({ operation: "edit", prompt: "edit", model: CONSTANTS.MODEL, timestamp: "2025-01-02T00:00:00.000Z", sourceImage: original });
      if (String(p).endsWith("generated.json")) return JSON.stringify({ operation: "generate", prompt: "original", model: CONSTANTS.MODEL, timestamp: "2025-01-01T00:00:00.000Z" });
      throw new Error("ENOENT");
    });

    const result = await callExecute(server, "get_image_history", { imagePath: edit1 });
    const text = result.content[0].text as string;
    expect(text).toContain("(file missing)");
    expect(text).toContain("[Original]");
  });

  it("handles circular sidecar references without looping forever", async () => {
    const a = path.join(os.homedir(), "a.png");
    const b = path.join(os.homedir(), "b.png");

    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith("a.json")) return JSON.stringify({ operation: "edit", prompt: "a", model: "m", timestamp: "t", sourceImage: b });
      if (String(p).endsWith("b.json")) return JSON.stringify({ operation: "edit", prompt: "b", model: "m", timestamp: "t", sourceImage: a });
      throw new Error("ENOENT");
    });

    // Should terminate without hanging
    await expect(
      callExecute(server, "get_image_history", { imagePath: a })
    ).resolves.toBeDefined();
  });
});

// -------------------------------------------------------------------
// validateApiKeyFormat
// -------------------------------------------------------------------

describe("validateApiKeyFormat", () => {
  const validKey = "AIza" + "A".repeat(35);

  it("returns valid:true for a correctly formatted key", () => {
    expect(validateApiKeyFormat(validKey).valid).toBe(true);
    expect(validateApiKeyFormat(validKey).warning).toBeUndefined();
  });

  it("returns valid:false when key does not start with 'AIza'", () => {
    const result = validateApiKeyFormat("XYza" + "A".repeat(35));
    expect(result.valid).toBe(false);
    expect(result.warning).toContain("AIza");
    expect(result.warning).toContain("aistudio.google.com");
  });

  it("returns valid:false when key is too short", () => {
    const result = validateApiKeyFormat("AIza" + "A".repeat(10));
    expect(result.valid).toBe(false);
    expect(result.warning).toContain("truncated");
    expect(result.warning).toContain(`${CONSTANTS.API_KEY_LENGTH}`);
  });

  it("returns valid:false when key is too long", () => {
    const result = validateApiKeyFormat("AIza" + "A".repeat(50));
    expect(result.valid).toBe(false);
    expect(result.warning).toContain("truncated");
  });

  it("returns valid:false for an empty string", () => {
    expect(validateApiKeyFormat("").valid).toBe(false);
  });

  it("reports the actual key length in the warning", () => {
    const shortKey = "AIzaABC";
    expect(validateApiKeyFormat(shortKey).warning).toContain(`${shortKey.length} characters`);
  });
});

// -------------------------------------------------------------------
// configure_gemini_token — key format warning surfacing
// -------------------------------------------------------------------

describe("NanoBananaMCP configure_gemini_token key format warnings", () => {
  let server: NanoBananaMCP;
  let stderrSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    server = new NanoBananaMCP();
    stubFsDefaults();
    stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    stderrSpy.mockRestore();
  });

  it("returns a clean success message for a well-formed key", async () => {
    const goodKey = "AIza" + "A".repeat(35);
    const result = await callExecute(server, "configure_gemini_token", { apiKey: goodKey });
    expect(result.content[0].text).toContain("configured successfully");
    expect(result.content[0].text).not.toContain("⚠️");
  });

  it("includes a warning in the response text for a malformed key", async () => {
    const result = await callExecute(server, "configure_gemini_token", { apiKey: "not-a-real-key" });
    expect(result.content[0].text).toContain("⚠️");
    expect(result.content[0].text).toContain("aistudio.google.com");
  });

  it("logs the format warning to stderr for a malformed key", async () => {
    await callExecute(server, "configure_gemini_token", { apiKey: "not-a-real-key" });
    const output = stderrSpy.mock.calls.map(([m]) => String(m)).join("");
    expect(output).toContain("API key format warning");
  });

  it("does not log a format warning to stderr for a well-formed key", async () => {
    await callExecute(server, "configure_gemini_token", { apiKey: "AIza" + "A".repeat(35) });
    const output = stderrSpy.mock.calls.map(([m]) => String(m)).join("");
    expect(output).not.toContain("API key format warning");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — clear_session
// -------------------------------------------------------------------

describe("NanoBananaMCP clear_session", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
  });

  it("clears lastImagePath and saves session", async () => {
    (server as any).lastImagePath = path.join(os.homedir(), "img.png");
    const result = await callExecute(server, "clear_session", {});
    expect((server as any).lastImagePath).toBeNull();
    const sessionCall = vi.mocked(fs.writeFile).mock.calls.find(([p]) =>
      String(p).endsWith(".nano-banana-session.json")
    );
    expect(sessionCall).toBeDefined();
    expect(result.content[0].text).toContain("cleared");
  });

  it("returns an 'already empty' message when no session target was set", async () => {
    const result = await callExecute(server, "clear_session", {});
    expect(result.content[0].text).toContain("already empty");
  });

  it("does not delete any image files", async () => {
    (server as any).lastImagePath = path.join(os.homedir(), "img.png");
    await callExecute(server, "clear_session", {});
    expect(fs.unlink).not.toHaveBeenCalled();
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — revert_to_original
// -------------------------------------------------------------------

describe("NanoBananaMCP revert_to_original", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
  });

  it("throws when no imagePath and no session target", async () => {
    await expect(callExecute(server, "revert_to_original", {})).rejects.toMatchObject({
      code: ErrorCode.InvalidRequest,
    });
  });

  it("returns 'already original' when image has no sourceImage sidecar", async () => {
    const imgPath = path.join(os.homedir(), "generated.png");
    vi.mocked(fs.readFile).mockResolvedValue(
      JSON.stringify({ operation: "generate", prompt: "p", model: "m", timestamp: "t" })
    );
    const result = await callExecute(server, "revert_to_original", { imagePath: imgPath });
    expect(result.content[0].text).toContain("already the original");
  });

  it("walks the chain and sets lastImagePath to the root image", async () => {
    const original = path.join(os.homedir(), "generated.png");
    const edit1 = path.join(os.homedir(), "edited-1.png");
    const edit2 = path.join(os.homedir(), "edited-2.png");

    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith("edited-2.json")) return JSON.stringify({ operation: "edit", prompt: "e2", model: "m", timestamp: "t", sourceImage: edit1 });
      if (String(p).endsWith("edited-1.json")) return JSON.stringify({ operation: "edit", prompt: "e1", model: "m", timestamp: "t", sourceImage: original });
      if (String(p).endsWith("generated.json")) return JSON.stringify({ operation: "generate", prompt: "orig", model: "m", timestamp: "t" });
      throw new Error("ENOENT");
    });

    const result = await callExecute(server, "revert_to_original", { imagePath: edit2 });
    expect((server as any).lastImagePath).toBe(path.resolve(original));
    expect(result.content[0].text).toContain("reverted to original");
    expect(result.content[0].text).toContain(path.resolve(original));
  });

  it("uses session lastImagePath when no imagePath argument given", async () => {
    const original = path.join(os.homedir(), "generated.png");
    const edit1 = path.join(os.homedir(), "edited-1.png");
    (server as any).lastImagePath = edit1;

    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith("edited-1.json")) return JSON.stringify({ operation: "edit", prompt: "e1", model: "m", timestamp: "t", sourceImage: original });
      if (String(p).endsWith("generated.json")) return JSON.stringify({ operation: "generate", prompt: "orig", model: "m", timestamp: "t" });
      throw new Error("ENOENT");
    });

    await callExecute(server, "revert_to_original", {});
    expect((server as any).lastImagePath).toBe(path.resolve(original));
  });

  it("throws when the original file no longer exists on disk", async () => {
    const original = path.join(os.homedir(), "generated.png");
    const edit1 = path.join(os.homedir(), "edited-1.png");

    vi.mocked(fs.access).mockRejectedValue(new Error("ENOENT"));
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith("edited-1.json")) return JSON.stringify({ operation: "edit", prompt: "e", model: "m", timestamp: "t", sourceImage: original });
      if (String(p).endsWith("generated.json")) return JSON.stringify({ operation: "generate", prompt: "p", model: "m", timestamp: "t" });
      throw new Error("ENOENT");
    });

    await expect(callExecute(server, "revert_to_original", { imagePath: edit1 })).rejects.toMatchObject({
      code: ErrorCode.InvalidRequest,
    });
  });

  it("saves session after reverting", async () => {
    const original = path.join(os.homedir(), "generated.png");
    const edit1 = path.join(os.homedir(), "edited-1.png");

    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith("edited-1.json")) return JSON.stringify({ operation: "edit", prompt: "e", model: "m", timestamp: "t", sourceImage: original });
      if (String(p).endsWith("generated.json")) return JSON.stringify({ operation: "generate", prompt: "p", model: "m", timestamp: "t" });
      throw new Error("ENOENT");
    });

    await callExecute(server, "revert_to_original", { imagePath: edit1 });
    const sessionCall = vi.mocked(fs.writeFile).mock.calls.find(([p]) =>
      String(p).endsWith(".nano-banana-session.json")
    );
    expect(sessionCall).toBeDefined();
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — export_images
// -------------------------------------------------------------------

describe("NanoBananaMCP export_images", () => {
  let server: NanoBananaMCP;
  let imagesDir: string;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
    imagesDir = (server as any).getImagesDirectory();
  });

  it("throws when no images directory exists and no specific paths given", async () => {
    vi.mocked(fs.readdir).mockRejectedValue(new Error("ENOENT"));
    await expect(
      callExecute(server, "export_images", { outputDir: "/tmp/export" })
    ).rejects.toMatchObject({ code: ErrorCode.InvalidRequest });
  });

  it("returns 'no images' when directory is empty", async () => {
    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readdir).mockResolvedValue([] as any);
    const result = await callExecute(server, "export_images", { outputDir: "/tmp/export" });
    expect(result.content[0].text).toContain("No images to export");
  });

  it("copies all images from the output directory to the export dir", async () => {
    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readdir).mockResolvedValue([
      { name: "generated-a.png", isFile: () => true } as any,
      { name: "edited-b.png", isFile: () => true } as any,
    ]);
    vi.mocked(fs.copyFile).mockResolvedValue(undefined as any);

    const result = await callExecute(server, "export_images", { outputDir: "/tmp/export" });
    expect(fs.copyFile).toHaveBeenCalledTimes(4); // 2 images + 2 sidecar attempts
    expect(result.content[0].text).toContain("Exported 2 image(s)");
    expect(result.content[0].text).toContain("/tmp/export");
  });

  it("exports only the specified imagePaths when provided", async () => {
    vi.mocked(fs.copyFile).mockResolvedValue(undefined as any);
    const imgA = path.join(imagesDir, "a.png");
    const imgB = path.join(imagesDir, "b.png");

    const result = await callExecute(server, "export_images", {
      outputDir: "/tmp/export",
      imagePaths: [imgA, imgB],
    });
    expect(result.content[0].text).toContain("Exported 2 image(s)");
  });

  it("creates the output directory if it does not exist", async () => {
    vi.mocked(fs.readdir).mockResolvedValue([
      { name: "img.png", isFile: () => true } as any,
    ]);
    vi.mocked(fs.copyFile).mockResolvedValue(undefined as any);

    await callExecute(server, "export_images", { outputDir: "/tmp/new-dir/export" });
    expect(fs.mkdir).toHaveBeenCalledWith(
      path.resolve("/tmp/new-dir/export"),
      expect.objectContaining({ recursive: true })
    );
  });

  it("reports skipped images when copyFile fails", async () => {
    vi.mocked(fs.readdir).mockResolvedValue([
      { name: "a.png", isFile: () => true } as any,
      { name: "b.png", isFile: () => true } as any,
    ]);
    vi.mocked(fs.copyFile).mockImplementation(async (src: any) => {
      if (String(src).endsWith("b.png")) throw new Error("EACCES");
    });

    const result = await callExecute(server, "export_images", { outputDir: "/tmp/export" });
    expect(result.content[0].text).toContain("Exported 1 image(s)");
    expect(result.content[0].text).toContain("⚠️");
    expect(result.content[0].text).toContain("b.png");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — list_generated_images filtering
// -------------------------------------------------------------------

describe("NanoBananaMCP list_generated_images filtering", () => {
  let server: NanoBananaMCP;
  let imagesDir: string;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
    imagesDir = (server as any).getImagesDirectory();

    vi.mocked(fs.access).mockResolvedValue(undefined as any);
    vi.mocked(fs.readdir).mockResolvedValue([
      { name: "generated-a.png", isFile: () => true } as any,
      { name: "edited-b.png", isFile: () => true } as any,
    ]);
    vi.mocked(fs.stat).mockResolvedValue({ size: 1024, mtime: new Date("2025-06-01T00:00:00Z") } as any);
  });

  it("filters to only 'generate' operations", async () => {
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith("generated-a.json")) return JSON.stringify({ operation: "generate", prompt: "gen", model: "m", timestamp: "2025-06-01T00:00:00Z" });
      if (String(p).endsWith("edited-b.json")) return JSON.stringify({ operation: "edit", prompt: "edit", model: "m", timestamp: "2025-06-01T00:00:00Z" });
      throw new Error("ENOENT");
    });

    const result = await callExecute(server, "list_generated_images", { operation: "generate" });
    const text = result.content[0].text as string;
    expect(text).toContain("generated-a.png");
    expect(text).not.toContain("edited-b.png");
  });

  it("filters to only 'edit' operations", async () => {
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith("generated-a.json")) return JSON.stringify({ operation: "generate", prompt: "gen", model: "m", timestamp: "2025-06-01T00:00:00Z" });
      if (String(p).endsWith("edited-b.json")) return JSON.stringify({ operation: "edit", prompt: "edit", model: "m", timestamp: "2025-06-01T00:00:00Z" });
      throw new Error("ENOENT");
    });

    const result = await callExecute(server, "list_generated_images", { operation: "edit" });
    const text = result.content[0].text as string;
    expect(text).toContain("edited-b.png");
    expect(text).not.toContain("generated-a.png");
  });

  it("filters by 'since' date using sidecar timestamp", async () => {
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith("generated-a.json")) return JSON.stringify({ operation: "generate", prompt: "new", model: "m", timestamp: "2025-07-01T00:00:00Z" });
      if (String(p).endsWith("edited-b.json")) return JSON.stringify({ operation: "edit", prompt: "old", model: "m", timestamp: "2025-01-01T00:00:00Z" });
      throw new Error("ENOENT");
    });

    const result = await callExecute(server, "list_generated_images", { since: "2025-06-01" });
    const text = result.content[0].text as string;
    expect(text).toContain("generated-a.png");
    expect(text).not.toContain("edited-b.png");
  });

  it("returns no-match message when filters exclude everything", async () => {
    vi.mocked(fs.readFile).mockResolvedValue(
      JSON.stringify({ operation: "generate", prompt: "p", model: "m", timestamp: "2025-01-01T00:00:00Z" })
    );

    const result = await callExecute(server, "list_generated_images", {
      operation: "edit",
      since: "2026-01-01",
    });
    expect(result.content[0].text).toContain("No images found");
    expect(result.content[0].text).toContain("operation=");
  });

  it("shows operation tag in each result row", async () => {
    vi.mocked(fs.readFile).mockResolvedValue(
      JSON.stringify({ operation: "generate", prompt: "p", model: "m", timestamp: "2025-01-01T00:00:00Z" })
    );

    const result = await callExecute(server, "list_generated_images", {});
    expect(result.content[0].text).toContain("[generate]");
  });
});

// -------------------------------------------------------------------
// NanoBananaMCP — generate_image_batch with referenceImages
// -------------------------------------------------------------------

describe("NanoBananaMCP generate_image_batch with referenceImages", () => {
  let server: NanoBananaMCP;

  beforeEach(() => {
    server = new NanoBananaMCP();
    configureServer(server);
    stubFsDefaults();
    vi.mocked(fs.readFile).mockImplementation(async (p: any) => {
      if (String(p).endsWith(".json")) throw new Error("ENOENT");
      return Buffer.from("fake ref image bytes");
    });
  });

  it("passes reference images to each generateImageCore call", async () => {
    const coreSpy = vi.spyOn(server as any, "generateImageCore").mockResolvedValue({
      savedPaths: [path.join(os.homedir(), "img.png")],
      imageContent: [],
      textContent: "",
      tokenUsage: { total: 50, prompt: 40, response: 10 },
    });

    const refPath = path.join(os.homedir(), "style.png");
    await callExecute(server, "generate_image_batch", {
      prompt: "a city",
      count: 2,
      referenceImages: [refPath],
    });

    expect(coreSpy).toHaveBeenCalledTimes(2);
    expect(coreSpy).toHaveBeenCalledWith("a city", [refPath]);
  });

  it("works without referenceImages (passes undefined)", async () => {
    const coreSpy = vi.spyOn(server as any, "generateImageCore").mockResolvedValue({
      savedPaths: [path.join(os.homedir(), "img.png")],
      imageContent: [],
      textContent: "",
      tokenUsage: { total: 50, prompt: 40, response: 10 },
    });

    await callExecute(server, "generate_image_batch", { prompt: "a cat", count: 1 });
    expect(coreSpy).toHaveBeenCalledWith("a cat", undefined);
  });
});
