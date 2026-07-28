import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
  CallToolRequest,
  CallToolResult,
  ErrorCode,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { GoogleGenAI } from "@google/genai";
import { z } from "zod";
import fs from "fs/promises";
import path from "path";
import os from "os";

export const CONSTANTS = {
  MODEL: "gemini-2.5-flash-preview-05-20",
  TEXT_MODEL: "gemini-2.5-flash",
  OUTPUT_FORMAT: "png" as const,
  TIMEOUT_MS: 120_000,
  RETRY_ATTEMPTS: 3,
  RETRY_BASE_DELAY_MS: 1_000,
  CONFIG_FILENAME: ".nano-banana-config.json",
  SESSION_FILENAME: ".nano-banana-session.json",
  // Gemini 2.5 Flash approximate pricing — USD per million tokens
  PRICING_INPUT_PER_M: 0.075,
  PRICING_OUTPUT_PER_M: 0.30,
  BATCH_MAX_COUNT: 5,
} as const;

// --- Logger ---

const LOG_LEVELS = { DEBUG: 0, INFO: 1, WARN: 2, ERROR: 3, SILENT: 4 } as const;
type LogLevelName = keyof typeof LOG_LEVELS;

export class Logger {
  private readonly level: number;
  readonly levelName: LogLevelName;

  constructor(levelOverride?: string) {
    const raw = (levelOverride ?? process.env.NANO_BANANA_LOG_LEVEL ?? "WARN").toUpperCase() as LogLevelName;
    this.level = LOG_LEVELS[raw] ?? LOG_LEVELS.WARN;
    this.levelName = (Object.entries(LOG_LEVELS).find(([, v]) => v === this.level)?.[0] ?? "WARN") as LogLevelName;
  }

  private write(levelName: string, msg: string, meta?: unknown): void {
    const ts = new Date().toISOString();
    const suffix = meta !== undefined ? ` ${JSON.stringify(meta)}` : "";
    process.stderr.write(`[nano-banana] ${ts} ${levelName.padEnd(5)} ${msg}${suffix}\n`);
  }

  debug(msg: string, meta?: unknown): void { if (this.level <= LOG_LEVELS.DEBUG) this.write("DEBUG", msg, meta); }
  info(msg: string, meta?: unknown): void  { if (this.level <= LOG_LEVELS.INFO)  this.write("INFO",  msg, meta); }
  warn(msg: string, meta?: unknown): void  { if (this.level <= LOG_LEVELS.WARN)  this.write("WARN",  msg, meta); }
  error(msg: string, meta?: unknown): void { if (this.level <= LOG_LEVELS.ERROR) this.write("ERROR", msg, meta); }
}

export const log = new Logger();

// --- Error classification ---

export function classifyApiError(error: unknown, operation: string): McpError {
  if (error instanceof McpError) return error;

  const msg = error instanceof Error ? error.message : String(error);
  const lower = msg.toLowerCase();

  if (lower.includes("timed out")) {
    return new McpError(ErrorCode.InternalError,
      `${operation} timed out. Try again or increase NANO_BANANA_TIMEOUT_MS (current: ${CONSTANTS.TIMEOUT_MS}ms).`);
  }
  if (lower.includes("429") || lower.includes("resource_exhausted") || lower.includes("too many requests") || lower.includes("rate limit")) {
    return new McpError(ErrorCode.InternalError,
      `Rate limit hit during ${operation}. The server retried automatically. Check your Gemini quota at aistudio.google.com.`);
  }
  if (lower.includes("api_key_invalid") || lower.includes("invalid api key") || lower.includes("401") || lower.includes("unauthenticated")) {
    return new McpError(ErrorCode.InvalidRequest,
      `Invalid Gemini API key. Reconfigure: execute({operation: "configure_gemini_token", arguments: {apiKey: "..."}}) or check GEMINI_API_KEY.`);
  }
  if (lower.includes("permission_denied") || lower.includes("403") || lower.includes("quota_exceeded") || lower.includes("billing")) {
    return new McpError(ErrorCode.InvalidRequest,
      `Gemini API access denied or quota exceeded during ${operation}. Check your plan at aistudio.google.com.`);
  }
  if (lower.includes("not_found") || lower.includes("404") || lower.includes("model not found")) {
    return new McpError(ErrorCode.InvalidRequest,
      `Gemini model not found during ${operation}. Check your NANO_BANANA_MODEL setting.`);
  }
  if (lower.includes("503") || lower.includes("unavailable") || lower.includes("service unavailable")) {
    return new McpError(ErrorCode.InternalError,
      `Gemini API temporarily unavailable during ${operation}. Try again in a few seconds.`);
  }
  if (lower.includes("500") || lower.includes("internal_error") || lower.includes("server error")) {
    return new McpError(ErrorCode.InternalError,
      `Gemini API server error during ${operation}. This is usually transient — try again.`);
  }

  return new McpError(ErrorCode.InternalError, `${operation} failed: ${msg}`);
}

// --- Types ---

interface InlineDataPart {
  inlineData: { data: string; mimeType: string };
}
interface TextPart {
  text: string;
}
type ContentPart = InlineDataPart | TextPart;

interface McpImageContent {
  type: "image";
  data: string;
  mimeType: string;
}
interface McpTextContent {
  type: "text";
  text: string;
}
type McpContent = McpImageContent | McpTextContent;

interface OperationParam {
  type: string;
  description: string;
  required: boolean;
  items?: { type: string };
}

interface OperationDef {
  description: string;
  tags: string[];
  params: Record<string, OperationParam>;
  handler: (args: Record<string, unknown>) => Promise<CallToolResult>;
}

export interface ImageSidecar {
  operation: "generate" | "edit";
  prompt: string;
  model: string;
  timestamp: string;
  tokenUsage?: { total: number; prompt: number; response: number };
  sourceImage?: string;
}

// --- Config schema ---

const OutputFormatSchema = z.enum(["png", "jpeg", "webp"]);

export const ConfigSchema = z.object({
  geminiApiKey: z.string().min(1, "Gemini API key is required"),
  model: z.string().optional().default(CONSTANTS.MODEL),
  outputDir: z.string().optional(),
  outputFormat: OutputFormatSchema.optional().default(CONSTANTS.OUTPUT_FORMAT),
  timeoutMs: z.number().positive().optional().default(CONSTANTS.TIMEOUT_MS),
  promptPrefix: z.string().optional().default(""),
  promptSuffix: z.string().optional().default(""),
});

export type Config = z.infer<typeof ConfigSchema>;

// --- Server class ---

export class NanoBananaMCP {
  private server: Server;
  private genAI: GoogleGenAI | null = null;
  private config: Config | null = null;
  private lastImagePath: string | null = null;
  private configSource: "environment" | "config_file" | "not_configured" = "not_configured";
  private operations: Record<string, OperationDef>;

  constructor() {
    this.server = new Server(
      { name: "nano-banana-mcp", version: "1.3.0" },
      { capabilities: { tools: {} } }
    );

    this.operations = {
      configure_gemini_token: {
        description: "Configure your Gemini API key for nano-banana image generation.",
        tags: ["config", "setup", "api", "key", "token"],
        params: {
          apiKey: { type: "string", description: "Your Gemini API key from Google AI Studio", required: true },
        },
        handler: (args) => this.opConfigureGeminiToken(args as { apiKey: string }),
      },
      generate_image: {
        description: "Generate a NEW image from scratch using a text prompt. Use this only when creating a completely new image.",
        tags: ["generate", "create", "image", "new", "text-to-image"],
        params: {
          prompt: { type: "string", description: "Text prompt describing the image to create", required: true },
        },
        handler: (args) => this.opGenerateImage(args as { prompt: string }),
      },
      edit_image: {
        description: "Edit a specific existing image file using a text prompt, optionally with reference images for style or content guidance.",
        tags: ["edit", "modify", "image", "existing", "file", "reference"],
        params: {
          imagePath: { type: "string", description: "Full file path to the image to edit", required: true },
          prompt: { type: "string", description: "Text describing the modifications to make", required: true },
          referenceImages: { type: "array", items: { type: "string" }, description: "Optional file paths to reference images for style transfer or content guidance", required: false },
        },
        handler: (args) => this.opEditImage(args as { imagePath: string; prompt: string; referenceImages?: string[] }),
      },
      continue_editing: {
        description: "Continue editing the last image generated or edited in this session. No file path needed — automatically uses the most recent image.",
        tags: ["edit", "iterate", "continue", "last", "session"],
        params: {
          prompt: { type: "string", description: "Text describing the modifications to make to the last image", required: true },
          referenceImages: { type: "array", items: { type: "string" }, description: "Optional file paths to reference images", required: false },
        },
        handler: (args) => this.opContinueEditing(args as { prompt: string; referenceImages?: string[] }),
      },
      delete_image: {
        description: "Delete a generated image (and its metadata sidecar) from the output directory. Clears the session's last-image pointer if the deleted file was the most recent.",
        tags: ["delete", "remove", "image", "cleanup", "files"],
        params: {
          imagePath: { type: "string", description: "Full file path to the image to delete (must be inside the output directory)", required: true },
        },
        handler: (args) => this.opDeleteImage(args as { imagePath: string }),
      },
      enhance_prompt: {
        description: "Use Gemini to expand and improve a rough prompt into a detailed, high-quality image generation prompt before generating.",
        tags: ["prompt", "enhance", "improve", "text", "ai"],
        params: {
          prompt: { type: "string", description: "The rough or simple prompt to improve", required: true },
          style: { type: "string", description: "Optional style hint (e.g. 'photorealistic', 'oil painting', 'anime')", required: false },
        },
        handler: (args) => this.opEnhancePrompt(args as { prompt: string; style?: string }),
      },
      get_configuration_status: {
        description: "Check whether the Gemini API key is configured and view active settings (model, output directory, format, timeout, log level).",
        tags: ["config", "status", "check", "settings"],
        params: {},
        handler: () => this.opGetConfigurationStatus(),
      },
      get_last_image_info: {
        description: "Get the file path, size, and modification time of the last generated or edited image in this session.",
        tags: ["info", "last", "image", "metadata", "session"],
        params: {},
        handler: () => this.opGetLastImageInfo(),
      },
      list_generated_images: {
        description: "List images previously generated or edited by nano-banana in the output directory, sorted newest first, with the prompt that created each one.",
        tags: ["list", "browse", "images", "history", "files"],
        params: {
          limit: { type: "number", description: "Maximum number of images to return (default: 20)", required: false },
        },
        handler: (args) => this.opListGeneratedImages(args as { limit?: number }),
      },
      generate_image_batch: {
        description: `Generate multiple variations of the same prompt in parallel and save all of them. Capped at ${CONSTANTS.BATCH_MAX_COUNT} images per call to avoid runaway costs. Sets the session's last image to the first successful result.`,
        tags: ["generate", "batch", "multiple", "variations", "parallel"],
        params: {
          prompt: { type: "string", description: "Text prompt describing the images to create", required: true },
          count: { type: "number", description: `Number of images to generate (default: 2, max: ${CONSTANTS.BATCH_MAX_COUNT})`, required: false },
        },
        handler: (args) => this.opGenerateImageBatch(args as { prompt: string; count?: number }),
      },
      get_image_history: {
        description: "Show the full edit lineage of an image by walking its sidecar chain. Returns each ancestor in order from the original generation through every edit that produced the current file.",
        tags: ["history", "chain", "edit", "lineage", "sidecar", "trace"],
        params: {
          imagePath: { type: "string", description: "Full file path to the image whose history you want to trace", required: true },
        },
        handler: (args) => this.opGetImageHistory(args as { imagePath: string }),
      },
    };

    this.setupHandlers();
  }

  // --- MCP tool registration (2 tools only) ---

  private setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: "search",
          description: "Discover available nano-banana operations. Returns each operation's name, description, and parameter list. Use this before execute to find the right operation and understand its required arguments. Optionally filter by keyword or set verbose:true for full parameter descriptions.",
          inputSchema: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "Optional keyword to filter operations (e.g. 'generate', 'edit', 'config', 'list'). Omit to return all operations.",
              },
              verbose: {
                type: "boolean",
                description: "Set to true to include full parameter descriptions instead of just names and types.",
              },
            },
            additionalProperties: false,
          },
        },
        {
          name: "execute",
          description: "Execute a nano-banana operation by name. Use search first to discover available operations and their required parameters.",
          inputSchema: {
            type: "object",
            properties: {
              operation: {
                type: "string",
                description: "Name of the operation to run (e.g. 'generate_image', 'edit_image', 'continue_editing')",
              },
              arguments: {
                type: "object",
                description: "Arguments for the operation. Keys and types depend on the operation — use search to see required fields.",
                additionalProperties: true,
              },
            },
            required: ["operation", "arguments"],
            additionalProperties: false,
          },
        },
      ] as Tool[],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest): Promise<CallToolResult> => {
      try {
        switch (request.params.name) {
          case "search":
            return this.handleSearch(request);
          case "execute":
            return await this.handleExecute(request);
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${request.params.name}`);
        }
      } catch (error) {
        if (error instanceof McpError) throw error;
        throw new McpError(ErrorCode.InternalError, `Tool execution failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    });
  }

  // --- search handler ---

  private handleSearch(request: CallToolRequest): CallToolResult {
    const { query, verbose = false } = (request.params.arguments ?? {}) as { query?: string; verbose?: boolean };
    const term = query?.toLowerCase().trim();

    const matches = Object.entries(this.operations).filter(([name, op]) => {
      if (!term) return true;
      return (
        name.includes(term) ||
        op.description.toLowerCase().includes(term) ||
        op.tags.some(t => t.includes(term))
      );
    });

    if (matches.length === 0) {
      return {
        content: [{
          type: "text",
          text: `No operations matched "${query}". Try search without a query to see all available operations.`,
        }],
      };
    }

    const lines: string[] = [
      `nano-banana operations${term ? ` matching "${query}"` : ""} (${matches.length}):`,
      "",
    ];

    for (const [name, op] of matches) {
      lines.push(`  ${name}`);
      lines.push(`    ${op.description}`);

      const required = Object.entries(op.params).filter(([, p]) => p.required);
      const optional = Object.entries(op.params).filter(([, p]) => !p.required);

      if (required.length > 0) {
        if (verbose) {
          lines.push("    Required:");
          required.forEach(([k, p]) => lines.push(`      ${k} (${p.type}) — ${p.description}`));
        } else {
          lines.push(`    Required: ${required.map(([k, p]) => `${k} (${p.type})`).join(", ")}`);
        }
      }
      if (optional.length > 0) {
        if (verbose) {
          lines.push("    Optional:");
          optional.forEach(([k, p]) => lines.push(`      ${k} (${p.type}) — ${p.description}`));
        } else {
          lines.push(`    Optional: ${optional.map(([k, p]) => `${k} (${p.type})`).join(", ")}`);
        }
      }
      lines.push("");
    }

    lines.push('Use execute({"operation": "<name>", "arguments": {...}}) to run an operation.');

    return { content: [{ type: "text", text: lines.join("\n") }] };
  }

  // --- execute handler ---

  private async handleExecute(request: CallToolRequest): Promise<CallToolResult> {
    const { operation, arguments: args } = request.params.arguments as {
      operation: string;
      arguments: Record<string, unknown>;
    };

    const op = this.operations[operation];
    if (!op) {
      const available = Object.keys(this.operations).join(", ");
      throw new McpError(
        ErrorCode.InvalidParams,
        `Unknown operation: "${operation}". Available: ${available}`
      );
    }

    const missing = Object.entries(op.params)
      .filter(([key, p]) => p.required && (args[key] === undefined || args[key] === null))
      .map(([key]) => key);

    if (missing.length > 0) {
      throw new McpError(
        ErrorCode.InvalidParams,
        `Operation "${operation}" is missing required arguments: ${missing.join(", ")}`
      );
    }

    log.info(`execute: ${operation}`);
    return await op.handler(args);
  }

  // --- Utility methods ---

  private withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
    let timerId: ReturnType<typeof setTimeout>;
    const timeout = new Promise<never>((_, reject) => {
      timerId = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(timerId!));
  }

  private async withRetry<T>(
    fn: () => Promise<T>,
    label: string,
    attempts: number = CONSTANTS.RETRY_ATTEMPTS,
    baseDelayMs: number = CONSTANTS.RETRY_BASE_DELAY_MS
  ): Promise<T> {
    let lastError: unknown;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        if (attempt === attempts) break;
        const delayMs = baseDelayMs * 2 ** (attempt - 1);
        log.warn(`${label} failed (attempt ${attempt}/${attempts}), retrying in ${delayMs}ms`, {
          error: error instanceof Error ? error.message : String(error),
        });
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
    throw lastError;
  }

  private validateImagePath(imagePath: string): void {
    const resolved = path.resolve(imagePath);
    const allowedRoots = [os.homedir(), os.tmpdir(), process.cwd(), this.getImagesDirectory()];
    if (!allowedRoots.some(root => resolved.startsWith(root))) {
      throw new McpError(ErrorCode.InvalidParams, `Image path is outside allowed directories: ${imagePath}`);
    }
  }

  private sidecarPath(imagePath: string): string {
    const dir = path.dirname(imagePath);
    const base = path.basename(imagePath, path.extname(imagePath));
    return path.join(dir, `${base}.json`);
  }

  private async writeImageSidecar(imagePath: string, data: ImageSidecar): Promise<void> {
    try {
      await fs.writeFile(this.sidecarPath(imagePath), JSON.stringify(data, null, 2), { mode: 0o600 });
      log.debug("Sidecar written", { path: this.sidecarPath(imagePath) });
    } catch (err) {
      log.warn("Failed to write image sidecar", {
        path: this.sidecarPath(imagePath),
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  private async readImageSidecar(imagePath: string): Promise<ImageSidecar | null> {
    try {
      const data = JSON.parse(await fs.readFile(this.sidecarPath(imagePath), "utf-8"));
      return data as ImageSidecar;
    } catch {
      return null;
    }
  }

  private async saveSession(): Promise<void> {
    const sessionPath = path.join(process.cwd(), CONSTANTS.SESSION_FILENAME);
    try {
      await fs.writeFile(sessionPath, JSON.stringify({ lastImagePath: this.lastImagePath }, null, 2), { mode: 0o600 });
      log.debug("Session saved", { lastImagePath: this.lastImagePath });
    } catch (err) {
      log.warn("Failed to save session", { error: err instanceof Error ? err.message : String(err) });
    }
  }

  private async loadSession(): Promise<void> {
    const sessionPath = path.join(process.cwd(), CONSTANTS.SESSION_FILENAME);
    try {
      const data = JSON.parse(await fs.readFile(sessionPath, "utf-8"));
      if (data.lastImagePath && typeof data.lastImagePath === "string") {
        try {
          await fs.access(data.lastImagePath);
          this.lastImagePath = data.lastImagePath;
          log.info("Session restored", { lastImagePath: this.lastImagePath });
        } catch {
          log.debug("Session has lastImagePath but file no longer exists — skipping restore");
        }
      }
    } catch {
      // No session file is normal on first run
    }
  }

  private applyPromptAffix(prompt: string): string {
    const prefix = this.config?.promptPrefix ?? "";
    const suffix = this.config?.promptSuffix ?? "";
    return `${prefix}${prompt}${suffix}`;
  }

  private buildSuccessResponse(opts: {
    operation: "generated" | "edited";
    prompt: string;
    savedFiles: string[];
    textContent: string;
    originalPath?: string;
    referenceImages?: string[];
    tokenUsage?: { total: number; prompt: number; response: number };
  }): string {
    const { operation, prompt, savedFiles, textContent, originalPath, referenceImages, tokenUsage } = opts;

    let text = operation === "generated"
      ? `Image generated with nano-banana!\n\nPrompt: "${prompt}"`
      : `Image edited with nano-banana!\n\nOriginal: ${originalPath}\nEdit prompt: "${prompt}"`;

    if (referenceImages && referenceImages.length > 0) {
      text += `\n\nReference images used:\n${referenceImages.map(f => `- ${f}`).join("\n")}`;
    }

    if (textContent) {
      text += `\n\nDescription: ${textContent}`;
    }

    if (savedFiles.length > 0) {
      const label = operation === "generated" ? "Image" : "Edited image";
      text += `\n\n📁 ${label} saved to:\n${savedFiles.map(f => `- ${f}`).join("\n")}`;
      text += "\n\n💡 View the image by opening the file path above";
      text += "\n\n🔄 To modify this image, use: execute({operation: \"continue_editing\", arguments: {prompt: \"...\"}})";
    } else {
      text += "\n\nNote: No image was returned. The model may have returned only text.";
      text += "\n\n💡 Tip: Try running the operation again — sometimes the first call needs to warm up the model.";
    }

    if (tokenUsage && tokenUsage.total > 0) {
      const inputCost = (tokenUsage.prompt / 1_000_000) * CONSTANTS.PRICING_INPUT_PER_M;
      const outputCost = (tokenUsage.response / 1_000_000) * CONSTANTS.PRICING_OUTPUT_PER_M;
      const totalCost = inputCost + outputCost;
      const costStr = totalCost < 0.001 ? "<$0.001" : `~$${totalCost.toFixed(3)}`;
      text += `\n\n📊 Tokens used: ${tokenUsage.total.toLocaleString()} (prompt: ${tokenUsage.prompt.toLocaleString()}, response: ${tokenUsage.response.toLocaleString()}) — est. ${costStr}`;
    }

    return text;
  }

  private ensureConfigured(): void {
    if (!this.config || !this.genAI) {
      throw new McpError(
        ErrorCode.InvalidRequest,
        "Gemini API key not configured. Run: execute({operation: \"configure_gemini_token\", arguments: {apiKey: \"...\"}}) or set GEMINI_API_KEY in your environment."
      );
    }
  }

  private getMimeType(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();
    if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
    if (ext === ".png") return "image/png";
    if (ext === ".webp") return "image/webp";
    return "image/jpeg";
  }

  private getImagesDirectory(): string {
    if (this.config?.outputDir) return this.config.outputDir;

    if (os.platform() === "win32") {
      return path.join(os.homedir(), "Documents", "nano-banana-images");
    }

    const cwd = process.cwd();
    if (cwd.startsWith("/usr/") || cwd.startsWith("/opt/") || cwd.startsWith("/var/")) {
      return path.join(os.homedir(), "nano-banana-images");
    }

    return path.join(cwd, "generated_imgs");
  }

  private async saveConfig(): Promise<void> {
    if (this.config) {
      const configPath = path.join(process.cwd(), CONSTANTS.CONFIG_FILENAME);
      await fs.writeFile(configPath, JSON.stringify(this.config, null, 2), { mode: 0o600 });
    }
  }

  private async loadConfig(): Promise<void> {
    const envApiKey = process.env.GEMINI_API_KEY;
    const envModel = process.env.NANO_BANANA_MODEL;
    const envOutputDir = process.env.NANO_BANANA_OUTPUT_DIR;
    const envOutputFormat = process.env.NANO_BANANA_OUTPUT_FORMAT;
    const envTimeoutMs = process.env.NANO_BANANA_TIMEOUT_MS;
    const envPromptPrefix = process.env.NANO_BANANA_PROMPT_PREFIX;
    const envPromptSuffix = process.env.NANO_BANANA_PROMPT_SUFFIX;

    const buildOverrides = (): Partial<Record<string, unknown>> => {
      const o: Partial<Record<string, unknown>> = {};
      if (envModel) o.model = envModel;
      if (envOutputDir) o.outputDir = envOutputDir;
      if (envOutputFormat) o.outputFormat = envOutputFormat;
      if (envTimeoutMs) o.timeoutMs = Number(envTimeoutMs);
      if (envPromptPrefix !== undefined) o.promptPrefix = envPromptPrefix;
      if (envPromptSuffix !== undefined) o.promptSuffix = envPromptSuffix;
      return o;
    };

    if (envApiKey) {
      try {
        this.config = ConfigSchema.parse({ geminiApiKey: envApiKey, ...buildOverrides() });
        this.genAI = new GoogleGenAI({ apiKey: this.config.geminiApiKey });
        this.configSource = "environment";
        log.info("Config loaded from environment variable");
        log.debug("Active config", {
          model: this.config.model,
          outputFormat: this.config.outputFormat,
          timeoutMs: this.config.timeoutMs,
          outputDir: this.config.outputDir ?? "(auto)",
          promptPrefix: this.config.promptPrefix || "(none)",
          promptSuffix: this.config.promptSuffix || "(none)",
        });
        return;
      } catch (err) {
        log.warn("GEMINI_API_KEY present but config parse failed, falling back to config file", {
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    try {
      const configPath = path.join(process.cwd(), CONSTANTS.CONFIG_FILENAME);
      const raw = JSON.parse(await fs.readFile(configPath, "utf-8"));
      this.config = ConfigSchema.parse({ ...raw, ...buildOverrides() });
      this.genAI = new GoogleGenAI({ apiKey: this.config.geminiApiKey });
      this.configSource = "config_file";
      log.info("Config loaded from file", { path: configPath });
      log.debug("Active config", {
        model: this.config.model,
        outputFormat: this.config.outputFormat,
        timeoutMs: this.config.timeoutMs,
        outputDir: this.config.outputDir ?? "(auto)",
        promptPrefix: this.config.promptPrefix || "(none)",
        promptSuffix: this.config.promptSuffix || "(none)",
      });
    } catch {
      this.configSource = "not_configured";
      log.warn("No API key found — server will start unconfigured");
    }
  }

  // --- Operation implementations ---

  private async opConfigureGeminiToken(args: { apiKey: string }): Promise<CallToolResult> {
    try {
      this.config = ConfigSchema.parse({ geminiApiKey: args.apiKey });
      this.genAI = new GoogleGenAI({ apiKey: args.apiKey });
      this.configSource = "config_file";
      await this.saveConfig();
      log.info("API key configured via tool");
      return {
        content: [{ type: "text", text: "Gemini API token configured successfully! You can now use nano-banana image generation features." }],
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        throw new McpError(ErrorCode.InvalidParams, `Invalid API key: ${error.errors[0]?.message}`);
      }
      throw error;
    }
  }

  // Core image generation — saves file + sidecar, returns paths and image content.
  // Does NOT update lastImagePath or save session (callers do that).
  private async generateImageCore(originalPrompt: string): Promise<{
    savedPaths: string[];
    imageContent: McpImageContent[];
    textContent: string;
    tokenUsage?: { total: number; prompt: number; response: number };
  }> {
    const prompt = this.applyPromptAffix(originalPrompt);
    const response = await this.withRetry(
      () => this.withTimeout(
        this.genAI!.models.generateContent({ model: this.config!.model, contents: prompt }),
        this.config!.timeoutMs,
        "generate_image"
      ),
      "generate_image"
    );

    const savedPaths: string[] = [];
    const imageContent: McpImageContent[] = [];
    let textContent = "";

    const imagesDir = this.getImagesDirectory();
    await fs.mkdir(imagesDir, { recursive: true, mode: 0o755 });

    const usageMetadata = response.usageMetadata;
    const tokenUsage = usageMetadata
      ? { total: usageMetadata.totalTokenCount ?? 0, prompt: usageMetadata.promptTokenCount ?? 0, response: usageMetadata.candidatesTokenCount ?? 0 }
      : undefined;

    for (const part of response.candidates?.[0]?.content?.parts ?? []) {
      if (part.text) textContent += part.text;
      if (part.inlineData?.data) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        const randomId = Math.random().toString(36).substring(2, 8);
        const ext = this.config!.outputFormat === "jpeg" ? "jpg" : this.config!.outputFormat;
        const filePath = path.join(imagesDir, `generated-${timestamp}-${randomId}.${ext}`);

        const imageBytes = Buffer.from(part.inlineData.data, "base64");
        await fs.writeFile(filePath, imageBytes);
        savedPaths.push(filePath);
        log.info("Image saved", { path: filePath, sizeKb: Math.round(imageBytes.length / 1024) });

        await this.writeImageSidecar(filePath, {
          operation: "generate",
          prompt: originalPrompt,
          model: this.config!.model,
          timestamp: new Date().toISOString(),
          tokenUsage,
        });

        imageContent.push({ type: "image", data: part.inlineData.data, mimeType: part.inlineData.mimeType || "image/png" });
      }
    }

    if (tokenUsage) log.debug("generate_image: token usage", tokenUsage);
    return { savedPaths, imageContent, textContent, tokenUsage };
  }

  private async opGenerateImage(args: { prompt: string }): Promise<CallToolResult> {
    this.ensureConfigured();
    log.debug("generate_image: calling Gemini API", { model: this.config!.model, timeoutMs: this.config!.timeoutMs });

    try {
      const { savedPaths, imageContent, textContent, tokenUsage } = await this.generateImageCore(args.prompt);

      if (savedPaths.length > 0) {
        this.lastImagePath = savedPaths[savedPaths.length - 1];
        await this.saveSession();
      }

      const content: McpContent[] = [
        { type: "text", text: this.buildSuccessResponse({ operation: "generated", prompt: args.prompt, savedFiles: savedPaths, textContent, tokenUsage }) },
        ...imageContent,
      ];
      return { content };
    } catch (error) {
      log.error("generate_image failed", { error: error instanceof Error ? error.message : String(error) });
      throw classifyApiError(error, "generate_image");
    }
  }

  private async opGenerateImageBatch(args: { prompt: string; count?: number }): Promise<CallToolResult> {
    this.ensureConfigured();

    const count = Math.min(Math.max(args.count ?? 2, 1), CONSTANTS.BATCH_MAX_COUNT);
    log.info("generate_image_batch starting", { count, model: this.config!.model });

    const results = await Promise.allSettled(
      Array.from({ length: count }, () => this.generateImageCore(args.prompt))
    );

    type CoreResult = { savedPaths: string[]; imageContent: McpImageContent[]; textContent: string; tokenUsage?: { total: number; prompt: number; response: number } };
    const successes: CoreResult[] = [];
    const failureMessages: string[] = [];

    for (const result of results) {
      if (result.status === "fulfilled") {
        successes.push(result.value);
      } else {
        const err = classifyApiError(result.reason, "generate_image_batch");
        failureMessages.push(err.message);
        log.warn("Batch item failed", { error: result.reason instanceof Error ? result.reason.message : String(result.reason) });
      }
    }

    if (successes.length === 0) {
      throw new McpError(ErrorCode.InternalError,
        `All ${count} batch generations failed. First error: ${failureMessages[0]}`);
    }

    // Set lastImagePath to the first successful image
    const firstSavedPath = successes.find(s => s.savedPaths.length > 0)?.savedPaths[0] ?? null;
    if (firstSavedPath) {
      this.lastImagePath = firstSavedPath;
      await this.saveSession();
    }

    const allSavedPaths = successes.flatMap(s => s.savedPaths);
    const totalTokens = successes.reduce((sum, s) => sum + (s.tokenUsage?.total ?? 0), 0);
    const promptTokens = successes.reduce((sum, s) => sum + (s.tokenUsage?.prompt ?? 0), 0);
    const responseTokens = successes.reduce((sum, s) => sum + (s.tokenUsage?.response ?? 0), 0);
    const tokenUsage = totalTokens > 0 ? { total: totalTokens, prompt: promptTokens, response: responseTokens } : undefined;

    const lines: string[] = [];
    if (successes.length === count) {
      lines.push(`Generated ${count}/${count} images successfully.\n`);
    } else {
      lines.push(`Generated ${successes.length}/${count} images (${failureMessages.length} failed).\n`);
    }

    lines.push(`Prompt: "${args.prompt}"`);

    if (allSavedPaths.length > 0) {
      lines.push(`\n📁 Images saved to:\n${allSavedPaths.map(p => `- ${p}`).join("\n")}`);
      lines.push(`\n💡 First image set as session target — use continue_editing to iterate on it.`);
    }

    if (failureMessages.length > 0) {
      lines.push(`\n⚠️  Failed generations:\n${failureMessages.map(m => `- ${m}`).join("\n")}`);
    }

    if (tokenUsage) {
      const inputCost = (tokenUsage.prompt / 1_000_000) * CONSTANTS.PRICING_INPUT_PER_M;
      const outputCost = (tokenUsage.response / 1_000_000) * CONSTANTS.PRICING_OUTPUT_PER_M;
      const totalCost = inputCost + outputCost;
      const costStr = totalCost < 0.001 ? "<$0.001" : `~$${totalCost.toFixed(3)}`;
      lines.push(`\n📊 Total tokens: ${tokenUsage.total.toLocaleString()} (prompt: ${tokenUsage.prompt.toLocaleString()}, response: ${tokenUsage.response.toLocaleString()}) — est. ${costStr}`);
    }

    const content: McpContent[] = [
      { type: "text", text: lines.join("\n") },
      ...successes.flatMap(s => s.imageContent),
    ];
    return { content };
  }

  private async opEditImage(args: { imagePath: string; prompt: string; referenceImages?: string[] }): Promise<CallToolResult> {
    this.ensureConfigured();
    this.validateImagePath(args.imagePath);
    const prompt = this.applyPromptAffix(args.prompt);
    log.debug("edit_image: loading main image", { path: args.imagePath, mimeType: this.getMimeType(args.imagePath) });

    try {
      const mainBytes = await fs.readFile(args.imagePath);
      log.debug("edit_image: main image loaded", { sizeKb: Math.round(mainBytes.length / 1024) });

      const imageParts: ContentPart[] = [
        { inlineData: { data: mainBytes.toString("base64"), mimeType: this.getMimeType(args.imagePath) } },
      ];

      const refPaths = args.referenceImages ?? [];
      for (let i = 0; i < refPaths.length; i++) {
        const refPath = refPaths[i];
        try {
          this.validateImagePath(refPath);
          const refBytes = await fs.readFile(refPath);
          log.debug(`edit_image: loaded reference image ${i + 1}/${refPaths.length}`, { path: refPath, sizeKb: Math.round(refBytes.length / 1024) });
          imageParts.push({ inlineData: { data: refBytes.toString("base64"), mimeType: this.getMimeType(refPath) } });
        } catch (error) {
          log.warn("Could not load reference image — skipping", {
            path: refPath,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }

      imageParts.push({ text: prompt });
      log.debug("edit_image: calling Gemini API", { model: this.config!.model, timeoutMs: this.config!.timeoutMs, imageParts: imageParts.length });

      const response = await this.withRetry(
        () => this.withTimeout(
          this.genAI!.models.generateContent({ model: this.config!.model, contents: [{ parts: imageParts }] }),
          this.config!.timeoutMs,
          "edit_image"
        ),
        "edit_image"
      );

      const content: McpContent[] = [];
      const savedFiles: string[] = [];
      let textContent = "";

      const imagesDir = this.getImagesDirectory();
      await fs.mkdir(imagesDir, { recursive: true, mode: 0o755 });

      const usageMetadata = response.usageMetadata;
      const tokenUsage = usageMetadata
        ? { total: usageMetadata.totalTokenCount ?? 0, prompt: usageMetadata.promptTokenCount ?? 0, response: usageMetadata.candidatesTokenCount ?? 0 }
        : undefined;

      for (const part of response.candidates?.[0]?.content?.parts ?? []) {
        if (part.text) {
          log.debug("edit_image: received text part");
          textContent += part.text;
        }

        if (part.inlineData?.data) {
          const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
          const randomId = Math.random().toString(36).substring(2, 8);
          const ext = this.config!.outputFormat === "jpeg" ? "jpg" : this.config!.outputFormat;
          const filePath = path.join(imagesDir, `edited-${timestamp}-${randomId}.${ext}`);

          const imageBytes = Buffer.from(part.inlineData.data, "base64");
          await fs.writeFile(filePath, imageBytes);
          savedFiles.push(filePath);
          this.lastImagePath = filePath;
          log.info("Image saved", { path: filePath, sizeKb: Math.round(imageBytes.length / 1024) });

          await this.writeImageSidecar(filePath, {
            operation: "edit",
            prompt: args.prompt,
            model: this.config!.model,
            timestamp: new Date().toISOString(),
            tokenUsage,
            sourceImage: args.imagePath,
          });

          content.push({ type: "image", data: part.inlineData.data, mimeType: part.inlineData.mimeType || "image/png" });
        }
      }

      if (tokenUsage) log.debug("edit_image: token usage", tokenUsage);
      if (savedFiles.length > 0) await this.saveSession();

      content.unshift({ type: "text", text: this.buildSuccessResponse({ operation: "edited", prompt: args.prompt, savedFiles, textContent, originalPath: args.imagePath, referenceImages: args.referenceImages, tokenUsage }) });
      return { content };
    } catch (error) {
      log.error("edit_image failed", { error: error instanceof Error ? error.message : String(error) });
      throw classifyApiError(error, "edit_image");
    }
  }

  private async opContinueEditing(args: { prompt: string; referenceImages?: string[] }): Promise<CallToolResult> {
    this.ensureConfigured();

    if (!this.lastImagePath) {
      throw new McpError(ErrorCode.InvalidRequest, "No previous image found. Generate or edit an image first, then use continue_editing.");
    }

    try {
      await fs.access(this.lastImagePath);
    } catch {
      throw new McpError(ErrorCode.InvalidRequest, `Last image not found at: ${this.lastImagePath}. Please generate a new image first.`);
    }

    return this.opEditImage({ imagePath: this.lastImagePath, prompt: args.prompt, referenceImages: args.referenceImages });
  }

  private async opDeleteImage(args: { imagePath: string }): Promise<CallToolResult> {
    const imagesDir = this.getImagesDirectory();
    const resolved = path.resolve(args.imagePath);

    if (!resolved.startsWith(imagesDir)) {
      throw new McpError(ErrorCode.InvalidParams,
        `Can only delete images inside the output directory: ${imagesDir}`);
    }

    try {
      await fs.access(resolved);
    } catch {
      throw new McpError(ErrorCode.InvalidParams, `File not found: ${args.imagePath}`);
    }

    await fs.unlink(resolved);
    log.info("Image deleted", { path: resolved });

    const sc = this.sidecarPath(resolved);
    try {
      await fs.unlink(sc);
      log.debug("Sidecar deleted", { path: sc });
    } catch {
      // No sidecar is fine
    }

    let note = "";
    if (this.lastImagePath === resolved) {
      this.lastImagePath = null;
      await this.saveSession();
      note = "\n\nThis was the last image in the session — the session pointer has been cleared.";
    }

    return {
      content: [{ type: "text", text: `Deleted: ${resolved}${note}` }],
    };
  }

  private async opEnhancePrompt(args: { prompt: string; style?: string }): Promise<CallToolResult> {
    this.ensureConfigured();
    log.debug("enhance_prompt: calling Gemini text API", { model: CONSTANTS.TEXT_MODEL });

    const styleHint = args.style ? ` Style: ${args.style}.` : "";
    const instruction = `You are an expert image prompt engineer. Expand and improve the user's rough prompt into a detailed, high-quality image generation prompt that will produce excellent results.${styleHint} Return ONLY the improved prompt — no explanations, no preamble, no quotes.`;

    try {
      const response = await this.withRetry(
        () => this.withTimeout(
          this.genAI!.models.generateContent({
            model: CONSTANTS.TEXT_MODEL,
            contents: `${instruction}\n\nUser prompt: ${args.prompt}`,
          }),
          this.config!.timeoutMs,
          "enhance_prompt"
        ),
        "enhance_prompt"
      );

      const enhanced = (response.candidates?.[0]?.content?.parts ?? [])
        .filter((p): p is { text: string } => typeof p.text === "string")
        .map(p => p.text)
        .join("")
        .trim();

      if (!enhanced) {
        throw new McpError(ErrorCode.InternalError, "enhance_prompt returned no text — try again.");
      }

      log.info("Prompt enhanced", { originalLength: args.prompt.length, enhancedLength: enhanced.length });

      return {
        content: [{
          type: "text",
          text: `Enhanced prompt:\n\n${enhanced}\n\nUse with: execute({operation: "generate_image", arguments: {prompt: "<paste above>"}})`,
        }],
      };
    } catch (error) {
      log.error("enhance_prompt failed", { error: error instanceof Error ? error.message : String(error) });
      throw classifyApiError(error, "enhance_prompt");
    }
  }

  private async opGetConfigurationStatus(): Promise<CallToolResult> {
    const isConfigured = this.config !== null && this.genAI !== null;
    let text: string;

    if (isConfigured) {
      const sourceNote =
        this.configSource === "environment"
          ? "\n📍 Source: Environment variable (GEMINI_API_KEY)"
          : "\n📍 Source: Local config file (.nano-banana-config.json) — consider using environment variables for better security.";

      text = [
        "✅ Gemini API token is configured and ready to use",
        sourceNote,
        `\n⚙️  Model: ${this.config!.model}`,
        `📂 Output directory: ${this.getImagesDirectory()}`,
        `🖼️  Output format: ${this.config!.outputFormat}`,
        `⏱️  Timeout: ${(this.config!.timeoutMs / 1000).toFixed(0)}s`,
        `✏️  Prompt prefix: ${this.config!.promptPrefix || "(none)"}`,
        `✏️  Prompt suffix: ${this.config!.promptSuffix || "(none)"}`,
        `📋 Log level: ${log.levelName} (set NANO_BANANA_LOG_LEVEL=DEBUG|INFO|WARN|ERROR|SILENT)`,
      ].join("\n");
    } else {
      text = [
        "❌ Gemini API key is not configured.",
        "",
        "Configure it via (in priority order):",
        '1. MCP env: "env": { "GEMINI_API_KEY": "your-key" }   ← recommended',
        "2. Shell env: export GEMINI_API_KEY=your-key",
        '3. Tool: execute({operation: "configure_gemini_token", arguments: {apiKey: "your-key"}})',
      ].join("\n");
    }

    return { content: [{ type: "text", text }] };
  }

  private async opGetLastImageInfo(): Promise<CallToolResult> {
    if (!this.lastImagePath) {
      return {
        content: [{ type: "text", text: "No image has been generated or edited in this session yet." }],
      };
    }

    try {
      const stats = await fs.stat(this.lastImagePath);
      const sidecar = await this.readImageSidecar(this.lastImagePath);
      let text = `Last image:\n  Path: ${this.lastImagePath}\n  Size: ${Math.round(stats.size / 1024)} KB\n  Modified: ${stats.mtime.toLocaleString()}`;
      if (sidecar) {
        text += `\n  Operation: ${sidecar.operation}\n  Prompt: "${sidecar.prompt}"`;
        if (sidecar.sourceImage) text += `\n  Source: ${sidecar.sourceImage}`;
      }
      return { content: [{ type: "text", text }] };
    } catch {
      return {
        content: [{
          type: "text",
          text: `Last image path: ${this.lastImagePath}\nStatus: file not found — it may have been moved or deleted.`,
        }],
      };
    }
  }

  private async opListGeneratedImages(args: { limit?: number }): Promise<CallToolResult> {
    const limit = args.limit ?? 20;
    const imagesDir = this.getImagesDirectory();

    try {
      await fs.access(imagesDir);
    } catch {
      return {
        content: [{ type: "text", text: `No output directory found at: ${imagesDir}\n\nGenerate an image first to create it.` }],
      };
    }

    const entries = await fs.readdir(imagesDir, { withFileTypes: true });
    const imageFiles = entries
      .filter(e => e.isFile() && /\.(png|jpe?g|webp)$/i.test(e.name))
      .map(e => e.name)
      .sort((a, b) => b.localeCompare(a));

    const sliced = imageFiles.slice(0, limit);

    if (sliced.length === 0) {
      return { content: [{ type: "text", text: `No images found in ${imagesDir}.` }] };
    }

    const details = await Promise.all(
      sliced.map(async (name) => {
        const fullPath = path.join(imagesDir, name);
        const stats = await fs.stat(fullPath);
        const marker = fullPath === this.lastImagePath ? " ← last" : "";
        const sidecar = await this.readImageSidecar(fullPath);
        const promptSnippet = sidecar?.prompt
          ? `\n    prompt: "${sidecar.prompt.length > 60 ? sidecar.prompt.substring(0, 60) + "…" : sidecar.prompt}"`
          : "";
        return `- ${fullPath} (${Math.round(stats.size / 1024)} KB, ${stats.mtime.toLocaleString()})${marker}${promptSnippet}`;
      })
    );

    const header = imageFiles.length > limit
      ? `Showing ${sliced.length} of ${imageFiles.length} image(s) in ${imagesDir} (newest first):`
      : `Found ${sliced.length} image(s) in ${imagesDir}:`;

    return { content: [{ type: "text", text: `${header}\n\n${details.join("\n")}` }] };
  }

  private async opGetImageHistory(args: { imagePath: string }): Promise<CallToolResult> {
    type ChainEntry = { filePath: string; sidecar: ImageSidecar | null; exists: boolean };
    const chain: ChainEntry[] = [];
    const visited = new Set<string>();
    let current = path.resolve(args.imagePath);

    while (current && !visited.has(current)) {
      visited.add(current);

      let exists = true;
      try { await fs.access(current); } catch { exists = false; }

      const sidecar = exists ? await this.readImageSidecar(current) : null;
      chain.push({ filePath: current, sidecar, exists });

      const next = sidecar?.sourceImage ? path.resolve(sidecar.sourceImage) : null;
      if (!next) break;
      current = next;
    }

    // Oldest first
    chain.reverse();

    if (chain.length === 1 && !chain[0].sidecar) {
      return {
        content: [{
          type: "text",
          text: `No history found for: ${args.imagePath}\n\nThis image has no metadata sidecar — it may have been generated before v1.2.0, or the sidecar was deleted.`,
        }],
      };
    }

    const targetPath = path.resolve(args.imagePath);
    const lines: string[] = [
      `Edit history for: ${args.imagePath}`,
      `${chain.length} image(s) in chain:`,
      "",
    ];

    chain.forEach(({ filePath, sidecar, exists }, i) => {
      const label = i === 0 ? "Original" : `Edit ${i}`;
      const currentMarker = filePath === targetPath ? " ← current" : "";
      const missingMarker = exists ? "" : " (file missing)";
      lines.push(`${i + 1}. [${label}]${currentMarker}${missingMarker}`);
      lines.push(`   Path: ${filePath}`);
      if (sidecar) {
        lines.push(`   Prompt: "${sidecar.prompt}"`);
        lines.push(`   Model: ${sidecar.model}`);
        lines.push(`   Time: ${new Date(sidecar.timestamp).toLocaleString()}`);
        if (sidecar.tokenUsage?.total) {
          lines.push(`   Tokens: ${sidecar.tokenUsage.total.toLocaleString()}`);
        }
      } else {
        lines.push("   (no metadata available)");
      }
      lines.push("");
    });

    return { content: [{ type: "text", text: lines.join("\n") }] };
  }

  public async run(): Promise<void> {
    log.info("nano-banana-mcp v1.3.0 starting", { logLevel: log.levelName });
    await this.loadConfig();
    await this.loadSession();
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    log.info("Server ready");
  }
}
