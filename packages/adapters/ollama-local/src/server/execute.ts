import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  AdapterExecutionContext,
  AdapterExecutionResult,
} from "@paperclipai/adapter-utils";
import {
  readAdapterExecutionTarget,
  runAdapterExecutionTargetProcess,
} from "@paperclipai/adapter-utils/execution-target";
import {
  asNumber,
  asString,
  buildPaperclipEnv,
  parseObject,
  renderTemplate,
  DEFAULT_PAPERCLIP_AGENT_PROMPT_TEMPLATE,
  joinPromptSections,
  renderPaperclipWakePrompt,
  readPaperclipRuntimeSkillEntries,
  readPaperclipSkillMarkdown,
} from "@paperclipai/adapter-utils/server-utils";
import { DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL } from "../index.js";

const __moduleDir = path.dirname(fileURLToPath(import.meta.url));

export interface OllamaMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: OllamaToolCall[];
  tool_call_id?: string; // For 'tool' role
}

export interface OllamaToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string; // JSON string
  };
}

export interface OllamaChunkLine {
  type: "chunk";
  content: string;
  tool_calls?: OllamaToolCall[];
}

export interface OllamaDoneLine {
  type: "done";
  model: string;
  prompt_eval_count: number;
  eval_count: number;
  total_duration_ns: number;
}

export interface OllamaErrorLine {
  type: "error";
  message: string;
}

export type OllamaStdoutLine =
  | OllamaChunkLine
  | OllamaDoneLine
  | OllamaErrorLine;

const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful AI assistant integrated into the Paperclip control plane. Respond concisely and helpfully.";

function renderPaperclipEnvNote(env: Record<string, string>): string {
  const paperclipKeys = Object.keys(env)
    .filter((key) => key.startsWith("PAPERCLIP_"))
    .sort();
  if (paperclipKeys.length === 0) return "";
  return [
    "Paperclip runtime note:",
    `The following PAPERCLIP_* environment variables are available in this run: ${paperclipKeys.join(", ")}`,
    "Do not assume these variables are missing without checking your shell environment.",
    "",
    "",
  ].join("\n");
}

function renderApiAccessNote(env: Record<string, string>): string {
  if (!env.PAPERCLIP_API_URL || !env.PAPERCLIP_API_KEY) return "";
  return [
    "Paperclip API access note:",
    "Use run_shell_command with curl to make Paperclip API requests.",
    "GET example:",
    `  run_shell_command({ command: "curl -s -H \\"Authorization: Bearer $PAPERCLIP_API_KEY\\" \\"$PAPERCLIP_API_URL/api/agents/me\\"" })`,
    "POST/PATCH example:",
    `  run_shell_command({ command: "curl -s -X POST -H \\"Authorization: Bearer $PAPERCLIP_API_KEY\\" -H 'Content-Type: application/json' -H \\"X-Paperclip-Run-Id: $PAPERCLIP_RUN_ID\\" -d '{...}' \\"$PAPERCLIP_API_URL/api/issues/{id}/checkout\\"" })`,
    "",
    "",
  ].join("\n");
}

function buildPaperclipOllamaTools() {
  return [
    {
      type: "function",
      function: {
        name: "run_shell_command",
        description: "Execute a shell command on the host machine. Use for file operations, build/test commands, and Paperclip API calls via curl.",
        parameters: {
          type: "object",
          properties: {
            command: {
              type: "string",
              description: "The shell command to execute.",
            },
          },
          required: ["command"],
        },
      },
    },
  ];
}

/**
 * Try to resolve a possibly-untagged model name (e.g. "llama3.2") to the exact
 * name Ollama has installed (e.g. "llama3.2:3b").  Falls back to the original
 * name if the tags API is unavailable or no match is found.
 */
async function resolveModelName(
  baseUrl: string,
  requested: string,
): Promise<string> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return requested;
    const body = (await res.json()) as Record<string, unknown>;
    if (!Array.isArray(body.models)) return requested;
    const names: string[] = (body.models as Record<string, unknown>[])
      .filter((m) => typeof m.name === "string")
      .map((m) => m.name as string);

    // 1. Exact match
    if (names.includes(requested)) return requested;

    // 2. Exact match ignoring case
    const lower = requested.toLowerCase();
    const exact = names.find((n) => n.toLowerCase() === lower);
    if (exact) return exact;

    // 3. Base-name match (strip tag from both sides)
    const requestedBase = requested.split(":")[0].toLowerCase();
    const baseMatch = names.find(
      (n) => n.split(":")[0].toLowerCase() === requestedBase,
    );
    if (baseMatch) {
      // console.log(`[ollama] Resolved ${requested} to ${baseMatch}`);
      return baseMatch;
    }
  } catch {
    // network error / timeout — continue with original name
  }
  return requested;
}

export async function execute(
  ctx: AdapterExecutionContext,
): Promise<AdapterExecutionResult> {
  const { runId, agent, runtime, config, context, onLog, onMeta } = ctx;

  const baseUrl = asString(config.baseUrl, DEFAULT_OLLAMA_BASE_URL).replace(
    /\/$/,
    "",
  );
  const rawModel = asString(config.model, DEFAULT_OLLAMA_MODEL).trim();
  const timeoutSec = asNumber(config.timeoutSec, 300);
  const temperature =
    typeof config.temperature === "number" && Number.isFinite(config.temperature)
      ? config.temperature
      : undefined;
  const systemPromptConfig = asString(config.system, DEFAULT_SYSTEM_PROMPT);
  const maxTurns = asNumber(config.maxTurnsPerRun, 20);

  const executionTarget = readAdapterExecutionTarget({
    executionTarget: ctx.executionTarget,
    legacyRemoteExecution: ctx.executionTransport?.remoteExecution,
  });

  // Resolve the model name against what Ollama actually has installed.
  const model = await resolveModelName(baseUrl, rawModel);
  if (model !== rawModel) {
    await onLog("stdout", `[paperclip] Resolved model "${rawModel}" to "${model}"\n`);
  }

  const instructionsFilePath = asString(config.instructionsFilePath, "").trim();
  const instructionsDir = instructionsFilePath ? `${path.dirname(instructionsFilePath)}/` : "";
  let instructionsPrefix = "";
  if (instructionsFilePath) {
    try {
      const instructionsContents = await fs.readFile(instructionsFilePath, "utf8");
      instructionsPrefix =
        `${instructionsContents}\n\n` +
        `The above agent instructions were loaded from ${instructionsFilePath}. ` +
        `Resolve any relative file references from ${instructionsDir}.\n\n`;
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      await onLog(
        "stderr",
        `[paperclip] Warning: could not read agent instructions file "${instructionsFilePath}": ${reason}\n`,
      );
    }
  }

  const promptTemplate = asString(
    config.promptTemplate,
    DEFAULT_PAPERCLIP_AGENT_PROMPT_TEMPLATE,
  );
  const templateData = {
    agentId: agent.id,
    companyId: agent.companyId,
    runId,
    company: { id: agent.companyId },
    agent,
    run: { id: runId },
    context,
  };
  const renderedPrompt = renderTemplate(promptTemplate, templateData);
// Rehydrate prior conversation history from session
const sessionParams = parseObject(runtime.sessionParams);
const priorMessages: OllamaMessage[] = (() => {
  if (!Array.isArray(sessionParams.messages)) return [];
  return (sessionParams.messages as unknown[]).filter(
    (m): m is OllamaMessage =>
      typeof m === "object" &&
      m !== null &&
      !Array.isArray(m) &&
      ["system", "user", "assistant", "tool"].includes((m as any).role) &&
      typeof (m as any).content === "string",
  );
})();

  const resumedSession = priorMessages.length > 0;

  const wakePrompt = renderPaperclipWakePrompt(context.paperclipWake, { resumedSession });
  const paperclipEnv = buildPaperclipEnv(agent);
  const paperclipEnvNote = renderPaperclipEnvNote(paperclipEnv);
  const apiAccessNote = renderApiAccessNote(paperclipEnv);

  // If resuming, we omit the heavy instructions and prompt template if a wake prompt exists,
  // as the model already has the context in its session history.
  const shouldUseResumeDeltaPrompt = resumedSession && wakePrompt.length > 0;
  const finalInstructionsPrefix = shouldUseResumeDeltaPrompt ? "" : instructionsPrefix;
  const finalRenderedPrompt = shouldUseResumeDeltaPrompt ? "" : renderedPrompt;

  const desiredSkillNames = Array.isArray(config.paperclipDesiredSkills)
    ? config.paperclipDesiredSkills.filter((s): s is string => typeof s === "string")
    : [];
  
  const skillPrompts: string[] = [];
  if (desiredSkillNames.length > 0) {
    for (const skillName of desiredSkillNames) {
      const markdown = await readPaperclipSkillMarkdown(__moduleDir, skillName);
      if (markdown) {
        skillPrompts.push(`### Skill: ${skillName}\n\n${markdown}`);
      }
    }
  }
  const skillsPrefix = skillPrompts.length > 0 
    ? "## Available Skills\n\n" + skillPrompts.join("\n\n") + "\n\n"
    : "";

  const prompt = joinPromptSections([
    finalInstructionsPrefix,
    skillsPrefix,
    wakePrompt,
    paperclipEnvNote,
    apiAccessNote,
    finalRenderedPrompt,
  ]);

  const messages: OllamaMessage[] = [
    { role: "system", content: systemPromptConfig },
    ...priorMessages,
    { role: "user", content: prompt },
  ];

  if (onMeta) {
    await onMeta({
      adapterType: "ollama_local",
      command: `POST ${baseUrl}/api/chat`,
      cwd: process.cwd(),
      commandNotes: [
        `Model: ${model}`,
        `Prior conversation turns: ${Math.floor(priorMessages.length / 2)}`,
        `Max turns: ${maxTurns}`,
        `Streaming: true`,
        ...(instructionsFilePath ? [`Loaded instructions from ${instructionsFilePath}`] : []),
      ],
      commandArgs: [],
      env: paperclipEnv,
      prompt,
      promptMetrics: {
        promptChars: prompt.length,
        heartbeatPromptChars: renderedPrompt.length,
      },
      context,
    });
  }

  const controller = new AbortController();
  let timedOut = false;
  const timeoutHandle =
    timeoutSec > 0
      ? setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, timeoutSec * 1000)
      : null;

  let totalPromptEvalCount = 0;
  let totalEvalCount = 0;
  let turnCount = 0;
  let finalAssistantSummary = "";

  try {
    while (turnCount < maxTurns) {
      turnCount++;
      let assistantContent = "";
      let toolCalls: OllamaToolCall[] = [];

      const requestBody: Record<string, unknown> = {
        model,
        messages,
        stream: true,
        tools: buildPaperclipOllamaTools(),
      };
      if (temperature !== undefined) {
        requestBody.options = { temperature };
      }

      const response = await fetch(`${baseUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      if (!response.ok) {
        const bodyText = await response.text().catch(() => "");
        const errMsg = bodyText.trim() || `HTTP ${response.status} ${response.statusText}`;
        throw new Error(`Ollama returned ${response.status}: ${errMsg}`);
      }

      if (!response.body) throw new Error("Ollama response has no body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line) continue;

          let parsed: Record<string, unknown>;
          try {
            parsed = JSON.parse(line) as Record<string, unknown>;
          } catch {
            await onLog("stdout", line + "\n");
            continue;
          }

          if (parsed.error) {
            const errorMsg = typeof parsed.error === "string" ? parsed.error : JSON.stringify(parsed.error);
            throw new Error(`Ollama error: ${errorMsg}`);
          }

          const isDone = parsed.done === true;
          const messageObj =
            typeof parsed.message === "object" && parsed.message !== null
              ? (parsed.message as Record<string, unknown>)
              : null;
          
          if (messageObj) {
            const contentChunk = typeof messageObj.content === "string" ? messageObj.content : "";
            if (contentChunk) {
              assistantContent += contentChunk;
              await onLog("stdout", JSON.stringify({ type: "chunk", content: contentChunk }) + "\n");
            }

            if (Array.isArray(messageObj.tool_calls)) {
              for (const tc of messageObj.tool_calls) {
                const toolCall = tc as OllamaToolCall;
                // Avoid duplicates if Ollama sends the same tool call in multiple chunks (rare but possible)
                if (!toolCalls.some(existing => existing.id === toolCall.id)) {
                  toolCalls.push(toolCall);
                  await onLog("stdout", JSON.stringify({
                    type: "tool_call",
                    name: toolCall.function.name,
                    toolCallId: toolCall.id,
                    input: toolCall.function.arguments,
                  }) + "\n");
                }
              }
            }
          }

          if (isDone) {
            totalPromptEvalCount += typeof parsed.prompt_eval_count === "number" ? parsed.prompt_eval_count : 0;
            totalEvalCount += typeof parsed.eval_count === "number" ? parsed.eval_count : 0;
            const doneLine: OllamaDoneLine = {
              type: "done",
              model: typeof parsed.model === "string" ? parsed.model : model,
              prompt_eval_count: typeof parsed.prompt_eval_count === "number" ? parsed.prompt_eval_count : totalPromptEvalCount,
              eval_count: typeof parsed.eval_count === "number" ? parsed.eval_count : totalEvalCount,
              total_duration_ns: typeof parsed.total_duration === "number" ? parsed.total_duration : 0,
            };
            await onLog("stdout", JSON.stringify(doneLine) + "\n");
          }
        }
      }

      messages.push({ role: "assistant", content: assistantContent, tool_calls: toolCalls.length > 0 ? toolCalls : undefined });
      finalAssistantSummary = assistantContent;

      if (toolCalls.length === 0) {
        break; // No more tools, agent is done for this heartbeat
      }

      // Execute tool calls
      for (const toolCall of toolCalls) {
        if (toolCall.function.name === "run_shell_command") {
          let args: Record<string, unknown>;
          try {
            args = JSON.parse(toolCall.function.arguments);
          } catch (err) {
            const errorMsg = `Failed to parse tool arguments: ${err instanceof Error ? err.message : String(err)}`;
            messages.push({ role: "tool", content: errorMsg, tool_call_id: toolCall.id });
            await onLog("stderr", `[paperclip] ${errorMsg}\n`);
            continue;
          }

          const command = asString(args.command, "");
          if (!command) {
            const errorMsg = "Missing 'command' argument for run_shell_command.";
            messages.push({ role: "tool", content: errorMsg, tool_call_id: toolCall.id });
            continue;
          }

          await onLog("stdout", `[paperclip] Executing: ${command}\n`);
          const proc = await runAdapterExecutionTargetProcess(runId, executionTarget, "sh", ["-c", command], {
            cwd: process.cwd(),
            env: { ...process.env, ...paperclipEnv },
            timeoutSec: 60, // Individual tool timeout
            onLog,
          });

          const result = `Exit code: ${proc.exitCode}\nSTDOUT:\n${proc.stdout}\nSTDERR:\n${proc.stderr}`;
          messages.push({ role: "tool", content: result, tool_call_id: toolCall.id });
          await onLog("stdout", JSON.stringify({ type: "tool_result", toolCallId: toolCall.id, content: result }) + "\n");
        } else {
          const errorMsg = `Unknown tool: ${toolCall.function.name}`;
          messages.push({ role: "tool", content: errorMsg, tool_call_id: toolCall.id });
          await onLog("stderr", `[paperclip] ${errorMsg}\n`);
        }
      }
    }
  } catch (err) {
    if (timeoutHandle) clearTimeout(timeoutHandle);
    if (timedOut) {
      return {
        exitCode: null,
        signal: null,
        timedOut: true,
        errorMessage: `Timed out after ${timeoutSec}s`,
        provider: "ollama",
        model,
      };
    }
    const msg = err instanceof Error ? err.message : String(err);
    return {
      exitCode: 1,
      signal: null,
      timedOut: false,
      errorMessage: msg,
      provider: "ollama",
      model,
    };
  } finally {
    if (timeoutHandle) clearTimeout(timeoutHandle);
  }

  // Build updated session params with appended message history (strip system prompt from session)
  const updatedMessages = messages.slice(1);

  return {
    exitCode: 0,
    signal: null,
    timedOut: false,
    usage: { inputTokens: totalPromptEvalCount, outputTokens: totalEvalCount },
    provider: "ollama",
    model,
    billingType: "subscription",
    sessionParams: updatedMessages.length > 0 ? { messages: updatedMessages } : null,
    summary: finalAssistantSummary.trim() || null,
    resultJson: {
      turns: turnCount,
      // If we want the runtime to see a specific status, we can try to parse it from the summary
      // but usually the agent will have called the Paperclip API if it needed to.
    },
  };
}
