import { describe, expect, it } from "vitest";
import { parseOllamaStdoutLine } from "./parse-stdout.js";

describe("parseOllamaStdoutLine", () => {
  const ts = "2026-06-03T12:00:00.000Z";

  it("parses tool_call events", () => {
    const entries = parseOllamaStdoutLine(
      JSON.stringify({
        type: "tool_call",
        name: "paperclip_api",
        toolCallId: "call-1",
        input: { method: "PATCH", path: "/api/issues/x" },
      }),
      ts,
    );
    expect(entries).toEqual([
      {
        kind: "tool_call",
        ts,
        name: "paperclip_api",
        toolUseId: "call-1",
        input: { method: "PATCH", path: "/api/issues/x" },
      },
    ]);
  });

  it("parses tool_result events", () => {
    const entries = parseOllamaStdoutLine(
      JSON.stringify({
        type: "tool_result",
        toolCallId: "call-1",
        content: "HTTP 200\n{\"status\":\"done\"}",
      }),
      ts,
    );
    expect(entries[0]).toMatchObject({
      kind: "tool_result",
      toolUseId: "call-1",
      isError: false,
    });
  });
});
