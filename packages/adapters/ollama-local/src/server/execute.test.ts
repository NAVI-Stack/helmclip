import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { execute } from './execute.js';
import type { AdapterExecutionContext } from '@paperclipai/adapter-utils';

describe('ollama_local execute', () => {
  const mockCtx: AdapterExecutionContext = {
    runId: 'test-run',
    agent: {
      id: 'test-agent',
      companyId: 'test-company',
      name: 'Test Agent',
    } as any,
    runtime: {
      sessionParams: {},
    } as any,
    config: {
      model: 'llama3.2',
    },
    context: {
      paperclipWake: {
        reason: 'manual',
      },
    } as any,
    onLog: vi.fn(),
    onMeta: vi.fn(),
  };

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('should execute a simple chat without tools', async () => {
    const mockFetch = vi.mocked(fetch);

    // Mock /api/tags
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: [{ name: 'llama3.2:latest' }] }),
    } as Response);

    // Mock /api/chat stream
    const chunks = [
      JSON.stringify({ message: { role: 'assistant', content: 'Hello' }, done: false }),
      JSON.stringify({ message: { role: 'assistant', content: ' world' }, done: true, prompt_eval_count: 10, eval_count: 5 }),
    ];

    const stream = new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(new TextEncoder().encode(chunk + '\n'));
        }
        controller.close();
      },
    });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: stream,
    } as Response);

    const result = await execute(mockCtx);

    expect(result.exitCode).toBe(0);
    expect(result.summary).toBe('Hello world');
    expect(result.usage).toEqual({ inputTokens: 10, outputTokens: 5 });
    expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/api/chat'), expect.any(Object));
  });

  it('should handle tool calls', async () => {
    const mockFetch = vi.mocked(fetch);

    // Mock /api/tags
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: [{ name: 'llama3.2:latest' }] }),
    } as Response);

    // Turn 1: Assistant calls tool
    const turn1Chunks = [
      JSON.stringify({
        message: {
          role: 'assistant',
          content: 'Thinking...',
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: { name: 'run_shell_command', arguments: '{"command": "echo hello"}' },
            },
          ],
        },
        done: true,
      }),
    ];

    // Turn 2: Assistant responds with tool result
    const turn2Chunks = [
      JSON.stringify({
        message: { role: 'assistant', content: 'The output was hello' },
        done: true,
      }),
    ];

    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        body: new ReadableStream({
          start(controller) {
            for (const chunk of turn1Chunks) controller.enqueue(new TextEncoder().encode(chunk + '\n'));
            controller.close();
          },
        }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        body: new ReadableStream({
          start(controller) {
            for (const chunk of turn2Chunks) controller.enqueue(new TextEncoder().encode(chunk + '\n'));
            controller.close();
          },
        }),
      } as Response);

    const result = await execute(mockCtx);

    expect(result.exitCode).toBe(0);
    expect(result.summary).toBe('The output was hello');
    // Verify tool was "executed" (mockCtx.onLog should have been called)
    expect(mockCtx.onLog).toHaveBeenCalledWith('stdout', expect.stringContaining('Executing: echo hello'));
  });

  it('should preserve tool-related roles in sessionParams', async () => {
    const mockFetch = vi.mocked(fetch);

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: [{ name: 'llama3.2:latest' }] }),
    } as Response);

    const chunks = [
      JSON.stringify({ message: { role: 'assistant', content: 'Done' }, done: true }),
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: new ReadableStream({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk + '\n'));
          controller.close();
        },
      }),
    } as Response);

    const result = await execute({
      ...mockCtx,
      runtime: {
        sessionParams: {
          messages: [
            { role: 'user', content: 'Run echo' },
            { role: 'assistant', content: '', tool_calls: [{ id: '1', type: 'function', function: { name: 'run_shell_command', arguments: '{"command":"echo"}' } }] },
            { role: 'tool', content: 'echo output', tool_call_id: '1' },
          ],
        },
      } as any,
    });

    expect(result.exitCode).toBe(0);
    // sessionParams in result should include the new assistant message AND the prior ones
    const sessionMessages = (result.sessionParams as any).messages;
    expect(sessionMessages.some((m: any) => m.role === 'tool')).toBe(true);
  });
});
