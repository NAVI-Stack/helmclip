import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  execute,
  executePaperclipApiCall,
  isAllowedPaperclipApiPath,
  isDispositionRecoveryContext,
} from './execute.js';
import type { AdapterExecutionContext } from '@paperclipai/adapter-utils';

vi.mock('@paperclipai/adapter-utils/execution-target', () => ({
  readAdapterExecutionTarget: vi.fn(() => ({ kind: 'local' })),
  runAdapterExecutionTargetProcess: vi.fn().mockResolvedValue({
    exitCode: 0,
    stdout: 'hello\n',
    stderr: '',
  }),
}));

describe('ollama disposition helpers', () => {
  it('detects disposition recovery context from handoff wake reason', () => {
    expect(
      isDispositionRecoveryContext({
        paperclipWake: { reason: 'finish_successful_run_handoff' },
      }),
    ).toBe(true);
    expect(isDispositionRecoveryContext({ handoffRequired: true })).toBe(true);
    expect(isDispositionRecoveryContext({ recoveryIntent: 'status_only' })).toBe(true);
    expect(isDispositionRecoveryContext({ recoveryIntent: 'status_only', handoffRequired: true })).toBe(true);
    expect(isDispositionRecoveryContext({ paperclipWake: { reason: 'issue_assigned' } })).toBe(false);
  });

  it('restricts paperclip_api paths to allowed api roots', () => {
    expect(isAllowedPaperclipApiPath('/api/issues/issue-1')).toBe(true);
    expect(isAllowedPaperclipApiPath('/api/issues/issue-1/checkout')).toBe(true);
    expect(isAllowedPaperclipApiPath('/api/agents/me')).toBe(true);
    expect(isAllowedPaperclipApiPath('/api/companies/some-company')).toBe(true);
    expect(isAllowedPaperclipApiPath('/other/path')).toBe(false);
  });

  it('executes paperclip_api via fetch', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '{"status":"done"}',
    });
    vi.stubGlobal('fetch', mockFetch);
    process.env.PAPERCLIP_API_KEY = 'test-key';

    const result = await executePaperclipApiCall({
      method: 'PATCH',
      path: '/api/issues/issue-1',
      body: { status: 'done', comment: 'done' },
      runId: 'run-1',
      paperclipEnv: { PAPERCLIP_API_URL: 'http://localhost:3100' },
    });

    expect(result).toContain('HTTP 200');
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:3100/api/issues/issue-1',
      expect.objectContaining({ method: 'PATCH' }),
    );
  });
});

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

  it('includes disposition appendix and paperclip_api on handoff recovery wakes', async () => {
    const mockFetch = vi.mocked(fetch);
    process.env.PAPERCLIP_API_KEY = 'test-key';

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: [{ name: 'llama3.2:latest' }] }),
    } as Response);

    const chunks = [
      JSON.stringify({ message: { role: 'assistant', content: 'Marking done.' }, done: true }),
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

    let capturedPrompt = '';
    const result = await execute({
      ...mockCtx,
      context: {
        paperclipWake: {
          reason: 'finish_successful_run_handoff',
          issue: { id: 'issue-1', identifier: 'PAP-1', title: 'Test', status: 'in_progress' },
          livenessContinuation: {
            instruction: 'Choose exactly one disposition.',
          },
          commentWindow: { requestedCount: 0, includedCount: 0, missingCount: 0 },
          comments: [],
          fallbackFetchNeeded: false,
        },
        issueId: 'issue-1',
        handoffRequired: true,
      } as any,
      onMeta: vi.fn(async (meta) => {
        capturedPrompt = String(meta.prompt ?? '');
      }),
      runtime: {
        sessionParams: {
          messages: [{ role: 'user', content: 'prior turn' }, { role: 'assistant', content: 'prior reply' }],
        },
      } as any,
    });

    expect(result.exitCode).toBe(0);
    expect(capturedPrompt).toContain('## Disposition required');
    expect(capturedPrompt).toContain('paperclip_api');
    expect(capturedPrompt).toContain('/api/issues/issue-1');
    expect(capturedPrompt).toContain('Choose exactly one disposition');
    expect(capturedPrompt).toContain('Execution contract:');
    const chatBody = JSON.parse(
      String(mockFetch.mock.calls.find((call) => String(call[0]).includes('/api/chat'))?.[1]?.body ?? '{}'),
    );
    const toolNames = (chatBody.tools as Array<{ function: { name: string } }>).map((t) => t.function.name);
    expect(toolNames).toContain('paperclip_api');
  });

  it('handles paperclip_api tool calls', async () => {
    const mockFetch = vi.mocked(fetch);
    process.env.PAPERCLIP_API_KEY = 'test-key';

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ models: [{ name: 'llama3.2:latest' }] }),
    } as Response);

    const turn1Chunks = [
      JSON.stringify({
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_api',
              type: 'function',
              function: {
                name: 'paperclip_api',
                arguments: JSON.stringify({
                  method: 'PATCH',
                  path: '/api/issues/issue-1',
                  body: { status: 'done', comment: 'Finished' },
                }),
              },
            },
          ],
        },
        done: true,
      }),
    ];

    const turn2Chunks = [
      JSON.stringify({ message: { role: 'assistant', content: 'Issue marked done.' }, done: true }),
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
        status: 200,
        text: async () => '{"id":"issue-1","status":"done"}',
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
    expect(result.summary).toBe('Issue marked done.');
    expect(mockCtx.onLog).toHaveBeenCalledWith(
      'stdout',
      expect.stringContaining('paperclip_api PATCH /api/issues/issue-1'),
    );
  });
});
