export const type = "ollama_local";
export const label = "Ollama (local)";

export const DEFAULT_OLLAMA_MODEL = "llama3.2";
export const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434";

export const models = [
  { id: DEFAULT_OLLAMA_MODEL, label: "Llama 3.2" },
  { id: "llama3.1", label: "Llama 3.1" },
  { id: "codellama", label: "Code Llama" },
  { id: "deepseek-coder-v2", label: "DeepSeek Coder V2" },
  { id: "mistral", label: "Mistral" },
  { id: "phi4", label: "Phi 4" },
  { id: "qwen2.5-coder", label: "Qwen 2.5 Coder" },
];

export const agentConfigurationDoc = `# ollama_local agent configuration

Adapter: ollama_local

Use when:
- You want to run a free/local LLM via Ollama on the host machine
- You need open-source models like Llama, Mistral, CodeLlama, DeepSeek, or Phi
- You have a GPU-equipped machine with Ollama installed and want zero API cost
- Privacy is critical and you cannot send data to external LLM providers
- You want to use any custom or fine-tuned model available in the Ollama registry

Don't use when:
- Ollama is not installed or the model has not been pulled locally

Core fields:
- baseUrl (string, optional): Ollama server base URL. Defaults to http://localhost:11434.
- model (string, optional): Ollama model to use. Defaults to llama3.2. Must be available via \`ollama pull <model>\`.
- instructionsFilePath (string, optional): absolute path to a markdown instructions file (AGENTS.md) injected at runtime.
- promptTemplate (string, optional): run prompt template supporting {{agent.*}}, {{context.*}}, etc.
- system (string, optional): system prompt injected as the first message.
- temperature (number, optional): sampling temperature (0.0–2.0). Uses model default when omitted.

Operational fields:
- timeoutSec (number, optional): run timeout in seconds. Defaults to 300 (5 minutes). Set 0 for no timeout.
- graceSec (number, optional): kept for API compatibility; unused (no subprocess).

Notes:
- Ollama must be running before the agent executes: \`ollama serve\`
- Pull models before first use: \`ollama pull llama3.2\`
- The adapter resolves the base URL in order: \`config.baseUrl\`, \`process.env.OLLAMA_HOST\`, \`process.env.OLLAMA_BASE_URL\`, then defaults to \`http://localhost:11434\`.
- Conversation history is stored in sessionParams and replayed across runs for context continuity.
- This adapter calls the Ollama HTTP API directly (POST /api/chat), not via subprocess.
- No API key is required for local Ollama.
`;
