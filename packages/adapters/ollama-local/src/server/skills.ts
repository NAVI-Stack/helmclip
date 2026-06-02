import type {
  AdapterSkillContext,
  AdapterSkillSnapshot,
} from "@paperclipai/adapter-utils";
import { readPaperclipRuntimeSkillEntries } from "@paperclipai/adapter-utils/server-utils";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __moduleDir = path.dirname(fileURLToPath(import.meta.url));

export async function listOllamaSkills(
  ctx: AdapterSkillContext,
): Promise<AdapterSkillSnapshot> {
  const availableEntries = await readPaperclipRuntimeSkillEntries(
    ctx.config,
    __moduleDir,
  );
  const desiredSkills = Array.isArray(ctx.config.paperclipDesiredSkills)
    ? ctx.config.paperclipDesiredSkills.filter((s): s is string => typeof s === "string")
    : [];

  return {
    adapterType: "ollama_local",
    supported: true,
    mode: "ephemeral",
    desiredSkills,
    entries: availableEntries.map((entry) => ({
      ...entry,
      managed: true,
      state: desiredSkills.includes(entry.key) ? "installed" : "available",
    })),
    warnings: [],
  };
}

export async function syncOllamaSkills(
  ctx: AdapterSkillContext,
  desiredSkills: string[],
): Promise<AdapterSkillSnapshot> {
  // Ephemeral mode doesn't need persistent sync, just return the list with updated state
  return await listOllamaSkills({
    ...ctx,
    config: { ...ctx.config, paperclipDesiredSkills: desiredSkills },
  });
}
