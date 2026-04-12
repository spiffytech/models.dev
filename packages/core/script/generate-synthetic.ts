#!/usr/bin/env bun

import { z } from "zod";
import path from "node:path";
import { mkdir } from "node:fs/promises";

import type { Model } from "../src/schema";
import type { ModelFamily } from "../src/family";

const modelsDir = path.join(
  import.meta.dirname,
  "..",
  "..",
  "..",
  "providers",
  "synthetic",
  "models"
);

type SyntheticModel = Omit<Model, "id">;

type ComparableModel = Pick<Model,
  | "name"
  | "attachment"
  | "reasoning"
  | "tool_call"
  | "structured_output"
  | "temperature"
  | "open_weights"
  | "modalities"
> & {
  limit: Pick<Model["limit"], "context" | "output">;
  cost?: Pick<Model["cost"], "input" | "output" | "cache_read" | "cache_write">;
};

function normalizeForComparison(model: Omit<Model, "id">): ComparableModel {
  return {
    name: model.name,
    attachment: model.attachment,
    reasoning: model.reasoning,
    tool_call: model.tool_call,
    structured_output: model.structured_output,
    temperature: model.temperature,
    open_weights: model.open_weights,
    modalities: model.modalities,
    limit: {
      context: model.limit.context,
      output: model.limit.output,
    },
    cost: model.cost ? {
      input: model.cost.input,
      output: model.cost.output,
      cache_read: model.cost.cache_read,
      cache_write: model.cost.cache_write,
    } : undefined,
  };
}

const SyntheticModelsResponse = z.object({
  data: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      input_modalities: z.array(z.string()),
      output_modalities: z.array(z.string()),
      context_length: z.number(),
      max_output_length: z.number().optional(),
      pricing: z.object({
        prompt: z.string(),
        completion: z.string(),
        input_cache_reads: z.string().optional(),
        input_cache_writes: z.string().optional(),
      }),
      supported_features: z.array(z.string()).optional().default([]),
      supported_sampling_parameters: z.array(z.string()).optional().default([]),
    })
  ),
});

type SyntheticModelsResponse = z.infer<typeof SyntheticModelsResponse>;

function parsePricing(priceStr: string): number | undefined {
  const value = parseFloat(priceStr.replace("$", ""));
  if (value === 0 || isNaN(value)) return undefined;
  return Math.round(value * 1_000_000 * 100) / 100;
}

function modelFilePath(modelId: string): string {
  const parts = modelId.split("/");
  if (parts.length === 1) {
    return path.join(modelsDir, `${modelId}.toml`);
  }
  const [org, ...rest] = parts;
  return path.join(modelsDir, org, `${rest.join("/")}.toml`);
}

function generateToml(modelId: string, model: SyntheticModel): string {
  const lines: string[] = [];

  lines.push(`name = "${model.name}"`);
  if (model.family) {
    lines.push(`family = "${model.family}"`);
  }
  lines.push(`attachment = ${model.attachment}`);
  lines.push(`reasoning = ${model.reasoning}`);
  lines.push(`tool_call = ${model.tool_call}`);
  if (model.structured_output !== undefined) {
    lines.push(`structured_output = ${model.structured_output}`);
  }
  if (model.temperature !== undefined) {
    lines.push(`temperature = ${model.temperature}`);
  }
  if (model.release_date) {
    lines.push(`release_date = "${model.release_date}"`);
  }
  if (model.knowledge) {
    lines.push(`knowledge = "${model.knowledge}"`);
  }
  lines.push(`last_updated = "${model.last_updated}"`);
  lines.push(`open_weights = ${model.open_weights}`);

  if (model.interleaved) {
    lines.push("");
    if (model.interleaved === true) {
      lines.push("[interleaved]");
    } else {
      lines.push("[interleaved]");
      lines.push(`field = "${model.interleaved.field}"`);
    }
  }

  if (model.cost) {
    lines.push("");
    lines.push("[cost]");
    lines.push(`input = ${model.cost.input}`);
    lines.push(`output = ${model.cost.output}`);
    if (model.cost.cache_read !== undefined) {
      lines.push(`cache_read = ${model.cost.cache_read}`);
    }
    if (model.cost.cache_write !== undefined) {
      lines.push(`cache_write = ${model.cost.cache_write}`);
    }
  }

  lines.push("");
  lines.push("[limit]");
  lines.push(`context = ${model.limit.context}`);
  lines.push(`output = ${model.limit.output}`);

  lines.push("");
  lines.push("[modalities]");
  lines.push(`input = ${JSON.stringify(model.modalities.input)}`);
  lines.push(`output = ${JSON.stringify(model.modalities.output)}`);

  return lines.join("\n") + "\n";
}

const response = await fetch("https://api.synthetic.new/openai/v1/models");
if (!response.ok) {
  console.error(
    `Failed to fetch models: ${response.status} ${response.statusText}`
  );
  process.exit(1);
}

const json = await response.json();
const parsed = SyntheticModelsResponse.safeParse(json);
if (!parsed.success) {
  console.error("Invalid response:", parsed.error.errors);
  process.exit(1);
}
const data: SyntheticModelsResponse = parsed.data;

console.log(`Fetched ${data.data.length} models. Syncing files...`);

const existingFiles = await Array.fromAsync(
  new Bun.Glob("**/*.toml").scan(modelsDir)
);
const existingModelIds = new Set(
  existingFiles.map((f) => {
    const parts = f.replace(/\.toml$/, "").split("/");
    if (parts.length === 1) return parts[0];
    return `${parts[0]}/${parts.slice(1).join("/")}`;
  })
);
const apiModelIds = new Set(data.data.map((m) => m.id));

let deleted = 0;
for (const existingId of existingModelIds) {
  if (!apiModelIds.has(existingId)) {
    const filePath = modelFilePath(existingId);
    await Bun.file(filePath).delete();
    console.log(`Deleted: ${existingId}`);
    deleted++;
  }
}

const emptyDirs = await Array.fromAsync(
  new Bun.Glob("**/").scan(modelsDir)
);
for (const dir of emptyDirs) {
  const dirPath = path.join(modelsDir, dir);
  const files = await Array.fromAsync(new Bun.Glob("*.toml").scan(dirPath));
  if (files.length === 0) {
    await Bun.file(dirPath).delete();
    console.log(`Deleted empty directory: ${dir}`);
  }
}

let created = 0;
let skipped = 0;
for (const apiModel of data.data) {
  const filePath = modelFilePath(apiModel.id);

  let existingData: Omit<Model, "id"> | null = null;
  try {
    const existingToml = await Bun.file(filePath).text();
    existingData = Bun.TOML.parse(existingToml) as Omit<Model, "id">;
  } catch {
    // File doesn't exist
  }

  const costInput = parsePricing(apiModel.pricing.prompt);
  const costOutput = parsePricing(apiModel.pricing.completion);
  const cacheRead = apiModel.pricing.input_cache_reads
    ? parsePricing(apiModel.pricing.input_cache_reads)
    : undefined;
  const cacheWrite = apiModel.pricing.input_cache_writes
    ? parsePricing(apiModel.pricing.input_cache_writes)
    : undefined;

  const hasReasoning = apiModel.supported_features.includes("reasoning");
  const hasInterleaved = hasReasoning;

  const syntheticModel: SyntheticModel = {
    name: apiModel.name,
    family: existingData?.family as ModelFamily | undefined,
    attachment: apiModel.input_modalities.includes("image"),
    reasoning: hasReasoning,
    interleaved: hasInterleaved ? { field: "reasoning_content" } : undefined,
    tool_call: apiModel.supported_features.includes("tools"),
    structured_output: apiModel.supported_features.includes("structured_outputs"),
    temperature: apiModel.supported_sampling_parameters.includes("temperature"),
    release_date: existingData?.release_date,
    knowledge: existingData?.knowledge,
    last_updated: new Date().toISOString().slice(0, 10),
    open_weights: true,
    cost: (costInput !== undefined || costOutput !== undefined) ? {
      input: costInput,
      output: costOutput,
      cache_read: cacheRead,
      cache_write: cacheWrite,
    } : undefined,
    limit: {
      context: apiModel.context_length,
      output: apiModel.max_output_length,
    },
    modalities: {
      input: apiModel.input_modalities as Model["modalities"]["input"],
      output: apiModel.output_modalities as Model["modalities"]["output"],
    },
  };

  if (existingData) {
    const normalizedExisting = normalizeForComparison(existingData);
    const normalizedIncoming = normalizeForComparison(syntheticModel);

    if (Bun.deepEquals(normalizedExisting, normalizedIncoming)) {
      console.log(`Skipped (no changes): ${apiModel.id}`);
      skipped++;
      continue;
    }
  }

  const dirPath = path.dirname(filePath);
  await mkdir(dirPath, { recursive: true });
  await Bun.write(filePath, generateToml(apiModel.id, syntheticModel));
  console.log(`Created: ${apiModel.id}`);
  created++;
}

console.log(`\nDone. Created: ${created}, Skipped: ${skipped}, Deleted: ${deleted}`);
