import { existsSync } from "fs";
import { readFile } from "fs/promises";
import path from "path";
import { DATASETS } from "./datasets";
import type { ExperimentResult, RuntimeSettings } from "./types";

const RESULTS_FILE = path.resolve(process.cwd(), "..", "results", "tabular_regression_experiments.csv");

const MODEL_FAMILY: Record<string, string> = {
  lightgbm: "gbdt",
  catboost: "gbdt",
  realmlp: "neural",
  tabm: "neural",
  tabr: "neural",
  dcnv2: "neural",
  node: "neural",
  ft_transformer: "transformer",
  tab_transformer: "transformer",
  tabnet: "transformer",
  tabpfn: "foundation",
  tabiclv2: "foundation",
};

export async function loadExperimentResults(): Promise<{ results: ExperimentResult[]; sourceLabel: string }> {
  if (!existsSync(RESULTS_FILE)) {
    return { results: [], sourceLabel: "results pending" };
  }

  const csv = await readFile(RESULTS_FILE, "utf8");
  const rows = parseCsv(csv);
  const results = rows
    .map(toExperimentResult)
    .sort((left, right) => numericSort(left.rmse, right.rmse));

  return { results, sourceLabel: path.relative(path.resolve(process.cwd(), ".."), RESULTS_FILE) };
}

function toExperimentResult(row: Record<string, string>): ExperimentResult {
  const model = row.model || "unknown";
  return {
    experimentId: row.experiment_id || `${row.csv}-${model}-${row.seed}`,
    dataset: datasetName(row),
    target: row.target || "target",
    model,
    family: MODEL_FAMILY[model] ?? "unknown",
    rmse: numberValue(row.rmse),
    mae: numberValue(row.mae),
    wape: numberValue(row.wape),
    trainTimeSec: numberValue(row.train_time_sec),
    predictTimeSec: numberValue(row.predict_time_sec),
    runtime: runtimeSettings(row),
  };
}

function datasetName(row: Record<string, string>) {
  const sourceText = `${row.csv ?? ""} ${row.experiment_id ?? ""}`.toLowerCase();
  const matchedDataset = DATASETS.find((dataset) => {
    const slug = dataset.slug.toLowerCase();
    const name = dataset.name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
    return sourceText.includes(slug) || sourceText.includes(name) || sourceText.includes(String(dataset.openmlId));
  });
  if (matchedDataset) {
    return matchedDataset.name;
  }

  const fromExperiment = row.experiment_id?.split("_").slice(0, -3).join("_");
  if (fromExperiment) {
    return fromExperiment;
  }
  return row.csv?.split(/[\\/]/).pop()?.replace(/\.csv$/i, "") || "dataset";
}

function runtimeSettings(row: Record<string, string>): RuntimeSettings | undefined {
  const settings: RuntimeSettings = {
    enableSdpa: row.enable_sdpa,
    sdpaBackend: row.sdpa_backend,
    enableAmp: row.enable_amp,
    ampDtype: row.amp_dtype,
    enableCompile: row.enable_compile,
    compileMode: row.compile_mode,
    matmulPrecision: row.matmul_precision,
  };
  return Object.values(settings).some(Boolean) ? settings : undefined;
}

function parseCsv(csv: string): Record<string, string>[] {
  const lines = csv.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) {
    return [];
  }
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function splitCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === "\"" && quoted && next === "\"") {
      current += "\"";
      index += 1;
      continue;
    }
    if (char === "\"") {
      quoted = !quoted;
      continue;
    }
    if (char === "," && !quoted) {
      values.push(current);
      current = "";
      continue;
    }
    current += char;
  }

  values.push(current);
  return values;
}

function numberValue(value: string | undefined) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function numericSort(left?: number, right?: number) {
  if (left === undefined && right === undefined) {
    return 0;
  }
  if (left === undefined) {
    return 1;
  }
  if (right === undefined) {
    return -1;
  }
  return left - right;
}
