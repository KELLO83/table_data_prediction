import { DATASETS } from "./datasets";
import { MODELS } from "./models";
import { loadExperimentResults } from "./results";
import { RUNTIME_PLAN } from "./runtimePlan";
import { loadWikiDocuments } from "./wiki";
import type { DashboardData } from "./types";

export async function loadDashboardData(): Promise<DashboardData> {
  const [{ results, sourceLabel }, wiki] = await Promise.all([loadExperimentResults(), loadWikiDocuments()]);
  const completedDatasets = completedDatasetSlugs(results.map((result) => result.dataset));

  return {
    datasets: DATASETS,
    models: MODELS,
    runtimePlan: RUNTIME_PLAN,
    results,
    wiki,
    sourceLabel,
    summary: {
      datasetCoverage: completedDatasets.length,
      completedDatasets,
      fastestTrainTimeSec: fastestTrainTime(results),
    },
  };
}

function completedDatasetSlugs(datasetNames: string[]) {
  const normalizedNames = new Set(datasetNames.map(normalize));
  return DATASETS.filter((dataset) => normalizedNames.has(normalize(dataset.slug)) || normalizedNames.has(normalize(dataset.name))).map(
    (dataset) => dataset.slug,
  );
}

function fastestTrainTime(results: Array<{ trainTimeSec?: number }>) {
  const times = results.map((result) => result.trainTimeSec).filter((time): time is number => typeof time === "number");
  return times.length > 0 ? Math.min(...times) : undefined;
}

function normalize(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
}
