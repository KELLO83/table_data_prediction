export type DatasetProfile = {
  slug: string;
  name: string;
  openmlId: number;
  openmlUrl: string;
  domain: string;
  target: string;
  rows: number;
  features: number;
  numericFeatures: number;
  categoricalFeatures: number;
  regime: string;
  portfolioRole: string;
  description: string;
  modelingStory: string;
  leakagePolicy: string;
  stage: "primary" | "expansion";
};

export type ModelProfile = {
  name: string;
  family: string;
  status: "ready" | "optional dependency" | "planned";
  bestFor: string;
  portfolioStory: string;
  runtimeNotes: string;
};

export type RuntimePlanItem = {
  setting: string;
  values: string;
  purpose: string;
  status: "planned" | "measured";
};

export type RuntimeSettings = {
  enableSdpa?: string;
  sdpaBackend?: string;
  enableAmp?: string;
  ampDtype?: string;
  enableCompile?: string;
  compileMode?: string;
  matmulPrecision?: string;
};

export type ExperimentResult = {
  experimentId: string;
  dataset: string;
  target: string;
  model: string;
  family: string;
  rmse?: number;
  mae?: number;
  wape?: number;
  trainTimeSec?: number;
  predictTimeSec?: number;
  runtime?: RuntimeSettings;
};

export type WikiDocument = {
  slug: string;
  title: string;
  excerpt: string;
  content: string;
};

export type DashboardSummary = {
  datasetCoverage: number;
  completedDatasets: string[];
  fastestTrainTimeSec?: number;
};

export type DashboardData = {
  datasets: DatasetProfile[];
  models: ModelProfile[];
  runtimePlan: RuntimePlanItem[];
  results: ExperimentResult[];
  wiki: WikiDocument[];
  summary: DashboardSummary;
  sourceLabel: string;
};
