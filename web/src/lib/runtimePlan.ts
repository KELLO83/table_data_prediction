import type { RuntimePlanItem } from "./types";

export const RUNTIME_PLAN: RuntimePlanItem[] = [
  {
    setting: "SDPA",
    values: "on / off",
    purpose: "Confirm no-weight MultiheadAttention path and backend choice.",
    status: "planned",
  },
  {
    setting: "AMP",
    values: "on / off",
    purpose: "Measure mixed precision speedup and accuracy stability.",
    status: "planned",
  },
  {
    setting: "AMP dtype",
    values: "float16 first, bfloat16 if needed",
    purpose: "Compare tensor-core speed against numerical stability.",
    status: "planned",
  },
  {
    setting: "torch.compile",
    values: "auto / forced on / off",
    purpose: "Separate compile overhead from steady-state runtime.",
    status: "planned",
  },
  {
    setting: "compile mode",
    values: "reduce-overhead",
    purpose: "Start with the mode most likely to help repeated small-model calls.",
    status: "planned",
  },
  {
    setting: "matmul precision",
    values: "high / highest",
    purpose: "Quantify TF32/internal matmul precision tradeoff on CUDA.",
    status: "planned",
  },
  {
    setting: "batch size",
    values: "default / larger CUDA-safe value",
    purpose: "Find throughput limits without mixing model-quality claims.",
    status: "planned",
  },
];
