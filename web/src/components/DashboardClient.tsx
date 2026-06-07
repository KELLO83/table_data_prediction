"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  BookOpenText,
  Boxes,
  Cpu,
  Database,
  ExternalLink,
  Filter,
  Gauge,
  Layers3,
  RefreshCw,
  SlidersHorizontal,
  Trophy,
  X,
} from "lucide-react";
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  DashboardData,
  DatasetProfile,
  ExperimentResult,
  ModelProfile,
  RuntimePlanItem,
  WikiDocument,
} from "@/lib/types";

const FAMILY_COLORS: Record<string, string> = {
  gbdt: "#2563eb",
  neural: "#16a34a",
  transformer: "#d97706",
  foundation: "#9333ea",
  unknown: "#64748b",
};

const METRIC_OPTIONS = [
  { value: "rmse", label: "RMSE" },
  { value: "mae", label: "MAE" },
  { value: "wape", label: "WAPE" },
  { value: "trainTimeSec", label: "Train Time" },
] as const;

type MetricKey = (typeof METRIC_OPTIONS)[number]["value"];
type MarkdownBlock =
  | { type: "heading"; level: number; text: string }
  | { type: "paragraph"; text: string }
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "table"; rows: string[][] };

export function DashboardClient({ data }: { data: DashboardData }) {
  const [datasetFilter, setDatasetFilter] = useState("all");
  const [familyFilter, setFamilyFilter] = useState("all");
  const [metricSort, setMetricSort] = useState<MetricKey>("rmse");
  const [selectedDatasetSlug, setSelectedDatasetSlug] = useState(data.datasets[1]?.slug ?? data.datasets[0]?.slug ?? "");
  const [selectedWikiSlug, setSelectedWikiSlug] = useState<string | null>(null);

  const filteredResults = useMemo(
    () => filterAndSortResults(data.results, datasetFilter, familyFilter, metricSort),
    [data.results, datasetFilter, familyFilter, metricSort],
  );
  const selectedDataset = data.datasets.find((dataset) => dataset.slug === selectedDatasetSlug) ?? data.datasets[0];
  const selectedWiki = data.wiki.find((document) => document.slug === selectedWikiSlug);
  const bestRun = filteredResults[0] ?? data.results[0];
  const families = Array.from(new Set(data.models.map((model) => model.family)));

  return (
    <main className="shell">
      <aside className="sidebar" aria-label="Primary">
        <div className="brandBlock">
          <div className="brandMark">IT</div>
          <div>
            <p className="eyebrow">Industrial</p>
            <h1>Tabular Regression Lab</h1>
          </div>
        </div>
        <nav className="navList">
          <a href="#overview"><Gauge size={18} />Overview</a>
          <a href="#datasets"><Database size={18} />Datasets</a>
          <a href="#models"><Layers3 size={18} />Models</a>
          <a href="#leaderboard"><Trophy size={18} />Leaderboard</a>
          <a href="#runtime"><Cpu size={18} />Runtime</a>
          <a href="#wiki"><BookOpenText size={18} />Wiki</a>
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Experiment Console</p>
            <h2>Industrial Tabular Regression Lab</h2>
          </div>
          <a className="iconButton" href="/api/dashboard" aria-label="Open dashboard API">
            <RefreshCw size={18} />
            API
          </a>
        </header>

        <section id="overview" className="kpiGrid">
          <MetricCard icon={<Database size={20} />} label="Datasets" value={data.datasets.length.toString()} tone="blue" />
          <MetricCard icon={<Activity size={20} />} label="Completed Runs" value={data.results.length.toString()} tone="green" />
          <MetricCard icon={<Trophy size={20} />} label={`Best ${metricLabel(metricSort)}`} value={formatResultMetric(bestRun, metricSort)} tone="amber" />
          <MetricCard icon={<BarChart3 size={20} />} label="Fastest Train" value={formatSeconds(data.summary.fastestTrainTimeSec)} tone="violet" />
        </section>

        <section className="filterBar" aria-label="Result filters">
          <div className="filterTitle">
            <Filter size={18} />
            <strong>Result Filters</strong>
          </div>
          <label>
            Dataset
            <select value={datasetFilter} onChange={(event) => setDatasetFilter(event.target.value)}>
              <option value="all">All datasets</option>
              {data.datasets.map((dataset) => (
                <option key={dataset.slug} value={dataset.name}>{dataset.name}</option>
              ))}
            </select>
          </label>
          <label>
            Family
            <select value={familyFilter} onChange={(event) => setFamilyFilter(event.target.value)}>
              <option value="all">All families</option>
              {families.map((family) => (
                <option key={family} value={family}>{family}</option>
              ))}
            </select>
          </label>
          <label>
            Sort
            <select value={metricSort} onChange={(event) => setMetricSort(event.target.value as MetricKey)}>
              {METRIC_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
        </section>

        <section className="twoColumn">
          <Panel title="Model Frontier" action={`${filteredResults.length} filtered runs`}>
            {filteredResults.length > 0 ? (
              <>
                <FrontierChart results={filteredResults} />
                <FamilyLegend results={filteredResults} />
              </>
            ) : (
              <EmptyResultState />
            )}
          </Panel>
          <Panel title="Benchmark Coverage" action={`${data.summary.datasetCoverage}/${data.datasets.length}`}>
            <div className="coverageList">
              {data.datasets.map((dataset) => (
                <button
                  className={`coverageRow buttonRow ${selectedDatasetSlug === dataset.slug ? "active" : ""}`}
                  key={dataset.slug}
                  onClick={() => setSelectedDatasetSlug(dataset.slug)}
                >
                  <div>
                    <strong>{dataset.name}</strong>
                    <span>{dataset.regime}</span>
                  </div>
                  <span className={data.summary.completedDatasets.includes(dataset.slug) ? "status done" : "status pending"}>
                    {data.summary.completedDatasets.includes(dataset.slug) ? "complete" : "planned"}
                  </span>
                </button>
              ))}
            </div>
          </Panel>
        </section>

        <section id="datasets" className="twoColumn datasetSection">
          <Panel title="Dataset Explorer" action={`${data.datasets.length} planned tables`}>
            <DatasetTable datasets={data.datasets} selectedSlug={selectedDatasetSlug} onSelect={setSelectedDatasetSlug} />
          </Panel>
          {selectedDataset ? <DatasetDetail dataset={selectedDataset} /> : null}
        </section>

        <section id="models" className="panel">
          <div className="panelHeader">
            <h3>Model Registry</h3>
            <span>{data.models.length} candidates</span>
          </div>
          <ModelRegistry models={data.models} />
        </section>

        <section id="leaderboard" className="panel">
          <div className="panelHeader">
            <h3>Leaderboard</h3>
            <span>{data.sourceLabel}</span>
          </div>
          {filteredResults.length > 0 ? <Leaderboard results={filteredResults} metricSort={metricSort} /> : <EmptyResultState />}
        </section>

        <section id="runtime" className="panel">
          <div className="panelHeader">
            <h3>Runtime Lab</h3>
            <span>SDPA / AMP / compile</span>
          </div>
          <RuntimeMatrix plan={data.runtimePlan} results={filteredResults} />
        </section>

        <section id="wiki" className="panel">
          <div className="panelHeader">
            <h3>LLM Wiki Notes</h3>
            <span>{data.wiki.length} maintained notes</span>
          </div>
          <div className="wikiGrid">
            {data.wiki.map((document) => (
              <button className="wikiItem" key={document.slug} onClick={() => setSelectedWikiSlug(document.slug)}>
                <p className="eyebrow">{document.slug}</p>
                <h4>{document.title}</h4>
                <pre>{document.excerpt}</pre>
              </button>
            ))}
          </div>
        </section>
      </section>

      {selectedWiki ? <WikiModal document={selectedWiki} onClose={() => setSelectedWikiSlug(null)} /> : null}
    </main>
  );
}

function MetricCard({ icon, label, value, tone }: { icon: React.ReactNode; label: string; value: string; tone: string }) {
  return (
    <article className={`metricCard ${tone}`}>
      <div className="metricIcon">{icon}</div>
      <p>{label}</p>
      <strong>{value}</strong>
    </article>
  );
}

function Panel({ title, action, children }: { title: string; action: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panelHeader">
        <h3>{title}</h3>
        <span>{action}</span>
      </div>
      {children}
    </section>
  );
}

function DatasetTable({
  datasets,
  selectedSlug,
  onSelect,
}: {
  datasets: DatasetProfile[];
  selectedSlug: string;
  onSelect: (slug: string) => void;
}) {
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Domain</th>
            <th>Rows</th>
            <th>Features</th>
            <th>Target</th>
            <th>Regime</th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((dataset) => (
            <tr className={selectedSlug === dataset.slug ? "selectedRow" : ""} key={dataset.slug} onClick={() => onSelect(dataset.slug)}>
              <td><strong>{dataset.name}</strong></td>
              <td>{dataset.domain}</td>
              <td>{formatInteger(dataset.rows)}</td>
              <td>{formatInteger(dataset.features)}</td>
              <td><code>{dataset.target}</code></td>
              <td>{dataset.regime}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DatasetDetail({ dataset }: { dataset: DatasetProfile }) {
  return (
    <section className="panel detailPanel">
      <div className="panelHeader">
        <h3>{dataset.name}</h3>
        <a href={dataset.openmlUrl} target="_blank" rel="noreferrer">
          OpenML <ExternalLink size={14} />
        </a>
      </div>
      <div className="detailBody">
        <p>{dataset.description}</p>
        <div className="detailStats">
          <Stat label="Rows" value={formatInteger(dataset.rows)} />
          <Stat label="Features" value={formatInteger(dataset.features)} />
          <Stat label="Numeric" value={formatInteger(dataset.numericFeatures)} />
          <Stat label="Categorical" value={formatInteger(dataset.categoricalFeatures)} />
        </div>
        <section>
          <h4>Portfolio Story</h4>
          <p>{dataset.modelingStory}</p>
        </section>
        <section>
          <h4>Leakage / ID Policy</h4>
          <p>{dataset.leakagePolicy}</p>
        </section>
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="statPill">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ModelRegistry({ models }: { models: ModelProfile[] }) {
  return (
    <div className="modelGrid">
      {models.map((model) => (
        <article className={`modelCard ${model.family}`} key={model.name}>
          <div className="modelCardHeader">
            <div>
              <strong>{model.name}</strong>
              <span>{model.family}</span>
            </div>
            <span className={`modelStatus ${statusClass(model.status)}`}>{model.status}</span>
          </div>
          <dl>
            <dt>Best for</dt>
            <dd>{model.bestFor}</dd>
            <dt>Portfolio</dt>
            <dd>{model.portfolioStory}</dd>
            <dt>Runtime</dt>
            <dd>{model.runtimeNotes}</dd>
          </dl>
        </article>
      ))}
    </div>
  );
}

function FamilyLegend({ results }: { results: ExperimentResult[] }) {
  const families = Array.from(new Set(results.map((result) => result.family)));

  return (
    <div className="familyLegend" aria-label="Model family legend">
      {families.map((family) => (
        <span key={family}>
          <i style={{ backgroundColor: FAMILY_COLORS[family] ?? FAMILY_COLORS.unknown }} />
          {family}
        </span>
      ))}
    </div>
  );
}

function FrontierChart({ results }: { results: ExperimentResult[] }) {
  const points = results
    .filter((result) => Number.isFinite(result.rmse) && Number.isFinite(result.trainTimeSec))
    .map((result) => ({
      ...result,
      x: result.trainTimeSec,
      y: result.rmse,
      familyColor: FAMILY_COLORS[result.family] ?? FAMILY_COLORS.unknown,
    }));

  return (
    <div className="chartBox">
      <ResponsiveContainer width="100%" height={310}>
        <ScatterChart margin={{ top: 16, right: 16, bottom: 20, left: 8 }}>
          <CartesianGrid stroke="#e2e8f0" />
          <XAxis dataKey="x" name="Train time" unit="s" tick={{ fill: "#64748b", fontSize: 12 }} />
          <YAxis dataKey="y" name="RMSE" tick={{ fill: "#64748b", fontSize: 12 }} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<ChartTooltip />} />
          <Scatter data={points}>
            {points.map((point) => (
              <Cell key={`${point.experimentId}-${point.model}`} fill={point.familyColor} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function ChartTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ExperimentResult }> }) {
  if (!active || !payload?.length) {
    return null;
  }
  const point = payload[0].payload;
  return (
    <div className="chartTooltip">
      <strong>{point.model}</strong>
      <span>{point.dataset}</span>
      <span>RMSE {formatMetric(point.rmse)}</span>
      <span>Train {formatSeconds(point.trainTimeSec)}</span>
    </div>
  );
}

function Leaderboard({ results, metricSort }: { results: ExperimentResult[]; metricSort: MetricKey }) {
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Dataset</th>
            <th>Model</th>
            <th>Family</th>
            <th className={metricSort === "rmse" ? "sortedCol" : ""}>RMSE</th>
            <th className={metricSort === "mae" ? "sortedCol" : ""}>MAE</th>
            <th className={metricSort === "wape" ? "sortedCol" : ""}>WAPE</th>
            <th className={metricSort === "trainTimeSec" ? "sortedCol" : ""}>Train</th>
            <th>Predict</th>
          </tr>
        </thead>
        <tbody>
          {results.slice(0, 30).map((result, index) => (
            <tr key={result.experimentId}>
              <td>{index + 1}</td>
              <td>{result.dataset}</td>
              <td><strong>{result.model}</strong></td>
              <td>{result.family}</td>
              <td>{formatMetric(result.rmse)}</td>
              <td>{formatMetric(result.mae)}</td>
              <td>{formatMetric(result.wape)}</td>
              <td>{formatSeconds(result.trainTimeSec)}</td>
              <td>{formatSeconds(result.predictTimeSec)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RuntimeMatrix({ plan, results }: { plan: RuntimePlanItem[]; results: ExperimentResult[] }) {
  const measuredRows = results.filter((result) => result.runtime);

  return (
    <div className="runtimeSection">
      <div className="runtimePlanGrid">
        {plan.map((item) => (
          <article className="runtimePlanItem" key={item.setting}>
            <div>
              <SlidersHorizontal size={17} />
              <strong>{item.setting}</strong>
            </div>
            <p>{item.values}</p>
            <span>{item.purpose}</span>
          </article>
        ))}
      </div>
      {measuredRows.length === 0 ? (
        <div className="runtimeEmpty compact">
          <Boxes size={22} />
          <strong>Runtime evidence pending</strong>
          <span>Training rows with SDPA, AMP, and compile fields will appear here.</span>
        </div>
      ) : (
        <div className="runtimeGrid">
          {measuredRows.slice(0, 8).map((result) => (
            <article className="runtimeItem" key={`${result.experimentId}-runtime`}>
              <div>
                <strong>{result.model}</strong>
                <span>{result.dataset}</span>
              </div>
              <dl>
                <dt>SDPA</dt><dd>{result.runtime?.enableSdpa ?? "n/a"}</dd>
                <dt>AMP</dt><dd>{result.runtime?.enableAmp ?? "n/a"}</dd>
                <dt>Compile</dt><dd>{result.runtime?.enableCompile ?? "n/a"}</dd>
                <dt>Backend</dt><dd>{result.runtime?.sdpaBackend ?? "pending"}</dd>
              </dl>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function WikiModal({ document, onClose }: { document: WikiDocument; onClose: () => void }) {
  return (
    <div className="modalBackdrop" role="presentation" onClick={onClose}>
      <section className="modalPanel" role="dialog" aria-modal="true" aria-label={document.title} onClick={(event) => event.stopPropagation()}>
        <header>
          <div>
            <p className="eyebrow">{document.slug}</p>
            <h3>{document.title}</h3>
          </div>
          <button className="closeButton" onClick={onClose} aria-label="Close note">
            <X size={18} />
          </button>
        </header>
        <MarkdownContent content={document.content} title={document.title} />
      </section>
    </div>
  );
}

function MarkdownContent({ content, title }: { content: string; title: string }) {
  return (
    <div className="markdownBody">
      {parseMarkdown(stripLeadingTitle(content, title)).map((block, index) => renderMarkdownBlock(block, index))}
    </div>
  );
}

function renderMarkdownBlock(block: MarkdownBlock, index: number) {
  if (block.type === "heading") {
    const Heading = block.level <= 2 ? "h4" : "h5";
    return <Heading key={index}>{renderInlineMarkdown(block.text)}</Heading>;
  }

  if (block.type === "list") {
    const List = block.ordered ? "ol" : "ul";
    return (
      <List key={index}>
        {block.items.map((item) => (
          <li key={item}>{renderInlineMarkdown(item)}</li>
        ))}
      </List>
    );
  }

  if (block.type === "table") {
    const [header, ...body] = block.rows;
    return (
      <div className="markdownTableWrap" key={index}>
        <table>
          <thead>
            <tr>
              {header.map((cell) => (
                <th key={cell}>{renderInlineMarkdown(cell)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((row) => (
              <tr key={row.join("|")}>
                {row.map((cell) => (
                  <td key={cell}>{renderInlineMarkdown(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return <p key={index}>{renderInlineMarkdown(block.text)}</p>;
}

function renderInlineMarkdown(text: string) {
  return text.split(/(`[^`]+`)/g).map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={`${part}-${index}`}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function parseMarkdown(markdown: string): MarkdownBlock[] {
  const lines = markdown.split(/\r?\n/);
  const blocks: MarkdownBlock[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) {
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2] });
      index += 1;
      continue;
    }

    if (line.startsWith("|")) {
      const tableLines = collectWhile(lines, index, (value) => value.trim().startsWith("|"));
      const rows = tableLines.map(parseTableRow).filter((row) => !isTableSeparator(row));
      if (rows.length > 0) {
        blocks.push({ type: "table", rows });
      }
      index += tableLines.length;
      continue;
    }

    if (/^-\s+/.test(line) || /^\d+\.\s+/.test(line)) {
      const ordered = /^\d+\.\s+/.test(line);
      const listBlock = parseListBlock(lines, index, ordered);
      blocks.push({ type: "list", ordered, items: listBlock.items });
      index = listBlock.nextIndex;
      continue;
    }

    const paragraphLines = collectWhile(lines, index, (value) => {
      const trimmed = value.trim();
      return Boolean(trimmed) && !isMarkdownBlockStart(trimmed);
    });
    blocks.push({ type: "paragraph", text: paragraphLines.map((value) => value.trim()).join(" ") });
    index += paragraphLines.length;
  }

  return blocks;
}

function collectWhile(lines: string[], startIndex: number, predicate: (line: string) => boolean) {
  const collected: string[] = [];
  for (let index = startIndex; index < lines.length; index += 1) {
    if (!predicate(lines[index])) {
      break;
    }
    collected.push(lines[index]);
  }
  return collected;
}

function parseListBlock(lines: string[], startIndex: number, ordered: boolean) {
  const items: string[] = [];
  const marker = ordered ? /^\d+\.\s+/ : /^-\s+/;
  let index = startIndex;

  while (index < lines.length) {
    const line = lines[index].trim();
    if (!marker.test(line)) {
      break;
    }

    const itemLines = [line.replace(marker, "")];
    index += 1;

    while (index < lines.length) {
      const nextLine = lines[index].trim();
      if (!nextLine) {
        index += 1;
        break;
      }
      if (marker.test(nextLine) || isHardMarkdownBoundary(nextLine)) {
        break;
      }
      itemLines.push(nextLine);
      index += 1;
    }

    items.push(itemLines.join(" "));
  }

  return { items, nextIndex: index };
}

function parseTableRow(line: string) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isTableSeparator(row: string[]) {
  return row.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function isMarkdownBlockStart(line: string) {
  return /^(#{1,4})\s+/.test(line) || line.startsWith("|") || /^-\s+/.test(line) || /^\d+\.\s+/.test(line);
}

function isHardMarkdownBoundary(line: string) {
  return /^(#{1,4})\s+/.test(line) || line.startsWith("|");
}

function stripLeadingTitle(content: string, title: string) {
  const lines = content.split(/\r?\n/);
  const firstLine = lines[0]?.trim();
  if (firstLine === `# ${title}`) {
    return lines.slice(1).join("\n").trim();
  }
  return content;
}

function EmptyResultState() {
  return (
    <div className="emptyState">
      <BarChart3 size={28} />
      <strong>Awaiting trained model results</strong>
      <span>Run the Python training pipeline to populate the dashboard from results CSV.</span>
    </div>
  );
}

function filterAndSortResults(results: ExperimentResult[], datasetFilter: string, familyFilter: string, metricSort: MetricKey) {
  return results
    .filter((result) => datasetFilter === "all" || result.dataset === datasetFilter)
    .filter((result) => familyFilter === "all" || result.family === familyFilter)
    .sort((left, right) => numericSort(metricValue(left, metricSort), metricValue(right, metricSort)));
}

function metricValue(result: ExperimentResult, metric: MetricKey) {
  return result[metric];
}

function metricLabel(metric: MetricKey) {
  return METRIC_OPTIONS.find((option) => option.value === metric)?.label ?? metric;
}

function formatResultMetric(result: ExperimentResult | undefined, metric: MetricKey) {
  if (!result) {
    return "pending";
  }
  return metric === "trainTimeSec" ? formatSeconds(result.trainTimeSec) : formatMetric(metricValue(result, metric));
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

function statusClass(status: ModelProfile["status"]) {
  return status.replace(/\s+/g, "-");
}

function formatMetric(value?: number) {
  if (value === undefined || !Number.isFinite(value)) {
    return "pending";
  }
  return value >= 100 ? value.toFixed(2) : value.toFixed(4);
}

function formatSeconds(value?: number) {
  if (value === undefined || !Number.isFinite(value)) {
    return "pending";
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)}s`;
}

function formatInteger(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}
