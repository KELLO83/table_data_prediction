import { existsSync } from "fs";
import { readFile } from "fs/promises";
import path from "path";
import type { WikiDocument } from "./types";

const ROOT = path.resolve(process.cwd(), "..");
const WIKI_FILES: Record<string, string> = {
  findings: "docs/llm_wiki/concepts/current_experiment_findings.md",
  runtime: "docs/llm_wiki/concepts/runtime_acceleration_ablation_protocol.md",
  plan: "docs/llm_wiki/experiment_notes/runtime/2026-06-07-runtime-acceleration-baseline-plan.md",
};

export async function loadWikiDocuments(): Promise<WikiDocument[]> {
  const documents = await Promise.all(Object.keys(WIKI_FILES).map(loadWikiDocument));
  return documents.filter((document): document is WikiDocument => Boolean(document));
}

export async function loadWikiDocument(slug: string): Promise<WikiDocument | null> {
  const wikiPath = WIKI_FILES[slug];
  if (!wikiPath) {
    return null;
  }

  const absolutePath = path.join(ROOT, wikiPath);
  if (!existsSync(absolutePath)) {
    return null;
  }

  const markdown = await readFile(absolutePath, "utf8");
  return {
    slug,
    title: titleFromMarkdown(markdown),
    excerpt: excerptFromMarkdown(markdown),
    content: markdown,
  };
}

function titleFromMarkdown(markdown: string) {
  const firstHeading = markdown.split(/\r?\n/).find((line) => line.startsWith("# "));
  return firstHeading?.replace(/^#\s+/, "") || "Untitled note";
}

function excerptFromMarkdown(markdown: string) {
  return markdown
    .split(/\r?\n/)
    .filter((line) => line.trim() && !line.startsWith("#"))
    .slice(0, 8)
    .join("\n");
}
