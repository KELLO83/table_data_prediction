import { NextResponse } from "next/server";
import { loadWikiDocument } from "@/lib/wiki";

type RouteContext = {
  params: Promise<{ slug: string }>;
};

export async function GET(_request: Request, context: RouteContext) {
  const { slug } = await context.params;
  const document = await loadWikiDocument(slug);

  if (!document) {
    return NextResponse.json({ error: "Wiki document not found" }, { status: 404 });
  }

  return NextResponse.json(document);
}
