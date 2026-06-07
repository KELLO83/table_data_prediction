import { NextResponse } from "next/server";
import { loadDashboardData } from "@/lib/dashboard";

export async function GET() {
  return NextResponse.json(await loadDashboardData());
}
