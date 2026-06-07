import { DashboardClient } from "@/components/DashboardClient";
import { loadDashboardData } from "@/lib/dashboard";

export default async function Page() {
  const data = await loadDashboardData();
  return <DashboardClient data={data} />;
}
