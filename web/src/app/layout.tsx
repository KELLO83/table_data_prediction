import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "Industrial Tabular Regression Lab",
  description: "Tabular regression experiment dashboard",
  icons: {
    icon: "/icon.svg",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
