import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Supply Chain Disruption System",
  description: "AI-based supply chain incident and delivery risk analysis.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
