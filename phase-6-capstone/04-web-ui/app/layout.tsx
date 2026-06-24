import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Phase6 Agent Console",
  description: "Enterprise knowledge Agent capstone UI.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
