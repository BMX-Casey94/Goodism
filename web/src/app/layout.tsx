import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Goodism Knowledge Base",
  description: "Evidence-first conversational assistant grounded in the Goodism corpus.",
  robots: { index: false, follow: false },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en-GB" suppressHydrationWarning>
      <body className="antialiased selection:bg-amber-200/70 selection:text-stone-950 dark:selection:bg-amber-800/60 dark:selection:text-zinc-50">
        {children}
      </body>
    </html>
  );
}
