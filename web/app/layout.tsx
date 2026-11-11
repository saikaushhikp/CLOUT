import type { Metadata, Viewport } from "next";
import { Google_Sans, Google_Sans_Code } from "next/font/google";
import "./globals.css";

// Downloaded at build time and served from this app (no runtime requests to Google). Subsets load lazily by
// unicode-range: latin-ext carries "Đ", greek "Σ", symbols the arrows. The build warns that it has no metric
// overrides for these families: text may reflow slightly when the font finishes loading.
const sans = Google_Sans({ subsets: ["latin", "latin-ext", "greek", "symbols"], variable: "--font-google-sans", display: "swap" });
const mono = Google_Sans_Code({ subsets: ["latin", "latin-ext"], variable: "--font-google-sans-code", display: "swap" });

export const metadata: Metadata = {
  title: "Clout",
  description: "Survey a hidden electorate, commit a budget manifesto, run a five-round campaign and win the vote.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f4f2ec" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0d14" },
  ],
};

// Applies the saved light/dark choice before first paint (no flash). Falls back to the OS setting.
const themeScript = `try{var t=localStorage.getItem("dumbocracy.theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t;}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${sans.variable} ${mono.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
