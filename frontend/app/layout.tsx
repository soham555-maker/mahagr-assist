import type { Metadata } from "next";
import { Inter, Noto_Sans_Devanagari, Spectral } from "next/font/google";
import "./globals.css";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const deva = Noto_Sans_Devanagari({ subsets: ["devanagari"], variable: "--font-deva", display: "swap" });
const serif = Spectral({ subsets: ["latin"], weight: ["500", "600"], variable: "--font-serif", display: "swap" });

export const metadata: Metadata = {
  title: "MahaGR Assist — grounded answers over Maharashtra Government Resolutions",
  description:
    "A multilingual, source-grounded question-answering assistant over 18,080 Maharashtra Government Resolutions. Every answer carries a citation, it abstains when the corpus does not cover the question, and it runs entirely on-premise.",
};

/**
 * ROOT layout: fonts, globals and nothing else.
 *
 * The top navigation deliberately does NOT live here. `/` is a public landing
 * page and `/login` is a full-screen form; both look wrong under the officer
 * portal's chrome. The nav belongs to the `(portal)` route group instead, so
 * /ask, /browse and /admin share it while the two public pages do not.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${deva.variable} ${serif.variable}`}>
      <body className="min-h-dvh font-sans antialiased">{children}</body>
    </html>
  );
}
