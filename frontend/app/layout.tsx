import type { Metadata } from "next";
import { Inter, Noto_Sans_Devanagari, Spectral } from "next/font/google";
import Link from "next/link";
import { MessagesSquare, Library } from "lucide-react";
import "./globals.css";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const deva = Noto_Sans_Devanagari({ subsets: ["devanagari"], variable: "--font-deva", display: "swap" });
const serif = Spectral({ subsets: ["latin"], weight: ["500", "600"], variable: "--font-serif", display: "swap" });

export const metadata: Metadata = {
  title: "MahaGR Assist",
  description: "Grounded, multilingual answers over Maharashtra Government Resolutions.",
};

function TopNav() {
  return (
    <header className="sticky top-0 z-20 bg-navy text-white">
      <nav className="mx-auto flex max-w-6xl items-center gap-6 px-5 py-3">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-teal-bright font-serif text-lg font-semibold text-navy">
            म
          </span>
          <span className="text-[15px] font-semibold tracking-tight">MahaGR&nbsp;Assist</span>
        </Link>
        <div className="ml-2 flex items-center gap-1 text-sm">
          <NavLink href="/" icon={<MessagesSquare size={16} />} label="Ask" />
          <NavLink href="/browse" icon={<Library size={16} />} label="Browse GRs" />
        </div>
        <span className="ml-auto hidden text-xs text-iceblue sm:block">
          Grounded · Multilingual · Explainable
        </span>
      </nav>
    </header>
  );
}

function NavLink({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) {
  return (
    <Link
      href={href}
      className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-iceblue transition-colors duration-200 hover:bg-navy-700 hover:text-white"
    >
      {icon}
      {label}
    </Link>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${deva.variable} ${serif.variable}`}>
      <body className="min-h-dvh font-sans">
        <TopNav />
        {children}
      </body>
    </html>
  );
}
