"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowRight, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

const SECTIONS = [
  { href: "#problem", label: "Problem" },
  { href: "#architecture", label: "Architecture" },
  { href: "#pipelines", label: "Pipelines" },
  { href: "#retrieval", label: "Retrieval" },
  { href: "#graph", label: "Graph" },
  { href: "#security", label: "Security" },
  { href: "#results", label: "Results" },
  { href: "#stack", label: "Stack" },
];

/**
 * Landing-page header. Transparent over the hero, solid once scrolled — the
 * hero is dark navy, so a solid bar there would just draw a seam across it.
 */
export function LandingHeader() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-40 transition-colors duration-300",
        scrolled ? "border-b border-navy-700/60 bg-navy/95 backdrop-blur" : "bg-transparent",
      )}
    >
      <nav className="mx-auto flex max-w-6xl items-center gap-6 px-5 py-3.5">
        <Link href="/" className="flex items-center gap-2.5 text-white">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-teal-bright font-serif text-lg font-semibold text-navy">
            म
          </span>
          <span className="text-[15px] font-semibold tracking-tight">MahaGR&nbsp;Assist</span>
        </Link>

        <div className="ml-auto hidden items-center gap-1 lg:flex">
          {SECTIONS.map((s) => (
            <a
              key={s.href}
              href={s.href}
              className="rounded-md px-2.5 py-1.5 text-[13px] text-iceblue transition-colors hover:bg-navy-700 hover:text-white"
            >
              {s.label}
            </a>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2 lg:ml-2">
          <Button asChild size="sm" className="bg-teal-bright text-navy hover:bg-teal-bright/90">
            <Link href="/ask">
              Open the portal <ArrowRight size={15} className="ml-1" />
            </Link>
          </Button>

          <Sheet open={open} onOpenChange={setOpen}>
            <SheetTrigger asChild className="lg:hidden">
              <Button size="icon" variant="ghost" className="text-white hover:bg-navy-700 hover:text-white" aria-label="Open navigation">
                <Menu size={18} />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-64">
              <SheetTitle className="text-left text-base">Sections</SheetTitle>
              <div className="mt-4 flex flex-col">
                {SECTIONS.map((s) => (
                  <a
                    key={s.href}
                    href={s.href}
                    onClick={() => setOpen(false)}
                    className="rounded-md px-3 py-2.5 text-sm text-ink transition-colors hover:bg-ice"
                  >
                    {s.label}
                  </a>
                ))}
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </nav>
    </header>
  );
}

export function LandingFooter() {
  return (
    <footer className="border-t border-navy-700/50 bg-navy px-5 py-12 text-iceblue">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-10 md:grid-cols-3">
          <div>
            <div className="flex items-center gap-2.5 text-white">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-teal-bright font-serif text-lg font-semibold text-navy">
                म
              </span>
              <span className="text-[15px] font-semibold tracking-tight">MahaGR Assist</span>
            </div>
            <p className="mt-3 max-w-xs text-[13px] leading-relaxed">
              Grounded, multilingual question answering over Maharashtra Government Resolutions.
              Built for VJTI AI Hackathon 2026, Problem Statement 3.
            </p>
          </div>

          <div className="text-[13px]">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-white">
              Dataset &amp; models
            </h3>
            <ul className="space-y-1.5">
              <li>orgpedia/mahGRs — public GRs from gr.maharashtra.gov.in</li>
              <li>BAAI bge-m3 · bge-reranker-v2-m3</li>
              <li>Qwen2.5-3B-Instruct via Ollama</li>
              <li>Tesseract OCR (mar + hin + eng)</li>
            </ul>
          </div>

          <div className="text-[13px]">
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-white">
              Original-work note
            </h3>
            <p className="leading-relaxed">
              The system, pipeline and portal are our own work. The two fee-table GRs used to
              demonstrate table handling are synthetic samples we created and are labelled as such;
              every other GR is an original government document.
            </p>
          </div>
        </div>

        <div className="mt-10 border-t border-navy-700/60 pt-6 text-[12px]">
          Runs entirely on-premise — no document, query or embedding leaves the machine.
        </div>
      </div>
    </footer>
  );
}
