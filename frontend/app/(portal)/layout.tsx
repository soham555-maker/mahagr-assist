import Link from "next/link";
import { MessagesSquare, Library } from "lucide-react";
import { UserMenu } from "@/components/UserMenu";
import { NavLink } from "@/components/nav-link";

/**
 * Chrome shared by the authenticated officer portal (/ask, /browse, /admin).
 * A route group, so it adds a nav without adding a URL segment — the pages
 * stay at /ask rather than moving to /portal/ask.
 */
export default function PortalLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <header className="sticky top-0 z-30 border-b border-navy-700/50 bg-navy text-white">
        <nav className="mx-auto flex max-w-6xl items-center gap-6 px-5 py-3">
          <Link
            href="/"
            className="flex items-center gap-2.5 rounded-md transition-opacity hover:opacity-90"
          >
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-teal-bright font-serif text-lg font-semibold text-navy">
              म
            </span>
            <span className="text-[15px] font-semibold tracking-tight">MahaGR&nbsp;Assist</span>
          </Link>
          <div className="ml-2 flex items-center gap-1 text-sm">
            <NavLink href="/ask" icon={<MessagesSquare size={16} />} label="Ask" />
            <NavLink href="/browse" icon={<Library size={16} />} label="Browse GRs" />
          </div>
          <UserMenu />
        </nav>
      </header>
      {children}
    </>
  );
}
