"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

/**
 * A portal nav link that marks the current section. Client-side only because it
 * reads the pathname; `aria-current` carries the state for screen readers, so
 * the highlight is not colour-only.
 */
export function NavLink({
  href,
  icon,
  label,
}: {
  href: string;
  icon: React.ReactNode;
  label: string;
}) {
  const pathname = usePathname();
  const active = pathname === href || pathname.startsWith(`${href}/`);

  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={cn(
        "flex items-center gap-1.5 rounded-md px-3 py-1.5 transition-colors duration-200",
        active ? "bg-navy-700 text-white" : "text-iceblue hover:bg-navy-700 hover:text-white",
      )}
    >
      {icon}
      {label}
    </Link>
  );
}
