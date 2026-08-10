"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { LogIn, LogOut, Settings, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export function UserMenu() {
  const [role, setRole] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // sessionStorage is not available during SSR, so the menu renders only after
  // mount. Without the guard React reports a hydration mismatch on every load.
  useEffect(() => {
    setRole(sessionStorage.getItem("mahagr_role"));
    setUsername(sessionStorage.getItem("mahagr_username"));
    setMounted(true);
  }, []);

  if (!mounted) return <div className="ml-auto h-8" />;

  if (!username) {
    return (
      <div className="ml-auto">
        <Button
          asChild
          size="sm"
          variant="ghost"
          className="text-iceblue hover:bg-navy-700 hover:text-white"
        >
          <Link href="/login">
            <LogIn size={15} className="mr-1.5" /> Sign in
          </Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="ml-auto">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            size="sm"
            variant="ghost"
            className="gap-2 text-iceblue hover:bg-navy-700 hover:text-white"
          >
            <span className="grid h-6 w-6 place-items-center rounded-full bg-teal-bright text-[11px] font-semibold text-navy">
              {username.slice(0, 2).toUpperCase()}
            </span>
            <span className="hidden sm:inline">{username}</span>
          </Button>
        </DropdownMenuTrigger>

        <DropdownMenuContent align="end" className="w-56">
          <DropdownMenuLabel className="font-normal">
            <p className="flex items-center gap-1.5 text-sm font-medium text-ink">
              <User size={13} /> {username}
            </p>
            <p className="mt-0.5 text-xs text-slate2">{role}</p>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />

          {role === "IT Admin" && (
            <DropdownMenuItem asChild>
              <Link href="/admin" className="cursor-pointer">
                <Settings size={14} className="mr-2" /> Admin
              </Link>
            </DropdownMenuItem>
          )}

          <DropdownMenuItem
            className="cursor-pointer text-destructive focus:text-destructive"
            onClick={() => {
              sessionStorage.clear();
              window.location.href = "/login";
            }}
          >
            <LogOut size={14} className="mr-2" /> Sign out
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
