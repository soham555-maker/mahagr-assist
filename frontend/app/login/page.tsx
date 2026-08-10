"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowLeft, Loader2, LogIn } from "lucide-react";
import { api } from "@/lib/api";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const data = new URLSearchParams();
      data.append("username", username);
      data.append("password", password);

      const res = await api.login(data);
      // sessionStorage, not localStorage: it dies with the tab, which narrows
      // what an XSS bug can reach and what a shared machine leaves behind.
      sessionStorage.setItem("mahagr_token", res.access_token);
      sessionStorage.setItem("mahagr_role", res.role);
      sessionStorage.setItem("mahagr_username", res.username);
      window.location.href = "/ask";
    } catch (err) {
      setError((err as Error).message || "Failed to log in");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-navy px-5 py-12">
      <Card className="w-full max-w-md border-line shadow-lg">
        <CardHeader className="items-center text-center">
          <div className="mb-2 grid h-12 w-12 place-items-center rounded-xl bg-teal-bright font-serif text-2xl font-semibold text-navy">
            म
          </div>
          <CardTitle className="font-serif text-2xl text-navy">MahaGR Assist</CardTitle>
          <CardDescription>Sign in to the officer portal</CardDescription>
        </CardHeader>

        <CardContent>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <form onSubmit={handleLogin} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                type="text"
                autoComplete="username"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <Button
              type="submit"
              disabled={loading}
              className="w-full bg-teal text-white hover:bg-navy"
            >
              {loading ? (
                <>
                  <Loader2 size={15} className="mr-1.5 animate-spin" /> Signing in…
                </>
              ) : (
                <>
                  <LogIn size={15} className="mr-1.5" /> Sign in
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Link
        href="/"
        className="mt-6 inline-flex items-center gap-1.5 text-[13px] text-iceblue transition-colors hover:text-white"
      >
        <ArrowLeft size={14} /> Back to the overview
      </Link>
    </div>
  );
}
