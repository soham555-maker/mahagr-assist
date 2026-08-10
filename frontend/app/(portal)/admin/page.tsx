"use client";

import { useEffect, useState } from "react";
import { Activity, ShieldCheck } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type AuditLog = {
  id: string;
  user_id: string;
  action: string;
  detail: string;
  ip: string;
  ts: number;
};

export default function AdminPage() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    // The role check here is a CONVENIENCE, not the control. /admin/audit-logs
    // is gated by require_role(["IT Admin"]) on the server, so editing this
    // value in devtools gets you a 403, not the log.
    const role = sessionStorage.getItem("mahagr_role");
    if (role && role !== "IT Admin") {
      setError("Access denied — the IT Admin role is required to view the audit trail.");
      setLoading(false);
      return;
    }

    const token = sessionStorage.getItem("mahagr_token");
    fetch(`${API_BASE}/admin/audit-logs`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
      .then((res) => {
        if (!res.ok) throw new Error("Could not load the audit log — access denied.");
        return res.json();
      })
      .then(setLogs)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="mx-auto max-w-6xl px-5 py-8">
      <div className="mb-2 flex items-center gap-2.5">
        <ShieldCheck className="text-teal" size={24} />
        <h1 className="font-serif text-2xl font-semibold text-navy">Admin</h1>
      </div>
      <p className="mb-8 max-w-2xl text-sm leading-relaxed text-slate2">
        Who asked what, when, and which GRs were cited. The trail deliberately records the question
        and the cited GR numbers — never the answer text or document bodies, so it cannot become a
        second uncontrolled copy of the corpus.
      </p>

      {error ? (
        <Alert variant="destructive">
          <AlertTitle>Access denied</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : (
        <Card className="border-line shadow-sm">
          <CardHeader className="border-b border-line bg-ice/60">
            <CardTitle className="flex items-center gap-2 text-base text-navy">
              <Activity size={17} className="text-teal" /> Recent activity
            </CardTitle>
            <CardDescription>
              {loading ? "Loading…" : `${logs.length} most recent audited action(s)`}
            </CardDescription>
          </CardHeader>

          <CardContent className="p-0">
            {loading ? (
              <div className="space-y-3 p-6">
                {[0, 1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-9 w-full" />
                ))}
              </div>
            ) : logs.length === 0 ? (
              <p className="px-6 py-12 text-center text-sm text-slate2">
                No audited actions yet. Ask a question in the portal and it will appear here.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="whitespace-nowrap">Timestamp</TableHead>
                      <TableHead>User</TableHead>
                      <TableHead>Action</TableHead>
                      <TableHead>IP</TableHead>
                      <TableHead>Detail</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {logs.map((log) => (
                      <TableRow key={log.id}>
                        <TableCell className="whitespace-nowrap font-mono text-xs text-slate2">
                          {new Date(log.ts * 1000).toLocaleString()}
                        </TableCell>
                        <TableCell className="whitespace-nowrap font-medium text-navy">
                          {log.user_id}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="secondary"
                            className="bg-teal/10 text-teal hover:bg-teal/10"
                          >
                            {log.action}
                          </Badge>
                        </TableCell>
                        <TableCell className="whitespace-nowrap font-mono text-xs text-slate2">
                          {log.ip}
                        </TableCell>
                        <TableCell>
                          <pre className="max-w-md overflow-x-auto whitespace-pre-wrap rounded-md bg-ice p-2 font-mono text-[11.5px] leading-relaxed text-ink">
                            {log.detail}
                          </pre>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </main>
  );
}
