"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import { listJobs, type Job, type JobStatus } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Loader2, ChevronRight, RefreshCw, FileText, Clock } from "lucide-react";

const STATUS_CONFIG: Record<JobStatus, { label: string; dot: string; text: string; bg: string }> = {
  queued:  { label: "Queued",  dot: "bg-muted-foreground",    text: "text-muted-foreground",     bg: "bg-muted/60" },
  running: { label: "Running", dot: "bg-cyan-700 animate-pulse", text: "text-cyan-700",           bg: "bg-cyan-700/10" },
  done:    { label: "Done",    dot: "bg-olive-300",            text: "text-olive-300",            bg: "bg-olive-300/10" },
  failed:  { label: "Failed",  dot: "bg-destructive",         text: "text-destructive",           bg: "bg-destructive/10" },
};

function StatusBadge({ status }: { status: JobStatus }) {
  const cfg = STATUS_CONFIG[status];
  return (
    <span className={cn("inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold", cfg.bg, cfg.text)}>
      <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", cfg.dot)} />
      {cfg.label}
    </span>
  );
}

function duration(job: Job) {
  if (!job.created_at || !job.updated_at) return "—";
  const ms = new Date(job.updated_at).getTime() - new Date(job.created_at).getTime();
  if (ms < 1000) return "<1s";
  const s = Math.round(ms / 1000);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = async (showSpinner = false) => {
    if (showSpinner) setRefreshing(true);
    await listJobs().then(setJobs).catch(console.error);
    setLoading(false);
    setRefreshing(false);
  };

  useEffect(() => { refresh(); }, []);

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Jobs</h1>
          <p className="text-muted-foreground mt-1">Track all document ingestion runs</p>
        </div>
        <button
          onClick={() => refresh(true)}
          disabled={refreshing}
          className="flex items-center gap-2 px-3.5 py-2 rounded-xl border border-border bg-card hover:bg-muted/40 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
        >
          <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="flex items-center gap-2.5 text-muted-foreground text-sm py-12 justify-center">
          <Loader2 size={16} className="animate-spin" />
          Loading jobs…
        </div>
      ) : jobs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
          <div className="w-14 h-14 rounded-2xl bg-muted flex items-center justify-center">
            <FileText size={22} className="text-muted-foreground" />
          </div>
          <div>
            <p className="font-semibold">No jobs yet</p>
            <p className="text-sm text-muted-foreground mt-1">Upload a document to kick off your first ingestion job.</p>
          </div>
          <Link href="/" className="text-sm text-cyan-700 hover:underline font-medium">
            Go to Upload →
          </Link>
        </div>
      ) : (
        <div className="rounded-2xl border border-border overflow-hidden bg-card">
          {/* Table header */}
          <div className="grid grid-cols-[2fr_1fr_80px_80px_140px_40px] gap-4 px-5 py-3 border-b border-border bg-muted/30">
            {["File", "Status", "Steps", "Duration", "Created", ""].map((h) => (
              <span key={h} className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{h}</span>
            ))}
          </div>

          {/* Rows */}
          {jobs.map((job) => {
            const doneSteps = job.steps.filter(s => s.status === "done").length;
            return (
              <Link
                key={job.job_id}
                href={"/jobs/" + job.job_id}
                className="grid grid-cols-[2fr_1fr_80px_80px_140px_40px] gap-4 items-center px-5 py-4 border-b border-border/60 last:border-0 hover:bg-muted/20 transition-colors group"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center shrink-0">
                    <FileText size={14} className="text-muted-foreground" />
                  </div>
                  <span className="font-medium truncate text-sm">{job.filename}</span>
                </div>
                <div><StatusBadge status={job.status} /></div>
                <div className="text-sm text-muted-foreground tabular-nums">
                  {doneSteps}<span className="text-muted-foreground/50">/5</span>
                </div>
                <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
                  <Clock size={12} />
                  {duration(job)}
                </div>
                <div className="text-sm text-muted-foreground">
                  {job.created_at ? formatDistanceToNow(new Date(job.created_at), { addSuffix: true }) : "—"}
                </div>
                <div className="flex justify-end">
                  <ChevronRight size={16} className="text-muted-foreground group-hover:text-foreground transition-colors" />
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
