"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Check, X, Loader2, ChevronDown, ChevronUp, AlertCircle, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { Progress } from "@/components/ui/progress";
import { getJob, type Job, type JobStatus, type Step } from "@/lib/api";
import { cn } from "@/lib/utils";

const STEP_ORDER = ["parsing", "chunking", "enriching", "embedding", "storing"];

const STATUS_CONFIG: Record<JobStatus, { label: string; dot: string; text: string; bg: string }> = {
  queued:  { label: "Queued",  dot: "bg-muted-foreground",       text: "text-muted-foreground", bg: "bg-muted/60" },
  running: { label: "Running", dot: "bg-cyan-700 animate-pulse", text: "text-cyan-700",          bg: "bg-cyan-700/10" },
  done:    { label: "Done",    dot: "bg-olive-300",              text: "text-olive-300",          bg: "bg-olive-300/10" },
  failed:  { label: "Failed",  dot: "bg-destructive",            text: "text-destructive",        bg: "bg-destructive/10" },
};

function StatusBadge({ status }: { status: JobStatus }) {
  const cfg = STATUS_CONFIG[status];
  return (
    <span className={cn("inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-semibold", cfg.bg, cfg.text)}>
      {status === "running"
        ? <Loader2 size={12} className="animate-spin" />
        : <span className={cn("w-2 h-2 rounded-full shrink-0", cfg.dot)} />
      }
      {cfg.label}
    </span>
  );
}

function StepIcon({ status, index }: { status: string; index: number }) {
  const base = "w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0";
  if (status === "done")    return <div className={cn(base, "bg-olive-300/20")}><Check size={14} className="text-olive-300" strokeWidth={2.5} /></div>;
  if (status === "failed")  return <div className={cn(base, "bg-destructive/15")}><X size={14} className="text-destructive" strokeWidth={2.5} /></div>;
  if (status === "running") return <div className={cn(base, "bg-cyan-700/20")}><Loader2 size={14} className="text-cyan-700 animate-spin" /></div>;
  return <div className={cn(base, "border-2 border-border bg-muted/30 text-muted-foreground/50")}>{index + 1}</div>;
}

function ElementTypeBar({ types }: { types: Record<string, number> }) {
  const total = Object.values(types).reduce((a, b) => a + b, 0);
  if (total === 0) return null;
  const colors: Record<string, string> = {
    text:  "bg-cyan-700",
    image: "bg-olive-300",
    table: "bg-mist-400",
  };
  return (
    <div className="space-y-2.5 mt-3">
      {Object.entries(types).map(([type, count]) => (
        <div key={type} className="flex items-center gap-3 text-xs">
          <span className="w-10 text-muted-foreground capitalize font-medium">{type}</span>
          <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all", colors[type] ?? "bg-muted-foreground")}
              style={{ width: `${(count / total) * 100}%` }}
            />
          </div>
          <span className="w-6 text-right font-semibold tabular-nums">{count}</span>
        </div>
      ))}
    </div>
  );
}

function MetadataGrid({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data).filter(([, v]) => typeof v !== "object");
  if (entries.length === 0) return null;
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-2 mt-3">
      {entries.map(([k, v]) => (
        <div key={k} className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground capitalize">{k.replace(/_/g, " ")}</span>
          <span className="font-semibold tabular-nums">{String(v)}</span>
        </div>
      ))}
    </div>
  );
}

function StepRow({ step, index, isLast }: { step: Step; index: number; isLast: boolean }) {
  const [open, setOpen] = useState(false);
  const hasMeta = step.status === "done" && Object.keys(step.metadata).length > 0;
  const elementTypes = step.metadata.element_types as Record<string, number> | undefined;
  const otherMeta = elementTypes
    ? Object.fromEntries(Object.entries(step.metadata).filter(([k]) => k !== "element_types"))
    : step.metadata;

  return (
    <div className="flex gap-4">
      {/* Timeline */}
      <div className="flex flex-col items-center">
        <StepIcon status={step.status} index={index} />
        {!isLast && <div className="w-px flex-1 bg-border mt-1.5 min-h-[1.5rem]" />}
      </div>

      {/* Content */}
      <div className={cn("flex-1 min-w-0", isLast ? "pb-0" : "pb-6")}>
        <div
          className={cn("flex items-center justify-between", hasMeta && "cursor-pointer group")}
          onClick={() => hasMeta && setOpen((o) => !o)}
        >
          <div className="flex items-center gap-2.5">
            <span className={cn(
              "text-sm font-semibold",
              step.status === "done" ? "text-foreground"
              : step.status === "running" ? "text-cyan-700"
              : step.status === "failed" ? "text-destructive"
              : "text-muted-foreground"
            )}>
              {step.label}
            </span>
            {step.completed_at && (
              <span className="text-xs text-muted-foreground font-mono">
                {new Date(step.completed_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
              </span>
            )}
          </div>
          {hasMeta && (
            <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors flex items-center gap-1">
              {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              {open ? "hide" : "details"}
            </span>
          )}
        </div>

        {open && hasMeta && (
          <div className="mt-2.5 rounded-xl bg-muted/30 border border-border p-3.5">
            {elementTypes && <ElementTypeBar types={elementTypes} />}
            {Object.keys(otherMeta).length > 0 && <MetadataGrid data={otherMeta} />}
          </div>
        )}
      </div>
    </div>
  );
}

export default function JobDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");

  const refresh = () => getJob(id).then(setJob).catch((e) => setError(e.message));

  useEffect(() => { refresh(); }, [id]);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") return;
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [job?.status]);

  if (error) return (
    <div className="p-8 flex items-center gap-2.5 text-sm text-destructive">
      <AlertCircle size={16} /> {error}
    </div>
  );
  if (!job) return (
    <div className="p-8 flex items-center gap-2.5 text-sm text-muted-foreground">
      <Loader2 size={16} className="animate-spin" /> Loading job…
    </div>
  );

  const stepMap = Object.fromEntries(job.steps.map((s) => [s.name, s]));
  const runningStep = job.status === "running"
    ? STEP_ORDER.find((name) => !stepMap[name])
    : undefined;

  const steps: Step[] = STEP_ORDER.map((name) =>
    stepMap[name] ?? {
      name,
      label: name.charAt(0).toUpperCase() + name.slice(1),
      status: name === runningStep ? "running" : "pending",
      metadata: {},
    }
  );

  const doneCount = steps.filter((s) => s.status === "done").length;
  const progressPct = Math.round((doneCount / STEP_ORDER.length) * 100);

  return (
    <div className="p-8 max-w-2xl mx-auto space-y-8">
      {/* Back */}
      <Link href="/jobs" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors">
        <ArrowLeft size={14} /> All Jobs
      </Link>

      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-start gap-3">
          <h1 className="text-2xl font-bold tracking-tight leading-tight flex-1 min-w-0 break-all">{job.filename}</h1>
          <StatusBadge status={job.status} />
        </div>
        <p className="text-xs text-muted-foreground font-mono">{job.job_id}</p>
      </div>

      {/* Progress */}
      <div className="rounded-2xl border border-border bg-card p-5 space-y-4">
        <div className="flex justify-between items-center text-sm">
          <span className="font-medium">Pipeline Progress</span>
          <span className={cn(
            "text-xl font-bold tabular-nums",
            doneCount === STEP_ORDER.length ? "text-olive-300" : "text-foreground"
          )}>
            {progressPct}%
          </span>
        </div>
        <Progress value={progressPct} className="h-2.5 rounded-full" />
        <p className="text-xs text-muted-foreground">{doneCount} of {STEP_ORDER.length} steps complete</p>
      </div>

      {/* Error */}
      {job.error && (
        <div className="flex gap-3 rounded-xl border border-destructive/30 bg-destructive/8 p-4 text-sm text-destructive">
          <AlertCircle size={16} className="shrink-0 mt-0.5" />
          <span>{job.error}</span>
        </div>
      )}

      {/* Steps */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <h2 className="text-sm font-semibold mb-5 text-muted-foreground uppercase tracking-wider">Steps</h2>
        <div>
          {steps.map((step, i) => (
            <StepRow key={step.name} step={step} index={i} isLast={i === steps.length - 1} />
          ))}
        </div>
      </div>
    </div>
  );
}
