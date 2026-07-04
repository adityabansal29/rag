"use client";

import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Upload, FileText, X, ArrowRight, CheckCircle2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { getPresignedUrl, uploadToS3, notifyUpload } from "@/lib/api";
import { cn } from "@/lib/utils";

type Stage = "idle" | "presign" | "upload" | "notify" | "done" | "error";

const ACCEPTED = [".pdf", ".doc", ".docx", ".txt"];

const PIPELINE_STEPS = ["Parse", "Chunk", "Enrich", "Embed", "Store"];

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function UploadPage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");

  const accept = (f: File) => {
    const ext = "." + f.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED.includes(ext)) {
      setError(`Unsupported file type. Accepted: ${ACCEPTED.join(", ")}`);
      return;
    }
    setFile(f);
    setError("");
    setStage("idle");
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) accept(f);
  }, []);

  const upload = async () => {
    if (!file) return;
    setError("");
    try {
      setStage("presign");
      const { url, key } = await getPresignedUrl(file.name);
      setStage("upload");
      await uploadToS3(url, file, setProgress);
      setStage("notify");
      const { job_id } = await notifyUpload(key, file.name);
      setStage("done");
      router.push("/jobs/" + job_id);
    } catch (err) {
      setStage("error");
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  };

  const stageLabel: Record<Stage, string> = {
    idle:    "",
    presign: "Preparing secure upload…",
    upload:  `Uploading — ${progress}%`,
    notify:  "Queuing for processing…",
    done:    "Redirecting to job…",
    error:   "",
  };

  const busy = stage !== "idle" && stage !== "error";

  return (
    <div className="flex items-center justify-center min-h-screen p-8">
      <div className="w-full max-w-xl space-y-8">

        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Upload a Document</h1>
          <p className="text-base text-muted-foreground">
            We'll run it through a 5-step pipeline and make it searchable via chat.
          </p>
        </div>

        {/* Pipeline preview */}
        <div className="flex items-center gap-1.5">
          {PIPELINE_STEPS.map((step, i) => (
            <div key={step} className="flex items-center gap-1.5">
              <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-muted text-muted-foreground">
                {step}
              </span>
              {i < PIPELINE_STEPS.length - 1 && (
                <ArrowRight size={10} className="text-border shrink-0" />
              )}
            </div>
          ))}
        </div>

        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !busy && inputRef.current?.click()}
          className={cn(
            "relative flex flex-col items-center justify-center gap-5 rounded-2xl border-2 border-dashed p-14 cursor-pointer transition-all duration-200",
            dragging
              ? "border-cyan-700 bg-cyan-700/5 scale-[1.01]"
              : file
              ? "border-olive-300/40 bg-olive-300/5"
              : "border-border hover:border-cyan-700/40 hover:bg-muted/20"
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED.join(",")}
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) accept(f); }}
          />

          {file ? (
            <>
              <div className="w-16 h-16 rounded-2xl bg-olive-300/15 flex items-center justify-center">
                <FileText size={30} className="text-olive-300" />
              </div>
              <div className="text-center space-y-1">
                <p className="font-semibold text-base">{file.name}</p>
                <p className="text-sm text-muted-foreground">{formatSize(file.size)}</p>
              </div>
              {!busy && (
                <button
                  onClick={(e) => { e.stopPropagation(); setFile(null); setStage("idle"); setError(""); }}
                  className="absolute top-4 right-4 w-8 h-8 rounded-full bg-muted flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                >
                  <X size={14} />
                </button>
              )}
            </>
          ) : (
            <>
              <div className={cn(
                "w-16 h-16 rounded-2xl flex items-center justify-center transition-colors",
                dragging ? "bg-cyan-700/20" : "bg-muted"
              )}>
                <Upload size={26} className={dragging ? "text-cyan-700" : "text-muted-foreground"} />
              </div>
              <div className="text-center space-y-1">
                <p className="font-semibold text-base">
                  {dragging ? "Drop it!" : "Drag & drop your file"}
                </p>
                <p className="text-sm text-muted-foreground">or click to browse your computer</p>
              </div>
              <p className="text-xs text-muted-foreground/70 font-mono tracking-wider">PDF · DOC · DOCX · TXT</p>
            </>
          )}
        </div>

        {/* Progress */}
        {busy && (
          <div className="space-y-3">
            <div className="flex justify-between items-center text-sm">
              <span className="text-muted-foreground">{stageLabel[stage]}</span>
              {stage === "upload" && (
                <span className="font-semibold tabular-nums text-foreground">{progress}%</span>
              )}
            </div>
            <Progress value={stage === "upload" ? progress : null} className="h-2 rounded-full" />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-xl bg-destructive/10 border border-destructive/20 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* CTA */}
        <button
          onClick={upload}
          disabled={!file || busy}
          className="w-full h-13 flex items-center justify-center gap-2.5 rounded-xl text-base font-semibold text-white gradient-primary-btn disabled:cursor-not-allowed disabled:opacity-40"
          style={{ height: "52px" }}
        >
          {stage === "done" ? (
            <CheckCircle2 size={18} />
          ) : (
            <Upload size={17} />
          )}
          {busy ? stageLabel[stage] : "Upload & Process"}
        </button>
      </div>
    </div>
  );
}
