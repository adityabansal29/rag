const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type JobStatus = "queued" | "running" | "done" | "failed";

export interface Step {
  name: string;
  label: string;
  status: string;
  completed_at?: string;
  metadata: Record<string, unknown>;
}

export interface Job {
  job_id: string;
  filename: string;
  s3_key: string;
  status: JobStatus;
  steps: Step[];
  created_at: string;
  updated_at: string;
  error?: string | null;
}

export interface Source {
  content: string;
  score: number;
  type: string;
  page: string;
  source: string;
}

export async function getPresignedUrl(filename: string) {
  const res = await fetch(`${API}/presigned-url?filename=${encodeURIComponent(filename)}`);
  if (!res.ok) throw new Error("Failed to get presigned URL");
  return res.json() as Promise<{ url: string; key: string }>;
}

export async function uploadToS3(url: string, file: File, onProgress: (pct: number) => void) {
  return new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    });
    xhr.addEventListener("load", () => (xhr.status < 300 ? resolve() : reject(new Error(`S3 upload failed: ${xhr.status}`))));
    xhr.addEventListener("error", () => reject(new Error("S3 upload network error")));
    xhr.open("PUT", url);
    xhr.send(file);
  });
}

export async function notifyUpload(key: string, filename: string) {
  const res = await fetch(`${API}/notify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key, filename }),
  });
  if (!res.ok) throw new Error("Failed to notify upload");
  return res.json() as Promise<{ job_id: string; status: string }>;
}

export async function listJobs(): Promise<Job[]> {
  const res = await fetch(`${API}/jobs`);
  if (!res.ok) throw new Error("Failed to fetch jobs");
  return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`${API}/job/${jobId}`);
  if (!res.ok) throw new Error("Job not found");
  return res.json();
}

export async function sendChat(message: string, threadId: string) {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, thread_id: threadId }),
  });
  if (res.status === 409) throw Object.assign(new Error("Session corrupted"), { code: "SESSION_CORRUPTED" });
  if (!res.ok) throw new Error("Chat request failed");
  return res.json() as Promise<{ answer: string; sources: Source[] }>;
}
