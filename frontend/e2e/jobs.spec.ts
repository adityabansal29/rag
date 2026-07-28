import { test, expect } from "@playwright/test";

const API = "http://localhost:8000";

test("shows empty state when no jobs", async ({ page }) => {
  await page.route(`${API}/jobs`, (route) => route.fulfill({ json: [] }));
  await page.goto("/jobs");
  await expect(page.getByText("No jobs yet")).toBeVisible();
  await expect(page.getByRole("link", { name: /Go to Upload/i })).toBeVisible();
});

test("renders job row with filename and status", async ({ page }) => {
  const mockJob = {
    job_id: "test-123",
    filename: "quarterly-report.pdf",
    s3_key: "quarterly-report.pdf",
    status: "done",
    steps: Array.from({ length: 5 }, (_, i) => ({
      name: `step${i}`, label: `Step${i}`, status: "done", metadata: {},
    })),
    created_at: "2024-01-15T10:00:00Z",
    updated_at: "2024-01-15T10:01:00Z",
  };
  await page.route(`${API}/jobs`, (route) => route.fulfill({ json: [mockJob] }));
  await page.goto("/jobs");
  await expect(page.getByText("quarterly-report.pdf")).toBeVisible();
  await expect(page.getByText("Done")).toBeVisible();
  // steps counter: 5/5
  await expect(page.getByText(/5/)).toBeVisible();
});

test("renders running job with correct badge", async ({ page }) => {
  const mockJob = {
    job_id: "run-456",
    filename: "doc.pdf",
    s3_key: "doc.pdf",
    status: "running",
    steps: [
      { name: "parsing", label: "Parse", status: "done", metadata: {} },
      { name: "chunking", label: "Chunk", status: "running", metadata: {} },
      ...Array.from({ length: 3 }, (_, i) => ({
        name: `step${i}`, label: `Step${i}`, status: "queued", metadata: {},
      })),
    ],
    created_at: "2024-01-15T10:00:00Z",
    updated_at: "2024-01-15T10:00:30Z",
  };
  await page.route(`${API}/jobs`, (route) => route.fulfill({ json: [mockJob] }));
  await page.goto("/jobs");
  await expect(page.getByText("Running")).toBeVisible();
});

test("refresh button reloads job list", async ({ page }) => {
  let callCount = 0;
  await page.route(`${API}/jobs`, (route) => {
    callCount++;
    route.fulfill({ json: [] });
  });
  await page.goto("/jobs");
  await page.getByRole("button", { name: /Refresh/i }).click();
  expect(callCount).toBe(2); // initial load + refresh
});
