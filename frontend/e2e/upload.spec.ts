import { test, expect } from "@playwright/test";

test("renders heading and all pipeline steps", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Upload a Document" })).toBeVisible();
  for (const step of ["Parse", "Chunk", "Enrich", "Embed", "Store"]) {
    await expect(page.getByText(step, { exact: true })).toBeVisible();
  }
});

test("rejects unsupported file type", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "virus.exe",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("not a doc"),
  });
  await expect(page.getByText(/Unsupported file type/)).toBeVisible();
  // upload button stays disabled
  await expect(page.getByRole("button", { name: /Upload & Process/i })).toBeDisabled();
});

test("accepts pdf and enables upload button", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "report.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.4 fake content"),
  });
  await expect(page.getByText("report.pdf")).toBeVisible();
  await expect(page.getByRole("button", { name: /Upload & Process/i })).toBeEnabled();
});

test("clears file when X button clicked", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "doc.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.4"),
  });
  await expect(page.getByText("doc.pdf")).toBeVisible();
  // click the round X button to clear the selected file
  await page.locator("button.rounded-full").click();
  // Wait for the dropzone to reset
  await expect(page.getByText(/Drag & drop/i)).toBeVisible();
});
