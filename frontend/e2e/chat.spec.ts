import { test, expect } from "@playwright/test";

test("shows empty state on first load", async ({ page }) => {
  await page.goto("/chat");
  await expect(page.getByRole("heading", { name: "Chat" })).toBeVisible();
  await expect(page.getByText("Ask anything")).toBeVisible();
  await expect(page.getByText("Questions are answered from your indexed documents.")).toBeVisible();
});

test("send button is disabled when input is empty", async ({ page }) => {
  await page.goto("/chat");
  await expect(page.getByRole("button", { name: "Send message" })).toBeDisabled();
});

test("send button enables after typing", async ({ page }) => {
  await page.goto("/chat");
  await page.locator("textarea").fill("hello");
  await expect(page.getByRole("button", { name: "Send message" })).toBeEnabled();
});

test("suggestion prompt fills textarea", async ({ page }) => {
  await page.goto("/chat");
  await page.getByText("Summarize the key findings").click();
  await expect(page.locator("textarea")).toHaveValue("Summarize the key findings");
  await expect(page.getByRole("button", { name: "Send message" })).toBeEnabled();
});

test("new chat button resets session", async ({ page }) => {
  await page.goto("/chat");
  const threadBefore = await page.locator("p.font-mono").textContent();
  await page.getByRole("button", { name: /New chat/i }).first().click();
  const threadAfter = await page.locator("p.font-mono").textContent();
  expect(threadBefore).not.toBe(threadAfter);
});
