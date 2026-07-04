# RAG Frontend

Next.js 16 App Router UI for the RAG pipeline system.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Upload page — drag-and-drop or file picker (PDF, DOCX, TXT), upload progress bar |
| `/jobs` | Job history — list of all ingestion jobs with status badges |
| `/jobs/[id]` | Job detail — vertical step tracker with per-step metadata; polls every 2s while running |
| `/chat` | Chat — conversational RAG with collapsible source citations (score, type, page, file) |

## Running

```bash
npm install
npm run dev   # http://localhost:3000
```

Requires the backend API running on `:8000`. Set `NEXT_PUBLIC_API_URL` in `.env.local` if you change the port.

## Stack

- Next.js App Router
- shadcn/ui + Tailwind v4
- Custom palette: `olive-300`, `mist-400`, `cyan-700` (defined in `app/globals.css`)
- `lib/api.ts` — typed API client (`Job`, `Step`, `Source`)

## Notes

- Chat `thread_id` is generated once per browser session via `crypto.randomUUID()` and stored in `sessionStorage`. The backend checkpointer (LangGraph + DynamoDB) uses it to maintain conversation state across requests.
- Job detail polls only while `status !== "done" && status !== "failed"`.
- Job list makes a single fetch on mount — no polling.
