# Fix Tracker

## Phase 1: Isolated Correctness Bugs
- [ ] Task 1: Fix vector store score threshold bugs (#3, #4) — `chroma_store.py`, `pinecone_store.py`
- [ ] Task 2: Fix semaphore private attr access (#8) — `enricher.py`
- [ ] Task 3: Fix SQS infinite retry + graceful shutdown (#1, #19) — `sqs_listener.py`
- [ ] **Checkpoint 1** — smoke test: ingest + query

## Phase 2: Shared Infrastructure
- [ ] Task 4: Extract shared config + fix env var handling (#9, #18) — `backend/config.py`, `state.py`, `sqs_listener.py`
- [ ] Task 5: Make worker async (#2) — `sqs_listener.py`
- [ ] Task 6: Configurable LLM model (#11) — `app.py`, `pipeline.py`, `.env.example`
- [ ] **Checkpoint 2** — single async event loop, clear env errors

## Phase 3: Data Correctness
- [ ] Task 7: Fix chunk ID collision (#5) — `models.py` ⚠️ requires re-index
- [ ] Task 8: Persist parent summaries in child metadata (#7) — `pipeline.py`
- [ ] **Checkpoint 3** — idempotent re-ingest, parent context in answers

## Phase 4: API and Search Logic
- [ ] Task 9: Structured `rag_search` output (#10) — `routes.py`
- [ ] Task 10: QueryRewriter failure signaling (#6) — `query_rewriter.py`, `fusion/pipeline.py`
- [ ] Task 11: Clarify fusion SearchParams (#12) — `fusion/pipeline.py`
- [ ] **Checkpoint 4** — chat returns structured sources, fusion degradation observable

## Phase 5: Observability and Performance
- [ ] Task 12: Replace print with logging (#14) — all files
- [ ] Task 13: DynamoDB Scan pagination (#17) — `routes.py`
- [ ] Task 14: Batch enrichment (#16) — `enricher.py`
- [ ] Task 15: Delete sync asyncio.run wrappers (#13) — `hybrid/pipeline.py`, `fusion/pipeline.py`
- [ ] **Final checkpoint** — full pipeline smoke test, no print() in production code

## Deferred
- Issue #15 (Chroma BM25 full-corpus scan) — needs external BM25 index, out of scope
