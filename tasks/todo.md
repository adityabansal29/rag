# Fix Tracker

## Phase 1: Isolated Correctness Bugs
- [x] Task 1: Fix vector store score threshold bugs (#3, #4) — `chroma_store.py`, `pinecone_store.py`
- [x] Task 2: Fix semaphore private attr access (#8) — `enricher.py`
- [x] Task 3: Fix SQS infinite retry + graceful shutdown (#1, #19) — `sqs_listener.py`
- [x] **Checkpoint 1** — 9/9 tests pass, py_compile clean

## Phase 2: Shared Infrastructure
- [x] Task 4: Extract shared config + fix env var handling (#9, #18) — `backend/config.py`, `state.py`, `sqs_listener.py`
- [x] Task 5: Make worker async (#2) — `sqs_listener.py`
- [x] Task 6: Configurable LLM model (#11) — `app.py`, `pipeline.py`, `.env.example`
- [x] **Checkpoint 2** — 21/21 tests pass, py_compile clean

## Phase 3: Data Correctness
- [x] Task 7: Fix chunk ID collision (#5) — `models.py` ⚠️ requires re-index
- [x] Task 8: Persist parent summaries in child metadata (#7) — `pipeline.py`
- [x] **Checkpoint 3** — 27/27 tests pass, py_compile clean

## Phase 4: API and Search Logic
- [x] Task 9: Structured `rag_search` output (#10) — `routes.py`
- [x] Task 10: QueryRewriter failure signaling (#6) — `query_rewriter.py`, `fusion/pipeline.py`
- [x] Task 11: Clarify fusion SearchParams (#12) — `fusion/pipeline.py`
- [x] **Checkpoint 4** — 34/34 tests pass; chat returns structured sources, fusion degradation observable, params helper extracted

## Phase 5: Observability and Performance
- [x] Task 12: Replace print with logging (#14) — all files; LOG_LEVEL env var; DEBUG for score tables, INFO for summaries, WARNING for degraded paths
- [x] Task 13: DynamoDB Scan pagination (#17) — `routes.py`; added `Limit=50` to scan call
- [ ] Task 14: Batch enrichment (#16) — `enricher.py`
- [ ] Task 15: Delete sync asyncio.run wrappers (#13) — `hybrid/pipeline.py`, `fusion/pipeline.py`
- [ ] **Final checkpoint** — full pipeline smoke test, no print() in production code

## Deferred
- Issue #15 (Chroma BM25 full-corpus scan) — needs external BM25 index, out of scope
