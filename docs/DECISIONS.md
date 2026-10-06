# Decisions

Only decisions with real alternatives and real consequences are recorded
here — not every implementation detail.

---

**Decision:** PostgreSQL + pgvector + built-in full-text search, one
database, no separate vector store.

**Reason:** Keeps the MVP architecture simple and colocates structured
metadata, full-text search, and vector search in one place with one
transaction model.

**Alternatives considered:** Qdrant/Weaviate/Chroma + Postgres for
metadata (two systems to run and keep consistent, no concrete benefit at
this scale); SQLite + a vector extension (weaker full-text/concurrency
story, and pgvector's HNSW index is more mature).

---

**Decision:** Dependencies come entirely from `nixpkgs.python313Packages`
via the flake's `python3.withPackages`; there is no `requirements.txt` /
`venv` / `pip install` anywhere in this repo.

**Reason:** Binary wheels from PyPI routinely fail to run on NixOS
(missing dynamic linker paths/shared libs). Every dependency this project
needs happens to already be packaged in nixpkgs, so there's no reason to
take on that failure mode at all. See CLAUDE.md footgun #2.

**Alternatives considered:** `uv`/`pip` inside the nix devShell with
`nix-ld` as a fallback for binary wheels -- more moving parts, deferred
unless a future dependency genuinely isn't in nixpkgs.

---

**Decision:** One FastAPI process serves both the JSON API and
server-rendered (Jinja2) HTML; browser-side JS only calls the JSON API.

**Reason:** Satisfies "separate frontend from memory service" (spec
section 16) without standing up a second process/build toolchain for an
MVP. The browser has zero business logic -- it's a thin client over HTTP,
same as a future mobile client would be.

**Alternatives considered:** A separate Vite/React SPA -- more moving
parts (build step, CORS, two dev servers) for no MVP-stage benefit.

---

**Decision:** Embeddings are computed from `caption + derived summary +
topics` only, never from raw attachment text; raw attachment text instead
feeds the full-text index via `searchable_text`.

**Reason:** Keeps embedding input short and semantically dense (embedding
models degrade on long/noisy input), while still making attachment
content searchable through the lexical channel.

**Alternatives considered:** Chunking attachments into multiple embedded
passages, like a typical RAG pipeline. Explicitly out of scope -- the
spec frames this project as *not* a RAG chatbot, and per-chunk retrieval
adds real complexity (chunk boundaries, per-chunk provenance) for a
document-heavy use case the MVP doesn't target.

---

**Decision:** A memory is persisted immediately on creation, even if the
LLM flags a clarification question; the clarification only enriches
`derived` afterward, it never gates initial persistence.

**Reason:** The spec's own ingestion diagram shows clarification before
"commit," but gating persistence on an answered question would mean the
authoritative caption could be lost if the user never answers, which
conflicts with the durability requirement ("memory persists after
restart") and the instruction to not interrogate the user unnecessarily.

**Alternatives considered:** A draft/staging state machine for
not-yet-committed memories. More correct to the letter of the spec
diagram, but meaningfully more state to design and test for an MVP whose
clarification feature is explicitly supposed to be simple. Flagged as an
open question in docs/TODO.md since it's a product call, not purely an
engineering one.

---

**Decision:** No separate reranker model/service. The hybrid fused score
(0.65 × normalized vector similarity + 0.35 × normalized full-text rank)
doubles as the "reranking" step the spec's pipeline diagram calls for.

**Reason:** A cross-encoder reranker is a real model to host, run, and
evaluate; score fusion over two already-computed signals is most of the
practical benefit for an MVP's data volumes (dozens-hundreds of memories,
not millions).

**Alternatives considered:** A local cross-encoder via Ollama/another
runtime. Deferred -- see docs/TODO.md; worth revisiting if eval results
show the fused score materially misordering results at larger scale.
