"""Memory structure: the associative graph and the temporal layer built on
top of what ingestion already extracted.

Everything in this package is *derived* data in the authority hierarchy's
lowest tier (see docs/ARCHITECTURE.md): it is rebuilt from
`MemoryUnit.semantic` / `MemoryUnit.explicit_metadata` and can be dropped and
regenerated without losing a single word the user wrote. Nothing here ever
writes to `authoritative_content`, `clarifications`, or `Capture.raw_text`.

Deliberately LLM-free. `app/ingestion/` spends the inference budget; this
package only reorganizes its output, which is what makes the graph free to
rebuild and deterministic to test.
"""
