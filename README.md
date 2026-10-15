# membase

**An exploration of augmented personal memory, at the boundary of
information retrieval, language models, and human-computer interaction.**

membase is a research prototype, not a product. It exists to investigate
a narrow, specific question: what does it take to build a system that
*remembers for you* — honestly, auditably, and without inventing things it
doesn't actually know — rather than one that merely searches text or
generates plausible-sounding answers about it?

This repository, and the design notes under `docs/`, are shared as a
record of that investigation: the architecture it converged on, the
decisions it reversed, and the evaluation results that shaped it. It is
not packaged, hardened, or supported for running your own instance, and
nothing here should be read as an invitation to do so.

## The problem this investigates

Most systems that sit between a person and a language model over their
own data fall into one of two failure modes. A notes app with search is
honest but dumb: it can't answer a question phrased differently from how
the note was written. A general RAG chatbot is clever but unreliable: it
will confidently answer from weak or irrelevant evidence, and there is
usually no clean way to tell, after the fact, exactly what it was looking
at when it did.

membase treats that gap as the actual research problem, not an
engineering detail to paper over. Three questions run through the whole
design:

- **Information retrieval**: how do you combine exact lexical matching
  with semantic/vector search over *both* what a person literally wrote
  and what a model derived from it, without letting the derived layer
  quietly become the thing being searched *instead of* the original?
  How do you make a system *refuse* to answer, and say so plainly, when
  its own evidence doesn't actually support a claim — rather than
  treating abstention as a generation-quality afterthought?
- **Language models as an inference layer, not a database**: every
  LLM-derived field in this system is explicitly provisional and
  regenerable, layered underneath what the person actually typed, never
  overwriting or standing in for it. What happens to a memory architecture
  when you take that separation seriously at every layer, not just at the
  top-level answer?
- **Human-computer interaction**: when is it worth interrupting someone
  to ask a clarifying question about something they just told the system,
  and when does that turn a two-second note into an interrogation? How
  much ambiguity should a recall system resolve silently, versus handing
  back to the person asking, versus simply admitting it doesn't know?

## What it actually does

- **Capture → one or more memory units.** A single piece of writing can
  describe several unrelated facts; the system decides whether to keep it
  as one coherent episode or split it into independently retrievable
  units — the person never has to make that call themselves.
- **An authority hierarchy, enforced structurally.** What a person typed
  outranks what they later clarified, which outranks what the model
  inferred, which outranks the fully-derived indexes built from all of
  it. Correcting or re-deriving the bottom layers never touches the top.
- **Clarification as gating, not chat.** A genuinely ambiguous term or
  date can hold a memory out of search until answered — bounded (at most
  a few rounds, never a batch-interrogation) and skippable, and a real
  answer becomes a permanent part of that memory's semantics rather than
  a transient conversational turn.
- **Grounded retrieval with explicit abstention.** Hybrid (vector +
  full-text) search over original text, derived semantics, and chunked
  attachment content, feeding a bounded agentic search loop that is
  instructed to answer *only* from what it actually retrieved — and to
  say plainly that no stored memory addresses the question, rather than
  guess, when that's the truth.
- **An iterative recall mode.** A separate, fully-implemented path that
  can hold a multi-round dialogue when several stored memories are
  plausible — asking one concrete discriminating question instead of
  picking arbitrarily. (A benchmark found it underperforming the simpler
  one-shot agent on a deliberately ambiguous test corpus; see
  `docs/DECISIONS.md` for why it's kept anyway, as a live comparison
  point rather than the default.)
- **A background "structurer" pass** that deepens an already-answerable
  memory's representation over time — finding related memories,
  enriching semantics — without ever blocking on it or touching whether
  the memory is already searchable.
- **Contradiction surfacing, not resolution.** When two stored memories
  disagree on a concrete fact, the system flags it to be stated plainly
  in a future answer, rather than silently picking a winner.

## Architecture, briefly

Every idea above is implemented as a real abstraction, not a prompt
trick: `Capture` (the verbatim original), `MemoryUnit` (one retrievable
piece of memory, carrying authoritative content, an append-only
clarification history, derived semantics, and a status gate from
`draft` through `ready`), a bounded retrieval `Agent`, and a persisted
`RecallSession` for the iterative mode. The full design, including the
parts that were tried and reversed, is in `docs/ARCHITECTURE.md` and
`docs/DECISIONS.md`; `docs/REPORT.md` has the engineering/evaluation
history across phases, and `docs/TODO.md` the open questions this
investigation hasn't settled yet.

## On deployment

This is explicitly not framed as a tool for everyday use, and the
documentation here is not written to encourage self-hosting it. The
value of this repository is the architecture and the reasoning behind
it — a research artifact to read, not a product to install.
