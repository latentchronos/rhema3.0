# Local Agent Progress

Rhema uses an optional local-only progress file so developers and AI agents can continue work accurately across chats or sessions.

## File Location

Use:

```text
.agent-progress/progress.md
```

This file is intentionally ignored through `.git/info/exclude`, not `.gitignore`. That keeps the continuation state private to the local checkout and avoids pushing it to GitHub.

## When To Update It

Update `.agent-progress/progress.md` only after a completed repo change.

Examples that should be recorded:

- Code was modified.
- Documentation was added or updated.
- Tests, scripts, configuration, or schemas were changed.
- A phase was completed or deliberately paused after concrete file changes.

Examples that should not be recorded:

- Discussion with no file changes.
- Research notes that have not changed the repo.
- Brainstorming.
- Half-finished work that has not reached a coherent stopping point.

## Required Entry Content

Each update should include:

- date
- agent or developer name
- phase or workstream
- files changed
- summary
- verification run
- known issues
- next exact step

The goal is not to create a long diary. The goal is to give the next developer or agent enough context to continue surgically.

## Continuation Workflow

When opening a new chat or starting a new agent session:

1. Read `.agent-progress/progress.md`.
2. Confirm the active workstream and next exact step.
3. Inspect the referenced files before modifying anything.
4. Complete the next coherent change.
5. Run the appropriate verification.
6. Update `.agent-progress/progress.md` after the change is done.

If `.agent-progress/progress.md` does not exist yet, create it from:

```text
docs/progress-template.md
```

## Public Versus Private Tracking

Use public docs for plans contributors should see:

```text
docs/production-readiness-roadmap.md
```

Use the local progress file for private continuation state:

```text
.agent-progress/progress.md
```

## Important Limitation

Because `.agent-progress/progress.md` is intentionally not pushed, it only helps agents and developers working in the same local checkout. Public project direction should still live in committed docs such as the roadmap.
