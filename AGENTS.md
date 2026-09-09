# AGENTS.md

Read `CLAUDE.md` first; it is the canonical operating guide for this repo.

Everything about the project — stack, structure, the statistical model, how to
create a new widget, and the editorial constraints from El Observador — lives
there. This file exists only so that Codex and other agents that look for
`AGENTS.md` are pointed at it, and must not carry instructions of its own.

Two more files worth reading before touching a widget:

- `widgets/seguridad/WIDGET_README.md` — the security widget in full: four
  models, calibration, the specification envelope, what it supports and what it
  does not.
- `widgets/_template/WIDGET_README.md` — the scaffold for a new widget, plus
  "Lo que aprendimos con el widget de seguridad", which is the part that
  generalises.
