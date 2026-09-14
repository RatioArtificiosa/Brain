# 🧠 Brain — Virtual Neural Runtime (VNR)

> **Can a computer maintain a logically gigantic neural system while physically
> instantiating only the neural computation that is currently relevant?**
> How far can that compression go before behavior changes — and what determines the limit?

**Brain** is an open experimental platform for *procedurally generated, dynamically
materialized* neural simulation. Instead of storing billions of synapses, it stores the
**rules that generate them** — then materializes neurons only where activity propagates,
and garbage-collects the rest. The seed architecture comes from a real animal: the complete
adult fruit-fly connectome. The goal is not to claim a human brain in a box. It is to find,
experimentally, the **compression-vs-fidelity knee**: how much virtual brain a normal
computer can host before the approximation breaks.

No supercomputer required. This runs on a desktop.

---

## ⚡ The idea in 30 seconds

```text
FLY CONNECTOME ──► circuit DNA ──► VIRTUAL BRAIN (billions of logical neurons)
                                              │
                        only the ACTIVE FRONTIER is real at any instant
                                              │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
               CPU / 64 GB RAM              GPU / 8 GB VRAM
          rules · populations · memory       spikes · plasticity
```

A neuron is a deterministic ID (`hash(seed, lineage)`) plus generative rules. When it
fires, its targets are *generated on demand* — then evicted when activity decays. Identity
persists; computational state virtualizes. Like virtual memory, but for neurons.

## 🔬 Scientific backbone

Every technique here is published and reproduced — the research contribution is their
combination, measured honestly:

| Result | Source |
|---|---|
| Procedural connectivity: **4.13M neurons / 24.2B synapses on one GPU** | Knight & Nowotny 2021, *Nature Computational Science* |
| Complete adult fly connectome: **~139K neurons / ~50M synapses / 8,400+ cell types** | FlyWire consortium, *Nature* 2024 |
| Whole-brain fly LIF model reproducing sensorimotor processing | Shiu et al. 2024, *Nature* |

What no existing framework (GeNN, Brian2, NEST, Norse, snnTorch, Lava) does all at once:
fly-derived generative rules **+** deterministic regeneration **+** promotion/demotion
plasticity cache **+** compression-vs-fidelity as the reported metric. That's this project.

## 🖥️ Hardware honesty

| | Target dev machine | Budget rule |
|---|---|---|
| RAM | 64 GB | 15% reserve → ~54 GB usable |
| GPU | NVIDIA 8 GB VRAM | 20% reserve → ~5 GB usable |
| OS | Windows 11 (+ WSL2 for GeNN later) | PyTorch-first, GeNN optional |

Memory is *not* the binding constraint — event throughput is (~10⁹ synaptic events/sec at
scale stays in kernels, never crosses the Python boundary per-event). The fly baseline
(139K neurons, ~800 MB explicit) runs anywhere; 1M virtual neurons need single-digit MB
resident at realistic 1–5% active density.

## 🗺️ Roadmap

| Phase | Goal | Status |
|---|---|---|
| PH0 Bootstrap | Repo, CLI, config, hardware discovery | ✅ Done |
| PH1 Deterministic core | IDs, LIF, events, frontier, materialization, procedural connectivity | 🔨 In progress |
| PH2 Validation suite | Oracle tests; nothing proceeds until green | ⬜ |
| PH3 Backends | NumPy oracle → PyTorch CUDA → Rust native spike → GeNN (WSL2) | ⬜ |
| PH4 Connectome + generators | FlyWire ingest, motif DNA, 8 scaling generators | ⬜ |
| PH5 Scaling science | **The knee graph**: compression vs behavioral fidelity, with controls | ⬜ |
| PH6 Cognition + decoder | Tasks, roles, neuromodulators, external LLM-as-observer decoder | ⬜ |
| PH7 Observatory UI | Live brain observatory (detail-on-demand visualization) | ⬜ |
| PH8 Voice + embodiment | Thought events → speech, micro-world, autonomous operation | ⬜ |

The single most important output: **one graph** — compression ratio (x) vs behavioral
fidelity (y) — and the location of its knee. A null result that maps the failure boundary
is explicitly acceptable science.

## 🚀 Quickstart

```powershell
pip install -e .
vnr doctor
```

```text
VNR SYSTEM DIAGNOSTICS
CPU ..................... Intel64 (12 threads)
RAM ..................... 63.9 GB
NVIDIA GPU .............. Quadro M5000
VRAM .................... 8.0 GB
STATUS: READY
```

```powershell
pytest        # full suite: unit + integration + scientific oracle tests
ruff check .  # lint must stay clean — no warnings land on main
```

## 📁 Layout

```text
src/vnr/        core/ (ids, lif, events, frontier) · memory/ · connectome/
                generate/ · backend/ (cpu → torch → rust → genn) ·
                cognition/ · decoder/ · experiments/ · observe/ · voice/ · ui/
tests/          unit · integration · scientific · performance
benchmarks/     cpu · gpu · procedural · memory · scaling
scripts/        data + ops tooling (incl. the agent watchdog)
```

## 🤖 How this repo is built

Autonomous agent development against a living plan: every subsystem goes
*interface → test → reference implementation → benchmark → optimize*, small
conventional commits, phase gates that actually gate (PH2 must be green before
FlyWire ingest; UI comes last). Progress: `G:\BRAIN\VNR\02-checklist.md` in the
project workspace; lab log: `notes-and-observations.md`.

## 📜 Rules of the lab

- Hypothesis, observation, measured result, and interpretation are recorded separately.
- Never claim human equivalence, consciousness, or sentience without evidence.
- The LLM is an *observer/decoder outside* the simulator — never the brain, never the memory.
- Scale is always reported decomposed (logical / explicit / population / procedural-dormant).
- No stubs on main. No silent scope drops.

## 📄 License

MIT — see [LICENSE](LICENSE).
