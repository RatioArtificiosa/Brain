# VNR Master Plan — the single document that always says what to do next

**Project:** Virtual Neural Runtime (VNR) · **Version:** 1.0 · **Date:** 2026-09-14
**Companion docs:** `00-research-brief.md` (why) · `02-checklist.md` (tracking) · `notes-and-observations.md` (log)
**Source spec:** the 112-section ChatGPT/VNR specification (cited as §N below)

## 0. How to work from this document (read this first, every session)

1. Open `02-checklist.md`. The first unchecked item is your current task. Its ID (e.g. `PH1-WI03`) maps to a work item below.
2. Read that work item fully: objective → exact steps → acceptance criteria → deliverables. Do the steps in order.
3. Before coding a subsystem: write the interface, write the smallest test, write the reference implementation, benchmark, then optimize (§96). Never invert this order.
4. When the acceptance criteria pass, check the item off in `02-checklist.md` and append a dated entry to `notes-and-observations.md` (what worked, what didn't, measurements).
5. If you discover a better approach mid-task, you are authorized to change the plan: update the work item, record the decision and reason in `notes-and-observations.md`, and continue. The checklist is a live instrument, not a cage.
6. Never leave stubs: a checked item means tested, documented, committed code. If you find adjacent unfinished work, finish it, log it, then continue.
7. End every task with the §97 report (task, files, tests, benchmark, limitations, next).

## 1. Mission, hypotheses, non-goals

Mission (§1): build an experimental platform whose **logical** neural scale can vastly exceed its **physically resident** state, seeded by the adult *Drosophila* connectome, running on 64 GB RAM + 8 GB VRAM, Windows 11. Primary hypothesis (§62): sparse event-driven execution + procedural connectivity preserves measurable dynamics with a small resident subset. Secondaries: connectome-derived structure beats matched random controls; materialization/eviction cuts memory without proportional task loss; selective promotion of learned connections beats indiscriminate persistence. Non-goals (§63): no claims of consciousness, human equivalence, or sentience; the LLM stays outside the cognitive loop as observer/decoder (§108–109); UI comes last (§106).

## 2. Locked foundational decisions (change only with a log entry)

- **D2.1** Neuron model order: LIF first (§33), Izhikevich/AdEx only after the pipeline validates. Rationale: validatable, published fly baseline exists (Shiu et al. 2024).
- **D2.2** GPU backend is PyTorch-first on native Windows; GeNN/PyGeNN is an optional adapter built later under WSL2 (§10, §17). Rationale: PyGeNN-on-Windows friction is a known schedule risk; torch is sufficient for Phases 1–4.
- **D2.3** Deterministic GPU RNG = counter-based (Philox, via `torch.Generator`) keyed by `(global_seed, source_id, target_population_id, connectivity_version)` (§23). Stateless, parallel-safe, replayable. No uncontrolled global randomness anywhere.
- **D2.4** Event timestamps are integer ticks, `DT = 0.1 ms` default (§24). No float scheduling.
- **D2.5** Virtual neuron IDs are uint64 deterministic hashes of `(root_seed, module_id, lineage_id, generation, local_index)` (§20). Same config + seed reconstructs the same neuron — the invariant the whole runtime rests on.
- **D2.6** Three state layers are separated from day one: structural (stable), long-term learned (compressed persistent), transient (evictable) (§26). Nothing transient may ever silently carry learned information.
- **D2.7** Replay tests demand bit-exactness for static networks and **statistical** tolerance (spike-train distance + task bands) once plasticity is enabled. Rationale: STDP update ordering makes bit-replay infeasible; insisting on it would fail forever.
- **D2.8** Memory safety margins: 20% VRAM reserve, 15% RAM reserve, configurable; the app must never consume the last available memory (§11, §31).
- **D2.9** Active-frontier budgeting assumes 1–5% active density for driven tasks, not the 0.0025% sparse-bench figure. Rationale: energy-bench sparsity is stimulus-dependent; the math still fits (see §5).
- **D2.10** Scale is always reported decomposed: logical / explicit / population / procedural-dormant with percentages (§87–88). Never a single neuron count.
- **D2.15** Language track: the reference oracle stays Python/NumPy forever (§96). Rust enters at
  PH3 as the *native hot-loop candidate* (event kernel, pools, materialization) via PyO3 —
  never as a rewrite of validated Python. Rust code is gated by rust-analyzer + `cargo clippy`
  + tests, same as Python's ruff + pytest. (Toolchain verified 2026-09-14: cargo/rustc/
  rust-analyzer 1.97.1. Full-Rust rewrite rejected: discards the validated oracle for no
  measured reason; revisit only with profiling data.)

## 3. Stack and environment

Python 3.13 (verified on this machine 2026-09-14; minimum 3.11 for `Self`/match usage). Core deps: `numpy`, `pyyaml`, `click` (CLI), `pytest`, `pandas` + `pyarrow` (parquet metrics), `matplotlib` (plots). GPU phase adds `torch` + CUDA 12.x (verify `torch.cuda.is_available()` on the Quadro M5000, compute capability 5.2 — flagged risk, verify in PH3). Data phase adds `caveclient`, `pyarrow`. Optional: `genn` (WSL2 only), `react` UI (Phase 7). Dev: `ruff`, git. Data layout: `G:\BRAIN\VNR\data\` (raw connectome, never committed), `G:\BRAIN\VNR\vnr\` (code repo, committed), artifacts under `vnr/artifacts/` git-ignored. FlyWire access requires free account + auth token: `vnr dataset auth` stores it in user config, never in repo.

## 4. Repository layout (final — scaffold exactly this)

```text
vnr/
├── pyproject.toml            # stdlib-first core; torch/caveclient as extras
├── README.md                 # mission, 5-min quickstart, hardware table
├── LICENSE (MIT)             # decide day one, log it
├── src/vnr/
│   ├── cli.py                # vnr doctor|dataset|connectome|generate|simulate|benchmark|experiment|report
│   ├── config.py             # YAML config + validation + config-hash
│   ├── hardware.py           # discovery → HardwareProfile + RuntimeBudget
│   ├── core/{ids.py,neuron.py(lif),events.py,frontier.py,materialize.py,procedural.py}
│   ├── memory/{tiers.py,evict.py,pools.py,synapse_cache.py,gc.py,checkpoint.py}
│   ├── connectome/{dataset.py,flywire.py,synthetic.py,toy.py,schema.py,analyze.py,motifs.py}
│   ├── generate/{replication.py,duplication.py,modular.py,spatial.py,motif.py,population.py,hybrid.py,random.py}
│   ├── backend/{base.py,cpu.py,torch_backend.py,genn.py}
│   ├── cognition/{working_memory.py,prediction.py,valuation.py,attention.py,association.py,goal.py,latent.py}
│   ├── decoder/{linear.py,semantic.py}
│   ├── experiments/{runner.py,registry.py,controls.py,tasks.py,metrics.py}
│   ├── observe/{logging.py,traces.py,telemetry.py,snapshots.py,stability.py,report.py}
│   └── ui/                   # Phase 7 only; directory may exist with README until then
├── tests/{unit,integration,scientific,performance}
├── benchmarks/{cpu,gpu,procedural,memory,scaling}
├── scripts/{download_data.py,preprocess_connectome.py,extract_motifs.py,benchmark.py,run_experiment.py,inspect_hardware.py}
└── artifacts/                # git-ignored experiment outputs
```

## 5. Performance budgets (binding numbers — recheck each phase)

Usable VRAM ≈ 5 GB; usable RAM ≈ 54 GB. Per-resident-neuron LIF state ≈ 32–64 B; per-resident-synapse ≈ 12–16 B. Fly explicit baseline ≈ 0.8 GB (fits anywhere). 1M virtual neurons at 5% active ≈ 3 MB state + event buffers (fits easily). Design ceiling for Phase 4: 1M spikes/sec → 10⁹ synaptic events/sec must stay in kernels, never cross the Python boundary per-event (§29). Python-side budget: orchestrate in batches ≥10⁴ events; pool all hot objects (`NeuronPool/EventPool/ActivationPool/TracePool/SynapticScratchPool`). Any benchmark that moves per-event work into Python and calls it done fails review.

## 6. Work breakdown

### PH0 — Bootstrap (days 1–3). Goal: runnable skeleton + diagnostics.
- **PH0-WI01 Repo + packaging.** Steps: `pyproject.toml` (name `vnr`, stdlib-first deps, extras `[gpu] [data]`), MIT LICENSE, README with quickstart, `.gitignore` (artifacts/, data/, checkpoints), `git init`, initial commit. Accept: `pip install -e .` succeeds on this machine. Deliverables: repo root files. Refs: §9–10.
- **PH0-WI02 CLI skeleton.** Steps: `click` group `vnr` with stubbed subcommands `doctor dataset connectome generate simulate benchmark experiment report`, each printing its contract (not stubs-that-lie: unimplemented ones exit 2 with "not implemented until PHx"). Accept: `vnr --help` lists all eight; `vnr doctor` runs (see WI04). Refs: §10.
- **PH0-WI03 Config system.** Steps: YAML load/validate with schema version, `config_hash` (sha256 of canonical YAML) embedded in every run record. Accept: roundtrip test + hash stability test. Refs: §81–82.
- **PH0-WI04 Hardware discovery + budgets.** Steps: collect CPU/RAM/GPU/VRAM/CUDA/driver/torch/compute-cap/disk (§11) into `HardwareProfile`; derive `RuntimeBudget` with D2.8 margins; `vnr doctor` prints the §10 STATUS block. Accept: on this machine reports Quadro M5000 / 8192 MiB / driver 537.99 / torch missing→"optional"; never claims more VRAM than `nvidia-smi` reports minus reserve. Refs: §10–11, §31.

### PH1 — Deterministic core (weeks 1–2). Goal: toy runtime with roundtrip guarantees.
- **PH1-WI01 Virtual IDs.** `hash64(blake2b-8)` over `(root_seed, module_id, lineage_id, generation, local_index)`; property test: 10M IDs/sec target, zero collisions on 1M samples, seed-change avalanches. Refs: §20.
- **PH1-WI02 LIF neuron.** Exact §33 dynamics, integer-tick schedule, deterministic params; unit tests vs closed-form decay. Refs: §33.
- **PH1-WI03 Event engine.** `NeuralEvent` dataclass per §24 (+causal_parent, trace ids), priority queue on int ticks, batch-drain API for future kernels. Accept: 100K-event ordering test. Refs: §24.
- **PH1-WI04 Active frontier.** Six-state machine DORMANT→…→EVICTED (§25) with hysteresis thresholds (no instant create/delete churn); counters for residency. Refs: §25.
- **PH1-WI05 Materialization/eviction.** `materialize_neuron(id)` builds transient state from structural rules + persistent store; eviction persists long-term layer first (§26). Accept: **§67 roundtrip test passes exactly** (static net). Refs: §21, §26–28.
- **PH1-WI06 Procedural connectivity.** `generate_outgoing_events(source, ctx)` pipeline per §22; seeded per D2.3. Accept: **§68 determinism test** (100 calls identical; seed change → controlled divergence). Refs: §22–23.

### PH2 — Validation suite (week 3). Goal: the oracle everything later is measured against.
- **PH2-WI01–WI05** implement §69 (virtualization-vs-explicit, exact for static), §70 (1M virtual / 50K budget runs without materializing all), §71 (synapse virtualization, statistical equivalence), §72 `StructuralFidelityScore` (per-metric distances, never prematurely collapsed), §73 graph-health gate (exploding/dead/pathological-synchrony/hub/isolation → fail with warnings). Refs: §66–73.
- **PH2 gate:** all five tests green on CPU. Nothing proceeds to FlyWire until this gate passes (per §100).

### PH3 — Backends (weeks 3–5). Goal: GPU acceleration without losing the oracle.
- **PH3-WI01 NumPy vectorized CPU backend.** Batched LIF updates over resident arrays; must reproduce oracle spike trains exactly. Benchmark B001–B003 (§60).
- **PH3-WI02 PyTorch CUDA backend.** Kernel-side event processing, Philox keyed RNG (D2.3), VRAM manager per §31 with live free/allocated/reserved queries. **First action: verify `torch.cuda.is_available()` on the Maxwell M5000; if CUDA kernels refuse sm_52, fall back to CUDA-11.x torch build or CPU+RAM scale-down, and log the decision.** Benchmarks B002/B004–B008. Refs: §29–32.
- **PH3-WI03 GeNN adapter (WSL2, optional).** Only after WI02 passes; never blocking. Refs: §17 in agent workflow.

### PH4 — Connectome + generators (weeks 5–8). Goal: biological seed and scaled variants.
- **PH4-WI01 Dataset abstraction + toy/synthetic/random** (§12) with `ToyDataset` mandatory for tests.
- **PH4-WI02 FlyWire ingestion.** `vnr dataset auth`, chunked resumable download to `G:\BRAIN\VNR\data\`, normalized schema (§13, preserve raw + record every filter), `vnr connectome stats` → `connectome_report.json/html` (§14). Accept: neuron/synapse counts within published tolerance of 139,255 / ~50M (record dataset version; resolve the 50M-vs-54.5M discrepancy in the manifest).
- **PH4-WI03 Motif DNA.** `MotifCatalog` with frequencies, clustering, region/type signatures (§15); must answer "instantiate 100K of M001 preserving type compatibility."
- **PH4-WI04 All eight generators** (§16) with full metadata (§16 block), duplication-divergence params (§17), modular expansion preserving densities (§18), and `CognitiveRole` abstraction — never fake human regions (§19). Each validated by §72 fidelity scores + §73 health gate.

### PH5 — Scaling science (weeks 8–14). Goal: the knee graph.
- **PH5-WI01 Four-way experiment §37** (explicit-dense / sparse / procedural / procedural+virtualized), same seed/stimulus, reporting behavior + spikes + memory + runtime + RTF (§59).
- **PH5-WI02 Scaling E004** (1×–100×) with all six controls (§38–39), recording the full §39 metric set.
- **PH5-WI03 Plasticity + cache**: STDP → reward-modulated → neuromodulated (§49); hybrid procedural/materialized strategy (§50); promotion/demotion lifecycle (§51); `SynapseCache` (§52). Switch replay tests to statistical tolerance here (D2.7).
- **PH5-WI04 Memory hierarchy + GC + checkpointing** (tiers §27, policies §28, pools §29, `NeuralGarbageCollector` §53 with hit/miss telemetry, seed-reconstructable checkpoints §54).
- **PH5-WI05 Observability**: telemetry, causal traces with sampling (§56–57), `NetworkStabilityMonitor` with pause-snapshot-diagnose (§74), auto reports (§83) that separate hypothesis/observation/interpretation (§5).
- **PH5 gate:** PLC/PSC-vs-fidelity knee (§89–90) located. **Pivot rule:** no ≥10× synaptic compression at ≥95% fidelity by end of PH5 → reframe thesis around §91 alternatives, log it, continue.

### PH6 — Cognition + decoder (weeks 14–20, only after PH5 gate).
Tasks A–E (§40) → roles (§41) → neuromodulators (§42) → `LatentNeuralState` (§43) → decoder stages 1–6 (§44–45) → emergence/multi-agent only as designed experiments (§46–47). LLM remains observer (§108–109). Population expand/collapse experiment (§77) and adaptive resolution (§75–76) live here.

### PH7 — Observatory UI (after first scaling graph, never before).
Local web UI (React+TS+Three.js+WebSocket or Streamlit interim), population heatmaps + detail-on-demand (§79–80, §105). Dark, minimal, Apple-grade; progressive disclosure.

## 7. Interface contracts (frozen at implementation time, versioned after)

`materialize_neuron(id)->NeuronRuntimeState` · `generate_outgoing_events(src,ctx)->list[NeuralEvent]` · `ConnectomeDataset` protocol (§12) · experiment YAML schema (§81) · run record (`run.json/metrics.parquet/events.parquet/state_summary.parquet/config.yaml`, §55) · report bundle (§83). Breaking a contract requires a minor version bump + migration note.

## 8. Testing, determinism, reproducibility policy

Every subsystem: interface → smallest test → reference impl → benchmark → optimize (§96); reference backend kept forever as oracle. Seeds: global + per-call keyed RNG (D2.3); every run records RunID/ExperimentID/commit/dataset hash/config hash/seed/hardware/software (§55). Tolerances: exact for static, statistical for plastic (D2.7). CI: `pytest` unit+integration on every commit; scientific/performance suites nightly with seed rotation.

## 9. Risk register

R1 Maxwell sm_52 vs modern torch CUDA builds → verify PH3-WI02 first action; fallback CUDA-11.x torch or CPU scale-down. R2 Python event-loop wall → batched kernel boundary (§5 budgets) + review gate. R3 FlyWire auth/volume → WI `dataset auth` + resumable download + toy-first ordering. R4 Eviction destroying learning → three-layer separation (D2.6) + promotion cache (§51–52) + statistical replay tests. R5 Scope creep into cognition/UI early → phase gates are hard gates; UI last (§106). R6 Single-machine failure → checkpoints (§54) + small commits (§98) + branch strategy (§99).

## 11. Language, voice, embodiment & autonomy (PH8 — strictly post-PH6)

Source: ChatGPT voice/interface proposal, reviewed 2026-09-14 (see notes entry 03). Core verdict:
the proposal's boundaries are correct and strengthen the project; its ElevenLabs architecture
claim verified (Speech Engine layers voice onto an existing LLM/server stack untouched —
elevenlabs.io/speech-engine), its prosody claim corrected (API exposes Stability / Similarity /
Style-Exaggeration / Speaker-Boost presets plus emotion-from-text, NOT continuous rate/pitch
control — design the mapping accordingly). Everything in PH8 is gated behind the PH5 knee gate
(real VNR + real thought events first); the vertical slice is the proof, not the starting point.

Locked decisions:
- **D2.11 Brain decides WHEN, language decides HOW.** The neural runtime's ThoughtEventManager
  alone may originate communication (novelty / prediction-error / memory-reactivation /
  goal-conflict / association-strength crossing threshold with hysteresis). The LLM never
  declares significance; it renders what it is given.
- **D2.12 LLM context quarantine.** VNR owns memory, emotion, goals, internal state. The LLM
  receives ONLY retrieved VNR memories (each carrying its memory ID) and current state values.
  Every factual claim in LLM output must cite a supplied memory ID — enforced by automated
  citation-audit tests using held-out facts the LLM knows but VNR never stored. Without this,
  every voice demo is suspect (LLM world knowledge masquerading as VNR memory). Risk R7.
- **D2.13 No firehose.** The LLM receives compressed cognitive states (~100–1,000 dims:
  population-rate vector + neuromodulator vector + memory-retrieval summary + top-k concept
  activations), never raw spikes/events. Rate-capped upstream of the LLM.
- **D2.14 Recurrence is a confound until proven otherwise.** Unidirectional fidelity
  (human-judged faithfulness + citation audit) must pass BEFORE bidirectional injection (WI07)
  or linguistic recurrence (WI08); all A/B claims then require recurrence-off re-runs. Risk R8.

Work items (each: objective → acceptance → deliverable):
- **PH8-WI01 Semantic state compressor.** Objective: deterministic VNR→vector encoder per D2.13
  with schema version. Accept: compress/decompress roundtrip preserves task-relevant rankings;
  output dimension configurable 100–1,000. Deliverable: `src/vnr/voice/compressor.py` + tests.
- **PH8-WI02 ThoughtEventManager.** Thresholds + hysteresis + cooldown + semantic dedup
  (near-identical consecutive states collapse; no "I… I… I…" stutter). Accept: step-function
  stimulus yields exactly one event per crossing episode; sustained plateau yields ≤1/minute
  at default cooldown. Deliverable: `src/vnr/voice/thought_events.py` + tests.
- **PH8-WI03 Unidirectional observer pipeline.** VNR → decoder → LLM renderer → text, with the
  prompt contract "respond ONLY from supplied state; invent nothing." Accept: faithfulness eval
  (human rubric, n≥50 states) + citation audit green. Deliverable: pipeline + eval harness.
  **Gate: WI03 fidelity must pass before WI07/WI08.**
- **PH8-WI04 Quarantine tests.** Held-out-fact probes (LLM must refuse or flag when asked about
  unretrieved objects), memory-ID citation checker in CI. Accept: 100% refusal on held-out set.
- **PH8-WI05 TTS layer (ElevenLabs primary).** Voice profiles as discrete presets mapped from
  (valence, arousal, uncertainty) regions to {voice_id, stability, similarity, style-exaggeration}
  + emotionally-faithful text rendering (emotion rides in the TEXT, per API reality). Latency:
  p95 thought→speech < 4 s via TTS streaming; cost meter per utterance + monthly cap config.
  Contingency (noted, not built): local TTS fallback. Deliverable: `src/vnr/voice/tts.py`.
- **PH8-WI06 Toggle matrix as experiment flags.** `OBSERVER / PARTICIPANT / RECURRENCE /
  AUTONOMOUS_SPEECH` in experiment YAML (extends §81 schema) + A/B experiment set (VNR alone,
  +decoder, +LLM observer, +feedback, +recurrence, +environment, +language). Every flag
  combination is a preregistered condition (§82). Deliverable: config schema + runner support.
- **PH8-WI07 Bidirectional injection API.** Semantic input → clamping of sensory/goal populations
  with source tags (injected vs sensed distinguishable in traces); STT path reuses it. Accept:
  injection produces the same downstream cascade as equivalent sensory drive (± tolerance).
  Gated on WI03.
- **PH8-WI08 Linguistic recurrence + self-questioning.** `QUESTION_EVENT` loop
  (state→question→retrieval→answer→new state); auditory/symbolic feedback channel. Accept:
  loop terminates (max-depth + convergence criteria); recurrence-off baselines re-run for all
  prior A/B claims. Gated on WI03.
- **PH8-WI09 Browser micro-world.** Minimal contract: grid world, object set (colors/shapes/
  reward tags), vision = downsampled occupancy → sensory populations, discrete actions
  {approach, avoid, inspect, wait}, reward channel. Closed sensorimotor loop from day one.
  Deliverable: `src/vnr/env/` + UI pane.
- **PH8-WI10 Curiosity.** Expected-information-gain threshold over uncertainty reduction →
  exploratory goals drawn from the existing action space. Accept: novel-object approach rate
  exceeds random baseline with all else equal.
- **PH8-WI11 Private-language ladder.** Two agents, channel ladder (4-bit → 8/16-bit → vector →
  sequences), message cost, emergence criteria (stability, reuse, compositionality probes).
  Extends spec §§46–47. Accept: food/danger/A/B/location communicated above chance with
  symbols stable across sessions.
- **PH8-WI12 LLM-as-teacher protocol.** Instruction/question/feedback via environment + reward
  only; direct weight influence forbidden (extends D2.12). Accept: learning curves match
  environment-only reward baselines in mechanism (plasticity traces), differing only in rate.
- **PH8-WI13 Vertical slice demo (the proof).** Real VNR event → real latent state → real
  THOUGHT_EVENT → LLM sentence → ElevenLabs speech, with frame-synchronized visualization of
  every stage + provenance chain display (neural→state→decoder→LLM→TTS). Accept: end-to-end
  runs unprompted from a novel stimulus; provenance panel shows true causal chain; p95 < 4 s.
- **PH8-WI14 Autonomous-mode session protocol.** Rate caps, kill switch, full provenance
  logging, human-present requirement for voice-on sessions. Accept: 30-min unattended-logged
  run with zero uncapped bursts.
- **PH8-WI15 Four-pane observatory extension.** Brain activity / neural state / internal
  representation / spoken output + audio waveform + source chain (extends PH7-WI01).

Added risks: **R7 LLM leakage** (mitigated by D2.12 + WI04); **R8 recurrence confound**
(mitigated by D2.14 + WI08 baselines); **R9 API cost/latency** (mitigated by WI05 meter +
streaming + caps); **R10 prosody granularity** (mitigated: discrete profiles + text-carried
emotion, verified against current API).

Note: `00-research-brief.md` v1.0 is intentionally left frozen as the founding snapshot; the
voice layer is recorded here and in the notes log. Regenerate the brief as v1.1 only on
owner request.

## 12. Done criteria

Per-phase gates above; V1 = §112 checklist (all 20 capabilities) where "GPU-accelerated" means PH3-WI02 benchmarks and "learning tasks" means §40 A–E at ≥2 scales with controls. Anything deferred is logged in `notes-and-observations.md` with reason and re-entry condition — never silently dropped. PH8 (voice/embodiment) is post-V1 research track: V1 completeness is judged without it, and the vertical slice (PH8-WI13) is the demonstration that the platform earned its voice layer.
