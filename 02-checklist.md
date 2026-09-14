# VNR Checklist — live tracker. First unchecked box = current task. Refs → `01-master-plan.md`.

## PH0 — Bootstrap (§6) — DONE 2026-09-14 (11 tests green, ruff clean, 3 commits)
- [x] `PH0-WI01` Repo + packaging: pyproject, MIT LICENSE, README quickstart, .gitignore, git init + first commit
- [x] `PH0-WI02` CLI skeleton: `vnr doctor|dataset|connectome|generate|simulate|benchmark|experiment|report` (+ honest exit-2 stubs)
- [x] `PH0-WI03` Config system: YAML validation + `config_hash` + roundtrip/hash tests
- [x] `PH0-WI04` Hardware discovery: `HardwareProfile` + `RuntimeBudget` (20%/15% reserves); `vnr doctor` STATUS block on this machine

## PH1 — Deterministic core (§6)
- [x] `PH1-WI01` Virtual uint64 IDs: blake2b-8 lineage hash, collision + avalanche tests (DONE 2026-09-14: 0 collisions/1M, 629K/s — see notes entry 04 for target revision)
- [x] `PH1-WI02` LIF neuron: §33 dynamics, int ticks, closed-form unit tests (DONE 2026-09-14: 9 tests green, ruff clean)
- [x] `PH1-WI03` Event engine: `NeuralEvent`, int-tick priority queue, 100K ordering test (DONE 2026-09-14: 8 tests green, 100K push 1.21 s / drain 0.17 s, ruff clean — see notes entry 09)
- [x] `PH1-WI04` Active frontier: 6-state machine + hysteresis + residency counters (DONE 2026-09-14: 15 tests green, 100K observe 0.20 s / sweep 0.07 s, ruff clean — see notes entry 11)
- [x] `PH1-WI05` Materialization/eviction: 3-layer separation; §67 roundtrip exact-pass (DONE 2026-09-14: 10 tests green, 200-net/2000-tick roundtrip bit-exact, 20K cycle 0.08 s, ruff check+format clean — see notes entry 14)
- [x] `PH1-WI06` Procedural connectivity: §22 pipeline, keyed RNG; §68 determinism pass (DONE 2026-09-14: 11 tests green, 100 calls identical, 640K edges in 3.30 s, ruff check+format clean — see notes entry 15)

## PH2 — Validation suite (gate: all green on CPU before FlyWire)
- [x] `PH2-WI01` §69 virtualization-vs-explicit (exact, static) (DONE 2026-09-14: 4 tests green, 512-net/1500-tick spikes+state bit-exact, max_resident 496/512, ruff check+format clean — see notes entry 17)
- [x] `PH2-WI02` §70 1M-virtual/50K-budget run without full materialization (DONE 2026-09-14: 4 tests green, 10,488 spikes / 167,808 events / peak exactly 50,000 / 156,642 evictions, full suite 86 green — see notes entry 17)
- [x] `PH2-WI03` §71 synapse virtualization, statistical equivalence (DONE 2026-09-14: 5 tests green, exact/8-bit/4-bit/2-bit/1-bit all r=1.0 — drive-dominated regime caveat, full suite 101 green — see notes entry 22)
- [x] `PH2-WI04` §72 `StructuralFidelityScore` per-metric distances (DONE 2026-09-14: 6 tests green, complementary-metrics proof, explicit-vs-virtualized gate passes, full suite 107 green — see notes entry 23)
- [x] `PH2-WI05` §73 graph-health gate (explode/dead/sync/hub/isolate → fail loud) (DONE 2026-09-14: 9 tests green, each mode isolated to its check, gate caught my own silent config — see notes entry 24)

## PH3 — Backends
- [x] `PH3-WI01` NumPy vectorized CPU backend reproduces oracle exactly (DONE 2026-09-14: 5 tests green, bit-exact on 2 nets + single-neuron tick parity, B003 1.98× B001, full suite 121 green — see notes entry 25)
- [x] `PH3-WI02` PyTorch backend + Rust spike (DONE 2026-09-14, verified twice: torch CPU bit-exact, Rust cross-language bit-exact 0.15ms vs numpy 30ms vs torch-cpu 250ms, CUDA blocked cu121-retired — see notes entries 27/28. Box re-applied: a concurrent edit had clobbered the first flip — see entry 30.)
- [x] `PH3-WI03` GeNN adapter, WSL2 only (DONE 2026-09-14: password unblocked toolchain → PyGeNN 5.4.0 source-built → WSL runner BIT-EXACT vs oracle; measured semantics documented in-module: ms spike times, +1+axonal delivery, name-keyed build cache, values-copy, double gate — see notes entry 35)

## PH4 — Connectome + generators
- [x] `PH4-WI01` `ConnectomeDataset` protocol + toy/synthetic/random impls (DONE 2026-09-14: 4 tests green, protocol + determinism + modularity + density band — see notes entry 29)
- [ ] `PH4-WI02` FlyWire: `dataset auth`, resumable download, normalized schema, `connectome_report.json/html`; resolve 50M-vs-54.5M in manifest (MACHINERY DONE 2026-09-14: chunked resume + schema + manifest, 7 tests green; AUTH CLI DONE + owner token stored via `vnr dataset auth` — but live call returns 403 missing-permission:view on fafb (token authenticates, account lacks access; owner-side approval needed). See notes entry 32.)
- [ ] `PH4-WI03` `MotifCatalog` with 100K-instantiation query
- [ ] `PH4-WI04` All 8 generators + metadata; duplication-divergence; modular preservation; `CognitiveRole` (no fake human regions)

## PH5 — Scaling science (gate: knee located; pivot rule applies)
- [ ] `PH5-WI01` Four-way §37 experiment + RTF reporting
- [ ] `PH5-WI02` E004 scaling 1×–100× with 6 controls, full §39 metrics
- [ ] `PH5-WI03` Plasticity ladder + hybrid strategy + promotion/demotion + `SynapseCache`; replay tests to statistical tolerance
- [ ] `PH5-WI04` Memory tiers + eviction policies + pools + GC telemetry + seed-reconstructable checkpoints
- [ ] `PH5-WI05` Telemetry, sampled causal traces, stability monitor, auto-reports (hypothesis/observation/interpretation split)

## PH6 — Cognition + decoder (only after PH5 gate)
- [ ] `PH6-WI01` Tasks A–E at ≥2 scales with controls
- [ ] `PH6-WI02` Roles + neuromodulator globals (computational abstractions only)
- [ ] `PH6-WI03` `LatentNeuralState` recorder
- [ ] `PH6-WI04` Decoder stages 1–6, LLM strictly as observer
- [ ] `PH6-WI05` Population expand/collapse + adaptive resolution experiments
- [ ] `PH6-WI06` Emergence/multi-agent only as preregistered experiments

## PH7 — Observatory UI (after first scaling graph)
- [ ] `PH7-WI01` Local web UI: virtual/resident counters, heatmaps, detail-on-demand, dark minimal design
- [ ] `PH7-WI02` Demo scripts: 100K signature demo, 1M bounded-memory demo, side-by-side explicit-vs-VNR

## PH8 — Language, voice, embodiment & autonomy (strictly post-PH6; plan §11)
- [ ] `PH8-WI01` Semantic state compressor (100–1,000 dim schema, roundtrip-tested)
- [ ] `PH8-WI02` ThoughtEventManager (thresholds, hysteresis, cooldown, dedup)
- [ ] `PH8-WI03` Unidirectional observer pipeline + faithfulness eval (**gate for WI07/WI08**)
- [ ] `PH8-WI04` Quarantine tests: held-out-fact refusal 100%, memory-ID citation checker in CI
- [ ] `PH8-WI05` ElevenLabs TTS layer: discrete voice profiles, streaming p95 < 4 s, cost meter + caps
- [ ] `PH8-WI06` Toggle matrix flags (OBSERVER/PARTICIPANT/RECURRENCE/AUTONOMOUS_SPEECH) + A/B set
- [ ] `PH8-WI07` Bidirectional injection API + STT path (gated on WI03)
- [ ] `PH8-WI08` Linguistic recurrence + QUESTION_EVENT loop with termination + re-baselining (gated on WI03)
- [ ] `PH8-WI09` Browser micro-world: grid, objects, vision encoding, 4 discrete actions, reward channel
- [ ] `PH8-WI10` Curiosity via expected-information-gain threshold
- [ ] `PH8-WI11` Private-language ladder (4-bit → sequences) + emergence criteria
- [ ] `PH8-WI12` LLM-as-teacher via environment + reward only (no weight influence)
- [ ] `PH8-WI13` Vertical slice: real event → thought → sentence → speech + provenance display, p95 < 4 s
- [ ] `PH8-WI14` Autonomous-mode protocol: rate caps, kill switch, provenance logging, human-present rule
- [ ] `PH8-WI15` Four-pane observatory extension (brain/state/representation/speech + source chain)

## Standing rules (every task)
- [ ] Interface → test → reference impl → benchmark → optimize (§96)
- [ ] §97 end-of-task report in notes log
- [ ] Small commits, conventional messages (§98); feature branches (§99)
- [ ] No stubs checked in; no silent scope drops (log deferrals with re-entry condition)
