# Notes & Observations — running lab log for VNR

**Protocol:** append a dated entry per work session/task. Each entry: what was done, measurements,
what worked, what failed and why, suggestions, deferred items with re-entry conditions.
This log is the project's memory — write it so a different agent can take over cold.

---

## 2026-09-14 · Entry 01 — Project founding + verification + Phase 0 kickoff

**Did:** Read the full ChatGPT export (5,858 lines) in `G:\BRAIN`. Independently verified the three
load-bearing claims via literature search: Knight & Nowotny 2021 procedural connectivity
(4.13M neurons / 24.2B synapses, single GPU) — confirmed; FlyWire adult fly (~139K neurons,
50M+ synapses) — confirmed; Shiu et al. 2024 whole-brain fly LIF — confirmed (134 cites).
Wrote `00-research-brief.md`, `01-master-plan.md`, `02-checklist.md` (this log's companions).

**Hardware probed on this machine:** Quadro M5000, 8192 MiB, driver 537.99 (CUDA 12.2 runtime),
64 GB RAM (per owner), Python 3.13.2, torch NOT installed. Compute capability 5.2 (Maxwell).

**Key decisions locked (see plan §2, D2.1–D2.10):** LIF-first; PyTorch-before-GeNN (GeNN→WSL2);
Philox counter-based keyed RNG for GPU determinism; int ticks @0.1ms; uint64 lineage IDs;
3-layer state separation; statistical (not exact) replay tolerance once plasticity is on;
20%/15% memory reserves; 1–5% active-density budgeting; decomposed scale reporting.

**Watch items:** (1) sm_52 vs modern torch CUDA builds — verify `torch.cuda.is_available()` at
PH3-WI02 before any GPU architecture is assumed; fallback is CUDA-11.x torch or CPU scale-down.
(2) Spec says 50M synapses in one place, 54.5M in another — resolve against dataset version
in the PH4-WI02 manifest. (3) Energy-bench "3.5 active neurons/step" is stimulus-dependent;
all budgets assume driven-task densities. (4) Missing from spec, added to plan: `vnr dataset auth`,
named Philox mechanism, UI-after-first-graph rule, PH5 pivot rule (≥10× compression @ ≥95%
fidelity or reframe), lesion/knockout motif controls (suggested for PH5-WI02).

**Suggestions for later:** Streamlit interim dashboard before the full React observatory (faster
feedback during PH5); nightly seed-rotation CI once scientific suites exist; preregister E004
hypotheses in the experiment registry before running (§82).

---

## 2026-09-14 · Entry 02 — PH0 executed: scaffold, doctor, tests, lint, commits

**Did:** Built `G:\BRAIN\VNR\vnr\` per plan §4 layout (core subset): `pyproject.toml` (click/pyyaml
runtime, torch/caveclient extras, `vnr` console script), MIT LICENSE (holder "VNR contributors" —
owner may replace with name), README quickstart, `.gitignore`, `src/vnr/{cli,config,hardware}`,
11 tests across `test_cli/test_config/test_hardware`. Installed editable (`pip install -e .`),
`pytest` 11 passed, `ruff` clean, 3 conventional commits (`8115fb1`, `b23779d`, `ce1375a`).

**Measurements:** `vnr doctor` on this machine: Intel64 (12 threads), RAM 63.9 GB, Quadro M5000,
VRAM 8.0 GB, driver 537.99, torch missing → CPU-only note, STATUS: READY. Budgets derived:
CPU ~54 GB / GPU ~6.4 GB usable after reserves. Unimplemented subcommands exit 2 with phase
pointer (verified by test).

**What worked:** stdlib-first approach (argparse-free click is fine; ctypes RAM fallback means
no psutil dependency); `config_hash` canonical-YAML scheme; honest-stub CLI pattern.

**What I did wrong (caught myself):** (1) Shipped `isinstance(exp, dict and True)` — a real
TypeError-at-runtime bug — caught on re-read before tests ran; fixed to explicit TypeError for
non-mapping roots. Lesson: re-read every validation branch before running. (2) Unused `noqa: S603`
(RUF100) — S-rules aren't in the default select; removed instead of broadening config.
(3) Test asserted "8192" for VRAM but `format_bytes` renders "8.0 GB" — fixed test to match the
contract, not the other way round (twice: contract was right both times).

**Suggestions:** keep the "re-read the diff before pytest" habit codified — it caught a live bug
lint couldn't. Next up: PH1-WI01 virtual IDs (blake2b-8 lineage hash).

---

## 2026-09-14 · Entry 03 — Voice/interface proposal: verification, reasoning, doc integration

**Input:** owner's ChatGPT thread part 2 (LLM-as-linguistic-interface, thought events, spontaneous
speech, inner-monologue modes, bidirectional language, VNR-owned memory/emotion, ElevenLabs voice,
state-dependent prosody, self-hearing loop, curiosity, browser micro-world, private language, A/B
matrix, 4-pane UI, autonomous mode, vertical slice).

**Verified via web search:** (1) ElevenLabs Speech Engine layers voice onto an existing LLM/server
stack with LLM+RAG untouched — CONFIRMED (elevenlabs.io/speech-engine). The proposed
VNR→decoder→Node→LLM→ElevenLabs chain is architecturally sound. (2) Prosody controls —
CORRECTED: the TTS API exposes Stability / Similarity Boost / Style Exaggeration / Speaker Boost
presets, and per docs "the underlying emotion comes from textual cues." There is NO continuous
rate/pitch/pause API. Consequence locked in plan: prosody mapping = 3–4 discrete voice profiles
from (valence, arousal, uncertainty) regions + emotionally-faithful rendered text (emotion rides
in the words) + per-utterance voice_id switching (supported). The proposal's continuous-mapping
table was the only technically wrong element; everything else survived.

**Reasoning — endorsed as designed:** brain-decides-WHEN / language-decides-HOW (D2.11); staged
toggles observer→participant→recurrence→autonomous as preregistered conditions; no-firehose rule
(D2.13); teacher-via-environment with no weight influence; inner-monologue OFF/OBSERVER/MONOLOGUE
as an experimental variable (language-into-the-loop becomes manipulable — genuinely good science);
vertical slice defined with real-everything requirement and placed after the PH5 gate.

**Reasoning — strengthened beyond the proposal:** (1) LLM leakage is the #1 threat to the whole
voice program (R7): LLM world knowledge will masquerade as VNR memory. Added D2.12 quarantine
(LLM sees only retrieved memories WITH their IDs) + WI04 held-out-fact refusal tests (100%
bar) + citation audit in CI. (2) Recurrence is a confound (R8): D2.14 requires unidirectional
fidelity proven first, then recurrence-off re-runs of all A/B claims. (3) Added latency budget
(p95 thought→speech < 4 s, streaming) and cost meter + caps (R9) with local-TTS noted as
contingency only. (4) STT path needs a named injection API with source tags (injected vs sensed
distinguishable in traces) — WI07. (5) Micro-world given a minimal contract (grid, object set,
downsampled-occupancy vision, 4 discrete actions, reward channel) so WI09 can't sprawl. (6) Added
autonomous session protocol (rate caps, kill switch, human-present rule) — WI14 — which the
proposal's autonomous-mode enthusiasm lacked.

**Doc changes:** plan gained §11 (PH8, WI01–WI15 + D2.11–D2.14 + R7–R10); checklist gained the
PH8 section; brief v1.0 deliberately left frozen (founding snapshot — regenerate as v1.1 only on
owner request). Also fixed my own editing slip: a duplicated standing-rules block briefly
inserted into the plan, removed immediately.

**Suggestion:** when WI03 arrives, run the faithfulness eval with an adversarial rubric (raters
try to elicit hallucinations), not just cooperative scoring — quarantine is only proven by
failed attacks.

---

## 2026-09-14 · Entry 04 — Watchdog built/tested/deployed; Rust verdict; PH1-WI01 done

**Watchdog (owner request: wake after 3 min paused, skip when working, never die trying).**
Built `vnr/scripts/watchdog.py` on a file protocol in `G:\BRAIN\VNR\.watch`: `status`
(ACTIVE/PAUSED, missing/garbage = PAUSED) + `heartbeat` (mtime = last work). ACTIVE → always
silent; PAUSED + fresh beat → silent; PAUSED + stale beat (>180 s) → exactly one WAKEUP line,
then silent until activity resumes (no spam, re-arms on fresh beat); missing files → still
alarms. Stdout carries ONLY alarm lines (a persistent monitor forwards each to me); all
diagnostics go to `watchdog.log`; `watchdog.health` proves liveness. Per-iteration try/except:
the loop survives anything short of kill. Honest limit stated to owner: a local script cannot
invoke the agent by itself — the monitor is the wake path; without it the watchdog only logs.

**Tests (17 pass, ruff clean):** 5 unit (active-skip, fresh-skip, stale-alarm, missing-files,
garbage-status) + 1 integration driving the real subprocess through the full episode lifecycle
(silence → silence → 1 alarm → no spam → resume → re-alarm → fault-path alarm). Two of my test
expectations were wrong, not the code: (1) a "fresh" beat ages past the threshold if untouched
— fixed by refreshing it during the silence window; (2) deleting files mid-episode correctly
does NOT re-alarm (same episode) — fixed by resuming first, then faulting. Both corrections
confirm the episode semantics are right.

**Deployed:** `.watch/` ACTIVE + fresh beat; watchdog running under a persistent monitor
(alarm = 15 s checks, 180 s pause threshold). Protocol going forward: I touch heartbeat with
every work step and set PAUSED only when genuinely yielding to the owner.

**Rust verdict (owner: "in Rust if you can + rust-analyzer checking").** Toolchain present and
current (cargo/rustc/rust-analyzer 1.97.1). Decision locked as plan D2.15: NO rewrite — Phase 0
oracle is validated Python and spec §96 keeps the reference forever; Rust enters at PH3 as the
native hot-loop candidate via PyO3, gated by rust-analyzer + clippy + benchmarks (must beat
torch on events/sec). Checklist PH3-WI02 amended. Revisit only with profiling data.

**PH1-WI01 done:** `core/ids.py` (blake2b-8 lineage hash, mod-2⁶⁴ wrapping) + 7 tests green:
determinism, uint64 range, negative/huge inputs, full seed avalanche, per-field sensitivity,
zero collisions on 1M, bulk==single agreement. Measured throughput 629K/s vs the aspirational
10M/s: REVISED with rationale — IDs generate at graph-build/materialization time, never in the
event hot loop (which only compares IDs), so 629K/s builds 1M neurons in ~1.6 s and is
sufficient; recorded here instead of silently lowering the bar. Commit `c58467a`.

**Next:** PH1-WI02 LIF neuron.

---

## 2026-09-14 · Entry 05 — Watchdog failure admitted, real wake-up built + proven

**Failure (owner caught it, correctly):** the v1 watchdog never woke anyone. Root cause, no
excuses: status latched on ACTIVE and v1 only alarmed on PAUSED — between turns nothing ever
sets PAUSED, so the monitor stayed silent forever. A watchdog that can't fire is decoration.

**Fixes:** (1) `watchdog.py` v2 adds the STALLED alarm: ACTIVE + heartbeat stale > 900 s
(env `VNR_WATCH_STALLED_AFTER_SEC`) fires once per episode — covers stuck workers and turns
that end without yielding. 25 tests green (new: active-fresh silent, stalled alarm, threshold
boundary), ruff clean, committed `4d5bd92`. The v1 monitor was killed and redeployed on the
new code. (2) Searched `.grok`: no watchdog/cron plugin exists — but the harness HAS a native
cron: `scheduler_create`/`/loop` ("sessionCrons: scheduled wakeup", min 60 s, 7-day expiry;
confirmed in `docs/user-guide/04-slash-commands.md`, `10-hooks.md`, `20-background-tasks.md`).

**Real wake-up, proven end to end:** created durable 5-minute task "VNR watchdog tick" —
touches heartbeat, reads status/checklist/notes, works the next item if actionable, ends
quietly if genuinely blocked on the owner. Proof: a throwaway 60-second probe task fired on
schedule and a worker wrote `scheduler-probe` at 08:32:35 with zero intervention; probe task
then deleted, probe file cleaned. Schedulers auto-expire after 7 days — calendar note: renew
or re-evaluate on 2026-09-21. Cancel anytime: scheduler ID `01a0a053-da49-7933-a845-8c46b3283a5f`.

**Two wake layers now:** file-level (monitor on watchdog stdout: WAKEUP after 3-min pause,
STALLED after 15-min idle-ACTIVE) + harness-level (5-min tick resumes checklist work). Either
suffices alone; together they cover monitor death and scheduler expiry respectively.

**Next:** PH1-WI02 LIF neuron.

---

## 2026-09-14 · Entry 06 — GitHub connected, showcase README live, push discipline wired

**Did:** Owner's repo `RatioArtificiosa/Brain` exists and is PUBLIC (verified, not assumed).
Connected local `vnr/` repo: remote `origin` added, branch `main`, pushed (verified `pushedAt`
on GitHub). Set repo description + 10 topics (computational-neuroscience, spiking-neural-
networks, connectomics, procedural-generation, flywire, python, pytorch, rust, gpu,
event-driven).

**README rewritten as showcase:** the one question, 30-second idea diagram, science-backbone
table (Knight & Nowotny / FlyWire / Shiu, all pre-verified), hardware-honesty table, phase
roadmap with live statuses, quickstart with real `vnr doctor` output, layout, agent-workflow
rules, lab rules. No fake badges (no CI exists — nothing claimed). Committed + pushed.

**Backup/showcase discipline:** the 5-minute scheduled tick prompt now ends every work item
with conventional commit + `git push origin main`, so GitHub is a live backup and a truthful
progress showcase. First push done this session.

**Wake-layer confidence (owner asked "absolutely sure?"):** stated precisely — (a) the monitor
fires on watchdog stdout the moment a WAKEUP/STALLED line prints, regardless of other
background tasks (proven: probe fired while monitor + tests ran); (b) the durable scheduler
fires every 5 min even across sessions and is independent of in-flight tasks; (c) boundaries:
monitor dies with the session, scheduler expires after 7 days (renew 2026-09-21). "Absolutely"
covers the mechanisms, not the machine being off.

**Next:** PH1-WI02 LIF neuron.

---

## 2026-09-14 · Entry 07 — PH1-WI02 done: exact integer-tick LIF oracle

**Supervisor audit (same day):** independently re-ran suite (34 passed) + ruff (clean);
commit `e5c8be8` verified real. Two defects found in the tick's work and fixed here:
(1) the tick did NOT push — `origin/main..HEAD` still contained `e5c8be8`; pushed manually
and hardened the tick prompt with a pre-exit check (`git log origin/main..HEAD` must be
empty). (2) Notes numbering collided (two "Entry 06" + chronological disorder from the
tick appending blindly); renumbered tick entry to 07 and restored order 01–07.
Checklist box was correctly checked with measurements. Lesson codified: ticks must read
headings before appending, and pushing is verified, not assumed.

## 2026-09-14 · Entry 08 — Audit part 2: docs had no backup; fixed with docs branch

**Gap found during the same audit:** only `vnr/` code was versioned — the plan, checklist, and
this log (the project's memory) existed solely as local files. Fixed: `G:\BRAIN\VNR\` is now
its own repo (`vnr/`, `.watch/`, `data/` ignored — no nested-repo mess), docs committed and
pushed to `origin/docs` on the same GitHub repo (branches: `main` = code, `docs` = docs).
Tick prompt extended: every work item ends with code commit + docs commit + push of BOTH
branches, verified via `git log origin/<branch>..HEAD` emptiness before ending the turn.

## 2026-09-14 · Entry 09 — WI03 adopted from a tick, verified, design-reviewed, shipped

**What happened:** a tick implemented PH1-WI03 (`core/events.py` + `test_events.py`) but ended
with the files uncommitted, unchecked, unlogged. I adopted the work instead of redoing it.

**Verification (mine):** 8 tests green; 100K-event ordering+FIFO passes (push 1.21 s, drain
0.17 s — heapq does ~80K pushes/s, ample for the reference oracle); ruff clean. One false
alarm of my own: two parallel reads displayed the files' contents swapped and I briefly
concluded the tick had crossed the filenames — re-checked via an independent channel
(`Get-Content` heads) and confirmed files are correct. Lesson: parallel read results can
mislead; verify anomalies through a second channel before acting.

**Design review:** (1) The tick used explicit `target_id` where spec §24 sketches
`target_population` — ENDORSED as correct sequencing (explicit delivery first; population
addressing arrives with the population model at PH6). Logged so PH6 doesn't inherit an
assumption silently. (2) `drain_all` via `sorted`+`clear` is O(n log n) — fine for the
oracle; backends will own performance. (3) Validation is thorough (bool exclusion, uint64
bounds, finite weights, tick monotonicity by construction). Committed, checked off above,
pushed to `main`; docs pushed to `docs`.

**Next:** PH1-WI04 active frontier.

**Did:** `vnr/src/vnr/core/neuron.py` (`LIFParams` frozen + validated, `LIFState`, `LIFNeuron.step/run/reset`)
implements spec §33 exactly: `dV/dt = (-V + I_syn)/tau` as the per-tick closed form
`V' = I + (V - I)·exp(-dt/tau)`, spike at `V >= threshold` with reset + integer-tick refractory
hold (`refractory_ms/dt` rounded, D2.4 — no float scheduling). `tests/test_neuron.py`: 9 tests vs
the `closed_form_voltage` oracle (decay, steady state, single-step analytic, spike+refractory clamp,
subthreshold silence, predicted spike tick, replay determinism, param validation).

**What I got wrong (caught by tests, not by reading):** two test drives assumed continuous-current
response — a single tick from rest rises only `I·(1-decay)` ≈ 0.005·I, so `step(0, 100.0)` correctly
did NOT spike and sparse `[0,1.5,0,3,0.2]` pulses never accumulated. Fixed the tests (drive 1000.0;
sustained 60-on/60-off blocks), not the model — the exact exponential integrator is per §33.
Lesson: for this integrator, think in steady states (`V∞ = I`), not pulse amplitudes.
Also fixed 2 ruff hits (`__all__` sort, `int(round())` → `round()`).

**Measurements:** full suite 34 passed in ~22 s (dominated by the 1M-ID collision test), ruff
check + format clean, committed `e5c8be8`. Defaults: tau 20 ms, threshold 1.0, reset/init 0.0,
dt 0.1 ms, refractory 2.0 ms (20 ticks).

**Design note for PH3:** this module is the frozen oracle — NumPy/CUDA backends must reproduce its
spike trains exactly on static nets. `I_syn` is in mV (asymptotic voltage), not nA; document the
unit convention at the backend boundary when synaptic weights arrive (PH1-WI03+).

**Next:** PH1-WI03 event engine.

---
