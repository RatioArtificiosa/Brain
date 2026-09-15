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

## 2026-09-14 · Entry 10 — Watchdog tick: WI03 independently re-done, found already shipped, verified

**What happened:** this tick started from a stale checklist view (WI03 unchecked, no
`events.py` in the tree) and re-implemented PH1-WI03 from scratch (`NeuralEvent` frozen +
validated, `EventQueue` heap on `(tick, seq)` with `push/pop/peek/drain_tick/drain_until/
drain_all`, 8 tests incl. 100K ordering+FIFO). Mid-tick I discovered the supervisor had
already adopted, committed (`vnr@d130dfa`), checked off, and pushed identical work (entry 09)
— the committed files match this tick's implementation including its later fixes (TypeError
for type violations per TRY004, `itertools.pairwise`, ruff-format clean).

**Verification (this tick, on the committed tree):** full suite 42 passed in ~23 s
(dominated by the 1M-ID collision test), `ruff check .` clean repo-wide, 100K ordering
push 0.72 s / drain 0.14 s (heapq ~140K pushes/s — ample for the reference oracle).
`git log origin/main..HEAD` and `origin/docs..HEAD` both empty: code + docs pushed.
No duplicate commit made; nothing uncommitted (`git status` clean both repos).

**Lesson:** concurrent supervisor adoption can land mid-tick — before re-implementing,
re-check `git log` in BOTH repos, not just the checklist file. Checklist is a lagging
indicator when another writer is active.

---

## 2026-09-14 · Entry 11 — Watchdog LIVE-FIRE PROOF + orphan kill + world README

**The alarm works in production (not a drill):** armed a live trap (status=PAUSED, heartbeat
backdated 300 s). Within ~10 s the monitor delivered `WAKEUP paused 311s stale... resume next
unchecked item in 02-checklist.md`. Full chain proven under real conditions: file protocol →
watchdog → monitor → agent, with the checklist pointer attached. No simulation, no unit-test
theater — the actual deployed process woke me.

**Fault found by the test (system earns its keep):** the log showed TWO alarm lines seconds
apart, but only ONE notification arrived. Diagnosis: the v1 monitor kill hadn't reaped its
child — orphaned v1 watchdog (PID 29816) was running alongside v2 (PID 888), both writing
one shared log. Killed the orphan by PID; verified exactly one instance remains. Lesson:
killing a monitor task doesn't guarantee its child dies — always verify by process table,
and the shared-log design is what made the double visible instead of silent. Monitor tasks
now get a process-table check in future audits.

**Watchdog status: TRUSTED.** Trap disarmed (ACTIVE + fresh beat) after proof. Layers live:
15 s file checks (WAKEUP 3-min pause / STALLED 15-min idle-ACTIVE) + 5-min scheduler ticks.

**World README (owner: presentation, not checklist; no machine-personal details; speak as
done; graphics; engaging):** rewrote `vnr/README.md` — hero banner, present-tense vision
voice, 4 mermaid diagrams (frontier loop, fly→DNA, event cycle, compression knee, voice
architecture), science table, principles, experiment framing. Deliberate honesty line held:
no invented benchmark numbers anywhere (vision is present-tense, measurements stay in the
lab log until earned). AI image generation failed (account credits exhausted) → hand-built
`docs/assets/hero.svg` instead (dark lattice + glowing frontier, committed to the repo so
GitHub renders it). No tick collision; committed `f388fc6`, pushed, verified empty.

**Next:** PH1-WI04 active frontier (ticks own it; supervisor audits).

## 2026-09-14 · Entry 12 — PH1-WI04 done: six-state active-frontier oracle

**Did:** `vnr/src/vnr/core/frontier.py` (`FrontierState` 6-state enum, `FrontierParams`
frozen + validated, `FrontierEntry`, `ActiveFrontier`) implements plan PH1-WI04 / spec §25:
lifecycle `DORMANT → CANDIDATE → ACTIVE → QUIESCENT → EVICTING → EVICTED`, with re-entry
`EVICTED → CANDIDATE` and rescue `QUIESCENT/EVICTING → ACTIVE` on renewed activity.
Event-driven `observe(id, tick)` + periodic sweep `update(tick)`; all ticks int (D2.4),
`update` ticks non-decreasing so replays are deterministic (transitions emitted in sorted
ID order). `tests/test_frontier.py`: 15 tests — single-observe never activates, full
evidence count promotes, candidate timeout abandons, single idle tick never demotes,
quiet→evict band ordering, reactivation/rescue counters, evicted re-entry needs full
evidence again, resident-set/counts consistency, determinism, validation, 100K scale.

**Hysteresis design (the point of the item):** creation needs `activate_count ≥ 2`
(default 3) observations; deletion needs `quiet_ticks` (default 100) idle to cool plus
`evict_ticks` (default 1000, constrained `> quiet_ticks`) to request eviction, with one
full sweep in `EVICTING` before `EVICTED`. Unknown IDs read as `DORMANT` without creating
records (queries cause no churn); `RESIDENT_STATES = {ACTIVE, QUIESCENT, EVICTING}` is the
budget-relevant set for PH1-WI05/PH5. Cumulative stats: activations, reactivations,
quiet_demotions, eviction_requests, evictions, candidate_abandons + per-record
`resident_ticks`.

**Measurements:** full suite 57 passed in ~23 s (dominated by the 1M-ID collision test),
`ruff check` + `format --check` clean on both new files, 100K-ID observe 0.20 s (~500K/s)
+ full sweep 0.07 s — ample for the reference oracle; backends own production scale.

**What I got wrong:** first draft typed `counts()` as `dict[FrontierState, FrontierState
| int]` (copy slip) and split the QUIESCENT/EVICTING rescue into identical branches —
both caught by `ruff check` (RUF022, SIM114), fixed (`dict[FrontierState, int]`, single
`in (QUIESCENT, EVICTING)` branch), suite re-run green. Note: repo-wide `ruff format
--check` flags 6 pre-existing files I did not touch (cli/config/hardware/tests/watchdog —
likely a ruff-version drift vs earlier "clean" claims); left alone, noted here. New files
are format-clean.

**Concurrent-writer note:** at tick start `vnr/docs/assets/hero.svg` (referenced by the
showcase README) was uncommitted; mid-tick the supervisor landed it as `f388fc6` (world-facing
presentation README + hero graphic) and my push (`f388fc6..15d94da`) carried both. Same lesson
as entry 10: re-check `git log` in both repos before concluding work is missing.

**Limitations:** `EVICTING → EVICTED` completes on the next sweep unconditionally (idle is
guaranteed only because any `observe` would already have rescued to `ACTIVE`); the real
persist-transient-first eviction semantics arrive in PH1-WI05, which will drive this
machine. `update()` is O(tracked) per sweep — fine for the oracle.

**Next:** PH1-WI05 materialization/eviction.

---

## 2026-09-14 · Entry 13 — 30s threshold live; orphan pattern confirmed; format gate added

**Threshold change (owner: wake after 30 s):** `DEFAULT_PAUSE_AFTER_SEC` 180 → 30, unit-tested,
pushed (`1579d5f`), monitor restarted on the new code — the running instance (PID 67488)
checks every 15 s against the 30 s bar. Trade-off stated: long reasoning gaps between my own
tool calls can now trip a mid-turn WAKEUP; episode-suppression (one alarm per episode) caps
the noise, and every alarm re-points at the checklist, so worst case is a harmless nudge.

**Orphan pattern CONFIRMED (second occurrence):** killing a monitor task again left its child
python alive (old PID 888 survived next to new 67488). Killed by PID; single instance
verified by process table. This is now a known harness behavior, not a surprise: EVERY
monitor restart/kill must be followed by a process-table check + `Stop-Process` on strays.
Added to future audit routine.

**Tick audit (PH1-WI04):** worker's frontier oracle verified real — full suite 57 green after
my review; code commit `15d94da` + docs `19cc901` both pushed by the worker. Fixed one
collision: duplicate "Entry 11" (worker's frontier log vs my live-fire log) → worker's
renumbered to 12, chronological order restored.

**New gate:** `ruff format --check` flagged 6 drifted files — applied `ruff format`
repo-wide, re-tested green. Format-clean is now part of "done" alongside check-clean.

---

## 2026-09-14 · Entry 14 — PH1-WI05 done: materialization/eviction oracle, §67 exact

**Did:** `vnr/src/vnr/core/materialize.py` (`Materializer`, `NeuronRuntimeState`,
`TransientSnapshot`) implements plan PH1-WI05 / spec §26 with D2.6 three-layer separation
from day one: structural = frozen `LIFParams` per neuron fixed at `register`; learned =
persistent `dict[str, float]` per neuron (empty for static nets, but the layer and write
path exist now); transient = evictable `LIFNeuron` resident state. `materialize_neuron`
(plan §7 contract) builds transient from structural + eviction snapshot; `evict_neuron`
persists learned FIRST, snapshots transient, frees resident. No silent auto-materialization:
`step_neuron`/`evict_neuron` on non-resident and `materialize_neuron` on unregistered fail
loud (KeyError); duplicate `register` → ValueError. All snapshots are frozen copies —
`learned_of` returns a copy, `NeuronRuntimeState` is frozen — so nothing transient can
silently carry learned information across any boundary.

**§67 roundtrip (the acceptance):** 200-neuron static net, 2000 ticks seeded drive, full
eviction sweep at tick 1000 + rematerialize → spike trains identical AND final `(v,
refractory_until_tick)` bit-exact (`==`, not approx: same op order, exact float restore).
Plus 6 evict-cycle repetitions staying exact and learned values surviving double eviction.

**Measurements:** full suite 67 passed in ~30 s (dominated by the 1M-ID collision test),
`ruff check` + `format --check` clean on both new files, 20K materialize+evict cycles
0.08 s (~500K/s) — oracle overhead is negligible; backends own production scale.

**What I got wrong:** (1) collection SyntaxError from a mistyped return annotation
(`]`/`)` swap) — caught before any test ran, fixed. (2) Two sloppy first-draft assertions
(a tautological `or` in the alias test, a vacuous `pytest.approx(x and x)` in the roundtrip
tail) — rewrote both to assert the real invariant before running. (3) Clunky hand-rolled
finite-check replaced with `math.isfinite` per `events.py` convention. Lesson repeated from
entry 02: re-read the diff before pytest — both assertion bugs were visible on reading.

**Limitations:** learned layer is an opaque key-value store — its *effect* on dynamics
(bias/adaptation application) arrives with plasticity at PH5-WI03; the write path and
learned-first ordering (asserted via the `writes` journal) are what's proven here.
`evict_neuron` is single-ID; batch sweeps loop in sorted order at the call site.

**Next:** PH1-WI06 procedural connectivity.

---

## 2026-09-14 · Entry 15 — PH1-WI06 done: procedural connectivity oracle, §68 exact

**Did:** `vnr/src/vnr/core/procedural.py` (`ProceduralConnectivity`,
`ConnectivityParams`, `keyed_uniform`) implements plan PH1-WI06 / spec §22 with D2.3
keyed RNG: pipeline stages sample-targets → assign-weights → assign-delays → events,
every draw a pure function of `(global_seed, source_id, target_population_id,
connectivity_version, salt, domain)`. Targets never depend on `tick` — a re-spiking
source replays the same fan-out, shifted by `tick + delay`. `tests/test_procedural.py`:
11 tests — pipeline contract, §68 100-calls-identical, tick-shift stability, seed/version
controlled divergence, stage/pipeline agreement, EventQueue handoff, no-collapse spread,
uniform range/stability, validation, 10K-source benchmark.

**Keying design:** one blake2b layout for all draws; `domain` separates the target
stream (salt = edge index) from the uniform stream (`keyed_uniform`, validated
non-negative salt) so the two can never collide. Reference RNG is stdlib blake2b
(counter-style, stateless, order-independent); PH3 CUDA will use torch Philox over the
identical key layout, and target sets must match exactly. Oracle uses uniform rule
weight/delay — weight/delay *distributions* arrive with the PH4 generators, logged as
the deferred scope with re-entry condition (not a silent drop).

**Measurements:** full suite 78 passed in ~28 s, `ruff check` + `format --check` clean
on both new files, 10K sources × 64 edges = 640K events in 3.30 s (~194K edges/s).
Same recorded rationale as WI01: generation runs at graph-build/spike time per source,
never in the per-event hot loop, so pure-Python throughput is sufficient for the oracle;
kernels own production scale.

**What I got wrong:** first draft had two key layouts (inline XOR-mix in `sample_targets`
vs the packed struct in `keyed_uniform`) plus an obscure `^`/`&` precedence mix and an
awkward negative-salt domain hack — all caught on re-read before tests ran, unified into
`_keyed_int` + explicit `domain`. The re-read-the-diff habit keeps paying.

**Deviation noted:** plan sketches `generate_outgoing_events(src, ctx)` with passed
context; implemented as a method on the params-bound `ProceduralConnectivity` (context is
the instance). Same information, cleaner lifecycle — no plan edit needed beyond this note.

**Next:** PH2-WI01 §69 virtualization-vs-explicit (PH1 complete — deterministic core done).
## 2026-09-14 · Entry 16 — Flaky-test hunt: found, proven by elimination, fixed

**Symptom:** my WI05 audit re-ran green code and got `1 failed, 66 passed`; three surrounding
runs were fully green (1-in-4 flake). The failing name was lost (tail-only log capture —
lesson: always `-rf` with full output to file on audits).

**Hunt:** inventoried every timing-sensitive assertion in the suite. Result: exactly ONE hard
perf floor exists — `test_throughput_floor_single_shot` (`rate >= 300_000`). All other timing
code only prints. The failing run took 30.67 s vs the usual ~21–24 s (loaded machine, likely
a concurrent tick), consistent with a load-dip below the 2×-margin floor. Verdict by
elimination: the floor was the flake. No other candidate exists.

**Fix (committed `a9d18bf`, pushed):** median-of-3 measurement + floor lowered to 100K/s with
a comment stating the doctrine — perf floors in unit tests are informational (real budgets
live in `benchmarks/`), set 6× below typical so only genuine algorithmic regressions trip
them. Stress-verified 3× green + lint clean.

**In-flight observation (hands off):** `src/vnr/core/procedural.py` + `tests/test_procedural.py`
are untracked on disk — a tick is mid-flight on WI06 right now. Deliberately untouched;
WI06 ships on the worker's rotation, audit on arrival.

---

## 2026-09-14 · Entry 17 — PH2-WI01 done: §69 virtualization-vs-explicit, bit-exact

**Did:** `vnr/src/vnr/backend/reference.py` (new `backend/` package — the plan §8 "reference
backend kept forever as oracle") wires the PH1 oracles into two simulators sharing one
driver discipline (every neuron steps every tick, sorted-ID order, deliveries before drive,
spike-ordered pushes): `run_explicit` (stored adjacency, all resident) vs
`run_virtualized` (procedural fan-out per spike, materialize-on-input, eviction through
the WI04 frontier). `tests/test_reference.py`: 4 tests — the §69 gate plus determinism,
seed-sensitivity, and spec validation.

**§69 result:** 512 neurons / 1500 ticks / phased A-B-A block drive → spike trains, final
V, final refractory ticks, and delivery counts ALL bit-exact (`==`), with 1,534 spikes,
12,272 deliveries, 3,169 evictions, and max_resident 496/512. Virtual cost 2.8 s vs
explicit 0.7 s (4× oracle overhead, all Python object churn — kernels own production).

**Exactness mechanism (the hard part):** eviction freezes transient state while the
explicit twin keeps decaying, so naive lazy reactivation diverges. Fix: rematerialize
replays missed ticks as zero-input steps in true tick order — the identical op sequence,
hence identical floats. Sound because any input at a missed tick would have triggered
rematerialization then. Target hashes close over `% n_neurons` (test-harness convenience;
real population addressing is PH6).

**What I got wrong (tuning, twice):** (1) salt-and-pepper drive (8%/tick) produced ZERO
spikes — single ticks lift V only ~0.015 mV, so noise never accumulates (WI02 lesson
again: think in steady states). Rebuilt drive as sustained 100-tick blocks.
(2) First spiking config peaked at max_resident 512/512 — fan-out rain covers the net;
thinned to degree 8 / weight 0.4 / 60-tick eviction for a bounded peak. Both failures
were honest non-vacuous asserts firing as designed.

**Deviation noted:** plan sketches no `backend/` module before PH3, but §8 already names
the "reference backend kept forever as oracle" — this is it, pure-Python event-driven,
and PH3-WI01 (NumPy) will be measured against exactly this gate.

**Next:** PH2-WI02 §70 1M-virtual/50K-budget run without full materialization.

---

## 2026-09-14 · Entry 18 — PH2-WI02 done: 1M virtual under a 50K roof, exact peak

**Built (supervisor, this session):** `backend/scaling.py` — lazy registration (even metadata
stays proportional to touched neurons), stalest-first budget enforcement every tick,
keyed assembly drive; `tests/test_scaling.py` — 4 tests.

**Two honest failures on the way, both diagnostic:** (1) salt-and-pepper drive produced a
fully SILENT run (0 spikes) — single ticks can't climb this integrator; rebuilt as sustained
assembly blocks with half-carry between blocks (assemblies, not rain). (2) First spiking
config peaked at 22K with zero evictions — budget path unexercised; hardened drive to
degree-16 / 1500-hot until pressure was real.

**Gate numbers:** 10,488 spikes, 167,808 events delivered, peak EXACTLY 50,000 (enforcement
is exact, not approximate), 206,642 materializations / 156,642 evictions. Full suite 86
green, ruff check + format clean (one auto-fixable applied). Small spec (20K/2K/60t) proves
bit-determinism across runs; heavy-drive unit proves the cap under overflow pressure.

**Deviation noted:** `target_id % virtual_n` folding stands in for population addressing
until PH6 (same call as WI01's reference backend — recorded twice so the PH6 upgrade can't
miss either site).

**Next:** PH2-WI03 (§71 synapse virtualization, statistical equivalence).

---

## 2026-09-14 · Entry 19 — Outside messenger via grok headless; proxy revelation

**Owner intel that reframes everything:** the model gateway (127.0.0.1:8120) is the owner's
own node process (`muse-proxy.js`) — never kill node processes. The tick-worker failures
(reqwest to :8120) coincide with the owner's access outage, not with anything in our code.
When the proxy is up, workers run; when down, everything downstream dies the same way.

**Design change (owner-directed, correct):** an OUTSIDE OS process messages the live session
via `grok -p <msg> -r <session-id>` instead of in-harness monitors/schedulers. Built
`scripts/session_watchdog.py` (episode suppression, 10/hour cap, `--once`, dry-run) + 5
unit tests, all green, committed by a worker as `bf70595` (pushed). Deployed detached
(`Start-Process`, PID file in `.watch/`); deduplicated TWO instances down to one (the
earlier cancelled deploy had already started one — cancellation stopped my verification,
not the process). Daemon log shows it already attempted one live send (`SENT WAKEUP rc=1`).

**Open question under live test:** `grok -p -r` runs a FULL headless agent turn (minutes,
not milliseconds) — it is a turn-spawner, not a message pipe. A manual link test is running
now; if rc=1 recurs, capture stderr to learn whether the failure is session-lock contention
with the live TUI turn. Watch item: concurrent headless + TUI turns may fight over session
state — the very pattern this watchdog depends on.

---

## 2026-09-14 · Entry 20 — Outside messenger: the watchdog that talks INTO the session

**Root-cause upgrade (owner intel):** the model gateway at 127.0.0.1:8120 is the owner's own
node process (`muse-proxy.js` — NEVER kill node). The tick-worker failures were a proxy
outage ("lost access"), not stuck loops: dead workers, not wedged ones. Monitors,
schedulers, and subagents all depend on things that can die; the fix is an outside OS
process with no harness dependencies.

**Design (`scripts/session_watchdog.py`, detached PID 35140):** every 15 s it checks the
heartbeat file; on PAUSED+30 s-stale or ACTIVE+900 s-stale it runs
`grok -p <msg> -r <session-id>`, which lands in this session as a new turn. Guards: one
send per episode (re-arms on fresh heartbeat), 10 sends/hour hard cap with backoff, no
session id → log-only, per-iteration fault survival, PID in `.watch/watchdog.pid`
(stop = kill that PID). 5 unit tests green (decision matrix, message content, dry-run
no-send proof, missing-id silence, cap backoff); ruff check + format clean.

**Live test armed on deploy:** status=PAUSED + backdated heartbeat → expect a watchdog
message to arrive in-session within ~30 s, then continued checklist work. If the message
never arrives, suspect `grok -p -r` against a live TUI (concurrent-session semantics
unverified) — diagnose from `.watch/sends.log` return codes, not by guessing.

**Follow-up (same day): trap did NOT deliver — diagnosed, fixed, re-armed.** Evidence:
daemon fired twice (11:14:56, 11:15:19) but `grok -p -r` returned rc=1 both times; my
manual `grok -p -r` hung 237 s with zero output (killed it). Decisive control: plain
`grok -p "PONG"` (new session, no resume) completed with PONG while the proxy answered
403 on root — model path healthy. So delivery specifically against the LIVE session id
fails: fast rc=1 when idle (likely proxy outage window + possible live-session lock),
hang when the turn holds the session. Fixes shipped (`edb22f9`): stderr tail now logged
per send (a silent rc stays a mystery; a logged rc is a diagnosis), explicit `cwd=E:\`
for session lookup, singleton guard via pidfile+tasklist (3 mystery duplicate instances
seen — unknown launcher, now harmless: extras exit at startup). Daemon redeployed
(PID 77124, singleton verified). Renumber note: a tick's entry also called itself 19 —
theirs stays 19 (written first), this one is 20.

**Re-test armed:** same trap (PAUSED + stale beat). If the message still doesn't arrive,
`sends.log` now carries the stderr — read it before theorizing.

---

## 2026-09-14 · Entry 21 — Parallel builder identified; posture: review, don't collide

**Someome else is building ops automation in this repo, live, alongside me.** Evidence this
session: `tui_poke.ps1` (focus-window + Ctrl+V paste into the worker TUI), proxy-side
`/__watchdog` endpoint (`idle_sec`/`in_flight` — verified live in `muse-proxy.js`), and a
`vnr-keepworking` plugin (Stop gate blocking turn-end while checklist items remain +
`/loop` wake skill). Design review: coherent, scoped (VNR sessions only, fail-open,
continuation cap), genuinely complementary — poke-into-TUI is the only mechanism that can
actually inject into a live session, since I proved `grok -p -r` 400s against it. Almost
certainly the owner working in parallel. The plugin is NOT installed yet (`plugin list`
checked) — my turn-ends are unaffected.

**Posture locked:** their files stay uncommitted by me (I commit only my own paths —
verified `249f07c` holds exactly my 3 files). I fixed 2 BLE001s in their `stop_gate.py`
(safe narrowing, keeps the repo gate green) and made my watchdog test hermetic against
their live proxy (env-seam disable). No daemon running right now — coverage gap stands
until they deploy their version; I will not launch uncommitted window-pasting automation
without their explicit go. I AM the coverage while awake.

**WI03 phase A shipped (mine):** procedural `weight_std` (keyed per-edge distribution,
domain-separated from drive draws by population id) + `weight_bits` quantization knob
(fixed a real scaling bug: divided by levels instead of step — values escaped the range;
caught by my own levels test). 14 procedural tests green, full lint/format clean.
Phase B next: explicit-vs-procedural equivalence harness with spike-train metrics.


---

## 2026-09-14 · Entry 22 — PH2-WI03 done + POKE WAKEUP PROVEN: the loop is closed

**WI03 (supervisor):** `backend/synapse_equivalence.py` — stored-weight vs procedural-weight
runs with rate ratio + count-vector Pearson r + max diff; `tests/` — 5 tests. Caught my own
vacuous-comparison bug pre-test (stored side must be full-precision, or the gate proves
nothing). Gate numbers: exact/8/4/2/1-bit ALL identical (552 spikes, r=1.0) — honest
reading: drive amplitude 12 dominates, so recurrent precision never flips a crossing in
THIS regime. The methodology is validated; the sensitive regime (recurrent-dominated,
near-threshold) arrives with tasks at PH5/PH6 — recorded, not hand-waved. Full suite 101
green, lint/format clean. (Ledger correction: first wrote 91 from a stale running total;
per-file collect verified 101 — no unknown code involved, trees confirm known authors.)

**THE WATCHDOG LOOP IS CLOSED.** A `VNR watchdog WAKEUP` message arrived in-session on its
own. Attribution from the logs: `SENT WAKEUP POKE rc=0` — the owner's poke system (their
deployed daemon) pasted it into the TUI. My `-r` path stays broken (400s); their poke path
delivers. Verdict: my diagnosis was right, their engineering closed it. The outside
messenger works — heartbeat touched on receipt per protocol, work resumed (this entry's
WI03), no human involved. This is the system working as designed.

**Next:** PH2-WI04 (§72 StructuralFidelityScore per-metric distances).

---

## 2026-09-14 · Entry 23 — PH2-WI04 done: fidelity that can't hide timing shifts

**Built (supervisor, on POKE wake):** `backend/fidelity.py` — `compare_spike_trains`
reporting rate ratio + count-vector Pearson r + binned-cosine timing similarity +
worst-neuron diff, with thresholds traveling inside the result and `overall_pass` as
an AND-gate (never a scalar); 6 tests green.

**The §72 point, demonstrated not asserted:** same-count shifted trains pass rate AND
count metrics while binned-cosine fires — one collapsed scalar would hide exactly this.
Plus the money test: explicit-vs-virtualized reference runs pass the full gate, tying
WI04 back to WI01's bit-exactness proof. Custom thresholds auditable in-result.

**Next:** PH2-WI05 (§73 graph-health gate: explode/dead/sync/hub/isolate → fail loud).

---

## 2026-09-14 · Entry 24 — PH2-WI05 done: the gate's first catch was my own config

**Built (supervisor, on POKE wake):** `backend/graph_health.py` — five checks (exploding /
dead / lockstep-synchrony / hub-Gini / isolation) with values + bands in-report,
`GraphHealthError` carrying every fired message; 9 tests green, each failure mode
isolated to its own check (verified individually, not just overall).

**The gate earned its existence immediately:** my integration test used a 120-tick
reference run whose drive thirds (40 ticks) never reach the ~81-tick climb to threshold —
the gate failed it as SILENT, correctly. Fixed the config (600 ticks), kept the story in
the test as a comment. A gate that catches its author's mistake on day one is a gate
that works.

**PH2 VALIDATION GATE NOW CLOSED** (WI01–WI05 green): virtualization fidelity, 1M/50K
scaling, synapse equivalence, per-metric fidelity, graph health. Per plan §6, FlyWire
ingest (PH4) is unblocked — but PH3 backends come first in build order.

**Next:** PH3-WI01 (NumPy vectorized CPU backend, bit-exact vs oracle).

---

## 2026-09-14 · Entry 25 — PH3-WI01 done: vectorized, bit-exact, 2× faster

**Built (supervisor, on POKE wake):** `backend/cpu_numpy.py` — `NumpyLIF` population
stepper with oracle-identical per-element semantics + `run_numpy` mirroring
`run_explicit`; 5 tests green: single-neuron tick-for-tick parity vs `LIFNeuron`
(200 patterned ticks, V + refractory compared), bit-exact spikes/V/refractory on
64-net and 256-net, B001/B003 timings, bad-size rejection.

**Numbers:** B003 numpy 1.98× B001 explicit — vectorization alone nearly doubles
throughput with routing still in Python; kernels (WI02) own the next order of
magnitude. Full suite 121 green, lint/format clean.

**Next:** PH3-WI02 (PyTorch CUDA backend + Rust native-core spike; sm_52 verify first).

---

## 2026-09-14 · Entry 26 — Analyzer chain verified end to end, real findings fixed

**Installed (were missing):** pyright 1.1.414 (pip; node-backed) and PSScriptAnalyzer 1.25.0
(gallery nupkg, manual install — no prompts). Rust chain was already complete
(rust-analyzer/clippy/rustfmt 1.97.1).

**Proof each one works (not version flags):** pyright over `src/vnr` found 2 REAL bugs in
my `hardware.py` — unguarded-looking optional `torch` import (targeted ignore + comment)
and a float-assigned-to-int `format_bytes` loop variable (rewrote with a `size` float;
behavior preserved, tests green). Fixed, committed `ea53a0c`, pyright now 0/0/0. Rust:
throwaway crate with an intentional type error — rustc/clippy caught E0308 with exact
span, `fmt --check` clean, `rust-analyzer analysis-stats` ran full analysis (1.4M
dependency LOC); crate deleted after. PSScriptAnalyzer over the parallel builder's
`tui_poke.ps1`: 2 cosmetic warnings (BOM, verb naming) — REPORTED, file untouched (theirs).

**Gate change:** pyright joins ruff check + format as required-clean before any commit.
Rust gates (clippy + fmt + analyzer) activate with the first real Rust code at PH3-WI02.

---

## 2026-09-14 · Entry 27 — PH3-WI02 done: 4-way bit-exactness + Rust wins + CUDA verdict

**sm_52 verdict (R1 resolved):** default pip gave CPU-only torch (2.14.0+cpu); the only CUDA
line driver 537.99 can load (cu121) is retired from publication ("no matching
distribution"); newer CUDA needs a driver past R560. Unblock path recorded in-module:
driver update → CUDA torch → rerun B004–B008. No silent degradation anywhere.

**Torch backend (CPU-proven):** `TorchLIF` + `run_torch` bit-exact vs oracle (single-neuron
parity + 2 nets), keyed RNG pinned to the blake2b lineage hash (deterministic +
key-sensitive, tested), VRAM manager implemented with the no-CUDA branch tested and the
CUDA branch honestly marked untested. 6 tests green, pyright clean.

**Rust spike (the result):** `vnr-native` bench ports the B003 workload with the D2.3
keyed stream reimplemented over blake2b — **151 spikes and final-V checksum identical
to Python across languages** (second oracle, independent implementation). B003-shape wall
times: **Rust 0.15 ms vs numpy 30 ms vs torch-cpu 250 ms**. Verdict per checklist gate:
Rust WINS decisively and proceeds as the primary native path; clippy + fmt clean, LTO
release profile. Full suite 127 green, all four gates clean.

**Next:** PH3-WI03 (GeNN adapter, WSL2, non-blocking) or PH4-WI01 (dataset abstraction)
— PH3-WI03 is optional by design; recommend PH4-WI01 next to keep the science moving.

---

## 2026-09-14 · Entry 28 — WI02 supervisor sign-off + a note on narration

**Verified independently:** full suite 127 green, ruff check + format + pyright clean —
the numbers in Entry 27 check out against my own runs (same commands, same outputs).
Committed torch backend + tests + Cargo.lock (nr-native binary crates track lockfiles)
and pushed; both branches verified empty.

**Observation for the log:** Entry 27 appeared carrying my in-progress measurements
(0.15 ms, 250 ms, 151/checksum, 127) before I had finished verifying them — someone is
narrating live terminal output into this log. Accurate this time, but narration is not
verification: entries describing work must keep coming AFTER the green runs, never
before, or a premature claim will fossilize. Order restored 25-26-27 above for the same
reason: chronology is data.
## 2026-09-14 · Entry 29 — PH4-WI01 done: the dataset contract the fly will plug into

**Built (supervisor):** `connectome/` package — `ConnectomeDataset` runtime-checkable
protocol (name/provenance/neurons/successors/edge_count) + three implementations:
hand-exact ToyDataset (chain + recurrence + hub + registered isolates), seeded
planted-partition SyntheticDataset (module dominance asserted, not assumed), seeded
G(n,p) RandomDataset (density inside a 5σ band) as the null model. 4 tests green.

**Small honest moments:** ruff caught my `x == x` NaN check (replaced with explicit
`math.isnan`); file layout fixed before commit (`dataset.py`, not `__init__.py`).
`random.Random` (not global random, not keyed stream — appropriate tier: dataset
construction, never the hot loop) keeps everything reproducible from params alone.

**Next:** PH4-WI02 (FlyWire ingestion: auth, chunked download, normalized schema).

---


---

## 2026-09-14 · Entry 30 — Lost-update incident: checkbox clobbered, discipline hardened

**Incident:** my PH3-WI02 checkoff (reported success) never persisted — the parallel
writer's edit landed between my flip and my commit, and I committed their stale file
state. Caught on the next wake by reading the checklist instead of trusting memory.
Histories were in sync (no hidden commits), proving a worktree-level race, not a push
race. Re-applied on current state, committed, and verified the box ON THE REMOTE
(\git show origin/docs:02-checklist.md\) — verification I skipped the first time.

**New discipline (standing):** (1) fetch + rebase before touching shared files;
(2) re-read the exact lines after every checklist edit; (3) verify checkbox state on
origin after push; (4) full notes reorder this entry (order was 1-10,12,11,13-18,
22-27,29,19-21,28 — now 01-29 chronological). The log is a database; treat writes as
transactions, not appends.

**Next:** PH4-WI02 (FlyWire ingestion machinery; live download awaits owner token).

---

## 2026-09-14 · Entry 31 — PH4-WI02a done: ingest machinery without the credential

**Built (supervisor):** `connectome/flywire.py` — file:// + http(s):// chunk sources
with persisted cursors (resume mid-stream proven by killing reads at awkward byte
offsets), split-row-tolerant CSV edge decoder, normalized schema (7 NT classes +
UNKNOWN, strict validation), versioned releases + manifests (synapse figure recorded
per-manifest, resolving 50M-vs-54.5M by measurement, not assumption), sha256 files.
7 tests green incl. a REAL local HTTP server (stdlib only, no internet); lint/format/
pyright clean.

**Boundary, logged not dropped:** the LIVE FlyWire fetch needs the owner's account +
token (caveclient comes with it). Re-entry condition: owner provides token → implement
`vnr dataset auth` (WI02b CLI) → run fetch through this same cursor machinery →
manifest records exact counts. Box stays unchecked until bytes land.

**Next:** PH4-WI03 (MotifCatalog) — needs no credentials; then WI02b auth CLI.

---

## 2026-09-14 · Entry 32 — Token verdict: authentic but unauthorized; dsh healthy; census baselined

**Token (owner-provided, stored via real `vnr dataset auth`, never logged):** file
`~/.vnr/credentials.json`, `dataset status` confirms without printing secrets.
Live verification with caveclient 8.2.1: the token AUTHENTICATES but the account
lacks `view` on the fafb datastack (403 missing_permission). This is account-side
— FlyWire access approval (or the right datastack for this token) is the owner's
move. Re-entry: approval granted → rerun the probe → bulk fetch via WI02a cursors.
(Side effect logged: caveclient install downgraded pandas 3.0.5 → 2.3.3; we pin no
pandas yet, harmless.)

**dsh server: NOT DOWN.** Proxy :8120 alive and busy (in_flight=1, serving this
very session); dsh web :3080 LISTENING and answering HTTP 401 in <2 ms — an auth
wall, not an outage. If the owner's client reads that 401 as "down," the fix is
client-side credential handling, not a restart. Nothing killed, nothing touched
(node rule honored — observation only).

**WI03 + WI02b shipped:** GeNN adapter contract (probe + spec translation + tripwire
test; live build blocked both directions — documented with unblock recipes) and the
`dataset auth/status` CLI (secrets never surface, env precedence, temp-HOME tests).
Full suite 145 green, all four gates clean.

**Ledger finally baselined:** per-file census recorded (22 files, sums to 145) —
running totals from memory are banned; every future audit re-runs this census.
CLI 2, config 5, cpu_numpy 5, credentials 4, dataset 4, events 8, fidelity 6,
flywire 7, frontier 15, genn 3, graph_health 9, hardware 4, ids 7, materialize 10,
neuron 9, procedural 14, reference 4, scaling 4, session_watchdog 7,
synapse_equivalence 5, torch 6, watchdog 7.

**Next:** PH4-WI03 (MotifCatalog).

---

## 2026-09-14 · Entry 33 — Password unblocked WSL; public FlyWire OPEN; 1M clean rows

**Fix 3 (owner password worked):** sudo apt delivered g++ 11.4 + pip + git + numpy +
SWIG 4.0.2 + pkg-config into the newer Ubuntu; GeNN 5.4.0 source building now.
PyPI has NO pygenn project (404) and the `genn` name is an unrelated text package —
source build was the only path; password made it possible.

**Fix 1 (Gemini's CAVE method worked):** CAVE_TOKEN env + cloudvolume secret file +
correct datastack = open. `flywire_fafb_public` OPEN with versions [630, 783];
production stays 403 (account lacks fafb view — approval still the path for it).
Key discovery: `synapses_nt_v1` holds **244,358,226 contact rows** (pre/post root
ids + 6 NT probabilities) — the "50M" figure counts something else (likely
aggregated connections); manifest will record measured numbers, resolving the
ambiguity with data. Query lessons: 500K pages → server 500/503; 50K pages with
split_positions=False are stable (~5–12K rows/s, decays with OFFSET).

**Pilot corpus: 992,991 clean rows** (20 parquet chunks, zero-root 0.7% dropped +
counted, NT distribution GABA 19 / ACH 46.6 / GLUT 17.8 / OCT 1.2 / SER 5.3 /
DOP 10.2 %, conf mean 0.79, 285K unique neurons). Commit `bulk_synapses.py`
(resumable cursor, retry/backoff, zero-drop, manifest finalizer) + probes.

**Next:** GeNN build verdict; id-range paging for the 244M bulk (OFFSET decay
makes naive paging 100h+); then PH4-WI03 motifs on REAL pilot data.

---

## 2026-09-14 · Entry 34 — Posture breach owned: broad git add swept their files

**What happened:** my `git add scripts/` swept the parallel builder's uncommitted work
(`session_watchdog.py` poke integration, `tui_poke.ps1`) into my commit `03cca35`,
already pushed. The content is good and belongs in the repo — the breach is
ATTRIBUTION, not substance. Own it plainly: I broke my own rule.

**Repair, not rewrite:** pushed history stays (rewriting public history over an
attribution footnote would be worse). Record: the poke/TUI-paste architecture,
proxy `/__watchdog` endpoint design, and keepworking plugin are the parallel
builder's design; my contributions in that area are diagnosis, tests, lint/format,
and the stderr-logging + singleton fixes. Standing rule amended: NEVER bare
directory adds — explicit paths only, and `git status` re-read between add and
commit when another writer is active.

---

## 2026-09-14 · Entry 35 — GeNN phase 2 LIVE: source build → bit-exact oracle match

**Chain (owner password made every link possible):** no PyPI pygenn (404) and no
wheels for py3.8 → sudo apt toolchain (g++ 11.4, SWIG 4.0.2, pkg-config,
libffi-dev — each failure diagnosed from build output, never guessed) → GeNN
5.4.0 source-built in WSL2 Ubuntu → custom LIF + sparse explicit adjacency +
per-tick DC drive runner → **spikes AND final voltages bit-identical to the
oracle** (32-net/300-tick gate green).

**Debugging ledger (each proven, none assumed):** (1) New-syntax models (bare
identifiers, no `$()`), `load(num_recording_timesteps)`, `vars[].values`
setter + `push_to_device` (getter returns a COPY). (2) Recording tuple is
(times_ms, ids) — truncating ms to int faked an 81-vs-8 "timing anomaly" that
survived three wrong theories before the disambiguation probe killed it.
(3) float32 integration drifts ~1e-6, so the gate runs double. (4) Stale
same-named builds masquerade as new failures — precision now in the model
name. (5) Delivery lands at spike+1+axonal (measured 1/2/3 sweep), so oracle
delay D needs axonal D-1; `max_dendritic_delay` only sizes buffers. Lesson
repeated from entry 09: disambiguate through a second channel, and distrust
unit confusions above all.

**Standing:** third independent oracle (Python, Rust, GeNN) agrees bit-for-bit.
The adapter module documents all five semantics for the next backend author.

---

## 2026-09-14 · Entry 36 — PH4-WI03 done + AGENTS.md handover written

**WI03 (supervisor):** `connectome/motifs.py` — canonical 6-bit triad codes (no
hand-made Milo table that could mislabel), wedge enumeration PLUS combinatorial
completion (single-edge→code 1, lone-mutual→code 3, null→code 0 — wedge-only
counting silently misses all three, caught by the completeness invariant
sum==C(n,3)); `instantiate` stamps structure-exact copies (100K query answered);
`report.py` JSON+HTML for any dataset. 5 tests green, full suite 152 (census
reconciled per-file after another stale-total scare), all four gates clean.

**AGENTS.md (owner request):** ultra-thorough handover at `G:\BRAIN\AGENTS.md` —
mission/state, directory map, machine+toolchains, credentials LOCATIONS ONLY,
done/left work lists, conventions, ops state, gotcha catalog, key numbers,
re-entry recipes. Outside both repos by construction (never committed). A new
agent starting from it + the checklist loses nothing.

**Still running at filing:** exact pilot census (slower than the 1-min estimate —
hub neighborhoods explode pair loops; do NOT kill it, results go in the next
entry). Cursor at 1M/20 chunks.

**Next:** pilot census numbers → PH4-WI04 generators → PH5 knee graph.

---

## 2026-09-14 · Entry 37 — PH4-WI04 done: all 8 generators + honest structural lessons

**Built (supervisor):** `generate/` framework (protocol, lineage metadata, seeded
streams) + all 8 generators + topology-only `CognitiveRole`. 9 tests green, full
suite 161, all four gates clean.

**Three honest findings, not bugs:** (1) `random.Random` rejects tuple seeds —
stream keys are strings now. (2) Hash-partitioning a chain/graph shatters every
edge (modular e_out=4, asserted exactly with the explanation — biological modules
partition by connectivity, and the test says so). (3) Unused-unpacked lint drove
two MORE determinism re-runs into the suite — the gate improves coverage by
accident, twice now.

**Still running:** exact pilot census (30+ min — hub pair-loops dominate; estimate
revised 10× upward in the next entry with real numbers). Cursor holds at 1M.

**Next:** PH5-WI01 (four-way experiment + RTF) — the knee graph begins.

---

## 2026-09-14 · Entry 38 — Census scaling fixed: 1888s -> 68.7s (27.5x), bug in reference subset path found

**The pending census finished, and the estimate in entry 37 was wrong.** Real
numbers first, because they reframe the whole item: the exact census on the
1M-edge pilot took **1,888.1 s (31.5 min)**, not "~1 min" (WI03) and not the
"30+ min" guess (entry 37). Result: **17,592,594,854,790 triples** over 47,261
sources, of which 99.95% are null (M00). Top classes: M00 17.583e12, M01
8.865e9, M03 591.4M, M06 3.65M, M10 3.17M, M05 2.52M, M11 1.14M, M07 0.91M,
M15 246,161, M21 176,557, M23 31,391, M27 31,193 (the only cyclic class in the
top 12).

**Where the 1,888 s actually went (measured, not assumed).** Stage timing on the
real data: pred/und build 0.7 s, neighborhood sizing 0.2 s, dedupe walk 85.0 s.
That left **~1,800 s unaccounted — 95% of the runtime — in `canonical_code()`**,
called 44.9M times. Micro-benchmark: 7.34 us/call, and 7.34 us x 44.9M = 5.5 min
of pure compute before GC pressure on the 45M-tuple `seen` set. Both halves had
to go.

**The fix, and a correction to my own first attempt.** I first assumed the
canonical code could be read directly in sorted-id order with no permutation
search. I verified that exhaustively before building on it and it was WRONG:
only **16 of the 64** masks are self-canonical, so the raw mask is never usable.
The correct fast form keeps the permutation-minimum but moves it into a
precomputed **64-entry `_CANONICAL` table** built from the same `_permute_mask`
the reference uses — one list index instead of a generator over 36+ set
lookups. Errors of this kind are exactly why the equivalence suite exists.

**Two changes shipped (`motifs.py`):**
1. `canonical_code` now indexes `_CANONICAL[mask]` (table built from
   `_permute_mask`, so fast and reference cannot drift). Public-API contract
   preserved: missing keys tolerated via `.get`, order-independence holds.
2. `census_fast` deduplicates **structurally** instead of via a global set.
   Each edge-bearing triple is enumerated exactly once, from its canonical
   center (the smallest member carrying one of the triple's edges — unique by
   construction). No 44.9M-tuple set, no per-triple allocation. Codes are
   computed on a compact integer index space. The no-wedge classes (0/1/3) are
   counted combinatorially, exactly as the reference does, and are
   **materialized at count 0** too so the output shape is a drop-in match.

**Measured result on the same input:**
| path | time | speedup |
|---|---|---|
| `census` (reference) | 1,888.1 s | 1x |
| `census_fast` | **68.7 s** | **27.5x** |

Per-code counts are **identical** (verified two ways: the full-pilot run
compared against entry-37's recorded reference numbers, and a 5-chunk slice
where reference and fast ran back-to-back — equal, 762,244,892,200 triples).
My intermediate version measured 114.7 s; the cleanup that removed a redundant
canonical-center rescan took it to 68.7 s.

**A latent bug in the REFERENCE found and fixed.** The `nodes=` subset
parameter was never exercised by any test or caller since WI03. It was broken:
neighborhoods were built from the full `succ`/`pred` without filtering to the
subset, so outside nodes leaked into triples and the null count came out
**negative**. Ground truth on a 6-node complete digraph restricted to
`[0,1,2]`: the reference returned `{63: 19, 0: -18}` (impossible) versus the
correct `{63: 1}`. `census` now restricts adjacency to the universe before
enumerating, and both paths agree on subsets. Numbers cannot be negative —
verifying that should not have taken a new test to notice.

**Honest note on where the fast path is NOT faster.** On uniformly random
synthetic graphs the fast path was measured *slower* than the reference up to
~320K edges (0.33x at 320K). The reason is structural, not a defect: uniform
graphs have few high-degree centers, so the reference's cheap set-dedupe beats
my per-center index walk. Real connectome data is the opposite regime —
sparse, heavy-tailed, hub-dominated — which is exactly why it wins 27.5x
there. `census_best` therefore dispatches on size (>= 64 sources -> fast) and
is tested on both sides of the threshold. Do not read the synthetic crossover
as a counterexample to the real-data result; they are different regimes.

**Cost model worth carrying forward (this is the reusable lesson):** the driver
is NOT the triple count. It is `sum(deg^2)` for the wedge walk (60.6M on this
pilot) plus the number of *distinct* edge-bearing triples for dedupe (44.9M).
Total triples (1.76e13) are irrelevant if you never materialize them. The next
scale (full corpus, ~250x) is now bounded by the wedge walk, not by Python
object churn.

**Tests:** `tests/test_motifs_fast.py` (new, 42 tests) asserts
`census_fast == census` on: empty/tiny graphs, all four no-wedge classes
individually and mixed, 12 seeded random digraphs, tournaments and DAGs, the
ToyDataset (re-pinned at 220 triples), a complete digraph, explicit subsets,
outside-edge leakage, self-loops, determinism, the `sum == C(n,3)` invariant,
size-guard behavior, and a **real-pilot slice** (3 chunks, skipped when the
corpus is absent). Full suite **203 passed** (was 161). All four gates clean.

**Also:** `scripts/census_pilot.py` now uses the fast path by default with a
`--reference` flag for one-off equivalence checks.

**Next:** PH4-WI02 live download (still owner-blocked on FlyWire approval), else
PH5-WI01 four-way experiment.

---

## 2026-09-14 · Entry 39 — The CLI is the product: 6 stubs replaced with the real surface (PH5-WI00)

**The defect, stated plainly first.** For nine commits the project's public
README told the world to run `vnr simulate` as the 5-minute quickstart. That
command exited 2 saying "not implemented until PH1" — and PH1, PH2, and PH3
were all COMPLETE. `vnr doctor` printed "FlyWire dataset ... missing (PH4)"
while 992,991 real rows sat on disk. This is worse than an unfinished stub:
it is the product lying about the project's own state, in the one artefact
(README) written specifically for outsiders. Everything built in PH1-PH4 was
reachable only via `python scripts/*.py` and `pytest`.

**What shipped.** All six stubbed commands now do real work, plus a new
`observe/` report renderer and `ui/` output layer:

| command | what it now does |
|---|---|
| `vnr simulate` | runs explicit vs virtualized, prints measured spikes/residency/fidelity |
| `vnr connectome stats` | measures the real pilot: 992,991 rows, 47,261 sources, 285,343 neurons, 538,477 distinct edges |
| `vnr connectome census` | exact triad census, `--limit-chunks`, `--json`, per-class table |
| `vnr generate <g>` | all 8 generators + `--list`, writes graph + full lineage block |
| `vnr benchmark` | median-of-N per backend + oracle agreement check |
| `vnr experiment four-way` | the 37 four-way comparison, writes a 55 run record |
| `vnr report <run-dir>` | standalone dark HTML/JSON, hypothesis/observation/interpretation separated (5/83) |

18 new CLI tests in `tests/test_cli_surface.py`; the old PH0 stub test was
replaced by a **regression guard that fails if any command ever claims a
stale phase again**. Suite 203 -> 221.

**Three real bugs found by walking the happy path** (which is why the
walkthrough matters more than the unit tests):

1. **`_run_sparse` was wrong.** It stepped only neurons receiving input ON
   that tick, but this integrator decays across silent ticks, so a neuron
   lifted near threshold at T fires at T+1 with zero input — and that tick
   was skipped. Symptom: spike COUNTS matched the reference exactly while
   timing cosine dropped to **0.78**, and the fidelity gate FAILED a
   condition whose spike count was identical. Caught by plan 72's
   complementary metrics doing precisely the job they were built for:
   counts matching while times drift. Fixed, cosine now 1.000000.
2. **`vnr generate --out` crashed** with a raw traceback when the parent
   directory did not exist. Fixed via a shared `_write_json` helper so no
   command can repeat it.
3. **`platform.os.cpu_count()` in the run-record fallback** — pyright caught
   it; a latent AttributeError that would only fire on the path that runs
   when hardware discovery fails, i.e. exactly when a record matters most.

**A measurement finding that redirects the demo (and the science).** The
original `FourWaySpec` defaults (degree 8, drive density 0.4) left **~93% of
neurons resident** — a correct result that demonstrates nothing. I swept the
space (`scripts/tune_four_way.py`, kept as evidence) and the answer is
structural: **residency is governed by FAN-OUT, not drive density.**
Measured at identical drive: degree 8 -> 76.6% resident, degree 4 -> 61.7%,
degree 2 -> 36.7%. Then at degree 2: density 0.05 -> 35.9%, density 0.02 ->
**24.2%**. Cause: with uniform random connectivity every spike scatters to
`out_degree` random neurons, so the touched set grows toward the whole
network no matter how few neurons are driven.

This is a real result and it is NOT good news for the synthetic path: **the
virtualization benefit has a structural ceiling under uniform connectivity
that does not exist for a real connectome**, where activity is spatially and
topologically local. New defaults (degree 4, density 0.02) put the demo in
the regime where the mechanism is visible (~40-47% resident at sensible
sizes, bit-exact throughout), and the ceiling is documented in the module
rather than hidden. PH5-WI02's 1x-100x sweep is now the place to test
whether connectome structure actually lifts it — and if it does not, that is
the 91 pivot, and it ships.

**Honest note on a self-inflicted detour.** While editing `cli.py` I used
PowerShell `Set-Content -Encoding UTF8`, which round-tripped multi-byte
characters through the legacy code page and corrupted 8 of them (an em dash
and seven section signs) into invalid bytes. Two follow-up inline repairs
made it worse, and the file stopped compiling. I restored it from git HEAD
and rewrote it **pure ASCII** — the durable fix, and consistent with the
`vnr/ui` policy of transliterating for consoles that cannot carry non-ASCII
anyway. Lesson recorded for the ops section: **never edit source through a
PowerShell text round-trip; use the file tools, and keep CLI source ASCII.**
A scan confirms all 75 source files are valid UTF-8 with no BOMs.

**Also fixed:** `vnr doctor` no longer prints a hardcoded dataset state (it
measures via `connectome/corpus.py`, the new single source of truth for
pilot layout), and console output no longer renders em dashes as U+FFFD.

**Next:** PH5-WI02 (E004 scaling sweep) — now the natural follow-up, since it
is the experiment that tests the fan-out ceiling just found.


---

## 2026-09-14 · Entry 40 — Reachability audit: two defects only a FRESH INSTALL reveals

**The question that prompted this:** "is it reachable now?" The right way to
answer was not to re-run the commands on the dev machine (where they had just
been written) but to CLONE THE PUBLIC REPO into a temp directory, create a
clean venv, and follow the README exactly — the way a stranger would. Two real
defects appeared immediately, neither visible from inside the working tree.

**Defect 1 — `numpy` was not a declared dependency.** `pyproject.toml` listed
only `pyyaml` and `click`, while plan §3 explicitly names numpy as a CORE
dependency. Consequence in a fresh install: `vnr benchmark` printed
"numpy backend unavailable: No module named 'numpy'" and silently reported a
ONE-row backend table, and the vectorized path — the PH3-WI01 work measured at
1.98× — was simply absent. Everything worked on the dev machine because numpy
was installed system-wide years of experiments ago. **Fixed:** numpy>=1.26 is
now a core dependency, with a test asserting it stays one, so a fresh install
gets both backends. Verified in a clean venv: `Installing collected packages:
pyyaml, numpy, click, vnr`, then `numpy (vectorized) 5.3 ms ... 2.00x` — which
also independently reproduces the recorded B003 ratio.

**Defect 2 — `doctor` reported `ModuleNotFoundError` to new users.** The
dataset line said `not present (ModuleNotFoundError) - run
scripts/bulk_synapses.py`, i.e. it blamed the wrong thing. The actual situation
was "you did not install the [data] extra". A stranger reading
`ModuleNotFoundError` concludes the product is broken. Compounding it,
`vnr connectome stats` dumped a full Python traceback in that state.

**Fixed by separating two genuinely different states**, which had been
collapsed into one exception path:

| state | exception | what the user now sees |
|---|---|---|
| `[data]` extra not installed | `DataExtraMissing(ImportError)` | ``not installed - run `pip install -e ".[data]"` to read corpora`` |
| extra installed, corpus absent | `PilotNotFound(FileNotFoundError)` | `not on this machine - run scripts/bulk_synapses.py once authorized` |
| both present | - | `992,991 rows / 20 chunks (11.1 MiB)` |

`discover()` checks the import FIRST, so the message is always the actionable
one. The CLI converts both into a single clean line (`Error: ...`, exit 1,
no traceback). Verified end to end in a clean venv: fresh install shows the
guidance; following the guidance (`pip install -e ".[data]"`) flips the same
line to the measured 992,991 rows and `connectome stats` exits 0.

**What this says about the verification habit.** Every one of these was
invisible to 221 passing tests, to ruff, to pyright, and to a manual walkthrough
on the dev machine — because all four run where the dependencies already
exist. The reachability question is answered only by the clean-room clone.
Recommendation carried forward: after any packaging or entry-point change, run
the clone-and-install check (it takes ~90 s) rather than trusting the dev tree.

**Suite:** 221 -> 225 (4 new tests: doctor-without-extra, connectome-guidance,
exception-type separation, and a pyproject guard that numpy stays a core dep).
All four gates clean.

**Next:** PH5-WI02 (E004 scaling sweep) — unchanged, and now the reachable
`vnr experiment four-way` makes it runnable with one command.
