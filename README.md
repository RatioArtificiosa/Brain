# VNR — Virtual Neural Runtime

Experimental platform for simulating biologically inspired neural architectures whose
**logical scale exceeds physically resident state**, via procedural connectivity, sparse
event-driven execution, dynamic materialization/eviction, and hierarchical memory.
Seed architecture: the adult *Drosophila* connectome (FlyWire).

Start with the research brief (`G:\BRAIN\VNR\00-research-brief.md`), then the master plan
(`G:\BRAIN\VNR\01-master-plan.md`). Live progress lives in `02-checklist.md`, the lab log
in `notes-and-observations.md`.

## 5-minute quickstart (Windows)

```powershell
pip install -e "G:\BRAIN\VNR\vnr"
vnr doctor
```

## Hardware

| Resource | This machine (2026-09-14) | Budget rule |
|---|---|---|
| RAM | 64 GB | 15% reserve → ~54 GB usable |
| GPU | NVIDIA Quadro M5000, 8 GB | 20% reserve → ~5 GB usable |
| Driver / CUDA | 537.99 / 12.2 runtime | — |
| OS / Python | Windows 11 / 3.13 | torch GPU support TBD at PH3 (sm_52) |

## CLI

```text
vnr doctor | dataset | connectome | generate | simulate | benchmark | experiment | report
```

`vnr doctor` is the first command that must work — and it does.
