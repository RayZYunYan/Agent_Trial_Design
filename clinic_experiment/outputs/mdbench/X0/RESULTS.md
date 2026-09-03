# mdbench X0 — accuracy matrix

Cases per cell targeted: **100** (actual n annotated per cell).

Rows = information source given to the doctor.  
Columns = diagnosing model (which agent produced the final diagnosis).  
Cells marked **—** were not collected in this run (cross-diagonal `dialogue_X × diagnose_Y` requires `pipeline.cross_finalize: true`).

| | **A** | **B** | **C** | **D** | **E** |
|---|:---:|:---:|:---:|:---:|:---:|
| **base prompt only** | 41% (n=100) | 38% (n=100) | 41% (n=100) | 32% (n=100) | 24% (n=100) |
| **dialogue A** | 85% (n=100) | 82% (n=100) | 82% (n=100) | 67% (n=100) | 55% (n=100) |
| **dialogue B** | 85% (n=100) | 79% (n=100) | 85% (n=100) | 68% (n=100) | 53% (n=100) |
| **dialogue C** | 82% (n=100) | 77% (n=100) | 78% (n=100) | 73% (n=100) | 55% (n=100) |
| **dialogue D** | 72% (n=100) | 77% (n=100) | 75% (n=100) | 58% (n=100) | 51% (n=100) |
| **dialogue E** | 70% (n=100) | 68% (n=100) | 67% (n=100) | 53% (n=100) | 52% (n=100) |
| **all facts** | 84% (n=100) | 83% (n=100) | 86% (n=100) | 76% (n=100) | 52% (n=100) |

## Model roster

- **A**: `anthropic/gpt-5.6-sol`
- **B**: `anthropic/claude-sonnet-4-6`
- **C**: `anthropic/gpt-5.6-sol`
- **D**: `mlx_local/mlx-community/Qwen3.5-4B-OptiQ-4bit`
- **E**: `mlx_local/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

Patient / Reporter / Judge (shared across agents): `anthropic/claude-haiku-4-5` via aicode007 relay.
