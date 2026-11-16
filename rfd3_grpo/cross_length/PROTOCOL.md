# Predeclared length-transfer check

Declared 2026-11-13 07:31 UTC, before any samples at these lengths.

Evaluate the six saved length-32 RFD3 adapters without retraining at unconditional lengths 24 and 40. For each length use eight new shared noise seeds, 4,000,000 through 4,000,007. Policy list: stock, full-window zero, late-window zero, main_s0, main_s1, main_s2, random_s0, single_s0, late_s0. Pair each adapted policy with the zero adapter using the same exploration window. Compare guarded reward, Rg, C-alpha bond MAE, clash fraction and the declared geometry heuristic. Use the same fixed checkpoint, 64 schedule points, two recycles and source code as the 32-residue study. No adapter updates or hyperparameter choice based on this data.

Run lengths sequentially on local MPS, length24 then length40, only if each can start before 2026-11-13 10:26 UTC. The experiment deadline stops new samples. If a model size, memory or other failure occurs, preserve the partial run and explain it. These are two input lengths, not evidence across diverse protein design conditions.
