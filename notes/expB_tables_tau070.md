
### cos>=0.7

| group / class | n (n with ≥12 segs) | PR rate | cycling texts (PR ≥ 0.30) | PR rate, first 2k words | PR per 1k tok (unbounded) | verbatim-repeat rate |
|---|---|---|---|---|---|---|
| base_nothink | ALL | 800 (264) | 0.69 | 0.92 | 0.69 | 2.90 | 0.14 |
| p50c_nothink | ALL | 800 (288) | 0.66 | 0.95 | 0.64 | 2.89 | 0.06 |
| base_think_preloop | ALL | 400 (369) | 0.90 | 0.99 | 0.89 | 5.57 | 0.16 |
| p50c_think | correct | 214 (197) | 0.85 | 0.98 | 0.86 | 6.30 | 0.12 |
| p50c_think | finished: wrong answer | 71 (71) | 0.92 | 1.00 | 0.94 | 7.26 | 0.26 |
| p50c_think | finished: no \boxed{} | 55 (54) | 0.86 | 1.00 | 0.84 | 6.28 | 0.41 |
| p50c_think | cut: degenerate whitespace | 20 (20) | 0.90 | 1.00 | 0.87 | 6.44 | 0.08 |
| p50c_think | cut: still reasoning | 24 (24) | 0.84 | 1.00 | 0.75 | 4.82 | 0.39 |
| p50c_think | cut: repetitive (paraphrase) | 9 (9) | 0.77 | 1.00 | 0.82 | 4.85 | 0.46 |
| p50c_think | cut: verbatim loop | 7 (7) | 0.49 | 0.86 | 0.72 | 5.48 | 0.63 |
| p50c_think | ALL | 400 (382) | 0.86 | 0.99 | 0.86 | 6.34 | 0.22 |
- still-reasoning minus base_nothink | ALL: ratio 1.21, diff 95% CI [0.089, 0.198]
- still-reasoning minus p50c_think | correct: ratio 0.98, diff 95% CI [-0.069, 0.031]
- still-reasoning minus p50c_nothink | ALL: ratio 1.27, diff 95% CI [0.123, 0.227]
- still-reasoning minus base_think_preloop | ALL: ratio 0.93, diff 95% CI [-0.107, -0.012]
- P3 p50c_think ALL vs base pre-loop: ratio 0.96, diff CI [-0.056, -0.014]

### cos>=0.7 strict12

| group / class | n (n with ≥12 segs) | PR rate | cycling texts (PR ≥ 0.30) | PR rate, first 2k words | PR per 1k tok (unbounded) | verbatim-repeat rate |
|---|---|---|---|---|---|---|
| base_nothink | ALL | 800 (264) | 0.69 | 0.91 | 0.69 | 2.89 | 0.14 |
| p50c_nothink | ALL | 800 (288) | 0.66 | 0.95 | 0.64 | 2.89 | 0.06 |
| base_think_preloop | ALL | 400 (369) | 0.89 | 0.99 | 0.88 | 5.52 | 0.16 |
| p50c_think | correct | 214 (197) | 0.85 | 0.98 | 0.85 | 6.28 | 0.12 |
| p50c_think | finished: wrong answer | 71 (71) | 0.89 | 1.00 | 0.93 | 7.24 | 0.26 |
| p50c_think | finished: no \boxed{} | 55 (54) | 0.82 | 1.00 | 0.84 | 6.22 | 0.41 |
| p50c_think | cut: degenerate whitespace | 20 (20) | 0.90 | 1.00 | 0.87 | 6.42 | 0.08 |
| p50c_think | cut: still reasoning | 24 (24) | 0.83 | 1.00 | 0.74 | 4.81 | 0.39 |
| p50c_think | cut: repetitive (paraphrase) | 9 (9) | 0.74 | 1.00 | 0.82 | 4.84 | 0.46 |
| p50c_think | cut: verbatim loop | 7 (7) | 0.42 | 0.71 | 0.71 | 5.43 | 0.63 |
| p50c_think | ALL | 400 (382) | 0.84 | 0.99 | 0.86 | 6.31 | 0.22 |
- still-reasoning minus base_nothink | ALL: ratio 1.21, diff 95% CI [0.087, 0.198]
- still-reasoning minus p50c_think | correct: ratio 0.98, diff 95% CI [-0.070, 0.034]
- still-reasoning minus p50c_nothink | ALL: ratio 1.27, diff 95% CI [0.118, 0.226]
- still-reasoning minus base_think_preloop | ALL: ratio 0.94, diff 95% CI [-0.109, -0.009]
- P3 p50c_think ALL vs base pre-loop: ratio 0.95, diff CI [-0.066, -0.023]

### cos>=0.75

| group / class | n (n with ≥12 segs) | PR rate | cycling texts (PR ≥ 0.30) | PR rate, first 2k words | PR per 1k tok (unbounded) | verbatim-repeat rate |
|---|---|---|---|---|---|---|
| base_nothink | ALL | 800 (264) | 0.53 | 0.77 | 0.51 | 2.19 | 0.14 |
| p50c_nothink | ALL | 800 (288) | 0.46 | 0.74 | 0.44 | 2.12 | 0.06 |
| base_think_preloop | ALL | 400 (369) | 0.81 | 0.98 | 0.80 | 4.91 | 0.16 |
| p50c_think | correct | 214 (197) | 0.74 | 0.96 | 0.74 | 5.54 | 0.12 |
| p50c_think | finished: wrong answer | 71 (71) | 0.86 | 1.00 | 0.85 | 6.86 | 0.26 |
| p50c_think | finished: no \boxed{} | 55 (54) | 0.79 | 1.00 | 0.73 | 5.86 | 0.41 |
| p50c_think | cut: degenerate whitespace | 20 (20) | 0.79 | 1.00 | 0.74 | 5.77 | 0.08 |
| p50c_think | cut: still reasoning | 24 (24) | 0.74 | 1.00 | 0.63 | 4.48 | 0.39 |
| p50c_think | cut: repetitive (paraphrase) | 9 (9) | 0.68 | 0.89 | 0.64 | 4.54 | 0.46 |
| p50c_think | cut: verbatim loop | 7 (7) | 0.42 | 0.86 | 0.55 | 5.03 | 0.63 |
| p50c_think | ALL | 400 (382) | 0.76 | 0.98 | 0.74 | 5.74 | 0.22 |
- still-reasoning minus base_nothink | ALL: ratio 1.40, diff 95% CI [0.139, 0.276]
- still-reasoning minus p50c_think | correct: ratio 0.99, diff 95% CI [-0.074, 0.060]
- still-reasoning minus p50c_nothink | ALL: ratio 1.59, diff 95% CI [0.204, 0.336]
- still-reasoning minus base_think_preloop | ALL: ratio 0.90, diff 95% CI [-0.141, -0.014]
- P3 p50c_think ALL vs base pre-loop: ratio 0.94, diff CI [-0.077, -0.020]

### cos>=0.75 strict12

| group / class | n (n with ≥12 segs) | PR rate | cycling texts (PR ≥ 0.30) | PR rate, first 2k words | PR per 1k tok (unbounded) | verbatim-repeat rate |
|---|---|---|---|---|---|---|
| base_nothink | ALL | 800 (264) | 0.52 | 0.76 | 0.51 | 2.17 | 0.14 |
| p50c_nothink | ALL | 800 (288) | 0.46 | 0.73 | 0.44 | 2.11 | 0.06 |
| base_think_preloop | ALL | 400 (369) | 0.80 | 0.98 | 0.79 | 4.82 | 0.16 |
| p50c_think | correct | 214 (197) | 0.73 | 0.96 | 0.73 | 5.51 | 0.12 |
| p50c_think | finished: wrong answer | 71 (71) | 0.82 | 1.00 | 0.84 | 6.78 | 0.26 |
| p50c_think | finished: no \boxed{} | 55 (54) | 0.74 | 1.00 | 0.72 | 5.79 | 0.41 |
| p50c_think | cut: degenerate whitespace | 20 (20) | 0.78 | 1.00 | 0.73 | 5.71 | 0.08 |
| p50c_think | cut: still reasoning | 24 (24) | 0.73 | 1.00 | 0.62 | 4.47 | 0.39 |
| p50c_think | cut: repetitive (paraphrase) | 9 (9) | 0.65 | 0.89 | 0.63 | 4.52 | 0.46 |
| p50c_think | cut: verbatim loop | 7 (7) | 0.35 | 0.71 | 0.54 | 4.78 | 0.63 |
| p50c_think | ALL | 400 (382) | 0.74 | 0.97 | 0.74 | 5.69 | 0.22 |
- still-reasoning minus base_nothink | ALL: ratio 1.41, diff 95% CI [0.139, 0.277]
- still-reasoning minus p50c_think | correct: ratio 1.00, diff 95% CI [-0.071, 0.065]
- still-reasoning minus p50c_nothink | ALL: ratio 1.59, diff 95% CI [0.201, 0.335]
- still-reasoning minus base_think_preloop | ALL: ratio 0.91, diff 95% CI [-0.135, -0.006]
- P3 p50c_think ALL vs base pre-loop: ratio 0.93, diff CI [-0.086, -0.028]

### lex>=0.5

| group / class | n (n with ≥12 segs) | PR rate | cycling texts (PR ≥ 0.30) | PR rate, first 2k words | PR per 1k tok (unbounded) | verbatim-repeat rate |
|---|---|---|---|---|---|---|
| base_nothink | ALL | 800 (264) | 0.03 | 0.02 | 0.03 | 0.11 | 0.14 |
| p50c_nothink | ALL | 800 (288) | 0.01 | 0.00 | 0.01 | 0.04 | 0.06 |
| base_think_preloop | ALL | 400 (369) | 0.12 | 0.12 | 0.08 | 0.59 | 0.16 |
| p50c_think | correct | 214 (197) | 0.08 | 0.06 | 0.02 | 0.68 | 0.12 |
| p50c_think | finished: wrong answer | 71 (71) | 0.24 | 0.39 | 0.03 | 2.29 | 0.26 |
| p50c_think | finished: no \boxed{} | 55 (54) | 0.30 | 0.48 | 0.06 | 2.76 | 0.41 |
| p50c_think | cut: degenerate whitespace | 20 (20) | 0.12 | 0.05 | 0.03 | 0.85 | 0.08 |
| p50c_think | cut: still reasoning | 24 (24) | 0.24 | 0.33 | 0.03 | 2.10 | 0.39 |
| p50c_think | cut: repetitive (paraphrase) | 9 (9) | 0.27 | 0.56 | 0.06 | 2.42 | 0.46 |
| p50c_think | cut: verbatim loop | 7 (7) | 0.19 | 0.29 | 0.11 | 2.97 | 0.63 |
| p50c_think | ALL | 400 (382) | 0.16 | 0.21 | 0.03 | 1.43 | 0.22 |
- still-reasoning minus base_nothink | ALL: ratio 7.02, diff 95% CI [0.138, 0.285]
- still-reasoning minus p50c_think | correct: ratio 3.20, diff 95% CI [0.095, 0.244]
- still-reasoning minus p50c_nothink | ALL: ratio 22.67, diff 95% CI [0.162, 0.308]
- still-reasoning minus base_think_preloop | ALL: ratio 2.04, diff 95% CI [0.053, 0.200]
- P3 p50c_think ALL vs base pre-loop: ratio 1.34, diff CI [0.018, 0.063]

### lex>=0.5 strict12

| group / class | n (n with ≥12 segs) | PR rate | cycling texts (PR ≥ 0.30) | PR rate, first 2k words | PR per 1k tok (unbounded) | verbatim-repeat rate |
|---|---|---|---|---|---|---|
| base_nothink | ALL | 800 (264) | 0.03 | 0.02 | 0.03 | 0.09 | 0.14 |
| p50c_nothink | ALL | 800 (288) | 0.01 | 0.00 | 0.01 | 0.03 | 0.06 |
| base_think_preloop | ALL | 400 (369) | 0.09 | 0.08 | 0.05 | 0.43 | 0.16 |
| p50c_think | correct | 214 (197) | 0.05 | 0.03 | 0.01 | 0.53 | 0.12 |
| p50c_think | finished: wrong answer | 71 (71) | 0.16 | 0.14 | 0.03 | 1.76 | 0.26 |
| p50c_think | finished: no \boxed{} | 55 (54) | 0.20 | 0.22 | 0.05 | 2.19 | 0.41 |
| p50c_think | cut: degenerate whitespace | 20 (20) | 0.11 | 0.05 | 0.02 | 0.77 | 0.08 |
| p50c_think | cut: still reasoning | 24 (24) | 0.23 | 0.29 | 0.03 | 2.05 | 0.39 |
| p50c_think | cut: repetitive (paraphrase) | 9 (9) | 0.23 | 0.22 | 0.06 | 2.32 | 0.46 |
| p50c_think | cut: verbatim loop | 7 (7) | 0.10 | 0.00 | 0.06 | 2.53 | 0.63 |
| p50c_think | ALL | 400 (382) | 0.11 | 0.10 | 0.02 | 1.16 | 0.22 |
- still-reasoning minus base_nothink | ALL: ratio 7.38, diff 95% CI [0.130, 0.275]
- still-reasoning minus p50c_think | correct: ratio 4.33, diff 95% CI [0.107, 0.254]
- still-reasoning minus p50c_nothink | ALL: ratio 23.03, diff 95% CI [0.151, 0.296]
- still-reasoning minus base_think_preloop | ALL: ratio 2.66, diff 95% CI [0.073, 0.219]
- P3 p50c_think ALL vs base pre-loop: ratio 1.31, diff CI [0.008, 0.045]

### lex>=0.6

| group / class | n (n with ≥12 segs) | PR rate | cycling texts (PR ≥ 0.30) | PR rate, first 2k words | PR per 1k tok (unbounded) | verbatim-repeat rate |
|---|---|---|---|---|---|---|
| base_nothink | ALL | 800 (264) | 0.01 | 0.01 | 0.01 | 0.04 | 0.14 |
| p50c_nothink | ALL | 800 (288) | 0.00 | 0.00 | 0.00 | 0.01 | 0.06 |
| base_think_preloop | ALL | 400 (369) | 0.02 | 0.01 | 0.01 | 0.12 | 0.16 |
| p50c_think | correct | 214 (197) | 0.02 | 0.01 | 0.00 | 0.19 | 0.12 |
| p50c_think | finished: wrong answer | 71 (71) | 0.08 | 0.00 | 0.00 | 0.87 | 0.26 |
| p50c_think | finished: no \boxed{} | 55 (54) | 0.11 | 0.00 | 0.00 | 1.24 | 0.41 |
| p50c_think | cut: degenerate whitespace | 20 (20) | 0.03 | 0.00 | 0.00 | 0.23 | 0.08 |
| p50c_think | cut: still reasoning | 24 (24) | 0.10 | 0.04 | 0.01 | 1.01 | 0.39 |
| p50c_think | cut: repetitive (paraphrase) | 9 (9) | 0.11 | 0.00 | 0.01 | 1.22 | 0.46 |
| p50c_think | cut: verbatim loop | 7 (7) | 0.12 | 0.14 | 0.05 | 2.43 | 0.63 |
| p50c_think | ALL | 400 (382) | 0.05 | 0.01 | 0.00 | 0.57 | 0.22 |
- still-reasoning minus base_nothink | ALL: ratio 11.43, diff 95% CI [0.043, 0.162]
- still-reasoning minus p50c_think | correct: ratio 4.73, diff 95% CI [0.029, 0.148]
- still-reasoning minus p50c_nothink | ALL: ratio 43.36, diff 95% CI [0.051, 0.169]
- still-reasoning minus base_think_preloop | ALL: ratio 4.30, diff 95% CI [0.027, 0.145]
- P3 p50c_think ALL vs base pre-loop: ratio 2.29, diff CI [0.021, 0.041]

### lex>=0.6 strict12

| group / class | n (n with ≥12 segs) | PR rate | cycling texts (PR ≥ 0.30) | PR rate, first 2k words | PR per 1k tok (unbounded) | verbatim-repeat rate |
|---|---|---|---|---|---|---|
| base_nothink | ALL | 800 (264) | 0.01 | 0.00 | 0.01 | 0.03 | 0.14 |
| p50c_nothink | ALL | 800 (288) | 0.00 | 0.00 | 0.00 | 0.01 | 0.06 |
| base_think_preloop | ALL | 400 (369) | 0.01 | 0.01 | 0.01 | 0.07 | 0.16 |
| p50c_think | correct | 214 (197) | 0.01 | 0.00 | 0.00 | 0.11 | 0.12 |
| p50c_think | finished: wrong answer | 71 (71) | 0.03 | 0.00 | 0.00 | 0.46 | 0.26 |
| p50c_think | finished: no \boxed{} | 55 (54) | 0.05 | 0.00 | 0.00 | 0.65 | 0.41 |
| p50c_think | cut: degenerate whitespace | 20 (20) | 0.03 | 0.00 | 0.00 | 0.20 | 0.08 |
| p50c_think | cut: still reasoning | 24 (24) | 0.10 | 0.04 | 0.01 | 0.95 | 0.39 |
| p50c_think | cut: repetitive (paraphrase) | 9 (9) | 0.07 | 0.00 | 0.01 | 0.99 | 0.46 |
| p50c_think | cut: verbatim loop | 7 (7) | 0.04 | 0.00 | 0.02 | 2.02 | 0.63 |
| p50c_think | ALL | 400 (382) | 0.03 | 0.00 | 0.00 | 0.35 | 0.22 |
- still-reasoning minus base_nothink | ALL: ratio 12.36, diff 95% CI [0.040, 0.149]
- still-reasoning minus p50c_think | correct: ratio 8.65, diff 95% CI [0.035, 0.146]
- still-reasoning minus p50c_nothink | ALL: ratio 41.76, diff 95% CI [0.046, 0.155]
- still-reasoning minus base_think_preloop | ALL: ratio 6.57, diff 95% CI [0.032, 0.141]
- P3 p50c_think ALL vs base pre-loop: ratio 1.98, diff CI [0.007, 0.021]
