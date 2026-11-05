main: full_runpod, n=20 texts x 4096 tok, 10619 loop positions; total +0.0590 (base 0.0817 -> p50c 0.1407); prod NLL 0.401 -> 0.437
replication run: n=30; 20 texts shared with main, 10 held-out texts (10431 loop positions)
cross-hardware, same texts: max |Δtotal diff| 0.0026, max |block attn16-23 diff| 0.0027

| component | block % (main) [95% CI] | inject % (main) | noise ctrl % | perm ctrl % | direct % | block % held-out | inject % held-out |
|---|---|---|---|---|---|---|---|
| mlp15 | 69 [48, 94] | 60 | 4 | 34 | 3 | 80 | 56 |
| mlp19 | 56 [38, 80] | 14 | 0 | 10 | 3 | 58 | 8 |
| attn16-23 | 91 [67, 118] | 53 | 11 | 35 | – | 96 | 55 |
| attn16-21 | 69 [47, 95] | 18 | 3 | 10 | – | 81 | 15 |
| attn22-23 | 70 [48, 95] | 20 | 4 | 15 | – | 72 | 13 |
| fullattn19+23 | 76 [53, 103] | 22 | 6 | 8 | – | 85 | 24 |
| attn19 | 43 [26, 64] | 4 | 1 | 1 | 16 | 54 | 3 |
| attn23 | 58 [37, 82] | 13 | 4 | 8 | 71 | 56 | 9 |
| mlp16-23(excl19) | -61 [-92, -32] | 32 | 21 | 18 | – | -72 | 22 |
| mlp16-18 | -7 [-23, 9] | 3 | 2 | 3 | – | -42 | 4 |
| mlp20-23 | -55 [-89, -25] | 28 | 15 | 12 | – | -41 | 18 |
| mlp22-23 | -79 [-112, -52] | 14 | 16 | 10 | – | -86 | 13 |
| mlp23 | -61 [-87, -35] | 5 | 16 | 15 | 5 | -56 | 4 |
| layers22-23 | 56 [32, 83] | 97 | 33 | 39 | – | 58 | 82 |
| all_downstream(excl mlp19) | 97 [73, 124] | 132 | 109 | 98 | – | 97 | 113 |
| h23.5 | 11 [3, 18] | 6 | 1 | 1 | 27 | 20 | 3 |
| h23.7 | 29 [14, 48] | 2 | 0 | 2 | -0 | 28 | 1 |
| h23.0 | 12 [5, 20] | 2 | 0 | 1 | 12 | 14 | 1 |
| h23.2 | -14 [-18, -10] | -1 | 0 | 1 | 1 | -23 | 0 |
| h23.6 | 12 [6, 18] | 0 | 0 | 1 | 4 | 11 | 1 |
| h19.4 | 17 [10, 26] | 2 | -0 | 0 | 6 | 32 | 2 |
| h19.6 | 13 [6, 22] | 1 | 0 | 0 | 0 | 10 | 1 |
| h18.14 | 17 [11, 23] | 2 | – | – | 0 | 23 | 1 |
| h16.6 | 19 [11, 28] | 2 | – | – | 1 | 26 | 1 |
| h17.6 | 13 [8, 19] | 1 | – | – | -0 | 18 | 1 |
| h22.2 | 11 [7, 16] | 2 | – | – | 2 | 11 | 1 |
| h22.8 | -14 [-20, -8] | -0 | – | – | 0 | -15 | 0 |

direct (last hop only): edit sites mlp15+mlp19 3.3 %, all Δ 100.0 %
edit L15 only 60 %, L19 only 31 %
AtP vs exact inject Spearman: {'base': 0.6730657423971378, 'edit': 0.6241111359570664, 'mean': 0.6359458855098391}
  AtP mlp15: base-grad +0.0110  p50c-grad +0.1454  exact inject +0.0356  exact block +0.0408
  AtP mlp19: base-grad +0.0050  p50c-grad +0.0526  exact inject +0.0081  exact block +0.0332
  AtP attn23: base-grad +0.0035  p50c-grad +0.0776  exact inject +0.0077  exact block +0.0341
  AtP mlp23: base-grad -0.0069  p50c-grad +0.0616  exact inject +0.0028  exact block -0.0359
  AtP attn19: base-grad +0.0013  p50c-grad +0.0558  exact inject +0.0025  exact block +0.0254

position split of inject (% total): all / at loop (query) positions only / at all other positions
  attn19: 4.2 / 3.4 / 0.3
  attn22: 7.5 / 6.0 / 1.1
  attn23: 13.0 / 13.0 / 0.0
  mlp15: 60.4 / 56.7 / 3.1
  mlp19: 13.7 / 12.9 / 1.2
  h23.5: 5.5 / 5.5 / 0.0
  h23.7: 1.8 / 1.8 / 0.0
  h19.4: 2.3 / 2.4 / -0.2

induction attention (query p -> earlier position holding the repeated token), mean over loop positions, per head; base -> p50c
  L15: h0 0.010->0.010  h1 0.013->0.013  h2 0.019->0.019  h3 0.021->0.021  h4 0.016->0.016  h5 0.050->0.050  h6 0.022->0.022  h7 0.128->0.128   sum 0.280->0.280 (+0 %)
    duplicate-token mass sum 0.032->0.032
  L19: h0 0.028->0.024  h1 0.018->0.016  h2 0.035->0.028  h3 0.030->0.023  h4 0.007->0.013  h5 0.033->0.026  h6 0.031->0.031  h7 0.026->0.021   sum 0.209->0.182 (-13 %)
    duplicate-token mass sum 0.023->0.026
  L23: h0 0.042->0.025  h1 0.028->0.027  h2 0.001->0.001  h3 0.025->0.025  h4 0.036->0.024  h5 0.071->0.035  h6 0.000->0.001  h7 0.039->0.023   sum 0.242->0.160 (-34 %)
    duplicate-token mass sum 0.143->0.141

logit lens of Δ_X (p50c − base output, linearised through the base final norm) at loop positions:
| X | ‖Δ‖ | cos(Δ,u15) | consistency | centred Δlogit of repeated token | median percentile of target | top promoted | top suppressed |
|---|---|---|---|---|---|---|---|
| ALL | 4.99 | -0.21 | 0.70 | -3.99 | 0.040 | '’:', ' evaluations', ' computations', '方法及', ' formulas', ' examinations' | '负', '不是', '蓝', '还是', '客', '指' |
| mlp15 | 1.58 | -0.98 | 0.98 | +0.68 | 0.938 | '呗', ' first', ' series', ' or', ' thereby', ' f' | '...*', '...**', 'urada', 'essin', 'ologe', 'kede' |
| mlp19 | 1.34 | -0.48 | 0.76 | +0.03 | 0.527 | ' discret', '对', '呗', 'ства', '与被', ' pair' | 'omentar', 'enheim', 'emies', '...*', '而是因为', 'oment' |
| attn19 | 0.93 | -0.13 | 0.61 | -0.14 | 0.515 | ' dec', 'lo', ' ', '1', '0', '去' | 'óis', 'manos', ' Planej', 'amentais', 'ierz', 'indert' |
| attn22 | 1.29 | +0.01 | 0.69 | -0.12 | 0.342 | '二者', 'antin', 'reve', '请参考', 'فه', '内容及' | ' hub', 'isp', ' select', 'usercontent', ' **', 'istan' |
| attn23 | 2.08 | +0.17 | 0.74 | -3.10 | 0.000 | 'ciso', ' –**', '者为', ' *«', ' **‘', '中国国家' | ' shr', ' ', ' constr', 'owitz', '嘛', '\n' |
| h23.5 | 0.95 | -0.02 | 0.66 | -1.32 | 0.000 | '�', 'toDouble', 'god', ' slike', 'arvi', '�' | '易', 'itions', '重要', '遇', 'olarity', 'uids' |
| h23.7 | 0.85 | -0.00 | 0.82 | -0.02 | 0.436 | 'ip', '攀', '重要', 'Jac', 'adj', 'un' | 'ergestellt', '้อ', '东西', 'ley', '缺氧', 'kering' |
| mlp22 | 1.51 | +0.14 | 0.69 | -0.95 | 0.057 | 'atiques', ' счастью', 'ínc', 'iennes', 'acies', 'onale' | ' o', ' O', ' st', 'er', ' St', ' e' |
| mlp23 | 3.69 | +0.24 | 0.82 | -1.33 | 0.325 | 'apers', 'venues', 'essions', 'inations', 'scriptions', 'anners' | '重要', '小', '高', '关注', '大', '好' |

Δlogit (centred) of discourse tokens, all-Δ / attn23 / mlp15: '</think>' -0.54/-1.23/+0.91, '\n\n' -1.64/-1.44/+0.71, ' So' -1.01/-0.76/+0.52, ' Therefore' +0.34/-0.64/+0.52, ' Wait' -2.74/-1.25/+0.17, 'Wait' -2.59/-1.23/+0.19, ' final' +0.40/-0.27/+0.77, ' answer' +0.94/+0.04/+0.47, ' Thus' +1.79/-0.17/+0.98, ' Let' -1.05/-1.01/+0.69, ' Hmm' -1.99/-0.87/+0.02, ' But' -1.21/-0.96/+0.66, ' first' +0.86/-0.84/+1.89, ' result' +3.06/-0.17/+1.41, ' value' +2.23/-0.28/+1.06, '**' -1.71/-0.60/+0.38, ' Final' -0.14/-0.63/+0.42, 'Final' -0.31/-0.61/+0.53, ' The' -0.37/-0.87/+0.76, 'So' -1.24/-0.57/+0.19

L15 h7: ablate in base Δloop +0.0131 (22 % of p50c), Δprod +0.0153; control L15 h3 ablation +0.0059; extra when ablated on top of p50c +0.0992; 64 L15 neurons' u-write at loop positions 0.259 -> 0.251 (-3 %), prod 0.102 -> 0.091; h7 output·u15 loop +0.093 vs prod +0.044 (cos +0.08)
  held-out: h7 ablate base +0.0109, extra on p50c +0.1567, total +0.0787
