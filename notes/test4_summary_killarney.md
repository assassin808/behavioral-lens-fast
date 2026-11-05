## paired: {"n_items": 40, "maxt": 8192, "n_loop_pos": 151030, "n_prod_pos": 154720} 
recon FVU at loop positions (base / edit acts): {"ours16": {"base": 0.14540326744318008, "edit": 0.563729539513588}, "pub18": {"base": 0.24358378052711488, "edit": 0.427348917722702}, "ours20": {"base": 0.1270520083606243, "edit": 0.5266635626554489}}

### ours16 (hidden_states[16], d_sae 16384, read against u15) cfg {"layer": 16, "d_sae": 16384, "k": 64, "fvu": 0.13639040929165777, "l0": 58.62402666831503, "dead_frac": 0.0, "n_train_tokens": 17400499, "steps": 4248, "n_eval": 400000, "scale": 0.10762589424848557}
max |cos(dec, u)| = 0.565; #features |cos|>0.3: 19; residual ‖u − proj on top-k greedy features‖ k=1..10: [0.83, 0.75, 0.7, 0.68, 0.66, 0.64, 0.62, 0.61, 0.61, 0.6]; cos(mean Δresid at loop pos, u) = -1.00; frac of top-20 edit-ON features in base top-100 productive: 0.25

**cos_plus_u**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 3673 | Mid-phrase token while restating earlier text | Y | 0.92 | +0.56 | +1.87 | 7.15→1.96 | 0.77→0.50 | et` as the bullet marker.\n    *   Wait, I'll use the text `*[[ This]] is an examp ⏐ et` as the bullet marker.\n    *   Wait, I'll use the text `*[[ This]] is an examp |
| 6592 | Noun in self-correction constraint-checking sentences | N | 0.58 | +0.49 | +0.59 | 0.64→0.01 | 0.28→0.01 | te safety guidelines).*\n    *I will not claim he is a circus[[ clown]] unless I ha ⏐ te safety guidelines).*\n    *I will not claim he is a circus[[ clown]] unless I ha |
| 1375 | Newline after a stuck or contradictory reasoning step | Y | 1.0 | +0.38 | -0.09 | 0.13→0.06 | 0.05→0.02 | iles North.\n    The distance covered by each is 80 and 150.\n[[    \n]]    I will p ⏐ travel in opposite directions, the distance covered is 230.\n[[    \n]]    Let's as |
| 7236 | Copying next number during repeated calculation | Y | 0.83 | +0.36 | +0.42 | 0.38→0.00 | 0.18→0.00 | *   Wait, $18 \times 9 = 162$.\n    *   Wait, $18 \times 9 = [[1]]62$.\n    * ⏐ .8 = 180.\nWait, 225 * 0.8 = 180.\nWait, 225 * 0.8 = [[1]]80.\n\nOkay, |
| 7770 | "Wait" tokens in repeated self-correction loops | Y | 0.83 | +0.36 | +0.13 | 0.08→0.01 | 0.04→0.01 | e^{i(2\pi/3)}$.\n    Wait, $e^{i2\pi/3} = e^{i(2\pi/3)}$.\n   [[ Wait]], $e^{i2 ⏐ Wait, $2^{24} = 2^{24}$.\n                                *  [[ Wait]], $2^{24 |
| 1444 | Closing math "$" before repeated "Wait," line | Y | 0.83 | +0.35 | +0.48 | 0.61→0.39 | 0.17→0.12 | ^{42}$.\nWait, $z^{85} + z^{-85} = (z^2)^{42} + (z^{-2})^{42}[[$.]]\nWait, $z^{ ⏐ ^{42}$.\nWait, $z^{85} + z^{-85} = (z^2)^{42} + (z^{-2})^{42}[[$.]]\nWait, $z^{ |
| 2562 | Indentation whitespace before repeated "Wait," line | Y | 0.83 | +0.34 | +0.47 | 0.94→0.80 | 0.17→0.15 | . Yes.\n    Wait, I need to check if "Part" is allowed. Yes.\n[[   ]] Wait, I nee ⏐ . Yes.\n    Wait, I need to check if "Part" is allowed. Yes.\n[[   ]] Wait, I nee |
| 13 | Newline before another "Wait," re-check line | Y | 0.83 | +0.34 | +0.45 | 0.48→0.42 | 0.12→0.11 |      *   $40 + 4 = 44$.\n            *   Wait, $40 + 4 = 44$.[[\n]]             ⏐      *   $40 + 4 = 44$.\n            *   Wait, $40 + 4 = 44$.[[\n]]             |

**cos_minus_u**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 8733 | Noun after "imply/part of a" hedging | N | 0.83 | -0.49 | -0.88 | 0.01→1.50 | 0.00→0.75 | re a trick where "after a month" refers to the specific time[[ frame]] relative to ⏐ e interpreted as "less than" but maybe the wording implies a[[ specific]] relationshi |
| 10763 | Binary operator/equals before next operand number | N | 0.92 | -0.44 | -0.50 | 0.01→0.07 | 0.00→0.05 | $.\n\n3.  **Calculate Remaining Books:**\n    *   Top Shelf: $8[[ -]] 2 = 6$ ⏐ en $m = n$ if $a=b$. Here $a=2$ and $b=2$.\n        Thus, $2x[[ =]] 4$.\n    * |
| 5917 | Period ending problem-statement sentence in analysis | N | 0.75 | -0.35 | -0.48 | 0.01→0.60 | 0.00→0.43 | ultiply the number of roses per bush by the number of bushes[[.]]\n        *   ⏐ is asking for the value of $x$ given the equation $2^8 = 4^x[[$.]] They want a |
| 7614 | Clause-boundary comma after complete main clause | N | 0.92 | -0.33 | -0.18 | 0.00→0.34 | 0.00→0.27 | day the remaining elements of ADGB became RAF Bomber Command[[ ,]] headquarter ⏐ ed on January 31 , 2012 in North America with an English dub[[ ,]] and the Ef  |
| 9790 | Mid-phrase token in memorized fixed phrase | N | 0.75 | -0.32 | -0.17 | 0.00→0.18 | 0.00→0.16 | >\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport[["]] content="wi ⏐  + \frac{r}{n}\right)^{nt}$$\nWhere:\n*   $A$ is the amount of[[ money]] accumulated |

**contrib_u_loop_minus_prod**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 3673 | Mid-phrase token while restating earlier text | Y | 0.92 | +0.56 | +1.87 | 7.15→1.96 | 0.77→0.50 | et` as the bullet marker.\n    *   Wait, I'll use the text `*[[ This]] is an examp ⏐ et` as the bullet marker.\n    *   Wait, I'll use the text `*[[ This]] is an examp |
| 8733 | Noun after "imply/part of a" hedging | N | 0.83 | -0.49 | -0.88 | 0.01→1.50 | 0.00→0.75 | re a trick where "after a month" refers to the specific time[[ frame]] relative to ⏐ e interpreted as "less than" but maybe the wording implies a[[ specific]] relationshi |
| 6592 | Noun in self-correction constraint-checking sentences | N | 0.58 | +0.49 | +0.59 | 0.64→0.01 | 0.28→0.01 | te safety guidelines).*\n    *I will not claim he is a circus[[ clown]] unless I ha ⏐ te safety guidelines).*\n    *I will not claim he is a circus[[ clown]] unless I ha |
| 2562 | Indentation whitespace before repeated "Wait," line | Y | 0.83 | +0.34 | +0.47 | 0.94→0.80 | 0.17→0.15 | . Yes.\n    Wait, I need to check if "Part" is allowed. Yes.\n[[   ]] Wait, I nee ⏐ . Yes.\n    Wait, I need to check if "Part" is allowed. Yes.\n[[   ]] Wait, I nee |
| 1444 | Closing math "$" before repeated "Wait," line | Y | 0.83 | +0.35 | +0.48 | 0.61→0.39 | 0.17→0.12 | ^{42}$.\nWait, $z^{85} + z^{-85} = (z^2)^{42} + (z^{-2})^{42}[[$.]]\nWait, $z^{ ⏐ ^{42}$.\nWait, $z^{85} + z^{-85} = (z^2)^{42} + (z^{-2})^{42}[[$.]]\nWait, $z^{ |
| 10763 | Binary operator/equals before next operand number | N | 0.92 | -0.44 | -0.50 | 0.01→0.07 | 0.00→0.05 | $.\n\n3.  **Calculate Remaining Books:**\n    *   Top Shelf: $8[[ -]] 2 = 6$ ⏐ en $m = n$ if $a=b$. Here $a=2$ and $b=2$.\n        Thus, $2x[[ =]] 4$.\n    * |
| 2293 | Subject token right before stating a computed value | N | 0.75 | +0.23 | +0.68 | 0.46→0.24 | 0.20→0.11 |     *   If he stops at 45, he traveled 45 miles.\n    *   The[[ distance]] between the ⏐     *   If he stops at 45, he traveled 45 miles.\n    *   The[[ distance]] between the |
| 13 | Newline before another "Wait," re-check line | Y | 0.83 | +0.34 | +0.45 | 0.48→0.42 | 0.12→0.11 |      *   $40 + 4 = 44$.\n            *   Wait, $40 + 4 = 44$.[[\n]]             ⏐      *   $40 + 4 = 44$.\n            *   Wait, $40 + 4 = 44$.[[\n]]             |

**edit_off_loop**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 3673 | Mid-phrase token while restating earlier text | Y | 0.92 | +0.56 | +1.87 | 7.15→1.96 | 0.77→0.50 | et` as the bullet marker.\n    *   Wait, I'll use the text `*[[ This]] is an examp ⏐ et` as the bullet marker.\n    *   Wait, I'll use the text `*[[ This]] is an examp |
| 6592 | Noun in self-correction constraint-checking sentences | N | 0.58 | +0.49 | +0.59 | 0.64→0.01 | 0.28→0.01 | te safety guidelines).*\n    *I will not claim he is a circus[[ clown]] unless I ha ⏐ te safety guidelines).*\n    *I will not claim he is a circus[[ clown]] unless I ha |
| 149 | Newline before repeating "Wait," line again | Y | 0.5 | +0.13 | +0.42 | 0.56→0.13 | 0.25→0.05 | $f$ decreases on $(x_1, x_2)$, then $f$ is smaller at $x_2$.[[\n]]Wait, if $f$ ⏐ $f$ decreases on $(x_1, x_2)$, then $f$ is smaller at $x_2$.[[\n]]Wait, if $f$ |
| 7236 | Copying next number during repeated calculation | Y | 0.83 | +0.36 | +0.42 | 0.38→0.00 | 0.18→0.00 | *   Wait, $18 \times 9 = 162$.\n    *   Wait, $18 \times 9 = [[1]]62$.\n    * ⏐ .8 = 180.\nWait, 225 * 0.8 = 180.\nWait, 225 * 0.8 = [[1]]80.\n\nOkay, |
| 11102 | Token right after "Wait," restating prior content | Y | 0.83 | +0.29 | +0.65 | 0.41→0.06 | 0.20→0.04 | ^2 \cdot 3 \cdot 5 = 60$.\n    $(60)^2 = 3600$.\n    Wait, $2^[[4]] \cdot 3^2 ⏐ ^2 \cdot 3 \cdot 5 = 60$.\n    $(60)^2 = 3600$.\n    Wait, $2^[[4]] \cdot 3^2 |
| 2293 | Subject token right before stating a computed value | N | 0.75 | +0.23 | +0.68 | 0.46→0.24 | 0.20→0.11 |     *   If he stops at 45, he traveled 45 miles.\n    *   The[[ distance]] between the ⏐     *   If he stops at 45, he traveled 45 miles.\n    *   The[[ distance]] between the |
| 1444 | Closing math "$" before repeated "Wait," line | Y | 0.83 | +0.35 | +0.48 | 0.61→0.39 | 0.17→0.12 | ^{42}$.\nWait, $z^{85} + z^{-85} = (z^2)^{42} + (z^{-2})^{42}[[$.]]\nWait, $z^{ ⏐ ^{42}$.\nWait, $z^{85} + z^{-85} = (z^2)^{42} + (z^{-2})^{42}[[$.]]\nWait, $z^{ |
| 5526 | Final token of a restated, copied phrase | Y | 0.75 | +0.20 | +0.45 | 0.58→0.36 | 0.17→0.11 | takes 0.5 hours to walk, and 0.5 hours to take care, that's [[1]] hour.\n    * ⏐ ces.\n        *   This implies that the 5 pies were cut into [[8]] pieces.\n    |
| 12009 | Single-digit numeral token in reasoning | Y | 1.0 | +0.13 | +0.02 | 0.35→0.21 | 0.09→0.07 |                    *   This implies the remaining money is $[[5]].\n           ⏐                    *   This implies the remaining money is $[[5]].\n           |
| 13840 | Digit copying inside repeated number restatement | Y | 0.67 | +0.29 | +0.43 | 0.19→0.05 | 0.09→0.03 |  then $x^{10} + (13x-1)^{10} = 1/13^{10} + 0$.\nSo $x=1/[[1]]3$ is not a  ⏐  then $x^{10} + (13x-1)^{10} = 1/13^{10} + 0$.\nSo $x=1/[[1]]3$ is not a  |

**edit_on_loop**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 8733 | Noun after "imply/part of a" hedging | N | 0.83 | -0.49 | -0.88 | 0.01→1.50 | 0.00→0.75 | re a trick where "after a month" refers to the specific time[[ frame]] relative to ⏐ e interpreted as "less than" but maybe the wording implies a[[ specific]] relationshi |
| 8156 | Content word completing a clause (noun/verb) | N | 0.75 | -0.14 | -0.28 | 0.01→1.34 | 0.01→0.76 | Benson between Omaha and Benson specifically to promote that[[ suburbs]] development ⏐ mp no longer performs its original function , but supplies a[[ fountain]] near the vi |
| 10080 | Mental-math decomposition: next number to split | N | 0.83 | -0.19 | -0.13 | 0.00→0.85 | 0.00→0.59 | .\n\n    Wait, let me recalculate $4 \times 45$.\n    $4 \times[[ ]]40 = 16 ⏐ gain.**\n        *   $3 \times 25$:\n            *   $3 \times[[ ]]10 = 30 |
| 13144 | Final digit of a computed arithmetic result | N | 0.75 | -0.19 | -0.14 | 0.01→0.61 | 0.00→0.44 |     $$\text{Total} = 500 + 440$$\n        $$\text{Total} = 94[[0]]$$\n\n    Fina ⏐ N_B \times P_B = 10 \times 60 + 5 \times 40 = 600 + 200 = 80[[0]]$.\n\n    *    |
| 5917 | Period ending problem-statement sentence in analysis | N | 0.75 | -0.35 | -0.48 | 0.01→0.60 | 0.00→0.43 | ultiply the number of roses per bush by the number of bushes[[.]]\n        *   ⏐ is asking for the value of $x$ given the equation $2^8 = 4^x[[$.]] They want a |
| 7142 | Noun completing a modified noun phrase | N | 0.75 | -0.32 | -0.21 | 0.01→0.59 | 0.00→0.44 |  @-@ General S. B. Woods had tampered with the government 's[[ files]] on the A &  ⏐ ails of her visit attracted much attention from the Parisian[[ press]] , which exp |
| 2996 | Sentence-ending period in Wikipedia prose | N | 0.5 | -0.07 | -0.01 | 0.00→0.48 | 0.00→0.40 | a on December 9 , 2014 , and in Europe on December 11 , 2014[[ .]] \n = = = Loc ⏐ ege , and Hunter College and at DePaul University in Chicago[[ .]] During the  |
| 1011 | Final summation step after listed computations | Y | 0.92 | -0.19 | -0.46 | 0.00→0.45 | 0.00→0.37 |  125.\n        *   Alex = $4 \times 125 - 2 = 498$.\n        *[[  ]] Total = $12 ⏐ Final confirmation:\n    May = 500\n    June = 500 - 60 = 440\n[[   ]] Total = 500 |
| 1729 | Abstract noun completing generic formal phrase | N | 0.75 | -0.19 | -0.15 | 0.00→0.39 | 0.00→0.33 | ssential because they provide a clear framework for decision[[ making]] within the  ⏐ g phase. We trust that your enthusiasm will drive meaningful[[ discussions]] throughout  |
| 15688 | Wikitext numeric separator "@.@" / "@,@" tokens | N | 0.67 | -0.17 | -0.16 | 0.00→0.34 | 0.00→0.27 | age or staining green with damage . The stem is 2 – 6 cm ( 0[[ @]].@ 8 – 2 ⏐ gue Baseball ( MLB ) pitcher . He stands 6 feet 2 inches ( 1[[ @]].@ 88 m ) |

**cos_dresid**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 8733 | Noun after "imply/part of a" hedging | N | 0.83 | -0.49 | -0.88 | 0.01→1.50 | 0.00→0.75 | re a trick where "after a month" refers to the specific time[[ frame]] relative to ⏐ e interpreted as "less than" but maybe the wording implies a[[ specific]] relationshi |
| 10763 | Binary operator/equals before next operand number | N | 0.92 | -0.44 | -0.50 | 0.01→0.07 | 0.00→0.05 | $.\n\n3.  **Calculate Remaining Books:**\n    *   Top Shelf: $8[[ -]] 2 = 6$ ⏐ en $m = n$ if $a=b$. Here $a=2$ and $b=2$.\n        Thus, $2x[[ =]] 4$.\n    * |
| 5917 | Period ending problem-statement sentence in analysis | N | 0.75 | -0.35 | -0.48 | 0.01→0.60 | 0.00→0.43 | ultiply the number of roses per bush by the number of bushes[[.]]\n        *   ⏐ is asking for the value of $x$ given the equation $2^8 = 4^x[[$.]] They want a |
| 7614 | Clause-boundary comma after complete main clause | N | 0.92 | -0.33 | -0.18 | 0.00→0.34 | 0.00→0.27 | day the remaining elements of ADGB became RAF Bomber Command[[ ,]] headquarter ⏐ ed on January 31 , 2012 in North America with an English dub[[ ,]] and the Ef  |
| 9790 | Mid-phrase token in memorized fixed phrase | N | 0.75 | -0.32 | -0.17 | 0.00→0.18 | 0.00→0.16 | >\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport[["]] content="wi ⏐  + \frac{r}{n}\right)^{nt}$$\nWhere:\n*   $A$ is the amount of[[ money]] accumulated |

**loop_selective**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 3673 | Mid-phrase token while restating earlier text | Y | 0.92 | +0.56 | +1.87 | 7.15→1.96 | 0.77→0.50 | et` as the bullet marker.\n    *   Wait, I'll use the text `*[[ This]] is an examp ⏐ et` as the bullet marker.\n    *   Wait, I'll use the text `*[[ This]] is an examp |
| 2293 | Subject token right before stating a computed value | N | 0.75 | +0.23 | +0.68 | 0.46→0.24 | 0.20→0.11 |     *   If he stops at 45, he traveled 45 miles.\n    *   The[[ distance]] between the ⏐     *   If he stops at 45, he traveled 45 miles.\n    *   The[[ distance]] between the |
| 11102 | Token right after "Wait," restating prior content | Y | 0.83 | +0.29 | +0.65 | 0.41→0.06 | 0.20→0.04 | ^2 \cdot 3 \cdot 5 = 60$.\n    $(60)^2 = 3600$.\n    Wait, $2^[[4]] \cdot 3^2 ⏐ ^2 \cdot 3 \cdot 5 = 60$.\n    $(60)^2 = 3600$.\n    Wait, $2^[[4]] \cdot 3^2 |
| 6592 | Noun in self-correction constraint-checking sentences | N | 0.58 | +0.49 | +0.59 | 0.64→0.01 | 0.28→0.01 | te safety guidelines).*\n    *I will not claim he is a circus[[ clown]] unless I ha ⏐ te safety guidelines).*\n    *I will not claim he is a circus[[ clown]] unless I ha |
| 1444 | Closing math "$" before repeated "Wait," line | Y | 0.83 | +0.35 | +0.48 | 0.61→0.39 | 0.17→0.12 | ^{42}$.\nWait, $z^{85} + z^{-85} = (z^2)^{42} + (z^{-2})^{42}[[$.]]\nWait, $z^{ ⏐ ^{42}$.\nWait, $z^{85} + z^{-85} = (z^2)^{42} + (z^{-2})^{42}[[$.]]\nWait, $z^{ |

**prod_selective**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 8733 | Noun after "imply/part of a" hedging | N | 0.83 | -0.49 | -0.88 | 0.01→1.50 | 0.00→0.75 | re a trick where "after a month" refers to the specific time[[ frame]] relative to ⏐ e interpreted as "less than" but maybe the wording implies a[[ specific]] relationshi |
| 10763 | Binary operator/equals before next operand number | N | 0.92 | -0.44 | -0.50 | 0.01→0.07 | 0.00→0.05 | $.\n\n3.  **Calculate Remaining Books:**\n    *   Top Shelf: $8[[ -]] 2 = 6$ ⏐ en $m = n$ if $a=b$. Here $a=2$ and $b=2$.\n        Thus, $2x[[ =]] 4$.\n    * |
| 9112 | Verb/noun in conditional "if subject X" restatement | Y | 0.58 | +0.31 | -0.48 | 0.10→0.03 | 0.05→0.01 | aining" pieces = pieces that were not placed.\n    *   If she[[ places]], they are p ⏐                                *   Wait, if he "donates", he[[ gives]] money away. |
| 5917 | Period ending problem-statement sentence in analysis | N | 0.75 | -0.35 | -0.48 | 0.01→0.60 | 0.00→0.43 | ultiply the number of roses per bush by the number of bushes[[.]]\n        *   ⏐ is asking for the value of $x$ given the equation $2^8 = 4^x[[$.]] They want a |
| 3792 | Number followed by parenthetical annotation or label | N | 0.83 | -0.06 | -0.47 | 0.06→0.20 | 0.05→0.14 | es 3)$.\n            *   Coeffs:\n                *   $2$: $1$[[ (]]from 10) + ⏐   *   Let's verify the counts.\n        *   Roller coaster: 2[[ (]]Pam) + 4 |

### pub18 (hidden_states[18], d_sae 65536, read against u15) cfg {}
max |cos(dec, u)| = 0.447; #features |cos|>0.3: 4; residual ‖u − proj on top-k greedy features‖ k=1..10: [0.89, 0.8, 0.75, 0.72, 0.7, 0.68, 0.66, 0.65, 0.64, 0.63]; cos(mean Δresid at loop pos, u) = -0.90; frac of top-20 edit-ON features in base top-100 productive: 0.25

**cos_plus_u**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 937 | Mid-clause verb/auxiliary in restated question | Y | 0.67 | +0.45 | +3.02 | 1.02→0.11 | 0.99→0.37 | ule that the objection is *valid* (sustained) if the defense[[ had]] a valid obj ⏐ ule that the objection is *valid* (sustained) if the defense[[ had]] a valid obj |
| 1498 | Inside conditional clause in reasoning bullets | N | 0.5 | +0.40 | +3.25 | 2.26→0.86 | 1.00→1.00 | he works for 40 years, she gets $50,000/year.\n    *   If she[[ quits]] after 30 ye ⏐ he works for 40 years, she gets $50,000/year.\n    *   If she[[ quits]] after 30 ye |
| 2005 | Start of new sentence's subject noun phrase | N | 0.42 | +0.27 | +1.10 | 2.65→1.91 | 1.00→1.00 |  , 2012 . It features guest star Chris Gethard as Trevor . \n[[ The]] series depi ⏐ the second ( Fortissimo ) was released on April 2 , 2008 . \n[[ The]] opening the |
| 1470 | Hyphen "@-@" token in Wikipedia compounds | N | 0.67 | +0.23 | -0.06 | 0.07→0.01 | 0.37→0.03 | s incorporated into the Gazette in December 1929 , the Sleaf[[ord]] Telegraph r ⏐ oks " in the " how @-@ to @-@ be @-@ the @-@ best @-@ you @-[[@]] can @-@ be  |
| 1539 | Final token before repeated phrase restarts | Y | 0.92 | +0.22 | +2.53 | 0.50→0.04 | 0.92→0.19 | boy #fashion #style #life\n        #playboy #fashion #style #[[life]]\n        #pl ⏐ boy #fashion #style #life\n        #playboy #fashion #style #[[life]]\n        #pl |
| 1699 | Function word mid-phrase when restating earlier text | Y | 0.5 | +0.20 | +3.27 | 0.97→0.34 | 0.99→0.88 |                         *   If we add them back, we have $F$[[ white]] flamingos.\n ⏐                         *   If we add them back, we have $F$[[ white]] flamingos.\n |
| 8948 | Mid-sentence tokens while repeating a "Wait" sentence | Y | 0.67 | +0.19 | +0.02 | 0.00→0.00 | 0.01→0.00 | \n    Wait, I need to check if I should include the blog post[[ title]].\n    No, ju ⏐    Wait, I should check if I need to mention that it's about[[ children]].\n    Okay.\n |
| 551 | Start of a restated or alternative problem quote | Y | 0.83 | +0.18 | +0.56 | 0.11→0.03 | 0.44→0.15 |                                                     *   Or "[[How]] many action ⏐          *   Pink = 240.\n                                *  [[ "]]There are 22 |

**cos_minus_u**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 12393 | Next item in a list or enumeration | N | 0.67 | -0.41 | -0.56 | 0.00→0.07 | 0.00→0.30 | 's check the options again.\n    *   Options: 10, 15, 20, 22,[[ ]]30, 35 ⏐ whaler , or New Zealand whaler , as well as by the shortened[[ "]] bronze " ,  |
| 3890 | Positions right before a content-listing start | N | 0.17 | -0.36 | -0.03 | 0.00→0.00 | 0.00→0.00 | commas.\n    *   Ensure each line ends properly.\n\n    *Line 1[[:*]] The sun ros ⏐  hashtags, starting with '#'<¦im_end¦>\n<¦im_start¦>assistant[[\n]]<think>\nThin |
| 24687 | Mid-token completion of predictable multi-token sequences | N | 0.75 | -0.23 | -0.01 | 0.00→0.00 | 0.00→0.00 | he code.\n</think>\n\n```html\n<!DOCTYPE html>\n<html lang="en">\n[[<head]]>\n    <meta  ⏐ as "an out-of-court statement offered to prove that a declar[[ant]] was told a  |
| 563 | Restating the problem's topic or text | Y | 0.5 | -0.20 | -0.47 | 0.02→0.15 | 0.08→0.58 | he Request:**\n    *   Topic: Advantages and disadvantages of[[ having]] supernatura ⏐ :\n\n1.  **Analyze the Request:**\n    *   Topic: Journal entry[[ about]] stress mana |
| 3584 | Mid-name token predicting continuation of proper noun | N | 0.58 | -0.17 | -0.07 | 0.00→0.00 | 0.00→0.00 | ot statistics and powers , was the game 's main attraction .[[ Tro]]ika tried to ⏐  Australia , the Netherlands , Ireland and New Zealand . The[[ music]] video for " |

**contrib_u_loop_minus_prod**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 1498 | Inside conditional clause in reasoning bullets | N | 0.5 | +0.40 | +3.25 | 2.26→0.86 | 1.00→1.00 | he works for 40 years, she gets $50,000/year.\n    *   If she[[ quits]] after 30 ye ⏐ he works for 40 years, she gets $50,000/year.\n    *   If she[[ quits]] after 30 ye |
| 937 | Mid-clause verb/auxiliary in restated question | Y | 0.67 | +0.45 | +3.02 | 1.02→0.11 | 0.99→0.37 | ule that the objection is *valid* (sustained) if the defense[[ had]] a valid obj ⏐ ule that the objection is *valid* (sustained) if the defense[[ had]] a valid obj |
| 1699 | Function word mid-phrase when restating earlier text | Y | 0.5 | +0.20 | +3.27 | 0.97→0.34 | 0.99→0.88 |                         *   If we add them back, we have $F$[[ white]] flamingos.\n ⏐                         *   If we add them back, we have $F$[[ white]] flamingos.\n |
| 2005 | Start of new sentence's subject noun phrase | N | 0.42 | +0.27 | +1.10 | 2.65→1.91 | 1.00→1.00 |  , 2012 . It features guest star Chris Gethard as Trevor . \n[[ The]] series depi ⏐ the second ( Fortissimo ) was released on April 2 , 2008 . \n[[ The]] opening the |
| 1539 | Final token before repeated phrase restarts | Y | 0.92 | +0.22 | +2.53 | 0.50→0.04 | 0.92→0.19 | boy #fashion #style #life\n        #playboy #fashion #style #[[life]]\n        #pl ⏐ boy #fashion #style #life\n        #playboy #fashion #style #[[life]]\n        #pl |
| 761 | Analytic commentary on a problem's phrasing or wording | N | 0.83 | -0.06 | -1.47 | 0.01→0.04 | 0.04→0.22 | ion aligns with the concept of **Service Perception**, which[[ focuses]] on how cust ⏐ t's injuries to the motorcyclist..."\n            *   This ph[[rasing]] "assigned 5 |
| 551 | Start of a restated or alternative problem quote | Y | 0.83 | +0.18 | +0.56 | 0.11→0.03 | 0.44→0.15 |                                                     *   Or "[[How]] many action ⏐          *   Pink = 240.\n                                *  [[ "]]There are 22 |
| 563 | Restating the problem's topic or text | Y | 0.5 | -0.20 | -0.47 | 0.02→0.15 | 0.08→0.58 | he Request:**\n    *   Topic: Advantages and disadvantages of[[ having]] supernatura ⏐ :\n\n1.  **Analyze the Request:**\n    *   Topic: Journal entry[[ about]] stress mana |

**edit_off_loop**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 1498 | Inside conditional clause in reasoning bullets | N | 0.5 | +0.40 | +3.25 | 2.26→0.86 | 1.00→1.00 | he works for 40 years, she gets $50,000/year.\n    *   If she[[ quits]] after 30 ye ⏐ he works for 40 years, she gets $50,000/year.\n    *   If she[[ quits]] after 30 ye |
| 937 | Mid-clause verb/auxiliary in restated question | Y | 0.67 | +0.45 | +3.02 | 1.02→0.11 | 0.99→0.37 | ule that the objection is *valid* (sustained) if the defense[[ had]] a valid obj ⏐ ule that the objection is *valid* (sustained) if the defense[[ had]] a valid obj |
| 2005 | Start of new sentence's subject noun phrase | N | 0.42 | +0.27 | +1.10 | 2.65→1.91 | 1.00→1.00 |  , 2012 . It features guest star Chris Gethard as Trevor . \n[[ The]] series depi ⏐ the second ( Fortissimo ) was released on April 2 , 2008 . \n[[ The]] opening the |
| 1699 | Function word mid-phrase when restating earlier text | Y | 0.5 | +0.20 | +3.27 | 0.97→0.34 | 0.99→0.88 |                         *   If we add them back, we have $F$[[ white]] flamingos.\n ⏐                         *   If we add them back, we have $F$[[ white]] flamingos.\n |
| 1539 | Final token before repeated phrase restarts | Y | 0.92 | +0.22 | +2.53 | 0.50→0.04 | 0.92→0.19 | boy #fashion #style #life\n        #playboy #fashion #style #[[life]]\n        #pl ⏐ boy #fashion #style #life\n        #playboy #fashion #style #[[life]]\n        #pl |
| 1969 | Informal quoted speech or lyrics, mid-utterance | N | 0.83 | +0.14 | +0.12 | 0.24→0.02 | 0.77→0.07 | rocket flying into a wall. Did it break?"\n    *   "It didn't[[.]] It exploded ⏐ riving home. Saw the car. I didn't steal it.\n        But the[[ cops]] found out.\n |
| 10806 | Subject position after "Wait," self-check clause | Y | 0.83 | +0.04 | +2.15 | 0.20→0.00 | 0.80→0.03 | 'll proceed.\n\n        Wait, I need to check if the separator[[ should]] be between  ⏐ 'll proceed.\n\n        Wait, I need to check if the separator[[ should]] be between  |
| 591 | Number-expected token during self-doubt rechecking | Y | 0.83 | +0.09 | +0.70 | 0.15→0.04 | 0.41→0.17 |  $\log_264$."\n    Wait, is it possible the user meant $\log_[[2]] 64$?\n ⏐  $\log_264$."\n    Wait, is it possible the user meant $\log_[[2]] 64$?\n |
| 551 | Start of a restated or alternative problem quote | Y | 0.83 | +0.18 | +0.56 | 0.11→0.03 | 0.44→0.15 |                                                     *   Or "[[How]] many action ⏐          *   Pink = 240.\n                                *  [[ "]]There are 22 |
| 1418 | Index/exponent separator in structured math sequences | Y | 0.58 | +0.09 | +0.29 | 0.11→0.04 | 0.32→0.14 | 1,1), (2,1)$.\n                        *   Line 6: $(0,2), (1[[,]]2), (2,2 ⏐ : $(0,0), (0,1), (0,2)$; $(1,0), (1,1), (1,2)$; $(2,0), (2[[,]]1), (2,2 |

**edit_on_loop**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 941 | Subject noun right before main verb | N | 0.75 | -0.12 | -0.24 | 0.00→0.18 | 0.00→0.65 | ic, manage user sessions, and test security protocols.*\nThis[[ task]] involves cr ⏐      <h3>Executive Summary</h3>\n        <p>\n            This[[ proposal]] outlines a  |
| 820 | Mid-sentence content word before a continuing phrase | N | 0.75 | -0.07 | +0.11 | 0.04→0.20 | 0.27→0.87 | es there and unroofed a further 75 , including damage to the[[ hospital]] . There wer ⏐ ier Reef , the hurricane downed palm trees and produced high[[ waves]] , with sign |
| 563 | Restating the problem's topic or text | Y | 0.5 | -0.20 | -0.47 | 0.02→0.15 | 0.08→0.58 | he Request:**\n    *   Topic: Advantages and disadvantages of[[ having]] supernatura ⏐ :\n\n1.  **Analyze the Request:**\n    *   Topic: Journal entry[[ about]] stress mana |
| 2022 | Anticipating a standard math concept/method | N | 0.75 | -0.04 | -0.34 | 0.03→0.14 | 0.15→0.65 | s a function of $r$ first). Wait, let's look at the standard[[ calculus]] approach fo ⏐ ree sides, it's an arbitrary shape (triangle).\n\n3.  **Recall[[ Relevant]] Mathematica |
| 1095 | Demonstrative/determiner starting a generalization about problems | N | 0.5 | -0.08 | -0.10 | 0.03→0.13 | 0.10→0.58 | nce between $(x, y)$ and $(40, 50)$.\nNow, what are the first[[ two]] terms?\nThey ⏐ , this implies finding $¦R - D¦$.\n    *   However, sometimes[[ these]] problems im |
| 1813 | Terse noun in bulleted planning/checklist items | N | 0.67 | -0.07 | -0.56 | 0.03→0.12 | 0.15→0.53 | omers).\n    *   Eliminate A (Diligence, Self-interest - Self[[-interest]] is wrong).\n ⏐ ting with Ethical: A, C (if Cultural isn't it), E (Financial[[ is]] different). |
| 2015 | Introducing an example or illustration | N | 0.33 | -0.03 | -0.05 | 0.01→0.10 | 0.05→0.50 | amples. Words might have changed meaning over centuries. For[[ instance]], the word ' ⏐ r. His leadership style was characterized by strategic fores[[ight]] and discipl |
| 1461 | Verb/connective right after subject, predicting "to" | N | 0.75 | -0.07 | -0.34 | 0.00→0.09 | 0.02→0.47 | $ of a person's daily caloric requirement, how many calories[[ fulfill]] a person's  ⏐ ll most probably hold in favor of\nOptions:\nA. the creditors,[[ because]] the agreeme |
| 1944 | Auxiliary/"already" signaling completed state | N | 0.75 | -0.03 | +0.01 | 0.01→0.09 | 0.06→0.48 | the time Timberlake was in the vocal booth , the basic track[[ was]] done , then ⏐ graded equipment , and installed new instruments . Once work[[ was]] completed , |
| 1982 | Generic/typical-use qualifiers before a specific noun | N | 0.83 | -0.04 | -0.31 | 0.00→0.08 | 0.03→0.48 |  Sometimes it leads to confusion.\n            *   People may[[ prefer]] single name ⏐ meworks that simplify development tasks.\n*   **Simplicity:**[[ For]] beginners,  |

**cos_dresid**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 3890 | Positions right before a content-listing start | N | 0.17 | -0.36 | -0.03 | 0.00→0.00 | 0.00→0.00 | commas.\n    *   Ensure each line ends properly.\n\n    *Line 1[[:*]] The sun ros ⏐  hashtags, starting with '#'<¦im_end¦>\n<¦im_start¦>assistant[[\n]]<think>\nThin |
| 12393 | Next item in a list or enumeration | N | 0.67 | -0.41 | -0.56 | 0.00→0.07 | 0.00→0.30 | 's check the options again.\n    *   Options: 10, 15, 20, 22,[[ ]]30, 35 ⏐ whaler , or New Zealand whaler , as well as by the shortened[[ "]] bronze " ,  |
| 24687 | Mid-token completion of predictable multi-token sequences | N | 0.75 | -0.23 | -0.01 | 0.00→0.00 | 0.00→0.00 | he code.\n</think>\n\n```html\n<!DOCTYPE html>\n<html lang="en">\n[[<head]]>\n    <meta  ⏐ as "an out-of-court statement offered to prove that a declar[[ant]] was told a  |
| 3584 | Mid-name token predicting continuation of proper noun | N | 0.58 | -0.17 | -0.07 | 0.00→0.00 | 0.00→0.00 | ot statistics and powers , was the game 's main attraction .[[ Tro]]ika tried to ⏐  Australia , the Netherlands , Ireland and New Zealand . The[[ music]] video for " |
| 62474 | Newline after sentence in Wikipedia narrative | N | None | -0.13 | +0.00 | 0.00→0.00 | 0.00→0.00 | rs , and a minute later , the order to open fire was given .[[ \n]] Markgraf op |

**loop_selective**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 1699 | Function word mid-phrase when restating earlier text | Y | 0.5 | +0.20 | +3.27 | 0.97→0.34 | 0.99→0.88 |                         *   If we add them back, we have $F$[[ white]] flamingos.\n ⏐                         *   If we add them back, we have $F$[[ white]] flamingos.\n |
| 1498 | Inside conditional clause in reasoning bullets | N | 0.5 | +0.40 | +3.25 | 2.26→0.86 | 1.00→1.00 | he works for 40 years, she gets $50,000/year.\n    *   If she[[ quits]] after 30 ye ⏐ he works for 40 years, she gets $50,000/year.\n    *   If she[[ quits]] after 30 ye |
| 937 | Mid-clause verb/auxiliary in restated question | Y | 0.67 | +0.45 | +3.02 | 1.02→0.11 | 0.99→0.37 | ule that the objection is *valid* (sustained) if the defense[[ had]] a valid obj ⏐ ule that the objection is *valid* (sustained) if the defense[[ had]] a valid obj |
| 1539 | Final token before repeated phrase restarts | Y | 0.92 | +0.22 | +2.53 | 0.50→0.04 | 0.92→0.19 | boy #fashion #style #life\n        #playboy #fashion #style #[[life]]\n        #pl ⏐ boy #fashion #style #life\n        #playboy #fashion #style #[[life]]\n        #pl |
| 10806 | Subject position after "Wait," self-check clause | Y | 0.83 | +0.04 | +2.15 | 0.20→0.00 | 0.80→0.03 | 'll proceed.\n\n        Wait, I need to check if the separator[[ should]] be between  ⏐ 'll proceed.\n\n        Wait, I need to check if the separator[[ should]] be between  |

**prod_selective**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 761 | Analytic commentary on a problem's phrasing or wording | N | 0.83 | -0.06 | -1.47 | 0.01→0.04 | 0.04→0.22 | ion aligns with the concept of **Service Perception**, which[[ focuses]] on how cust ⏐ t's injuries to the motorcyclist..."\n            *   This ph[[rasing]] "assigned 5 |
| 770 | Mid-phrase token before a short commentary continuation | N | 0.67 | -0.02 | -1.41 | 0.01→0.04 | 0.05→0.21 |   "shadows stretched long across floor."\n    *   "lights off[[ inside]] bunker."\n   ⏐ e now.\n\n    *   Draft 1 (Internal):\n        hey guys what is[[ happening]] today? #pla |
| 1034 | Subject noun after determiner, predicate next | N | 0.42 | -0.05 | -1.06 | 0.01→0.02 | 0.04→0.14 | \n    We have to find a friend to share the bread,\n    With a[[ brother]] who loves t ⏐ gal mechanism.\n\n    Wait, I found a source that says "If the[[ prosecution]] presents ev |
| 58808 | Referring to the problem statement's wording | N | 0.67 | -0.00 | -0.97 | 0.01→0.01 | 0.03→0.04 |   Maybe if she buys 400 cartons? No.\n\n    *   Okay, the math[[ is]] clear.\n\n    ⏐                                    *   Let's restart the RRT[[ analysis]] for $x^2 - |
| 6262 | Noun/quantity before a coordinating "AND" clause | N | 0.83 | +0.11 | -0.90 | 0.01→0.00 | 0.03→0.01 | could mean Kael brought apples equal to twice what Morisette[[ brought]], and then * ⏐ s the set of students who did *not* participate in MATHCOUNT[[S]] and did *no |

### ours20 (hidden_states[20], d_sae 16384, read against u19) cfg {"layer": 20, "d_sae": 16384, "k": 64, "fvu": 0.1334768019059227, "l0": 58.35170675783741, "dead_frac": 0.0, "n_train_tokens": 17400499, "steps": 4248, "n_eval": 400000, "scale": 0.17770493030548096}
max |cos(dec, u)| = 0.648; #features |cos|>0.3: 6; residual ‖u − proj on top-k greedy features‖ k=1..10: [0.76, 0.69, 0.67, 0.66, 0.64, 0.63, 0.62, 0.61, 0.6, 0.6]; cos(mean Δresid at loop pos, u) = -0.95; frac of top-20 edit-ON features in base top-100 productive: 0.45

**cos_plus_u**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 6329 | Hesitant self-doubt restatement before re-checking | Y | 0.83 | +0.59 | +1.52 | 3.90→0.03 | 0.82→0.02 | cles]".\n    *   Yes.\n    *   Could it be "2 containers" = "2[[ containers]] [of importe ⏐ cles]".\n    *   Yes.\n    *   Could it be "2 containers" = "2[[ containers]] [of importe |
| 12884 | Mid-clause in conditional "if X, he is..." restatement | Y | 0.83 | +0.58 | +3.26 | 8.42→0.09 | 0.95→0.07 | ng *towards* home?\n            *   Wait, if he drives *away*[[ from]] home, he is ⏐ ng *towards* home?\n            *   Wait, if he drives *away*[[ from]] home, he is |
| 12572 | Next token copies earlier math expression | Y | 0.75 | +0.42 | +0.50 | 1.15→0.01 | 0.26→0.01 | et's denote $P_2(0, y)$.\nThen distance $P_2 Z = \sqrt{y^2 + [[3]]0^2}$.\n ⏐ et's denote $P_2(0, y)$.\nThen distance $P_2 Z = \sqrt{y^2 + [[3]]0^2}$.\n |
| 13205 | Newline after a "Wait, if..." self-doubt sentence | Y | 0.67 | +0.39 | -0.38 | 0.11→0.03 | 0.04→0.01 | .\nWait, if the graph is $y = 2\sin(3x+\pi)+1$, then $c=\pi$.[[\n]]But why woul ⏐ ?\nWait, if the graph is $y = 2\sin(3x+\pi)+1$, then $c=\pi$.[[\n]]But maybe th |
| 14345 | Noun in quoted/restated problem text | Y | 0.67 | +0.28 | +0.49 | 0.88→0.11 | 0.28→0.07 | \n    So I'm safe.\n\n    Let's go.\n</think>\n\nWrite an acoustic[[ song]] about the K ⏐  Their main objective is to examine the shifting patterns of[[ structure]] over time.  |
| 9457 | Start of a parallel "If..." conditional clause | Y | 0.83 | +0.26 | +0.03 | 0.03→0.00 | 0.02→0.00 |  he stands still for 2 hours, he is at home.\n            *  [[ If]] he wants to ⏐ t the destination, he is at the destination.\n            *  [[ If]] he wants to |
| 15190 | Newline ending conclusion before "Wait" self-correction | Y | 0.58 | +0.26 | -0.01 | 0.04→0.02 | 0.02→0.01 | ipse is narrow in the x-direction.\nSo $x$ is the minor axis.[[\n]]Wait, so $a$ ⏐ se is narrower in the x-direction.\nSo $x$ is bounded by $a$.[[\n]]So the ellip |
| 13840 | Space before a repeated number | Y | 0.83 | +0.23 | +0.29 | 0.14→0.00 | 0.06→0.00 | 140 * 6 = $840.\n                            *   Wait, $140 *[[ ]]6 = $840 ⏐ 140 * 6 = $840.\n                            *   Wait, $140 *[[ ]]6 = $840 |

**cos_minus_u**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 2458 | Mid-expression math/LaTeX token before operand | N | 0.42 | -0.65 | -1.18 | 0.10→4.70 | 0.02→0.95 |  **Formulate Equations:**\n    *   Volume of a sphere: $V = \[[frac]]{4}{3}\pi ⏐ + 5¦ = 5.25$.\n                        *   $¦3(1/4 - 2)¦ = ¦3[[(-]]7/4)¦ = ¦- |
| 8733 | Interpreting word-problem phrasing as implying relationships | N | 0.75 | -0.36 | -0.69 | 0.00→0.70 | 0.00→0.49 | ens"?\n            *   Actually, usually these problems imply[[ specific]] relationshi ⏐ *   Let's check if the problem implies a set of equations or[[ just]] relative co |
| 189 | Abstract noun concluding a relation-type judgment | N | 0.83 | -0.28 | -0.37 | 0.00→0.05 | 0.00→0.04 | confirms the $732 is part of the $366,000.\nSo it is a direct[[ ratio]].\n\nCalculati ⏐  is it the number of bottles?\n        *   Regardless, it's a[[ multiplier]].\n        *  |
| 10632 | Announcing one more verification check | Y | 0.83 | -0.24 | -0.28 | 0.00→0.18 | 0.00→0.14 | likely intended to be solved correctly as $x=2$.\n\n    Let me[[ just]] pause and t ⏐  is correct.\n    My derivation $x=2$ is correct.\n\n    Let me[[ just]] pause and t |
| 8127 | Newline before indented bullet in reasoning re-check | Y | 0.83 | -0.22 | -0.18 | 0.00→0.00 | 0.00→0.00 | luate the standard BCG classification terms vs. the options.[[\n]]    *   Stan ⏐  6... Then he sees 2." This is a standard math word problem.[[\n]]    *   Let  |

**contrib_u_loop_minus_prod**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 12884 | Mid-clause in conditional "if X, he is..." restatement | Y | 0.83 | +0.58 | +3.26 | 8.42→0.09 | 0.95→0.07 | ng *towards* home?\n            *   Wait, if he drives *away*[[ from]] home, he is ⏐ ng *towards* home?\n            *   Wait, if he drives *away*[[ from]] home, he is |
| 2458 | Mid-expression math/LaTeX token before operand | N | 0.42 | -0.65 | -1.18 | 0.10→4.70 | 0.02→0.95 |  **Formulate Equations:**\n    *   Volume of a sphere: $V = \[[frac]]{4}{3}\pi ⏐ + 5¦ = 5.25$.\n                        *   $¦3(1/4 - 2)¦ = ¦3[[(-]]7/4)¦ = ¦- |
| 6329 | Hesitant self-doubt restatement before re-checking | Y | 0.83 | +0.59 | +1.52 | 3.90→0.03 | 0.82→0.02 | cles]".\n    *   Yes.\n    *   Could it be "2 containers" = "2[[ containers]] [of importe ⏐ cles]".\n    *   Yes.\n    *   Could it be "2 containers" = "2[[ containers]] [of importe |
| 12572 | Next token copies earlier math expression | Y | 0.75 | +0.42 | +0.50 | 1.15→0.01 | 0.26→0.01 | et's denote $P_2(0, y)$.\nThen distance $P_2 Z = \sqrt{y^2 + [[3]]0^2}$.\n ⏐ et's denote $P_2(0, y)$.\nThen distance $P_2 Z = \sqrt{y^2 + [[3]]0^2}$.\n |
| 8733 | Interpreting word-problem phrasing as implying relationships | N | 0.75 | -0.36 | -0.69 | 0.00→0.70 | 0.00→0.49 | ens"?\n            *   Actually, usually these problems imply[[ specific]] relationshi ⏐ *   Let's check if the problem implies a set of equations or[[ just]] relative co |
| 14345 | Noun in quoted/restated problem text | Y | 0.67 | +0.28 | +0.49 | 0.88→0.11 | 0.28→0.07 | \n    So I'm safe.\n\n    Let's go.\n</think>\n\nWrite an acoustic[[ song]] about the K ⏐  Their main objective is to examine the shifting patterns of[[ structure]] over time.  |
| 16073 | Whitespace/newline right before repeated "Wait" | Y | 0.75 | +0.16 | +0.37 | 0.88→0.41 | 0.14→0.08 | Wait, $x^2 - 6x + 5 = 0$.\n        Wait, $x^2 - 6x + 5 = 0$.\n[[       ]] Wait, $x^2 ⏐ ss expensive.\n    Wait, $O$ is less.\n    Wait, $O$ is less.\n[[   ]] Wait, $O$ i |
| 189 | Abstract noun concluding a relation-type judgment | N | 0.83 | -0.28 | -0.37 | 0.00→0.05 | 0.00→0.04 | confirms the $732 is part of the $366,000.\nSo it is a direct[[ ratio]].\n\nCalculati ⏐  is it the number of bottles?\n        *   Regardless, it's a[[ multiplier]].\n        *  |

**edit_off_loop**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 12884 | Mid-clause in conditional "if X, he is..." restatement | Y | 0.83 | +0.58 | +3.26 | 8.42→0.09 | 0.95→0.07 | ng *towards* home?\n            *   Wait, if he drives *away*[[ from]] home, he is ⏐ ng *towards* home?\n            *   Wait, if he drives *away*[[ from]] home, he is |
| 6329 | Hesitant self-doubt restatement before re-checking | Y | 0.83 | +0.59 | +1.52 | 3.90→0.03 | 0.82→0.02 | cles]".\n    *   Yes.\n    *   Could it be "2 containers" = "2[[ containers]] [of importe ⏐ cles]".\n    *   Yes.\n    *   Could it be "2 containers" = "2[[ containers]] [of importe |
| 12572 | Next token copies earlier math expression | Y | 0.75 | +0.42 | +0.50 | 1.15→0.01 | 0.26→0.01 | et's denote $P_2(0, y)$.\nThen distance $P_2 Z = \sqrt{y^2 + [[3]]0^2}$.\n ⏐ et's denote $P_2(0, y)$.\nThen distance $P_2 Z = \sqrt{y^2 + [[3]]0^2}$.\n |
| 14345 | Noun in quoted/restated problem text | Y | 0.67 | +0.28 | +0.49 | 0.88→0.11 | 0.28→0.07 | \n    So I'm safe.\n\n    Let's go.\n</think>\n\nWrite an acoustic[[ song]] about the K ⏐  Their main objective is to examine the shifting patterns of[[ structure]] over time.  |
| 16073 | Whitespace/newline right before repeated "Wait" | Y | 0.75 | +0.16 | +0.37 | 0.88→0.41 | 0.14→0.08 | Wait, $x^2 - 6x + 5 = 0$.\n        Wait, $x^2 - 6x + 5 = 0$.\n[[       ]] Wait, $x^2 ⏐ ss expensive.\n    Wait, $O$ is less.\n    Wait, $O$ is less.\n[[   ]] Wait, $O$ i |
| 7922 | Mid-expression token in repeated "Wait," restatement | Y | 0.83 | +0.22 | +0.50 | 0.36→0.00 | 0.24→0.00 | 2x + 24 = 0$ has no solution.\n    Wait, $15x^2 - 12x + 24 = [[0]]$ has no sol ⏐ 2x + 24 = 0$ has no solution.\n    Wait, $15x^2 - 12x + 24 = [[0]]$ has no sol |
| 11529 | Start of next parallel enumerated statement | Y | 0.75 | +0.14 | +0.64 | 0.30→0.00 | 0.19→0.00 | is:\n    Amy is 5 years older than Jackson.\n    James is 10.\n[[   ]] James is 1  ⏐ \n                                                        *  [[ If]] she coaches |
| 6680 | Newline/indent before another "Wait," restart | Y | 0.75 | +0.14 | +0.39 | 0.40→0.13 | 0.14→0.05 | $.\nSo if $a < b$, then $a$ is the semi-minor axis.\nWait, no.[[\n]]Let's check  ⏐ $.\nSo if $a < b$, then $a$ is the semi-minor axis.\nWait, no.[[\n]]Let's check  |
| 15922 | Copying a token mid-expression during restatement | Y | 0.67 | +0.02 | +0.52 | 0.35→0.12 | 0.17→0.07 | s of the faces are $12 \times 8$, $12 \times 10$, $10 \times[[ ]]8$.\n         ⏐ s of the faces are $12 \times 8$, $12 \times 10$, $10 \times[[ ]]8$.\n         |
| 6870 | Space before inline math variable | Y | 0.75 | +0.15 | +0.35 | 0.20→0.02 | 0.09→0.01 |  question.\n    User wrote: "Solve the following equation for[[ $]]x$: [equatio ⏐ for any hidden tricks.\n    "Solve the following equation for[[ $]]x$"\n    "rea |

**edit_on_loop**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 2458 | Mid-expression math/LaTeX token before operand | N | 0.42 | -0.65 | -1.18 | 0.10→4.70 | 0.02→0.95 |  **Formulate Equations:**\n    *   Volume of a sphere: $V = \[[frac]]{4}{3}\pi ⏐ + 5¦ = 5.25$.\n                        *   $¦3(1/4 - 2)¦ = ¦3[[(-]]7/4)¦ = ¦- |
| 2210 | Function word early in a question or prompt | N | 0.83 | -0.10 | -0.14 | 0.00→2.81 | 0.00→0.93 |  is (X)" where X is the correct letter choice.\n\nQuestion: If[[ the]] spot rates  ⏐ e correct letter choice.\n\nQuestion:  Some of key differences[[ between]] Islamic fin |
| 1376 | Hyphen inside Wikipedia "@-@" compound words | N | 0.83 | -0.14 | -0.26 | 0.00→1.37 | 0.00→0.82 | ance of preserving archaeology , making many monuments out @[[-]]@ of @-@ bou ⏐ oads , and becomes the new king of the now blossoming Lost @[[-]]@ Hope . \n B |
| 15418 | Digits of final boxed numeric answer | Y | 0.92 | -0.12 | -0.18 | 0.01→0.93 | 0.01→0.61 | rds.\n*   Difference = $2040 - 900 = 1140$ yards.\n\n\boxed{114[[0]]} ⏐ feet}$\n    *   Total length = $280 \text{ feet}$.\n\n\boxed{28[[0]]} |
| 9072 | Predicting "answer/result" noun after determiner | N | 0.92 | -0.15 | -0.29 | 0.01→0.84 | 0.01→0.57 |  are two interpretations of the problem statement leading to[[ different]] answers.\n   ⏐ based on the fraction.\n    3. Compute the difference.\n    4.[[ Box]] the answer. |
| 1602 | Counting quantity of countable items next | N | 0.67 | -0.11 | -0.06 | 0.23→0.97 | 0.06→0.56 | le of 30 items.\n    And each item is in a container.\n    How[[ many]] containers? ⏐ l?"\n        If I buy a uniform, I pay $150.\n        If I buy[[ multiple]], I pay more |
| 8733 | Interpreting word-problem phrasing as implying relationships | N | 0.75 | -0.36 | -0.69 | 0.00→0.70 | 0.00→0.49 | ens"?\n            *   Actually, usually these problems imply[[ specific]] relationshi ⏐ *   Let's check if the problem implies a set of equations or[[ just]] relative co |
| 15425 | Label noun right before colon in list | N | 0.75 | -0.07 | -0.30 | 0.04→0.73 | 0.02→0.52 | lops small leaks that increase over time.\n    *   **First 15[[ years]]:** Zero los ⏐ $40$.\n    Now calculate the cost for each item.\n    Watermel[[ons]]: $4 \text{ |
| 7480 | Planning the final answer's presentation, mid-phrase | N | 0.83 | -0.13 | -0.14 | 0.00→0.63 | 0.00→0.45 | n the relationships described.\n    *   Solve for the unknown[[s]] step-by-ste ⏐ alculate the cost of each component (hat, jacket, and pants)[[ individually]] and then su |
| 10915 | Final term of distributed polynomial multiplication | N | 0.83 | +0.01 | -0.26 | 0.02→0.60 | 0.02→0.47 |  $-1(2x^2) = -2x^2$.\n        *   $-1(x) = -x$.\n        *   $[[-]]1(-1) =  ⏐ (4x^2) = -4x^2$.\n            $-1(-21x) = 21x$.\n            $[[-]]1(17) = |

**cos_dresid**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 2458 | Mid-expression math/LaTeX token before operand | N | 0.42 | -0.65 | -1.18 | 0.10→4.70 | 0.02→0.95 |  **Formulate Equations:**\n    *   Volume of a sphere: $V = \[[frac]]{4}{3}\pi ⏐ + 5¦ = 5.25$.\n                        *   $¦3(1/4 - 2)¦ = ¦3[[(-]]7/4)¦ = ¦- |
| 8733 | Interpreting word-problem phrasing as implying relationships | N | 0.75 | -0.36 | -0.69 | 0.00→0.70 | 0.00→0.49 | ens"?\n            *   Actually, usually these problems imply[[ specific]] relationshi ⏐ *   Let's check if the problem implies a set of equations or[[ just]] relative co |
| 189 | Abstract noun concluding a relation-type judgment | N | 0.83 | -0.28 | -0.37 | 0.00→0.05 | 0.00→0.04 | confirms the $732 is part of the $366,000.\nSo it is a direct[[ ratio]].\n\nCalculati ⏐  is it the number of bottles?\n        *   Regardless, it's a[[ multiplier]].\n        *  |
| 8127 | Newline before indented bullet in reasoning re-check | Y | 0.83 | -0.22 | -0.18 | 0.00→0.00 | 0.00→0.00 | luate the standard BCG classification terms vs. the options.[[\n]]    *   Stan ⏐  6... Then he sees 2." This is a standard math word problem.[[\n]]    *   Let  |
| 10632 | Announcing one more verification check | Y | 0.83 | -0.24 | -0.28 | 0.00→0.18 | 0.00→0.14 | likely intended to be solved correctly as $x=2$.\n\n    Let me[[ just]] pause and t ⏐  is correct.\n    My derivation $x=2$ is correct.\n\n    Let me[[ just]] pause and t |

**loop_selective**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 12884 | Mid-clause in conditional "if X, he is..." restatement | Y | 0.83 | +0.58 | +3.26 | 8.42→0.09 | 0.95→0.07 | ng *towards* home?\n            *   Wait, if he drives *away*[[ from]] home, he is ⏐ ng *towards* home?\n            *   Wait, if he drives *away*[[ from]] home, he is |
| 6329 | Hesitant self-doubt restatement before re-checking | Y | 0.83 | +0.59 | +1.52 | 3.90→0.03 | 0.82→0.02 | cles]".\n    *   Yes.\n    *   Could it be "2 containers" = "2[[ containers]] [of importe ⏐ cles]".\n    *   Yes.\n    *   Could it be "2 containers" = "2[[ containers]] [of importe |
| 12222 | Mid-sentence in self-doubt "Wait" reconsideration | Y | 0.67 | +0.06 | +0.64 | 0.26→0.14 | 0.15→0.08 |                    *   Wait, $M_1$ is $(0.75, 0.433)$. $M_3$[[ is]] $(-0.75 ⏐                    *   Wait, $M_1$ is $(0.75, 0.433)$. $M_3$[[ is]] $(-0.75 |
| 11529 | Start of next parallel enumerated statement | Y | 0.75 | +0.14 | +0.64 | 0.30→0.00 | 0.19→0.00 | is:\n    Amy is 5 years older than Jackson.\n    James is 10.\n[[   ]] James is 1  ⏐ \n                                                        *  [[ If]] she coaches |
| 15922 | Copying a token mid-expression during restatement | Y | 0.67 | +0.02 | +0.52 | 0.35→0.12 | 0.17→0.07 | s of the faces are $12 \times 8$, $12 \times 10$, $10 \times[[ ]]8$.\n         ⏐ s of the faces are $12 \times 8$, $12 \times 10$, $10 \times[[ ]]8$.\n         |

**prod_selective**

| feat | label (sonnet) | rep? | det. bal.acc | cos u | d′ loop-vs-prod | mean act loop base→edit | freq loop base→edit | top contexts |
|---|---|---|---|---|---|---|---|---|
| 2458 | Mid-expression math/LaTeX token before operand | N | 0.42 | -0.65 | -1.18 | 0.10→4.70 | 0.02→0.95 |  **Formulate Equations:**\n    *   Volume of a sphere: $V = \[[frac]]{4}{3}\pi ⏐ + 5¦ = 5.25$.\n                        *   $¦3(1/4 - 2)¦ = ¦3[[(-]]7/4)¦ = ¦- |
| 8733 | Interpreting word-problem phrasing as implying relationships | N | 0.75 | -0.36 | -0.69 | 0.00→0.70 | 0.00→0.49 | ens"?\n            *   Actually, usually these problems imply[[ specific]] relationshi ⏐ *   Let's check if the problem implies a set of equations or[[ just]] relative co |
| 11435 | End of first alternative, predicting "or" | N | 0.83 | -0.10 | -0.55 | 0.01→0.49 | 0.01→0.39 | ther you are looking for a place to rest, a place to explore[[,]] or a place  ⏐  years ($n$):** $17$\n*   **Annual interest rate ($r$):** $5\[[%]]$ or $0.0 |
| 9081 | Second item in parallel contrasting clause | Y | 0.67 | +0.03 | -0.51 | 0.04→0.03 | 0.03→0.02 | retation A: The 5 Democrats must form a line of 5, and the 5[[ Republicans]] must form a ⏐ form a contiguous block of 5 people in a row, and the set of[[ Republicans]] must form a |
| 12311 | Number or quantity about to be stated, in final answer | Y | 0.83 | -0.11 | -0.45 | 0.01→0.06 | 0.00→0.06 | easoning:\n\n1.  **Identify the starting amount:** Jean starts[[ with]] **30** loll ⏐ l Income**: The total income from Mr. Reynolds' Auto Service[[ was]] $366,0 |

### Prediction tallies
- ours16: P4a max|cos|=0.56 (<0.5?) ; rep-labelled in top-5 +cos: 4/5 ; P4b rep-labelled in top-10 OFF: 8/10 ; P4c top-20 ON in prod-top100: 0.25
- pub18: P4a max|cos|=0.45 (<0.5?) ; rep-labelled in top-5 +cos: 2/5 ; P4b rep-labelled in top-10 OFF: 7/10 ; P4c top-20 ON in prod-top100: 0.25
- ours20: P4a max|cos|=0.65 (<0.5?) ; rep-labelled in top-5 +cos: 5/5 ; P4b rep-labelled in top-10 OFF: 10/10 ; P4c top-20 ON in prod-top100: 0.45

Sonnet detection balanced accuracy: mean 0.73, median 0.75, n=118

### Opus spot-check vs Sonnet

| feature | sonnet label | opus label | rep? S/O | bal.acc S/O |
|---|---|---|---|---|
| pub18:1539 | Final token before repeated phrase restarts | End of repeated unit in verbatim loop | Y/Y | 0.92/0.83 |
| ours20:10915 | Final term of distributed polynomial multiplication | Final term in term-by-term expansion | N/N | 0.83/0.75 |
| ours20:9081 | Second item in parallel contrasting clause | Second item in parallel contrastive structure | Y/N | 0.67/0.75 |
| ours16:1375 | Newline after a stuck or contradictory reasoning step | Line break before a reasoning pivot | Y/N | 1.0/1.0 |
| ours20:13205 | Newline after a "Wait, if..." self-doubt sentence | Line break after "Wait, if…" restatement | Y/Y | 0.67/1.0 |
| pub18:1813 | Terse noun in bulleted planning/checklist items | Mid-phrase inside terse, parallel bullet-list items | N/N | 0.67/0.75 |
| pub18:8948 | Mid-sentence tokens while repeating a "Wait" sentence | Tokens inside verbatim-repeated or re-quoted earlier text | Y/Y | 0.67/0.67 |
| ours16:8733 | Noun after "imply/part of a" hedging | Hypothesized hidden structure in problem wording | N/N | 0.83/0.75 |
| ours16:12009 | Single-digit numeral token in reasoning | Single-digit number tokens in math reasoning | Y/N | 1.0/1.0 |
| ours16:9790 | Mid-phrase token in memorized fixed phrase | Mid-span of memorized fixed phrases | N/N | 0.75/0.67 |
| pub18:937 | Mid-clause verb/auxiliary in restated question |  | Y/N | 0.67/1.0 |
| pub18:3890 | Positions right before a content-listing start | Lead-in before an elaborating description or list | N/N | 0.17/0.17 |
| ours16:9008 | Content word completing a short simple sentence | Content words ending short drafted filler sentences | N/N | 0.83/0.75 |
| pub18:2920 | Mid-reasoning math tokens before elaboration | Clause boundary right after inline math expression | N/N | 0.58/0.67 |
| pub18:12393 | Next item in a list or enumeration | Predicting next item in parallel enumeration | N/N | 0.67/0.67 |
| ours16:6592 | Noun in self-correction constraint-checking sentences | Key noun under scrutiny in self-checking clauses | N/Y | 0.58/0.75 |
| ours20:2677 | Sentence-ending punctuation in reasoning deliberation | End of tentative interpretation or method proposal | N/N | 0.83/0.92 |
| ours16:11268 | Plural subject just before predicate verb | Plural reference to previously listed items | N/N | 1.0/1.0 |
| ours16:1011 | Final summation step after listed computations | Final aggregation step after listed sub-results | Y/Y | 0.92/0.58 |
| pub18:24687 | Mid-token completion of predictable multi-token sequences | Mid-span tokens of memorized, highly predictable sequences | N/N | 0.75/0.83 |

repetition-related flag agreement 0.75 (n=20); mean det. bal.acc Sonnet 0.75 vs Opus 0.78; Pearson 0.71

figure: notes/fig/test4_killarney_features.png
