## meta {"n_loop_items": 40, "n_prod_items": 20, "n_wiki_items": 20, "maxt": 8192, "pos_per_item": 200, "conclude": [" =", "**", " is", " The", " result", " final", " So", " total", " done", " answer", " correct", " Final", " Thus", " Therefore", " conclusion", " equals", " Answer", " Done", " Hence", " boxed", "</think>"], "cont": ["\n", "  ", "    ", "\n\n", " re", " *", " -", " again", " check", " But", " double", " another", " Or", " wait", " Let", " maybe", " Also", " verify", " Maybe", " Wait", "Wait", " Actually", " Alternatively", " Hmm"]} 
s = 2.3179283142089844 s/|h| = 0.5682997553447532

### Lens fidelity: held-out KL(model || lens) [nats], tuned-lens fit
| hs | logit | tuned |
|---|---|---|
| 12 | 12.99 | 1.65 |
| 13 | 13.05 | 1.59 |
| 14 | 13.26 | 1.53 |
| 15 | 12.59 | 1.49 |
| 16 | 10.85 | 0.66 |
| 17 | 10.55 | 0.56 |
| 18 | 9.09 | 0.51 |
| 19 | 7.52 | 0.44 |
| 20 | 6.80 | 0.22 |
| 21 | 5.73 | 0.17 |
| 22 | 4.07 | 0.12 |
| 23 | 1.95 | 0.08 |

### Bare-direction reads of u15 (top 10 tokens), per lens

**logit**

| hs | +u top | −u top | cls(+u) |
|---|---|---|---|
| 16 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |
| 17 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |
| 18 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |
| 19 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |
| 20 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |
| 21 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |
| 22 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |
| 23 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |
| 24 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |

**tuned**

| hs | +u top | −u top | cls(+u) |
|---|---|---|---|
| 16 | `离 岛 │ 还是 总 | 実は 可调 |^ 不如` | `encies èches inations agues  methodologies  classifications  artifacts させて頂きます lications owitz` | -3.27 |
| 17 | `气 但 还是 离 │ 气的 下的 真的是 也不是 贪` | `inations  wikipedia encies  Marks èches  :", ively agues  artifacts  mechanics` | -2.88 |
| 18 | `?... ...     [' │ IS anti som 总 ...,` | `inations  wikipedia  artifacts  Marks èches  series  Falcon encies icans ciate` | -2.48 |
| 19 | `ille IS 但可以 离 anti 不如 ILL eux interactive ỗng` | ` Marks  wikipedia  mechanics encies ively owitz 呗  marks inations èches` | -1.43 |
| 20 | `│ ...** gangen ...( 回事 ?... eel 其他地方 还是 gang` | ` thereby ively 呗  Marks  (“  wikipedia 以确定  pertinent  requisite 即可完成` | -1.60 |
| 21 | `│ ...** IS 是 的一部分 还是 ...* || ISK -**` | ` thereby 呗  wikipedia หน่อย owitz  통하여 即可获得 ubr  requisite стройства` | -1.19 |
| 22 | `│ ...** 不如  самом || ...*  justru 而是因为 还是 -**` | `呗  wikipedia  thereby owitz  requisite 化处理 ively  appropriately ubr 即可获得` | -1.65 |
| 23 | `is ...* IS t ...** e 而是因为 还是 │ 不如` | `呗  thereby стройства ádi  requisite owitz  align 지요  accordingly  selection` | -1.41 |
| 24 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |

**jlens_pub**

| hs | +u top | −u top | cls(+u) |
|---|---|---|---|
| 16 | `**? 还不是 ?** !? 根本不是 ?). ？** 还是在 }? 还是` | ` examinations  methodologies  publications  analyses  exhibitions  computations encies  presentations  respective  accordingly` | -3.31 |
| 17 | `其实是 **? 还不是 还是在 真的是 实际上是 !? ...** 真的有 根本不是` | ` methodologies  computations  examinations encies  publications  analyses -products  respective  exhibitions  requisite` | -2.94 |
| 18 | `**? ...** ?** 还不是 还是在 ?... 其实是 已经是 >? 正在` | ` methodologies  algorithms  computations  analyses  analysts  examinations  conditions  presentations  publications  techniques` | -1.89 |
| 19 | `**? ...** 已经是 现在是 其实是 负 这里的 ?** Actually 还是` | ` methodologies  computations  conditions  analysts  analyses  algorithms  calculations  examinations  estimates  estim` | -1.76 |
| 20 | `**? ...** 不如 実は 其实是 Actually 不是 转 是 真的是` | ` methodologies  products  computations -products  estimates  prototypes  algorithms  conditions  calculations  analysts` | -1.67 |
| 21 | `**? ...** ** 不如 不是 ?** **- 是 Actually 真的是` | ` prototypes  methodologies  computations  algorithms  datasets  smartphones  analysts  benchmarks  accordingly imens` | -1.11 |
| 22 | `>? **? 不是 Actually || 是 IS 不如 ?= 完` | ` prototypes  algorithms  computations 方法及  convictions  smartphones  methodologies owitz  benchmarks encies` | -1.71 |
| 23 | `'? >? 是 不如 ?- ?... || ?= │ 2` | `呗  convictions ivs  accordingly 衣裙 стройства owitz 以确定 imens  viewpoints` | -1.33 |
| 24 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |

**jlens_ours**

| hs | +u top | −u top | cls(+u) |
|---|---|---|---|
| 16 | `不是 还是 >? 还不是  Wait 蓝 根本不是  Actually  hanem 是指` | `方法及  computations ’: 得出结论  examinations 便可  methodologies  formulas  designs  assertions` | -4.41 |
| 17 | `不是 还是 其实是 真的是 还不是 Actually 还是在 蓝 的还是 实际上是` | ` computations  methodologies 性及 方法及  designs 分析及 设计及 -products  formulas 以求` | -3.50 |
| 18 | `不是 还是 ...** 是指 是 正在 蓝 >? ... 也是` | ` methodologies -products  designs  computations 方法及 分析及  conditions  analysts  algorithms 设计及` | -2.55 |
| 19 | `蓝 Actually 负 不是 还是 是指 是 不如 转 正在` | `-products  methodologies  computations 分析及 方法及  estim  conditions  analysts inius  convictions` | -2.36 |
| 20 | `不是 不如 蓝 是 Actually IS 是指 还是 负 也是` | `-products  computations  methodologies  products  prototypes 方法及  designs  calculations imens  estimates` | -1.98 |
| 21 | `不是 不如 IS 是 明 Actually 蓝 亲 完 改` | ` prototypes -products imens inius 方法及  computations  (“  methodologies .):  calculations` | -1.61 |
| 22 | `不是 是 IS 不如 || 完 Actually >? 2 OFF` | ` prototypes 方法及 以确定  convictions  computations  viewpoints 衣裙 moire  methodologies -products` | -2.06 |
| 23 | `是 2 || >? 不如 │ '? ... There �` | `以确定 ivs 衣裙 方法及 стройства 呗  viewpoints  convictions станции roids` | -1.82 |
| 24 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |

**jpp_pub**

| hs | +u top | −u top | cls(+u) |
|---|---|---|---|
| 16 | ` , **, ### ","  ),  ", ”, ##  ，  **,` | `.@ .'.  '@ ..' @g .' :@ @c @" ...'` | -0.74 |
| 17 | ` ؟  hanem  . 已经是 原来是  .. ?-  , 而是因为 其实是` | ` computations  algorithms ："  methodologies  sensit  viewpoints 便将 性及  exemplary  disciplines` | -2.36 |
| 18 | `...**  hanem ?... **? ?- 同上 -** **- 同样是 >?` | ` algorithms  computations 以确定  classifiers  reliably  analyses  methodologies  estim  exemplary  calculations` | -1.60 |
| 19 | `@  @ @( � @m ,@ (@  #{@  @( @$` | ` accordingly  ,  first  \n\n\n\n  reliably  appropriately  amic  demonstrated  conditions  fulfilled` | -1.30 |
| 20 | `...** **? -** **- ** ?** ?...  hanem  **- "**` | ` products  methodologies  :",  estimates  conditions 开口道  computations  outcomes  thereby  artifacts` | -1.29 |
| 21 | `...** ** **? ?** -** "** **- ？** '** !**` | ` ,  ”  ’  .  :",  ： ,’’  (“  ，  “.` | +0.09 |
| 22 | ` ,  ،  、 •  )  ， ## \\",\\"  .  ),` | `:** ：" ):** :</ ：</ .:** .** 。」 ，《 .):` | -1.25 |
| 23 | `ugada iste  sino '? UNT ...* 其他地方 illante )... isode` | `呗 ：“ .:  [@  calculations ：</  studies  convictions 以确定  fundamentals` | -1.21 |
| 24 | `...* ...** urada essin ologe kede 而是因为 tant zada gangen` | `呗  first  series  or  thereby  f 지요 即可  easily 而不` | -0.02 |

### Base-relative reads at real loop states (h + s·(±u15)), top-10 by mean Δp, and CONCLUDE−CONTINUE Δlogp score

**logit**

| hs | −u: up (Δp) | +u: up (Δp) | cls(−u) | cls(+u) | cls(−u) prod-pos | cls(+u) prod-pos |
|---|---|---|---|---|---|---|
| 16 | `\n ,  if /she ."). .). /her .". ., 教于` | `...* ...** ologe cante zada 而是因为 同期的 ?... kede ijden` | +0.175 | -0.274 | +0.008 | -0.104 |
| 17 | `, /she  if  WAIT ."). .).  and  wait “A  of` | `...* ...** cante ologe zada ?... 而是因为 venta ificante  **.**` | +0.122 | -0.165 | +0.013 | -0.036 |
| 18 | `\n ,  = /she  if  and  “  of ，“ .".` | `...* ...** ologe  **.** zada venta 而是因为 cante …* ändern` | +0.101 | -0.136 | +0.013 | -0.019 |
| 19 | `\n ,  = .). /she .".  if  and  of /her` | `...* ...**  **.** venta 而是因为 ologe ?... cante *". urunk` | +0.092 | -0.131 | +0.010 | -0.008 |
| 20 | ` = .). WAIT  +  and  if  WAIT (wait 1 .".` | `...*  **.** ...** venta …* ossid cante ?... 五颜六色的 *".` | +0.073 | -0.094 | +0.012 | -0.023 |
| 21 | ` = WAIT  $ $.                  WAIT  if          +  and` | `...*  **.** ...** 五颜六色的 venta ossid …* ”… istou cante` | +0.052 | -0.067 | +0.012 | -0.019 |
| 22 | `\n  $          = 1  Wait  he , $.   ` | `...* ...**  **.** ".** 五颜六色的 她还是 他还是 ionario 各种各样的 *".` | +0.031 | -0.038 | +0.006 | -0.004 |
| 23 | `\n . , 1        $     Wait        ` | `...* ...**  **.** (let 她还是 venta ..., 价格是 urada isata` | +0.009 | -0.010 | +0.003 | +0.013 |

**tuned**

| hs | −u: up (Δp) | +u: up (Δp) | cls(−u) | cls(+u) | cls(−u) prod-pos | cls(+u) prod-pos |
|---|---|---|---|---|---|---|
| 16 | `        \n\n  $ ) 4                                                                                                                                   This                                                                                  $\\ ."` | `     *  is             *  min *. \n  Wait ".` | +1.598 | -1.199 | +1.320 | -1.197 |
| 17 | `        4  orders \n\n                                                                                         \n  bird                                                                                                                                   year  This` | `     min  is  Wait ? ". *. ...                             8` | +1.347 | -1.003 | +1.045 | -0.940 |
| 18 | `                            3 4  orders \n\n  This              bird  \\` | `    2 ...  min *. P  is ".  there  Wait` | +1.141 | -0.796 | +0.849 | -0.740 |
| 19 | `        ic 6  $\\ N  the 3 4  bird                    ` | `     gone 2                             t ... *.             5 8` | +0.683 | -0.449 | +0.491 | -0.437 |
| 20 | ` $  This $         ."  the ally  " 1 ,` | ` * " ...     *.  is 2 ?  let  =` | +0.507 | -0.359 | +0.393 | -0.336 |
| 21 | ` $ ." ally  This  Thirty  (  No 6  of  or` | ` *  is " ...  Is  thirty \n\n *. ? t` | +0.325 | -0.205 | +0.254 | -0.207 |
| 22 | ` This ally ,  $  Let  (  me  this        No` | ` is  *  Is ...    I      there  " ?` | +0.414 | -0.299 | +0.315 | -0.275 |
| 23 | ` This \n  Let ally ,  $  let  of  it  me` | ` is  Is ... \n\n  there  * ? 2 's  $\\` | +0.321 | -0.231 | +0.249 | -0.213 |

**jlens_pub**

| hs | −u: up (Δp) | +u: up (Δp) | cls(−u) | cls(+u) | cls(−u) prod-pos | cls(+u) prod-pos |
|---|---|---|---|---|---|---|
| 16 | `.” ,  ($ .).  “ ，“ "</  Thus ."</  interpretations` | `**? ?**  Wait ？** ".** "? ?). ? ?! !?` | +1.698 | -1.476 | +1.426 | -1.496 |
| 17 | `, \n . .” .</ ，“ ”  ($  Thus ’ve` | `**?  Wait ...** ?** ？** ** **!  **- Wait ".**` | +1.416 | -1.227 | +1.141 | -1.204 |
| 18 | `\n . ，“ , ：“ ” .” $x  if 。”` | `** **? ...** ?**  Wait ？** ".** "** .** **-` | +0.808 | -0.817 | +0.602 | -0.723 |
| 19 | `. , \n ，“ ” .” ：“  Calculation  if nd` | `**? ** .** ...** ?** Wait **。 **. ？** Okay` | +0.701 | -0.702 | +0.524 | -0.602 |
| 20 | `, \n ” ，“  “     .”  $  Calculation .</` | `** Wait **?  WAIT .** ...** wait Okay  **- 5` | +0.510 | -0.528 | +0.414 | -0.466 |
| 21 | `\n ,  “ ”      $ .”         ，“  \\$` | `** Wait  WAIT 1 2 Okay **? 5 \\ ,I` | +0.302 | -0.315 | +0.239 | -0.273 |
| 22 | `,  $ $.  (\\  volume  pick  \\$  fourth $x $n` | `1 2 5 Wait ,I  the 0 4  *  WAIT` | +0.412 | -0.383 | +0.333 | -0.333 |
| 23 | `   ,  $ .  "  Let \n </think> :  This` | `Wait  WAIT 4 1  There 5 \n\n  Is Okay  "...` | +0.272 | -0.239 | +0.240 | -0.227 |

**jlens_ours**

| hs | −u: up (Δp) | +u: up (Δp) | cls(−u) | cls(+u) | cls(−u) prod-pos | cls(+u) prod-pos |
|---|---|---|---|---|---|---|
| 16 | `.” .).  “  Thus ,” ”, ”. ’ll ，“ .".` | ` Wait "?  _,  -> **? ? ?). }?  wait ...).` | +2.183 | -1.723 | +1.974 | -1.865 |
| 17 | `.”  Thus  “ ,  dozen ”, ” .). ’ll ,”` | ` Wait  _, "? ...** }? 0 5  $? ? Wait` | +1.614 | -1.260 | +1.386 | -1.295 |
| 18 | `.” ”  “ ，“  Thus  dozen ”,  Seems .).  Wow` | ` Wait 0 ...**  _, 2  WAIT }. **? 5  is` | +1.098 | -0.931 | +0.877 | -0.889 |
| 19 | `.). .". ., nd .</ .”  “ ” rd ，“` | `0  Wait 2  _, 。 5 Okay  Okay }. 1` | +0.955 | -0.826 | +0.742 | -0.752 |
| 20 | `.,  “ .</ .). ”  fourth ，“ .: .” .".` | `1 5 2 0  Is Wait 4 3  is  WAIT` | +0.651 | -0.578 | +0.535 | -0.531 |
| 21 | `”  “ .,  $  fourth .? .”         $x .</` | `  2 5  Is  the 4 Wait 0  is 1` | +0.473 | -0.415 | +0.385 | -0.381 |
| 22 | `,  $ $.  fourth  +  Eth  five  us  price  volume` | ` * 4 0 1 's  Is Wait  the 2  is` | +0.510 | -0.450 | +0.412 | -0.398 |
| 23 | `    $ ,  This  Let \n  *(  it </think>  No` | ` *  Wait  Is \n\n Wait  There  there    is  =` | +0.379 | -0.340 | +0.323 | -0.311 |

**jpp_pub**

| hs | −u: up (Δp) | +u: up (Δp) | cls(−u) | cls(+u) | cls(−u) prod-pos | cls(+u) prod-pos |
|---|---|---|---|---|---|---|
| 16 | ` or . ,  。 \n\n  if  `,  :  -,  and` | ` , **, ### ","  ), ##  ",  "," ”,  ,[` | +0.488 | -0.320 | +0.416 | -0.349 |
| 17 | `\n . ,  $  dozen  a  of  Thus  divisible ?` | ` .  ,  ؟  ?  .* ?$  ,-  .$ wait  There` | +0.958 | -0.707 | +0.888 | -0.787 |
| 18 | `, \n .  dozen  $ ：“  satisfy  of  " $x` | `...**  **.** **? ?$ **- -**  **- \n\n ?** \n  \n` | +0.632 | -0.558 | +0.526 | -0.517 |
| 19 | `. ,   \n  Wait \n  $   \n\n  If $" 6` | `@ @$ :@  #{@ ,@ #@  @ @m  @$ �` | +0.602 | -0.466 | +0.525 | -0.471 |
| 20 | `, \n      $         .  “             ”  *“` | `...** **  **- **? "**  ** ?** .** -** Wait` | +0.367 | -0.364 | +0.318 | -0.343 |
| 21 | `\n ,  $  Wait         \r\n      has  -  +` | `...** "** ** .** ?** **? !** **. -** **-` | +0.022 | -0.085 | -0.021 | -0.040 |
| 22 | ` Wait . &apos          ,"  .**  ",                   ` | ` , • ## \n\n \\",\\"  )  ، � \\" ”…` | +0.306 | -0.223 | +0.291 | -0.228 |
| 23 | `,  $ . $      `$  a  if  to "` | `"...  **.** "?  Is ...*  "?"  ?  * \n\n  ...` | +0.212 | -0.168 | +0.211 | -0.179 |

**rlens**

| hs | −u: up (Δp) | +u: up (Δp) | cls(−u) | cls(+u) | cls(−u) prod-pos | cls(+u) prod-pos |
|---|---|---|---|---|---|---|
| 16 | `\n\n  verify  I Final  This .  the  (  Let </think>` | ` Wait  is Wait  No  $  I      Let  the `.` | +1.645 | -0.667 | +1.059 | -0.895 |
| 17 | `\n\n  I  So  Let Final  ( .  This $.  Okay` | ` is Wait  Wait  No  I  $  correct \n  The  If` | +1.043 | -0.485 | +0.723 | -0.714 |
| 18 | `\n\n  I  So . $.  Okay  (  This Final  Let` | ` is Wait  The  maybe  correct  If        over            ` | +0.713 | -0.331 | +0.533 | -0.517 |
| 19 | `\n\n  I Let .  Let  (  So  This  Okay 3` | ` is Wait  Wait  maybe  The  I  over  If  a 'm` | +0.726 | -0.323 | +0.416 | -0.490 |
| 20 | `Let  I  Let .  (  So \n\n Final  let 0` | ` is Wait  I  The    maybe  If 'm  $  a` | +0.645 | -0.396 | +0.378 | -0.450 |
| 21 | ` I Let \n\n  ( .  Let  Okay Final  So  =` | ` is Wait  The  maybe   'm  over  If  I ...` | +0.540 | -0.320 | +0.305 | -0.367 |
| 22 | `\n\n  Okay Let .  I  let Final  Let  =  (` | ` is Wait  maybe    -  I 'm  The  over  actually` | +0.546 | -0.271 | +0.325 | -0.393 |
| 23 | `Let .  Wait  I  ph  = \n\n  This  Let  seems` | ` is  -  I    The 'm  over  the  a Okay` | +0.560 | -0.280 | +0.371 | -0.310 |

### P2a tally: layers (of 8) where cls(−u) > cls(+u), loop positions
- logit: base-relative 8/8; bare 8/8
- tuned: base-relative 8/8; bare 8/8
- jlens_pub: base-relative 8/8; bare 8/8
- jlens_ours: base-relative 8/8; bare 8/8
- jpp_pub: base-relative 8/8; bare 7/8
- rlens: base-relative 8/8; bare None/8

### Δresid (p50c − base) at loop positions
| hs | rel ‖Δ‖ | cos(Δ,−u15) | corr(lens Δ, lens −u) logit / tuned / jlens_pub / jlens_ours / jpp | top-8 tokens up (tuned, Δp) | top-8 down (tuned, Δp) |
|---|---|---|---|---|---|
| 16 | 0.53 | +0.99 | +1.00 / +1.00 / +1.00 / +1.00 / +1.00 | `        \n\n  $ )  $\\  orders  their  have` | `    i              grows  buys \n  has  cost` |
| 17 | 0.52 | +0.89 | +0.91 / +0.90 / +0.94 / +0.94 / +0.93 | `        \n\n 4  orders  sentences  bird . end` | `    \n             el  changed $.  grows  seats` |
| 18 | 0.51 | +0.83 | +0.86 / +0.85 / +0.84 / +0.84 / +0.83 | `        \n\n  have 4  orders                                         1  out` | `     changed              buys  cost  grows \n  seats` |
| 19 | 0.49 | +0.75 | +0.82 / +0.78 / +0.79 / +0.77 / +0.84 | `\n\n 4          $\\  bird  sheets  the times` | `\n  buys  he  changed  Wait             He  from` |
| 20 | 0.58 | +0.77 | +0.78 / +0.73 / +0.76 / +0.76 / +0.75 | `\n\n          $  This  the 1       .` | ` Wait \n  he      * 5 $. population` |
| 21 | 0.57 | +0.73 | +0.77 / +0.70 / +0.73 / +0.72 / +0.75 | `\n\n  $  verify  the  This 1 0  Let` | ` Wait  * \n  he  check 5 ,  his` |
| 22 | 0.56 | +0.69 | +0.74 / +0.63 / +0.66 / +0.66 / +0.70 | `\n\n  verify  This  the 0  Over  us  a` | ` Wait \n  check ,  he  is  been  "` |
| 23 | 0.62 | +0.56 | +0.63 / +0.58 / +0.61 / +0.61 / +0.65 | `\n\n  verify .  the  This  ( $.  a` | ` Wait \n  check  has  is  roll population  he` |
| 24 | 0.82 | +0.22 | +0.23 / +0.43 / +0.59 / +0.63 / +0.47 | `\n\n .  verify  This </think>  (  I  Let` | ` Wait \n  check Wait  has  is  changed  hive` |

final Δlogp vs R-lens(−u at layer L) correlation: {'16': 0.98, '17': 0.93, '18': 0.85, '19': 0.8, '20': 0.74, '21': 0.7, '22': 0.65, '23': 0.59}
next-token (loop continuation) Δlogp at loop positions: -0.068

### Variance share of u in the residual (base, generated positions of all analysis items)
| hs | n | u15 share | rank of u15 among eigvals | u19 share | top-1 PC share | mean h·u15 loop / non-loop gen / wiki |
|---|---|---|---|---|---|---|
| 13 | 399153 | 0.0424 (43.4× iso) | 2 | 0.0355 | 0.126 | 1.10 / 0.46 / 0.31 |
| 14 | 399153 | 0.0423 (43.3× iso) | 2 | 0.0358 | 0.127 | 1.31 / 0.58 / 0.39 |
| 15 | 399153 | 0.0363 (37.2× iso) | 3 | 0.0303 | 0.104 | 1.41 / 0.58 / 0.33 |
| 16 | 399153 | 0.0513 (52.6× iso) | 2 | 0.0412 | 0.103 | 2.25 / 0.94 / 0.53 |
| 17 | 399153 | 0.0480 (49.1× iso) | 2 | 0.0464 | 0.107 | 2.56 / 1.10 / 0.66 |
| 18 | 399153 | 0.0448 (45.9× iso) | 2 | 0.0503 | 0.109 | 2.78 / 1.24 / 0.74 |
| 19 | 399153 | 0.0347 (35.5× iso) | 3 | 0.0428 | 0.096 | 2.79 / 1.29 / 0.88 |
| 20 | 399153 | 0.0418 (42.8× iso) | 3 | 0.0614 | 0.111 | 3.65 / 1.69 / 1.14 |
| 21 | 399153 | 0.0350 (35.8× iso) | 3 | 0.0502 | 0.100 | 3.78 / 1.80 / 1.40 |
| 22 | 399153 | 0.0300 (30.7× iso) | 3 | 0.0438 | 0.098 | 4.03 / 2.00 / 1.56 |
| 23 | 399153 | 0.0191 (19.6× iso) | 5 | 0.0288 | 0.079 | 3.30 / 1.56 / 1.23 |
| 24 | 399153 | 0.0046 (4.7× iso) | 34 | 0.0073 | 0.069 | 1.01 / 0.63 / 0.21 |

figures: notes/fig/test2_killarney_variance.png notes/fig/test2_killarney_axis_dresid.png
