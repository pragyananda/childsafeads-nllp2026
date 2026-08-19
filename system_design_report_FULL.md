# System design report

# **ChildSafeAds, NLLP @ EMNLP 2026**

*Final evaluation score **0.6223** mean macro-F1 · routed hybrid — fine-tuned encoder \+ classical ML \+ QLoRA LLM, each used where it measurably wins · 15 fine-tuning runs in the submitted system, 58 across the study · 1×16GB GPU · \~22 GPU-hours to build, \~45 including everything rejected · no paid API, no instance leaves the machine*

*Team: **pragyananda** · code and this report: **https://github.com/pragyananda/childsafeads-nllp2026** · every number below is reproducible from the commands in Section 14\.*

Full ablations: [`system_design_report_FULL.md`](system_design_report_FULL.md).

## **Summary**

### **Evaluation results**

Five submissions were permitted. The first four were spent measuring components
against the evaluation set one at a time, which is why every effect below
decomposes exactly (Section 11). The fifth is the system this report describes.

| Metric                    | Submission 4 (encoder only) | **Final submission** | Held-out prediction       | Delivered                      |
| :------------------------ | :-------------------------- | :------------------------- | :------------------------ | :----------------------------- |
| **mean\_macro\_f1** | 0.6030                      | **0.6223**           | —                        | **\+0.0193**             |
| st1\_macro\_f1            | 0.5906                      | 0.5906                     | unchanged by construction | **exact**                |
| st2\_macro\_f1            | **0.7272**            | 0.6979                     | \+0.070, 5/5 folds        | **−0.0293 — reversed** |
| st3\_macro\_f1            | 0.4913                      | **0.5784**           | \+0.072, 20/20 splits     | **\+0.0871 — 121%**     |
| st3\_family\_macro\_f1    | 0.5798                      | 0.6501                     | —                        | \+0.0703                       |
| Baseline                  | 0.093                       | 0.093                      |                           |                                |

Because ST1 predictions are byte-identical to the scored run, the mean
decomposes with no residual: (−0.0293 \+ 0.0871)/3 \= \+0.0193, against the
\+0.0193 reported. **ST3 moved from the weakest sub-task in the field to
mid-field and carried the whole gain. The ST2 change cost 0.0098 on the mean;
shipping ST3 alone would have scored 0.6321.**

The two components were validated by the same criterion — consistency across
resampled channel-grouped splits — on instruments of 2,857 and 504 instances
respectively. **The larger instrument is the one that failed.** Section 6b
explains why, and it is the most useful result in this report.

### **Principal findings**

|                                                     |                                                    |
| :-------------------------------------------------- | :------------------------------------------------- |
| ST2 gain from the crawl                             | **\+0.128** (p \< 0.0001)                    |
| ST3*loss* from the crawl                          | **−0.031** (p \= 0.018)                     |
| Crawl needed for 89% of the benefit                 | **20% of the corpus**                        |
| Legal definitions supplied to a prompted model      | **\+0.038** on ST3                           |
| Statutory citations added on top of definitions     | **−0.002**                                  |
| none (ST1) recall before correction                 | **0.14**                                     |
| no\_flag (ST3) recall before correction             | **0.16**                                     |
| Held-out estimate made*before* first submission   | 0.60 (actual 0.5962)                               |
| Classical TF-IDF vs the same encoder, same folds    | ST2**\+0.070**, ST1 **−0.019**        |
| That ST2 gain, measured on the evaluation set       | **−0.029 — the criterion's first failure** |
| Effective sample size behind it (rarest ST2 class)  | **17**, not 2,857                            |
| Of that ST2 gain, attributable to recalibration     | \+0.035 (thresholds reach it too)                  |
| Of that ST2 gain, attributable to complementarity   | \+0.035 (thresholds cannot)                        |
| QLoRA LLM standalone ST3 vs the encoder it improves | 0.444 vs 0.502                                     |

Three sub-tasks over the same sponsored segments. Only two of them are worth paying to crawl the web for — and the third performs *worse* when you do. Nor do they want the same *model*: the sub-task an encoder wins by 0.019 is next to the one it loses by 0.070, on identical features and identical folds.

## **1\. The question we set out to answer**

The task frames itself as a system-design problem: if you ran a regulator monitoring commercial content aimed at minors, what could you achieve at each level of data access, and what would it cost?

The dataset makes that concrete by grouping fields into four levels ordered by collection cost — the segment transcript, then video metadata, then channel context, then the crawled destination page. Level 1 is already in hand for any video you can transcribe. Level 4 requires resolving and fetching an outbound link, which at platform scale is the expensive part, and in the wild fails for dead or bot-blocking pages.

We treated "which levels are worth buying" as the primary research question and accuracy as the instrument for answering it, rather than the other way round. The answer turned out to differ per sub-task, which is not something a single leaderboard number can express.

## **2\. The finding**

We fine-tuned the same multi-task architecture at each access level and repeated every configuration across seeds. Across 21 runs, the three sub-tasks disagree about what data they want.

| Configuration          | Collection cost | Runs | ST1                    | ST2                    | ST3                    |
| :--------------------- | :-------------- | ---: | :--------------------- | :--------------------- | :--------------------- |
| L1 — transcript only  | lowest          |    9 | 0.604 ±.013           | 0.621 ±.014           | **0.427 ±.021** |
| L2 —\+ video metadata | low             |    1 | 0.594                  | 0.637                  | 0.416                  |
| L4 —\+ page, 1024 tok | highest         |    3 | 0.638 ±.027           | 0.719 ±.004           | 0.381 ±.006           |
| L4 —\+ page, 2048 tok | highest         |    8 | **0.686 ±.059** | **0.749 ±.014** | 0.396 ±.027           |

*Dev macro-F1, mean ± standard deviation across independent runs. ST1 is macro-averaged over reference-present labels, matching the leaderboard implementation.*Tested directly, with Welch's t-test across runs:

- **ST2 — product category.** The crawl is worth **\+0.128** (p \< 0.0001, d \= 9.0). Unambiguous, and unsurprising: the shop page states what is being sold.
- **ST1 — commercial type.** The crawl is worth **\+0.083** (p \= 0.005). Real, but noisy — ST1's run-to-run spread is by far the largest of the three.
- **ST3 — compliance flags.** The crawl **costs** 0.031 (p \= 0.018). At the family level the same direction holds (−0.025, p \= 0.063).

> **Consequence for a monitoring system.** The sub-task with actual legal consequence — is this advertising disclosed, does it exhort children to buy — is the one that needs *nothing beyond the transcript*. Compliance triage can run at the cheapest tier, over the whole platform. The expensive crawl should be reserved for identifying *what* is being sold, which a regulator can defer until a segment has already been flagged.

The mechanism is plain enough in the inputs. ST3 turns on what the creator said and where a disclosure appeared; product-page boilerplate is a distractor competing for the same context window. Doubling the context from 1024 to 2048 tokens, which admits more page text, moved ST2 up and ST3 down in the same runs.

## **3\. The system this implies: screen cheaply, escalate selectively**

The finding above does not recommend "run level 4 on everything", which is what a leaderboard submission does. It recommends a **cascade**: screen every segment at level 1, and pay to crawl only the segments where crawling changes an answer.

We simulated exactly that. ST3 is always answered at level 1\. ST1 and ST2 are answered at level 1 unless an instance is escalated, in which case the crawled model answers instead. The crawl budget is the free parameter.

| Crawl budget | Targeted escalation | Flagged-first   | Random (control) |
| -----------: | :------------------ | :-------------- | :--------------- |
|           0% | 0.540               | 0.540           | 0.540            |
|          10% | **0.577**     | 0.577           | 0.546            |
|          20% | 0.588               | **0.590** | 0.551            |
|          30% | 0.582               | 0.580           | 0.553            |
|          50% | 0.590               | 0.589           | 0.580            |
|          70% | 0.594               | 0.590           | 0.586            |
|         100% | 0.596               | 0.596           | 0.596            |

*Escalation policies: by uncertainty of the cheap tier (top-2 ST1 margin plus distance of ST2 scores from the decision boundary); by whether level-1 screening raised any compliance flag; and at random as a control.*
**Crawling 20% of the corpus retains 89% of what crawling all of it buys** (0.590 against 0.596, from a 0.540 floor). Which 20% matters as much as how much: at a 10% budget, targeted escalation scores 0.577 against random's 0.546, so most of the available gain comes from choosing well rather than from crawling more.

For an authority the practical reading is that the outbound-link crawl — the component that is slow, fails on dead or bot-blocking pages, and carries the most friction under platform terms — can be a targeted follow-up on roughly one segment in five, rather than a precondition for monitoring at all.

## **4\. What it costs to act on the output**

Macro-F1 is the leaderboard's currency, not an enforcement authority's. What determines whether a monitoring system is deployable is the size of the human review queue it generates at a recall target the regulator is willing to defend.

For the two practices prohibited outright under UCPD Annex I, using the level-1 model:

| Flag                     | Prevalence | Recall target | Precision | Reviewed per 1,000 screened |
| :----------------------- | ---------: | ------------: | --------: | --------------------------: |
| undisclosed\_advertising |      14.7% |           80% |      0.39 |                         301 |
|                          |            |           90% |      0.30 |                         446 |
|                          |            |           95% |      0.25 |                         556 |
| direct\_exhortation      |      15.3% |           80% |      0.22 |                         567 |
|                          |            |           90% |      0.21 |                         656 |
|                          |            |           95% |      0.19 |                         753 |

Undisclosed advertising is tractable: catching four in five costs a reviewer about 300 segments per thousand screened, and roughly two in five of those are genuine. Direct exhortation is not deployable at any of these operating points — at 80% recall more than three quarters of the queue is noise. That is consistent with it being the flag whose definition is a three-part contextual test rather than a surface property of the text.

### Severity-weighted scoring

The taxonomy grades each flag by how the law operates, and macro-F1 discards that: it treats a blacklisted practice under UCPD Annex I exactly like a soft-law HFSS case. Re-weighting by severity (per se 3, conditional 2, soft law 1):

| ST3 answered from    | Unweighted | Severity-weighted |
| :------------------- | :--------- | :---------------- |
| Level 1 (transcript) | 0.479      | **0.498**   |
| Level 4 (full crawl) | 0.449      | 0.458             |

The level-1 advantage **grows** under severity weighting, from \+0.030 to \+0.040. The reason is visible per flag: undisclosed\_advertising, a *per se* prohibition, scores 0.65 from the transcript against 0.51 with the crawl. The cheap tier is better precisely where the law is strictest, which strengthens the case for screening at level 1 rather than weakening it.

## **5\. The system we submitted**

The system is **routed twice**: once by data access level, and once by model
family. Neither routing was a design preference; each is the output of a
measurement reported below.

|               | model                                                                                                         | data levels | why                                                                                          |
| :------------ | :------------------------------------------------------------------------------------------------------------ | :---------- | :------------------------------------------------------------------------------------------- |
| **ST1** | ModernBERT-large, 6 seeds averaged                                                                            | 1–4        | the crawl is worth\+0.083; the classical arm *loses* 0.019 here                            |
| **ST2** | that encoder, averaged 50/50 with a classical TF-IDF arm                                                      | 1–4        | the blend is worth\+0.070 over the encoder, 5/5 folds                                        |
| **ST3** | ModernBERT-large at level 1, 6 seeds, with the two disclosure flags averaged against a QLoRA-tuned Qwen2.5-7B | 1 (LLM: 2)  | the crawl*costs* ST3 0.031; the LLM beats the encoder on disclosure and only on disclosure |

**The backbone.** A shared ModernBERT-large with three heads — softmax for ST1, sigmoid for ST2 and ST3 — trained jointly with pos-weighted BCE so that flags appearing 40 times in training are not drowned out by ones appearing 1,277 times. Entering all three sub-tasks costs barely more than entering one, and since omitted sub-tasks score zero, a single-sub-task entry is capped at a mean of 0.333. Two instances are trained, differing only in which levels they read, and each is a six-seed probability average; Section 7 explains why averaging is preferred to selecting among seeds.

**The classical arm (ST2).** TF-IDF over three views — word n-grams on the page prose, word n-grams on the transcript, and **character n-grams (3–5, `char_wb`) over the outbound URL** — into per-label logistic regression. The URL view is the one that matters: brand identity is a substring, and two thirds of evaluation instances promote a domain seen in training. Section 6a decomposes the \+0.070.

**The LLM arm (ST3 disclosure only).** Qwen2.5-7B-Instruct, 4-bit NF4, LoRA r=16, fine-tuned as an eight-way *sequence classifier* rather than a generator, so that it emits probabilities that can be averaged rather than strings that must be parsed. Its input carries the taxonomy's written tests verbatim. Section 10a reports why it is used on two flags out of eight.

**Constraint layer.** ST3 probabilities are thresholded at 0.5 — except `inadequate_disclosure`, at 0.40 — then four rules are applied in this order:

1. YouTube's paid-promotion label rules out undisclosed\_advertising.
2. The two disclosure flags are made mutually exclusive; **undisclosed\_advertising is always the survivor** (\+0.008, 19/20 held-out halves, against keeping the higher-scoring one).
3. A near-empty transcript replaces the whole set with insufficient\_context.
4. Otherwise, if no\_flag reaches probability 0.6, it replaces the whole set (Section 11).

Rules 3 and 4 are exclusive of each other and of everything above them, matching the taxonomy's rule that no\_flag and insufficient\_context each stand alone. ST1 and ST2 receive no post-processing; ST2 falls back to its highest-margin label if nothing clears 0.5.

Every decision constant in that layer — 0.40, 0.6, `keep_undisc` — was fixed by the same protocol: adopt only on ≥18/20 resampled channel-grouped splits. Nothing was fitted by gradient on dev.

**What changed from the scored 0.6030 system, and what did not.** ST1 is untouched, deliberately: the evaluation set is 0.108 harder than dev on ST1 (Section 11), so we have no instrument that can validate an ST1 change, and the one time we shipped an ST1 change on a dev-only signal it cost 0.041. ST2 gains the classical arm. ST3 gains three epochs of training, the QLoRA disclosure blend, and two constants. On dev the ST3 arm moves **0.457 → 0.530**; on the evaluation set it moved **0.4913 → 0.5784**.

**Measured afterwards, one of these two changes was wrong.** The ST3 arm
over-delivered and the ST2 arm regressed by 0.029. We have left the system as
submitted rather than describing the version we would build now, and Section 6b
reports the failure in full, because the reason the ST2 component passed
validation is the most transferable thing we learned. The system we *would*
now submit is this one with the ST2 blend removed: it scores **0.6321**.

### Constraints applied after the model — kept

Three rules the taxonomy states outright, enforced on the output rather than hoped for from the loss:

- YouTube's own paid-promotion flag rules out undisclosed\_advertising. In training this holds in **0 of 1,281** such instances — a hard constraint, not a tendency.
- undisclosed\_advertising and inadequate\_disclosure are mutually exclusive by definition; we keep whichever the model scores higher.
- A near-empty transcript implies insufficient\_context.

### Transcript length as a feature — kept

insufficient\_context instances have a median transcript of **2 words** against a corpus median of 222\. No bag-of-words model can represent length, and the encoder never learned it either — the label sat at F1 0.00 in every run. A single length threshold lifts it to 0.38, and ST3 overall by \+0.042.

## **6\. What we tried and threw away**

Three components that looked well-motivated and did not survive measurement. Reporting them is the point of a design report; each cost real time and each result is reusable.

### Outbound-domain gazetteer — rejected

Splits are channel-disjoint but *not* brand-disjoint: 68% of dev and 65% of test instances promote a product domain that also occurs in training. A gazetteer mapping domains to rare labels is worth **\+0.13** over a TF-IDF baseline on those labels — and **\+0.001** over the fine-tuned encoder, which memorises the same brands unaided (gambling 0.00 → 0.55, age\_restricted 0.00 → 0.55 with no gazetteer at all). A hand-built lexicon is a real win for a cheap model and redundant for a fine-tuned one.

### Knowledge-based keyword lexicon — rejected

For the two labels the gazetteer could not reach, we wrote a lexicon from the taxonomy's own definitions — alcohol, vaping, betting, weapons. It reached F1 0.12 on age\_restricted and got *worse* at level 4, because product pages mention alcohol and age limits incidentally. Surface keywords do not track the legal category.

### Per-label decision thresholds — rejected

Macro-F1 weights a 5-instance flag exactly like a 260-instance one, so tuning each label's threshold looks like the obvious lever. On dev it appears to buy \~0.05. Fitted on one channel-grouped half of dev and scored on the other, it transfers at **\+0.005** with three seeds and **−0.034** with six. A single global threshold — one free parameter instead of twenty — fails the same way: 0.6 wins on the fitting half and loses on the held-out half. We ship a fixed 0.5.

The encoder consequently over-predicts relative to the training priors — 1.91 ST2 labels per instance against a gold 1.32 — which for a while we defended as correct behaviour, on the grounds that macro-F1 rewards recall on rare classes. That defence was half right and we could not tell which half until we had something to compare against. Fold-honest per-label thresholds are worth \+0.035 (4/5 folds), so the miscalibration *was* costing score; the classical blend is worth \+0.070 while also fixing the calibration, so thresholds were the wrong instrument rather than the wrong idea. Section 6a separates the two effects.

### Jointly optimised thresholds — rejected

Per-label tuning treats each threshold independently, which is exact under macro averaging only if the labels are decided independently. ST3's are not: no\_flag and insufficient\_context are exclusive of everything else, so raising one threshold changes which instances the others can claim. Optimising all eight jointly by coordinate ascent respects that coupling and converges quickly.

It reaches **ST3 \= 0.545 on dev**, against 0.468 for the shipped decision rule — the largest apparent gain any configuration in this project produced, and a mean of 0.716. It is also eight free parameters fitted to 504 instances, five of which carry hfss\_food\_marketing and seven insufficient\_context. We did not submit it. The three corrections we *did* measure on the evaluation set moved −0.041, −0.005 and \+0.021 against dev gains of \+0.051, \+0.026 and \+0.031, and a gain of this size fitted this way has no better claim.

We record it because the failure mode is instructive rather than embarrassing: the more faithfully a tuning procedure models the metric's structure, the more effectively it fits the sample it is tuned on. Sophistication in the optimiser is not evidence about generalisation.

## **6a. Which model each sub-task wants**

Sections 2–4 ask which *data* each sub-task wants. The same question can be
asked of the model, and it has an equally split answer — which is not visible
from a leaderboard, because a leaderboard reports one number for a system that
has already made the choice.

We trained a deliberately cheap classical arm — TF-IDF into per-label logistic
regression, three views: word n-grams over the page prose, word n-grams over the
transcript, and character n-grams (3–5) over the outbound URL — and evaluated it
on the **same channel-grouped 5-fold split over train\+dev (2,857 out-of-fold
instances)** used for every encoder configuration. Identical folds, identical
instances, identical decision rule, so the two are directly comparable and can
be averaged fold-honestly.

|               | ModernBERT-large | classical TF-IDF |               50/50 blend | folds won by the blend |
| :------------ | ---------------: | ---------------: | ------------------------: | :--------------------- |
| **ST1** |  **0.614** |  0.595 (−0.019) |           0.603 (−0.011) | **2/5**          |
| **ST2** |            0.661 |  0.686 (\+0.025) | **0.731 (\+0.070)** | **5/5**          |

*Mean of five held-out folds. The blend was adopted for ST2 and rejected for ST1.*

> **Read this section with Section 6b open.** The ST2 blend was adopted on the
> evidence below and **lost 0.029 on the evaluation set**. The dissociation
> between the sub-tasks is real and survives; the magnitude and the sign of the
> ST2 effect do not. We have left the analysis as it stood at adoption time,
> because the interesting question is not what the right answer was but why this
> evidence was not enough to find it.

**Same features, opposite verdicts.** The two sub-tasks read the same input and
are annotated from the same product page, so a single number for "does classical
ML work here" would be meaningless. ST2 asks *what category of thing is this* —
a question a brand name answers outright, and 68% of dev and 65% of test
instances promote a domain seen in training, so character n-grams over the URL
retrieve the answer nearly by lookup. ST1 asks *what kind of commercial
relationship is this* — whether an offer exists at all, whether it is a good, a
service or a subscription — which the same brand name does not settle, and where
an encoder's composition over the page is doing real work.

### Is the ST2 gain complementarity, or just calibration?

The blend also halves the label rate, from 1.91 per instance to 1.35 against a
gold 1.32, so an obvious deflationary reading is that averaging with a
better-calibrated model is a threshold adjustment in disguise. That is testable:
fit per-label thresholds on four folds, apply them to the fifth, and see how much
of the gain a purely calibration-based fix recovers.

| ST2, fold-honest               |            mean | vs encoder        | folds won     |
| :----------------------------- | --------------: | :---------------- | :------------ |
| encoder at 0.5                 |           0.661 | —                | —            |
| encoder\+ per-label thresholds |           0.696 | \+0.035           | 4/5           |
| **50/50 blend at 0.5**   | **0.731** | **\+0.070** | **5/5** |

**The gain splits almost exactly in half.** \+0.035 is recalibration, which
thresholds also reach. The other \+0.035 is complementarity — information the
encoder does not have, which no adjustment of the encoder's own decision
boundary can recover — and the blend beats the threshold-corrected encoder on
4/5 folds. Both halves are real and only one of them is a free lunch.

We report the decomposition rather than the headline because the headline alone
would have been misleading in a specific way: an ensemble that improves a
miscalibrated model is a weaker claim than an ensemble that improves a calibrated
one, and the difference is not visible unless you construct the calibrated
control.

The decomposition turned out to matter more than we intended. On the evaluation
set the two halves had *opposite signs*: the recalibration half pulled the label
rate down toward the training prior, which extinguished a 17-instance class and
cost the sub-task 0.029. Splitting a gain into parts you can name is worth doing
even when both parts look positive, because it is the only way to notice later
which part was load-bearing.

### Why this was found late, and what that says about ablation order

Our first classical baseline was a twenty-minute throwaway on day one: ST2
0.636, untuned, against an encoder that had not been tuned either. The family
was dropped on that basis and not revisited for a week. The comparison was fair
and the conclusion was wrong, because both arms improved under tuning and they
did not improve by the same amount. **A baseline measured once, early, is
evidence about that baseline and not about the family.** The 0.070 that was
available for the whole project sat behind a single unrepeated comparison.

## **6b. Where the adoption criterion failed, and why**

Section 6a adopted the ST2 blend on the strongest evidence in this project:
\+0.070 mean over five held-out folds of a 2,857-instance channel-grouped
cross-validation, winning **5 folds out of 5**, with the gain decomposed into a
recalibration half and a complementarity half. Section 10a adopted the ST3
hybrid on 20 resampled halves of a 504-instance dev set, \+0.072, **20/20**.

Both were measured on the evaluation set in the same submission. ST3 delivered
**\+0.087** — more than predicted. ST2 delivered **−0.029**.

|            | instrument                 | instances | prediction     | delivered         |
| :--------- | :------------------------- | --------: | :------------- | :---------------- |
| ST3 hybrid | 20 resampled halves of dev |       504 | \+0.072, 20/20 | **\+0.087** |
| ST2 blend  | 5-fold CV over train\+dev  |     2,857 | \+0.070, 5/5   | **−0.029** |

The bigger, more careful, more expensive instrument is the one that lied. That
inversion is worth more than the score it cost.

### The mechanism is visible in the prediction counts

The blend moves ST2 from 1.89 labels per instance to 1.43, against a training
prior of 1.32. Per class, on the 503 evaluation instances:

| ST2 class          |      train n |     encoder |       blend |        change |
| :----------------- | -----------: | ----------: | ----------: | ------------: |
| **gambling** | **17** | **3** | **0** | **−3** |
| gambling\_adjacent |          109 |          47 |          24 |          −23 |
| health             |          319 |          72 |          46 |          −26 |
| creator\_community |          316 |         122 |          87 |          −35 |
| apps               |          825 |         172 |         128 |          −44 |

`gambling` is predicted **zero times**, which fixes its F1 at exactly 0. ST2 is
macro-averaged over twelve classes, so one class dropping out costs up to
**1/12 \= 0.083**. The observed ST2 loss is 0.029. A single rare class,
extinguished by an operating-point shift, accounts for the entire regression
without any need to invoke a distribution shift in the blend's other eleven
columns.

### Why 2,857 instances could not see it

Because the metric is macro-F1, **the effective sample size is set by the rarest
class, not by the corpus.** `gambling` has 17 instances in train\+dev. Spread
over five folds that is roughly three per fold — enough for the class to fire
occasionally, not enough for five folds to agree about whether it fires. The
instrument reported 5/5 because the eleven well-populated classes moved
together and dominated the pooled score; the twelfth, carrying 8.3% of the
metric, was below the resolution of every fold.

This is a sharper statement of Section 7's variance result, and it corrects it.
Section 7 says consistency across resampled splits is the reliable signal.
That remains true — it was right five times out of six, including on the ST3
change in this same submission. What must be added is that **consistency is only
informative about the classes the resamples actually contain.** A criterion
computed over a pooled macro average inherits the sample size of its weakest
column. We would now require, before adopting any multi-label change: no class
may lose more than half its predictions, and no class may go to zero.

### A claim we withdrew, and should not have

An earlier draft of Section 6 said that the encoder's over-prediction relative
to the training prior — 1.89 ST2 labels per instance against 1.32 — was correct
behaviour rather than a defect, on the grounds that macro-F1 rewards recall on
rare classes and matching the prior would cost score. On the strength of the
cross-validation instrument we revised that: fold-honest thresholds recovered
\+0.035, so the miscalibration appeared to be costing score after all.

**The original claim was right.** Over-prediction is what keeps a 17-instance
class alive in the output, and on the evaluation set that was worth more than
the calibration it wasted. The blend's two halves are not equally valuable: the
complementarity half is plausibly still real, and the recalibration half —
pulling the label rate down toward the training prior — is what removed
`gambling` and cost the sub-task. Averaging with a better-calibrated model
delivers both halves inseparably, which is why the component could not be
salvaged by turning one of them off.

The general form, and the reason this section exists: **a validation instrument
can be enlarged, resampled and cross-checked and still be systematically blind,
if the quantity that decides the outcome is thinner than the instrument's
resolution.** Ours was built to defeat run-to-run noise, which it did. It was
never built to measure a class with 17 examples, and nothing about the 5/5 fold
record announced that limitation.

## **7\. How much of a leaderboard gap is real**

The most portable result here is not about this dataset. It is about how much noise a 2,353-instance training set produces, and how easy that noise is to mistake for progress.

Re-running an *identical* configuration — same code, same seed — gave a mean of 0.618 on one run and 0.575 on another. torch.manual\_seed does not make training deterministic: GPU atomics, cuDNN kernel selection and dropout ordering all vary.

| Sub-task | Identical-config spread | Apparent level effect | Verdict           |
| :------- | :---------------------- | :-------------------- | :---------------- |
| ST1      | 0.089                   | 0.091                 | indistinguishable |
| ST3      | 0.056                   | 0.009                 | indistinguishable |
| ST2      | 0.016                   | 0.104                 | real              |

Every conclusion in this report that rested on single runs was withdrawn and re-measured across seeds; the surviving claims are the ones with t-tests attached. Two practical consequences follow.

First, **a dev-leaderboard gap under roughly 0.05 between two teams carries little information** about which system is better. Second, a submission strategy that picks whichever run scored best on dev is selecting partly on noise and should be expected to regress on test. Our own dev submission of 0.6185 came from an unusually favourable run combined with dev-fitted thresholds; we expect the honest system to land near 0.60, and we would rather say so in advance than explain it afterwards.

Averaging probabilities across six seeds, instead of selecting among them, is the cheapest defence available and the reason the submitted system is an ensemble.

## **8\. Notes on the release**

Three observations that cost us time and may save someone else's.

- **dev.jsonl is not valid JSONL.** Six of its 504 records span multiple physical lines, so the shipped load\_data.py and check\_submission.py both raise JSONDecodeError on the first instance. Naive line-parsing silently yields 503 of 504 — the dangerous failure, since it produces slightly wrong local scores with no error. train.jsonl and test.jsonl are clean. Streaming with json.JSONDecoder().raw\_decode reads all three correctly.
- **Channel-disjoint is not brand-disjoint.** Two-thirds of dev and test instances promote a domain seen in training, and 119 of the 170 domains with three or more training instances carry a single ST1 label. A substantial share of ST1 and ST2 is reachable by memorising brands rather than by understanding the segment, which is worth knowing when reading any score on this benchmark.
- **ST1's metric.** Macro-averaging over all five ST1 labels rather than reference-present ones changes the same predictions by \~0.13, because other has 2 training and 0 dev instances. We confirmed the leaderboard uses reference-present labels by matching a submission scored at 0.6185 to a local 0.6183.

## **9\. Cost, and what we would do with more**

The submitted system costs about **22 GPU-hours on a single 16GB card** (RTX PRO 2000 Blackwell), all of it one-off training:

| Component                                            | Runs |     Each |    Total |
| :--------------------------------------------------- | ---: | -------: | -------: |
| ModernBERT-large, level 4 @2048 (ST1, ST2)           |    6 | \~52 min |    5.2 h |
| ModernBERT-large, level 1 @1024, 6 epochs (ST3)      |    6 | \~78 min |    7.8 h |
| QLoRA on Qwen2.5-7B-Instruct, 4-bit (ST3 disclosure) |    3 |    \~3 h |    9.0 h |
| TF-IDF\+ logistic regression (ST2)                   |    1 | \~10 min | CPU only |

Inference over all 3,360 instances is **\~2 GPU-hours**, nine tenths of it the 7B component; the encoders and the classical arm together are under 20 minutes. The marginal cost of scoring one more video is a transcript, two encoder forward passes and one 7B forward pass.

The 7B model is 82% of training cost for a component that touches two flags out of eight. That is defensible here because those two flags are *per se* prohibitions and the encoder was weakest on them, but it is the first thing to drop under a compute constraint: the encoder-only ST3 arm retains 0.502 of the blend's 0.530.

The wider study cost approximately **45 GPU-hours**, the difference being the level-2 and level-4/1024 ablation runs, the 5-fold cross-validation instrument of Section 6a (10 further encoder trainings), six train+dev models built but not used, an abandoned DeBERTa-v3 second architecture, and \~2 hours of LLM inference for Section 10 (\~9 min per 500 instances). No paid API was used at any stage, and no instance left the machine.

Level-1 models also train roughly three times faster than level-4 ones, so the cheap tier is cheaper twice over — once in collection, once in compute. Since the compliance sub-task is the one that runs on every video, and it is the one that needs only level 1, the economics compound in the right direction.

## **10\. Does legal context help? A measured answer**

The organisers pose this as an open question and do not score it. We ran the experiment: a local instruct model (Qwen2.5-7B, 4-bit, on the same 16GB card) predicting ST3 zero-shot, in three conditions that differ **only** in how much legal grounding the system prompt carries. Model, decoding, instances, parsing and the constraint layer are identical across conditions, so any difference is attributable to the legal context alone.

| Flag                                     | Fine-tuned encoder | Bare flag names | \+ taxonomy definitions | \+ legal provisions |
| :--------------------------------------- | :----------------- | :-------------- | :---------------------- | :------------------ |
| misleading\_claim                        | **0.76**     | 0.02            | 0.09                    | 0.21                |
| undisclosed\_advertising                 | **0.65**     | 0.48            | 0.36                    | 0.42                |
| age\_restricted\_or\_prohibited\_product | **0.45**     | 0.15            | 0.37                    | 0.32                |
| inadequate\_disclosure                   | **0.41**     | 0.13            | 0.22                    | 0.14                |
| direct\_exhortation                      | 0.34               | 0.41            | 0.41                    | **0.45**      |
| no\_flag                                 | 0.22               | 0.35            | **0.42**          | 0.37                |
| **ST3 macro-F1**                   | **0.427**    | 0.254           | 0.292                   | 0.290               |
| ST3 family                               | **0.645**    | 0.406           | 0.431                   | 0.419               |

**Definitions help; citations do not.** Giving the model the taxonomy's plain-language definitions is worth **\+0.038** over bare flag names. Adding the instruments and provisions from legal\_provisions.json on top is worth **−0.002** — nothing. For most flags, what carries the signal is the written test, not the statutory reference.

**With one exception, and it is the informative one.** On direct\_exhortation, the provisions condition produces **0.45**, the best score any system in this report achieves on that flag, encoder included. That is the flag whose definition is explicitly a three-part contextual test rather than a lexical property — and it is the only place where statutory grounding earns its tokens.

**The zero-shot model loses badly overall** (0.290 against 0.427), which is the expected result: 2,353 in-domain training examples beat a rulebook. But the two systems fail in different places, and that is exploitable.

### The hybrid, and why it is principled rather than opportunistic

The LLM wins precisely on the flags where an encoder is structurally blind: those defined by a written test it cannot read. Substituting the LLM's direct\_exhortation decision into the encoder's output:

| ST3 system                          | ST3             | direct\_exhortation | no\_flag |
| :---------------------------------- | :-------------- | :------------------ | :------- |
| Encoder alone                       | 0.439           | 0.34                | 0.24     |
| \+ LLM (taxonomy) for exhortation   | 0.461           | 0.41                | 0.34     |
| \+ LLM (provisions) for exhortation | **0.465** | **0.45**      | 0.33     |

**\+0.026 on ST3**, consistent across both conditions, with no\_flag improving as a side effect: removing the encoder's spurious exhortation calls lets the compliant class surface. The LLM is zero-shot and never saw dev, so this is not a fitted result.

**Measured on the evaluation set, the substitution did not transfer.** The mean fell from 0.5962 to 0.5945; ST1 and ST2 predictions were identical between the two submissions, placing the entire movement in ST3, which lost approximately 0.005 while gaining 0.026 on dev.

The dev gain was below the reliability threshold established in Section 7 — roughly 0.05 on this data — and, unlike the corrections in Section 11, this component was never put through the channel-grouped held-out protocol. A plausible mechanism combined with a sub-threshold dev gain proved not to be evidence of generalisation.

The per-flag observation may still hold: the LLM's advantage on direct\_exhortation is consistent across all three prompting conditions. What the evaluation set does not support is that substituting it improves the system as a whole.

### Do demonstrations help? A dissociation between two kinds of label

The comparison above supplies legal *definitions*. A second experiment supplies labelled *examples* instead — the same model, the same prompt condition, the same constraint layer, with eight in-context demonstrations drawn from train. Two selection strategies: eight fixed at random, and eight retrieved per instance by TF-IDF similarity over transcripts.

| Flag                                     | Fine-tuned encoder | Zero-shot      | Random-8 | Retrieved-8 |
| :--------------------------------------- | :----------------- | :------------- | :------- | :---------- |
| misleading\_claim                        | **0.76**     | 0.09           | 0.69     | 0.70        |
| undisclosed\_advertising                 | **0.65**     | 0.36           | 0.33     | 0.41        |
| age\_restricted\_or\_prohibited\_product | **0.45**     | 0.37           | 0.29     | 0.38        |
| inadequate\_disclosure                   | **0.41**     | 0.22           | 0.28     | 0.31        |
| direct\_exhortation                      | 0.34               | **0.41** | 0.15     | 0.33        |
| hfss\_food\_marketing                    | **0.26**     | 0.07           | 0.00     | 0.15        |
| **ST3 macro-F1**                   | **0.427**    | 0.292          | 0.292    | 0.368       |
| ST3 family                               | **0.645**    | 0.431          | 0.496    | 0.530       |

**Eight examples move misleading\_claim from 0.09 to 0.70.** The model's legal reasoning is unchanged between the two conditions; only its exposure to how the flag is actually applied differs. This converts the observation below from an inference into a demonstration: the fine-tuned encoder's advantage on this flag is knowledge of an annotation convention, not superior legal analysis.

**But demonstrations are not uniformly helpful, and the pattern is informative.** direct\_exhortation is the single flag where the zero-shot model beats everything in this report — 0.41, above the encoder's 0.34 and above both few-shot conditions. It is also the only flag whose taxonomy entry is an explicit three-part test rather than a category description. Given the written test the model applies it well; given examples it drifts toward the annotators' noisier application of it, losing 0.08 with retrieval and 0.26 at random.

The two label types behave oppositely:

- **Policy-like labels**, where the definition is broad and the annotation pipeline supplies the operative threshold, are learned from examples and essentially cannot be reasoned to. misleading\_claim fires on 54% of instances; no reading of UCPD Arts. 6–7 predicts that rate.
- **Test-like labels**, where the taxonomy states a decision procedure, are better served by the procedure than by examples.

**Random demonstrations are worth nothing in aggregate** (0.292, identical to zero-shot): the misleading\_claim gain is exactly cancelled by the collapse of direct\_exhortation and hfss\_food\_marketing. Retrieval recovers most of it (0.368). The useful statement is therefore not "examples help" but "*similar* examples help, on the labels that are conventions rather than tests".

Retrieved few-shot still trails the fine-tuned encoder by 0.059. Eight demonstrations do not substitute for 2,353 training instances, and we did not pursue in-context learning further for the submitted system.

### 10a. Fine-tuning the LLM: a specialist that improves a better generalist

Everything above is zero-shot or few-shot. The obvious next question is what
happens when the language model is actually trained on the task, and the answer
turned out to be the most useful negative-looking result in the project.

Qwen2.5-7B-Instruct, 4-bit NF4, LoRA r=16 on the attention and MLP projections,
fine-tuned on the 2,353 training instances as an **eight-way sequence classifier
rather than a generator**. That choice is what makes the rest possible: a
generative head emits strings that must be parsed and cannot be averaged, while a
classification head emits calibrated probabilities that ensemble with the
encoder's directly. The input carries the taxonomy's written tests verbatim.
Three seeds, ~3 GPU-hours each on the same 16GB card.

| Flag                                     | gold n | encoder (level 1) | **QLoRA LLM** |
| :--------------------------------------- | -----: | ----------------: | ------------------: |
| undisclosed\_advertising                 |     74 |             0.610 |     **0.850** |
| inadequate\_disclosure                   |    118 |             0.369 |     **0.482** |
| direct\_exhortation                      |     77 |   **0.353** |               0.324 |
| misleading\_claim                        |    260 |   **0.769** |               0.685 |
| age\_restricted\_or\_prohibited\_product |     16 |   **0.600** |               0.160 |
| hfss\_food\_marketing                    |      5 |   **0.333** |               0.222 |
| no\_flag                                 |    127 |   **0.556** |               0.491 |
| insufficient\_context                    |      7 |   **0.435** |               0.400 |
| **ST3 macro**                      |        |   **0.503** |               0.452 |

*Dev, six-seed encoder against three-seed LLM, both at a flat 0.5 with no constraint layer.*

**The LLM is worse overall and better where it matters.** It loses on six flags
out of eight and by 0.051 on the macro — a model we would have discarded if we
had read only the summary row. On the disclosure family it wins by 0.240 and
0.113, and those are the two flags the whole system was weakest on.

The mechanism is not mysterious. `inadequate_disclosure` is defined as *a
disclosure exists but is buried*, and `undisclosed_advertising` turns on whether
one appears anywhere at all. Both are **written tests over the presence and
placement of a specific string**, stated explicitly in the taxonomy, and both are
decidable by a model that can read the test. The encoder cannot read a rulebook;
it can only infer the rule from 2,353 examples of it being applied. Conversely
`age_restricted` (16 instances) and `hfss` (5) are categories to be recognised,
not tests to be applied, and the LLM simply has not seen enough of them — 0.160
against the encoder's 0.600.

### Routing, not ensembling

The distinction matters, and the two produce different numbers. Under the shipped
constraint layer, on 20 channel-grouped held-out halves of dev:

| ST3 arm                                                   |             dev | vs encoder-only                | splits won      |
| :-------------------------------------------------------- | --------------: | :----------------------------- | :-------------- |
| encoder ep6 alone                                         |           0.502 | —                             | —              |
| \+ LLM averaged into **all eight** flags            |           0.541 | \+0.039 but **sd 0.040** | **10/20** |
| \+ LLM averaged into **the disclosure family only** | **0.530** | \+0.029, sd 0.009              | **20/20** |

**The all-flags blend has the larger mean and we did not ship it.** Its
per-split standard deviation is four times larger and it wins exactly half its
splits — the signature, established in Section 7 and confirmed three times on the
evaluation set, of a change that will not transfer. The disclosure-only blend
wins less on average and wins *every* split, because it only touches the two
flags where the LLM's advantage has a stated mechanism. Restricting an ensemble
to where you can say why it should help is the difference between 20/20 and a
coin flip.

The general form: **a component that is worse everywhere on average can still be
the best available component somewhere specific, and the summary metric actively
hides this.** ST3 macro-F1 says 0.452 against 0.503 — discard it. The per-flag
breakdown says it holds the best score in the project on the flag carrying a *per
se* prohibition under UCPD Annex I. An evaluation that reports only the mean
would have thrown away the single most useful model we trained.

### And the constraint layer is not decoration

Once the arm was strong enough for the question to be worth asking, we re-tested
the rules against it:

| ST3 arm                                |             dev | splits won vs shipped |
| :------------------------------------- | --------------: | :-------------------- |
| shipped: blend\+ full constraint layer | **0.530** | —                    |
| blend, format constraints only         |           0.473 | 1/20                  |

The layer is worth **\+0.057 on 19 of 20 splits**. Earlier in the project, over a
weaker arm and with only the exclusivity rules in place, the same measurement
came out neutral (0.503 raw against 0.499) and we recorded it as such. The
difference is the three decision constants added since — `inadequate_disclosure`
at 0.40, `no_flag` at 0.6, and always resolving the disclosure tie toward
`undisclosed_advertising`. Each was adopted on ≥19/20 splits, and their joint
effect is the largest single component of the ST3 arm.

### What misleading\_claim reveals about the benchmark

The most informative number in the table is 0.76 against 0.09. The encoder learns that the annotation pipeline applies misleading\_claim to 54% of instances; a model reasoning from the legal definition almost never applies it. Both are internally consistent — they are answering different questions. Our scores therefore measure **agreement with an annotation policy**, not agreement with law, and a system optimised for this benchmark is partly learning a labelling convention. This is not a criticism of the release, which documents its automatic annotation openly; it is a caution about how any score on it, including ours, should be read.

### A note on where inference runs

Every model in this report runs locally. Beyond cost, that is a compliance property: the corpus is CC BY-NC-SA from SponsorBlock with video-derived fields under a research-use agreement, and the terms prohibit redistribution outside the team. Sending transcripts and crawled page text to a third-party API endpoint raises a question that local inference simply does not. For a public authority operating under platform terms, where the model runs is part of the system design, not an implementation detail.

## **11\. The system will not say "nothing here"**

The largest single defect in the system was present from the first training run and surfaced only under a per-class error audit. It is worth recording both the defect and the fact that component-level experimentation did not reveal it. A breakdown of ST1 by class:

| ST1 class                      |            n |       Precision |          Recall |              F1 |
| :----------------------------- | -----------: | --------------: | --------------: | --------------: |
| digital\_content\_or\_services |          248 |           0.909 |           0.923 |           0.916 |
| physical\_goods                |          218 |           0.888 |           0.945 |           0.916 |
| physical\_services             |           24 |           0.833 |           0.625 |           0.714 |
| **none**                 | **14** | **1.000** | **0.143** | **0.250** |

none — no identifiable commercial offer, typically a dead or parked page — is recovered twice in fourteen. Precision is perfect: when the system predicts none it is always correct; it simply almost never does. Because ST1 macro-averages over four classes, that one class accounts for 0.19 of ST1.

The same shape appears in ST3. no\_flag — commercial content that appears compliant — has 127 instances, F1 0.24, recall 0.16, and is emitted 41 times against 127\.

**Two mechanisms, one behaviour.** In ST1 the classification head trains under unweighted cross-entropy while the ST2 and ST3 heads receive pos\_weight, so the rare classes are rarely the argmax. In ST3 no\_flag is *derived* rather than predicted — our decision rule emits it only when no other flag fires, and misleading\_claim alone fires on 54% of instances, leaving the compliant class structurally starved.

**A third mechanism: the ensembling itself.** Averaging probabilities across seeds — the step we adopted precisely because selecting among noisy runs is unreliable — suppresses the rare classes further. Training a variant with class-weighted ST1 cross-entropy and comparing the two ways of combining its five seeds:

| ST1 combination           |        macro-F1 |         none F1 | none predicted (14 gold) |
| :------------------------ | --------------: | --------------: | -----------------------: |
| individual seeds, mean of |           0.735 |              — |                       — |
| probability averaging     |           0.694 |           0.235 |                        3 |
| majority vote             | **0.745** | **0.435** |                        9 |

The ensemble scores *below the average of its own members* under probability averaging, because averaging shrinks exactly the extreme probabilities a rare class depends on: five seeds that each confidently predict none on different instances average to a confident prediction on none of them. Majority voting preserves them, and recovers 9 of 14.

We report this as a mechanism rather than a fix. On ten channel-grouped halves, class-weighted plus voting gained −0.005 with sd 0.088 and improved only 6/10 splits, so by the criterion established below it does not qualify. But it explains why a defect that looks like a loss-function problem was not repaired by correcting the loss function: two independent parts of the pipeline suppress the same classes, and fixing one leaves the other in place.

For a compliance system this is the least acceptable direction of bias. A monitor that under-predicts "no identifiable offer" and "appears compliant" does not fail neutrally: it inflates the review queue and, if its outputs were taken at face value, would systematically over-attribute violations to creators. Macro-F1 does not surface this, because the metric rewards recovering rare *positive* classes, which is where most of our effort went.

### Correcting it, and which correction transferred

Both defects admit post-hoc corrections that require no retraining. We validated each on ten channel-grouped halves of dev, with constants fixed a priori rather than tuned per split:

| Correction                     | Held-out gain  | sd              | Splits improved |
| :----------------------------- | :------------- | :-------------- | :-------------- |
| no\_flag emitted when p ≥ 0.6 | \+0.031 on ST3 | **0.010** | **10/10** |
| ST1 logit adjustment, τ\= 0.3 | \+0.051 on ST1 | 0.070           | 7/10            |

We then measured both on the evaluation set. Because macro-F1 for each sub-task depends only on that sub-task's predictions, and our submissions differed in one component at a time, the effects decompose exactly:

| Submission                                      | ST1              | ST2              | ST3              | Mean             |
| :---------------------------------------------- | :--------------- | :--------------- | :--------------- | :--------------- |
| 1\. Routed seed ensemble                        | 0.5906           | 0.7272           | 0.4707           | 0.5962           |
| 2\. \+ LLM substitution for direct\_exhortation | 0.5906           | 0.7272           | 0.4657\*         | 0.5945           |
| 3\. \+ ST1 τ=0.3 and no\_flag t=0.6            | 0.5501           | 0.7272           | 0.4913           | 0.5895           |
| **4\. \+ no\_flag t=0.6 only**            | **0.5906** | **0.7272** | **0.4913** | **0.6030** |

\* *inferred from the mean; ST1 and ST2 predictions were identical to submission 1\.*

| Component                                | Effect on the evaluation set |
| :--------------------------------------- | :--------------------------- |
| no\_flag t \= 0.6                        | **\+0.0206 on ST3**    |
| ST1 logit adjustment τ\= 0.3            | **−0.0405 on ST1**    |
| LLM substitution for direct\_exhortation | **−0.005 on ST3**     |

**Held-out consistency predicted transfer; effect size did not.** The correction with the tightest variance and a perfect split record (no\_flag: sd 0.010, 10/10) transferred. The one with the *largest* dev gain but loose variance (ST1 τ: \+0.051 dev, sd 0.070, 7/10) reversed sign on test and cost more than the other gained. The LLM substitution, which was never subjected to the held-out protocol at all, also failed.

This is the practical form of the variance result in Section 7\. On a benchmark of this size, the reliable selection signal is *how consistently* a change wins across resampled held-out splits, not how much it wins by on any one of them. Reporting a single dev number — as a leaderboard submission implicitly does — carries almost no information about whether a change will generalise.

Submissions 2 and 3 were regressions. They are also what made submission 4 certain rather than speculative: measuring both corrections on the evaluation set in a single bundled submission allowed the good component to be isolated arithmetically, and submission 4's score was known to four decimal places before it was uploaded.

### Why dev-based selection was unreliable here

Comparing dev and evaluation performance per sub-task explains the pattern:

Comparing the submitted system's dev and evaluation scores, sub-task by sub-task:

| Sub-task       | Dev             | Evaluation       | Difference                          |
| :------------- | :-------------- | :--------------- | :---------------------------------- |
| ST1            | 0.699           | 0.5906           | evaluation**harder**, −0.108 |
| ST2            | 0.649           | 0.7272           | evaluation easier,\+0.078           |
| ST3            | 0.468           | 0.4913           | evaluation easier,\+0.023           |
| **mean** | **0.605** | **0.6030** | **−0.002**                   |

The two splits are channel-disjoint and are not interchangeable samples. Yet the *mean* agrees to 0.002, because the per-sub-task errors are large and happen to cancel. The same held for the first submission: dev 0.596 against 0.5962.

This resolves an apparent contradiction. Our channel-grouped estimation protocol was well calibrated at the level of a whole system — it predicted both submitted scores to within 0.002 — while being unreliable for selecting a component on a single sub-task, because ST1 alone is 0.108 harder on the evaluation set. A system-level estimate and a component-level one are not the same measurement, and only the first was trustworthy here.

## **12\. The monitoring system**

Sections 2 to 4 argue that compliance screening needs only the transcript and that the crawl should be a targeted follow-up. A submission file does not demonstrate that. `monitor.py` implements the pipeline the argument implies and reports its cost in the units an authority budgets — crawls issued and analyst-segments queued.

```
every segment  ──▶ [1] SCREEN   ST3 model, level 1, transcript only
                       ├──▶ confident compliant ....... closed, no cost
                       ├──▶ low confidence ............ ABSTAIN → human
                       └──▶ flagged
                            [2] TRIAGE   P(any per se prohibition)
                            [3] ENRICH   crawl, budget-capped, top of queue only
                            [4] QUEUE    ranked, with what-is-being-sold attached
```

Run over dev at a 20% crawl budget and 15% abstention: 26% of segments close at tier 1 with no crawl, 59% are flagged, 15% are routed to a human unjudged, and 200 crawls plus 738 analyst items are generated per 1,000 segments screened.

**Flagging 59% of segments is not triage.** The reference itself marks misleading\_claim on 54%, so a faithful system reproduces a firehose. What makes the pipeline usable is not the flagging decision but the **order** of the queue, which is the only thing an analyst with a fixed budget experiences.

### Ranking, and what it is worth

Ranking every segment by the probability that any *per se* prohibition applies — a noisy-or over undisclosed\_advertising and direct\_exhortation:

|               Reviewed | True per se found |     Precision | Lift over random |
| ---------------------: | ----------------: | ------------: | ---------------: |
|            top 25 (5%) |                18 |           72% |           2.25× |
| **top 50 (10%)** |      **38** | **76%** | **2.53×** |
|          top 100 (20%) |                61 |           61% |           1.91× |
|          top 200 (40%) |                88 |           44% |           1.47× |
|                all 504 |               135 |         26.8% |           1.00× |

**Reviewing 10% of the corpus surfaces 28% of all *per se* violations at 76% precision**, against a 26.8% base rate. That is an operating point an authority can budget against, and it is reachable without crawling anything.

Two negative results shaped this. Ranking by a severity-weighted sum over all six flags — the obvious construction, and our first — achieves only 1.3× lift at k=50, because the conditional flags are numerous and dilute the *per se* signal. And measuring lift *within* the already-flagged subset understates it: a deployed system scores every segment and works down one list, rather than filtering and then ordering.

### What the system does not do

It triages; it does not adjudicate. At 80% recall the *per se* flags run at precision 0.39 and 0.22 (Section 4), so most of any queue drawn deep enough to be exhaustive is not a violation. The abstention channel exists for the same reason: a compliance tool that cannot decline to judge routes its own errors into enforcement. Neither property is visible in macro-F1, and neither would be present in a system built to optimise it.

## **13\. Limitations**

- The L2 configuration was run once. Its position in the level ordering is not established, and we do not claim it.
- ST3's advantage at level 1 is significant at p \= 0.018 across 17 runs but modest in size (0.031). It supports "the crawl does not help ST3" more strongly than it supports any precise ordering among the cheap levels.
- Labels are automatically annotated with human validation, so all scores measure agreement with an annotation pipeline, not with law. Our system partly learns that pipeline's conventions — misleading\_claim fires on 54% of instances, a rate reflecting a labelling policy as much as a legal threshold.
- Our first submission scored **0.5962** against a held-out estimate of 0.60 made before submitting; the fourth scored **0.6030**. The agreement of the first is evidence that the channel-grouped estimation protocol is calibrated at the level of a whole system, even though Section 11 shows it was not reliable for selecting individual components on ST1.
- Five evaluation submissions were available. The first four measured components one at a time; two of them were regressions, which is how the exact decomposition in Section 11 was obtained. In retrospect the two corrections there should have been submitted separately rather than bundled, since they carried very different held-out support; bundling them cost one upload and briefly obscured which component was responsible.
- **The ST2 blend was a mistake and we shipped it**, on 5/5 folds of a 2,857-instance instrument that could not resolve a 17-instance class (Section 6b). It cost 0.029 on ST2 and 0.0098 on the mean. Submissions were exhausted, so the corrected system — identical minus that component, scoring 0.6321 — is reported but not leaderboard-verified.
- The ST2 blend weight was fixed at 0.5 and never tuned. Given the above this was fortunate rather than principled: tuning it would have added a fitted parameter to a component whose validation was already blind in the dimension that mattered.
- Six adoption decisions were made on the consistency criterion and measured on the evaluation set. Five transferred; one reversed. We report the hit rate rather than only the successes, since a criterion that is right five times in six is a useful instrument and a criterion advertised as reliable is not the same thing.
- The LLM comparison uses one 7B model at 4-bit precision. A larger or full-precision model might change the balance, though not obviously the direction: the encoder's advantage comes from having seen the annotation policy, which no amount of model capacity supplies.

*All figures are dev macro-F1 under the leaderboard metric, computed with a local replica validated against a scored submission. Significance by Welch's t-test across independent training runs.*

## **14\. Reproducing the submitted system**

Everything below runs on one 16GB GPU with no network access after the model
weights are cached, and reproduces the submitted file exactly — we verified this
by rebuilding it and diffing all 503 predictions.

```
# 1. encoders (6 seeds each; ~13 GPU-hours total)
for s in 0 1 2 3 4 5; do
  python src/encoder.py --level 4 --maxlen 2048 --seed $s              # ST1, ST2
  python src/encoder.py --level 1 --maxlen 1024 --epochs 6 --seed $s --tag _ep6
done

# 2. QLoRA specialist for the ST3 disclosure family (3 seeds; ~9 GPU-hours)
for s in 0 1 2; do python src/llm_finetune.py --seed $s; done

# 3. classical arm for ST2 (CPU, ~10 min)
python src/classical.py --task st2 --save

# 4. route ST3: encoder everywhere, encoder+LLM on the two disclosure flags
python src/build_st3_hybrid.py --split test \\
  --out work/probs_test_L1_ModernBERT-large_len1024_seed0_hyb3.npz

# 5. assemble
python src/make_test_submission.py --tag "" --seeds 0 1 2 3 4 5 \\
  --st3-tag _hyb3 --classical-st2 work/classical_test_st2.npz \\
  --no-flag-t 0.6 --disclosure-tiebreak keep_undisc --inadequate-t 0.40 \\
  --out SUBMIT_BEST.jsonl

# 6. validate against the organisers' checker
python starting_kit_test/check_submission.py \\
  work/SUBMIT_BEST.jsonl data/public_data_test/test.jsonl
```

The validation instrument is separate from the build. `src/encoder.py --cv-folds 5 --cv-fold K` produces the out-of-fold predictions over train\+dev
that every adoption decision in Sections 6a and 10a was made on; `src/data.py`
holds the streaming loader that reads the malformed `dev.jsonl` correctly
(Section 8), and `src/metrics.py` the local replica of the leaderboard metric,
validated to 0.0002 against a scored submission.

One note for anyone rerunning this: probability files must be named with the
data level, the max length, the seed *and* the configuration tag. An earlier
version of our naming scheme omitted length and seed, and a 2048-token run
silently overwrote a 1024-token one — producing a plausible ablation table for
an experiment that had not been run. It is the kind of error no test catches and
no result looks wrong under.
