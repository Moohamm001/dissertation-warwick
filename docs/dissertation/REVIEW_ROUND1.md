# Review round 1 — 2026-07-12 (two independent reviewers)

*Reviewer A: external-examiner audit (numbers/citations/consistency/viva). Reviewer B:
top-tier adversarial review (claims/logic/evidence). Verdicts: A "structurally sound, not
internally sealed"; B "Major revision". Every finding verified against code/reports before
action — one B finding was factually corrected (bootstrap DOES re-select the threshold per
resample, decision.py). Status after the same-day revision pass:*

| # | Finding (severity) | Status |
|---|---|---|
| A-F1 | SHAP entirely uncited (method carrying RQ3) | ✅ Lundberg & Lee 2017 crawled (`W2618851150`), cited §2.5/§3.7, D18 **[PROPOSED — awaiting approval]** |
| A-F2 | Demo threshold 0.617 traceable to no report | ✅ `decide` now emits the operating point → `decision_policy.md`; draft cross-references it |
| A-F3 | Two raw Briers in §4.4 (0.124 vs 0.1220) | ✅ 0.122 throughout §4.4 |
| A-M1/B-C1(part) | Four PR-AUC bases never reconciled | ✅ pooled-OOF vs fold-median sentence at §4.2; numerics-only basis note at §4.3 |
| A-M2 | Two thresholds (0.617/0.676) unreconciled | ✅ §3.10 explains catch-rate point + §4.5 generalisation |
| A-M3/B-C4(i) | Bootstrap procedure unspecified | ✅ verified in code (re-selected per resample); stated in §4.5 + report; lower bound named load-bearing |
| B-C4(ii) | 0.5-cut is a strawman baseline | ✅ top-decile baseline added to `decision_policy.md`: saving at R=20 is **3.9%** — quoted honestly in §4.5/§5.3 |
| A-M8/B-C4(iii) | Non-monotone savings unexplained | ✅ moving-denominator explanation in §4.5 + report |
| A-M4/B-C3 | 245-events fit: 3 points, non-monotone, analytic answer is 200 | ✅ basis disclosed in §4.3; hedged to "order 200–250 / 4–5×" in abstract, §4.3, §6.1, §6.3; 0.155-vs-0.189 reconciled |
| A-M5/B-C1 | Untuned challenger + roadmap deviation; unpaired comparison | ✅ owned in §4.2 + §5.1 ("unpowered to crown a winner"); paired fold test queued (§6.3 item 4) |
| A-M6 | EPV denominators (3.8 vs 2.0) | ✅ both stated §2.3 |
| A-M7 | Wasikowski year mismatch | ✅ 2010 in text, refs, `index.yaml` |
| A-M10/B-C2(lit) | "rarely surfaced" overclaim | ✅ softened; framed against §2.4 sources |
| B-C2 | Within-minority ECE near-tautological as "calibration" | ✅ verified in `metrics.py`; honestly defined in §3.5; §4.4/§5.2/§6.1/abstract reframed as *structural trade-off measured at an extreme point* |
| B-C5 | Fairness verdict granularity-specific; no protected attributes | ✅ both boundaries stated in §4.6 + §6.1 |
| B-C6 | Reason-code stability untested; Revenue/Sales collinearity | ✅ boundary stated in §4.7 (cluster-level claim); stability check queued |
| B-C7 | Credit Score null = range restriction (unexcluded) | ✅ full paragraph §5.4; conditional phrasing in §4.2 |
| B-C8 | Label conditions on resolution; eval ≠ deployment population | ✅ acknowledged §3.2, analysed §5.4, limitation (7) §5.6; score-distribution diagnostic queued |
| B-C9 | "Pre-registered" overclaim | ✅ "pre-specified" throughout |
| A-minors | trebles/construction/Fernández et al./under-a-second/adjectives/stale data_quality status | ✅ all fixed; `clean.py` template corrected + report regenerated |
| B-C10 | Permutation test | Survives as-is (both reviewers) |

## Round 2 — 2026-07-12 (same two reviewers, re-audit of the revised draft)

**Verdicts:** Examiner — "internally sealed on numbers; no FATAL"; adversarial reviewer —
**Minor revision** (up from Major). Both independently caught the same failure: the round-1
ledger claimed the "pre-registered→pre-specified" sweep was complete when four instances
survived — a response letter overstating its own compliance. All round-2 findings actioned
same-day:

| Finding | Status |
|---|---|
| "pre-registered" ×4 (abstract, §2.8, §4.4, §6.2) — flagged by BOTH reviewers | ✅ swept; verified by grep |
| Abstract lagged the body ×4: 19.0% without decile context; single-champion "dominant risk driver"; "event-level calibration error" wording; "early-delinquency" | ✅ abstract rewritten: decile context + interval added; cluster-level claim; "confidence shortfall" wording; "delinquency-detection model" |
| "early-delinquency" undefined (label carries no timing) | ✅ renamed in abstract/§1.2; definitional note added after the RQs (§1.3) since RQ1 quotes the project plan verbatim |
| `behind` (n=1) leave-one-out never run | ✅ run (`followup_checks.md` §4): PR-AUC median 0.117→0.099 (inside fold band), recall unchanged 0.600 — nothing turns on it; §5.6(4) updated |
| 3.9%-vs-decile had no uncertainty interval | ✅ bootstrapped (per-resample re-selection): **6.8% [1.2, 19.4]** — above zero (reviewer predicted it would cover zero; it does not) but modest and location-optimistic; §4.5, §5.3 and abstract state it at the right size |
| "lower bound is load-bearing" mischaracterised a location bias | ✅ reworded in §4.5 + report: "the true saving may lie below the quoted bounds" |
| §5.3 bolded principle overclaimed vs a strawman baseline | ✅ restated: principle holds, demonstrated at the right size; queue-depth derivation is the durable contribution |
| §4.7 "non-estimable everywhere" contradicted §4.6's qualified verdict | ✅ "at every cell of the audit's chosen granularity" |
| Ch5 residual "~245" drift ×2; Fernández et al.; 16.4% coincidence clause | ✅ all fixed |

**Still open after round 2:** D19–D23 curation approvals (the RQ2 prior-art, rare-events,
CV-inference, Hand 2006, and green-domain citations — gated on user approval; §5.2's
"consistent with the calibration sources reviewed in §2.4" remains a placeholder until they
land); Appendix C screenshots; Warwick word-limit/format check.

## Round 3 — 2026-07-17 (WMG template + plain-language + contribution framing)

**Verdicts:** examiner — "internally sealed on substance"; adversarial reviewer — **Minor
revision, at the accept boundary**. All findings actioned same session:

| Finding | Status |
|---|---|
| §3.12 test count "44 tests" vs Appendix D "49 tests" (examiner MAJOR) | ✅ verified `pytest --co` = 49; §3.12 → 49 |
| §7.3 "projection of §4.3" — stale after Results/Analysis split (examiner MAJOR) | ✅ → §4.4 (the projection's actual home) |
| §3.7 "coverage is carries almost no information" — stranded "is" from the plain-language swap (examiner MINOR) | ✅ "is" deleted |
| §2.6 "though Chapter 5 notes… coefficient stability" — pointer misdirects (examiner MINOR) | ✅ → §4.8 |
| §7.2 lacks the replication-vs-novelty demarcation that §2.4/§2.5 earn (BOTH reviewers, highest-leverage) | ✅ §7.2 now states plainly which findings *confirm* prior art (Wallace & Dahabreh, van den Goorbergh, Hand/EPV) and names the three *original* contributions: joint characterisation at N=50 on a real green-loan book, two entry-condition-tested non-estimability verdicts, and pricing declined claims in events (~200–250) |
| Abstract "beating… the decile rule by a modest 6.8%" asserts more than the body (location-optimistic interval) (adversarial cosmetic) | ✅ hedged to "modest, and less certain, 6.8%" |

Both reviewers independently confirmed: all 9 new citations (D19–D23) resolve body↔references;
every quantitative claim traces to a report; the plain-language edits changed no number or claim;
the Intro/Summary subsections add no unsupported material; §7.2/§1.4 no longer contradict
§4.6/§5.4. docx rebuilt (11,985 words, 9 H1, 23 captions, 3 TOC fields); 49 tests pass.

**Residual after the professional-formatting pass (2026-07-17):**
- ✅ *Formatting* — build.py now emits A4, 2.54 cm margins, Times New Roman 12 / 1.5 / justified,
  bold heading hierarchy, italic centred captions, roman front-matter + arabic body page
  numbering (two sections), and `updateFields=true` so Word offers to populate the
  TOC/lists/page numbers on open (the F9 step is now automatic).
- ✅ *Appendix B* — drafted the ethics-waiver rationale (secondary, anonymised, no human
  participants); only the actual approval reference/email is left as a blank to insert.
- ⛔ *Cannot be done here (need the student):* Appendix A ethics-training certificate and the
  Appendix B approval reference (fabricating either is prohibited); Appendix E three demo
  screenshots (app verified working end-to-end, but the screenshot-capture tool is broken in
  this environment — capture on the submission machine); name + student ID on the front pages.
- Word count: core chapters (1–7) ≈ 10,324 + abstract 457 — well under a typical 15,000 MSc
  limit; confirm the exact limit in the WMG handbook.

## Round 4 (mechanical) — 2026-07-18, self-run while reviewer agents were quota-blocked

A scripted examiner-equivalent sweep over all 9 chapter files. **Clean bill of health:**
- **Cross-references:** all 56 section headings enumerated; every `§x.y` in the body resolves to
  a real section — zero stragglers.
- **Section numbering:** Ch2–7 subsections are strictly sequential (no gap, no duplicate); the
  Introduction and Summary subsections are present in every one of Ch2–6.
- **Citations:** no in-text author-year is missing from `08_references.md` (the only two
  script flags — "Calster 2016", "Culloch 2006" — are regex tokenisation artefacts of the
  multi-word "Van Calster" and "McCulloch"; both are correctly in the references).
- **Number tracing:** every headline figure traces to a source report; the six script flags were
  all formatting artefacts (report "0 of 27" vs chapter "0 of 27 industries"; chapter 0.676
  rounds report 0.6763; unicode-minus −0.02; "14 of 25" = report "14/25"; 1.030 rounds 1.0303).
- **Projection wording:** no stray "245"; "200–250" used consistently.

Mechanically sealed. The remaining round-4 step is the adversarial reviewer's argument-level
confirmation of the §7.2 demarcation, deferred until the agent quota resets (15:30 London).

## Open items (not closable by editing)
1. ~~Curation approvals~~ ✅ **closed 2026-07-12**: D6 + D18 approved and promoted (curated
   21 → 23); all decisions D1–D18 now citation-closed.
2. ~~Queued analyses~~ ✅ **closed 2026-07-12**: `python -m emerald_ai followup-checks` →
   `reports/followup_checks.md`. Paired RQ1 test **confirms** no-winner (baseline wins 14/25
   shared folds, sign-test p ≈ 0.69, difference bands straddle zero); reason-code ranking
   **stable** (mean pairwise Spearman 0.89, Revenue top-3 in 100% of folds; cluster rank-1 only
   60% — cluster-level claim retained); population diagnostic **quantifies** the C8 mismatch
   (terminal-outcome threshold flags 16.4% of the full book vs the intended 10%, KS 0.065).
   Draft updated in §4.7, §5.1, §5.4, §5.6.
3. **Literature round 2 (B-C2/B-§4):** candidate additions for imbalance-vs-calibration prior
   art (van den Goorbergh 2022; Wallace & Dahabreh 2012), rare-events logistic (King & Zeng
   2001; Firth 1993), CV comparison inference (Nadeau & Bengio 2003; Dietterich 1998), Hand
   2006, and 1–2 green-loan domain papers from the brain's green-finance theme — none cited yet
   (protocol: crawl → propose → approve → cite).
4. Appendix C screenshots; Warwick word-limit/format check.
