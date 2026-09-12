# DDXPlus Data Audit Report

**Project:** CoDx — Probabilistic Coherence as a Test of Clinical Reasoning in LLMs  
**Generated:** 2026-09-12T03:49:01+00:00  
**Corpus:** DDXPlus, all three released splits  
**Hardware:** NVIDIA RTX PRO 5000 Blackwell, 48935 MiB, 595.91.07

## Summary

| Severity | Count |
|---|---|
| CRITICAL | 2 |
| MAJOR | 3 |
| MINOR | 1 |
| INFO | 13 |

| # | Check | Severity | Finding |
|---|---|---|---|
| 1 | `generator_determinism` | CRITICAL | 21,004 replicate case pairs; the generator returns a DIFFERENT differential for 25.82% of them (mean JSD 0.04895, p90 0.21979, max 0.52613); the top-1 diagnosis flips in 0.48% of replicate pairs |
| 2 | `kb_probability_release` | CRITICAL | release_conditions.json declares 888 (pathology, evidence) slots and supplies conditional probabilities for 0 of them |
| 3 | `class_balance` | MAJOR | 49 pathologies; prior range 0.00025-0.06326 (imbalance ratio 251.6x); prior entropy 5.38 bits of 5.61 max |
| 4 | `duplicates` | MAJOR | 17324 (1.3403%) of records are exact duplicates of another record; 3,861 distinct cases appear in more than one split (8,705 records), i.e. train/test leakage |
| 5 | `naive_bayes_reconstruction` | MAJOR | reconstructing the likelihood tables from corpus frequencies and recombining them by naive Bayes reproduces the shipped differentials at mean JSD 0.5463 (top-1 agreement 74.5%, top-5 Jaccard 0.257) against a generator noise floor of 0.0490 -- NOT adequate |
| 6 | `age_plausibility` | MINOR | 181,916 assertions of smoking / occupational-exposure / travel history in patients under 10 years old across 8 screened questions |
| 7 | `demographics` | INFO | age range [0, 109], median 39; sex values ['F', 'M']; out-of-range ages: 0 |
| 8 | `design_decision` | INFO | A2/A3 operate in the ABSOLUTE regime using the empirical-posterior oracle; naive-Bayes reconstruction is retained only as an approximate baseline (it alone would have forced the RELATIVE regime) |
| 9 | `differential_wellformed` | INFO | differentials sum to 1 within 1e-6 for 1292579 (100.0000%); length mismatches: 0; non-monotone orderings: 0; negatives: 0 |
| 10 | `empirical_oracle_feasibility` | INFO | conditioning directly on the corpus gives an assumption-free posterior; |E|=1: 100% scorable (median support 115346), |E|=2: 100% scorable (median support 23320), |E|=3: 96% scorable (median support 9136), |E|=4: 93% scorable (median support 4837), |E|=5: 85% scorable (median support 2705), |E|=6: 67% scorable (median support 1062), |E|=8: 42% scorable (median support 313), |E|=10: 13% scorable (median support 83) at a 500-patient minimum |
| 11 | `evidence_load` | INFO | evidences per patient: median 20, IQR 15-25, range 2-47; 1,722 patients above the Tukey fence (40) |
| 12 | `evidence_token_validity` | INFO | 516 distinct evidence tokens over 223 question codes; unknown codes: 0, invalid values: 0, malformed: 0 |
| 13 | `kb_coverage` | INFO | 0 (pathology, evidence) pairs observed in data but not declared in the knowledge base; 0 declared pathologies never observed; 0 declared evidences never observed |
| 14 | `oracle_vs_shipped_differential` | INFO | the empirical 3-finding posterior differs from the shipped full-case differential at mean JSD 0.539 over 4,903 comparable items |
| 15 | `schema_and_nulls` | INFO | 1,292,579 records; null fields: none; empty evidence lists: 0; empty differentials: 0 |
| 16 | `severity_coverage` | INFO | severity distribution (1 = most severe): sev1=99,860, sev2=262,456, sev3=389,404, sev4=401,406, sev5=139,453; 362,316 cases at severity <= 2 are available for the severity-weighted incoherence analysis |
| 17 | `sex_plausibility` | INFO | 0 sex-implausible evidence assertions (4 female-specific and 0 male-specific questions screened) |
| 18 | `split_shift` | INFO | pathology-marginal population stability index: train->test=0.0086, train->validate=0.0126  (<0.1 negligible, 0.1-0.25 moderate, >0.25 material) |
| 19 | `true_pathology_in_differential` | INFO | ground-truth pathology absent from its own differential in 0 (0.0000%); it is rank-1 in 73.60% of the rest (median rank 1) |

---

## Findings in detail

### 1. `generator_determinism` — CRITICAL

21,004 replicate case pairs; the generator returns a DIFFERENT differential for 25.82% of them (mean JSD 0.04895, p90 0.21979, max 0.52613); the top-1 diagnosis flips in 0.48% of replicate pairs

Two patients identical in evidence multiset, age, sex, initial evidence and true pathology can receive materially different ground-truth differentials. No released field explains the difference, so it is irreducible from the corpus alone. Consequence for this project: the ground truth itself has an order-of-1e-2 JSD noise floor. Any model order-effect must be reported against it, alongside the model's own test-retest floor. Consequence for the wider literature: DDXPlus differential-matching metrics have a hard ceiling that is not 1.0, and papers reporting near-perfect differential reproduction should be read with that in mind.

```json
{
  "n_replicate_pairs": 21004,
  "frac_pairs_disagreeing": 0.2581889163968768,
  "jsd_mean": 0.04895434666622638,
  "jsd_median": 0.0,
  "jsd_p90": 0.21978756874473587,
  "jsd_p99": 0.32775485055364706,
  "jsd_max": 0.5261316621231531,
  "l1_mean": 0.1684420244737566,
  "top1_flip_rate": 0.004760997905160921
}
```

### 2. `kb_probability_release` — CRITICAL

release_conditions.json declares 888 (pathology, evidence) slots and supplies conditional probabilities for 0 of them

This is the week-1 gate named in the proposal. The generator's likelihood tables are NOT part of the public DDXPlus release: every slot is an empty object. Exact posteriors for arbitrary evidence subsets therefore cannot be read off the knowledge base, and axioms A2 and A3 need another reference. Two were tested; see `naive_bayes_reconstruction` and `empirical_oracle_feasibility`.

```json
{
  "n_slots": 888,
  "n_with_probabilities": 0
}
```

### 3. `class_balance` — MAJOR

49 pathologies; prior range 0.00025-0.06326 (imbalance ratio 251.6x); prior entropy 5.38 bits of 5.61 max

Case sampling for the coherence battery must be stratified by pathology and by severity, otherwise the headline order-effect number is dominated by the handful of common respiratory pathologies and the severity-weighted analysis is starved of high-severity cases.

```json
{
  "n_pathologies": 49,
  "imbalance_ratio": 251.59076923076927,
  "prior_entropy_bits": 5.37797447067042,
  "severity_mass": {
    "1": 0.07725639980225579,
    "2": 0.20304832431905515,
    "3": 0.3012612768735992,
    "4": 0.3105465894154245,
    "5": 0.1078874095896653
  }
}
```

| pathology                | len   | prior    | severity |
|--------------------------|-------|----------|----------|
| URTI                     | 81767 | 0.063259 | 5        |
| Viral pharyngitis        | 78222 | 0.060516 | 4        |
| Anemia                   | 64410 | 0.049831 | 4        |
| HIV (initial infection)  | 36784 | 0.028458 | 3        |
| Anaphylaxis              | 35271 | 0.027287 | 1        |
| Localized edema          | 35253 | 0.027273 | 4        |
| Pulmonary embolism       | 34872 | 0.026979 | 2        |
| Influenza                | 33956 | 0.02627  | 3        |
| Bronchitis               | 33537 | 0.025946 | 4        |
| GERD                     | 32948 | 0.02549  | 3        |
| Acute otitis media       | 32907 | 0.025458 | 4        |
| Pneumonia                | 32787 | 0.025366 | 3        |
| Acute dystonic reactions | 32565 | 0.025194 | 2        |
| Panic attack             | 31643 | 0.024481 | 5        |
| Acute laryngitis         | 30753 | 0.023792 | 4        |
| Allergic sinusitis       | 30750 | 0.02379  | 4        |
| Pericarditis             | 28912 | 0.022368 | 4        |
| Guillain-Barré syndrome  | 28025 | 0.021681 | 2        |
| Sarcoidosis              | 27215 | 0.021055 | 4        |
| Possible NSTEMI / STEMI  | 27114 | 0.020977 | 1        |
| Unstable angina          | 26872 | 0.020789 | 2        |
| Atrial fibrillation      | 26476 | 0.020483 | 3        |
| Cluster headache         | 26470 | 0.020478 | 3        |
| Chronic rhinosinusitis   | 26043 | 0.020148 | 5        |
| Inguinal hernia          | 25618 | 0.019819 | 3        |

### 4. `duplicates` — MAJOR

17324 (1.3403%) of records are exact duplicates of another record; 3,861 distinct cases appear in more than one split (8,705 records), i.e. train/test leakage

A case is keyed on (evidence multiset, age, sex, pathology, initial evidence). Cross-split repetition means any supervised model trained on DDXPlus train and scored on DDXPlus test has memorisation headroom. It does not affect the coherence measurements in this project, which are within-case and label-free, but it is reported because downstream users of the corpus are affected.

```json
{
  "n_exact_duplicate_records": 17324,
  "n_cases_in_multiple_splits": 3861,
  "n_records_in_multiple_splits": 8705
}
```

### 5. `naive_bayes_reconstruction` — MAJOR

reconstructing the likelihood tables from corpus frequencies and recombining them by naive Bayes reproduces the shipped differentials at mean JSD 0.5463 (top-1 agreement 74.5%, top-5 Jaccard 0.257) against a generator noise floor of 0.0490 -- NOT adequate

The per-table estimates are precise (binomial SE ~3e-4); the failure is the recombination rule, not the estimation. Sampled evidence pairs within a pathology have mean |phi| 0.030 (p95 0.167, max 1.000), with 8.7% of pairs exceeding |phi|=0.1, so conditional independence given the pathology is violated materially and summing log-likelihood-ratios over ~20 correlated findings compounds the error. Temperature scaling does not rescue it, which confirms the defect is structural rather than a calibration issue. Consequence: naive Bayes is retained ONLY as an explicitly approximate baseline and as the object of the independence-correction ablation, never as ground truth.

```json
{
  "validation": {
    "n_cases": 20000,
    "present_only": true,
    "jsd_mean": 0.5462949577244856,
    "jsd_median": 0.6167761222477275,
    "jsd_p90": 0.7738178771449036,
    "top1_agreement": 0.7445,
    "top5_jaccard": 0.256915873015873
  },
  "independence": {
    "n_pairs_tested": 3600,
    "phi_mean_abs": 0.0296465679272286,
    "phi_median_abs": 0.006114326405204689,
    "phi_p95_abs": 0.16672074786549956,
    "phi_max_abs": 1.0,
    "frac_pairs_abs_phi_gt_0.1": 0.08722222222222223,
    "mutual_information_nats_mean": 0.0037327927233939376,
    "mutual_information_nats_p95": 0.0236024717303552
  },
  "generator_noise_floor_jsd": 0.04895434666622638
}
```

### 6. `age_plausibility` — MINOR

181,916 assertions of smoking / occupational-exposure / travel history in patients under 10 years old across 8 screened questions

Reported as a realism limitation of the synthetic generator, not as a defect that threatens the coherence measurement: order invariance holds regardless of whether the case is clinically plausible. It does matter for the MIMIC external-validity arm, which is the reason that arm exists.

```json
{
  "n_under_10": 181916,
  "screened_codes": [
    "E_191",
    "E_198",
    "E_199",
    "E_200",
    "E_204",
    "E_222",
    "E_49",
    "E_79"
  ]
}
```

| code  | n       | min_age | p01_age | question                          |
|-------|---------|---------|---------|-----------------------------------|
| E_49  | 124744  | 0       | 0.0     | Do you attend or work in a day... |
| E_204 | 1292579 | 0       | 0.0     | Have you traveled out of the c... |
| E_198 | 29137   | 0       | 0.0     | Do you work in agriculture?       |
| E_222 | 87448   | 0       | 0.0     | Are you exposed to secondhand ... |
| E_79  | 363510  | 0       | 0.0     | Do you smoke cigarettes?          |
| E_191 | 30739   | 18      | 19.0    | Are you a former smoker?          |
| E_199 | 12238   | 30      | 30.0    | Do you work in construction?      |
| E_200 | 13767   | 30      | 30.0    | Do you work in the mining sect... |

### 7. `demographics` — INFO

age range [0, 109], median 39; sex values ['F', 'M']; out-of-range ages: 0

```json
{
  "age_quantiles": {
    "min": 0.0,
    "p1": 0.0,
    "p25": 22.0,
    "p50": 39.0,
    "p75": 56.0,
    "p99": 98.0,
    "max": 109.0
  },
  "sex_values": [
    "F",
    "M"
  ],
  "n_age_out_of_range": 0
}
```

### 8. `design_decision` — INFO

A2/A3 operate in the ABSOLUTE regime using the empirical-posterior oracle; naive-Bayes reconstruction is retained only as an approximate baseline (it alone would have forced the RELATIVE regime)

A1 (order invariance) and A4 (positional anchoring) require no ground truth and are unaffected by any of this. A2 (update fidelity) and A3 (redundancy insensitivity) use the empirical oracle with a 500-patient minimum support and Wilson intervals on every target, restricted to evidence subsets of size <= 6. Every divergence figure in the paper is reported against three anchors: the model's test-retest floor, the generator's own replicate noise floor measured here (0.0490 JSD), and a shuffled-evidence ceiling.

```json
{
  "regime_naive_bayes_would_force": "RELATIVE",
  "generator_noise_floor_jsd": 0.04895434666622638
}
```

### 9. `differential_wellformed` — INFO

differentials sum to 1 within 1e-6 for 1292579 (100.0000%); length mismatches: 0; non-monotone orderings: 0; negatives: 0

```json
{
  "n_not_normalised": 0,
  "n_length_mismatch": 0,
  "n_non_monotone": 0,
  "max_abs_dev_from_1": 8.881784197001252e-16,
  "k_quantiles": {
    "min": 1.0,
    "p25": 4.0,
    "p50": 8.0,
    "p75": 13.0,
    "p95": 21.0,
    "max": 34.0
  }
}
```

### 10. `empirical_oracle_feasibility` — INFO

conditioning directly on the corpus gives an assumption-free posterior; |E|=1: 100% scorable (median support 115346), |E|=2: 100% scorable (median support 23320), |E|=3: 96% scorable (median support 9136), |E|=4: 93% scorable (median support 4837), |E|=5: 85% scorable (median support 2705), |E|=6: 67% scorable (median support 1062), |E|=8: 42% scorable (median support 313), |E|=10: 13% scorable (median support 83) at a 500-patient minimum

Evidence sets up to |E|=5 keep at least 80% of items scorable, which covers the absolute-reference requirements of A2 and A3. A1 and A4 need no ground truth at all and so run at any evidence-set size, including the full ~20-finding cases and the MIMIC narratives. This is the resolution of the week-1 gate: A2/A3 are ABSOLUTE measures, not relative ones, and the reference carries no independence assumption for a reviewer to attack -- it is the generator's own joint distribution, counted.

```json
{
  "min_support": 500,
  "max_k_80pct_scorable": 5,
  "by_k": {
    "1": {
      "median_support": 115346.0,
      "p10_support": 21572.0,
      "frac_scorable": 1.0
    },
    "2": {
      "median_support": 23319.5,
      "p10_support": 6432.200000000003,
      "frac_scorable": 1.0
    },
    "3": {
      "median_support": 9136.0,
      "p10_support": 1542.4000000000003,
      "frac_scorable": 0.9649122807017544
    },
    "4": {
      "median_support": 4837.0,
      "p10_support": 769.8,
      "frac_scorable": 0.9323308270676691
    },
    "5": {
      "median_support": 2705.0,
      "p10_support": 311.30000000000007,
      "frac_scorable": 0.8492462311557789
    },
    "6": {
      "median_support": 1061.5,
      "p10_support": 125.00000000000001,
      "frac_scorable": 0.6709183673469388
    },
    "8": {
      "median_support": 313.0,
      "p10_support": 39.0,
      "frac_scorable": 0.41689373297002724
    },
    "10": {
      "median_support": 83.0,
      "p10_support": 5.0,
      "frac_scorable": 0.13353115727002968
    }
  }
}
```

| evidence_set_size | median_support | p10_support | frac_scorable |
|-------------------|----------------|-------------|---------------|
| 1                 | 115346.0       | 21572.0     | 1.0           |
| 2                 | 23319.5        | 6432.2      | 1.0           |
| 3                 | 9136.0         | 1542.4      | 0.964912      |
| 4                 | 4837.0         | 769.8       | 0.932331      |
| 5                 | 2705.0         | 311.3       | 0.849246      |
| 6                 | 1061.5         | 125.0       | 0.670918      |
| 8                 | 313.0          | 39.0        | 0.416894      |
| 10                | 83.0           | 5.0         | 0.133531      |

### 11. `evidence_load` — INFO

evidences per patient: median 20, IQR 15-25, range 2-47; 1,722 patients above the Tukey fence (40)

This distribution sets the feasible range for the evidence-set-size ablation and bounds the cost of ELR-Fusion, which issues one query per finding.

```json
{
  "quantiles": {
    "min": 2.0,
    "p1": 5.0,
    "p25": 15.0,
    "p50": 20.0,
    "p75": 25.0,
    "p99": 37.0,
    "max": 47.0
  },
  "tukey_fence": 40.0,
  "n_above_fence": 1722,
  "mean": 19.774255964238936
}
```

### 12. `evidence_token_validity` — INFO

516 distinct evidence tokens over 223 question codes; unknown codes: 0, invalid values: 0, malformed: 0

```json
{
  "n_distinct_tokens": 516,
  "unknown_codes": [],
  "invalid_values": [],
  "malformed": []
}
```

| tok          | n       |
|--------------|---------|
| E_204_@_V_10 | 1194404 |
| E_53         | 1000618 |
| E_57_@_V_123 | 737724  |
| E_66         | 503146  |
| E_201        | 402018  |
| E_54_@_V_161 | 391628  |
| E_79         | 363510  |
| E_54_@_V_192 | 281871  |
| E_91         | 274187  |
| E_181        | 254993  |
| E_131_@_V_10 | 221357  |
| E_129        | 212305  |
| E_54_@_V_179 | 208850  |
| E_54_@_V_181 | 196997  |
| E_55_@_V_89  | 192707  |
| E_54_@_V_183 | 187164  |
| E_135_@_V_12 | 179737  |
| E_54_@_V_154 | 174378  |
| E_124        | 170880  |
| E_55_@_V_29  | 170116  |
| E_50         | 163694  |
| E_78         | 150052  |
| E_55_@_V_101 | 149601  |
| E_55_@_V_159 | 148383  |
| E_55_@_V_55  | 147882  |

### 13. `kb_coverage` — INFO

0 (pathology, evidence) pairs observed in data but not declared in the knowledge base; 0 declared pathologies never observed; 0 declared evidences never observed

Declared-but-unobserved is benign. Observed-but-undeclared means the structural KB cannot be used as a hard support constraint when reconstructing likelihoods: zeroing out undeclared evidence would assign probability 0 to cases that the generator actually produced.

```json
{
  "n_violating_pairs": 0,
  "unseen_pathologies": [],
  "unseen_evidence_codes": []
}
```

### 14. `oracle_vs_shipped_differential` — INFO

the empirical 3-finding posterior differs from the shipped full-case differential at mean JSD 0.539 over 4,903 comparable items

This is expected and is not a defect: the two quantities condition on different evidence. The shipped differential conditions on the patient's entire finding set, the empirical oracle on the 3-finding subset actually presented. The comparison is reported so that the report does not silently imply the two are interchangeable. Each axiom states which reference it uses.

```json
{
  "n_items": 4903,
  "jsd_mean": 0.539152245348315,
  "jsd_median": 0.5867368589938775,
  "median_support": 9687.0
}
```

### 15. `schema_and_nulls` — INFO

1,292,579 records; null fields: none; empty evidence lists: 0; empty differentials: 0

```json
{
  "n_rows": 1292579,
  "nulls": {
    "row_id": 0,
    "age": 0,
    "sex": 0,
    "pathology": 0,
    "initial_evidence": 0,
    "evidences": 0,
    "ddx_names": 0,
    "ddx_probs": 0,
    "split": 0
  },
  "empty_evidences": 0,
  "empty_differentials": 0
}
```

### 16. `severity_coverage` — INFO

severity distribution (1 = most severe): sev1=99,860, sev2=262,456, sev3=389,404, sev4=401,406, sev5=139,453; 362,316 cases at severity <= 2 are available for the severity-weighted incoherence analysis

```json
{
  "counts": {
    "1": 99860,
    "2": 262456,
    "3": 389404,
    "4": 401406,
    "5": 139453
  },
  "n_high_severity": 362316
}
```

| severity | len    |
|----------|--------|
| 1        | 99860  |
| 2        | 262456 |
| 3        | 389404 |
| 4        | 401406 |
| 5        | 139453 |

### 17. `sex_plausibility` — INFO

0 sex-implausible evidence assertions (4 female-specific and 0 male-specific questions screened)

Screened by question wording, so it is a lower bound and may contain false positives where a question merely mentions an anatomical term. Any true positives are generator artefacts and the affected cases are excluded from the clinical plausibility subset used for the clinician-review arm.

```json
{
  "n_violations": 0,
  "female_specific_codes": [
    "E_163",
    "E_145",
    "E_167",
    "E_141"
  ],
  "male_specific_codes": [],
  "example_questions": {
    "E_163": "Have you had any vaginal discharge?",
    "E_145": "Do you have very abundant or very long menstruation periods?",
    "E_167": "Do you think you are pregnant or are you currently pregnant?",
    "E_141": "Did you have your first menstrual period before the age of 12?"
  }
}
```

### 18. `split_shift` — INFO

pathology-marginal population stability index: train->test=0.0086, train->validate=0.0126  (<0.1 negligible, 0.1-0.25 moderate, >0.25 material)

```json
{
  "psi": {
    "train->test": 0.008589547528444237,
    "train->validate": 0.012601717449373686
  },
  "split_sizes": {
    "test": 134529,
    "validate": 132448,
    "train": 1025602
  }
}
```

| pathology                         | test | validate | train |
|-----------------------------------|------|----------|-------|
| Boerhaave                         | 2083 | 2075     | 15080 |
| Localized edema                   | 3734 | 3694     | 27825 |
| SLE                               | 1564 | 1579     | 11867 |
| Bronchospasm / acute asthma ex... | 2222 | 2209     | 19875 |
| Scombroid food poisoning          | 2486 | 2250     | 18535 |
| HIV (initial infection)           | 3919 | 3852     | 29013 |
| Bronchitis                        | 3594 | 3543     | 26400 |
| Guillain-Barré syndrome           | 2601 | 2557     | 22867 |
| Unstable angina                   | 2880 | 2748     | 21244 |
| Influenza                         | 3554 | 3590     | 26812 |
| Stable angina                     | 2386 | 2340     | 16995 |
| Possible NSTEMI / STEMI           | 2911 | 2943     | 21260 |
| Acute pulmonary edema             | 2598 | 2500     | 19018 |
| Myasthenia gravis                 | 2215 | 2159     | 18566 |
| Acute laryngitis                  | 3217 | 3407     | 24129 |
| Pneumonia                         | 3542 | 3484     | 25761 |
| Epiglottitis                      | 2364 | 2248     | 17209 |
| Acute otitis media                | 3516 | 3474     | 25917 |
| Spontaneous pneumothorax          | 1343 | 1405     | 10162 |
| Anemia                            | 6842 | 6903     | 50665 |
| Acute rhinosinusitis              | 1829 | 1866     | 13578 |
| Allergic sinusitis                | 2411 | 2136     | 26203 |
| Anaphylaxis                       | 3799 | 3754     | 27718 |
| Pulmonary neoplasm                | 1918 | 1891     | 14457 |
| Spontaneous rib fracture          | 778  | 782      | 5712  |

### 19. `true_pathology_in_differential` — INFO

ground-truth pathology absent from its own differential in 0 (0.0000%); it is rank-1 in 73.60% of the rest (median rank 1)

Rank-1 agreement well below 100% is expected and is not an error: the differential is the generator's posterior, while PATHOLOGY is the sampled ground truth. It does bound achievable top-1 accuracy for any method, and that bound belongs in the results table.

```json
{
  "n_truth_not_in_ddx": 0,
  "top1_rate": 0.7360424391855352,
  "mean_rank": 1.5774587085199434,
  "median_rank": 1.0,
  "top5_rate": 0.9731730130228017
}
```

---

## Provenance

```json
{
  "generated_utc": "2026-09-12T03:49:01+00:00",
  "git_commit": "8cd9c1b",
  "python": "3.12.14",
  "platform": "Linux-7.0.0-31-generic-x86_64-with-glibc2.43",
  "gpu": "NVIDIA RTX PRO 5000 Blackwell, 48935 MiB, 595.91.07",
  "polars": "1.44.2",
  "numpy": "2.3.5"
}
```