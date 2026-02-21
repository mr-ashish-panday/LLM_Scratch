# Framework Quick Reference

## PICO — Clinical Trials & Direct Comparisons

| Component | Definition | Title Example Fragment |
|:---|:---|:---|
| **P** — Population | Specific demographic, species, clinical group | "pediatric patients with allergic rhinitis" |
| **I** — Intervention | Primary independent variable, treatment, exposure | "Benzathine Penicillin Prophylaxis" |
| **C** — Comparison | Control group, placebo, alternative treatment | Often omitted for brevity unless comparison is the novelty |
| **O** — Outcome | Measured dependent variable, clinical endpoint | "mortality prediction," "clinical remission rates" |

---

## PICOTTS — Extended PICO for Complex Studies

Adds to PICO:

| Component | Definition |
|:---|:---|
| **T** — Time/Duration | Temporal scope of the study (e.g., "12-month follow-up") |
| **T** — Type of Study | Methodological design (e.g., "Randomized Controlled Trial") |
| **S** — Setting | Geographic or institutional context (e.g., "Tertiary Referral Center") |

---

## SPICED — Public Health & Epidemiology

| Component | Definition | Include in Title? |
|:---|:---|:---|
| **S** — Setting | Environment, institution, geography | Only if it uniquely modifies outcomes |
| **P** — Population | Target demographic or sample cohort | Yes — always |
| **I** — Intervention | Active agent, social policy, treatment | Yes — always |
| **C** — Condition | Disease, psychological state, systemic phenomenon | Yes — always |
| **E** — End-point | Primary evaluated outcome, systemic impact | Yes — always |
| **D** — Design | Methodological structure | Append as subtitle after colon |

> **Restraint Rule**: Do NOT force every SPICED element into the title. Omit Setting unless geography uniquely influences results. A pharma study "in Toronto" adds nothing; a tropical disease study "in Sub-Saharan Africa" is essential context.

---

## SPIDER — Qualitative & Mixed-Methods

| Component | Definition | Replaces in PICO |
|:---|:---|:---|
| **S** — Sample | The specific group studied | Population |
| **P** — Phenomenon of Interest | The social/psychological phenomenon | Intervention |
| **I** — Design | Qualitative methodology used | — |
| **D** — Evaluation | How the phenomenon was assessed | Outcome |
| **E** — Research Type | Qualitative, quantitative, or mixed | — |

---

## ECLIPSE — Health Policy & Service Evaluation

| Component | Definition |
|:---|:---|
| **E** — Expectation | What improvement/change is expected |
| **C** — Client group | Who receives the service |
| **L** — Location | Where the service is delivered |
| **I** — Impact | What the service achieves |
| **P** — Professionals | Who delivers the service |
| **SE** — Service | What service is being evaluated |

---

## SPICE — Project Proposals & Quality Improvement

| Component | Definition |
|:---|:---|
| **S** — Setting | Context of the project |
| **P** — Perspective | Whose viewpoint is analyzed |
| **I** — Intervention/Exposure/Interest | What is being studied |
| **C** — Comparison | What is the alternative |
| **E** — Evaluation | How success is measured |

---

## APA Variable Format — Social Sciences

**Structure**: `Effect of [Independent Variable] on [Dependent Variable] in [Population]`

Use when the research has clear IV → DV relationships in Psychology, Education, Sociology, or Behavioral Sciences.

---

## Decision Tree

```
Is the research clinical/medical?
├─ YES → Is there a clear intervention vs. control?
│   ├─ YES → Is it a simple RCT? → PICO
│   │         Is it longitudinal/complex? → PICOTTS
│   └─ NO → Is it public health/epidemiology? → SPICED
│            Is it health policy/admin? → ECLIPSE
├─ NO → Is it qualitative/mixed-methods?
│   ├─ YES → SPIDER
│   └─ NO → Is it a project proposal/QI?
│       ├─ YES → SPICE
│       └─ NO → Social science with clear variables? → APA Variables
│                Humanities/interpretive? → Colon Paradigm
```
