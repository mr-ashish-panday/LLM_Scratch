---
name: Research Title Architect
description: Generates optimized, publication-ready research paper titles using bibliometric frameworks (PICO, SPICED, SPIDER, ECLIPSE), ASEO principles, and discipline-specific conventions. Triggered for any academic titling task.
---

# Research Title Architect

## Purpose

This skill provides the agent with a complete, rigorous system for generating optimized academic research paper titles. When a user is working on a research paper, manuscript, or academic publication and needs a title, the agent activates this skill to produce 3-5 publication-ready title variants that are:
- **Bibliometrically optimized** for maximum citation impact
- **Algorithmically tuned** for Academic Search Engine Optimization (ASEO)
- **Discipline-calibrated** for STEM, Social Sciences, or Humanities conventions
- **Journal-compliant** with specific character/formatting constraints

---

## When to Activate

Trigger this skill when the user:
- Asks for help naming/titling a research paper
- Is drafting or finalizing an academic manuscript
- Asks for feedback on an existing research title
- Needs to optimize a title for a specific journal or database
- Is preparing a submission and mentions title requirements

---

## The Golden Rule

> [!CAUTION]
> **NEVER finalize a title before the paper is written.** The title must be crafted AFTER the abstract, data analysis, and conclusion are complete. A working title during drafting is fine, but the publication-ready title must reflect the final, proven arguments — not aspirational claims.

---

## Step 1: Identify the Research Type → Select Framework

Before generating anything, determine the research type and select the appropriate framework:

| Research Type | Framework | When to Use |
|:---|:---|:---|
| Clinical trials, RCTs, direct comparisons | **PICO** | Clear intervention vs. control with measured outcome |
| Complex/longitudinal clinical studies | **PICOTTS** | Need to specify time, study type, and setting |
| Public health, epidemiology, policy evaluation | **SPICED** | Broad observational studies, multi-variable analyses |
| Qualitative / mixed-methods research | **SPIDER** | Subjective experiences, social phenomena, behaviors |
| Health policy, service evaluation, management | **ECLIPSE** | Operational research, administrative evaluations |
| Project proposals, quality improvement | **SPICE** | Emphasizing perspective and viewpoint being analyzed |
| Social science (Psychology, Education, etc.) | **APA Variables** | Clear independent/dependent variables |
| Humanities (Literature, History, Philosophy) | **Colon Paradigm** | Interpretive analysis, textual criticism |

> [!TIP]
> See `resources/framework_reference.md` for full component breakdowns of each framework.

---

## Step 2: The 6-Step Title Generation Algorithm

Execute these steps sequentially. Do NOT skip steps.

### 2.1 — Core Narrative Extraction
Draft **3 complete sentences** summarizing the paper's intent. These must answer:
- **Who** — the population/sample/cohort
- **What** — the intervention, phenomenon, or variable
- **Where** — the setting (only if it modifies outcomes)
- **How** — the methodology/design

Use the selected framework to ensure all required components are captured.

### 2.2 — Keyword Harvesting & Database Verification
- Extract primary variables, phenomena, and interventions from the 3 sentences
- Cross-reference terms against **MeSH databases** (for medical/life sciences) or discipline-specific thesauri
- Test keywords on PubMed/Google Scholar — if results are too generic (millions of hits), **refine to more specific terms**
  - ❌ "cancer risk" → ✅ "neoplastic polyp growth rate"
  - ❌ "gall stones" → ✅ "gallbladder polyps growth rate"
- Draft 5-7 **separate metadata keywords** that are NOT already in the title (eliminate the 92% redundancy error)

### 2.3 — Syntactic Compression
Combine the 3 narrative sentences into a single comprehensive string. It will be 30-40 words at this stage — that's expected.

### 2.4 — Lexical Pruning & Typological Alignment

**Kill these filler phrases immediately:**
- "A study of…" / "An investigation into…" / "Research on…"
- "Observations regarding…" / "More about…" / "…revisited"
- Leading "The" (wastes prime keyword space)
- Unnecessary prepositions — use compounded noun phrases instead

**Then align with typology:**

| Typology | Structure | Citation Impact | Use When |
|:---|:---|:---|:---|
| **Descriptive** | Noun phrases, no active verbs. States variables/methods. | Baseline standard | Conservative journals, clinical trials, when objectivity is paramount |
| **Declarative** | Active verbs stating the main finding/conclusion | **Highest citations** | Biology, rapid-communication journals, when results are novel and strong |
| **Interrogative** | Phrased as a question | **Lowest citations — AVOID** | Only for review articles/editorials. Never for primary research. |

> [!WARNING]
> **Interrogative titles are a citation killer.** They reduce keyword density AND signal uncertainty. Unless the user specifically requests one for a review/editorial, always steer toward Descriptive or Declarative.

### 2.5 — Algorithmic Testing & ASEO Verification

Run this checklist on the pruned title:

| Check | Rule | Why |
|:---|:---|:---|
| **First 65 characters** | Must contain the most critical 1-2 key phrases | Search engines assign disproportionate weight to early-position words |
| **Total length** | 10-16 words / 100-120 characters (general) | Optimal for both human cognition and algorithmic indexing |
| **Journal-specific limit** | See §4 Journal Constraints table | Non-compliance = desk rejection |
| **No weak keywords leading** | Avoid starting with: incidence, prevalence, outcomes, rapid, new, assessment, effects, method | These appear in millions of titles and provide zero distinguishing power |
| **No non-standard abbreviations** | Only universally recognized abbreviations allowed (DNA, RNA, HIV, AIDS, FDA, ECG, EEG) | Unrecognized abbreviations kill discoverability |
| **No formulas/special characters** | Strip mathematical notation from the title | Database parsers corrupt non-standard characters in metadata exports |
| **Keyword metadata is non-redundant** | The 5-7 keywords must NOT repeat title/abstract terms | 92% of authors waste this indexing real estate — don't be one of them |

### 2.6 — Generate Variants & Present

Produce **3-5 distinct title variants**:
1. One **descriptive** (noun-heavy, methods-focused)
2. One **declarative** (active verb, states the finding)
3. One **colon-structured** (hook : specifics) — if discipline allows
4. Optionally: one ultra-short variant for high-impact journals (Nature/Science)
5. Optionally: one expanded variant for mega-journals (PLOS ONE)

For each variant, briefly note:
- Character count and word count
- Which databases/journals it's optimized for
- Which typology it uses
- Any trade-offs made

---

## Step 3: Discipline Calibration

The agent MUST calibrate the title style to the user's academic discipline:

### STEM (Science, Technology, Engineering, Medicine)
- **Objective**: Rapid transmission of empirical facts
- **Style**: High density of common nouns, specific taxonomic data, zero filler, emotional detachment
- **Length**: Short (10-15 words)
- **Rules**: 
  - Use exact gene markers, chemical compounds, binomial species names
  - ❌ "The control of organ development in fish"
  - ✅ "The novel gene 'exdpf' regulates pancreas development in zebrafish"
- **Avoid**: Adjectives, flourishes, rhetorical devices

### Social Sciences (Psychology, Sociology, Education)
- **Objective**: Clear articulation of variables and theoretical relationships
- **Style**: Relational phrasing (Effect of X on Y), clear IV/DV identification
- **Length**: Moderate (12-18 words)
- **Structure**: "Effect of [IV] on [DV] in [Population]: A [Design]"
- **Follow**: APA style — title case for manuscript, sentence case for reference lists

### Humanities (Literature, History, Philosophy, Arts)
- **Objective**: Persuasive framing of an interpretive argument
- **Style**: Descriptive adjectives, literary allusions, proper nouns, rhetorical devices welcome
- **Length**: Longer (15-25+ words)
- **Structure**: Evocative hook : Specific textual analysis (colon paradigm is dominant)
- **Rules**:
  - Short story/poem/essay titles → quotation marks in the title
  - Novel/play/film titles → italics
  - Follow MLA or Chicago style

---

## Step 4: Journal-Specific Constraints

If the user specifies a target journal, apply these hard constraints:

| Journal | Max Length | Special Rules |
|:---|:---|:---|
| **Nature** | ~66 characters (incl. spaces) | NO punctuation. NO abbreviations. Single flowing sentence only. |
| **Science** | <100 chars (120 absolute max) | Prefer brevity, allow standard structure |
| **Cell** | 120 characters | Standard constraints |
| **NEJM** | ~15 words avg (up to 148 in complex trials) | Must include study design. Descriptive preferred. |
| **The Lancet** | Strict word count per type | Descriptive only. No passive voice. No abbreviations. No questions. |
| **JMIR** | 280 characters | Empirical: descriptive + title case + study type. Viewpoints: flexible. |
| **PLOS ONE** | 250 characters | Allows detailed, niche descriptive titles |

> [!IMPORTANT]
> **Nature's rules are the strictest in publishing.** Zero punctuation means NO colons, NO hyphens, NO parentheses. The entire title must be a single, perfectly flowing phrase under 66 characters. Always generate a Nature-compliant variant if the user targets multidisciplinary journals.

---

## Step 5: Capitalization Rules

| Style | Rule | Used In |
|:---|:---|:---|
| **Title Case** | Capitalize first, last, and all principal words (nouns, verbs, adjectives, adverbs). Lowercase articles, prepositions (<4 letters), conjunctions. | APA manuscripts, social sciences, most submission forms |
| **Sentence Case** | Capitalize only the first word, proper nouns, and the first word after a colon. | Natural science journals, medical publications, all reference lists |

> [!NOTE]
> Inconsistent capitalization is a hallmark of amateurish writing and triggers formatting rejections. Always verify the target journal's style guide.

---

## Step 6: Fatal Error Checklist

Before finalizing, scan the title for these deal-breakers:

| # | Error | Fix |
|:---|:---|:---|
| 1 | Starts with "A study of…" or "The" | Delete. Lead with the primary variable/keyword. |
| 2 | Contains non-standard abbreviations | Spell out or remove. Only DNA/RNA/HIV/AIDS/FDA/ECG/EEG allowed. |
| 3 | Phrased as a question | Rewrite as descriptive or declarative. Questions = lowest citations. |
| 4 | Contains humor/puns/clickbait | Remove. Risks alienating non-native speakers and damaging credibility. |
| 5 | Key phrases are after character 65 | Restructure to front-load critical keywords. |
| 6 | Geographic location included without scientific necessity | Remove unless the setting uniquely modifies outcomes. |
| 7 | Sample size or P-values in title | Remove. These belong in the abstract. |
| 8 | Exceeds target journal character limit | Prune further. Non-compliance = desk rejection. |
| 9 | Keywords metadata duplicates title words | Replace with synonyms, MeSH terms, or broader categorical identifiers. |
| 10 | Exaggerates scope beyond what data supports | Narrow claims to match actual findings. Academic clickbait is career poison. |

---

## Output Format

When generating titles, present results in this structure:

```
## Research Title Variants

**Framework Used**: [PICO/SPICED/SPIDER/etc.]
**Discipline**: [STEM/Social Science/Humanities]
**Target Journal**: [If specified]

### Variant 1 — Declarative
> [Title here]
- Words: X | Characters: X
- Optimized for: [databases/journals]
- Rationale: [why this structure]

### Variant 2 — Descriptive
> [Title here]
- Words: X | Characters: X
- ...

### Variant 3 — Colon-Structured
> [Title here]
- Words: X | Characters: X
- ...

### Recommended Keywords (Non-Redundant)
1. [keyword not in title]
2. [keyword not in title]
3. ...

### ⚠️ Title Elements Intentionally Excluded
- [Element]: [Reason for exclusion]
```

---

## Agent Behavior

When this skill is active:
1. **Always ask for the abstract/summary first** if the user hasn't provided paper details — you cannot title what you haven't read
2. **Never output a single title** — always produce 3-5 variants with rationale
3. **Flag interrogative requests** — if the user asks for a question-based title, warn about citation impact but comply if they insist
4. **Check journal constraints proactively** — ask the target journal if not provided
5. **Be honest about trade-offs** — if brevity sacrifices specificity, say so explicitly
6. **Treat the title as engineered metadata**, not creative writing — precision over poetry
