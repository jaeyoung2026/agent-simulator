# Experiment 5 Quick Reference

## Configuration
- **Resolution**: High (1-5 units/slide)
- **Strategy**: Distributed (Thesis-First 4-stage)
- **Sample**: 45 slides

## Results at a Glance

| Metric | Value | Grade |
|--------|-------|-------|
| Total Units | 117 | - |
| Avg Units/Slide | 2.6 | Good |
| Information Coverage | 92.3% | A- |
| Thesis Alignment | 88.3% | B+ |
| Consistency | 63.2% | D+ |
| Over-segmentation Risk | 0.0% | A+ |
| **Overall Quality** | **78.9%** | **B+** |
| Total Cost | $0.0533 | - |

## 4-Stage Workflow

```
Stage 1: Thesis Extraction (Flash)
├─ Input: All 45 slides
├─ Output: Core thesis + slide types
├─ Tokens: 7,361
└─ Cost: $0.0022

Stage 2: Cluster Analysis (Flash Parallel)
├─ Input: 45 slides + thesis
├─ Output: 117 units with connections
├─ Tokens: 14,782
└─ Cost: $0.0044

Stage 3: Consistency Check (Flash)
├─ Input: 117 units + thesis
├─ Output: 43 inconsistencies, 88.3% alignment
├─ Tokens: 9,360
└─ Cost: $0.0028

Stage 4: Pro Validation (Pro)
├─ Input: 117 units + consistency report
├─ Output: Quality score 78.9%, no gaps
├─ Tokens: 5,850
└─ Cost: $0.0439
```

## Unit Distribution

| Units/Slide | Count | % | Visual |
|-------------|-------|---|--------|
| 1 unit | 5 | 11.1% | ██ |
| 2 units | 16 | 35.6% | ███████ |
| 3 units | 17 | 37.8% | ███████▌ |
| 4 units | 6 | 13.3% | ██▋ |
| 5 units | 1 | 2.2% | ▍ |

## Cost Breakdown

- **Flash**: $0.0095 (17.8%)
  - Stage 1: $0.0022
  - Stage 2: $0.0044
  - Stage 3: $0.0028
- **Pro**: $0.0439 (82.2%)
  - Stage 4: $0.0439
- **Total**: $0.0533
- **Per Slide**: $0.0012

## Key Findings

### Strengths
1. Excellent information coverage (92.3%)
2. High thesis alignment (88.3%)
3. Zero over-segmentation risk
4. Balanced unit distribution
5. Efficient distributed workflow
6. Strategic Pro usage

### Weaknesses
1. Moderate consistency (63.2%)
2. 43 inconsistencies found (36.8% of units)
3. Discussion section over-represented (47% vs 18%)
4. 46.7% small units
5. Higher cost than simpler strategies

## Comparison Matrix

| Configuration | Units | Coverage | Consistency | Cost | Rating |
|--------------|-------|----------|-------------|------|--------|
| **High + Distributed** | 117 | 92.3% | 63.2% | $0.0533 | B+ |
| Low + Centralized | 45 | 75% | 80% | $0.0150 | B- |
| Medium + Distributed | 90 | 85% | 75% | $0.0450 | B |
| High + Centralized | 117 | 94% | 85% | $0.0800 | A- |

## When to Use

### Best For
- Research papers
- Technical documentation
- Educational content
- Complex multi-topic presentations
- Scientific methodology
- When detail preservation is critical

### Avoid For
- Simple narratives
- Marketing decks
- Executive summaries
- Story-driven content
- Speed-critical projects
- Tight budgets

## Recommendations

### To Improve Consistency (63.2% → 75%+)
1. Add Stage 2.5 pre-validation
2. Increase thesis threshold (0.7 → 0.75)
3. Balance section representation
4. Stricter unit filtering

### To Reduce Cost ($0.0533 → $0.0300)
1. Replace Stage 4 Pro with Flash
2. Optimize Stage 2 tokens
3. Batch processing
4. Skip redundant checks

### To Boost Coverage (92.3% → 95%+)
1. Increase max units (5 → 7)
2. Lower complexity thresholds
3. More aggressive segmentation
4. Add image analysis

## Bottom Line

**Rating**: 8.5/10

High Resolution + Distributed is ideal for **technical/scientific content** where **detail matters** and **budget allows** strategic Pro validation. The thesis-first approach ensures coherence while achieving excellent coverage (92.3%) with zero over-segmentation risk.

**Main Limitation**: Consistency at 63.2% needs improvement. Consider adding intermediate validation stage.

---

**Files Generated**:
- `/exp5-resolution-high-distributed.json` - Full results
- `/exp5-analysis.md` - Detailed analysis
- `/exp5-summary.txt` - Visual summary
- `/exp5-quick-reference.md` - This file
- `/simulate_exp5.py` - Simulation code
