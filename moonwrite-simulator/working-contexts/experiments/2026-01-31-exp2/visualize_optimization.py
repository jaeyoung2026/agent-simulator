#!/usr/bin/env python3
"""
Visualization of Cost Optimization
Compares Optimized vs Full Pro approaches
"""

import json

# Load results
with open('exp5-premium-distributed-optimized.json', 'r') as f:
    data = json.load(f)

cost = data['cost_estimate']
opt = cost['optimization_analysis']

print("\n" + "="*80)
print("PREMIUM + DISTRIBUTED: OPTIMIZED MODEL ALLOCATION")
print("="*80)

# Model allocation visualization
print("\n┌─ MODEL ALLOCATION ─────────────────────────────────────────────────────────┐")
print("│                                                                             │")
print("│  Step 1: Thesis Extraction        [Flash/Sonnet]  ████████████████████     │")
print("│  Step 2: Cluster Analysis (||)    [Flash/Sonnet]  ██████████               │")
print("│  Step 3: Consistency Validation   [Flash/Sonnet]  ██████████               │")
print("│  Step 4: Quality Verification     [Pro/Opus]      ███                       │")
print("│                                                                             │")
print("│  Flash: ████████████████████████████████████████ 90.0%                     │")
print("│  Pro:   █████                                     10.0%                     │")
print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

# Cost comparison
print("\n┌─ COST COMPARISON ──────────────────────────────────────────────────────────┐")
print("│                                                                             │")

optimized_cost = opt['optimized_cost']
full_pro_cost = opt['full_pro_cost']
savings = opt['cost_savings']

# Visual bars
opt_bar_len = int((optimized_cost / full_pro_cost) * 50)
full_bar_len = 50

print(f"│  Optimized:  ${'█' * opt_bar_len:<50s}  ${optimized_cost:.4f}          │")
print(f"│  Full Pro:   ${'█' * full_bar_len:<50s}  ${full_pro_cost:.4f}          │")
print("│                                                                             │")
print(f"│  💰 SAVINGS: ${savings:.4f} ({opt['savings_percentage']:.1f}%)                                         │")
print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

# Token distribution
print("\n┌─ TOKEN DISTRIBUTION ───────────────────────────────────────────────────────┐")
print("│                                                                             │")

flash_total = cost['flash_tokens']['total']
pro_total = cost['pro_tokens']['total']
total_tokens = flash_total + pro_total

print(f"│  Step 1 (Flash):  {cost['flash_tokens']['breakdown']['step1']:>6,} tokens  ████████████████     │")
print(f"│  Step 2 (Flash):  {cost['flash_tokens']['breakdown']['step2']:>6,} tokens  ██████               │")
print(f"│  Step 3 (Flash):  {cost['flash_tokens']['breakdown']['step3']:>6,} tokens  ██████               │")
print(f"│  Step 4 (Pro):    {cost['pro_tokens']['breakdown']['step4']:>6,} tokens  ██                   │")
print("│                                                                             │")
print(f"│  Total: {total_tokens:,} tokens                                                   │")
print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

# Quality metrics
print("\n┌─ QUALITY METRICS ──────────────────────────────────────────────────────────┐")
print("│                                                                             │")

qm = data['quality_metrics']

def quality_bar(score, label, width=40):
    filled = int(score * width)
    bar = '█' * filled + '░' * (width - filled)
    return f"│  {label:<25} [{bar}] {score:.2f}  │"

print(quality_bar(qm['overall_quality'], "Overall Quality"))
print(quality_bar(qm['consistency_score'], "Consistency"))
print(quality_bar(qm['alignment_rate'], "Alignment Rate"))
print(quality_bar(qm['avg_self_critique'], "Self-Critique Avg"))
print(quality_bar(qm['avg_principle_score'], "Writing Principles"))
print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

# Processing stats
print("\n┌─ PROCESSING STATISTICS ────────────────────────────────────────────────────┐")
print("│                                                                             │")
print(f"│  Slides Processed:        {data['total_slides']:<10}                                       │")
print(f"│  Semantic Units:          {data['total_units']:<10}                                       │")
print(f"│  Avg Units/Slide:         {data['avg_units_per_slide']:<10.2f}                                       │")
print(f"│  Images Analyzed:         {data['step1_results']['image_analysis']['total_images']:<10}                                       │")
print(f"│  Clusters Created:        {len(data['step2_results']['cluster_analyses']):<10}                                       │")
print(f"│  Domains Identified:      {len(data['step2_results']['domain_summary']):<10}                                       │")
print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

# Domain breakdown
print("\n┌─ DOMAIN ANALYSIS ──────────────────────────────────────────────────────────┐")
print("│                                                                             │")

domain_summary = data['step2_results']['domain_summary']
max_clusters = max(domain_summary.values())

for domain, count in sorted(domain_summary.items(), key=lambda x: x[1], reverse=True):
    bar_len = int((count / max_clusters) * 40)
    bar = '█' * bar_len
    print(f"│  {domain:<25} {bar:<40} {count:>2}  │")

print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

# Connection strength
print("\n┌─ THESIS CONNECTION STRENGTH ───────────────────────────────────────────────┐")
print("│                                                                             │")

conn_dist = data['step2_results']['connection_strength_distribution']
total_clusters = sum(conn_dist.values())

for strength in ['high', 'medium', 'low']:
    if strength in conn_dist:
        count = conn_dist[strength]
        pct = (count / total_clusters * 100) if total_clusters > 0 else 0
        bar_len = int(pct / 2)
        bar = '█' * bar_len
        print(f"│  {strength.capitalize():<15} {bar:<50} {count:>2} ({pct:>5.1f}%)  │")

print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

# Step 4 results (Pro model)
print("\n┌─ STEP 4: QUALITY VERIFICATION (Pro/Opus) ──────────────────────────────────┐")
print("│                                                                             │")

s4 = data['step4_results']
print(f"│  Quality Score:           {s4['quality_score']:.2f}                                          │")
print(f"│  Total Gaps Found:        {s4['summary']['total_gaps']:<10}                                       │")
print(f"│  High Severity Gaps:      {s4['summary']['high_severity_gaps']:<10}                                       │")
print(f"│  Medium Severity Gaps:    {s4['summary']['medium_severity_gaps']:<10}                                       │")
print(f"│  Sections Verified:       {s4['summary']['sections_verified']:<10}                                       │")
print(f"│  Needs Work:              {s4['summary']['sections_needing_work']:<10}                                       │")
print("│                                                                             │")

# Writing principles
print("│  Writing Principles Evaluation:                                             │")
for principle, details in s4['writing_principles_evaluation'].items():
    score = details['score']
    bar_len = int(score * 30)
    bar = '█' * bar_len + '░' * (30 - bar_len)
    print(f"│    {principle.replace('_', ' ').title():<25} [{bar}] {score:.2f}      │")

print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

# Key takeaways
print("\n┌─ KEY TAKEAWAYS ────────────────────────────────────────────────────────────┐")
print("│                                                                             │")
print("│  ✓ 84.6% cost reduction compared to full Pro approach                      │")
print("│  ✓ 90% of processing done with Flash (Sonnet) - cost-efficient             │")
print("│  ✓ 10% reserved for Pro (Opus) - strategic quality verification            │")
print("│  ✓ Quality maintained at 0.77 (Good) with excellent consistency (0.89)     │")
print("│  ✓ 96% alignment rate - most units properly connected to thesis            │")
print("│  ✓ Scalable pattern for large-scale analysis projects                      │")
print("│                                                                             │")
print("│  RECOMMENDATION: Ideal for budget-conscious research analysis with         │")
print("│                  100+ slides, achieving professional quality at            │")
print("│                  a fraction of the cost.                                    │")
print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

print("\n")
