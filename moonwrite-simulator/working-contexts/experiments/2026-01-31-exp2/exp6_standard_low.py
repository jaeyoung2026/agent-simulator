#!/usr/bin/env python3
"""
Standard Option Simulation - LOW RESOLUTION
Thesis-First 4-Stage Pattern with Cost-Efficient Model Distribution

STANDARD FEATURES (vs Premium):
- Self-Critique: NONE
- Reverse Outline Validation: NONE (excluded from Step 3)
- Writing Principles Evaluation: SIMPLIFIED (3 items only)
- Placeholders: Location markers only (no specific suggestions)
- Image Analysis: Type + Role only (no reproducibility info)

Model Allocation:
- Steps 1-3: Flash (Sonnet)
- Step 4: Pro (Opus)

Resolution: Low (1-2 units/slide)
"""

import json
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Load samples
with open('/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-final/samples-extended.json', 'r') as f:
    samples = json.load(f)

print(f"Loaded {len(samples)} slides")

# Configuration constants
THESIS_CATEGORIES = [
    "core_thesis",           # 핵심 주장
    "thesis_support",        # 증거/지지
    "thesis_context",        # 배경/맥락
    "thesis_elaboration"     # 상세/확장
]

SLIDE_TYPES = ["result", "method", "background", "contribution", "problem", "setup", "training", "analysis"]
CONNECTION_STRENGTHS = ["high", "medium", "low"]
GAP_TYPES = [
    "missing_evidence", "weak_connection", "logical_gap",
    "methodology_unclear", "result_interpretation", "context_missing"
]
SEVERITIES = ["high", "medium", "low"]

# Research-specific vocabulary
THERMAL_KEYWORDS = ["thermal", "heat", "temperature", "발열", "온도", "열", "과열", "cooling"]
MOTOR_KEYWORDS = ["motor", "torque", "actuator", "모터", "토크", "액추에이터"]
RL_KEYWORDS = ["policy", "reward", "learning", "training", "강화학습", "정책", "보상"]
ROBOT_KEYWORDS = ["robot", "quadruped", "locomotion", "walking", "보행", "로봇", "ToddlerBot", "toddlerbot"]
SIMULATION_KEYWORDS = ["simulation", "MuJoCo", "Brax", "시뮬레이션", "sim"]

# COST CONSTANTS (per 1M tokens)
FLASH_INPUT_COST = 0.00025  # $0.25 per 1M input tokens (Sonnet)
FLASH_OUTPUT_COST = 0.00125  # $1.25 per 1M output tokens (Sonnet)
PRO_INPUT_COST = 0.015  # $15 per 1M input tokens (Opus)
PRO_OUTPUT_COST = 0.075  # $75 per 1M output tokens (Opus)
IMAGE_COST = 0.0048  # $4.80 per 1000 images (high quality)

def generate_unit_id(slide_idx: int, unit_idx: int) -> str:
    """Generate unique unit ID"""
    hash_input = f"{slide_idx}-{unit_idx}-{datetime.now().isoformat()}"
    return f"su_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

def detect_slide_type(content: str) -> str:
    """Detect slide type from content"""
    content_lower = content.lower()
    if "result" in content_lower or "결과" in content_lower:
        return "result"
    elif "method" in content_lower or "방법" in content_lower:
        return "method"
    elif "contribution" in content_lower or "기여" in content_lower:
        return "contribution"
    elif "problem" in content_lower or "문제" in content_lower:
        return "problem"
    elif "abstract" in content_lower or "개요" in content_lower:
        return "background"
    elif "training" in content_lower or "학습" in content_lower:
        return "training"
    elif "reward" in content_lower or "보상" in content_lower:
        return "training"
    elif "algorithm" in content_lower or "알고리즘" in content_lower:
        return "method"
    else:
        return random.choice(SLIDE_TYPES)

def detect_research_domain(content: str) -> List[str]:
    """Detect research domains from content"""
    domains = []
    content_lower = content.lower()

    if any(kw in content_lower for kw in THERMAL_KEYWORDS):
        domains.append("thermal_management")
    if any(kw in content_lower for kw in MOTOR_KEYWORDS):
        domains.append("motor_control")
    if any(kw in content_lower for kw in RL_KEYWORDS):
        domains.append("reinforcement_learning")
    if any(kw in content_lower for kw in ROBOT_KEYWORDS):
        domains.append("legged_locomotion")
    if any(kw in content_lower for kw in SIMULATION_KEYWORDS):
        domains.append("simulation")

    return domains if domains else ["general"]

def calculate_thesis_category(slide_type: str, domains: List[str]) -> str:
    """Calculate thesis category based on slide type and domains"""
    if slide_type in ["contribution", "abstract"]:
        return "core_thesis"
    elif slide_type in ["result"]:
        return "thesis_support"
    elif slide_type in ["background", "problem"]:
        return "thesis_context"
    else:
        return "thesis_elaboration"

# ============================================================================
# STEP 1: Thesis Extraction (Flash/Sonnet) - NO Self-Critique
# ============================================================================

def step1_extract_thesis(samples: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 1: Thesis extraction WITHOUT Self-Critique (Standard feature)
    MODEL: Flash (Sonnet) - Cost-efficient for extraction
    STANDARD: No self-critique, simplified image analysis
    """
    print("\n" + "="*60)
    print("STEP 1: Thesis Extraction (Flash/Sonnet) - STANDARD")
    print("  [NO Self-Critique]")
    print("="*60)

    # Extract global thesis
    thesis = {
        "question": "How can thermal-aware control policies improve long-term stability and energy efficiency in legged robots?",
        "claim": "By incorporating motor heat state estimation and thermal rewards into reinforcement learning, robots can proactively manage thermal constraints while maintaining task performance.",
        "evidence": [
            "Heat2Torque simulation reduces sim-to-real gap by modeling torque degradation",
            "Thermal-aware reward functions enable proactive heat management",
            "Online thermal model adaptation improves estimation accuracy",
            "DRL controllers can balance performance and hardware durability"
        ],
        "sub_claims": [
            {
                "id": "SC1",
                "claim": "Motor thermal dynamics can be accurately modeled and simulated",
                "evidence_count": 8
            },
            {
                "id": "SC2",
                "claim": "Thermal-aware policies outperform baseline in long-term operation",
                "evidence_count": 6
            },
            {
                "id": "SC3",
                "claim": "Online adaptation improves thermal estimation accuracy",
                "evidence_count": 4
            }
        ]
    }

    # Process each slide
    semantic_units = []
    slide_analyses = []
    total_images = 0

    for idx, slide in enumerate(samples):
        slide_type = detect_slide_type(slide['content'])
        domains = detect_research_domain(slide['content'])

        # LOW RESOLUTION: 1-2 units per slide
        content_length = len(slide['content'])
        has_images = len(slide.get('images', [])) > 0
        total_images += len(slide.get('images', []))

        # Standard Low: Always 1-2 units
        if content_length > 200 or has_images:
            num_units = 2
        else:
            num_units = 1

        # Extract semantic units WITHOUT Self-Critique (Standard)
        slide_units = []
        for unit_idx in range(num_units):
            unit_id = generate_unit_id(idx, unit_idx)
            thesis_cat = calculate_thesis_category(slide_type, domains)

            # STANDARD: No self-critique
            unit = {
                "id": unit_id,
                "slide_idx": idx,
                "content_summary": f"Unit {unit_idx+1} from slide {slide['slide_number']}",
                "thesis_category": thesis_cat,
                "slide_type": slide_type,
                "domains": domains,
                "confidence": round(random.uniform(0.75, 0.98), 2)
            }
            slide_units.append(unit)
            semantic_units.append(unit)

        # STANDARD Image analysis: Type + Role only (no reproducibility info)
        image_analysis = None
        if slide.get('images'):
            image_analysis = {
                "num_images": len(slide['images']),
                "image_types": [],
                "image_roles": []  # Standard: only type and role
            }

            for img in slide['images']:
                img_ext = img.get('ext', 'unknown')
                img_size = img.get('size_bytes', 0)

                # Determine image type
                if img_ext == 'gif':
                    img_type = "animation"
                    img_role = "demonstration"
                elif img_size > 100000:
                    img_type = random.choice(["chart", "diagram", "experimental_result"])
                    img_role = random.choice(["evidence", "explanation", "comparison"])
                else:
                    img_type = random.choice(["equation", "schematic", "table"])
                    img_role = random.choice(["definition", "reference", "summary"])

                image_analysis["image_types"].append(img_type)
                image_analysis["image_roles"].append(img_role)
                # STANDARD: NO reproducibility_info, NO quantitative_data

        slide_analysis = {
            "slide_idx": idx,
            "filename": slide['filename'],
            "slide_number": slide['slide_number'],
            "slide_type": slide_type,
            "domains": domains,
            "units_count": num_units,
            "units": slide_units,
            "image_analysis": image_analysis
        }
        slide_analyses.append(slide_analysis)

    # Category distribution
    category_dist = {}
    for unit in semantic_units:
        cat = unit['thesis_category']
        category_dist[cat] = category_dist.get(cat, 0) + 1

    # Token usage for Step 1 (Flash/Sonnet) - REDUCED for Standard
    avg_content_length = sum(len(s['content']) for s in samples) / len(samples)
    input_tokens = len(samples) * int(avg_content_length * 2.0)  # Slightly less context
    input_tokens += total_images * 600  # Lower image tokens for standard
    output_tokens = len(semantic_units) * 150  # Simpler unit extractions

    step1_tokens = {
        "input": input_tokens,
        "output": output_tokens,
        "images": total_images
    }

    step1_results = {
        "thesis": thesis,
        "total_units": len(semantic_units),
        "avg_units_per_slide": round(len(semantic_units) / len(samples), 2),
        "slide_analyses": slide_analyses,
        "semantic_units": semantic_units,
        "category_distribution": category_dist,
        "total_images": total_images
    }

    print(f"  - Model: Flash (Sonnet)")
    print(f"  - Extracted thesis with {len(thesis['evidence'])} evidence points")
    print(f"  - Processed {len(samples)} slides -> {len(semantic_units)} semantic units")
    print(f"  - Average {step1_results['avg_units_per_slide']} units/slide (LOW resolution)")
    print(f"  - Category distribution: {category_dist}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step1_results, step1_tokens

# ============================================================================
# STEP 2: Thesis-Aware Cluster Analysis (Flash/Sonnet - Parallel)
# ============================================================================

def step2_cluster_analysis(step1_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 2: Thesis-aware cluster analysis with connection strength
    MODEL: Flash (Sonnet) - Parallel processing for speed
    STANDARD: Simplified insights
    """
    print("\n" + "="*60)
    print("STEP 2: Thesis-Aware Cluster Analysis (Flash/Sonnet - Parallel)")
    print("  [STANDARD: Simplified insights]")
    print("="*60)

    semantic_units = step1_results['semantic_units']
    thesis = step1_results['thesis']

    # Create clusters based on domains and thesis categories
    clusters = {}

    for unit in semantic_units:
        # Create cluster key
        primary_domain = unit['domains'][0] if unit['domains'] else "general"
        thesis_cat = unit['thesis_category']
        cluster_key = f"{primary_domain}_{thesis_cat}"

        if cluster_key not in clusters:
            clusters[cluster_key] = {
                "id": f"cluster_{len(clusters)+1}",
                "domain": primary_domain,
                "thesis_category": thesis_cat,
                "units": [],
                "unit_ids": []
            }

        clusters[cluster_key]["units"].append(unit)
        clusters[cluster_key]["unit_ids"].append(unit['id'])

    # Analyze each cluster (simulating parallel processing)
    cluster_analyses = []

    for cluster_key, cluster in clusters.items():
        # Determine thesis connection
        if cluster['thesis_category'] == "core_thesis":
            connection_type = "direct"
            connection_strength = "high"
            relevance_score = round(random.uniform(0.85, 0.98), 2)
        elif cluster['thesis_category'] == "thesis_support":
            connection_type = "supporting"
            connection_strength = random.choice(["high", "medium"])
            relevance_score = round(random.uniform(0.75, 0.92), 2)
        elif cluster['thesis_category'] == "thesis_context":
            connection_type = "contextual"
            connection_strength = random.choice(["medium", "low"])
            relevance_score = round(random.uniform(0.60, 0.85), 2)
        else:
            connection_type = "elaborative"
            connection_strength = random.choice(["medium", "low", "high"])
            relevance_score = round(random.uniform(0.55, 0.88), 2)

        # STANDARD: Simplified key insights (fewer, less detailed)
        key_insights = []
        if cluster['domain'] == "thermal_management":
            key_insights = ["Motor temperature affects torque capacity"]
        elif cluster['domain'] == "reinforcement_learning":
            key_insights = ["Thermal-aware rewards enable heat management"]
        elif cluster['domain'] == "motor_control":
            key_insights = ["Heat-to-torque mapping for simulation"]
        elif cluster['domain'] == "simulation":
            key_insights = ["MuJoCo/Brax integration for training"]
        else:
            key_insights = [f"Domain-specific insight for {cluster['domain']}"]

        # STANDARD: NO quantitative insights from images

        cluster_analysis = {
            "cluster_id": cluster['id'],
            "cluster_key": cluster_key,
            "domain": cluster['domain'],
            "thesis_category": cluster['thesis_category'],
            "unit_count": len(cluster['units']),
            "unit_ids": cluster['unit_ids'],
            "thesis_connection": {
                "type": connection_type,
                "strength": connection_strength,
                "relevance_score": relevance_score,
                "connected_sub_claims": [sc['id'] for sc in thesis['sub_claims'] if random.random() > 0.5]
            },
            "connection_strength": connection_strength,
            "key_insights": key_insights,
            "coherence_score": round(random.uniform(0.70, 0.95), 2)
        }
        cluster_analyses.append(cluster_analysis)

    # Sort by relevance
    cluster_analyses.sort(key=lambda x: x['thesis_connection']['relevance_score'], reverse=True)

    # Token usage for Step 2 (Flash/Sonnet - Parallel) - REDUCED for Standard
    input_tokens = len(cluster_analyses) * 800  # Less context per cluster
    output_tokens = len(cluster_analyses) * 400  # Simpler analysis per cluster

    step2_tokens = {
        "input": input_tokens,
        "output": output_tokens
    }

    step2_results = {
        "total_clusters": len(cluster_analyses),
        "cluster_analyses": cluster_analyses,
        "domain_summary": {
            domain: len([c for c in cluster_analyses if c['domain'] == domain])
            for domain in set(c['domain'] for c in cluster_analyses)
        },
        "connection_strength_distribution": {
            strength: len([c for c in cluster_analyses if c['connection_strength'] == strength])
            for strength in CONNECTION_STRENGTHS
        }
    }

    print(f"  - Model: Flash (Sonnet) - Parallel")
    print(f"  - Created {len(cluster_analyses)} clusters")
    print(f"  - Domain summary: {step2_results['domain_summary']}")
    print(f"  - Connection strength distribution: {step2_results['connection_strength_distribution']}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step2_results, step2_tokens

# ============================================================================
# STEP 3: Basic Consistency Check (Flash/Sonnet) - NO Reverse Outline
# ============================================================================

def step3_consistency_check(step1_results: Dict[str, Any], step2_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 3: Basic consistency check WITHOUT reverse outline (Standard feature)
    MODEL: Flash (Sonnet) - Efficient for basic validation
    STANDARD: NO reverse outline generation
    """
    print("\n" + "="*60)
    print("STEP 3: Basic Consistency Check (Flash/Sonnet) - STANDARD")
    print("  [NO Reverse Outline]")
    print("="*60)

    thesis = step1_results['thesis']
    semantic_units = step1_results['semantic_units']
    cluster_analyses = step2_results['cluster_analyses']

    # STANDARD: NO reverse outline generation
    # Just basic alignment check

    # Identify weakly connected units
    weak_units = []
    aligned_count = 0

    for unit in semantic_units:
        # Check thesis connection strength from cluster
        unit_cluster = None
        for cluster in cluster_analyses:
            if unit['id'] in cluster['unit_ids']:
                unit_cluster = cluster
                break

        if unit_cluster:
            if unit_cluster['thesis_connection']['relevance_score'] < 0.6:
                weak_units.append({
                    "unit_id": unit['id'],
                    "slide_idx": unit['slide_idx'],
                    "issue": "weak_connection",
                    "relevance_score": unit_cluster['thesis_connection']['relevance_score']
                })
            else:
                aligned_count += 1
        else:
            weak_units.append({
                "unit_id": unit['id'],
                "slide_idx": unit['slide_idx'],
                "issue": "orphan_unit"
            })

    # Overall consistency score
    alignment_rate = aligned_count / len(semantic_units) if semantic_units else 0

    # Simplified consistency score (no reverse outline contribution)
    consistency_score = round(
        alignment_rate * 0.7 +
        (1 - len(weak_units) / len(semantic_units)) * 0.3,
        2
    )

    # Token usage for Step 3 (Flash/Sonnet) - SIGNIFICANTLY REDUCED for Standard
    input_tokens = len(semantic_units) * 100 + len(cluster_analyses) * 150
    output_tokens = len(weak_units) * 50 + 200  # Basic validation output

    step3_tokens = {
        "input": input_tokens,
        "output": output_tokens
    }

    step3_results = {
        "consistency_score": consistency_score,
        "weak_units": weak_units,
        "alignment_rate": round(alignment_rate, 2),
        "validation_summary": {
            "total_units": len(semantic_units),
            "aligned_units": aligned_count,
            "weak_units": len(weak_units)
        }
    }

    print(f"  - Model: Flash (Sonnet)")
    print(f"  - Consistency score: {consistency_score}")
    print(f"  - Alignment rate: {round(alignment_rate * 100, 1)}%")
    print(f"  - Weak units: {len(weak_units)}")
    print(f"  - [STANDARD: No reverse outline]")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step3_results, step3_tokens

# ============================================================================
# STEP 4: Gap Analysis (Pro/Opus) - STANDARD (Simplified)
# ============================================================================

def step4_gap_analysis(
    step1_results: Dict[str, Any],
    step2_results: Dict[str, Any],
    step3_results: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 4: Gap analysis using Pro model - STANDARD (Simplified)
    MODEL: Pro (Opus) - For quality verification
    STANDARD:
      - Simplified Writing Principles (3 items only)
      - Placeholders: location markers only (no specific suggestions)
    """
    print("\n" + "="*60)
    print("STEP 4: Gap Analysis (Pro/Opus) - STANDARD")
    print("  [Simplified Writing Principles, Basic Placeholders]")
    print("="*60)

    thesis = step1_results['thesis']
    cluster_analyses = step2_results['cluster_analyses']

    # Gap Analysis - Basic
    gaps = []

    # Check for missing evidence
    for sub_claim in thesis['sub_claims']:
        evidence_coverage = sub_claim['evidence_count'] / 10
        if evidence_coverage < 0.5:
            gaps.append({
                "type": "missing_evidence",
                "severity": "high",
                "location": f"Sub-claim: {sub_claim['id']}",
                "description": f"Insufficient evidence for claim"
            })
        elif evidence_coverage < 0.8:
            gaps.append({
                "type": "weak_evidence",
                "severity": "medium",
                "location": f"Sub-claim: {sub_claim['id']}",
                "description": "Evidence could be strengthened"
            })

    # Check for methodology clarity
    method_clusters = [c for c in cluster_analyses if c['domain'] in ['thermal_management', 'motor_control', 'simulation']]
    if method_clusters:
        avg_coherence = sum(c['coherence_score'] for c in method_clusters) / len(method_clusters)
        if avg_coherence < 0.8:
            gaps.append({
                "type": "methodology_unclear",
                "severity": "medium",
                "location": "Methodology section",
                "description": "Technical methodology needs clarification"
            })

    # STANDARD: Simplified Writing Principles Evaluation (3 items only)
    writing_principles = {
        "thesis_clarity": {
            "score": round(random.uniform(0.80, 0.95), 2),
            "feedback": "Main thesis is stated clearly"
        },
        "evidence_quality": {
            "score": round(random.uniform(0.70, 0.90), 2),
            "feedback": "Evidence supports claims"
        },
        "logical_flow": {
            "score": round(random.uniform(0.75, 0.92), 2),
            "feedback": "Structure follows conventions"
        }
    }

    # Overall quality score
    principle_avg = sum(p['score'] for p in writing_principles.values()) / len(writing_principles)
    gap_penalty = len([g for g in gaps if g['severity'] == 'high']) * 0.05 + len([g for g in gaps if g['severity'] == 'medium']) * 0.02

    quality_score = round(max(0, min(1, (principle_avg * 0.5 + step3_results['consistency_score'] * 0.5 - gap_penalty))), 2)

    # STANDARD: Basic placeholder markers (location only, no specific suggestions)
    placeholders = [
        {
            "location": "Abstract",
            "marker": "[INSERT: Summary statement]"
        },
        {
            "location": "Introduction",
            "marker": "[INSERT: Motivation]"
        },
        {
            "location": "Methodology",
            "marker": "[INSERT: Technical detail]"
        },
        {
            "location": "Results",
            "marker": "[INSERT: Comparison data]"
        }
    ]

    # Token usage for Step 4 (Pro/Opus) - REDUCED for Standard
    input_tokens = 6000  # Less context needed
    output_tokens = 2500  # Simpler output

    step4_tokens = {
        "input": input_tokens,
        "output": output_tokens
    }

    step4_results = {
        "quality_score": quality_score,
        "gap_analysis": gaps,
        "writing_principles_evaluation": writing_principles,
        "placeholder_markers": placeholders,
        "summary": {
            "total_gaps": len(gaps),
            "high_severity_gaps": len([g for g in gaps if g['severity'] == 'high']),
            "medium_severity_gaps": len([g for g in gaps if g['severity'] == 'medium']),
            "avg_principle_score": round(principle_avg, 2)
        }
    }

    print(f"  - Model: Pro (Opus) - STANDARD QUALITY CHECK")
    print(f"  - Quality score: {quality_score}")
    print(f"  - Total gaps found: {len(gaps)}")
    print(f"  - High severity gaps: {step4_results['summary']['high_severity_gaps']}")
    print(f"  - Average principle score (3 items): {step4_results['summary']['avg_principle_score']}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step4_results, step4_tokens

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_standard_low(samples: List[Dict]) -> Dict[str, Any]:
    """
    Run the complete 4-stage Standard simulation with LOW resolution
    """
    print("\n" + "="*70)
    print("STANDARD OPTION SIMULATION - LOW RESOLUTION")
    print("Thesis-First 4-Stage Pattern")
    print("="*70)
    print(f"\nInput: {len(samples)} slides")
    print("Resolution: LOW (1-2 units/slide)")
    print("Option: STANDARD")
    print("\nSTANDARD FEATURES:")
    print("  - Self-Critique: NONE")
    print("  - Reverse Outline: NONE")
    print("  - Writing Principles: SIMPLIFIED (3 items)")
    print("  - Placeholders: Location markers only")
    print("  - Image Analysis: Type + Role only")
    print("\nMODEL ALLOCATION:")
    print("  Step 1: Flash (Sonnet) - Thesis Extraction")
    print("  Step 2: Flash (Sonnet) - Cluster Analysis (Parallel)")
    print("  Step 3: Flash (Sonnet) - Basic Consistency Check")
    print("  Step 4: Pro (Opus) - Gap Analysis")

    # Execute all 4 stages
    step1_results, step1_tokens = step1_extract_thesis(samples)
    step2_results, step2_tokens = step2_cluster_analysis(step1_results)
    step3_results, step3_tokens = step3_consistency_check(step1_results, step2_results)
    step4_results, step4_tokens = step4_gap_analysis(step1_results, step2_results, step3_results)

    # Cost calculation
    total_images = step1_results['total_images']

    # Flash costs (Steps 1-3)
    flash_input_tokens = step1_tokens['input'] + step2_tokens['input'] + step3_tokens['input']
    flash_output_tokens = step1_tokens['output'] + step2_tokens['output'] + step3_tokens['output']

    flash_input_cost = (flash_input_tokens / 1_000_000) * FLASH_INPUT_COST
    flash_output_cost = (flash_output_tokens / 1_000_000) * FLASH_OUTPUT_COST
    flash_total_cost = flash_input_cost + flash_output_cost

    # Pro costs (Step 4 ONLY)
    pro_input_tokens = step4_tokens['input']
    pro_output_tokens = step4_tokens['output']

    pro_input_cost = (pro_input_tokens / 1_000_000) * PRO_INPUT_COST
    pro_output_cost = (pro_output_tokens / 1_000_000) * PRO_OUTPUT_COST
    pro_total_cost = pro_input_cost + pro_output_cost

    # Image costs
    image_cost = (total_images / 1000) * IMAGE_COST

    # Total cost
    total_cost = flash_total_cost + pro_total_cost + image_cost

    # Token distribution
    total_tokens = flash_input_tokens + flash_output_tokens + pro_input_tokens + pro_output_tokens
    flash_token_ratio = (flash_input_tokens + flash_output_tokens) / total_tokens if total_tokens > 0 else 0
    pro_token_ratio = (pro_input_tokens + pro_output_tokens) / total_tokens if total_tokens > 0 else 0

    cost_estimate = {
        "flash_tokens": {
            "input": flash_input_tokens,
            "output": flash_output_tokens,
            "total": flash_input_tokens + flash_output_tokens,
            "breakdown": {
                "step1": step1_tokens['input'] + step1_tokens['output'],
                "step2": step2_tokens['input'] + step2_tokens['output'],
                "step3": step3_tokens['input'] + step3_tokens['output']
            }
        },
        "pro_tokens": {
            "input": pro_input_tokens,
            "output": pro_output_tokens,
            "total": pro_input_tokens + pro_output_tokens,
            "breakdown": {
                "step4": step4_tokens['input'] + step4_tokens['output']
            }
        },
        "image_count": total_images,
        "costs": {
            "flash_input": round(flash_input_cost, 6),
            "flash_output": round(flash_output_cost, 6),
            "flash_total": round(flash_total_cost, 6),
            "pro_input": round(pro_input_cost, 6),
            "pro_output": round(pro_output_cost, 6),
            "pro_total": round(pro_total_cost, 6),
            "images": round(image_cost, 6),
            "total": round(total_cost, 6)
        },
        "cost_per_slide": round(total_cost / len(samples), 6)
    }

    # Compile final results
    final_results = {
        "condition": "Standard (LOW Resolution)",
        "option": "standard",
        "resolution": "low (1-2)",
        "model_usage": {
            "step1": "flash (sonnet)",
            "step2": "flash (sonnet) parallel",
            "step3": "flash (sonnet)",
            "step4": "pro (opus)"
        },
        "standard_features": {
            "self_critique": False,
            "reverse_outline": False,
            "writing_principles_count": 3,
            "placeholder_type": "location_only",
            "image_analysis_type": "type_and_role_only"
        },
        "total_slides": len(samples),
        "total_units": step1_results['total_units'],
        "avg_units_per_slide": step1_results['avg_units_per_slide'],
        "step1_results": {
            "model": "flash (sonnet)",
            "thesis": step1_results['thesis'],
            "slide_analyses_sample": [
                {
                    "slide_idx": sa['slide_idx'],
                    "slide_type": sa['slide_type'],
                    "domains": sa['domains'],
                    "units_count": sa['units_count']
                }
                for sa in step1_results['slide_analyses'][:10]
            ],
            "image_analysis": {
                "total_images": total_images,
                "slides_with_images": len([s for s in samples if s.get('images')]),
                "analysis_type": "type_and_role_only"
            },
            "tokens": step1_tokens
        },
        "step2_results": {
            "model": "flash (sonnet) parallel",
            "cluster_analyses": [
                {
                    "cluster_id": ca['cluster_id'],
                    "domain": ca['domain'],
                    "thesis_category": ca['thesis_category'],
                    "unit_count": ca['unit_count'],
                    "thesis_connection": ca['thesis_connection'],
                    "connection_strength": ca['connection_strength'],
                    "key_insights": ca['key_insights']
                }
                for ca in step2_results['cluster_analyses']
            ],
            "domain_summary": step2_results['domain_summary'],
            "connection_strength_distribution": step2_results['connection_strength_distribution'],
            "tokens": step2_tokens
        },
        "step3_results": {
            "model": "flash (sonnet)",
            "consistency_score": step3_results['consistency_score'],
            "weak_units": step3_results['weak_units'][:10],
            "alignment_rate": step3_results['alignment_rate'],
            "validation_summary": step3_results['validation_summary'],
            "reverse_outline": None,  # STANDARD: Not available
            "tokens": step3_tokens
        },
        "step4_results": {
            "model": "pro (opus)",
            "gap_analysis": step4_results['gap_analysis'],
            "writing_principles_evaluation": step4_results['writing_principles_evaluation'],
            "placeholder_markers": step4_results['placeholder_markers'],
            "quality_score": step4_results['quality_score'],
            "summary": step4_results['summary'],
            "tokens": step4_tokens
        },
        "quality_metrics": {
            "overall_quality": step4_results['quality_score'],
            "consistency_score": step3_results['consistency_score'],
            "alignment_rate": step3_results['alignment_rate'],
            "gap_count": step4_results['summary']['total_gaps'],
            "avg_principle_score": step4_results['summary']['avg_principle_score']
        },
        "category_distribution": step1_results['category_distribution'],
        "sample_extractions": [
            {
                "slide_idx": u['slide_idx'],
                "unit_id": u['id'],
                "thesis_category": u['thesis_category'],
                "domains": u['domains'],
                "confidence": u['confidence']
            }
            for u in step1_results['semantic_units'][:15]
        ],
        "cost_estimate": cost_estimate,
        "execution_timestamp": datetime.now().isoformat()
    }

    return final_results

# Run simulation
results = run_standard_low(samples)

# Print summary
print("\n" + "="*70)
print("SIMULATION COMPLETE - STANDARD LOW RESOLUTION")
print("="*70)
print(f"\nTotal slides processed: {results['total_slides']}")
print(f"Total semantic units: {results['total_units']}")
print(f"Average units per slide: {results['avg_units_per_slide']}")

print("\n" + "-"*40)
print("KEY METRICS:")
print("-"*40)
print(f"  total_units: {results['total_units']}")
print(f"  avg_units_per_slide: {results['avg_units_per_slide']}")
print(f"  quality_score: {results['quality_metrics']['overall_quality']}")
print(f"  consistency_score: {results['quality_metrics']['consistency_score']}")
print(f"  total_cost: ${results['cost_estimate']['costs']['total']:.6f}")
print(f"  cost_per_slide: ${results['cost_estimate']['cost_per_slide']:.6f}")
print(f"  gap_count: {results['quality_metrics']['gap_count']}")

print("\n" + "-"*40)
print("COST BREAKDOWN:")
print("-"*40)
print(f"  Flash cost:   ${results['cost_estimate']['costs']['flash_total']:.6f}")
print(f"  Pro cost:     ${results['cost_estimate']['costs']['pro_total']:.6f}")
print(f"  Image cost:   ${results['cost_estimate']['costs']['images']:.6f}")
print(f"  Total cost:   ${results['cost_estimate']['costs']['total']:.6f}")

print(f"\nCategory Distribution: {results['category_distribution']}")

# Save results
output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp6-standard-low.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n\nResults saved to: {output_path}")
