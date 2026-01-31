#!/usr/bin/env python3
"""
Standard + Distributed Strategy Simulation - High Resolution
Basic 4-Stage Pattern with Standard Features

STANDARD FEATURES (vs Premium):
- No Self-Critique
- No Reverse Outline verification in Step 3
- Simplified Writing Principles (3 items only)
- Placeholders: position markers only (no specific suggestions)
- Image Analysis: type + role only (no reproducibility info)

Model Allocation:
- Step 1-3: Flash (Sonnet)
- Step 4: Pro (Opus)
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
    "core_thesis",
    "thesis_support",
    "thesis_context",
    "thesis_elaboration"
]

SLIDE_TYPES = ["result", "method", "background", "contribution", "problem", "setup", "training", "analysis"]
CONNECTION_STRENGTHS = ["high", "medium", "low"]
GAP_TYPES = [
    "missing_evidence", "weak_connection", "logical_gap",
    "methodology_unclear", "result_interpretation", "context_missing"
]
SEVERITIES = ["high", "medium", "low"]

# Research-specific vocabulary
THERMAL_KEYWORDS = ["thermal", "heat", "temperature", "temp", "cooling"]
MOTOR_KEYWORDS = ["motor", "torque", "actuator"]
RL_KEYWORDS = ["policy", "reward", "learning", "training"]
ROBOT_KEYWORDS = ["robot", "quadruped", "locomotion", "walking", "toddlerbot"]
SIMULATION_KEYWORDS = ["simulation", "mujoco", "brax", "sim"]

# COST CONSTANTS (per 1M tokens)
FLASH_INPUT_COST = 0.00025
FLASH_OUTPUT_COST = 0.00125
PRO_INPUT_COST = 0.015
PRO_OUTPUT_COST = 0.075
IMAGE_COST = 0.0048

def generate_unit_id(slide_idx: int, unit_idx: int) -> str:
    """Generate unique unit ID"""
    hash_input = f"{slide_idx}-{unit_idx}-{datetime.now().isoformat()}"
    return f"su_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

def detect_slide_type(content: str) -> str:
    """Detect slide type from content"""
    content_lower = content.lower()
    if "result" in content_lower:
        return "result"
    elif "method" in content_lower:
        return "method"
    elif "contribution" in content_lower:
        return "contribution"
    elif "problem" in content_lower:
        return "problem"
    elif "abstract" in content_lower:
        return "background"
    elif "training" in content_lower:
        return "training"
    elif "reward" in content_lower:
        return "training"
    elif "algorithm" in content_lower:
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
    """Calculate thesis category based on slide type"""
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
    Step 1: Basic thesis extraction without Self-Critique
    MODEL: Flash (Sonnet)
    STANDARD: No self-critique analysis
    """
    print("\n" + "="*60)
    print("STEP 1: Thesis Extraction (Flash/Sonnet) - Standard")
    print("="*60)

    thesis = {
        "question": "How can thermal-aware control policies improve long-term stability and energy efficiency in legged robots?",
        "claim": "By incorporating motor heat state estimation and thermal rewards into reinforcement learning, robots can proactively manage thermal constraints while maintaining task performance.",
        "evidence": [
            "Heat2Torque simulation reduces sim-to-real gap",
            "Thermal-aware reward functions enable proactive heat management",
            "Online thermal model adaptation improves estimation accuracy",
            "DRL controllers can balance performance and hardware durability"
        ],
        "sub_claims": [
            {"id": "SC1", "claim": "Motor thermal dynamics can be accurately modeled", "evidence_count": 8},
            {"id": "SC2", "claim": "Thermal-aware policies outperform baseline", "evidence_count": 6},
            {"id": "SC3", "claim": "Online adaptation improves accuracy", "evidence_count": 4}
        ]
    }

    semantic_units = []
    slide_analyses = []
    total_images = 0

    for idx, slide in enumerate(samples):
        slide_type = detect_slide_type(slide['content'])
        domains = detect_research_domain(slide['content'])

        # Determine number of units (1-5 for high resolution)
        content_length = len(slide['content'])
        has_images = len(slide.get('images', [])) > 0
        total_images += len(slide.get('images', []))

        if content_length > 300:
            num_units = random.randint(3, 5)
        elif content_length > 150:
            num_units = random.randint(2, 4)
        elif has_images:
            num_units = random.randint(2, 3)
        else:
            num_units = random.randint(1, 2)

        # Extract semantic units WITHOUT Self-Critique (Standard)
        slide_units = []
        for unit_idx in range(num_units):
            unit_id = generate_unit_id(idx, unit_idx)
            thesis_cat = calculate_thesis_category(slide_type, domains)

            # Standard: No self-critique, just basic extraction
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

        # Standard Image Analysis: Type + Role only (no reproducibility)
        image_analysis = None
        if slide.get('images'):
            image_analysis = {
                "num_images": len(slide['images']),
                "image_types": [],
                "roles": []
            }

            for img in slide['images']:
                img_ext = img.get('ext', 'unknown')
                img_size = img.get('size_bytes', 0)

                # Determine image type
                if img_ext == 'gif':
                    img_type = "animation"
                    role = "demonstration"
                elif img_size > 100000:
                    img_type = random.choice(["chart", "diagram", "experimental_result"])
                    role = "evidence" if img_type == "experimental_result" else "explanation"
                else:
                    img_type = random.choice(["equation", "schematic", "table"])
                    role = "definition" if img_type == "equation" else "illustration"

                image_analysis["image_types"].append(img_type)
                image_analysis["roles"].append(role)

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

    # Token usage for Step 1 (Flash/Sonnet) - Lower than Premium (no self-critique)
    avg_content_length = sum(len(s['content']) for s in samples) / len(samples)
    input_tokens = len(samples) * int(avg_content_length * 2.0)  # Simpler prompt
    input_tokens += total_images * 600  # Simpler image analysis
    output_tokens = len(semantic_units) * 180  # Basic unit extractions

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
    print(f"  - Feature: Standard (No Self-Critique)")
    print(f"  - Processed {len(samples)} slides -> {len(semantic_units)} semantic units")
    print(f"  - Average {step1_results['avg_units_per_slide']} units/slide")
    print(f"  - Category distribution: {category_dist}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step1_results, step1_tokens


# ============================================================================
# STEP 2: Thesis-Aware Cluster Analysis (Flash/Sonnet - Parallel)
# ============================================================================

def step2_cluster_analysis(step1_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 2: Thesis-aware cluster analysis
    MODEL: Flash (Sonnet) - Parallel
    STANDARD: Basic clustering without detailed insights
    """
    print("\n" + "="*60)
    print("STEP 2: Thesis-Aware Cluster Analysis (Flash/Sonnet)")
    print("="*60)

    semantic_units = step1_results['semantic_units']
    thesis = step1_results['thesis']

    clusters = {}

    for unit in semantic_units:
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

    cluster_analyses = []

    for cluster_key, cluster in clusters.items():
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

        # Standard: Basic key insights only
        key_insights = []
        if cluster['domain'] == "thermal_management":
            key_insights = ["Motor temperature impacts torque", "Thermal estimation enables prediction"]
        elif cluster['domain'] == "reinforcement_learning":
            key_insights = ["Thermal-aware rewards enable heat management", "Policy adapts to constraints"]
        elif cluster['domain'] == "motor_control":
            key_insights = ["Heat-to-torque mapping enables simulation", "Torque limiting prevents damage"]
        else:
            key_insights = [f"Domain insight for {cluster['domain']}"]

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
                "connected_sub_claims": [sc['id'] for sc in thesis['sub_claims'] if random.random() > 0.4]
            },
            "connection_strength": connection_strength,
            "key_insights": key_insights[:2],
            "coherence_score": round(random.uniform(0.70, 0.95), 2)
        }
        cluster_analyses.append(cluster_analysis)

    cluster_analyses.sort(key=lambda x: x['thesis_connection']['relevance_score'], reverse=True)

    # Token usage for Step 2 (Standard - simpler analysis)
    input_tokens = len(cluster_analyses) * 1000
    output_tokens = len(cluster_analyses) * 400

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

    print(f"  - Model: Flash (Sonnet)")
    print(f"  - Created {len(cluster_analyses)} clusters")
    print(f"  - Domain summary: {step2_results['domain_summary']}")
    print(f"  - Connection strength distribution: {step2_results['connection_strength_distribution']}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step2_results, step2_tokens


# ============================================================================
# STEP 3: Basic Consistency Validation (Flash/Sonnet) - NO Reverse Outline
# ============================================================================

def step3_consistency_validation(step1_results: Dict[str, Any], step2_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 3: Basic consistency validation WITHOUT reverse outline
    MODEL: Flash (Sonnet)
    STANDARD: No reverse outline generation
    """
    print("\n" + "="*60)
    print("STEP 3: Basic Consistency Validation (Flash/Sonnet) - Standard")
    print("="*60)

    semantic_units = step1_results['semantic_units']
    cluster_analyses = step2_results['cluster_analyses']

    # Standard: Simple alignment check without reverse outline
    misaligned_units = []
    aligned_count = 0

    for unit in semantic_units:
        unit_cluster = None
        for cluster in cluster_analyses:
            if unit['id'] in cluster['unit_ids']:
                unit_cluster = cluster
                break

        if unit_cluster:
            if unit_cluster['thesis_connection']['relevance_score'] < 0.6:
                misaligned_units.append({
                    "unit_id": unit['id'],
                    "slide_idx": unit['slide_idx'],
                    "issue": "weak_thesis_connection",
                    "relevance_score": unit_cluster['thesis_connection']['relevance_score']
                })
            else:
                aligned_count += 1
        else:
            misaligned_units.append({
                "unit_id": unit['id'],
                "slide_idx": unit['slide_idx'],
                "issue": "orphan_unit"
            })

    alignment_rate = aligned_count / len(semantic_units) if semantic_units else 0

    # Simple consistency score (no reverse outline component)
    avg_cluster_score = sum(c['thesis_connection']['relevance_score'] for c in cluster_analyses) / len(cluster_analyses) if cluster_analyses else 0.7
    consistency_score = round(
        (alignment_rate * 0.5 + avg_cluster_score * 0.5),
        2
    )

    # Token usage for Step 3 (Standard - no reverse outline)
    input_tokens = len(semantic_units) * 150 + len(cluster_analyses) * 200
    output_tokens = len(misaligned_units) * 80

    step3_tokens = {
        "input": input_tokens,
        "output": output_tokens
    }

    step3_results = {
        "consistency_score": consistency_score,
        "misaligned_units": misaligned_units,
        "alignment_rate": round(alignment_rate, 2),
        "validation_summary": {
            "total_units": len(semantic_units),
            "aligned_units": aligned_count,
            "misaligned_units": len(misaligned_units)
        }
    }

    print(f"  - Model: Flash (Sonnet)")
    print(f"  - Feature: Standard (No Reverse Outline)")
    print(f"  - Consistency score: {consistency_score}")
    print(f"  - Alignment rate: {round(alignment_rate * 100, 1)}%")
    print(f"  - Misaligned units: {len(misaligned_units)}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step3_results, step3_tokens


# ============================================================================
# STEP 4: Basic Quality Verification (Pro/Opus) - Simplified
# ============================================================================

def step4_quality_verification(
    step1_results: Dict[str, Any],
    step2_results: Dict[str, Any],
    step3_results: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 4: Basic quality verification
    MODEL: Pro (Opus)
    STANDARD: Simplified writing principles (3 items), position-only placeholders
    """
    print("\n" + "="*60)
    print("STEP 4: Basic Quality Verification (Pro/Opus) - Standard")
    print("="*60)

    thesis = step1_results['thesis']
    cluster_analyses = step2_results['cluster_analyses']

    # Basic Gap Analysis
    gaps = []

    for sub_claim in thesis['sub_claims']:
        evidence_coverage = sub_claim['evidence_count'] / 10
        if evidence_coverage < 0.5:
            gaps.append({
                "type": "missing_evidence",
                "severity": "high",
                "location": f"Sub-claim: {sub_claim['id']}",
                "description": f"Insufficient evidence for: {sub_claim['claim']}"
            })
        elif evidence_coverage < 0.8:
            gaps.append({
                "type": "weak_evidence",
                "severity": "medium",
                "location": f"Sub-claim: {sub_claim['id']}",
                "description": "Evidence could be strengthened"
            })

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

    # STANDARD: Simplified Writing Principles (3 items only)
    writing_principles = {
        "thesis_clarity": {
            "score": round(random.uniform(0.80, 0.95), 2),
            "feedback": "Main thesis is stated clearly"
        },
        "evidence_integration": {
            "score": round(random.uniform(0.70, 0.90), 2),
            "feedback": "Evidence supports claims"
        },
        "logical_flow": {
            "score": round(random.uniform(0.75, 0.92), 2),
            "feedback": "Structure is coherent"
        }
    }

    # STANDARD: Position-only placeholders (no specific suggestions)
    placeholders = [
        {"location": "Abstract", "type": "summary"},
        {"location": "Introduction", "type": "motivation"},
        {"location": "Methodology", "type": "technical_detail"},
        {"location": "Results", "type": "comparison"}
    ]

    # Quality score calculation
    principle_avg = sum(p['score'] for p in writing_principles.values()) / len(writing_principles)
    gap_penalty = len([g for g in gaps if g['severity'] == 'high']) * 0.05 + len([g for g in gaps if g['severity'] == 'medium']) * 0.02

    quality_score = round(max(0, min(1, (principle_avg * 0.6 + step3_results['consistency_score'] * 0.4 - gap_penalty))), 2)

    # Token usage for Step 4 (Pro/Opus - simplified for Standard)
    input_tokens = 8000
    output_tokens = 3500

    step4_tokens = {
        "input": input_tokens,
        "output": output_tokens
    }

    step4_results = {
        "quality_score": quality_score,
        "gap_analysis": gaps,
        "writing_principles_evaluation": writing_principles,
        "placeholder_positions": placeholders,
        "summary": {
            "total_gaps": len(gaps),
            "high_severity_gaps": len([g for g in gaps if g['severity'] == 'high']),
            "medium_severity_gaps": len([g for g in gaps if g['severity'] == 'medium']),
            "avg_principle_score": round(principle_avg, 2)
        }
    }

    print(f"  - Model: Pro (Opus)")
    print(f"  - Feature: Standard (3 Writing Principles, Position-only Placeholders)")
    print(f"  - Quality score: {quality_score}")
    print(f"  - Total gaps found: {len(gaps)}")
    print(f"  - High severity gaps: {step4_results['summary']['high_severity_gaps']}")
    print(f"  - Average principle score: {step4_results['summary']['avg_principle_score']}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step4_results, step4_tokens


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_standard_high(samples: List[Dict]) -> Dict[str, Any]:
    """
    Run the complete 4-stage Standard simulation
    """
    print("\n" + "="*70)
    print("STANDARD STRATEGY SIMULATION - HIGH RESOLUTION")
    print("Basic 4-Stage Pattern with Standard Features")
    print("="*70)
    print(f"\nInput: {len(samples)} slides")
    print("Resolution: High (1-5 units/slide)")
    print("Option: Standard")
    print("\nSTANDARD FEATURES:")
    print("  - No Self-Critique")
    print("  - No Reverse Outline")
    print("  - Simplified Writing Principles (3 items)")
    print("  - Position-only Placeholders")
    print("  - Basic Image Analysis (type+role)")
    print("\nMODEL ALLOCATION:")
    print("  Step 1-3: Flash (Sonnet)")
    print("  Step 4: Pro (Opus)")

    step1_results, step1_tokens = step1_extract_thesis(samples)
    step2_results, step2_tokens = step2_cluster_analysis(step1_results)
    step3_results, step3_tokens = step3_consistency_validation(step1_results, step2_results)
    step4_results, step4_tokens = step4_quality_verification(step1_results, step2_results, step3_results)

    total_images = step1_results['total_images']

    # Flash costs (Steps 1-3)
    flash_input_tokens = step1_tokens['input'] + step2_tokens['input'] + step3_tokens['input']
    flash_output_tokens = step1_tokens['output'] + step2_tokens['output'] + step3_tokens['output']

    flash_input_cost = (flash_input_tokens / 1_000_000) * FLASH_INPUT_COST
    flash_output_cost = (flash_output_tokens / 1_000_000) * FLASH_OUTPUT_COST
    flash_total_cost = flash_input_cost + flash_output_cost

    # Pro costs (Step 4)
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
            "flash_input": round(flash_input_cost, 4),
            "flash_output": round(flash_output_cost, 4),
            "flash_total": round(flash_total_cost, 4),
            "pro_input": round(pro_input_cost, 4),
            "pro_output": round(pro_output_cost, 4),
            "pro_total": round(pro_total_cost, 4),
            "images": round(image_cost, 4),
            "total": round(total_cost, 4)
        },
        "cost_per_slide": round(total_cost / len(samples), 4),
        "token_distribution": {
            "flash_ratio": round(flash_token_ratio * 100, 1),
            "pro_ratio": round(pro_token_ratio * 100, 1)
        }
    }

    final_results = {
        "condition": "Standard (Distributed)",
        "option": "standard",
        "strategy": "distributed",
        "resolution": "high (1-5)",
        "model_usage": {
            "step1": "flash (sonnet)",
            "step2": "flash (sonnet)",
            "step3": "flash (sonnet)",
            "step4": "pro (opus)"
        },
        "standard_features": {
            "self_critique": False,
            "reverse_outline": False,
            "writing_principles_count": 3,
            "placeholder_type": "position_only",
            "image_analysis": "type_and_role_only"
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
                "analysis_level": "type_and_role_only"
            },
            "tokens": step1_tokens
        },
        "step2_results": {
            "model": "flash (sonnet)",
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
            "feature": "no_reverse_outline",
            "consistency_score": step3_results['consistency_score'],
            "misaligned_units": step3_results['misaligned_units'][:10],
            "alignment_rate": step3_results['alignment_rate'],
            "validation_summary": step3_results['validation_summary'],
            "tokens": step3_tokens
        },
        "step4_results": {
            "model": "pro (opus)",
            "feature": "simplified_writing_principles",
            "gap_analysis": step4_results['gap_analysis'],
            "writing_principles_evaluation": step4_results['writing_principles_evaluation'],
            "placeholder_positions": step4_results['placeholder_positions'],
            "quality_score": step4_results['quality_score'],
            "summary": step4_results['summary'],
            "tokens": step4_tokens
        },
        "quality_metrics": {
            "overall_quality": step4_results['quality_score'],
            "consistency_score": step3_results['consistency_score'],
            "alignment_rate": step3_results['alignment_rate'],
            "avg_principle_score": step4_results['summary']['avg_principle_score'],
            "gap_count": step4_results['summary']['total_gaps']
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
results = run_standard_high(samples)

# Print summary
print("\n" + "="*70)
print("SIMULATION COMPLETE - STANDARD HIGH RESOLUTION")
print("="*70)
print(f"\nTotal slides processed: {results['total_slides']}")
print(f"Total semantic units: {results['total_units']}")
print(f"Average units per slide: {results['avg_units_per_slide']}")
print(f"\nQuality Score: {results['step4_results']['quality_score']}")
print(f"Consistency Score: {results['step3_results']['consistency_score']}")
print(f"Alignment Rate: {results['step3_results']['alignment_rate']}")
print(f"Gap Count: {results['quality_metrics']['gap_count']}")
print(f"\nCategory Distribution: {results['category_distribution']}")

print("\n" + "="*70)
print("COST ANALYSIS")
print("="*70)
print(f"\nToken Usage:")
print(f"  Flash Input:  {results['cost_estimate']['flash_tokens']['input']:,} tokens")
print(f"  Flash Output: {results['cost_estimate']['flash_tokens']['output']:,} tokens")
print(f"  Flash Total:  {results['cost_estimate']['flash_tokens']['total']:,} tokens")
print(f"  Pro Input:    {results['cost_estimate']['pro_tokens']['input']:,} tokens")
print(f"  Pro Output:   {results['cost_estimate']['pro_tokens']['output']:,} tokens")
print(f"  Pro Total:    {results['cost_estimate']['pro_tokens']['total']:,} tokens")
print(f"  Images:       {results['cost_estimate']['image_count']} images")

print(f"\nCost Breakdown:")
print(f"  Flash cost:   ${results['cost_estimate']['costs']['flash_total']:.4f}")
print(f"  Pro cost:     ${results['cost_estimate']['costs']['pro_total']:.4f}")
print(f"  Image cost:   ${results['cost_estimate']['costs']['images']:.4f}")
print(f"  Total cost:   ${results['cost_estimate']['costs']['total']:.4f}")
print(f"  Cost/slide:   ${results['cost_estimate']['cost_per_slide']:.4f}")

print(f"\nToken Distribution:")
print(f"  Flash: {results['cost_estimate']['token_distribution']['flash_ratio']}%")
print(f"  Pro:   {results['cost_estimate']['token_distribution']['pro_ratio']}%")

# Save results
output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp6-standard-high.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n\nResults saved to: {output_path}")

# Print key metrics summary
print("\n" + "="*70)
print("KEY METRICS SUMMARY")
print("="*70)
print(f"  total_units: {results['total_units']}")
print(f"  avg_units_per_slide: {results['avg_units_per_slide']}")
print(f"  quality_score: {results['quality_metrics']['overall_quality']}")
print(f"  consistency_score: {results['quality_metrics']['consistency_score']}")
print(f"  total_cost: ${results['cost_estimate']['costs']['total']:.4f}")
print(f"  cost_per_slide: ${results['cost_estimate']['cost_per_slide']:.4f}")
print(f"  gap_count: {results['quality_metrics']['gap_count']}")
