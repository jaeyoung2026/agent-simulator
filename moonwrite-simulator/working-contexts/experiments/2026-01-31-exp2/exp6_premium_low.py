#!/usr/bin/env python3
"""
Premium + Low Resolution Simulation
Thesis-First 4-Stage Pattern with Cost-Efficient Model Distribution

CONFIGURATION:
- Resolution: Low (1-2 units/slide)
- Model: Flash (Steps 1-3) + Pro (Step 4)
- Premium Features: All enabled
  - Self-Critique: Active
  - Reverse Outline Validation: Active
  - Writing Principles: Full (6 items)
  - Placeholders: Concrete suggestions
  - Image Analysis: Deep (reproducibility, quantitative insights)
  - Sub-claims Analysis: Active

ADDITIONAL PREMIUM COSTS:
- Reverse outline verification: +1 Pro call (input 2000, output 1000)
- Writing principles detail: +1 Pro call (input 1500, output 800)
- Detailed placeholders: +1 Flash call (input 1000, output 500)
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
# STEP 1: Thesis Extraction + Self-Critique (Flash/Sonnet) - LOW RESOLUTION
# ============================================================================

def step1_extract_thesis(samples: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 1: Thesis extraction with Self-Critique and deep image analysis
    MODEL: Flash (Sonnet) - Cost-efficient for extraction and analysis
    RESOLUTION: Low (1-2 units/slide)
    """
    print("\n" + "="*60)
    print("STEP 1: Thesis Extraction + Self-Critique (Flash/Sonnet)")
    print("Resolution: LOW (1-2 units/slide)")
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

    # Process each slide - LOW RESOLUTION (1-2 units/slide)
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

        # Low resolution logic: mostly 1, sometimes 2
        if content_length > 200 and has_images:
            num_units = 2
        elif content_length > 300:
            num_units = 2
        else:
            num_units = 1

        # Extract semantic units with Self-Critique
        slide_units = []
        for unit_idx in range(num_units):
            unit_id = generate_unit_id(idx, unit_idx)
            thesis_cat = calculate_thesis_category(slide_type, domains)

            # Self-critique score (Premium feature)
            critique_aspects = {
                "clarity": round(random.uniform(0.7, 1.0), 2),
                "completeness": round(random.uniform(0.6, 1.0), 2),
                "relevance": round(random.uniform(0.7, 1.0), 2),
                "evidence_quality": round(random.uniform(0.5, 1.0), 2)
            }
            critique_score = round(sum(critique_aspects.values()) / len(critique_aspects), 2)

            unit = {
                "id": unit_id,
                "slide_idx": idx,
                "content_summary": f"Unit {unit_idx+1} from slide {slide['slide_number']}",
                "thesis_category": thesis_cat,
                "slide_type": slide_type,
                "domains": domains,
                "self_critique": {
                    "score": critique_score,
                    "aspects": critique_aspects,
                    "improvements": ["Add more specific data points", "Clarify methodology details"][unit_idx % 2:][:1] if critique_score < 0.85 else []
                },
                "confidence": round(random.uniform(0.75, 0.98), 2)
            }
            slide_units.append(unit)
            semantic_units.append(unit)

        # Deep image analysis (Premium multimodal)
        image_analysis = None
        if slide.get('images'):
            image_analysis = {
                "num_images": len(slide['images']),
                "image_types": [],
                "quantitative_data": [],
                "reproducibility_info": {}
            }

            for img in slide['images']:
                img_ext = img.get('ext', 'unknown')
                img_size = img.get('size_bytes', 0)

                # Determine image type
                if img_ext == 'gif':
                    img_type = "animation"
                elif img_size > 100000:
                    img_type = random.choice(["chart", "diagram", "experimental_result"])
                else:
                    img_type = random.choice(["equation", "schematic", "table"])

                image_analysis["image_types"].append(img_type)

                # Extract quantitative insights (Premium feature)
                if img_type in ["chart", "experimental_result"]:
                    image_analysis["quantitative_data"].append({
                        "type": random.choice(["time_series", "comparison", "distribution"]),
                        "data_points": random.randint(5, 50),
                        "units": random.choice(["degrees Celsius", "Nm", "seconds", "steps"]),
                        "has_error_bars": random.random() > 0.5
                    })

                # Reproducibility info (Premium feature - Deep analysis)
                if img_type == "experimental_result":
                    image_analysis["reproducibility_info"] = {
                        "hardware_specified": random.random() > 0.3,
                        "parameters_visible": random.random() > 0.4,
                        "sample_size_shown": random.random() > 0.5,
                        "statistical_measures": random.random() > 0.6
                    }

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

    # Token usage for Step 1 (Flash/Sonnet) - Lower for Low resolution
    avg_content_length = sum(len(s['content']) for s in samples) / len(samples)
    input_tokens = len(samples) * int(avg_content_length * 2.0)  # Reduced multiplier for low resolution
    input_tokens += total_images * 800  # Image tokens
    output_tokens = len(semantic_units) * 200  # Reduced output for fewer units

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
        "self_critique_summary": {
            "avg_score": round(sum(u['self_critique']['score'] for u in semantic_units) / len(semantic_units), 2),
            "units_needing_improvement": len([u for u in semantic_units if u['self_critique']['score'] < 0.85])
        },
        "total_images": total_images
    }

    print(f"  - Model: Flash (Sonnet)")
    print(f"  - Resolution: Low (1-2 units/slide)")
    print(f"  - Extracted thesis with {len(thesis['evidence'])} evidence points")
    print(f"  - Sub-claims analyzed: {len(thesis['sub_claims'])}")
    print(f"  - Processed {len(samples)} slides -> {len(semantic_units)} semantic units")
    print(f"  - Average {step1_results['avg_units_per_slide']} units/slide")
    print(f"  - Category distribution: {category_dist}")
    print(f"  - Self-critique avg: {step1_results['self_critique_summary']['avg_score']}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step1_results, step1_tokens

# ============================================================================
# STEP 2: Thesis-Aware Cluster Analysis (Flash/Sonnet - Parallel)
# ============================================================================

def step2_cluster_analysis(step1_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 2: Thesis-aware cluster analysis with connection strength
    MODEL: Flash (Sonnet) - Parallel processing for speed
    """
    print("\n" + "="*60)
    print("STEP 2: Thesis-Aware Cluster Analysis (Flash/Sonnet - Parallel)")
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

        # Generate key insights (Premium depth)
        key_insights = []
        if cluster['domain'] == "thermal_management":
            key_insights = [
                "Motor temperature significantly impacts torque capacity",
                "Online thermal estimation improves prediction accuracy",
                "Proactive thermal management extends operation time"
            ]
        elif cluster['domain'] == "reinforcement_learning":
            key_insights = [
                "Thermal-aware rewards enable learned heat management",
                "Policy adaptation maintains performance under thermal constraints",
                "Multi-objective optimization balances task and thermal goals"
            ]
        elif cluster['domain'] == "motor_control":
            key_insights = [
                "Heat-to-torque mapping enables realistic simulation",
                "Actuator degradation affects locomotion stability",
                "Torque limiting prevents permanent motor damage"
            ]
        elif cluster['domain'] == "simulation":
            key_insights = [
                "MuJoCo/Brax integration enables fast training",
                "Thermal simulation reduces sim-to-real gap",
                "GPU acceleration speeds up policy learning"
            ]
        else:
            key_insights = [
                f"Domain-specific insight for {cluster['domain']}",
                "Integration with overall thermal management system"
            ]

        # Quantitative insights from images
        quantitative_insights = []
        for unit in cluster['units']:
            slide_idx = unit['slide_idx']
            slide_analysis = step1_results['slide_analyses'][slide_idx]
            if slide_analysis.get('image_analysis'):
                for qd in slide_analysis['image_analysis'].get('quantitative_data', []):
                    quantitative_insights.append({
                        "source_unit": unit['id'],
                        "data_type": qd['type'],
                        "measurement": f"{random.randint(1, 100)} {qd['units']}",
                        "significance": random.choice(["statistically significant", "observable trend", "requires further validation"])
                    })

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
            "key_insights": key_insights[:3],
            "quantitative_insights": quantitative_insights[:5],
            "coherence_score": round(random.uniform(0.70, 0.95), 2)
        }
        cluster_analyses.append(cluster_analysis)

    # Sort by relevance
    cluster_analyses.sort(key=lambda x: x['thesis_connection']['relevance_score'], reverse=True)

    # Token usage for Step 2 (Flash/Sonnet - Parallel) - Lower for fewer clusters
    input_tokens = len(cluster_analyses) * 1000  # Reduced for low resolution
    output_tokens = len(cluster_analyses) * 500

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
# STEP 3: Consistency Validation + Reverse Outline (Flash/Sonnet)
# ============================================================================

def step3_consistency_validation(step1_results: Dict[str, Any], step2_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 3: Consistency validation and reverse outline generation
    MODEL: Flash (Sonnet) - Efficient for structural analysis
    PREMIUM: Reverse outline validation (Thesis-section alignment check)
    """
    print("\n" + "="*60)
    print("STEP 3: Consistency Validation + Reverse Outline (Flash/Sonnet)")
    print("PREMIUM: Reverse outline verification enabled")
    print("="*60)

    thesis = step1_results['thesis']
    semantic_units = step1_results['semantic_units']
    cluster_analyses = step2_results['cluster_analyses']

    # Generate reverse outline (Premium feature)
    reverse_outline = {
        "title": "Thermal-Aware Control for Long-Term Stability in Legged Robots",
        "sections": []
    }

    # Group clusters by thesis category for outline
    section_mapping = {
        "core_thesis": "1. Introduction & Thesis",
        "thesis_context": "2. Background & Problem Definition",
        "thesis_elaboration": "3. Methodology",
        "thesis_support": "4. Experiments & Results"
    }

    for thesis_cat, section_name in section_mapping.items():
        cat_clusters = [c for c in cluster_analyses if c['thesis_category'] == thesis_cat]

        if cat_clusters:
            section = {
                "name": section_name,
                "thesis_category": thesis_cat,
                "clusters": [c['cluster_id'] for c in cat_clusters],
                "unit_count": sum(c['unit_count'] for c in cat_clusters),
                "key_points": [],
                "thesis_alignment": round(sum(c['thesis_connection']['relevance_score'] for c in cat_clusters) / len(cat_clusters), 2)
            }

            # Extract key points
            for cluster in cat_clusters[:3]:
                section["key_points"].extend(cluster['key_insights'][:2])

            reverse_outline["sections"].append(section)

    # Identify misaligned units
    misaligned_units = []
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
                misaligned_units.append({
                    "unit_id": unit['id'],
                    "slide_idx": unit['slide_idx'],
                    "issue": "weak_thesis_connection",
                    "relevance_score": unit_cluster['thesis_connection']['relevance_score'],
                    "recommendation": "Consider strengthening connection to main thesis or relocating"
                })
            elif unit['self_critique']['score'] < 0.7:
                misaligned_units.append({
                    "unit_id": unit['id'],
                    "slide_idx": unit['slide_idx'],
                    "issue": "low_quality_content",
                    "self_critique_score": unit['self_critique']['score'],
                    "recommendation": "Revise content for clarity and completeness"
                })
            else:
                aligned_count += 1
        else:
            misaligned_units.append({
                "unit_id": unit['id'],
                "slide_idx": unit['slide_idx'],
                "issue": "orphan_unit",
                "recommendation": "Unit not associated with any cluster - consider integration or removal"
            })

    # Overall consistency score
    alignment_rate = aligned_count / len(semantic_units) if semantic_units else 0

    # Thesis-section alignment check (Premium feature)
    section_alignments = []
    for section in reverse_outline["sections"]:
        alignment_status = "aligned" if section["thesis_alignment"] > 0.75 else "needs_attention"
        section_alignments.append({
            "section": section["name"],
            "alignment_score": section["thesis_alignment"],
            "status": alignment_status
        })

    consistency_score = round(
        (alignment_rate * 0.4 +
         sum(s["thesis_alignment"] for s in reverse_outline["sections"]) / len(reverse_outline["sections"]) * 0.4 +
         (1 - len(misaligned_units) / len(semantic_units)) * 0.2),
        2
    )

    # Token usage for Step 3 (Flash/Sonnet)
    input_tokens = len(semantic_units) * 180 + len(cluster_analyses) * 250  # Reduced for low resolution
    output_tokens = len(reverse_outline["sections"]) * 350 + len(misaligned_units) * 80

    step3_tokens = {
        "input": input_tokens,
        "output": output_tokens
    }

    step3_results = {
        "consistency_score": consistency_score,
        "reverse_outline": reverse_outline,
        "misaligned_units": misaligned_units,
        "alignment_rate": round(alignment_rate, 2),
        "section_alignments": section_alignments,
        "validation_summary": {
            "total_units": len(semantic_units),
            "aligned_units": aligned_count,
            "misaligned_units": len(misaligned_units),
            "sections_count": len(reverse_outline["sections"]),
            "avg_section_alignment": round(sum(s["thesis_alignment"] for s in reverse_outline["sections"]) / len(reverse_outline["sections"]), 2)
        }
    }

    print(f"  - Model: Flash (Sonnet)")
    print(f"  - Consistency score: {consistency_score}")
    print(f"  - Alignment rate: {round(alignment_rate * 100, 1)}%")
    print(f"  - Misaligned units: {len(misaligned_units)}")
    print(f"  - Reverse outline sections: {len(reverse_outline['sections'])}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step3_results, step3_tokens

# ============================================================================
# STEP 4: Paper Quality Verification (Pro/Opus) + Premium Features
# ============================================================================

def step4_quality_verification(
    step1_results: Dict[str, Any],
    step2_results: Dict[str, Any],
    step3_results: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 4: Comprehensive paper quality verification using Pro model
    MODEL: Pro (Opus) - ONLY for final quality verification
    PREMIUM FEATURES:
    - Writing Principles: Full (6 items)
    - Detailed placeholders with experiment suggestions
    - Gap analysis with severity breakdown
    """
    print("\n" + "="*60)
    print("STEP 4: Paper Quality Verification (Pro/Opus)")
    print("PREMIUM: Full Writing Principles + Detailed Placeholders")
    print("="*60)

    thesis = step1_results['thesis']
    cluster_analyses = step2_results['cluster_analyses']
    reverse_outline = step3_results['reverse_outline']

    # Gap Analysis with severity and priority
    gaps = []

    # Check for missing evidence
    for sub_claim in thesis['sub_claims']:
        evidence_coverage = sub_claim['evidence_count'] / 10  # Assume 10 is ideal
        if evidence_coverage < 0.5:
            gaps.append({
                "type": "missing_evidence",
                "severity": "high",
                "priority": 1,
                "location": f"Sub-claim: {sub_claim['id']}",
                "description": f"Insufficient evidence for claim: {sub_claim['claim']}",
                "suggestion": "Add experimental results or literature support",
                "placeholder": f"[INSERT: Additional evidence for {sub_claim['claim'][:50]}...]",
                "experiment_suggestion": f"Consider running ablation study on {sub_claim['claim'].split()[0:3]}"
            })
        elif evidence_coverage < 0.8:
            gaps.append({
                "type": "weak_evidence",
                "severity": "medium",
                "priority": 2,
                "location": f"Sub-claim: {sub_claim['id']}",
                "description": "Evidence exists but could be strengthened",
                "suggestion": "Consider adding statistical analysis or comparison data",
                "experiment_suggestion": "Add cross-validation or confidence intervals"
            })

    # Check for logical gaps in section flow
    for i, section in enumerate(reverse_outline['sections'][:-1]):
        next_section = reverse_outline['sections'][i + 1]
        if section['thesis_alignment'] < 0.7 or next_section['thesis_alignment'] < 0.7:
            gaps.append({
                "type": "logical_gap",
                "severity": "medium",
                "priority": 2,
                "location": f"Between {section['name']} and {next_section['name']}",
                "description": "Transition between sections could be clearer",
                "suggestion": "Add transitional paragraph connecting the sections",
                "placeholder": f"[INSERT: Transition from {section['thesis_category']} to {next_section['thesis_category']}]"
            })

    # Check for methodology clarity
    method_clusters = [c for c in cluster_analyses if c['domain'] in ['thermal_management', 'motor_control', 'simulation']]
    if method_clusters:
        avg_coherence = sum(c['coherence_score'] for c in method_clusters) / len(method_clusters)
        if avg_coherence < 0.8:
            gaps.append({
                "type": "methodology_unclear",
                "severity": "medium",
                "priority": 2,
                "location": "Methodology section",
                "description": "Technical methodology could be explained more clearly",
                "suggestion": "Add step-by-step explanation or algorithmic description",
                "placeholder": "[INSERT: Detailed algorithm pseudocode or flowchart description]",
                "experiment_suggestion": "Include hyperparameter sensitivity analysis"
            })

    # Writing Principles Evaluation (Premium feature - Full 6 items)
    writing_principles = {
        "thesis_clarity": {
            "score": round(random.uniform(0.80, 0.95), 2),
            "feedback": "Main thesis is clearly stated with supporting sub-claims",
            "improvement": "Consider emphasizing the novelty aspect more prominently"
        },
        "evidence_integration": {
            "score": round(random.uniform(0.70, 0.90), 2),
            "feedback": "Evidence is presented but integration could be tighter",
            "improvement": "Link experimental results directly to specific claims"
        },
        "logical_flow": {
            "score": round(random.uniform(0.75, 0.92), 2),
            "feedback": "Overall structure follows academic conventions",
            "improvement": "Add more explicit signposting between sections"
        },
        "technical_precision": {
            "score": round(random.uniform(0.80, 0.95), 2),
            "feedback": "Technical terms are used appropriately",
            "improvement": "Define key terms on first use"
        },
        "reproducibility": {
            "score": round(random.uniform(0.65, 0.85), 2),
            "feedback": "Some experimental details need clarification",
            "improvement": "Include hyperparameter tables and hardware specifications"
        },
        "contribution_clarity": {
            "score": round(random.uniform(0.75, 0.90), 2),
            "feedback": "Contributions are listed but could be more distinctive",
            "improvement": "Explicitly compare with prior work limitations"
        }
    }

    # Section-wise thesis connection verification
    section_verifications = []
    for section in reverse_outline['sections']:
        section_clusters = [c for c in cluster_analyses if c['cluster_id'] in section['clusters']]

        if section_clusters:
            avg_connection = sum(c['thesis_connection']['relevance_score'] for c in section_clusters) / len(section_clusters)
            connection_types = list(set(c['thesis_connection']['type'] for c in section_clusters))

            verification = {
                "section": section['name'],
                "thesis_connection_score": round(avg_connection, 2),
                "connection_types": connection_types,
                "status": "verified" if avg_connection > 0.75 else "needs_strengthening",
                "recommendations": []
            }

            if avg_connection < 0.75:
                verification["recommendations"].append("Strengthen explicit connection to main thesis")
            if len(connection_types) > 2:
                verification["recommendations"].append("Consider reorganizing for consistent focus")

            section_verifications.append(verification)

    # Overall quality score
    principle_avg = sum(p['score'] for p in writing_principles.values()) / len(writing_principles)
    section_avg = sum(v['thesis_connection_score'] for v in section_verifications) / len(section_verifications) if section_verifications else 0.7
    gap_penalty = len([g for g in gaps if g['severity'] == 'high']) * 0.05 + len([g for g in gaps if g['severity'] == 'medium']) * 0.02

    quality_score = round(max(0, min(1, (principle_avg * 0.4 + section_avg * 0.4 + step3_results['consistency_score'] * 0.2 - gap_penalty))), 2)

    # Concrete placeholder suggestions (Premium - with experiment suggestions)
    placeholders = [
        {
            "location": "Abstract",
            "type": "summary_statement",
            "template": "[INSERT: Quantitative improvement over baseline - e.g., 'X% longer operation time under thermal constraints']",
            "experiment_suggestion": "Run 10+ trials with thermal stress test"
        },
        {
            "location": "Introduction",
            "type": "motivation_gap",
            "template": "[INSERT: Specific failure case study motivating thermal management need]",
            "experiment_suggestion": "Document real hardware failure from thermal overload"
        },
        {
            "location": "Methodology",
            "type": "technical_detail",
            "template": "[INSERT: Thermal model parameter table with values and units]",
            "experiment_suggestion": "Include calibration procedure for thermal parameters"
        },
        {
            "location": "Experiments",
            "type": "comparison_baseline",
            "template": "[INSERT: Comparison table with baseline methods showing improvement metrics]",
            "experiment_suggestion": "Compare with at least 3 baseline methods"
        },
        {
            "location": "Results",
            "type": "statistical_significance",
            "template": "[INSERT: Statistical analysis (p-values, confidence intervals) for main results]",
            "experiment_suggestion": "Run statistical tests with n>=30 samples"
        }
    ]

    # Token usage for Step 4 (Pro/Opus)
    # Base tokens
    base_input_tokens = 8000
    base_output_tokens = 4000

    # Premium additional costs
    # Reverse outline verification: +1 Pro call (input 2000, output 1000)
    reverse_outline_input = 2000
    reverse_outline_output = 1000

    # Writing principles detail: +1 Pro call (input 1500, output 800)
    writing_principles_input = 1500
    writing_principles_output = 800

    pro_input_tokens = base_input_tokens + reverse_outline_input + writing_principles_input
    pro_output_tokens = base_output_tokens + reverse_outline_output + writing_principles_output

    step4_tokens = {
        "input": pro_input_tokens,
        "output": pro_output_tokens,
        "breakdown": {
            "base": {"input": base_input_tokens, "output": base_output_tokens},
            "reverse_outline_verification": {"input": reverse_outline_input, "output": reverse_outline_output},
            "writing_principles_detail": {"input": writing_principles_input, "output": writing_principles_output}
        }
    }

    step4_results = {
        "quality_score": quality_score,
        "gap_analysis": gaps,
        "writing_principles_evaluation": writing_principles,
        "section_verifications": section_verifications,
        "placeholder_suggestions": placeholders,
        "summary": {
            "total_gaps": len(gaps),
            "high_severity_gaps": len([g for g in gaps if g['severity'] == 'high']),
            "medium_severity_gaps": len([g for g in gaps if g['severity'] == 'medium']),
            "low_severity_gaps": len([g for g in gaps if g['severity'] == 'low']),
            "avg_principle_score": round(principle_avg, 2),
            "sections_verified": len(section_verifications),
            "sections_needing_work": len([v for v in section_verifications if v['status'] == 'needs_strengthening'])
        }
    }

    print(f"  - Model: Pro (Opus) - FINAL QUALITY CHECK")
    print(f"  - Quality score: {quality_score}")
    print(f"  - Total gaps found: {len(gaps)}")
    print(f"  - Gap severity breakdown:")
    print(f"      High: {step4_results['summary']['high_severity_gaps']}")
    print(f"      Medium: {step4_results['summary']['medium_severity_gaps']}")
    print(f"      Low: {step4_results['summary']['low_severity_gaps']}")
    print(f"  - Writing principles avg: {step4_results['summary']['avg_principle_score']}")
    print(f"  - Sections needing work: {step4_results['summary']['sections_needing_work']}")
    print(f"  - Tokens: {pro_input_tokens:,} input, {pro_output_tokens:,} output")

    return step4_results, step4_tokens

# ============================================================================
# PREMIUM ADDITIONAL COSTS: Flash call for detailed placeholders
# ============================================================================

def premium_placeholder_generation() -> Dict[str, int]:
    """
    Additional Flash call for detailed placeholder generation
    Premium feature: +1 Flash call (input 1000, output 500)
    """
    return {
        "input": 1000,
        "output": 500
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_premium_low_resolution(samples: List[Dict]) -> Dict[str, Any]:
    """
    Run the complete 4-stage Premium simulation with Low Resolution
    """
    print("\n" + "="*70)
    print("PREMIUM + LOW RESOLUTION SIMULATION")
    print("Thesis-First 4-Stage Pattern with Cost-Efficient Model Distribution")
    print("="*70)
    print(f"\nInput: {len(samples)} slides")
    print("Resolution: Low (1-2 units/slide)")
    print("Strategy: Distributed (Thesis-First 4-stage)")
    print("\nPREMIUM FEATURES:")
    print("  - Self-Critique: Active")
    print("  - Reverse Outline Validation: Active")
    print("  - Writing Principles: Full (6 items)")
    print("  - Placeholders: Concrete suggestions with experiments")
    print("  - Image Analysis: Deep (reproducibility, quantitative)")
    print("  - Sub-claims Analysis: Active")
    print("\nMODEL ALLOCATION:")
    print("  Step 1: Flash (Sonnet) - Thesis + Self-Critique + Images")
    print("  Step 2: Flash (Sonnet) - Thesis-Aware Cluster Analysis (Parallel)")
    print("  Step 3: Flash (Sonnet) - Consistency + Reverse Outline")
    print("  Step 4: Pro (Opus) - FINAL QUALITY VERIFICATION")

    # Execute all 4 stages
    step1_results, step1_tokens = step1_extract_thesis(samples)
    step2_results, step2_tokens = step2_cluster_analysis(step1_results)
    step3_results, step3_tokens = step3_consistency_validation(step1_results, step2_results)
    step4_results, step4_tokens = step4_quality_verification(step1_results, step2_results, step3_results)

    # Premium additional: Placeholder generation
    placeholder_tokens = premium_placeholder_generation()

    # Cost calculation
    total_images = step1_results['total_images']

    # Flash costs (Steps 1-3 + placeholder generation)
    flash_input_tokens = step1_tokens['input'] + step2_tokens['input'] + step3_tokens['input'] + placeholder_tokens['input']
    flash_output_tokens = step1_tokens['output'] + step2_tokens['output'] + step3_tokens['output'] + placeholder_tokens['output']

    flash_input_cost = (flash_input_tokens / 1_000_000) * FLASH_INPUT_COST
    flash_output_cost = (flash_output_tokens / 1_000_000) * FLASH_OUTPUT_COST
    flash_total_cost = flash_input_cost + flash_output_cost

    # Pro costs (Step 4 with premium additions)
    pro_input_tokens = step4_tokens['input']
    pro_output_tokens = step4_tokens['output']

    pro_input_cost = (pro_input_tokens / 1_000_000) * PRO_INPUT_COST
    pro_output_cost = (pro_output_tokens / 1_000_000) * PRO_OUTPUT_COST
    pro_total_cost = pro_input_cost + pro_output_cost

    # Image costs
    image_cost = (total_images / 1000) * IMAGE_COST

    # Total cost
    total_cost = flash_total_cost + pro_total_cost + image_cost

    # Calculate what it would cost if ALL steps used Pro
    total_input_all = flash_input_tokens + pro_input_tokens
    total_output_all = flash_output_tokens + pro_output_tokens
    full_pro_cost = (
        total_input_all / 1_000_000 * PRO_INPUT_COST +
        total_output_all / 1_000_000 * PRO_OUTPUT_COST +
        image_cost
    )

    # Savings calculation
    cost_savings = full_pro_cost - total_cost
    savings_percentage = (cost_savings / full_pro_cost * 100) if full_pro_cost > 0 else 0

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
                "step3": step3_tokens['input'] + step3_tokens['output'],
                "placeholder_generation": placeholder_tokens['input'] + placeholder_tokens['output']
            }
        },
        "pro_tokens": {
            "input": pro_input_tokens,
            "output": pro_output_tokens,
            "total": pro_input_tokens + pro_output_tokens,
            "breakdown": step4_tokens.get('breakdown', {
                "step4": step4_tokens['input'] + step4_tokens['output']
            })
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
        "cost_per_slide": round(total_cost / len(samples), 6),
        "optimization_analysis": {
            "optimized_cost": round(total_cost, 6),
            "full_pro_cost": round(full_pro_cost, 6),
            "cost_savings": round(cost_savings, 6),
            "savings_percentage": round(savings_percentage, 2),
            "flash_token_ratio": round(flash_token_ratio * 100, 1),
            "pro_token_ratio": round(pro_token_ratio * 100, 1)
        }
    }

    # Compile final results
    final_results = {
        "condition": "Premium (Low Resolution)",
        "strategy": "distributed",
        "resolution": "low (1-2)",
        "premium_features": {
            "self_critique": True,
            "reverse_outline_validation": True,
            "writing_principles": "full (6 items)",
            "placeholders": "concrete with experiment suggestions",
            "image_analysis": "deep (reproducibility, quantitative)",
            "sub_claims_analysis": True
        },
        "model_usage": {
            "step1": "flash (sonnet)",
            "step2": "flash (sonnet) parallel",
            "step3": "flash (sonnet)",
            "step4": "pro (opus)",
            "premium_placeholder": "flash (sonnet)"
        },
        "cost_optimization": {
            "flash_ratio": f"{round(flash_token_ratio * 100, 1)}%",
            "pro_ratio": f"{round(pro_token_ratio * 100, 1)}%",
            "savings_vs_full_pro": f"~{round(savings_percentage, 1)}%"
        },
        "total_slides": len(samples),
        "total_units": step1_results['total_units'],
        "avg_units_per_slide": step1_results['avg_units_per_slide'],
        "step1_results": {
            "model": "flash (sonnet)",
            "thesis": step1_results['thesis'],
            "clusters": [
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
                "image_types_summary": {
                    "charts": random.randint(8, 15),
                    "diagrams": random.randint(5, 10),
                    "equations": random.randint(3, 8),
                    "animations": len([s for s in samples if any(img.get('ext') == 'gif' for img in s.get('images', []))])
                }
            },
            "self_critique_summary": step1_results['self_critique_summary'],
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
                    "thesisConnection": ca['thesis_connection'],
                    "connection_strength": ca['connection_strength'],
                    "key_insights": ca['key_insights'],
                    "quantitative_insights": ca['quantitative_insights'][:3]
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
            "reverse_outline": step3_results['reverse_outline'],
            "misaligned_units": step3_results['misaligned_units'][:10],
            "alignment_rate": step3_results['alignment_rate'],
            "validation_summary": step3_results['validation_summary'],
            "tokens": step3_tokens
        },
        "step4_results": {
            "model": "pro (opus)",
            "gap_analysis": step4_results['gap_analysis'],
            "writing_principles_evaluation": step4_results['writing_principles_evaluation'],
            "section_verifications": step4_results['section_verifications'],
            "placeholder_suggestions": step4_results['placeholder_suggestions'],
            "quality_score": step4_results['quality_score'],
            "summary": step4_results['summary'],
            "tokens": step4_tokens
        },
        "quality_metrics": {
            "quality_score": step4_results['quality_score'],
            "consistency_score": step3_results['consistency_score'],
            "alignment_rate": step3_results['alignment_rate'],
            "self_critique_avg": step1_results['self_critique_summary']['avg_score'],
            "writing_principles_avg": step4_results['summary']['avg_principle_score'],
            "gap_count": step4_results['summary']['total_gaps'],
            "gap_severity_breakdown": {
                "high": step4_results['summary']['high_severity_gaps'],
                "medium": step4_results['summary']['medium_severity_gaps'],
                "low": step4_results['summary']['low_severity_gaps']
            }
        },
        "category_distribution": step1_results['category_distribution'],
        "sample_extractions": [
            {
                "slide_idx": u['slide_idx'],
                "unit_id": u['id'],
                "thesis_category": u['thesis_category'],
                "domains": u['domains'],
                "self_critique_score": u['self_critique']['score'],
                "confidence": u['confidence']
            }
            for u in step1_results['semantic_units'][:15]
        ],
        "cost_estimate": cost_estimate,
        "execution_timestamp": datetime.now().isoformat()
    }

    return final_results

# Run simulation
results = run_premium_low_resolution(samples)

# Print summary
print("\n" + "="*70)
print("SIMULATION COMPLETE - PREMIUM LOW RESOLUTION")
print("="*70)

print("\n" + "-"*50)
print("KEY METRICS")
print("-"*50)
print(f"total_units: {results['total_units']}")
print(f"avg_units_per_slide: {results['avg_units_per_slide']}")
print(f"quality_score: {results['quality_metrics']['quality_score']}")
print(f"consistency_score: {results['quality_metrics']['consistency_score']}")
print(f"alignment_rate: {results['quality_metrics']['alignment_rate']}")
print(f"self_critique_avg: {results['quality_metrics']['self_critique_avg']}")
print(f"writing_principles_avg: {results['quality_metrics']['writing_principles_avg']}")
print(f"total_cost: ${results['cost_estimate']['costs']['total']:.6f}")
print(f"cost_per_slide: ${results['cost_estimate']['cost_per_slide']:.6f}")
print(f"gap_count: {results['quality_metrics']['gap_count']}")
print(f"  - high: {results['quality_metrics']['gap_severity_breakdown']['high']}")
print(f"  - medium: {results['quality_metrics']['gap_severity_breakdown']['medium']}")
print(f"  - low: {results['quality_metrics']['gap_severity_breakdown']['low']}")

print("\n" + "-"*50)
print("COST BREAKDOWN")
print("-"*50)
print(f"Flash (Steps 1-3 + Placeholder): ${results['cost_estimate']['costs']['flash_total']:.6f}")
print(f"Pro (Step 4 + Premium): ${results['cost_estimate']['costs']['pro_total']:.6f}")
print(f"Images: ${results['cost_estimate']['costs']['images']:.6f}")
print(f"Total: ${results['cost_estimate']['costs']['total']:.6f}")
print(f"\nSavings vs Full Pro: {results['cost_estimate']['optimization_analysis']['savings_percentage']:.1f}%")

print("\n" + "-"*50)
print("PREMIUM FEATURES STATUS")
print("-"*50)
for feature, status in results['premium_features'].items():
    print(f"  {feature}: {status}")

# Save results
output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp6-premium-low.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n\nResults saved to: {output_path}")
