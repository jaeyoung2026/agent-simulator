#!/usr/bin/env python3
"""
Premium Medium Resolution Simulation
Thesis-First 4-Stage Pattern with Premium Features

CONFIGURATION:
- Resolution: Medium (1-3 units/slide)
- Model Distribution: Flash Steps 1-3, Pro Step 4
- Premium Features: All enabled
  - Self-Critique: Per-unit quality evaluation
  - Reverse Outline Verification: Thesis-section alignment
  - Writing Principles: Full 6-item evaluation
  - Placeholders: Specific suggestions with experiments
  - Image Analysis: Deep (reproducibility, quantitative insights)
  - Sub-claims Analysis: Enabled

ADDITIONAL COSTS (Premium-specific):
- Reverse Outline Verification: +1 Pro call (input 2000, output 1000)
- Writing Principles Detail: +1 Pro call (input 1500, output 800)
- Detailed Placeholders: +1 Flash call (input 1000, output 500)
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
    "core_thesis",           # Core claim
    "thesis_support",        # Evidence/support
    "thesis_context",        # Background/context
    "thesis_elaboration"     # Detail/expansion
]

SLIDE_TYPES = ["result", "method", "background", "contribution", "problem", "setup", "training", "analysis"]
CONNECTION_STRENGTHS = ["high", "medium", "low"]
GAP_TYPES = [
    "missing_evidence", "weak_connection", "logical_gap",
    "methodology_unclear", "result_interpretation", "context_missing"
]
SEVERITIES = ["high", "medium", "low"]

# Research-specific vocabulary
THERMAL_KEYWORDS = ["thermal", "heat", "temperature", "heat", "temp", "overheat", "cooling"]
MOTOR_KEYWORDS = ["motor", "torque", "actuator"]
RL_KEYWORDS = ["policy", "reward", "learning", "training", "RL", "DRL"]
ROBOT_KEYWORDS = ["robot", "quadruped", "locomotion", "walking", "ToddlerBot", "toddlerbot"]
SIMULATION_KEYWORDS = ["simulation", "MuJoCo", "Brax", "sim", "mjx"]

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
# STEP 1: Thesis Extraction + Self-Critique + Sub-claims (Flash/Sonnet)
# ============================================================================

def step1_extract_thesis(samples: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 1: Thesis extraction with Self-Critique and deep image analysis
    MODEL: Flash (Sonnet)
    RESOLUTION: Medium (1-3 units/slide)
    PREMIUM: Self-Critique, Sub-claims Analysis, Deep Image Analysis
    """
    print("\n" + "="*60)
    print("STEP 1: Thesis Extraction + Self-Critique (Flash/Sonnet)")
    print("Resolution: MEDIUM (1-3 units/slide)")
    print("="*60)

    # Extract global thesis with sub-claims (Premium feature)
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
                "evidence_count": 8,
                "supporting_slides": [],
                "confidence": 0.0
            },
            {
                "id": "SC2",
                "claim": "Thermal-aware policies outperform baseline in long-term operation",
                "evidence_count": 6,
                "supporting_slides": [],
                "confidence": 0.0
            },
            {
                "id": "SC3",
                "claim": "Online adaptation improves thermal estimation accuracy",
                "evidence_count": 4,
                "supporting_slides": [],
                "confidence": 0.0
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

        # Determine number of units (1-3 for MEDIUM resolution)
        content_length = len(slide['content'])
        has_images = len(slide.get('images', [])) > 0
        total_images += len(slide.get('images', []))

        # MEDIUM resolution: 1-3 units per slide
        if content_length > 200:
            num_units = 3
        elif content_length > 100 or has_images:
            num_units = 2
        else:
            num_units = 1

        # Extract semantic units with Self-Critique (Premium feature)
        slide_units = []
        for unit_idx in range(num_units):
            unit_id = generate_unit_id(idx, unit_idx)
            thesis_cat = calculate_thesis_category(slide_type, domains)

            # Self-critique score (Premium feature) - detailed per-aspect evaluation
            critique_aspects = {
                "clarity": round(random.uniform(0.7, 1.0), 2),
                "completeness": round(random.uniform(0.6, 1.0), 2),
                "relevance": round(random.uniform(0.7, 1.0), 2),
                "evidence_quality": round(random.uniform(0.5, 1.0), 2)
            }
            critique_score = round(sum(critique_aspects.values()) / len(critique_aspects), 2)

            # Determine improvement suggestions based on score
            improvements = []
            if critique_aspects['clarity'] < 0.8:
                improvements.append("Clarify the main point with more precise language")
            if critique_aspects['completeness'] < 0.8:
                improvements.append("Add more supporting details or context")
            if critique_aspects['relevance'] < 0.8:
                improvements.append("Strengthen connection to main thesis")
            if critique_aspects['evidence_quality'] < 0.8:
                improvements.append("Include quantitative data or citations")

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
                    "improvements": improvements[:2] if critique_score < 0.85 else []
                },
                "confidence": round(random.uniform(0.75, 0.98), 2)
            }
            slide_units.append(unit)
            semantic_units.append(unit)

            # Link to sub-claims (Premium feature)
            for sub_claim in thesis['sub_claims']:
                if (sub_claim['id'] == 'SC1' and 'thermal_management' in domains) or \
                   (sub_claim['id'] == 'SC2' and 'reinforcement_learning' in domains) or \
                   (sub_claim['id'] == 'SC3' and 'simulation' in domains):
                    if idx not in sub_claim['supporting_slides']:
                        sub_claim['supporting_slides'].append(idx)
                    sub_claim['confidence'] = min(1.0, sub_claim['confidence'] + 0.05)

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
                        "has_error_bars": random.random() > 0.5,
                        "statistical_significance": random.choice(["p<0.05", "p<0.01", "not reported"])
                    })

                # Reproducibility info (Premium feature - Deep)
                if img_type in ["experimental_result", "chart"]:
                    image_analysis["reproducibility_info"] = {
                        "hardware_specified": random.random() > 0.3,
                        "parameters_visible": random.random() > 0.4,
                        "sample_size_shown": random.random() > 0.5,
                        "statistical_measures": random.random() > 0.6,
                        "experiment_conditions": random.choice(["fully specified", "partially specified", "not specified"]),
                        "data_availability": random.choice(["available", "upon request", "not mentioned"])
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

    # Update sub-claim confidence scores
    for sub_claim in thesis['sub_claims']:
        sub_claim['confidence'] = round(min(1.0, sub_claim['confidence']), 2)

    # Category distribution
    category_dist = {}
    for unit in semantic_units:
        cat = unit['thesis_category']
        category_dist[cat] = category_dist.get(cat, 0) + 1

    # Token usage for Step 1 (Flash/Sonnet)
    avg_content_length = sum(len(s['content']) for s in samples) / len(samples)
    input_tokens = len(samples) * int(avg_content_length * 2.0)  # Content + prompt
    input_tokens += total_images * 600  # Image tokens for Flash
    output_tokens = len(semantic_units) * 200  # Unit extractions (medium resolution)

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
        "sub_claims_summary": {
            sc['id']: {
                "supporting_slides": len(sc['supporting_slides']),
                "confidence": sc['confidence']
            } for sc in thesis['sub_claims']
        },
        "total_images": total_images
    }

    print(f"  - Model: Flash (Sonnet)")
    print(f"  - Resolution: Medium (1-3 units/slide)")
    print(f"  - Extracted thesis with {len(thesis['sub_claims'])} sub-claims")
    print(f"  - Processed {len(samples)} slides -> {len(semantic_units)} semantic units")
    print(f"  - Average {step1_results['avg_units_per_slide']} units/slide")
    print(f"  - Self-critique avg: {step1_results['self_critique_summary']['avg_score']}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step1_results, step1_tokens


# ============================================================================
# STEP 2: Thesis-Aware Cluster Analysis (Flash/Sonnet - Parallel)
# ============================================================================

def step2_cluster_analysis(step1_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 2: Thesis-aware cluster analysis with connection strength
    MODEL: Flash (Sonnet) - Parallel processing
    """
    print("\n" + "="*60)
    print("STEP 2: Thesis-Aware Cluster Analysis (Flash/Sonnet - Parallel)")
    print("="*60)

    semantic_units = step1_results['semantic_units']
    thesis = step1_results['thesis']

    # Create clusters based on domains and thesis categories
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

    # Analyze each cluster
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

        # Generate key insights based on domain
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
            "quantitative_insights": quantitative_insights[:3],
            "coherence_score": round(random.uniform(0.70, 0.95), 2)
        }
        cluster_analyses.append(cluster_analysis)

    # Sort by relevance
    cluster_analyses.sort(key=lambda x: x['thesis_connection']['relevance_score'], reverse=True)

    # Token usage for Step 2 (Flash/Sonnet - Parallel)
    input_tokens = len(cluster_analyses) * 1000
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
    print(f"  - Connection distribution: {step2_results['connection_strength_distribution']}")
    print(f"  - Tokens: {input_tokens:,} input, {output_tokens:,} output")

    return step2_results, step2_tokens


# ============================================================================
# STEP 3: Consistency Validation + Reverse Outline (Flash/Sonnet)
# ============================================================================

def step3_consistency_validation(step1_results: Dict[str, Any], step2_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 3: Consistency validation and reverse outline generation
    MODEL: Flash (Sonnet)
    PREMIUM: Reverse Outline Verification (Thesis-Section Alignment)
    """
    print("\n" + "="*60)
    print("STEP 3: Consistency Validation + Reverse Outline (Flash/Sonnet)")
    print("Premium: Reverse Outline Verification Enabled")
    print("="*60)

    thesis = step1_results['thesis']
    semantic_units = step1_results['semantic_units']
    cluster_analyses = step2_results['cluster_analyses']

    # Generate reverse outline
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

            for cluster in cat_clusters[:3]:
                section["key_points"].extend(cluster['key_insights'][:2])

            reverse_outline["sections"].append(section)

    # Identify misaligned units
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
                "recommendation": "Unit not associated with any cluster"
            })

    alignment_rate = aligned_count / len(semantic_units) if semantic_units else 0

    # Thesis-section alignment check (Premium feature)
    section_alignments = []
    for section in reverse_outline["sections"]:
        alignment_status = "aligned" if section["thesis_alignment"] > 0.75 else "needs_attention"
        section_alignments.append({
            "section": section["name"],
            "alignment_score": section["thesis_alignment"],
            "status": alignment_status,
            "connected_sub_claims": [sc['id'] for sc in thesis['sub_claims'] if random.random() > 0.3]
        })

    consistency_score = round(
        (alignment_rate * 0.4 +
         sum(s["thesis_alignment"] for s in reverse_outline["sections"]) / len(reverse_outline["sections"]) * 0.4 +
         (1 - len(misaligned_units) / len(semantic_units)) * 0.2),
        2
    )

    # Token usage for Step 3 (Flash/Sonnet)
    input_tokens = len(semantic_units) * 150 + len(cluster_analyses) * 250
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
# STEP 4: Paper Quality Verification (Pro/Opus)
# ============================================================================

def step4_quality_verification(
    step1_results: Dict[str, Any],
    step2_results: Dict[str, Any],
    step3_results: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Step 4: Comprehensive paper quality verification using Pro model
    MODEL: Pro (Opus)
    PREMIUM: Full Writing Principles (6 items), Detailed Placeholders
    """
    print("\n" + "="*60)
    print("STEP 4: Paper Quality Verification (Pro/Opus)")
    print("Premium: Full Writing Principles + Detailed Placeholders")
    print("="*60)

    thesis = step1_results['thesis']
    cluster_analyses = step2_results['cluster_analyses']
    reverse_outline = step3_results['reverse_outline']

    # Gap Analysis with severity and priority
    gaps = []

    # Check for missing evidence based on sub-claims
    for sub_claim in thesis['sub_claims']:
        evidence_coverage = sub_claim['evidence_count'] / 10
        if evidence_coverage < 0.5:
            gaps.append({
                "type": "missing_evidence",
                "severity": "high",
                "priority": 1,
                "location": f"Sub-claim: {sub_claim['id']}",
                "description": f"Insufficient evidence for claim: {sub_claim['claim']}",
                "suggestion": "Add experimental results or literature support",
                "placeholder": f"[INSERT: Additional evidence for {sub_claim['claim'][:50]}...]",
                "experiment_suggestion": f"Consider running ablation study on {sub_claim['id']} component"
            })
        elif evidence_coverage < 0.8:
            gaps.append({
                "type": "weak_evidence",
                "severity": "medium",
                "priority": 2,
                "location": f"Sub-claim: {sub_claim['id']}",
                "description": "Evidence exists but could be strengthened",
                "suggestion": "Consider adding statistical analysis or comparison data"
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
                "placeholder": "[INSERT: Detailed algorithm pseudocode or flowchart description]"
            })

    # Writing Principles Evaluation (Premium feature - Full 6 items)
    writing_principles = {
        "thesis_clarity": {
            "score": round(random.uniform(0.80, 0.95), 2),
            "feedback": "Main thesis is clearly stated with supporting sub-claims",
            "improvement": "Consider emphasizing the novelty aspect more prominently",
            "examples": ["Highlight the gap in current thermal management approaches", "Explicitly state the novel contribution in the first paragraph"]
        },
        "evidence_integration": {
            "score": round(random.uniform(0.70, 0.90), 2),
            "feedback": "Evidence is presented but integration could be tighter",
            "improvement": "Link experimental results directly to specific claims",
            "examples": ["Add direct references from results to sub-claims", "Include quantitative comparisons with baselines"]
        },
        "logical_flow": {
            "score": round(random.uniform(0.75, 0.92), 2),
            "feedback": "Overall structure follows academic conventions",
            "improvement": "Add more explicit signposting between sections",
            "examples": ["Use transitional phrases to connect methodology to results", "Add section previews at the start of each major section"]
        },
        "technical_precision": {
            "score": round(random.uniform(0.80, 0.95), 2),
            "feedback": "Technical terms are used appropriately",
            "improvement": "Define key terms on first use",
            "examples": ["Define 'thermal-aware policy' in the introduction", "Clarify the difference between core and winding temperature"]
        },
        "reproducibility": {
            "score": round(random.uniform(0.65, 0.85), 2),
            "feedback": "Some experimental details need clarification",
            "improvement": "Include hyperparameter tables and hardware specifications",
            "examples": ["Add table with all training hyperparameters", "Specify exact hardware configuration for experiments"]
        },
        "contribution_clarity": {
            "score": round(random.uniform(0.75, 0.90), 2),
            "feedback": "Contributions are listed but could be more distinctive",
            "improvement": "Explicitly compare with prior work limitations",
            "examples": ["Create comparison table with existing methods", "Highlight unique aspects of the proposed approach"]
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
            "experiment_suggestion": "Run long-term endurance test comparing thermal-aware vs baseline policy"
        },
        {
            "location": "Introduction",
            "type": "motivation_gap",
            "template": "[INSERT: Specific failure case study motivating thermal management need]",
            "experiment_suggestion": "Document real-world motor failure incident with thermal logs"
        },
        {
            "location": "Methodology",
            "type": "technical_detail",
            "template": "[INSERT: Thermal model parameter table with values and units]",
            "experiment_suggestion": "Calibrate thermal model parameters using motor characterization tests"
        },
        {
            "location": "Experiments",
            "type": "comparison_baseline",
            "template": "[INSERT: Comparison table with baseline methods showing improvement metrics]",
            "experiment_suggestion": "Implement and test against 2-3 baseline approaches from literature"
        },
        {
            "location": "Results",
            "type": "statistical_significance",
            "template": "[INSERT: Statistical analysis (p-values, confidence intervals) for main results]",
            "experiment_suggestion": "Run multiple trials (n>=5) and compute statistical significance"
        }
    ]

    # Token usage for Step 4 (Pro/Opus)
    # Base tokens
    input_tokens = 8000
    output_tokens = 4000

    # Premium additional costs
    # Reverse outline verification: +1 Pro call
    reverse_outline_input = 2000
    reverse_outline_output = 1000

    # Writing principles detail: +1 Pro call
    writing_principles_input = 1500
    writing_principles_output = 800

    step4_tokens = {
        "input": input_tokens + reverse_outline_input + writing_principles_input,
        "output": output_tokens + reverse_outline_output + writing_principles_output
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
    print(f"  - High severity gaps: {step4_results['summary']['high_severity_gaps']}")
    print(f"  - Average principle score: {step4_results['summary']['avg_principle_score']}")
    print(f"  - Writing principles evaluated: 6 (Full)")
    print(f"  - Tokens: {step4_tokens['input']:,} input, {step4_tokens['output']:,} output")

    return step4_results, step4_tokens


# ============================================================================
# PREMIUM ADDITIONAL FEATURES (Extra API calls)
# ============================================================================

def premium_detailed_placeholders(step4_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Premium Feature: Detailed placeholder suggestions
    MODEL: Flash (Sonnet) - 1 additional call
    """
    print("\n" + "-"*40)
    print("PREMIUM: Detailed Placeholders (Flash)")
    print("-"*40)

    # Enhanced placeholders with specific suggestions
    enhanced_placeholders = []
    for ph in step4_results['placeholder_suggestions']:
        enhanced = {
            **ph,
            "specific_content_suggestions": [
                f"Option 1: {random.choice(['Add graph showing', 'Include table with', 'Present data for'])} {ph['type']}",
                f"Option 2: {random.choice(['Reference study', 'Cite work', 'Compare with'])} for {ph['location']}",
                f"Option 3: {random.choice(['Provide example', 'Show case study', 'Demonstrate'])} related to {ph['type']}"
            ],
            "priority": random.choice(["critical", "important", "optional"]),
            "estimated_effort": random.choice(["1 hour", "2-3 hours", "1 day"])
        }
        enhanced_placeholders.append(enhanced)

    tokens = {
        "input": 1000,
        "output": 500
    }

    print(f"  - Enhanced {len(enhanced_placeholders)} placeholders with specific suggestions")
    print(f"  - Tokens: {tokens['input']:,} input, {tokens['output']:,} output")

    return {"enhanced_placeholders": enhanced_placeholders}, tokens


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_premium_medium_simulation(samples: List[Dict]) -> Dict[str, Any]:
    """
    Run the complete 4-stage Premium Medium resolution simulation
    """
    print("\n" + "="*70)
    print("PREMIUM MEDIUM RESOLUTION SIMULATION")
    print("Thesis-First 4-Stage Pattern with All Premium Features")
    print("="*70)
    print(f"\nInput: {len(samples)} slides")
    print("Resolution: MEDIUM (1-3 units/slide)")
    print("Strategy: Distributed (Thesis-First 4-stage)")
    print("\nMODEL ALLOCATION:")
    print("  Step 1: Flash (Sonnet) - Thesis + Self-Critique + Sub-claims + Images")
    print("  Step 2: Flash (Sonnet) - Thesis-Aware Cluster Analysis (Parallel)")
    print("  Step 3: Flash (Sonnet) - Consistency + Reverse Outline Verification")
    print("  Step 4: Pro (Opus) - Quality + Writing Principles (6) + Gap Analysis")
    print("  Premium: Flash - Detailed Placeholders")
    print("\nPREMIUM FEATURES:")
    print("  - Self-Critique: Per-unit quality evaluation")
    print("  - Sub-claims Analysis: Tracking evidence per claim")
    print("  - Reverse Outline Verification: Thesis-section alignment")
    print("  - Writing Principles: Full 6-item evaluation")
    print("  - Placeholders: Specific suggestions with experiments")
    print("  - Image Analysis: Deep (reproducibility, quantitative)")

    # Execute all 4 stages
    step1_results, step1_tokens = step1_extract_thesis(samples)
    step2_results, step2_tokens = step2_cluster_analysis(step1_results)
    step3_results, step3_tokens = step3_consistency_validation(step1_results, step2_results)
    step4_results, step4_tokens = step4_quality_verification(step1_results, step2_results, step3_results)

    # Premium additional feature
    placeholder_results, placeholder_tokens = premium_detailed_placeholders(step4_results)

    # Cost calculation
    total_images = step1_results['total_images']

    # Flash costs (Steps 1-3 + Premium placeholder)
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
                "premium_placeholders": placeholder_tokens['input'] + placeholder_tokens['output']
            }
        },
        "pro_tokens": {
            "input": pro_input_tokens,
            "output": pro_output_tokens,
            "total": pro_input_tokens + pro_output_tokens,
            "breakdown": {
                "step4_base": 8000 + 4000,
                "reverse_outline_verification": 2000 + 1000,
                "writing_principles_detail": 1500 + 800
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
        "cost_per_slide": round(total_cost / len(samples), 6),
        "model_distribution": {
            "flash_token_ratio": round(flash_token_ratio * 100, 1),
            "pro_token_ratio": round(pro_token_ratio * 100, 1)
        }
    }

    # Compile final results
    final_results = {
        "experiment_id": "exp6-premium-medium",
        "condition": "Premium (Medium Resolution)",
        "strategy": "distributed",
        "resolution": "medium (1-3)",
        "premium_features": {
            "self_critique": True,
            "reverse_outline_verification": True,
            "writing_principles_full": True,
            "detailed_placeholders": True,
            "deep_image_analysis": True,
            "sub_claims_analysis": True
        },
        "model_usage": {
            "step1": "flash (sonnet)",
            "step2": "flash (sonnet) parallel",
            "step3": "flash (sonnet)",
            "step4": "pro (opus)",
            "premium_placeholders": "flash (sonnet)"
        },
        "total_slides": len(samples),
        "total_units": step1_results['total_units'],
        "avg_units_per_slide": step1_results['avg_units_per_slide'],
        "step1_results": {
            "model": "flash (sonnet)",
            "thesis": step1_results['thesis'],
            "self_critique_summary": step1_results['self_critique_summary'],
            "sub_claims_summary": step1_results['sub_claims_summary'],
            "image_analysis": {
                "total_images": total_images,
                "slides_with_images": len([s for s in samples if s.get('images')])
            },
            "tokens": step1_tokens
        },
        "step2_results": {
            "model": "flash (sonnet) parallel",
            "total_clusters": step2_results['total_clusters'],
            "domain_summary": step2_results['domain_summary'],
            "connection_strength_distribution": step2_results['connection_strength_distribution'],
            "tokens": step2_tokens
        },
        "step3_results": {
            "model": "flash (sonnet)",
            "consistency_score": step3_results['consistency_score'],
            "alignment_rate": step3_results['alignment_rate'],
            "reverse_outline": step3_results['reverse_outline'],
            "section_alignments": step3_results['section_alignments'],
            "validation_summary": step3_results['validation_summary'],
            "tokens": step3_tokens
        },
        "step4_results": {
            "model": "pro (opus)",
            "quality_score": step4_results['quality_score'],
            "gap_analysis": step4_results['gap_analysis'],
            "writing_principles_evaluation": step4_results['writing_principles_evaluation'],
            "section_verifications": step4_results['section_verifications'],
            "placeholder_suggestions": step4_results['placeholder_suggestions'],
            "summary": step4_results['summary'],
            "tokens": step4_tokens
        },
        "premium_results": {
            "enhanced_placeholders": placeholder_results['enhanced_placeholders'],
            "tokens": placeholder_tokens
        },
        "quality_metrics": {
            "quality_score": step4_results['quality_score'],
            "consistency_score": step3_results['consistency_score'],
            "alignment_rate": step3_results['alignment_rate'],
            "self_critique_avg": step1_results['self_critique_summary']['avg_score'],
            "writing_principles_avg": step4_results['summary']['avg_principle_score']
        },
        "gap_summary": {
            "gap_count": step4_results['summary']['total_gaps'],
            "severity_breakdown": {
                "high": step4_results['summary']['high_severity_gaps'],
                "medium": step4_results['summary']['medium_severity_gaps'],
                "low": step4_results['summary']['low_severity_gaps']
            }
        },
        "category_distribution": step1_results['category_distribution'],
        "cost_estimate": cost_estimate,
        "execution_timestamp": datetime.now().isoformat()
    }

    return final_results


# Run simulation
results = run_premium_medium_simulation(samples)

# Print summary
print("\n" + "="*70)
print("SIMULATION COMPLETE - PREMIUM MEDIUM RESOLUTION")
print("="*70)

print("\n### KEY METRICS ###")
print(f"total_units: {results['total_units']}")
print(f"avg_units_per_slide: {results['avg_units_per_slide']}")
print(f"quality_score: {results['quality_metrics']['quality_score']}")
print(f"consistency_score: {results['quality_metrics']['consistency_score']}")
print(f"alignment_rate: {results['quality_metrics']['alignment_rate']}")
print(f"self_critique_avg: {results['quality_metrics']['self_critique_avg']}")
print(f"writing_principles_avg: {results['quality_metrics']['writing_principles_avg']}")
print(f"total_cost: ${results['cost_estimate']['costs']['total']:.6f}")
print(f"cost_per_slide: ${results['cost_estimate']['cost_per_slide']:.6f}")
print(f"gap_count: {results['gap_summary']['gap_count']}")
print(f"  - high: {results['gap_summary']['severity_breakdown']['high']}")
print(f"  - medium: {results['gap_summary']['severity_breakdown']['medium']}")
print(f"  - low: {results['gap_summary']['severity_breakdown']['low']}")

print("\n### TOKEN USAGE ###")
print(f"Flash tokens: {results['cost_estimate']['flash_tokens']['total']:,}")
print(f"Pro tokens: {results['cost_estimate']['pro_tokens']['total']:,}")
print(f"Flash ratio: {results['cost_estimate']['model_distribution']['flash_token_ratio']}%")
print(f"Pro ratio: {results['cost_estimate']['model_distribution']['pro_token_ratio']}%")

print("\n### COST BREAKDOWN ###")
print(f"Flash cost: ${results['cost_estimate']['costs']['flash_total']:.6f}")
print(f"Pro cost: ${results['cost_estimate']['costs']['pro_total']:.6f}")
print(f"Image cost: ${results['cost_estimate']['costs']['images']:.6f}")
print(f"Total cost: ${results['cost_estimate']['costs']['total']:.6f}")

# Save results
output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp6-premium-medium.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n\nResults saved to: {output_path}")
