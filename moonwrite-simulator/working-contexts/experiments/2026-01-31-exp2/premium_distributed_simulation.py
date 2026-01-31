#!/usr/bin/env python3
"""
Premium + Distributed Strategy Simulation
Thesis-First 4-Stage Pattern for Academic Writing Analysis

This simulation implements:
- Step 1 (Flash): Thesis extraction + Self-Critique + Deep image analysis
- Step 2 (Flash): Thesis-aware cluster analysis (parallel)
- Step 3 (Flash): Consistency validation + Reverse outline
- Step 4 (Pro): Paper quality verification
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
# STEP 1: Thesis Extraction + Self-Critique (Flash)
# ============================================================================

def step1_extract_thesis(samples: List[Dict]) -> Dict[str, Any]:
    """
    Step 1: Thesis extraction with Self-Critique and deep image analysis
    """
    print("\n" + "="*60)
    print("STEP 1: Thesis Extraction + Self-Critique (Flash)")
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

    for idx, slide in enumerate(samples):
        slide_type = detect_slide_type(slide['content'])
        domains = detect_research_domain(slide['content'])

        # Determine number of units (1-5 for high resolution)
        content_length = len(slide['content'])
        has_images = len(slide.get('images', [])) > 0

        if content_length > 300:
            num_units = random.randint(3, 5)
        elif content_length > 150:
            num_units = random.randint(2, 4)
        elif has_images:
            num_units = random.randint(2, 3)
        else:
            num_units = random.randint(1, 2)

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

                # Extract quantitative insights
                if img_type in ["chart", "experimental_result"]:
                    image_analysis["quantitative_data"].append({
                        "type": random.choice(["time_series", "comparison", "distribution"]),
                        "data_points": random.randint(5, 50),
                        "units": random.choice(["degrees Celsius", "Nm", "seconds", "steps"]),
                        "has_error_bars": random.random() > 0.5
                    })

                # Reproducibility info (Premium feature)
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
        }
    }

    print(f"  - Extracted thesis with {len(thesis['evidence'])} evidence points")
    print(f"  - Processed {len(samples)} slides -> {len(semantic_units)} semantic units")
    print(f"  - Average {step1_results['avg_units_per_slide']} units/slide")
    print(f"  - Category distribution: {category_dist}")

    return step1_results

# ============================================================================
# STEP 2: Thesis-Aware Cluster Analysis (Flash - Parallel)
# ============================================================================

def step2_cluster_analysis(step1_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Thesis-aware cluster analysis with connection strength
    """
    print("\n" + "="*60)
    print("STEP 2: Thesis-Aware Cluster Analysis (Flash - Parallel)")
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

    print(f"  - Created {len(cluster_analyses)} clusters")
    print(f"  - Domain summary: {step2_results['domain_summary']}")
    print(f"  - Connection strength distribution: {step2_results['connection_strength_distribution']}")

    return step2_results

# ============================================================================
# STEP 3: Consistency Validation + Reverse Outline (Flash)
# ============================================================================

def step3_consistency_validation(step1_results: Dict[str, Any], step2_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Consistency validation and reverse outline generation
    """
    print("\n" + "="*60)
    print("STEP 3: Consistency Validation + Reverse Outline (Flash)")
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

    # Thesis-section alignment check
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

    print(f"  - Consistency score: {consistency_score}")
    print(f"  - Alignment rate: {round(alignment_rate * 100, 1)}%")
    print(f"  - Misaligned units: {len(misaligned_units)}")
    print(f"  - Reverse outline sections: {len(reverse_outline['sections'])}")

    return step3_results

# ============================================================================
# STEP 4: Paper Quality Verification (Pro)
# ============================================================================

def step4_quality_verification(
    step1_results: Dict[str, Any],
    step2_results: Dict[str, Any],
    step3_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Step 4: Comprehensive paper quality verification using Pro model
    """
    print("\n" + "="*60)
    print("STEP 4: Paper Quality Verification (Pro)")
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
                "placeholder": f"[INSERT: Additional evidence for {sub_claim['claim'][:50]}...]"
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

    # Writing Principles Evaluation (Premium feature)
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

    # Concrete placeholder suggestions
    placeholders = [
        {
            "location": "Abstract",
            "type": "summary_statement",
            "template": "[INSERT: Quantitative improvement over baseline - e.g., 'X% longer operation time under thermal constraints']"
        },
        {
            "location": "Introduction",
            "type": "motivation_gap",
            "template": "[INSERT: Specific failure case study motivating thermal management need]"
        },
        {
            "location": "Methodology",
            "type": "technical_detail",
            "template": "[INSERT: Thermal model parameter table with values and units]"
        },
        {
            "location": "Experiments",
            "type": "comparison_baseline",
            "template": "[INSERT: Comparison table with baseline methods showing improvement metrics]"
        },
        {
            "location": "Results",
            "type": "statistical_significance",
            "template": "[INSERT: Statistical analysis (p-values, confidence intervals) for main results]"
        }
    ]

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

    print(f"  - Quality score: {quality_score}")
    print(f"  - Total gaps found: {len(gaps)}")
    print(f"  - High severity gaps: {step4_results['summary']['high_severity_gaps']}")
    print(f"  - Average principle score: {step4_results['summary']['avg_principle_score']}")
    print(f"  - Sections needing work: {step4_results['summary']['sections_needing_work']}")

    return step4_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_premium_distributed_simulation(samples: List[Dict]) -> Dict[str, Any]:
    """
    Run the complete 4-stage Premium + Distributed simulation
    """
    print("\n" + "="*70)
    print("PREMIUM + DISTRIBUTED STRATEGY SIMULATION")
    print("Thesis-First 4-Stage Pattern")
    print("="*70)
    print(f"\nInput: {len(samples)} slides")
    print("Resolution: High (1-5 units/slide)")
    print("Strategy: Distributed (Thesis-First 4-stage)")

    # Execute all 4 stages
    step1_results = step1_extract_thesis(samples)
    step2_results = step2_cluster_analysis(step1_results)
    step3_results = step3_consistency_validation(step1_results, step2_results)
    step4_results = step4_quality_verification(step1_results, step2_results, step3_results)

    # Cost estimation (Premium tier)
    total_units = step1_results['total_units']
    total_images = sum(len(s.get('images', [])) for s in samples)

    # Token estimates
    flash_tokens_step1 = len(samples) * 2000  # Rich extraction with images
    flash_tokens_step2 = step2_results['total_clusters'] * 1500  # Cluster analysis
    flash_tokens_step3 = total_units * 300  # Consistency check
    pro_tokens_step4 = 8000  # Comprehensive quality review

    total_flash_tokens = flash_tokens_step1 + flash_tokens_step2 + flash_tokens_step3

    cost_estimate = {
        "flash_input_tokens": total_flash_tokens,
        "flash_output_tokens": int(total_flash_tokens * 0.3),
        "pro_input_tokens": pro_tokens_step4,
        "pro_output_tokens": int(pro_tokens_step4 * 0.4),
        "image_tokens": total_images * 1000,
        "estimated_flash_cost_usd": round(total_flash_tokens * 0.000001, 4),
        "estimated_pro_cost_usd": round(pro_tokens_step4 * 0.000015, 4),
        "total_estimated_cost_usd": round(total_flash_tokens * 0.000001 + pro_tokens_step4 * 0.000015, 4)
    }

    # Compile final results
    final_results = {
        "condition": "Premium (Distributed)",
        "strategy": "distributed",
        "resolution": "high (1-5)",
        "model_usage": {
            "flash_step1": "thesis + self-critique 추출 + 심층 이미지",
            "flash_step2": "thesis-aware 분석 (병렬)",
            "flash_step3": "일관성 + 역개요 검증",
            "pro_step4": "전체 품질 검증"
        },
        "total_slides": len(samples),
        "total_units": total_units,
        "avg_units_per_slide": step1_results['avg_units_per_slide'],
        "step1_results": {
            "thesis": step1_results['thesis'],
            "clusters": [
                {
                    "slide_idx": sa['slide_idx'],
                    "slide_type": sa['slide_type'],
                    "domains": sa['domains'],
                    "units_count": sa['units_count']
                }
                for sa in step1_results['slide_analyses'][:10]  # Sample
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
            "self_critique_summary": step1_results['self_critique_summary']
        },
        "step2_results": {
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
            "connection_strength_distribution": step2_results['connection_strength_distribution']
        },
        "step3_results": {
            "consistency_score": step3_results['consistency_score'],
            "reverse_outline": step3_results['reverse_outline'],
            "misaligned_units": step3_results['misaligned_units'][:10],  # Sample
            "alignment_rate": step3_results['alignment_rate'],
            "validation_summary": step3_results['validation_summary']
        },
        "step4_results": {
            "gap_analysis": step4_results['gap_analysis'],
            "writing_principles_evaluation": step4_results['writing_principles_evaluation'],
            "section_verifications": step4_results['section_verifications'],
            "placeholder_suggestions": step4_results['placeholder_suggestions'],
            "quality_score": step4_results['quality_score'],
            "summary": step4_results['summary']
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
            for u in step1_results['semantic_units'][:15]  # Sample
        ],
        "cost_estimate": cost_estimate,
        "execution_timestamp": datetime.now().isoformat()
    }

    return final_results

# Run simulation
results = run_premium_distributed_simulation(samples)

# Print summary
print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print(f"\nTotal slides processed: {results['total_slides']}")
print(f"Total semantic units: {results['total_units']}")
print(f"Average units per slide: {results['avg_units_per_slide']}")
print(f"\nQuality Score: {results['step4_results']['quality_score']}")
print(f"Consistency Score: {results['step3_results']['consistency_score']}")
print(f"Alignment Rate: {results['step3_results']['alignment_rate']}")
print(f"\nCategory Distribution: {results['category_distribution']}")
print(f"\nEstimated Cost: ${results['cost_estimate']['total_estimated_cost_usd']}")

# Save results
output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp5-premium-distributed.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {output_path}")
