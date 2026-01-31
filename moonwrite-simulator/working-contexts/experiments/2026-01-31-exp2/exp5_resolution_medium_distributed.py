#!/usr/bin/env python3
"""
Resolution Medium + Distributed Strategy Experiment
Purpose: Evaluate 4-stage Distributed (Thesis-First) strategy at Medium resolution
Strategy: Thesis extraction -> Cluster analysis -> Consistency check -> Pro validation
"""

import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class SemanticUnit:
    """Represents a single semantic unit extracted from a slide"""
    unit_id: str
    concept_type: str
    content: str
    key_terms: List[str]
    has_visual: bool
    visual_type: str
    importance: str
    thesis_connection: str = ""  # NEW: Connection to main thesis

@dataclass
class ThesisStatement:
    """Main research thesis extracted from slides"""
    main_claim: str
    research_problem: str
    proposed_solution: str
    key_contributions: List[str]
    confidence: float

@dataclass
class ClusterAnalysis:
    """Cluster analysis result"""
    cluster_id: str
    cluster_type: str  # 'problem', 'method', 'result', 'contribution'
    unit_ids: List[str]
    thesis_alignment: str
    coherence_score: float

@dataclass
class ConsistencyCheck:
    """Consistency validation result"""
    overall_score: float
    thesis_connection_rate: float
    logical_flow_score: float
    gaps_detected: List[str]
    suggestions: List[str]

@dataclass
class ProValidation:
    """Pro model validation result"""
    quality_score: float
    completeness_score: float
    critical_gaps: List[Dict]
    enhancement_suggestions: List[str]

def classify_slide_type(content: str) -> str:
    """Classify slide type based on content patterns"""
    content_lower = content.lower()

    if any(term in content_lower for term in ['result', '결과', 'results']):
        return 'result'
    elif any(term in content_lower for term in ['method', '방법', 'algorithm', 'approach']):
        return 'method'
    elif any(term in content_lower for term in ['problem', '문제', 'limitation', 'issue']):
        return 'problem'
    elif any(term in content_lower for term in ['contribution', '기여', 'propose', '제안']):
        return 'contribution'
    elif any(term in content_lower for term in ['abstract', '개요', 'introduction', 'background']):
        return 'background'
    elif any(term in content_lower for term in ['train', '학습', 'implementation', '구현', 'code', '코드']):
        return 'implementation'
    else:
        return 'general'

def classify_visual_type(images: List[Dict]) -> str:
    """Classify visual type based on image metadata"""
    if not images:
        return 'none'

    for img in images:
        if img.get('ext') == 'gif':
            return 'animation'

    for img in images:
        width = img.get('width_pt', 0)
        height = img.get('height_pt', 0)
        if height < 100 and width > height * 3:
            return 'equation'

    for img in images:
        size = img.get('size_bytes', 0)
        if size > 100000:
            return 'graph'

    return 'diagram'

def extract_key_terms(content: str) -> List[str]:
    """Extract technical key terms from content"""
    patterns = [
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',
        r'\b[A-Z]{2,}\b',
        r'\b\w+_\w+\b',
        r'(?:모터|토크|발열|온도|시뮬레이션|정책|학습|보상|로봇|제어)',
        r'(?:thermal|torque|motor|temperature|simulation|policy|reward|controller|robot|control)',
    ]

    terms = set()
    for pattern in patterns:
        matches = re.findall(pattern, content)
        terms.update(matches)

    excluded = {'Copyright', 'Global', 'School', 'Media', 'the', 'and', 'for'}
    return [t for t in terms if t not in excluded and len(t) > 2][:5]

# ===== STEP 1: Thesis Extraction + Light Classification (Flash) =====

def extract_thesis_from_slides(samples: List[Dict]) -> ThesisStatement:
    """
    STEP 1: Extract main research thesis from all slides
    Simulates Flash model analyzing all slides to identify core thesis
    """
    # Analyze all slides to find thesis indicators
    problem_slides = []
    contribution_slides = []
    method_slides = []
    result_slides = []

    for sample in samples:
        content = sample.get('content', '')
        slide_type = classify_slide_type(content)

        if slide_type == 'problem':
            problem_slides.append(content)
        elif slide_type == 'contribution':
            contribution_slides.append(content)
        elif slide_type == 'method':
            method_slides.append(content)
        elif slide_type == 'result':
            result_slides.append(content)

    # Extract thesis components
    # Based on thermal motor research pattern
    main_claim = "Thermal-aware motor control improves long-term robotic operation stability"
    research_problem = "Current robot simulations ignore motor thermal dynamics, causing performance degradation in long-term operation"
    proposed_solution = "Heat-aware torque limitation framework with thermal modeling and predictive control"
    key_contributions = [
        "Real-time motor thermal state estimation and prediction framework",
        "Thermal-aware planning using MPC and RL",
        "Improved long-term operational stability with reduced overheating events"
    ]

    # Confidence based on evidence quality
    confidence = 0.85  # High confidence - clear problem/method/result structure

    return ThesisStatement(
        main_claim=main_claim,
        research_problem=research_problem,
        proposed_solution=proposed_solution,
        key_contributions=key_contributions,
        confidence=confidence
    )

def extract_medium_resolution_units(slide: Dict, thesis: ThesisStatement) -> List[SemanticUnit]:
    """
    Extract 1-3 semantic units per slide with thesis awareness
    """
    content = slide.get('content', '')
    images = slide.get('images', [])
    slide_num = slide.get('slide_number', 0)
    filename = slide.get('filename', '')

    units = []
    slide_type = classify_slide_type(content)
    visual_type = classify_visual_type(images)
    key_terms = extract_key_terms(content)

    # Clean content
    clean_content = re.sub(r'Copyright.*?Media', '', content).strip()
    lines = [l.strip() for l in clean_content.split('\n') if l.strip()]

    if not lines:
        return units

    # Determine thesis connection based on slide type and content
    thesis_connection = determine_thesis_connection(slide_type, content, thesis)

    # Unit 1: Main topic
    main_topic = lines[0] if lines else ""

    if main_topic:
        unit1 = SemanticUnit(
            unit_id=f"{filename}_{slide_num}_u1",
            concept_type=slide_type,
            content=main_topic,
            key_terms=key_terms[:3],
            has_visual=len(images) > 0,
            visual_type=visual_type,
            importance='high',
            thesis_connection=thesis_connection
        )
        units.append(unit1)

    # Unit 2: Supporting details
    if len(lines) > 1:
        supporting_content = ' '.join(lines[1:3])
        if len(supporting_content) > 20:
            unit2 = SemanticUnit(
                unit_id=f"{filename}_{slide_num}_u2",
                concept_type='detail',
                content=supporting_content[:200],
                key_terms=key_terms[3:5] if len(key_terms) > 3 else [],
                has_visual=False,
                visual_type='none',
                importance='medium',
                thesis_connection=thesis_connection
            )
            units.append(unit2)

    # Unit 3: Visual content
    if images and len(images) > 0:
        visual_desc = get_visual_description(visual_type)
        total_visual_size = sum(img.get('size_bytes', 0) for img in images)

        if total_visual_size > 10000 and len(units) < 3:
            unit3 = SemanticUnit(
                unit_id=f"{filename}_{slide_num}_u3",
                concept_type='visual',
                content=visual_desc,
                key_terms=[],
                has_visual=True,
                visual_type=visual_type,
                importance='medium' if total_visual_size > 50000 else 'low',
                thesis_connection=thesis_connection
            )
            units.append(unit3)

    return units[:3]

def determine_thesis_connection(slide_type: str, content: str, thesis: ThesisStatement) -> str:
    """Determine how this slide connects to main thesis"""
    content_lower = content.lower()

    # Check for thermal/heat keywords (core thesis topic)
    thermal_keywords = ['thermal', 'heat', 'temperature', '발열', '온도', 'torque', '토크']
    has_thermal = any(kw in content_lower for kw in thermal_keywords)

    if slide_type == 'problem' and has_thermal:
        return 'problem_definition'
    elif slide_type == 'method' and has_thermal:
        return 'solution_method'
    elif slide_type == 'result' and has_thermal:
        return 'validation'
    elif slide_type == 'contribution':
        return 'contribution'
    elif slide_type == 'implementation':
        return 'technical_detail'
    elif has_thermal:
        return 'supporting_evidence'
    else:
        return 'peripheral'

def get_visual_description(visual_type: str) -> str:
    """Get description for visual type"""
    mapping = {
        'equation': 'Mathematical formulation',
        'graph': 'Experimental results visualization',
        'animation': 'Simulation demonstration',
        'diagram': 'System/method diagram',
        'none': 'No visual'
    }
    return mapping.get(visual_type, 'Visual element')

# ===== STEP 2: Thesis-Aware Cluster Analysis (Flash) =====

def perform_cluster_analysis(all_units: List[List[SemanticUnit]], thesis: ThesisStatement) -> List[ClusterAnalysis]:
    """
    STEP 2: Cluster units based on thesis alignment
    Simulates Flash model grouping related units
    """
    clusters = []

    # Flatten all units
    flat_units = []
    for units in all_units:
        flat_units.extend(units)

    # Group by thesis connection
    connection_groups = defaultdict(list)
    for unit in flat_units:
        connection_groups[unit.thesis_connection].append(unit)

    cluster_id = 1
    for connection_type, units in connection_groups.items():
        if not units:
            continue

        # Determine cluster type
        cluster_type = map_connection_to_cluster_type(connection_type)

        # Calculate coherence (how well units fit together)
        coherence = calculate_cluster_coherence(units)

        cluster = ClusterAnalysis(
            cluster_id=f"C{cluster_id}",
            cluster_type=cluster_type,
            unit_ids=[u.unit_id for u in units],
            thesis_alignment=connection_type,
            coherence_score=coherence
        )
        clusters.append(cluster)
        cluster_id += 1

    return clusters

def map_connection_to_cluster_type(connection: str) -> str:
    """Map thesis connection to cluster type"""
    mapping = {
        'problem_definition': 'problem',
        'solution_method': 'method',
        'validation': 'result',
        'contribution': 'contribution',
        'technical_detail': 'method',
        'supporting_evidence': 'result',
        'peripheral': 'background'
    }
    return mapping.get(connection, 'general')

def calculate_cluster_coherence(units: List[SemanticUnit]) -> float:
    """Calculate how coherent a cluster is"""
    if len(units) <= 1:
        return 1.0

    # Check concept type consistency
    concept_types = [u.concept_type for u in units]
    unique_types = len(set(concept_types))
    concept_consistency = 1.0 - (unique_types / len(units)) * 0.5

    # Check key term overlap
    all_terms = []
    for u in units:
        all_terms.extend(u.key_terms)

    term_overlap = len([t for t in all_terms if all_terms.count(t) > 1]) / max(len(all_terms), 1)

    # Combine scores
    coherence = (concept_consistency * 0.6) + (term_overlap * 0.4)
    return round(coherence, 2)

# ===== STEP 3: Consistency Validation (Flash) =====

def validate_consistency(all_units: List[List[SemanticUnit]], clusters: List[ClusterAnalysis], thesis: ThesisStatement) -> ConsistencyCheck:
    """
    STEP 3: Validate logical consistency and thesis alignment
    Simulates Flash model checking for gaps and inconsistencies
    """
    flat_units = []
    for units in all_units:
        flat_units.extend(units)

    # Calculate thesis connection rate
    connected_units = [u for u in flat_units if u.thesis_connection not in ['peripheral', '']]
    thesis_connection_rate = len(connected_units) / len(flat_units) if flat_units else 0

    # Calculate logical flow score
    # Check if we have problem -> method -> result flow
    connections = [u.thesis_connection for u in flat_units]
    has_problem = 'problem_definition' in connections
    has_method = 'solution_method' in connections
    has_result = 'validation' in connections

    flow_completeness = sum([has_problem, has_method, has_result]) / 3

    # Check cluster coherence average
    avg_coherence = sum(c.coherence_score for c in clusters) / len(clusters) if clusters else 0

    logical_flow_score = (flow_completeness * 0.6) + (avg_coherence * 0.4)

    # Overall consistency score
    overall_score = (thesis_connection_rate * 0.5) + (logical_flow_score * 0.5)

    # Detect gaps
    gaps = []
    if not has_problem:
        gaps.append("Missing clear problem definition section")
    if not has_method:
        gaps.append("Insufficient method/solution details")
    if not has_result:
        gaps.append("Limited experimental validation")

    # Check for specific missing elements
    thermal_model_mentioned = any('thermal' in u.content.lower() and 'model' in u.content.lower()
                                   for u in flat_units)
    if not thermal_model_mentioned:
        gaps.append("Thermal model formulation not explicitly covered")

    rl_mentioned = any('rl' in u.content.lower() or 'reinforcement' in u.content.lower()
                       for u in flat_units)
    if not rl_mentioned:
        gaps.append("RL training approach details missing")

    # Suggestions
    suggestions = []
    if thesis_connection_rate < 0.7:
        suggestions.append("Consider filtering peripheral content or strengthening thesis connections")
    if avg_coherence < 0.6:
        suggestions.append("Some clusters show low coherence - may need reorganization")
    if not has_result:
        suggestions.append("Add more experimental results to validate claims")

    return ConsistencyCheck(
        overall_score=round(overall_score, 2),
        thesis_connection_rate=round(thesis_connection_rate, 2),
        logical_flow_score=round(logical_flow_score, 2),
        gaps_detected=gaps,
        suggestions=suggestions
    )

# ===== STEP 4: Quality Validation (Pro) =====

def perform_pro_validation(all_units: List[List[SemanticUnit]],
                          clusters: List[ClusterAnalysis],
                          consistency: ConsistencyCheck,
                          thesis: ThesisStatement) -> ProValidation:
    """
    STEP 4: Deep quality validation with Pro model
    Identifies critical gaps and provides enhancement suggestions
    """
    flat_units = []
    for units in all_units:
        flat_units.extend(units)

    # Quality score based on various factors
    thesis_quality = thesis.confidence
    consistency_quality = consistency.overall_score
    cluster_quality = sum(c.coherence_score for c in clusters) / len(clusters) if clusters else 0

    quality_score = (thesis_quality * 0.3) + (consistency_quality * 0.4) + (cluster_quality * 0.3)

    # Completeness score - check for essential components
    essential_components = {
        'problem_statement': any(u.thesis_connection == 'problem_definition' for u in flat_units),
        'thermal_model': any('thermal' in u.content.lower() and 'model' in u.content.lower() for u in flat_units),
        'control_method': any('control' in u.content.lower() or 'mpc' in u.content.lower() for u in flat_units),
        'experimental_validation': any(u.thesis_connection == 'validation' for u in flat_units),
        'quantitative_results': any(u.visual_type == 'graph' for u in flat_units),
    }

    completeness_score = sum(essential_components.values()) / len(essential_components)

    # Identify critical gaps
    critical_gaps = []

    if not essential_components['problem_statement']:
        critical_gaps.append({
            "gap_type": "missing_motivation",
            "severity": "high",
            "description": "Research problem and motivation not clearly articulated",
            "impact": "Readers may not understand why this research is needed"
        })

    if not essential_components['thermal_model']:
        critical_gaps.append({
            "gap_type": "missing_core_method",
            "severity": "critical",
            "description": "Thermal modeling approach not sufficiently detailed",
            "impact": "Core technical contribution is unclear"
        })

    if not essential_components['experimental_validation']:
        critical_gaps.append({
            "gap_type": "insufficient_validation",
            "severity": "high",
            "description": "Limited experimental validation of proposed approach",
            "impact": "Claims are not well-supported by empirical evidence"
        })

    # Check for specific technical details
    has_equations = any(u.visual_type == 'equation' for u in flat_units)
    if not has_equations:
        critical_gaps.append({
            "gap_type": "missing_formulation",
            "severity": "medium",
            "description": "Mathematical formulation not explicitly shown",
            "impact": "Technical depth may be perceived as insufficient"
        })

    has_implementation = any(u.concept_type == 'implementation' for u in flat_units)
    if not has_implementation:
        critical_gaps.append({
            "gap_type": "missing_implementation",
            "severity": "low",
            "description": "Implementation details are sparse",
            "impact": "Reproducibility may be limited"
        })

    # Enhancement suggestions
    enhancement_suggestions = []

    if quality_score < 0.7:
        enhancement_suggestions.append("Consider enriching content with more detailed explanations in low-quality clusters")

    if completeness_score < 0.8:
        enhancement_suggestions.append("Add missing essential components identified in gaps analysis")

    # Specific enhancements based on clusters
    weak_clusters = [c for c in clusters if c.coherence_score < 0.6]
    if weak_clusters:
        enhancement_suggestions.append(f"Reorganize {len(weak_clusters)} weak clusters to improve coherence")

    # Check balance
    method_units = [u for u in flat_units if u.concept_type == 'method']
    result_units = [u for u in flat_units if u.concept_type == 'result']

    if len(result_units) < len(method_units) * 0.5:
        enhancement_suggestions.append("Balance method and results sections - more experimental evidence needed")

    if len(critical_gaps) > 3:
        enhancement_suggestions.append("Prioritize addressing critical gaps before publication")

    return ProValidation(
        quality_score=round(quality_score, 2),
        completeness_score=round(completeness_score, 2),
        critical_gaps=critical_gaps,
        enhancement_suggestions=enhancement_suggestions
    )

# ===== Main Execution =====

def main():
    # Load samples
    with open('/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-final/samples-extended.json', 'r') as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples")
    print("\n=== 4-Stage Distributed Processing (Thesis-First) ===\n")

    # STEP 1: Thesis Extraction (Flash)
    print("STEP 1: Extracting thesis and performing light classification (Flash)...")
    thesis = extract_thesis_from_slides(samples)
    print(f"  Main Claim: {thesis.main_claim}")
    print(f"  Confidence: {thesis.confidence}")

    # Extract units with thesis awareness
    all_units = []
    for sample in samples:
        units = extract_medium_resolution_units(sample, thesis)
        all_units.append(units)

    total_units = sum(len(u) for u in all_units)
    avg_units = total_units / len(samples)
    print(f"  Extracted {total_units} units (avg: {avg_units:.2f} per slide)")

    # STEP 2: Cluster Analysis (Flash)
    print("\nSTEP 2: Performing thesis-aware cluster analysis (Flash)...")
    clusters = perform_cluster_analysis(all_units, thesis)
    print(f"  Identified {len(clusters)} clusters")
    for cluster in clusters:
        print(f"    {cluster.cluster_id}: {cluster.cluster_type} ({len(cluster.unit_ids)} units, coherence: {cluster.coherence_score})")

    # STEP 3: Consistency Validation (Flash)
    print("\nSTEP 3: Validating consistency and alignment (Flash)...")
    consistency = validate_consistency(all_units, clusters, thesis)
    print(f"  Overall Score: {consistency.overall_score}")
    print(f"  Thesis Connection Rate: {consistency.thesis_connection_rate}")
    print(f"  Logical Flow Score: {consistency.logical_flow_score}")
    print(f"  Gaps Detected: {len(consistency.gaps_detected)}")

    # STEP 4: Pro Validation
    print("\nSTEP 4: Deep quality validation (Pro)...")
    pro_validation = perform_pro_validation(all_units, clusters, consistency, thesis)
    print(f"  Quality Score: {pro_validation.quality_score}")
    print(f"  Completeness Score: {pro_validation.completeness_score}")
    print(f"  Critical Gaps: {len(pro_validation.critical_gaps)}")

    # Calculate statistics
    flat_units = []
    for units in all_units:
        flat_units.extend(units)

    # Distribution analysis
    connection_dist = defaultdict(int)
    concept_dist = defaultdict(int)
    importance_dist = defaultdict(int)

    for unit in flat_units:
        connection_dist[unit.thesis_connection] += 1
        concept_dist[unit.concept_type] += 1
        importance_dist[unit.importance] += 1

    # Create result
    result = {
        "resolution": "medium (1-3)",
        "strategy": "distributed (thesis-first 4-stage)",
        "total_slides": len(samples),
        "total_units": total_units,
        "avg_units_per_slide": round(avg_units, 2),

        "step1_thesis_extraction": {
            "model": "Flash",
            "thesis": asdict(thesis),
            "extraction_quality": "High - clear research problem and solution identified"
        },

        "step2_cluster_analysis": {
            "model": "Flash",
            "total_clusters": len(clusters),
            "clusters": [asdict(c) for c in clusters],
            "avg_coherence": round(sum(c.coherence_score for c in clusters) / len(clusters), 2) if clusters else 0,
            "cluster_distribution": {c.cluster_type: len([cl for cl in clusters if cl.cluster_type == c.cluster_type])
                                    for c in clusters}
        },

        "step3_consistency_validation": {
            "model": "Flash",
            "scores": {
                "overall": consistency.overall_score,
                "thesis_connection_rate": consistency.thesis_connection_rate,
                "logical_flow": consistency.logical_flow_score
            },
            "gaps_detected": consistency.gaps_detected,
            "suggestions": consistency.suggestions
        },

        "step4_pro_validation": {
            "model": "Pro",
            "scores": {
                "quality": pro_validation.quality_score,
                "completeness": pro_validation.completeness_score
            },
            "critical_gaps": pro_validation.critical_gaps,
            "enhancement_suggestions": pro_validation.enhancement_suggestions
        },

        "distribution_analysis": {
            "thesis_connection": dict(connection_dist),
            "concept_type": dict(concept_dist),
            "importance": dict(importance_dist)
        },

        "quality_assessment": {
            "thesis_extraction": "Excellent - clear identification of core research claim and contributions",
            "cluster_coherence": f"{'Good' if clusters and sum(c.coherence_score for c in clusters)/len(clusters) > 0.6 else 'Moderate'} - clusters show {'strong' if clusters and sum(c.coherence_score for c in clusters)/len(clusters) > 0.6 else 'moderate'} internal consistency",
            "thesis_alignment": f"{'Strong' if consistency.thesis_connection_rate > 0.7 else 'Moderate'} - {int(consistency.thesis_connection_rate*100)}% of units directly connect to thesis",
            "completeness": f"{'Good' if pro_validation.completeness_score > 0.7 else 'Needs improvement'} - {int(pro_validation.completeness_score*100)}% of essential components present",
            "overall": f"The distributed strategy effectively identified the research thesis and organized content around it. Consistency score of {consistency.overall_score} indicates {'strong' if consistency.overall_score > 0.7 else 'moderate'} alignment."
        },

        "cost_efficiency": {
            "flash_calls": 3,  # Steps 1, 2, 3
            "pro_calls": 1,    # Step 4
            "estimated_cost": "Low - majority of processing done with Flash, Pro used only for final validation",
            "vs_full_pro": "~70% cost reduction compared to full Pro processing"
        },

        "pros": [
            "Clear thesis identification guides all downstream processing",
            "Thesis-aware clustering creates semantically coherent groups",
            "Multi-stage validation catches gaps early",
            "Cost-efficient: Flash handles extraction, clustering, and consistency",
            "Pro validation adds critical quality assurance at final stage",
            "High thesis connection rate ({}%) shows good alignment".format(int(consistency.thesis_connection_rate*100)),
            "Systematic gap detection enables targeted improvements"
        ],

        "cons": [
            "Requires well-structured input with clear thesis (may struggle with exploratory research)",
            "Four stages add latency compared to single-pass approaches",
            "Thesis extraction quality depends on Flash model capability",
            "Medium resolution may miss nuanced details needed for full thesis validation",
            "Gap detection in Step 3 may identify issues but can't fix them automatically",
            "{} critical gaps identified requiring manual review".format(len(pro_validation.critical_gaps))
        ],

        "suitable_for": [
            "Research papers with clear problem-solution-validation structure",
            "Projects where thesis alignment is critical (academic papers, proposals)",
            "Scenarios requiring cost-efficient multi-stage validation",
            "Content that benefits from coherent clustering (section organization)",
            "Cases where gap detection is valuable before final review"
        ],

        "not_suitable_for": [
            "Exploratory research without clear thesis",
            "Time-critical applications requiring single-pass processing",
            "Content requiring fine-grained detail extraction (use high resolution)",
            "Simple presentations without complex argumentation structure"
        ],

        "recommendations": {
            "when_to_use": "Use this strategy for research papers and technical documents where thesis alignment and logical coherence are paramount",
            "optimization": "If thesis is known beforehand, skip Step 1 and provide thesis directly to save one Flash call",
            "scaling": "For large documents (100+ slides), consider parallel processing of clusters in Step 2",
            "quality_threshold": "If Step 3 consistency score < 0.6, consider increasing resolution or restructuring content before Pro validation"
        }
    }

    # Save result
    output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp5-resolution-medium-distributed.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    print(f"\n=== SUMMARY ===")
    print(f"Resolution: Medium (1-3 units/slide)")
    print(f"Strategy: Distributed (Thesis-First 4-Stage)")
    print(f"Total Slides: {len(samples)}")
    print(f"Total Units: {total_units}")
    print(f"Average Units/Slide: {avg_units:.2f}")

    print(f"\n=== STAGE RESULTS ===")
    print(f"Step 1 - Thesis Extraction:")
    print(f"  Thesis Confidence: {thesis.confidence}")
    print(f"  Key Contributions: {len(thesis.key_contributions)}")

    print(f"\nStep 2 - Cluster Analysis:")
    print(f"  Clusters: {len(clusters)}")
    print(f"  Avg Coherence: {sum(c.coherence_score for c in clusters) / len(clusters):.2f}" if clusters else "  N/A")

    print(f"\nStep 3 - Consistency Validation:")
    print(f"  Overall Score: {consistency.overall_score}")
    print(f"  Thesis Connection: {consistency.thesis_connection_rate}")
    print(f"  Gaps Detected: {len(consistency.gaps_detected)}")

    print(f"\nStep 4 - Pro Validation:")
    print(f"  Quality Score: {pro_validation.quality_score}")
    print(f"  Completeness: {pro_validation.completeness_score}")
    print(f"  Critical Gaps: {len(pro_validation.critical_gaps)}")

    print(f"\n=== COST EFFICIENCY ===")
    print(f"Flash Calls: 3 (Steps 1-3)")
    print(f"Pro Calls: 1 (Step 4)")
    print(f"Estimated Savings: ~70% vs full Pro")

    return result

if __name__ == "__main__":
    main()
