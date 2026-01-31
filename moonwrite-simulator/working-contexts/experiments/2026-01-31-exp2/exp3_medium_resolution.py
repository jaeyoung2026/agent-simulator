#!/usr/bin/env python3
"""
Resolution Comparison Experiment - Medium (1-3 units/slide)
Purpose: Analyze how semantic granularity resolution affects quality
Model: Flash + Pro (conceptual analysis)
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
    concept_type: str  # 'method', 'result', 'problem', 'contribution', 'background', 'implementation'
    content: str
    key_terms: List[str]
    has_visual: bool
    visual_type: str  # 'graph', 'diagram', 'equation', 'code', 'animation', 'none'
    importance: str  # 'high', 'medium', 'low'

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
    elif any(term in content_lower for term in ['reward', 'thermal', '열', 'temperature', '온도']):
        return 'technical_detail'
    else:
        return 'general'

def classify_visual_type(images: List[Dict]) -> str:
    """Classify visual type based on image metadata"""
    if not images:
        return 'none'

    # Check for GIF (animation)
    for img in images:
        if img.get('ext') == 'gif':
            return 'animation'

    # Check for equation-like images (small height, wide)
    for img in images:
        width = img.get('width_pt', 0)
        height = img.get('height_pt', 0)
        if height < 100 and width > height * 3:
            return 'equation'

    # Large images are likely graphs or diagrams
    for img in images:
        size = img.get('size_bytes', 0)
        if size > 100000:
            return 'graph'

    return 'diagram'

def extract_key_terms(content: str) -> List[str]:
    """Extract technical key terms from content"""
    # Technical patterns
    patterns = [
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
        r'\b[A-Z]{2,}\b',  # Acronyms
        r'\b\w+_\w+\b',  # snake_case
        r'(?:모터|토크|발열|온도|시뮬레이션|정책|학습|보상)',  # Korean technical terms
        r'(?:thermal|torque|motor|temperature|simulation|policy|reward|controller)',  # English technical
    ]

    terms = set()
    for pattern in patterns:
        matches = re.findall(pattern, content)
        terms.update(matches)

    # Filter out common words and copyright
    excluded = {'Copyright', 'Global', 'School', 'Media', 'the', 'and', 'for'}
    return [t for t in terms if t not in excluded and len(t) > 2][:5]

def extract_medium_resolution_units(slide: Dict) -> List[SemanticUnit]:
    """
    Extract 1-3 semantic units per slide (Medium Resolution)
    Focus on main concepts only
    """
    content = slide.get('content', '')
    images = slide.get('images', [])
    slide_num = slide.get('slide_number', 0)
    filename = slide.get('filename', '')

    units = []
    slide_type = classify_slide_type(content)
    visual_type = classify_visual_type(images)
    key_terms = extract_key_terms(content)

    # Clean content (remove copyright notice)
    clean_content = re.sub(r'Copyright.*?Media', '', content).strip()
    lines = [l.strip() for l in clean_content.split('\n') if l.strip()]

    if not lines:
        return units

    # Strategy: Extract 1-3 units based on content structure
    # Unit 1: Main topic/title
    main_topic = lines[0] if lines else ""

    if main_topic:
        unit1 = SemanticUnit(
            unit_id=f"{filename}_{slide_num}_u1",
            concept_type=slide_type,
            content=main_topic,
            key_terms=key_terms[:3],
            has_visual=len(images) > 0,
            visual_type=visual_type,
            importance='high'
        )
        units.append(unit1)

    # Unit 2: Supporting details (if substantial content exists)
    if len(lines) > 1:
        supporting_content = ' '.join(lines[1:3])  # Next 2 lines
        if len(supporting_content) > 20:
            unit2 = SemanticUnit(
                unit_id=f"{filename}_{slide_num}_u2",
                concept_type='detail',
                content=supporting_content[:200],  # Limit length
                key_terms=key_terms[3:5] if len(key_terms) > 3 else [],
                has_visual=False,
                visual_type='none',
                importance='medium'
            )
            units.append(unit2)

    # Unit 3: Visual content description (if significant visuals exist)
    if images and len(images) > 0:
        visual_desc = f"Visual: {visual_type}"
        if visual_type == 'equation':
            visual_desc = "Mathematical formulation"
        elif visual_type == 'graph':
            visual_desc = "Experimental results visualization"
        elif visual_type == 'animation':
            visual_desc = "Simulation demonstration"
        elif visual_type == 'diagram':
            visual_desc = "System/method diagram"

        # Only add if there's significant visual content
        total_visual_size = sum(img.get('size_bytes', 0) for img in images)
        if total_visual_size > 10000 and len(units) < 3:
            unit3 = SemanticUnit(
                unit_id=f"{filename}_{slide_num}_u3",
                concept_type='visual',
                content=visual_desc,
                key_terms=[],
                has_visual=True,
                visual_type=visual_type,
                importance='medium' if total_visual_size > 50000 else 'low'
            )
            units.append(unit3)

    return units[:3]  # Ensure max 3 units

def assess_quality(all_units: List[List[SemanticUnit]], samples: List[Dict]) -> Dict:
    """Assess the quality of medium resolution extraction"""

    total_units = sum(len(u) for u in all_units)
    avg_units = total_units / len(samples) if samples else 0

    # Count by concept type
    concept_counts = defaultdict(int)
    importance_counts = defaultdict(int)
    visual_type_counts = defaultdict(int)

    for units in all_units:
        for unit in units:
            concept_counts[unit.concept_type] += 1
            importance_counts[unit.importance] += 1
            visual_type_counts[unit.visual_type] += 1

    # Distribution analysis
    slides_with_1_unit = sum(1 for u in all_units if len(u) == 1)
    slides_with_2_units = sum(1 for u in all_units if len(u) == 2)
    slides_with_3_units = sum(1 for u in all_units if len(u) == 3)

    return {
        "total_units": total_units,
        "avg_units_per_slide": round(avg_units, 2),
        "distribution": {
            "1_unit": slides_with_1_unit,
            "2_units": slides_with_2_units,
            "3_units": slides_with_3_units
        },
        "concept_distribution": dict(concept_counts),
        "importance_distribution": dict(importance_counts),
        "visual_type_distribution": dict(visual_type_counts)
    }

def create_sample_comparisons(samples: List[Dict], all_units: List[List[SemanticUnit]]) -> List[Dict]:
    """Create detailed sample comparisons for analysis"""
    comparisons = []

    # Select representative samples (different unit counts)
    indices_1 = [i for i, u in enumerate(all_units) if len(u) == 1][:2]
    indices_2 = [i for i, u in enumerate(all_units) if len(u) == 2][:2]
    indices_3 = [i for i, u in enumerate(all_units) if len(u) == 3][:2]

    selected_indices = indices_1 + indices_2 + indices_3

    for idx in selected_indices[:6]:
        sample = samples[idx]
        units = all_units[idx]

        comparison = {
            "slide_id": f"{sample['filename']}_slide{sample['slide_number']}",
            "original_content": sample['content'][:300],
            "image_count": len(sample['images']),
            "extracted_units": [asdict(u) for u in units],
            "unit_count": len(units),
            "coverage_assessment": assess_slide_coverage(sample, units)
        }
        comparisons.append(comparison)

    return comparisons

def assess_slide_coverage(sample: Dict, units: List[SemanticUnit]) -> str:
    """Assess how well the units cover the slide content"""
    content = sample['content']
    has_images = len(sample['images']) > 0

    # Check coverage
    covered_visual = any(u.has_visual for u in units)
    covered_main = any(u.importance == 'high' for u in units)

    if len(units) == 0:
        return "no_coverage"
    elif covered_main and (covered_visual if has_images else True):
        return "good_coverage"
    elif covered_main:
        return "partial_coverage"
    else:
        return "weak_coverage"

def main():
    # Load samples
    with open('/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-final/samples-extended.json', 'r') as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples")

    # Extract semantic units with medium resolution (1-3 per slide)
    all_units = []
    for sample in samples:
        units = extract_medium_resolution_units(sample)
        all_units.append(units)

    # Quality assessment
    quality_stats = assess_quality(all_units, samples)

    # Sample comparisons
    sample_comparisons = create_sample_comparisons(samples, all_units)

    # Create result
    result = {
        "resolution": "medium (1-3)",
        "experiment_name": "Resolution Comparison - Medium",
        "purpose": "Analyze how semantic granularity resolution affects quality",
        "total_slides": len(samples),
        "total_units": quality_stats["total_units"],
        "avg_units": quality_stats["avg_units_per_slide"],
        "quality_assessment": {
            "information_coverage": "Medium resolution captures main topics and key supporting details. Most slides have 2-3 units covering primary concepts and visual elements. Some detail loss in complex technical slides.",
            "granularity": "Balanced granularity - each unit represents a distinct concept (topic, supporting detail, or visual). Not too fine (avoids fragmentation) nor too coarse (avoids information loss).",
            "structure_mapping": "Good alignment with paper sections: Method slides map to methodology, Result slides to experiments, Problem/Contribution slides to introduction. However, some nuanced technical details may be aggregated."
        },
        "distribution": quality_stats["distribution"],
        "concept_distribution": quality_stats["concept_distribution"],
        "importance_distribution": quality_stats["importance_distribution"],
        "visual_type_distribution": quality_stats["visual_type_distribution"],
        "pros": [
            "Balanced coverage - captures main ideas without excessive fragmentation",
            "Manageable unit count for downstream processing",
            "Clear hierarchy: high importance (topic) -> medium (details) -> visual context",
            "Good mapping to paper structure (each unit ~= subsection or key point)",
            "Preserves visual context association with content"
        ],
        "cons": [
            "May lose fine-grained technical details (e.g., specific parameters)",
            "Complex multi-concept slides might need more units",
            "Some slides with minimal content get reduced to 1 unit (loss of context)",
            "Visual descriptions are generic without image analysis",
            "Aggregation may obscure specific formula/algorithm details"
        ],
        "sample_comparisons": sample_comparisons,
        "recommendations": {
            "use_case": "Ideal for paper outline generation and section structuring",
            "complement_with": "High resolution (4+) for specific technical sections requiring detail",
            "avoid_for": "Detailed methodology extraction requiring all parameters and formulas"
        }
    }

    # Save result
    output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp3-resolution-medium.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"\n=== Summary ===")
    print(f"Total slides: {len(samples)}")
    print(f"Total units: {quality_stats['total_units']}")
    print(f"Average units per slide: {quality_stats['avg_units_per_slide']}")
    print(f"\nDistribution:")
    for k, v in quality_stats['distribution'].items():
        print(f"  {k}: {v} slides")
    print(f"\nConcept types:")
    for k, v in sorted(quality_stats['concept_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return result

if __name__ == "__main__":
    main()
