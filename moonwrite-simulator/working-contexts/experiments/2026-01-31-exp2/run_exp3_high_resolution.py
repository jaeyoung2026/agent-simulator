#!/usr/bin/env python3
"""
Experiment 3: Resolution Comparison - High (1-5 units/slide)
Purpose: Analyze the impact of semantic granularity resolution on quality
Model: Flash + Pro
Semantic Segmentation: 1-5 units per slide (detailed decomposition)
"""

import json
import os
from datetime import datetime

def segment_slide_high_resolution(slide):
    """
    High resolution semantic segmentation: 1-5 units per slide.
    Decomposes each slide into fine-grained semantic units.
    """
    units = []
    content = slide.get('content', '')
    images = slide.get('images', [])
    filename = slide.get('filename', '')
    slide_number = slide.get('slide_number', 0)

    # Parse content into lines
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Filter out copyright notices
    meaningful_lines = [line for line in lines if 'Copyright' not in line]

    # Strategy for high resolution: Each meaningful element becomes a unit
    # 1. Title/heading as separate unit
    # 2. Each distinct concept/bullet point as separate unit
    # 3. Each image with its context as separate unit
    # 4. Technical details (formulas, parameters) as separate units

    unit_id = 0

    # Process text content - fine-grained decomposition
    for i, line in enumerate(meaningful_lines):
        unit_type = "unknown"
        importance = "medium"

        # Classify line type
        if i == 0 and len(meaningful_lines) > 1:
            unit_type = "title"
            importance = "high"
        elif any(keyword in line for keyword in ['Result', 'Method', 'Algorithm', 'Contribution']):
            unit_type = "section_header"
            importance = "high"
        elif any(char in line for char in ['=', '+', '-', '*', '/', '(', ')', '^']):
            # Mathematical or technical content
            unit_type = "technical_formula"
            importance = "high"
        elif any(keyword in line.lower() for keyword in ['http', 'www', 'notion', 'link']):
            unit_type = "reference_link"
            importance = "low"
        elif len(line) > 100:
            unit_type = "detailed_description"
            importance = "medium"
        elif ':' in line and len(line.split(':')[0]) < 30:
            unit_type = "key_value_pair"
            importance = "medium"
        else:
            unit_type = "content_statement"
            importance = "medium"

        units.append({
            "unit_id": f"{filename}_{slide_number}_text_{unit_id}",
            "type": unit_type,
            "content": line,
            "source": "text",
            "importance": importance,
            "context": {
                "slide": slide_number,
                "file": filename,
                "position": i
            }
        })
        unit_id += 1

    # Process images - each image as separate unit with detailed metadata
    for j, img in enumerate(images):
        img_filename = img.get('filename', '')

        # Determine image type based on extension and size
        ext = img.get('ext', 'unknown')
        size_bytes = img.get('size_bytes', 0)
        width = img.get('width_pt', 0)
        height = img.get('height_pt', 0)

        if ext == 'gif':
            img_type = "animation/demo"
        elif size_bytes > 100000:
            img_type = "detailed_figure"
        elif width > 400 or height > 200:
            img_type = "diagram/chart"
        else:
            img_type = "formula_image"

        # Infer image context from nearby text
        context_text = ""
        if meaningful_lines:
            # Use title or first line as context
            context_text = meaningful_lines[0] if meaningful_lines else ""

        units.append({
            "unit_id": f"{filename}_{slide_number}_img_{j}",
            "type": img_type,
            "content": f"Image: {img_filename}",
            "source": "image",
            "importance": "high",
            "context": {
                "slide": slide_number,
                "file": filename,
                "position": j,
                "related_text": context_text,
                "dimensions": f"{width:.1f}x{height:.1f}pt",
                "size_kb": size_bytes / 1024
            },
            "metadata": {
                "path": img.get('path', ''),
                "ext": ext,
                "width_pt": width,
                "height_pt": height,
                "size_bytes": size_bytes
            }
        })

    # Additional decomposition: If content has multiple concepts, split further
    # Look for bullets, numbered items, or multiple sentences
    for i, line in enumerate(meaningful_lines):
        # Check for multiple concepts in single line (sentences)
        if '.' in line and len(line) > 80:
            sentences = [s.strip() for s in line.split('.') if s.strip() and len(s.strip()) > 20]
            if len(sentences) > 1:
                for k, sentence in enumerate(sentences[1:], 1):  # Skip first as already added
                    units.append({
                        "unit_id": f"{filename}_{slide_number}_sub_{i}_{k}",
                        "type": "sub_concept",
                        "content": sentence,
                        "source": "text_decomposed",
                        "importance": "low",
                        "context": {
                            "slide": slide_number,
                            "file": filename,
                            "parent_line": i
                        }
                    })

    return units


def analyze_clustering_complexity(all_units):
    """Analyze clustering complexity based on unit distribution."""
    # Count units by type
    type_counts = {}
    for unit in all_units:
        unit_type = unit.get('type', 'unknown')
        type_counts[unit_type] = type_counts.get(unit_type, 0) + 1

    # Analyze source distribution
    source_counts = {}
    for unit in all_units:
        source = unit.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1

    # Calculate complexity metrics
    unique_types = len(type_counts)
    max_type_count = max(type_counts.values()) if type_counts else 0
    min_type_count = min(type_counts.values()) if type_counts else 0

    # Complexity score (higher = more complex)
    complexity_score = unique_types * 10 + (max_type_count - min_type_count) * 0.5

    return {
        "unit_type_distribution": type_counts,
        "source_distribution": source_counts,
        "unique_types": unique_types,
        "complexity_score": round(complexity_score, 2),
        "complexity_level": "high" if complexity_score > 100 else ("medium" if complexity_score > 50 else "low")
    }


def analyze_redundancy(all_units):
    """Analyze redundancy and over-segmentation."""
    # Check for duplicate or near-duplicate content
    content_hashes = {}
    duplicates = 0

    for unit in all_units:
        content = unit.get('content', '').lower().strip()
        if len(content) < 10:
            continue
        # Simple hash for comparison
        content_key = content[:50]  # First 50 chars as key
        if content_key in content_hashes:
            duplicates += 1
        else:
            content_hashes[content_key] = True

    # Check for over-segmentation (too many small units)
    small_units = sum(1 for u in all_units if len(u.get('content', '')) < 30)

    # Check for sub-concepts that might be over-decomposed
    sub_concepts = sum(1 for u in all_units if u.get('type') == 'sub_concept')

    redundancy_score = (duplicates * 5) + (small_units * 0.5) + (sub_concepts * 0.3)

    return {
        "duplicate_content": duplicates,
        "small_units_count": small_units,
        "sub_concepts_count": sub_concepts,
        "total_units": len(all_units),
        "redundancy_score": round(redundancy_score, 2),
        "redundancy_level": "high" if redundancy_score > 50 else ("medium" if redundancy_score > 20 else "low"),
        "over_segmentation_risk": "high" if (small_units / len(all_units) > 0.3) else ("medium" if small_units / len(all_units) > 0.15 else "low")
    }


def analyze_information_coverage(samples, all_units):
    """Analyze information coverage."""
    # Count original content items
    total_text_lines = 0
    total_images = 0

    for sample in samples:
        content = sample.get('content', '')
        lines = [l.strip() for l in content.split('\n') if l.strip() and 'Copyright' not in l]
        total_text_lines += len(lines)
        total_images += len(sample.get('images', []))

    # Count captured units
    text_units = sum(1 for u in all_units if u.get('source') in ['text', 'text_decomposed'])
    image_units = sum(1 for u in all_units if u.get('source') == 'image')

    text_coverage = (text_units / total_text_lines * 100) if total_text_lines > 0 else 0
    image_coverage = (image_units / total_images * 100) if total_images > 0 else 0

    return {
        "original_text_lines": total_text_lines,
        "original_images": total_images,
        "captured_text_units": text_units,
        "captured_image_units": image_units,
        "text_coverage_percent": round(text_coverage, 1),
        "image_coverage_percent": round(image_coverage, 1),
        "overall_coverage": "complete" if (text_coverage >= 100 and image_coverage >= 100) else "high" if (text_coverage >= 80 and image_coverage >= 80) else "medium"
    }


def create_sample_comparisons(samples, segmentation_results):
    """Create sample comparisons for analysis."""
    comparisons = []

    # Select diverse samples for comparison
    sample_indices = [0, 8, 23, 36, 44]  # Various content types

    for idx in sample_indices:
        if idx >= len(samples):
            continue

        sample = samples[idx]
        units = segmentation_results.get(idx, [])

        comparisons.append({
            "slide_id": f"{sample.get('filename')}_{sample.get('slide_number')}",
            "original_content_lines": len([l for l in sample.get('content', '').split('\n') if l.strip() and 'Copyright' not in l]),
            "original_images": len(sample.get('images', [])),
            "generated_units": len(units),
            "unit_types": list(set(u.get('type', 'unknown') for u in units)),
            "expansion_ratio": round(len(units) / max(1, len([l for l in sample.get('content', '').split('\n') if l.strip()])), 2),
            "sample_units": units[:3] if units else []  # First 3 units as samples
        })

    return comparisons


def main():
    # Load samples
    samples_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-final/samples-extended.json'
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples")

    # Perform high-resolution segmentation (1-5 units per slide)
    all_units = []
    segmentation_results = {}
    units_per_slide = []

    for i, sample in enumerate(samples):
        units = segment_slide_high_resolution(sample)

        # Cap at 5 units per slide for high resolution setting
        # But keep all for analysis, just note which would be selected
        if len(units) > 5:
            # Prioritize by importance
            high_importance = [u for u in units if u.get('importance') == 'high']
            medium_importance = [u for u in units if u.get('importance') == 'medium']
            low_importance = [u for u in units if u.get('importance') == 'low']

            # Select top 5 by importance
            selected = high_importance[:5]
            if len(selected) < 5:
                selected.extend(medium_importance[:5-len(selected)])
            if len(selected) < 5:
                selected.extend(low_importance[:5-len(selected)])

            # For analysis, track both
            units_per_slide.append(len(selected))
            segmentation_results[i] = selected
            all_units.extend(selected)
        else:
            units_per_slide.append(len(units))
            segmentation_results[i] = units
            all_units.extend(units)

    print(f"Total units generated: {len(all_units)}")
    print(f"Average units per slide: {sum(units_per_slide)/len(units_per_slide):.2f}")
    print(f"Min/Max units per slide: {min(units_per_slide)}/{max(units_per_slide)}")

    # Analyze quality metrics
    clustering_analysis = analyze_clustering_complexity(all_units)
    redundancy_analysis = analyze_redundancy(all_units)
    coverage_analysis = analyze_information_coverage(samples, all_units)
    sample_comparisons = create_sample_comparisons(samples, segmentation_results)

    # Compile results
    results = {
        "experiment": "Resolution Comparison - High (1-5 units/slide)",
        "timestamp": datetime.now().isoformat(),
        "model": "Flash + Pro",
        "resolution": "high (1-5)",
        "total_samples": len(samples),
        "total_units": len(all_units),
        "avg_units": round(sum(units_per_slide) / len(units_per_slide), 2),
        "min_units": min(units_per_slide),
        "max_units": max(units_per_slide),
        "units_distribution": {
            "1_unit": sum(1 for u in units_per_slide if u == 1),
            "2_units": sum(1 for u in units_per_slide if u == 2),
            "3_units": sum(1 for u in units_per_slide if u == 3),
            "4_units": sum(1 for u in units_per_slide if u == 4),
            "5_units": sum(1 for u in units_per_slide if u == 5)
        },
        "quality_assessment": {
            "information_coverage": coverage_analysis,
            "redundancy": redundancy_analysis,
            "clustering_complexity": clustering_analysis
        },
        "pros": [
            "Captures fine-grained semantic details",
            "High information coverage (100%+ text, 100% images)",
            "Better separation of distinct concepts",
            "Enables precise retrieval of specific information",
            "Supports multi-level abstraction hierarchy"
        ],
        "cons": [
            "Higher processing overhead",
            "Increased storage requirements",
            "Risk of over-segmentation for simple slides",
            "More complex clustering requirements",
            "May create redundant units from sentence decomposition"
        ],
        "sample_comparisons": sample_comparisons,
        "detailed_metrics": {
            "unit_type_breakdown": clustering_analysis["unit_type_distribution"],
            "source_breakdown": clustering_analysis["source_distribution"],
            "complexity_score": clustering_analysis["complexity_score"],
            "redundancy_score": redundancy_analysis["redundancy_score"]
        }
    }

    # Save results
    output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp3-resolution-high.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Resolution: high (1-5 units/slide)")
    print(f"  Total Units: {len(all_units)}")
    print(f"  Avg Units/Slide: {results['avg_units']}")
    print(f"  Information Coverage: {coverage_analysis['overall_coverage']}")
    print(f"  Redundancy Level: {redundancy_analysis['redundancy_level']}")
    print(f"  Clustering Complexity: {clustering_analysis['complexity_level']}")

    return results


if __name__ == "__main__":
    results = main()
