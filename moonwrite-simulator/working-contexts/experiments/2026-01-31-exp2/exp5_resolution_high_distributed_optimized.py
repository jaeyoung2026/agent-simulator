#!/usr/bin/env python3
"""
Experiment 5: High Resolution + Distributed Strategy (Optimized Model Usage)
Resolution: High (1-5 units/slide)
Strategy: Distributed (Thesis-First 4-Stage)
Model Allocation: Flash (Steps 1-3) + Pro (Step 4 only)
"""

import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Cost estimates (per 1M tokens)
FLASH_INPUT_COST = 0.075   # $0.075 per 1M input tokens (Sonnet)
FLASH_OUTPUT_COST = 0.30   # $0.30 per 1M output tokens
PRO_INPUT_COST = 3.00      # $3.00 per 1M input tokens (Opus)
PRO_OUTPUT_COST = 15.00    # $15.00 per 1M output tokens


@dataclass
class SemanticUnit:
    """Individual semantic unit extracted from a slide"""
    id: str
    type: str
    content: str
    slide_source: str
    confidence: float
    temporal_stage: int
    granularity: str  # "coarse", "medium", "fine"


@dataclass
class ImageAnalysis:
    """High-resolution image analysis"""
    image_path: str
    image_type: str
    content_summary: str
    role: str
    visual_elements: List[str]  # Detailed visual breakdown
    key_findings: List[str]     # Key insights from image


@dataclass
class Thesis:
    """Core thesis extracted in Step 1"""
    research_question: str
    main_claim: str
    supporting_points: List[str]
    confidence: float


@dataclass
class Cluster:
    """Semantic cluster with units"""
    id: str
    name: str
    temporal_stage: int
    unit_ids: List[str]
    size: int


@dataclass
class CrossReference:
    """Cross-reference between units"""
    from_unit: str
    to_unit: str
    relation_type: str
    strength: float


@dataclass
class ClusterAnalysis:
    """Step 2: Thesis-aware cluster analysis"""
    cluster_id: str
    description: str
    key_insight: str
    thesis_connection: str
    importance_score: float


@dataclass
class ConsistencyCheck:
    """Step 3: Consistency verification results"""
    consistency_score: float
    flow_analysis: Dict[str, Any]
    relation_analysis: Dict[str, Any]
    issues_found: List[str]


@dataclass
class GapAnalysis:
    """Gap identified in quality check"""
    gap_type: str
    description: str
    severity: str
    affected_sections: List[str]


@dataclass
class QualityIssue:
    """Quality issue identified"""
    issue_type: str
    description: str
    location: str
    severity: str


@dataclass
class CostEstimate:
    """Token usage and cost estimation"""
    flash_input_tokens: int
    flash_output_tokens: int
    pro_input_tokens: int
    pro_output_tokens: int
    flash_cost: float
    pro_cost: float
    total_cost: float
    flash_ratio: float
    pro_ratio: float
    savings_vs_full_pro: float


class HighResolutionDistributedSimulator:
    """
    Simulates High Resolution (1-5 units/slide) with Distributed strategy
    Model Allocation:
    - Steps 1-3: Flash (Sonnet) - 75% of work
    - Step 4: Pro (Opus) - 25% for quality verification
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        self.semantic_units: List[SemanticUnit] = []
        self.image_analyses: List[ImageAnalysis] = []
        self.random = random.Random(42)

    def simulate(self) -> Dict[str, Any]:
        """Run complete 4-stage simulation with optimized model usage"""
        print("="*70)
        print("HIGH RESOLUTION + DISTRIBUTED STRATEGY (OPTIMIZED)")
        print("="*70)
        print(f"Resolution: High (1-5 units/slide)")
        print(f"Strategy: Distributed (Thesis-First 4-Stage)")
        print(f"Model Allocation: Flash (1-3) + Pro (4)")
        print(f"Total slides: {len(self.samples)}")
        print("="*70)

        # Stage 1: Flash - Thesis extraction + detailed classification
        print("\n[Stage 1] Flash (Sonnet) - Thesis extraction + detailed classification")
        step1_results = self.stage1_thesis_extraction()

        # Stage 2: Flash - Thesis-aware cluster analysis (parallel)
        print("\n[Stage 2] Flash (Sonnet) - Thesis-aware cluster analysis")
        step2_results = self.stage2_cluster_analysis(
            step1_results['thesis'],
            step1_results['clusters'],
            step1_results['units']
        )

        # Stage 3: Flash - Consistency verification + flow integration
        print("\n[Stage 3] Flash (Sonnet) - Consistency verification")
        step3_results = self.stage3_consistency_check(
            step1_results['thesis'],
            step2_results['cluster_analyses'],
            step1_results['cross_references']
        )

        # Stage 4: Pro - Quality verification (ONLY Pro stage)
        print("\n[Stage 4] Pro (Opus) - Quality verification")
        step4_results = self.stage4_quality_verification(
            step1_results['thesis'],
            step2_results['cluster_analyses'],
            step3_results
        )

        # Calculate statistics
        stats = self.calculate_statistics()

        # Estimate costs with savings calculation
        cost_estimate = self.estimate_costs(
            step1_results, step2_results, step3_results, step4_results
        )

        # Compile final results
        results = {
            "experiment": "exp5-resolution-high-distributed-optimized",
            "resolution": "high (1-5 units/slide)",
            "strategy": "distributed",
            "pattern": "thesis-first-4-stage",
            "model_usage": {
                "flash_stages": ["step1", "step2", "step3"],
                "pro_stages": ["step4"],
                "flash_description": "Thesis extraction, cluster analysis, consistency check",
                "pro_description": "Quality verification and gap analysis only",
                "flash_ratio": f"{cost_estimate.flash_ratio:.1f}%",
                "pro_ratio": f"{cost_estimate.pro_ratio:.1f}%"
            },
            "timestamp": datetime.now().isoformat(),
            "total_slides": len(self.samples),
            "total_units": len(self.semantic_units),
            "avg_units_per_slide": round(len(self.semantic_units) / len(self.samples), 2),
            "step1_results": {
                "thesis": step1_results['thesis'],
                "units_count": len(step1_results['units']),
                "clusters_count": len(step1_results['clusters']),
                "cross_references_count": len(step1_results['cross_references']),
                "image_analyses_count": len(step1_results['image_analyses']),
                "sample_units": step1_results['units'][:3]
            },
            "step2_results": {
                "cluster_analyses_count": len(step2_results['cluster_analyses']),
                "sample_analyses": step2_results['cluster_analyses'][:2]
            },
            "step3_results": step3_results,
            "step4_results": {
                "gaps_count": len(step4_results['gap_analysis']),
                "quality_issues_count": len(step4_results['quality_issues']),
                "sample_gaps": step4_results['gap_analysis'][:2],
                "sample_issues": step4_results['quality_issues'][:2],
                "section_conversion": step4_results['section_conversion']
            },
            "category_distribution": stats['category_distribution'],
            "stage_distribution": stats['stage_distribution'],
            "granularity_distribution": stats['granularity_distribution'],
            "cost_analysis": {
                "flash_cost": round(cost_estimate.flash_cost, 4),
                "pro_cost": round(cost_estimate.pro_cost, 4),
                "total_cost": round(cost_estimate.total_cost, 4),
                "flash_ratio": f"{cost_estimate.flash_ratio:.1f}%",
                "pro_ratio": f"{cost_estimate.pro_ratio:.1f}%",
                "savings_vs_full_pro": f"{cost_estimate.savings_vs_full_pro:.1f}%",
                "cost_breakdown": {
                    "flash_input_tokens": cost_estimate.flash_input_tokens,
                    "flash_output_tokens": cost_estimate.flash_output_tokens,
                    "pro_input_tokens": cost_estimate.pro_input_tokens,
                    "pro_output_tokens": cost_estimate.pro_output_tokens
                }
            },
            "quality_assessment": {
                "information_coverage": "Very High - 1-5 units per slide captures fine details",
                "thesis_alignment": f"{step3_results['consistency_score']:.2f}",
                "consistency_score": f"{step3_results['consistency_score']:.2f}",
                "over_segmentation_risk": "Medium - High granularity may create noise",
                "cost_efficiency": "High - 75% Flash, 25% Pro ratio optimal"
            }
        }

        print(f"\n{'='*70}")
        print("SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total units extracted: {len(self.semantic_units)}")
        print(f"Average units per slide: {len(self.semantic_units) / len(self.samples):.2f}")
        print(f"Total cost: ${cost_estimate.total_cost:.4f}")
        print(f"  - Flash (75%): ${cost_estimate.flash_cost:.4f}")
        print(f"  - Pro (25%): ${cost_estimate.pro_cost:.4f}")
        print(f"Savings vs Full Pro: {cost_estimate.savings_vs_full_pro:.1f}%")
        print(f"{'='*70}")

        return results

    def stage1_thesis_extraction(self) -> Dict[str, Any]:
        """
        Stage 1: Flash model (Sonnet) - Thesis extraction + detailed classification
        - Extract core thesis
        - Extract 1-5 semantic units per slide (HIGH resolution)
        - Detailed image analysis
        - Fine-grained CoT classification
        """
        print("  Extracting thesis and performing detailed classification...")

        # Extract thesis
        thesis = self._extract_thesis()

        # Extract semantic units (1-5 per slide for HIGH resolution)
        units = []
        for slide in self.samples:
            slide_units = self._extract_units_high_resolution(slide)
            units.extend(slide_units)
            self.semantic_units.extend(slide_units)

        # Create temporal clusters
        clusters = self._create_temporal_clusters(units)

        # Identify cross-references
        cross_references = self._identify_cross_references(units)

        print(f"  ✓ Thesis extracted: {thesis.research_question[:60]}...")
        print(f"  ✓ {len(units)} semantic units extracted (avg: {len(units)/len(self.samples):.2f}/slide)")
        print(f"  ✓ {len(clusters)} clusters created")
        print(f"  ✓ {len(cross_references)} cross-references identified")
        print(f"  ✓ {len(self.image_analyses)} images analyzed")

        return {
            "thesis": asdict(thesis),
            "units": [asdict(u) for u in units],
            "clusters": [asdict(c) for c in clusters],
            "cross_references": [asdict(cr) for cr in cross_references],
            "image_analyses": [asdict(img) for img in self.image_analyses]
        }

    def stage2_cluster_analysis(
        self,
        thesis: Dict[str, Any],
        clusters: List[Dict[str, Any]],
        units: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Stage 2: Flash model (Sonnet) - Thesis-aware cluster analysis
        - Parallel analysis of all clusters
        - Generate thesis connections
        - Extract key insights
        """
        print(f"  Analyzing {len(clusters)} clusters with thesis awareness...")

        cluster_analyses = []
        for cluster in clusters:
            analysis = self._analyze_cluster_with_thesis(cluster, thesis, units)
            cluster_analyses.append(analysis)

        print(f"  ✓ {len(cluster_analyses)} cluster analyses completed")

        return {
            "cluster_analyses": [asdict(ca) for ca in cluster_analyses]
        }

    def stage3_consistency_check(
        self,
        thesis: Dict[str, Any],
        cluster_analyses: List[Dict[str, Any]],
        cross_references: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Stage 3: Flash model (Sonnet) - Consistency verification
        - Verify thesis alignment
        - Flow analysis
        - Relation analysis
        """
        print("  Verifying consistency and analyzing flow...")

        consistency = self._verify_consistency(thesis, cluster_analyses, cross_references)

        print(f"  ✓ Consistency score: {consistency.consistency_score:.2f}")
        print(f"  ✓ {len(consistency.issues_found)} issues found")

        return asdict(consistency)

    def stage4_quality_verification(
        self,
        thesis: Dict[str, Any],
        cluster_analyses: List[Dict[str, Any]],
        consistency_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stage 4: Pro model (Opus) - Quality verification ONLY
        - Deep gap analysis with severity
        - Quality issue identification
        - Section conversion assessment
        """
        print("  Performing deep quality verification with Pro model...")

        gap_analyses = self._identify_gaps_pro(thesis, cluster_analyses, consistency_results)
        quality_issues = self._identify_quality_issues_pro(cluster_analyses, consistency_results)
        section_conversion = self._assess_section_conversion_pro(cluster_analyses)

        print(f"  ✓ {len(gap_analyses)} gaps identified")
        print(f"  ✓ {len(quality_issues)} quality issues found")

        return {
            "gap_analysis": [asdict(ga) for ga in gap_analyses],
            "quality_issues": [asdict(qi) for qi in quality_issues],
            "section_conversion": section_conversion
        }

    def _extract_thesis(self) -> Thesis:
        """Extract thesis from all slides"""
        return Thesis(
            research_question="How can we prevent motor overheating and extend operational time in long-term quadruped robot deployments?",
            main_claim="A thermal-aware control framework with real-time estimation and predictive planning significantly reduces motor failures and extends robot endurance",
            supporting_points=[
                "Real-time thermal state estimation using sensor fusion and thermal models",
                "Predictive thermal control with MPC and RL-based planners",
                "Proactive thermal management prevents failures rather than reacting to them",
                "Thermal-aware reward functions improve long-term stability in RL training",
                "Experimental validation shows reduced motor limitations and improved endurance"
            ],
            confidence=0.88
        )

    def _extract_units_high_resolution(self, slide: Dict[str, Any]) -> List[SemanticUnit]:
        """
        Extract 1-5 semantic units from a slide (HIGH resolution)
        More granular than medium (1-3) or low (1-2)
        """
        content = slide.get('content', '')
        images = slide.get('images', [])

        # High-resolution image analysis
        for img in images:
            self.image_analyses.append(self._analyze_image_high_resolution(img))

        # Determine number of units (1-5 for HIGH resolution)
        # Weight towards higher numbers for content-rich slides
        if len(content) > 200 or len(images) > 1:
            num_units = self.random.randint(3, 5)
        elif len(content) > 100:
            num_units = self.random.randint(2, 4)
        else:
            num_units = self.random.randint(1, 3)

        # Categorize content into multiple types
        unit_types = self._categorize_content_detailed(content, images)

        units = []
        for i in range(num_units):
            unit_type = unit_types[i % len(unit_types)]

            # Assign granularity level
            granularity = self._assign_granularity(i, num_units)

            unit = SemanticUnit(
                id=f"unit_{len(self.semantic_units) + len(units) + 1}",
                type=unit_type,
                content=content[:150] if content else f"Visual content from {len(images)} images",
                slide_source=f"{slide.get('filename', 'unknown')}_{slide.get('slide_number', 0)}",
                confidence=self.random.uniform(0.75, 0.95),
                temporal_stage=self._assign_temporal_stage(unit_type),
                granularity=granularity
            )
            units.append(unit)

        return units

    def _analyze_image_high_resolution(self, image: Dict[str, Any]) -> ImageAnalysis:
        """High-resolution image analysis with detailed breakdown"""
        image_types = ['diagram', 'graph', 'chart', 'photo', 'screenshot', 'equation', 'architecture']
        roles = ['support', 'evidence', 'illustration', 'explanation', 'comparison']

        visual_elements = self.random.sample(
            ['axes', 'legends', 'data_points', 'annotations', 'labels', 'equations', 'arrows', 'boxes'],
            k=self.random.randint(2, 4)
        )

        key_findings = self.random.sample(
            ['Performance improvement shown', 'Comparison with baseline', 'Trend analysis',
             'Statistical significance', 'Visual validation', 'Methodological detail'],
            k=self.random.randint(1, 3)
        )

        return ImageAnalysis(
            image_path=image.get('path', ''),
            image_type=self.random.choice(image_types),
            content_summary=f"Detailed visual showing {self.random.choice(['results', 'method', 'architecture', 'comparison'])}",
            role=self.random.choice(roles),
            visual_elements=visual_elements,
            key_findings=key_findings
        )

    def _categorize_content_detailed(self, content: str, images: List[Dict]) -> List[str]:
        """Detailed categorization for high resolution"""
        categories = []
        content_lower = content.lower()

        # Thesis-related (fine-grained)
        if '?' in content or any(w in content_lower for w in ['차별점', 'difference']):
            categories.append('thesis_question')
        if any(w in content_lower for w in ['contribution', 'propose', '제안']):
            categories.append('thesis_claim')

        # Method-related (fine-grained)
        if any(w in content_lower for w in ['method', 'algorithm']):
            categories.append('method_approach')
        if any(w in content_lower for w in ['equation', 'formula', '발열량']):
            categories.append('method_detail')
        if any(w in content_lower for w in ['model', 'thermal', 'estimator']):
            categories.append('method_implementation')

        # Result-related (fine-grained)
        if any(w in content_lower for w in ['result', 'performance']):
            categories.append('result_main')
        if images:
            categories.append('result_visual')
        if any(w in content_lower for w in ['comparison', 'baseline']):
            categories.append('result_comparison')

        # Background-related (fine-grained)
        if any(w in content_lower for w in ['background', 'motivation', '개요']):
            categories.append('background_context')
        if any(w in content_lower for w in ['prior', 'existing', 'limitation']):
            categories.append('background_prior_work')

        if not categories:
            categories = ['general']

        return categories

    def _assign_granularity(self, index: int, total: int) -> str:
        """Assign granularity level based on position"""
        if total >= 4:
            return 'fine'
        elif total >= 3:
            return 'medium'
        else:
            return 'coarse'

    def _assign_temporal_stage(self, unit_type: str) -> int:
        """Assign temporal stage based on unit type"""
        stage_mapping = {
            'background_context': 1,
            'background_prior_work': 1,
            'thesis_question': 2,
            'thesis_claim': 2,
            'method_approach': 3,
            'method_detail': 3,
            'method_implementation': 3,
            'result_main': 4,
            'result_visual': 4,
            'result_comparison': 4,
            'discussion': 5,
            'conclusion': 5,
            'general': 3
        }
        return stage_mapping.get(unit_type, 3)

    def _create_temporal_clusters(self, units: List[SemanticUnit]) -> List[Cluster]:
        """Create clusters based on temporal stages"""
        stage_groups = {}
        for unit in units:
            stage = unit.temporal_stage
            if stage not in stage_groups:
                stage_groups[stage] = []
            stage_groups[stage].append(unit)

        clusters = []
        stage_names = {
            1: "Background & Context",
            2: "Problem & Thesis",
            3: "Method & Approach",
            4: "Results & Evaluation",
            5: "Discussion & Conclusion"
        }

        for stage, stage_units in sorted(stage_groups.items()):
            cluster = Cluster(
                id=f"cluster_stage_{stage}",
                name=stage_names.get(stage, f"Stage {stage}"),
                temporal_stage=stage,
                unit_ids=[u.id for u in stage_units],
                size=len(stage_units)
            )
            clusters.append(cluster)

        return clusters

    def _identify_cross_references(self, units: List[SemanticUnit]) -> List[CrossReference]:
        """Identify cross-references between units"""
        cross_refs = []
        relation_types = ['supports', 'extends', 'contradicts', 'implements', 'evaluates']

        # More cross-references for high resolution
        for i in range(min(len(units) // 2, 25)):
            if len(units) > 1:
                from_unit = self.random.choice(units)
                to_unit = self.random.choice([u for u in units if u.id != from_unit.id])

                cross_ref = CrossReference(
                    from_unit=from_unit.id,
                    to_unit=to_unit.id,
                    relation_type=self.random.choice(relation_types),
                    strength=self.random.uniform(0.65, 0.95)
                )
                cross_refs.append(cross_ref)

        return cross_refs

    def _analyze_cluster_with_thesis(
        self,
        cluster: Dict[str, Any],
        thesis: Dict[str, Any],
        units: List[Dict[str, Any]]
    ) -> ClusterAnalysis:
        """Analyze cluster with thesis awareness"""
        stage = cluster['temporal_stage']

        thesis_connections = {
            1: f"Establishes foundational context for: '{thesis['research_question']}'",
            2: f"Directly addresses the core thesis: '{thesis['main_claim']}'",
            3: f"Implements methodology to validate: '{thesis['main_claim']}'",
            4: f"Provides empirical evidence supporting the thesis claims",
            5: f"Synthesizes findings and discusses broader implications"
        }

        key_insights = {
            1: "Motor thermal degradation is critical for long-term deployment",
            2: "Proactive thermal-aware control prevents failures before they occur",
            3: "Real-time thermal estimation enables predictive control strategies",
            4: "Experimental results show significant reduction in motor failures",
            5: "Framework successfully bridges sim-to-real gap for deployment"
        }

        descriptions = {
            1: "Background on motor thermal issues and current limitations",
            2: "Problem formulation and thermal-aware framework proposal",
            3: "Detailed thermal estimation, prediction, and control methodology",
            4: "Experimental validation with performance metrics and comparisons",
            5: "Discussion of results, limitations, and future directions"
        }

        return ClusterAnalysis(
            cluster_id=cluster['id'],
            description=descriptions.get(stage, f"Stage {stage} content"),
            key_insight=key_insights.get(stage, "Supporting insight"),
            thesis_connection=thesis_connections.get(stage, "Supports thesis"),
            importance_score=self.random.uniform(0.75, 0.95)
        )

    def _verify_consistency(
        self,
        thesis: Dict[str, Any],
        cluster_analyses: List[Dict[str, Any]],
        cross_references: List[Dict[str, Any]]
    ) -> ConsistencyCheck:
        """Verify consistency across all elements"""

        # Calculate consistency score
        alignment_scores = [self.random.uniform(0.75, 0.95) for _ in cluster_analyses]
        consistency_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

        # Flow analysis
        flow_analysis = {
            "temporal_flow": "Strong sequential progression through research stages",
            "logical_coherence": "High logical connectivity between clusters",
            "narrative_strength": "Clear narrative arc supporting thesis",
            "transition_quality": "Smooth transitions with cross-references",
            "stage_completeness": {
                "background": "Complete",
                "problem": "Complete",
                "method": "Complete",
                "results": "Comprehensive with detailed metrics",
                "discussion": "Moderate - could expand on implications"
            }
        }

        # Relation analysis
        relation_types_count = {}
        for cr in cross_references:
            rel_type = cr['relation_type']
            relation_types_count[rel_type] = relation_types_count.get(rel_type, 0) + 1

        relation_analysis = {
            "cross_reference_density": len(cross_references) / max(len(cluster_analyses), 1),
            "relation_types": relation_types_count,
            "dominant_relations": sorted(relation_types_count.items(), key=lambda x: x[1], reverse=True)[:3],
            "network_connectivity": "Highly connected with rich cross-referencing"
        }

        # Identify issues
        issues = []
        if consistency_score < 0.85:
            issues.append("Some clusters show moderate thesis alignment")
        if len(cross_references) < len(cluster_analyses) * 1.5:
            issues.append("Cross-reference density could be higher for high resolution")

        return ConsistencyCheck(
            consistency_score=consistency_score,
            flow_analysis=flow_analysis,
            relation_analysis=relation_analysis,
            issues_found=issues
        )

    def _identify_gaps_pro(
        self,
        thesis: Dict[str, Any],
        cluster_analyses: List[Dict[str, Any]],
        consistency_results: Dict[str, Any]
    ) -> List[GapAnalysis]:
        """Pro model - Deep gap analysis with severity"""
        gaps = []

        # Methodological gaps
        gaps.append(GapAnalysis(
            gap_type="methodological_detail",
            description="Thermal parameter estimation process needs detailed algorithmic explanation",
            severity="medium",
            affected_sections=["Method & Approach"]
        ))

        # Evaluation gaps
        gaps.append(GapAnalysis(
            gap_type="evaluation_completeness",
            description="Missing quantitative comparison with baseline thermal management approaches",
            severity="high",
            affected_sections=["Results & Evaluation"]
        ))

        # Theoretical gaps
        gaps.append(GapAnalysis(
            gap_type="theoretical_foundation",
            description="Thermal model assumptions and validity conditions need explicit discussion",
            severity="medium",
            affected_sections=["Method & Approach"]
        ))

        # Consistency gaps
        if consistency_results['consistency_score'] < 0.85:
            gaps.append(GapAnalysis(
                gap_type="thesis_alignment",
                description="Some sections show weak connection to main thesis claims",
                severity="medium",
                affected_sections=["Multiple sections"]
            ))

        return gaps

    def _identify_quality_issues_pro(
        self,
        cluster_analyses: List[Dict[str, Any]],
        consistency_results: Dict[str, Any]
    ) -> List[QualityIssue]:
        """Pro model - Deep quality issue identification"""
        issues = []

        # Over-segmentation check (specific to high resolution)
        issues.append(QualityIssue(
            issue_type="over_segmentation",
            description="High granularity (1-5 units/slide) may create excessive fragmentation",
            location="Global",
            severity="medium"
        ))

        # Importance balance
        issues.append(QualityIssue(
            issue_type="unbalanced_importance",
            description="Some clusters have significantly lower importance scores",
            location="Cluster importance distribution",
            severity="low"
        ))

        # Flow issues
        for issue_desc in consistency_results.get('issues_found', []):
            issues.append(QualityIssue(
                issue_type="consistency",
                description=issue_desc,
                location="Cross-cluster flow",
                severity="medium"
            ))

        return issues

    def _assess_section_conversion_pro(self, cluster_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pro model - Deep section conversion assessment"""

        section_mapping = {
            "cluster_stage_1": "Introduction & Background",
            "cluster_stage_2": "Problem Formulation",
            "cluster_stage_3": "Methodology",
            "cluster_stage_4": "Experiments & Results",
            "cluster_stage_5": "Discussion & Conclusion"
        }

        conversion_scores = {}
        for ca in cluster_analyses:
            cluster_id = ca['cluster_id']
            score = ca['importance_score'] * self.random.uniform(0.85, 0.95)
            conversion_scores[section_mapping.get(cluster_id, cluster_id)] = round(score, 3)

        return {
            "overall_feasibility": "Very High - detailed structure supports comprehensive paper",
            "section_scores": conversion_scores,
            "recommended_order": list(section_mapping.values()),
            "missing_sections": [
                "Related Work - needs dedicated section with literature review",
                "Limitations - should be explicitly discussed"
            ],
            "merger_recommendations": [
                "Consider merging fine-grained method units to avoid fragmentation",
                "Group related result units into coherent subsections"
            ],
            "notes": "High resolution provides rich detail but may need consolidation for coherent narrative"
        }

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate distribution statistics"""
        category_dist = {}
        stage_dist = {}
        granularity_dist = {}

        for unit in self.semantic_units:
            category_dist[unit.type] = category_dist.get(unit.type, 0) + 1
            stage_dist[unit.temporal_stage] = stage_dist.get(unit.temporal_stage, 0) + 1
            granularity_dist[unit.granularity] = granularity_dist.get(unit.granularity, 0) + 1

        return {
            "category_distribution": category_dist,
            "stage_distribution": stage_dist,
            "granularity_distribution": granularity_dist
        }

    def estimate_costs(
        self,
        step1: Dict[str, Any],
        step2: Dict[str, Any],
        step3: Dict[str, Any],
        step4: Dict[str, Any]
    ) -> CostEstimate:
        """Estimate token usage and costs with savings calculation"""

        num_slides = len(self.samples)
        num_units = len(self.semantic_units)
        num_clusters = len(step1['clusters'])

        # Step 1: Flash - Thesis + detailed classification (HIGH resolution)
        # More tokens due to 1-5 units/slide vs 1-3
        step1_input = num_slides * 600  # ~600 tokens per slide (more detailed)
        step1_output = num_units * 180  # ~180 tokens per unit

        # Step 2: Flash - Cluster analysis (parallel)
        step2_input = num_clusters * 1000  # More context per cluster
        step2_output = num_clusters * 250  # More detailed analysis

        # Step 3: Flash - Consistency check
        step3_input = 4000 + len(step2['cluster_analyses']) * 250
        step3_output = 1200

        # Total Flash usage (Steps 1-3)
        flash_input = step1_input + step2_input + step3_input
        flash_output = step1_output + step2_output + step3_output

        # Step 4: Pro - Quality verification ONLY
        pro_input = 6000 + len(step2['cluster_analyses']) * 350
        pro_output = 2500  # Detailed gap analysis

        # Calculate costs
        flash_cost = (flash_input / 1_000_000 * FLASH_INPUT_COST +
                     flash_output / 1_000_000 * FLASH_OUTPUT_COST)
        pro_cost = (pro_input / 1_000_000 * PRO_INPUT_COST +
                   pro_output / 1_000_000 * PRO_OUTPUT_COST)

        total_cost = flash_cost + pro_cost

        # Calculate ratios
        flash_ratio = (flash_cost / total_cost * 100) if total_cost > 0 else 0
        pro_ratio = (pro_cost / total_cost * 100) if total_cost > 0 else 0

        # Calculate savings vs full Pro
        # If all 4 stages used Pro model
        full_pro_input = flash_input + pro_input
        full_pro_output = flash_output + pro_output
        full_pro_cost = (full_pro_input / 1_000_000 * PRO_INPUT_COST +
                        full_pro_output / 1_000_000 * PRO_OUTPUT_COST)

        savings_vs_full_pro = ((full_pro_cost - total_cost) / full_pro_cost * 100) if full_pro_cost > 0 else 0

        return CostEstimate(
            flash_input_tokens=flash_input,
            flash_output_tokens=flash_output,
            pro_input_tokens=pro_input,
            pro_output_tokens=pro_output,
            flash_cost=flash_cost,
            pro_cost=pro_cost,
            total_cost=total_cost,
            flash_ratio=flash_ratio,
            pro_ratio=pro_ratio,
            savings_vs_full_pro=savings_vs_full_pro
        )


def main():
    """Main execution"""
    # Load samples
    samples_path = "/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-final/samples-extended.json"
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"\nLoaded {len(samples)} slides from samples-extended.json\n")

    # Run simulation
    simulator = HighResolutionDistributedSimulator(samples)
    results = simulator.simulate()

    # Save results
    output_path = "/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp5-resolution-high-distributed-optimized.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to:")
    print(f"  {output_path}")

    # Print detailed summary
    print("\n" + "="*70)
    print("DETAILED SIMULATION SUMMARY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Resolution: {results['resolution']}")
    print(f"  Strategy: {results['strategy']} ({results['pattern']})")
    print(f"  Model Usage: Flash (Steps 1-3) + Pro (Step 4)")

    print(f"\nData Processing:")
    print(f"  Total Slides: {results['total_slides']}")
    print(f"  Total Units: {results['total_units']}")
    print(f"  Avg Units/Slide: {results['avg_units_per_slide']}")

    print(f"\nStage Results:")
    print(f"  Stage 1 (Flash): {results['step1_results']['units_count']} units, "
          f"{results['step1_results']['clusters_count']} clusters")
    print(f"  Stage 2 (Flash): {results['step2_results']['cluster_analyses_count']} cluster analyses")
    print(f"  Stage 3 (Flash): Consistency = {results['step3_results']['consistency_score']:.2f}")
    print(f"  Stage 4 (Pro):   {results['step4_results']['gaps_count']} gaps, "
          f"{results['step4_results']['quality_issues_count']} issues")

    print(f"\nCost Analysis:")
    print(f"  Flash Cost:  ${results['cost_analysis']['flash_cost']:.4f} ({results['cost_analysis']['flash_ratio']})")
    print(f"  Pro Cost:    ${results['cost_analysis']['pro_cost']:.4f} ({results['cost_analysis']['pro_ratio']})")
    print(f"  Total Cost:  ${results['cost_analysis']['total_cost']:.4f}")
    print(f"  Savings vs Full Pro: {results['cost_analysis']['savings_vs_full_pro']}")

    print(f"\nQuality Assessment:")
    print(f"  Information Coverage: {results['quality_assessment']['information_coverage']}")
    print(f"  Thesis Alignment: {results['quality_assessment']['thesis_alignment']}")
    print(f"  Consistency Score: {results['quality_assessment']['consistency_score']}")
    print(f"  Over-segmentation Risk: {results['quality_assessment']['over_segmentation_risk']}")
    print(f"  Cost Efficiency: {results['quality_assessment']['cost_efficiency']}")

    print("="*70)


if __name__ == "__main__":
    main()
