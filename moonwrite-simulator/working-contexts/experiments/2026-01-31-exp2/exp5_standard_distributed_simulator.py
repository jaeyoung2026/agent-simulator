#!/usr/bin/env python3
"""
Experiment 5: Standard + Distributed Strategy Simulation
Thesis-First 4-Stage Pattern with Flash and Pro models
"""

import json
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Cost estimates (per 1M tokens)
FLASH_INPUT_COST = 0.075  # $0.075 per 1M input tokens
FLASH_OUTPUT_COST = 0.30  # $0.30 per 1M output tokens
PRO_INPUT_COST = 3.00     # $3.00 per 1M input tokens
PRO_OUTPUT_COST = 15.00   # $15.00 per 1M output tokens


@dataclass
class SemanticUnit:
    """Individual semantic unit extracted from a slide"""
    id: str
    type: str  # thesis_question, thesis_claim, method, result, background, etc.
    content: str
    slide_source: str
    confidence: float
    temporal_stage: int  # 1-5 stage classification


@dataclass
class ImageAnalysis:
    """Standard image analysis"""
    image_path: str
    image_type: str  # diagram, graph, photo, screenshot, etc.
    content_summary: str
    role: str  # support, evidence, illustration


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
    relation_type: str  # supports, contradicts, extends, etc.
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
    severity: str  # high, medium, low
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


class StandardDistributedSimulator:
    """
    Simulates Standard option with Distributed (Thesis-First) strategy
    4-stage pattern: Flash -> Flash -> Flash -> Pro
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        self.semantic_units: List[SemanticUnit] = []
        self.image_analyses: List[ImageAnalysis] = []
        self.random = random.Random(42)  # For reproducibility

    def simulate(self) -> Dict[str, Any]:
        """Run complete 4-stage simulation"""
        print("Starting Standard + Distributed Strategy Simulation...")
        print(f"Total slides: {len(self.samples)}")

        # Stage 1: Flash - Thesis extraction + lightweight classification
        print("\n[Stage 1] Flash - Thesis extraction + lightweight classification")
        step1_results = self.stage1_thesis_extraction()

        # Stage 2: Flash - Thesis-aware cluster analysis (parallel)
        print("\n[Stage 2] Flash - Thesis-aware cluster analysis")
        step2_results = self.stage2_cluster_analysis(
            step1_results['thesis'],
            step1_results['clusters'],
            step1_results['units']
        )

        # Stage 3: Flash - Consistency verification + flow integration
        print("\n[Stage 3] Flash - Consistency verification")
        step3_results = self.stage3_consistency_check(
            step1_results['thesis'],
            step2_results['cluster_analyses'],
            step1_results['cross_references']
        )

        # Stage 4: Pro - Quality verification
        print("\n[Stage 4] Pro - Quality verification")
        step4_results = self.stage4_quality_verification(
            step1_results['thesis'],
            step2_results['cluster_analyses'],
            step3_results
        )

        # Calculate statistics
        stats = self.calculate_statistics()

        # Estimate costs
        cost_estimate = self.estimate_costs(
            step1_results, step2_results, step3_results, step4_results
        )

        # Compile final results
        results = {
            "condition": "Standard (Distributed)",
            "strategy": "distributed",
            "pattern": "thesis-first",
            "resolution": "medium (1-3 units/slide)",
            "model_usage": {
                "flash_step1": "thesis + lightweight classification",
                "flash_step2": "thesis-aware cluster analysis",
                "flash_step3": "consistency verification",
                "pro_step4": "quality verification"
            },
            "timestamp": datetime.now().isoformat(),
            "total_slides": len(self.samples),
            "total_units": len(self.semantic_units),
            "avg_units_per_slide": len(self.semantic_units) / len(self.samples),
            "step1_results": step1_results,
            "step2_results": step2_results,
            "step3_results": step3_results,
            "step4_results": step4_results,
            "category_distribution": stats['category_distribution'],
            "stage_distribution": stats['stage_distribution'],
            "sample_extractions": self.get_sample_extractions(5),
            "cost_estimate": asdict(cost_estimate)
        }

        print(f"\n✓ Simulation complete!")
        print(f"  Total units: {len(self.semantic_units)}")
        print(f"  Avg units/slide: {len(self.semantic_units) / len(self.samples):.2f}")
        print(f"  Estimated cost: ${cost_estimate.total_cost:.4f}")

        return results

    def stage1_thesis_extraction(self) -> Dict[str, Any]:
        """
        Stage 1: Flash model extracts thesis and performs lightweight classification
        - Core thesis (question + claim) extraction
        - Extract 1-3 semantic units per slide
        - Standard image analysis
        - CoT-based classification
        """
        print("  Extracting thesis and classifying units...")

        # Simulate thesis extraction
        thesis = self._extract_thesis()

        # Extract semantic units from slides (1-3 per slide)
        units = []
        for slide in self.samples:
            slide_units = self._extract_units_from_slide(slide)
            units.extend(slide_units)
            self.semantic_units.extend(slide_units)

        # Create clusters based on temporal stages
        clusters = self._create_temporal_clusters(units)

        # Identify cross-references
        cross_references = self._identify_cross_references(units)

        print(f"  ✓ Thesis extracted")
        print(f"  ✓ {len(units)} semantic units extracted")
        print(f"  ✓ {len(clusters)} clusters created")
        print(f"  ✓ {len(cross_references)} cross-references identified")

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
        Stage 2: Flash model performs thesis-aware cluster analysis (parallel)
        - Generate thesisConnection for each cluster
        - Extract keyInsight
        - Create cluster descriptions
        """
        print(f"  Analyzing {len(clusters)} clusters in parallel...")

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
        Stage 3: Flash model performs consistency verification
        - Verify thesisConnection ↔ thesis alignment
        - Flow analysis (flowAnalysis)
        - Relation analysis (relationAnalysis)
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
        Stage 4: Pro model performs quality verification
        - Gap analysis (with severity)
        - Quality issues identification
        - Section conversion feasibility
        """
        print("  Performing quality verification with Pro model...")

        gap_analyses = self._identify_gaps(thesis, cluster_analyses, consistency_results)
        quality_issues = self._identify_quality_issues(cluster_analyses, consistency_results)
        section_conversion = self._assess_section_conversion(cluster_analyses)

        print(f"  ✓ {len(gap_analyses)} gaps identified")
        print(f"  ✓ {len(quality_issues)} quality issues found")

        return {
            "gap_analysis": [asdict(ga) for ga in gap_analyses],
            "quality_issues": [asdict(qi) for qi in quality_issues],
            "section_conversion": section_conversion
        }

    def _extract_thesis(self) -> Thesis:
        """Simulate thesis extraction from all slides"""
        # Analyze content to extract thesis
        research_questions = [
            "How can we prevent motor overheating in long-term robot operations?",
            "Can thermal-aware control improve robot endurance and safety?",
            "What is the impact of motor temperature on quadruped robot performance?"
        ]

        main_claims = [
            "A thermal-aware framework can extend robot operational time by predicting and preventing motor overheating",
            "Real-time thermal estimation and predictive control significantly reduce motor failures",
            "Integrating thermal models into RL training improves long-term robot stability"
        ]

        supporting_points = [
            "Real-time thermal estimation using sensor data and heat models",
            "Predictive thermal control with MPC and RL planners",
            "Reduced motor limitations and improved long-term stability",
            "Thermal-aware reward functions in RL training"
        ]

        return Thesis(
            research_question=self.random.choice(research_questions),
            main_claim=self.random.choice(main_claims),
            supporting_points=supporting_points,
            confidence=self.random.uniform(0.75, 0.95)
        )

    def _extract_units_from_slide(self, slide: Dict[str, Any]) -> List[SemanticUnit]:
        """Extract 1-3 semantic units from a slide (medium resolution)"""
        content = slide.get('content', '')
        images = slide.get('images', [])

        # Analyze images (standard analysis)
        for img in images:
            self.image_analyses.append(self._analyze_image_standard(img))

        # Determine number of units (1-3 for medium resolution)
        num_units = self.random.randint(1, 3)

        # Categorize based on content
        unit_types = self._categorize_content(content, images)

        units = []
        for i in range(num_units):
            unit_type = unit_types[i % len(unit_types)]

            unit = SemanticUnit(
                id=f"unit_{len(self.semantic_units) + len(units) + 1}",
                type=unit_type,
                content=content[:200] if content else f"Visual content from {len(images)} images",
                slide_source=f"{slide.get('filename', 'unknown')}_{slide.get('slide_number', 0)}",
                confidence=self.random.uniform(0.7, 0.95),
                temporal_stage=self._assign_temporal_stage(unit_type)
            )
            units.append(unit)

        return units

    def _analyze_image_standard(self, image: Dict[str, Any]) -> ImageAnalysis:
        """Standard image analysis: type + content + role"""
        image_types = ['diagram', 'graph', 'chart', 'photo', 'screenshot', 'equation']
        roles = ['support', 'evidence', 'illustration', 'explanation']

        return ImageAnalysis(
            image_path=image.get('path', ''),
            image_type=self.random.choice(image_types),
            content_summary=f"Visual showing {self.random.choice(['results', 'method', 'architecture', 'comparison'])}",
            role=self.random.choice(roles)
        )

    def _categorize_content(self, content: str, images: List[Dict]) -> List[str]:
        """Categorize content into semantic types using CoT"""
        categories = []

        content_lower = content.lower()

        # Thesis-related
        if any(word in content_lower for word in ['algorithm', 'difference', '차별점', '?']):
            categories.append('thesis_question')
        if any(word in content_lower for word in ['contribution', 'propose', '제안']):
            categories.append('thesis_claim')

        # Method-related
        if any(word in content_lower for word in ['method', 'algorithm', 'model', 'thermal']):
            categories.append('method_approach')
        if any(word in content_lower for word in ['equation', 'formula', '발열량']):
            categories.append('method_detail')

        # Result-related
        if any(word in content_lower for word in ['result', 'performance', 'comparison']):
            categories.append('result_main')
        if images:
            categories.append('result_visual')

        # Background-related
        if any(word in content_lower for word in ['background', 'motivation', '개요']):
            categories.append('background_context')
        if any(word in content_lower for word in ['prior', 'existing', 'limitation']):
            categories.append('background_prior_work')

        if not categories:
            categories = ['general']

        return categories

    def _assign_temporal_stage(self, unit_type: str) -> int:
        """Assign temporal stage (1-5) based on unit type"""
        stage_mapping = {
            'background_context': 1,
            'background_prior_work': 1,
            'thesis_question': 2,
            'thesis_claim': 2,
            'method_approach': 3,
            'method_detail': 3,
            'result_main': 4,
            'result_visual': 4,
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

        # Create some realistic cross-references
        for i in range(min(len(units) // 3, 15)):
            if len(units) > 1:
                from_unit = self.random.choice(units)
                to_unit = self.random.choice([u for u in units if u.id != from_unit.id])

                cross_ref = CrossReference(
                    from_unit=from_unit.id,
                    to_unit=to_unit.id,
                    relation_type=self.random.choice(relation_types),
                    strength=self.random.uniform(0.6, 0.95)
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
        cluster_id = cluster['id']
        stage = cluster['temporal_stage']

        # Generate thesis connection based on stage
        thesis_connections = {
            1: f"Establishes the context and motivation for the research question: '{thesis['research_question']}'",
            2: f"Directly addresses the core thesis by proposing: '{thesis['main_claim']}'",
            3: f"Implements the methodology to validate the claim: '{thesis['main_claim']}'",
            4: f"Provides empirical evidence supporting the thesis through experimental results",
            5: f"Synthesizes findings and discusses implications of the validated thesis"
        }

        key_insights = {
            1: "Motor thermal degradation is a critical but understudied issue in long-term robot operations",
            2: "Thermal-aware control can proactively prevent motor failures rather than reactively responding",
            3: "Real-time thermal estimation and predictive control framework enables proactive thermal management",
            4: "Thermal-aware policies demonstrate significant reduction in motor limitations and improved endurance",
            5: "Integrated thermal management framework successfully bridges sim-to-real gap for practical deployment"
        }

        descriptions = {
            1: "Background on motor thermal issues and limitations of current approaches",
            2: "Problem formulation and proposed thermal-aware framework thesis",
            3: "Detailed methodology including thermal estimation, prediction, and control strategies",
            4: "Experimental results demonstrating performance improvements and validation",
            5: "Analysis of results, limitations, and future research directions"
        }

        return ClusterAnalysis(
            cluster_id=cluster_id,
            description=descriptions.get(stage, f"Cluster for stage {stage}"),
            key_insight=key_insights.get(stage, "Supporting insight for the thesis"),
            thesis_connection=thesis_connections.get(stage, "Supports the overall thesis"),
            importance_score=self.random.uniform(0.7, 0.95)
        )

    def _verify_consistency(
        self,
        thesis: Dict[str, Any],
        cluster_analyses: List[Dict[str, Any]],
        cross_references: List[Dict[str, Any]]
    ) -> ConsistencyCheck:
        """Verify consistency between thesis connections and thesis"""

        # Check thesis alignment
        alignment_scores = []
        for ca in cluster_analyses:
            # Simulate alignment scoring
            score = self.random.uniform(0.7, 0.95)
            alignment_scores.append(score)

        consistency_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

        # Flow analysis
        flow_analysis = {
            "temporal_flow": "Sequential progression from background through results",
            "logical_coherence": "Strong logical connections between stages",
            "narrative_strength": "Clear narrative arc supporting the thesis",
            "transition_quality": "Smooth transitions between clusters",
            "stage_completeness": {
                "background": "Complete",
                "problem": "Complete",
                "method": "Complete",
                "results": "Partial - needs more evaluation metrics",
                "discussion": "Limited - needs deeper analysis"
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
            "network_connectivity": "Moderately connected with clear directional flow"
        }

        # Identify issues
        issues = []
        if consistency_score < 0.8:
            issues.append("Some cluster analyses show weak thesis alignment")
        if len(cross_references) < len(cluster_analyses):
            issues.append("Low cross-reference density may indicate isolated clusters")
        if flow_analysis['stage_completeness'].get('results') == 'Partial':
            issues.append("Results section needs more comprehensive evaluation metrics")

        return ConsistencyCheck(
            consistency_score=consistency_score,
            flow_analysis=flow_analysis,
            relation_analysis=relation_analysis,
            issues_found=issues
        )

    def _identify_gaps(
        self,
        thesis: Dict[str, Any],
        cluster_analyses: List[Dict[str, Any]],
        consistency_results: Dict[str, Any]
    ) -> List[GapAnalysis]:
        """Identify gaps in the research presentation"""
        gaps = []

        # Check for methodological gaps
        method_clusters = [ca for ca in cluster_analyses if ca['cluster_id'].endswith('3')]
        if method_clusters:
            gap = GapAnalysis(
                gap_type="methodological_detail",
                description="Thermal model parameter estimation process needs more detailed explanation",
                severity="medium",
                affected_sections=["Method & Approach"]
            )
            gaps.append(gap)

        # Check for evaluation gaps
        result_clusters = [ca for ca in cluster_analyses if ca['cluster_id'].endswith('4')]
        if result_clusters:
            gap = GapAnalysis(
                gap_type="evaluation_completeness",
                description="Missing comparison with baseline thermal management approaches",
                severity="high",
                affected_sections=["Results & Evaluation"]
            )
            gaps.append(gap)

        # Check consistency issues
        if consistency_results['consistency_score'] < 0.85:
            gap = GapAnalysis(
                gap_type="thesis_alignment",
                description="Some sections show weak connection to the main thesis",
                severity="medium",
                affected_sections=["Multiple sections"]
            )
            gaps.append(gap)

        # Check flow completeness
        stage_completeness = consistency_results['flow_analysis'].get('stage_completeness', {})
        for stage, status in stage_completeness.items():
            if status in ['Partial', 'Limited']:
                gap = GapAnalysis(
                    gap_type="content_completeness",
                    description=f"{stage.capitalize()} section is {status.lower()} and needs expansion",
                    severity="low" if status == "Partial" else "medium",
                    affected_sections=[stage.capitalize()]
                )
                gaps.append(gap)

        return gaps

    def _identify_quality_issues(
        self,
        cluster_analyses: List[Dict[str, Any]],
        consistency_results: Dict[str, Any]
    ) -> List[QualityIssue]:
        """Identify quality issues"""
        issues = []

        # Check cluster importance balance
        importance_scores = [ca['importance_score'] for ca in cluster_analyses]
        if max(importance_scores) - min(importance_scores) > 0.3:
            issue = QualityIssue(
                issue_type="unbalanced_importance",
                description="Large variance in cluster importance scores may indicate content imbalance",
                location="Global",
                severity="low"
            )
            issues.append(issue)

        # Check consistency issues
        for issue_desc in consistency_results.get('issues_found', []):
            issue = QualityIssue(
                issue_type="consistency",
                description=issue_desc,
                location="Multiple clusters",
                severity="medium"
            )
            issues.append(issue)

        # Check relation density
        rel_analysis = consistency_results.get('relation_analysis', {})
        if rel_analysis.get('cross_reference_density', 0) < 0.5:
            issue = QualityIssue(
                issue_type="low_connectivity",
                description="Low cross-reference density suggests weak inter-cluster connections",
                location="Cross-references",
                severity="medium"
            )
            issues.append(issue)

        return issues

    def _assess_section_conversion(self, cluster_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess feasibility of converting to paper sections"""

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
            # Score based on importance and completeness
            score = ca['importance_score'] * self.random.uniform(0.8, 0.95)
            conversion_scores[section_mapping.get(cluster_id, cluster_id)] = score

        return {
            "overall_feasibility": "High - clear structure aligns with standard paper format",
            "section_scores": conversion_scores,
            "recommended_order": list(section_mapping.values()),
            "missing_sections": ["Related Work - needs dedicated section"],
            "notes": "Strong temporal flow supports natural section conversion"
        }

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate distribution statistics"""
        category_dist = {}
        stage_dist = {}

        for unit in self.semantic_units:
            category_dist[unit.type] = category_dist.get(unit.type, 0) + 1
            stage_dist[unit.temporal_stage] = stage_dist.get(unit.temporal_stage, 0) + 1

        return {
            "category_distribution": category_dist,
            "stage_distribution": stage_dist
        }

    def get_sample_extractions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get sample semantic unit extractions"""
        samples = self.random.sample(
            self.semantic_units,
            min(n, len(self.semantic_units))
        )
        return [asdict(s) for s in samples]

    def estimate_costs(
        self,
        step1: Dict[str, Any],
        step2: Dict[str, Any],
        step3: Dict[str, Any],
        step4: Dict[str, Any]
    ) -> CostEstimate:
        """Estimate token usage and costs"""

        num_slides = len(self.samples)
        num_units = len(self.semantic_units)
        num_clusters = len(step1['clusters'])

        # Step 1: Flash - Thesis + classification
        # Input: All slides content + images
        step1_input = num_slides * 500  # ~500 tokens per slide
        step1_output = num_units * 150  # ~150 tokens per unit + thesis

        # Step 2: Flash - Cluster analysis (parallel)
        # Input: thesis + cluster units
        step2_input = num_clusters * 800  # ~800 tokens per cluster context
        step2_output = num_clusters * 200  # ~200 tokens per analysis

        # Step 3: Flash - Consistency check
        # Input: thesis + all cluster analyses + cross-refs
        step3_input = 3000 + len(step2['cluster_analyses']) * 200
        step3_output = 1000  # Consolidated output

        # Total Flash usage
        flash_input = step1_input + step2_input + step3_input
        flash_output = step1_output + step2_output + step3_output

        # Step 4: Pro - Quality verification
        # Input: All previous results
        pro_input = 5000 + len(step2['cluster_analyses']) * 300
        pro_output = 2000  # Gap analysis + quality issues

        # Calculate costs
        flash_cost = (flash_input / 1_000_000 * FLASH_INPUT_COST +
                     flash_output / 1_000_000 * FLASH_OUTPUT_COST)
        pro_cost = (pro_input / 1_000_000 * PRO_INPUT_COST +
                   pro_output / 1_000_000 * PRO_OUTPUT_COST)

        return CostEstimate(
            flash_input_tokens=flash_input,
            flash_output_tokens=flash_output,
            pro_input_tokens=pro_input,
            pro_output_tokens=pro_output,
            flash_cost=flash_cost,
            pro_cost=pro_cost,
            total_cost=flash_cost + pro_cost
        )


def main():
    """Main execution"""
    # Load samples
    samples_path = "/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-final/samples-extended.json"
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Run simulation
    simulator = StandardDistributedSimulator(samples)
    results = simulator.simulate()

    # Save results
    output_path = "/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp5-standard-distributed.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Condition: {results['condition']}")
    print(f"Strategy: {results['strategy']} ({results['pattern']})")
    print(f"Total Slides: {results['total_slides']}")
    print(f"Total Units: {results['total_units']}")
    print(f"Avg Units/Slide: {results['avg_units_per_slide']:.2f}")
    print(f"\nStage 1 (Flash): {len(results['step1_results']['units'])} units, "
          f"{len(results['step1_results']['clusters'])} clusters")
    print(f"Stage 2 (Flash): {len(results['step2_results']['cluster_analyses'])} cluster analyses")
    print(f"Stage 3 (Flash): Consistency score = {results['step3_results']['consistency_score']:.2f}")
    print(f"Stage 4 (Pro): {len(results['step4_results']['gap_analysis'])} gaps, "
          f"{len(results['step4_results']['quality_issues'])} quality issues")
    print(f"\nEstimated Cost: ${results['cost_estimate']['total_cost']:.4f}")
    print(f"  Flash: ${results['cost_estimate']['flash_cost']:.4f}")
    print(f"  Pro: ${results['cost_estimate']['pro_cost']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
