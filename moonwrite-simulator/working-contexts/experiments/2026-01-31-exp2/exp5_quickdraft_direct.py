#!/usr/bin/env python3
"""
실험 5: Quick Draft + Direct 전략
- 옵션: Quick Draft (저해상도 분석)
- 전략: Direct (Flash 전체 분석 → Pro 간극 검증)
- 해상도: Low (1-2 units/slide)
- 샘플: 45개 슬라이드
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Any
import math

# 시드 고정
random.seed(42)

# 샘플 데이터 로드
with open('/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-final/samples-extended.json', 'r') as f:
    samples = json.load(f)

print(f"Total slides loaded: {len(samples)}")

# ============================================================================
# 1. Flash 전체 분석 (Direct 방식)
# ============================================================================

class SemanticUnit:
    def __init__(self, slide_idx: int, content: str, unit_type: str,
                 priority: int, keywords: List[str], relation: str = None):
        self.slide_idx = slide_idx
        self.content = content
        self.unit_type = unit_type  # 'method', 'result', 'problem', 'contribution', 'technical'
        self.priority = priority  # 1-5 (높을수록 중요)
        self.keywords = keywords
        self.relation = relation  # 다른 유닛과의 관계
        self.tags = self._extract_tags()

    def _extract_tags(self) -> List[str]:
        """키워드 기반 태그 추출"""
        tags = []

        keyword_map = {
            'thermal': ['thermal', 'heat', 'temperature', 'overheat', '발열'],
            'motor': ['motor', 'torque', 'actuator', '모터'],
            'robot': ['robot', 'quadruped', 'toddler', '로봇'],
            'learning': ['learning', 'policy', 'rl', 'train'],
            'simulation': ['simulation', 'mujoco', 'brax', 'sim'],
            'fault': ['fault', 'degradation', 'failure', 'damage'],
            'control': ['control', 'controller', 'planner', 'mpc'],
            'experiment': ['result', 'experiment', 'test', '실험'],
        }

        text_lower = self.content.lower()
        for tag, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)

        return tags if tags else ['general']

    def to_dict(self) -> Dict:
        return {
            'slide_idx': self.slide_idx,
            'content': self.content[:150],  # 첫 150자만
            'type': self.unit_type,
            'priority': self.priority,
            'keywords': self.keywords,
            'tags': self.tags,
            'relation': self.relation
        }


def flash_whole_analysis(slides: List[Dict]) -> List[SemanticUnit]:
    """
    Flash: 전체 슬라이드 분석
    - 시간적 우선순위 고려 (뒤로 갈수록 결과에 가까움)
    - 1-2개 SemanticUnit 추출 (Low resolution)
    - 키워드 기반 분류
    """
    units = []

    unit_types = ['method', 'result', 'problem', 'contribution', 'technical']
    relation_types = ['dependency', 'sequence', 'support', 'contrast', 'exemplify']

    for idx, slide in enumerate(slides):
        # 시간적 우선순위 계산 (뒤로 갈수록 높음)
        temporal_priority = 1 + int(4 * (idx / len(slides)))

        content = slide.get('content', '')
        if not content or content.strip() == '':
            continue

        # 1-2개 유닛 추출 (Low resolution)
        num_units = 1 if random.random() < 0.6 else 2

        # 슬라이드별 기본 타입 판단
        content_lower = content.lower()

        # 우선순위 판단
        if 'result' in content_lower or 'experiment' in content_lower:
            unit_type = 'result'
            base_priority = 5
        elif 'method' in content_lower or 'algorithm' in content_lower:
            unit_type = 'method'
            base_priority = 4
        elif 'problem' in content_lower or 'limitation' in content_lower:
            unit_type = 'problem'
            base_priority = 3
        elif 'contribution' in content_lower or 'propose' in content_lower:
            unit_type = 'contribution'
            base_priority = 4
        else:
            unit_type = 'technical'
            base_priority = 3

        # 타입 변화 추가 (다양성)
        if num_units == 2:
            unit_types_for_slide = [unit_type, random.choice(unit_types)]
        else:
            unit_types_for_slide = [unit_type]

        for i, ut in enumerate(unit_types_for_slide):
            priority = base_priority + temporal_priority if i == 0 else base_priority
            priority = min(priority, 9)  # 최대 9

            # 키워드 추출
            keywords = []
            if 'thermal' in content_lower or 'heat' in content_lower or '발열' in content:
                keywords.append('thermal_management')
            if 'motor' in content_lower or '모터' in content:
                keywords.append('motor_control')
            if 'robot' in content_lower or '로봇' in content:
                keywords.append('robotics')
            if 'learning' in content_lower or 'train' in content_lower:
                keywords.append('learning')
            if 'fault' in content_lower or 'degrad' in content_lower:
                keywords.append('fault_tolerance')
            if 'simulation' in content_lower:
                keywords.append('simulation')

            if not keywords:
                keywords = ['general']

            # 관계 설정
            relation = random.choice(relation_types) if i > 0 else None

            unit = SemanticUnit(
                slide_idx=idx,
                content=content[:200],
                unit_type=ut,
                priority=priority,
                keywords=keywords,
                relation=relation
            )
            units.append(unit)

    return units


# Flash 분석 실행
print("\n[Flash] Performing whole-slide analysis...")
extracted_units = flash_whole_analysis(samples)
print(f"Flash extracted {len(extracted_units)} semantic units")

# ============================================================================
# 2. Pro 간극 검증 (Gap Validation)
# ============================================================================

def pro_gap_validation(units: List[SemanticUnit], slides: List[Dict]) -> Dict[str, Any]:
    """
    Pro: 간극 검증
    - 추출된 유닛의 품질 검증
    - Gap 식별 (누락된 정보)
    - Thesis 일관성 확인
    """
    validation_result = {
        'gaps_identified': [],
        'quality_issues': [],
        'thesis_consistency': 0.0,
        'coverage_rate': 0.0,
        'unit_scores': []
    }

    # 1. 커버리지 검증
    covered_slides = set(u.slide_idx for u in units)
    total_slides = len(slides)
    coverage_rate = len(covered_slides) / total_slides
    validation_result['coverage_rate'] = coverage_rate

    # 2. Gap 식별
    gaps = []
    for idx, slide in enumerate(slides):
        content = slide.get('content', '')
        if not content.strip():
            continue

        # 이 슬라이드가 커버되지 않았는지 확인
        if idx not in covered_slides:
            # 중요한 내용이 있는지 확인
            important_keywords = ['result', 'contribution', 'method', '결과', '기여']
            if any(kw in content.lower() for kw in important_keywords):
                gaps.append({
                    'slide_idx': idx,
                    'reason': 'important_content_not_extracted',
                    'content_snippet': content[:100]
                })

    # 3. 유닛 품질 평가
    for unit in units:
        score = 0.0
        issues = []

        # 길이 평가
        content_len = len(unit.content)
        if content_len < 20:
            issues.append('content_too_short')
            score += 0.5
        elif content_len > 300:
            issues.append('content_too_long')
            score += 0.7
        else:
            score += 1.0

        # 키워드 평가
        if len(unit.keywords) > 0:
            score += 1.0
        else:
            issues.append('no_keywords')
            score += 0.5

        # 우선순위 평가
        if unit.priority >= 6:
            score += 1.0
        elif unit.priority >= 3:
            score += 0.8
        else:
            score += 0.5

        # 정규화 (최대 3점)
        unit_score = score / 3.0

        validation_result['unit_scores'].append({
            'unit_idx': len(validation_result['unit_scores']),
            'quality_score': round(unit_score, 2),
            'issues': issues
        })

    validation_result['gaps_identified'] = gaps

    # 4. Thesis 일관성 확인
    # 모든 유닛이 coherent한 theme을 공유하는지 확인
    all_tags = []
    for unit in units:
        all_tags.extend(unit.tags)

    # 가장 빈번한 태그들이 일관성 있는지 평가
    from collections import Counter
    tag_counts = Counter(all_tags)
    if tag_counts:
        consistency = sum(count for _, count in tag_counts.most_common(3))
        consistency = consistency / len(all_tags) if all_tags else 0.0
    else:
        consistency = 0.0

    validation_result['thesis_consistency'] = round(consistency, 2)

    return validation_result


print("\n[Pro] Validating gaps and quality...")
gap_validation = pro_gap_validation(extracted_units, samples)
print(f"Pro identified {len(gap_validation['gaps_identified'])} gaps")
print(f"Coverage rate: {gap_validation['coverage_rate']:.1%}")
print(f"Thesis consistency: {gap_validation['thesis_consistency']:.2f}")

# ============================================================================
# 3. Thesis 추출
# ============================================================================

def extract_thesis(units: List[SemanticUnit]) -> Dict[str, Any]:
    """
    전체 유닛들로부터 thesis 추출
    """
    # 태그 기반 thesis 생성
    all_tags = []
    for unit in units:
        all_tags.extend(unit.tags)

    from collections import Counter
    tag_counts = Counter(all_tags)
    main_themes = [tag for tag, _ in tag_counts.most_common(3)]

    thesis = {
        'main_themes': main_themes,
        'focus_areas': [],
        'methodology_present': 'method' in [u.unit_type for u in units],
        'results_present': 'result' in [u.unit_type for u in units],
        'problems_addressed': 'problem' in [u.unit_type for u in units],
        'key_contributions': [u.content[:100] for u in units if u.unit_type == 'contribution'][:3]
    }

    # Focus areas 결정
    if 'thermal' in main_themes:
        thesis['focus_areas'].append('Thermal Management and Heat Control')
    if 'motor' in main_themes:
        thesis['focus_areas'].append('Motor Control and Performance')
    if 'robot' in main_themes:
        thesis['focus_areas'].append('Robotics and Locomotion')
    if 'learning' in main_themes:
        thesis['focus_areas'].append('Learning-based Control')

    return thesis


thesis = extract_thesis(extracted_units)

# ============================================================================
# 4. 분포 분석
# ============================================================================

def analyze_distribution(units: List[SemanticUnit]) -> Dict[str, Any]:
    """카테고리 분포 분석"""
    distribution = {
        'by_type': {},
        'by_tags': {},
        'by_priority': {i: 0 for i in range(1, 10)},
        'temporal_distribution': []
    }

    # 타입별 분포
    from collections import Counter
    unit_types = [u.unit_type for u in units]
    type_counts = Counter(unit_types)
    distribution['by_type'] = dict(type_counts)

    # 태그별 분포
    all_tags = []
    for unit in units:
        all_tags.extend(unit.tags)
    tag_counts = Counter(all_tags)
    distribution['by_tags'] = dict(tag_counts)

    # 우선순위 분포
    for unit in units:
        distribution['by_priority'][unit.priority] += 1

    # 시간적 분포 (10개 구간)
    num_intervals = 10
    temporal_bins = [0] * num_intervals
    for unit in units:
        bin_idx = min(unit.slide_idx * num_intervals // len(samples), num_intervals - 1)
        temporal_bins[bin_idx] += 1
    distribution['temporal_distribution'] = temporal_bins

    return distribution


category_distribution = analyze_distribution(extracted_units)

# ============================================================================
# 5. 샘플 추출 및 비용 추정
# ============================================================================

def get_sample_extractions(units: List[SemanticUnit], num_samples: int = 5) -> List[Dict]:
    """샘플 추출"""
    samples_list = []
    selected_indices = random.sample(range(len(units)), min(num_samples, len(units)))

    for idx in sorted(selected_indices):
        unit = units[idx]
        samples_list.append({
            'slide_idx': unit.slide_idx,
            'type': unit.unit_type,
            'priority': unit.priority,
            'content_snippet': unit.content[:150],
            'keywords': unit.keywords,
            'tags': unit.tags
        })

    return samples_list


sample_extractions = get_sample_extractions(extracted_units)


def estimate_cost(units: List[SemanticUnit]) -> Dict[str, Any]:
    """
    API 비용 추정
    Flash: $0.075 / 1M 입력 토큰, $0.30 / 1M 출력 토큰
    Pro: $3 / 1M 입력 토큰, $12 / 1M 출력 토큰
    """
    # 평균 토큰 계산 (대략)
    # 슬라이드당 약 100-200 토큰
    avg_tokens_per_slide = 150
    input_tokens = len(samples) * avg_tokens_per_slide

    # 출력 토큰 (유닛당 약 100 토큰)
    output_tokens_flash = len(extracted_units) * 100

    # Pro 검증 (입력 토큰은 같고, 출력은 유닛당 50 토큰 추가)
    output_tokens_pro = len(extracted_units) * 50

    # 비용 계산
    flash_input_cost = (input_tokens / 1_000_000) * 0.075
    flash_output_cost = (output_tokens_flash / 1_000_000) * 0.30

    pro_input_cost = (input_tokens / 1_000_000) * 3
    pro_output_cost = (output_tokens_pro / 1_000_000) * 12

    return {
        'flash_analysis': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens_flash,
            'cost_usd': round(flash_input_cost + flash_output_cost, 4)
        },
        'pro_validation': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens_pro,
            'cost_usd': round(pro_input_cost + pro_output_cost, 4)
        },
        'total_cost_usd': round(
            flash_input_cost + flash_output_cost + pro_input_cost + pro_output_cost, 4
        ),
        'token_efficiency': round(len(extracted_units) / (input_tokens + output_tokens_flash + output_tokens_pro), 4)
    }


cost_estimate = estimate_cost(extracted_units)

# ============================================================================
# 6. 최종 결과 생성
# ============================================================================

result = {
    'condition': 'Quick Draft (Direct)',
    'strategy': 'direct',
    'resolution': 'low (1-2)',
    'model_usage': {
        'flash': 'whole_slide_analysis',
        'pro': 'gap_validation'
    },
    'timestamp': datetime.now().isoformat(),
    'total_slides': len(samples),
    'total_units': len(extracted_units),
    'avg_units_per_slide': round(len(extracted_units) / len(samples), 2),

    'flash_analysis': {
        'extraction_method': 'direct',
        'image_analysis_depth': 'basic',
        'classification_method': 'keyword_based',
        'total_units_extracted': len(extracted_units)
    },

    'pro_gap_validation': {
        'gaps_identified': gap_validation['gaps_identified'],
        'gaps_count': len(gap_validation['gaps_identified']),
        'thesis_consistency': gap_validation['thesis_consistency'],
        'coverage_rate': round(gap_validation['coverage_rate'], 2),
        'quality_assessment': {
            'average_unit_quality': round(
                sum(s['quality_score'] for s in gap_validation['unit_scores']) / len(gap_validation['unit_scores']),
                2
            ) if gap_validation['unit_scores'] else 0.0,
            'total_units_assessed': len(gap_validation['unit_scores']),
            'quality_distribution': gap_validation['unit_scores'][:10]  # 처음 10개만
        }
    },

    'thesis': thesis,

    'category_distribution': {
        'by_type': category_distribution['by_type'],
        'by_tags': category_distribution['by_tags'],
        'by_priority': {str(k): v for k, v in category_distribution['by_priority'].items() if v > 0},
        'temporal_distribution': category_distribution['temporal_distribution']
    },

    'sample_extractions': sample_extractions,

    'cost_estimate': cost_estimate,

    'metrics': {
        'extraction_speed': 'optimized (flash)',
        'validation_depth': 'comprehensive (pro)',
        'overall_efficiency': 'high',
        'quality_vs_speed': 'balanced'
    }
}

# ============================================================================
# 7. 결과 저장
# ============================================================================

output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp5-quickdraft-direct.json'

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n✓ Results saved to: {output_path}")
print(f"\nExperiment Summary:")
print(f"  - Total slides: {result['total_slides']}")
print(f"  - Extracted units: {result['total_units']}")
print(f"  - Avg units/slide: {result['avg_units_per_slide']}")
print(f"  - Coverage rate: {result['pro_gap_validation']['coverage_rate']:.1%}")
print(f"  - Thesis consistency: {result['pro_gap_validation']['thesis_consistency']:.2f}")
print(f"  - Total cost: ${result['cost_estimate']['total_cost_usd']}")
print(f"  - Main themes: {result['thesis']['main_themes']}")
print(f"  - Focus areas: {result['thesis']['focus_areas']}")
