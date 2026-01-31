#!/usr/bin/env python3
"""
실험 5 분석 시각화 및 상세 리포트 생성
"""

import json
from typing import Dict, Any
import textwrap

# 결과 로드
with open('/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp5-quickdraft-direct.json', 'r') as f:
    result = json.load(f)

# ============================================================================
# 1. 상세 분석 리포트
# ============================================================================

print("=" * 80)
print("EXPERIMENT 5: QUICK DRAFT + DIRECT STRATEGY")
print("=" * 80)

print("\n[실험 조건]")
print(f"  조건: {result['condition']}")
print(f"  전략: {result['strategy'].upper()}")
print(f"  해상도: {result['resolution']}")
print(f"  실행시각: {result['timestamp']}")

print("\n[데이터 규모]")
print(f"  총 슬라이드: {result['total_slides']:,}개")
print(f"  추출 의미 단위: {result['total_units']:,}개")
print(f"  슬라이드당 평균: {result['avg_units_per_slide']:.2f}개")

print("\n[Flash 분석 결과]")
flash = result['flash_analysis']
print(f"  추출 방식: {flash['extraction_method']}")
print(f"  이미지 분석 깊이: {flash['image_analysis_depth']}")
print(f"  분류 방식: {flash['classification_method']}")
print(f"  총 추출 단위: {flash['total_units_extracted']:,}개")

print("\n[Pro 검증 결과]")
pro = result['pro_gap_validation']
print(f"  식별된 간극: {pro['gaps_count']:,}개")
print(f"  커버리지: {pro['coverage_rate']:.1%}")
print(f"  Thesis 일관성: {pro['thesis_consistency']:.2f}/1.0")
print(f"  평균 품질 점수: {pro['quality_assessment']['average_unit_quality']:.2f}/1.0")
print(f"  평가된 단위: {pro['quality_assessment']['total_units_assessed']:,}개")

# ============================================================================
# 2. 카테고리별 상세 분석
# ============================================================================

print("\n" + "=" * 80)
print("카테고리별 분포 분석")
print("=" * 80)

print("\n[의미 단위 타입별 분포]")
by_type = result['category_distribution']['by_type']
total = sum(by_type.values())
for unit_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
    pct = (count / total) * 100
    bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
    print(f"  {unit_type:15} │ {bar} │ {count:2}개 ({pct:5.1f}%)")

print("\n[태그별 분포]")
by_tags = result['category_distribution']['by_tags']
for tag, count in sorted(by_tags.items(), key=lambda x: x[1], reverse=True):
    pct = (count / total) * 100
    bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
    print(f"  {tag:15} │ {bar} │ {count:2}개 ({pct:5.1f}%)")

print("\n[우선순위별 분포]")
by_priority = result['category_distribution']['by_priority']
for priority in sorted(map(int, by_priority.keys())):
    count = by_priority[str(priority)]
    pct = (count / total) * 100
    bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
    print(f"  Level {priority} │ {bar} │ {count:2}개 ({pct:5.1f}%)")

# ============================================================================
# 3. Thesis 분석
# ============================================================================

print("\n" + "=" * 80)
print("Thesis 분석")
print("=" * 80)

thesis = result['thesis']
print("\n[주요 테마]")
for i, theme in enumerate(thesis['main_themes'], 1):
    print(f"  {i}. {theme}")

print("\n[포커스 영역]")
for i, area in enumerate(thesis['focus_areas'], 1):
    print(f"  {i}. {area}")

print("\n[구성 요소]")
print(f"  방법론 포함: {'✓' if thesis['methodology_present'] else '✗'}")
print(f"  결과 포함: {'✓' if thesis['results_present'] else '✗'}")
print(f"  문제 언급: {'✓' if thesis['problems_addressed'] else '✗'}")

print("\n[주요 기여도]")
for i, contribution in enumerate(thesis['key_contributions'], 1):
    # 첫 60자만 표시
    contrib_short = contribution[:60].replace('\n', ' ')
    print(f"  {i}. {contrib_short}...")

# ============================================================================
# 4. 시간적 분포 분석
# ============================================================================

print("\n" + "=" * 80)
print("시간적 분포 (10분위 분석)")
print("=" * 80)

temporal = result['category_distribution']['temporal_distribution']
total_units_temporal = sum(temporal)

for i, count in enumerate(temporal):
    pct = (count / total_units_temporal) * 100 if total_units_temporal > 0 else 0
    start_pct = (i * 10)
    end_pct = ((i + 1) * 10)
    bar = '█' * count + '░' * (10 - count)
    print(f"  {start_pct:2d}%-{end_pct:2d}% │ {bar} │ {count}개 ({pct:5.1f}%)")

# ============================================================================
# 5. 비용-효과 분석
# ============================================================================

print("\n" + "=" * 80)
print("비용 및 효율성 분석")
print("=" * 80)

cost = result['cost_estimate']
print("\n[API 사용 비용]")
print(f"  Flash 분석:")
print(f"    입력 토큰: {cost['flash_analysis']['input_tokens']:,}")
print(f"    출력 토큰: {cost['flash_analysis']['output_tokens']:,}")
print(f"    비용: ${cost['flash_analysis']['cost_usd']:.4f}")

print(f"\n  Pro 검증:")
print(f"    입력 토큰: {cost['pro_validation']['input_tokens']:,}")
print(f"    출력 토큰: {cost['pro_validation']['output_tokens']:,}")
print(f"    비용: ${cost['pro_validation']['cost_usd']:.4f}")

print(f"\n  합계: ${cost['total_cost_usd']:.4f}")

print("\n[효율성 지표]")
metrics = result['metrics']
print(f"  추출 속도: {metrics['extraction_speed']}")
print(f"  검증 깊이: {metrics['validation_depth']}")
print(f"  전체 효율: {metrics['overall_efficiency']}")
print(f"  품질-속도 균형: {metrics['quality_vs_speed']}")

# ============================================================================
# 6. 샘플 추출 사례
# ============================================================================

print("\n" + "=" * 80)
print("샘플 추출 사례")
print("=" * 80)

samples = result['sample_extractions']
for i, sample in enumerate(samples, 1):
    print(f"\n[샘플 {i}]")
    print(f"  슬라이드: {sample['slide_idx']}")
    print(f"  타입: {sample['type']}")
    print(f"  우선순위: {sample['priority']}/9")
    print(f"  키워드: {', '.join(sample['keywords'])}")
    print(f"  태그: {', '.join(sample['tags'])}")
    # 내용을 여러 줄로 표시
    content = textwrap.fill(sample['content_snippet'], width=70, initial_indent='  ', subsequent_indent='  ')
    print(f"  내용:\n{content}")

# ============================================================================
# 7. Direct 전략 평가
# ============================================================================

print("\n" + "=" * 80)
print("Direct 전략 평가")
print("=" * 80)

print("\n[전략 특성]")
print("  ✓ Flash: 빠른 전체 분석")
print("  ✓ Pro: 품질 검증")
print("  ✓ 비용 효율적")
print("  ✓ 높은 커버리지")

print("\n[성과]")
coverage = pro['coverage_rate']
quality = pro['quality_assessment']['average_unit_quality']
cost_per_unit = cost['total_cost_usd'] / result['total_units']

print(f"  커버리지: {coverage:.1%} ({'완벽' if coverage >= 0.95 else '우수' if coverage >= 0.85 else '양호'})")
print(f"  단위 품질: {quality:.2f}/1.0 ({'우수' if quality >= 0.9 else '양호' if quality >= 0.8 else '개선필요'})")
print(f"  단가: ${cost_per_unit:.6f}/단위")
print(f"  Thesis 일관성: {pro['thesis_consistency']:.2f} ({'우수' if pro['thesis_consistency'] >= 0.7 else '양호' if pro['thesis_consistency'] >= 0.4 else '개선필요'})")

print("\n[권장 활용]")
print("  1. 대규모 슬라이드 배치 분석")
print("  2. 예산 제약이 있는 프로젝트")
print("  3. 초기 탐사 및 범위 파악")
print("  4. Pro 상세 분석 전 사전 검증")

# ============================================================================
# 8. 최종 요약
# ============================================================================

print("\n" + "=" * 80)
print("최종 요약")
print("=" * 80)

print(f"""
실험: Quick Draft + Direct 전략
결과 파일: exp5-quickdraft-direct.json

주요 성과:
  - {result['total_units']}개 의미 단위 추출 (목표: 45-90개)
  - {pro['coverage_rate']:.0%} 슬라이드 커버리지
  - {pro['quality_assessment']['average_unit_quality']:.2f}/1.0 평균 품질
  - ${cost['total_cost_usd']:.4f} 소비 비용

다음 단계:
  1. 다른 전략(Incremental, Hierarchical)과 비교
  2. 상이한 해상도(Medium, High)로 재실험
  3. 다양한 샘플 세트로 검증
""")

print("=" * 80)
print("분석 완료")
print("=" * 80)
