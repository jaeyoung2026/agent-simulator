# Moonwriter 최종 스펙

> **확정일**: 2026-01-31
> **버전**: 1.0
> **근거**: exp1-6 실험 결과 종합

---

## 1. 옵션 정의

### 1.1 3가지 논문 생성 옵션

| 옵션 | 전략 | 모델 배분 | 해상도 | 용도 |
|------|------|----------|--------|------|
| **Quick Draft** | Direct | Flash 전체 → Pro 검증 | **Low (1-2)** | 빠른 스크리닝 |
| **Standard** | Distributed | Flash 1-3 + Pro 4 | **Medium (1-3)** | 일반 논문 작성 |
| **Premium** | Distributed | Flash 1-3 + Pro 4 + 추가 기능 | **Medium (1-3)** | 고품질 제출물 |

> **해상도 고정**: 각 옵션에 최적화된 해상도가 고정되어 별도 선택 불가

---

## 2. 옵션별 상세 스펙

### 2.1 Quick Draft

```
전략: Direct (Flash 전체 분석 → Pro 간극 검증)
```

| 기능 | 스펙 |
|------|------|
| 의미 추출 | 키워드 기반 직접 추출 |
| Thesis 추출 | 자동 (키워드) |
| 분류 방식 | 키워드 100% |
| 이미지 분석 | 기본 (유형 분류) |
| Gap 분석 | 없음 |
| Writing Principles | 최소 |
| thesisConnection | 없음 |

**품질 지표** (실험 결과):
- 품질 점수: 0.96
- 비용: ~$0.06
- 용도: 빠른 구조 확인, 초안 스케치

### 2.2 Standard

```
전략: Distributed (Thesis-First 4단계)
모델: Flash (Step 1-3) + Pro (Step 4)
```

| 기능 | 스펙 |
|------|------|
| 의미 추출 | CoT 기반 추출 |
| Thesis 추출 | CoT (질문 + 주장) |
| 분류 방식 | 하이브리드 (키워드 48% + LLM 52%) |
| 이미지 분석 | 표준 (유형 + 역할) |
| Gap 분석 | 기본 (4개 영역) |
| Writing Principles | 3개 항목 |
| thesisConnection | 있음 |
| 역개요 검증 | **없음** |
| Self-Critique | **없음** |
| 플레이스홀더 | 위치 표시만 |

**Writing Principles (3개)**:
1. thesis_clarity
2. evidence_integration
3. logical_flow

**품질 지표** (실험 결과 - Medium 해상도):
- 품질 점수: 0.76
- 일관성: 0.91
- 비용: ~$0.0006
- Thesis 연결률: 49%

### 2.3 Premium

```
전략: Distributed (Thesis-First 4단계)
모델: Flash (Step 1-3) + Pro (Step 4) + 추가 Pro 호출
```

| 기능 | 스펙 |
|------|------|
| 의미 추출 | Self-Critique 포함 추출 |
| Thesis 추출 | 상세 (질문 + 주장 + 증거 + Sub-claims) |
| 분류 방식 | Thesis-First 4단계 |
| 이미지 분석 | 심층 (유형 + 역할 + 재현성 + 정량적 인사이트) |
| Gap 분석 | 상세 (5개 영역, 심각도 + 우선순위) |
| Writing Principles | **6개 항목** |
| thesisConnection | 있음 (연결 강도 포함) |
| 역개요 검증 | **있음** |
| Self-Critique | **있음** |
| 플레이스홀더 | **구체적 제안 (실험 제안 포함)** |

**Writing Principles (6개)**:
1. thesis_clarity
2. evidence_integration
3. logical_flow
4. technical_precision
5. reproducibility
6. contribution_clarity

**품질 지표** (실험 결과 - Medium 해상도):
- 품질 점수: 0.75
- 일관성: 0.92
- Thesis 정렬률: **100%**
- Writing Principles 평균: 0.80
- Self-Critique 평균: 0.82
- 비용: ~$0.0009

---

## 3. 기능 차별화 매트릭스

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Quick Draft    Standard       Premium            │
├─────────────────────────────────────────────────────────────────────┤
│ 해상도             Low (1-2)      Medium (1-3)   Medium (1-3)       │
│ 전략               Direct         Distributed    Distributed        │
│ 모델               Flash→Pro      Flash 1-3+Pro4 Flash 1-3+Pro4     │
├─────────────────────────────────────────────────────────────────────┤
│ Thesis 추출        자동           CoT            상세+Sub-claims    │
│ thesisConnection   ✗              ✓              ✓ (강도 포함)      │
│ 이미지 분석        유형만         유형+역할      심층+재현성        │
│ Gap 분석           ✗              기본 (4)       상세 (5+심각도)    │
│ Writing Principles 최소           3개            6개                │
│ 역개요 검증        ✗              ✗              ✓                  │
│ Self-Critique      ✗              ✗              ✓                  │
│ 플레이스홀더       ✗              위치만         구체적 제안        │
├─────────────────────────────────────────────────────────────────────┤
│ 비용 비율          0.5x           1x (기준)      1.5-1.7x           │
│ 용도               빠른 스크리닝  일반 논문      고품질 제출물      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 모델 배분 전략

### 4.1 Distributed 전략 (Standard/Premium)

```
Step 1: Flash (Sonnet) - Thesis 추출 + 의미 단위 분류
Step 2: Flash (Sonnet) - Thesis-Aware 클러스터 분석 (병렬)
Step 3: Flash (Sonnet) - 일관성 검증 + 흐름 통합
Step 4: Pro (Opus)     - 최종 품질 검증
```

| 단계 | 모델 | 역할 | 토큰 비율 |
|------|------|------|----------|
| 1-3 | Flash (Sonnet) | 추출, 분류, 검증 | ~85-90% |
| 4 | Pro (Opus) | 품질 검증 | ~10-15% |

### 4.2 Premium 추가 호출

| 추가 기능 | 모델 | Input | Output |
|----------|------|-------|--------|
| 역개요 검증 | Pro | 2,000 | 1,000 |
| Writing Principles 상세 | Pro | 1,500 | 800 |
| 상세 플레이스홀더 | Flash | 1,000 | 500 |

### 4.3 비용 효율

- 전체 Pro 사용 대비 **84-87% 비용 절감**
- Flash로 대부분 처리, Pro는 최종 검증만

---

## 5. 비용 구조

### 5.1 옵션별 비용 (45 슬라이드 기준)

| 옵션 | 해상도 (고정) | 비용 | 슬라이드당 |
|------|--------------|------|-----------|
| **Quick Draft** | Low (1-2) | ~$0.06 | ~$0.0013 |
| **Standard** | Medium (1-3) | ~$0.0006 | ~$0.00001 |
| **Premium** | Medium (1-3) | ~$0.0009 | ~$0.00002 |

### 5.2 가격 정책 권장

| 옵션 | 가격 비율 | 근거 |
|------|----------|------|
| Quick Draft | 0.5x | 빠른 처리, Low 해상도, 기본 기능 |
| Standard | 1x (기준) | Medium 해상도, 균형 잡힌 품질 |
| Premium | 1.5-1.7x | Medium 해상도, 추가 기능 가치 |

---

## 6. 시나리오별 권장

### 6.1 옵션 선택 가이드

| 시나리오 | 권장 옵션 | 근거 |
|---------|----------|------|
| 빠른 구조 확인 | **Quick Draft** | 최소 비용, Low 해상도, 빠른 처리 |
| **일반 논문 작성** | **Standard** | Medium 해상도, 균형 잡힌 품질-비용 |
| 중요 학회 제출 | **Premium** | Medium 해상도, 100% Thesis 정렬 |
| 저널 제출 | **Premium** | 역개요 검증, 재현성 분석 |
| 대량 스크리닝 | **Quick Draft** | Low 해상도, 비용 효율 |

### 6.2 기본 권장

> **Standard**가 기본 권장 옵션
>
> - Medium 해상도 (1-3 units/slide) 고정
> - Thesis-First 4단계로 구조화
> - Flash 85% + Pro 15%로 비용 최적화
> - 일관성 0.91, 품질 0.76
> - 논문 변환 가능한 thesisConnection

---

## 7. 파이프라인 흐름

### 7.1 Standard 파이프라인

```
입력: 슬라이드 데이터
    ↓
[Step 1] Flash: Thesis 추출 + CoT 의미 분류
    ↓
[Step 2] Flash: 하이브리드 클러스터링 (병렬)
    ↓
[Step 3] Flash: 일관성 검증 + 흐름 통합
    ↓
[Step 4] Pro: 기본 Gap 분석 + Writing Principles (3개)
    ↓
출력: 논문 초안 + thesisConnection + Gap 리포트
```

### 7.2 Premium 파이프라인

```
입력: 슬라이드 데이터
    ↓
[Step 1] Flash: Thesis 추출 + Self-Critique + 심층 이미지 분석
    ↓
[Step 2] Flash: Thesis-First 클러스터링 (병렬)
    ↓
[Step 3] Flash: 일관성 검증 + 역개요 생성
    ↓
[Step 4] Pro: Gap 분석 (심각도) + Writing Principles (6개)
    ↓
[Step 4+] Pro: 역개요 검증 + Writing Principles 상세 평가
    ↓
[Step 4++] Flash: 구체적 플레이스홀더 생성
    ↓
출력: 논문 초안 + thesisConnection + Gap 리포트
      + 역개요 + Self-Critique + 플레이스홀더 제안
```

---

## 8. 품질 지표 정의

### 8.1 공통 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| quality_score | 전체 품질 점수 | > 0.7 |
| consistency_score | 일관성 점수 | > 0.8 |
| gap_count | 발견된 Gap 수 | 낮을수록 좋음 |

### 8.2 Premium 전용 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| alignment_rate | Thesis 정렬률 | > 90% |
| self_critique_avg | Self-Critique 평균 | > 0.8 |
| writing_principles_avg | Writing Principles 평균 | > 0.8 |

---

## 9. 실험 결과 요약

### 9.1 exp6 최종 결과 (옵션별 고정 해상도)

| 옵션 | 해상도 (고정) | Quality | Consistency | Alignment | Cost |
|------|--------------|---------|-------------|-----------|------|
| **Quick Draft** | Low (1-2) | 0.83 | 1.00 | - | ~$0.06 |
| **Standard** | Medium (1-3) | 0.76 | 0.91 | - | $0.0006 |
| **Premium** | Medium (1-3) | 0.75 | 0.92 | **100%** | $0.0009 |

### 9.2 핵심 발견

1. **Medium 해상도에서 Premium 효과 극대화** - 100% Thesis 정렬
2. **모델 배분 동일 유지 효과적** - Flash 1-3 + Pro 4
3. **기능 차별화로 Premium 가치 창출** - 역개요, Writing Principles 6개
4. **해상도 고정으로 옵션 단순화** - 사용자 선택 부담 감소

---

## 10. 참고 문서

| 문서 | 위치 |
|------|------|
| 실험 1-2 결과 | `/working-contexts/experiments/2026-01-31-final/` |
| 실험 3-5 결과 | `/working-contexts/experiments/2026-01-31-exp2/` |
| 실험 6 결과 | `/working-contexts/experiments/2026-01-31-exp2/exp6-*.json` |
| 비교 리포트 | `/working-contexts/experiments/2026-01-31-exp2/exp6-comparison-report.md` |
| 클러스터링 전략 | `/reference/clustering-strategy.md` |
| Writing Principles | `/reference/writing-principles.md` |

---

> **요약**: Moonwriter는 Quick Draft (Low) / Standard (Medium) / Premium (Medium) 3가지 옵션을 제공하며, 각 옵션에 최적화된 해상도가 고정된다. Quick Draft는 빠른 스크리닝용, Standard는 일반 논문 작성의 기본 권장 옵션, Premium은 역개요 검증, Writing Principles 6개, Self-Critique 등 추가 기능으로 고품질 제출물을 지원한다.
