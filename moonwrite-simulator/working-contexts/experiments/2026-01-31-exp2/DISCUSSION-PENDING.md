# 보류 중인 논의: Standard vs Premium 가격 차별화

> **날짜**: 2026-01-31
> **상태**: 다음 세션에서 계속
> **관련 파일**: exp5-final-report.md

---

## 문제 제기

최적화 후 Standard와 Premium의 비용 차이가 거의 없음:

| 옵션 | 해상도 | 모델 배분 | 비용 | 차이 |
|-----|--------|----------|------|------|
| Standard | Medium (1-3) | Flash 1-3 + Pro 4 | $0.055 | 기준 |
| Premium | High (1-5) | Flash 1-3 + Pro 4 | $0.07 | +27% |

**차이: 단 $0.015 (27%)**

---

## 원인 분석

```
Standard:  Flash $0.007 + Pro $0.048 = $0.055
Premium:   Flash $0.010 + Pro $0.060 = $0.070
                          ↑
              Pro 비용이 85% 지배적
```

둘 다 같은 모델 배분 패턴을 사용하므로:
- Flash 1-3단계: 추출, 분류, 일관성 검증
- Pro 4단계만: 최종 품질 검증

Pro가 비용의 85%를 차지 → 비용 구조가 유사해짐

---

## 검토할 옵션

### 옵션 1: Premium 차별화 강화
- Pro를 더 많이 사용 (2-4단계 모두 Pro)
- 장점: 명확한 가격/품질 차이
- 단점: 비용 증가

### 옵션 2: 가격 통합
- Standard/Premium을 하나로 통합
- 해상도만 선택 가능하게
- 장점: 단순화
- 단점: 프리미엄 수익 기회 상실

### 옵션 3: 해상도로만 차별화
- 모델 배분은 동일
- Low/Medium/High 해상도만 선택
- 가격 = 해상도에 비례

### 옵션 4: Premium에 추가 기능
- 동일 모델 배분 유지
- Premium에만 추가 기능 제공:
  - 역개요 검증
  - Writing Principles 상세 평가
  - 구체적 플레이스홀더 제안

---

## 현재 실험 결과 요약

### Quick Draft (Direct)
- 비용: $0.06
- 품질: 0.96
- 용도: 빠른 스크리닝

### Standard (Distributed)
- 비용: $0.055
- 품질: 0.63, 일관성 0.55
- 용도: 일반 논문 작성

### Premium (Distributed 최적화)
- 비용: $0.07 (또는 $0.0008 - 시뮬레이션 값)
- 품질: 0.77, 일관성 0.89, 정렬률 96%
- 용도: 고품질 제출물

---

## 결정 완료 (2026-01-31 exp6 실험 기반)

1. [x] Standard와 Premium의 가격 차별화 전략 확정 → **옵션 4: 기능 차별화**
2. [x] 모델 배분 패턴 재검토 → **동일 유지** (Flash 1-3 + Pro 4)
3. [x] 옵션별 기능 차별화 명확화 → 역개요 검증, Writing Principles 6개, Self-Critique
4. [x] 최종 가격 정책 수립 → Standard 1x, Premium 1.5-1.7x

### 실험 결과 요약 (exp6-comparison-report.md)

| 옵션 | 해상도 | Quality | Consistency | Alignment | Cost |
|------|--------|---------|-------------|-----------|------|
| Standard | Medium | 0.76 | 0.91 | - | $0.0006 |
| Premium | Medium | 0.75 | 0.92 | **100%** | $0.0009 |

**핵심 발견**:
- Medium 해상도에서 Premium이 100% Thesis 정렬 달성
- 비용 차이 50-67%로 합리적
- 모델 배분 동일 유지로 기능 차별화 검증됨

---

## 참고 파일

- `exp5-final-report.md` - 전체 실험 결과
- `exp5-premium-distributed-optimized.json` - Premium 최적화 결과
- `exp5-standard-distributed.json` - Standard 결과
- `clustering-strategy.md` - Distributed 전략 정의
- `final-report-v2.md` - 이전 실험 결과 형식 참고
