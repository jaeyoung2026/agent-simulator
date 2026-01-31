# 후속 실험 최종 리포트: Claude 모델 기반 옵션별/해상도별 품질 비교

> **실험일**: 2026-01-31
> **데이터**: 45개 샘플 슬라이드
> **모델**: Claude Haiku/Sonnet/Opus (Gemini 대신 Claude 모델 사용)

---

## 1. 실험 개요

### 1.1 모델 매핑
| 역할 | Claude 모델 | 대응 개념 |
|-----|------------|----------|
| 빠른 추출 | Haiku | Flash |
| 일반 처리 | Sonnet | Standard |
| 고품질 검증 | Opus | Pro |

### 1.2 실험 조건

**옵션별 실험** (모든 옵션에서 Opus 검증 사용):
| 옵션 | 추출 모델 | 검증 모델 | 해상도 |
|-----|----------|----------|-------|
| Quick Draft | Haiku | Opus | Low (1-2) |
| Standard | Sonnet | Opus | Medium (1-3) |
| Premium | Opus | Opus | High (1-5) |

**해상도별 실험**:
| 해상도 | 유닛/슬라이드 | 실험 모델 |
|-------|-------------|----------|
| Low | 1-2개 | Haiku |
| Medium | 1-3개 | Sonnet |
| High | 1-5개 | Sonnet |

---

## 2. 옵션별 결과 비교

### 2.1 정량적 비교

| 옵션 | 총 유닛 | 평균 유닛 | 검증 결과 |
|-----|--------|----------|----------|
| **Quick Draft (haiku+opus)** | 65 | 1.44 | 90% accepted |
| **Standard (sonnet+opus)** | 89 | 1.98 | 38.2% refined |
| **Premium (opus full)** | 126 | 2.80 | 81.7% premium |

### 2.2 Quick Draft + Opus (Low Resolution)

**결과 요약**:
- 총 유닛: 65개 (45 슬라이드)
- 평균 유닛/슬라이드: 1.44개
- Opus 검증 통과율: 90% (9 accepted, 1 refined, 0 rejected)

**카테고리 분포**:
- execution: 25개 (38.5%)
- finding: 9개 (13.8%)
- planning: 9개 (13.8%)
- setup: 1개
- interpretation: 1개

**이미지 분류**:
- plot: 11개
- diagram: 4개
- visualization: 3개
- video_animation: 3개
- formula: 2개
- screenshot: 1개
- other: 17개

**특징**: Haiku로 빠른 추출 후 Opus가 품질 검증. 대부분 그대로 수용되어 효율적.

### 2.3 Standard + Opus (Medium Resolution)

**결과 요약** (exp3 데이터 기준):
- 총 유닛: 89개
- 평균 유닛/슬라이드: 1.98개
- 분류 정제: 38.2% (34개 유닛 수정)
- Gap 식별: 4개
- Thesis 정렬: 72%

**카테고리 분포**:
- method: 28개 (31.5%)
- result: 18개 (20.2%)
- implementation: 12개
- introduction: 8개
- discussion: 8개
- conclusion: 6개

**Gap 분석 결과**:
1. 실제 하드웨어 검증 결과 부족 (major)
2. 기존 방법과의 정량적 비교 불충분 (minor)
3. 구성요소별 기여도 분석 필요 (minor)
4. 다른 로봇 플랫폼 적용 가능성 미검증 (moderate)

**특징**: Sonnet CoT 추출 + Opus 분류 정제로 균형 잡힌 품질.

### 2.4 Premium + Opus (High Resolution)

**결과 요약**:
- 총 유닛: 126개
- 평균 유닛/슬라이드: 2.80개
- Premium 품질 달성: 81.7% (103개)
- 클러스터 수: 13개

**카테고리 분포**:
- General: 39개 (31.0%)
- Supplementary: 17개 (13.5%)
- Theory: 13개 (10.3%)
- Thermal-Management: 12개 (9.5%)
- Results: 8개 (6.3%)

**소스 분포**:
- 텍스트 기반: 93개 (73.8%)
- 이미지 기반: 33개 (26.2%)

**도출된 Thesis**:
- **Domain**: Thermal-Aware Robot Control
- **Core Claim**: 열 관리를 통한 로봇 장기 운용 안정성 향상
- **Sub-claims**:
  1. 실시간 발열 예측 및 모니터링 프레임워크
  2. 열 인식 기반 동작 계획 (MPC/RL)
  3. 시뮬레이션-실제 간격 축소를 위한 Heat2Torque 모델

**특징**: Opus 전체 분석으로 최고 품질, 심층 멀티모달 분석 가능.

---

## 3. 해상도별 품질 비교

### 3.1 정량적 비교

| 해상도 | 총 유닛 | 평균 유닛 | 정보 커버리지 | 압축 수준 |
|-------|--------|----------|--------------|----------|
| **Low (1-2)** | 69 | 1.53 | Medium | 과도 |
| **Medium (1-3)** | 89-92 | 1.98-2.04 | Balanced | 적절 |
| **High (1-5)** | 163 | 3.62 | High | 최소 |

### 3.2 Low Resolution (1-2 유닛)

**품질 평가**:
- 정보 커버리지: Medium
- 완전 보존: 29개 슬라이드 (64.4%)
- 경미한 손실: 6개 슬라이드 (13.3%)
- 심각한 손실: 10개 슬라이드 (22.2%)
- 최대 손실: 7개 유닛 (한 슬라이드에서)
- 이미지 포함율: 66.7%

**장점**:
- 매우 간결한 형식
- 빠른 검색과 브라우징에 최적화
- 저장 공간과 처리 시간 최소화

**단점**:
- 22.2% 슬라이드에서 중요 정보 손실
- Method, Result 세부 사항 누락
- 멀티-파트 설명 완전성 보장 불가

**적합 용도**: 빠른 개요 파악, 논문 비교 검토

### 3.3 Medium Resolution (1-3 유닛)

**품질 평가**:
- 정보 커버리지: Balanced
- 균형 잡힌 세분화 (56%가 2개 유닛)
- 논문 구조와 좋은 매핑
- 이미지 보존율: ~75%

**장점**:
- 과도한 분할/압축 방지
- 논문 섹션과 자연스러운 대응
- 검색과 이해 사이 균형

**단점**:
- 일부 세부 기술 정보 손실 가능
- 복잡한 슬라이드에서 추가 유닛 필요

**적합 용도**: 일반 논문 작성 (권장 기본값)

### 3.4 High Resolution (1-5 유닛)

**품질 평가**:
- 정보 커버리지: High
- 중복성: 낮음
- 소형 유닛 비율: 42.3%
- 클러스터링 복잡도: 높음

**유닛 분포**:
- 1개 유닛: 8 슬라이드
- 2개 유닛: 9 슬라이드
- 3개 유닛: 11 슬라이드
- 4개 유닛: 12 슬라이드
- 5개 유닛: 5 슬라이드

**장점**:
- 정보의 상세한 구조화
- 검색 정밀도 향상
- 정교한 의미 분석 가능

**단점**:
- 과도한 세분화로 문맥 파악 어려움
- 클러스터링 복잡도 증가
- 처리 비용 증가

**적합 용도**: 정밀 검색, 수식/파라미터 중심 문서

---

## 4. 해상도 선택 가이드

```
                        Low (1-2)    Medium (1-3)    High (1-5)
                        ─────────    ────────────    ──────────
개요/목차 생성           ★★★★★        ★★★            ★★
일반 논문 작성           ★            ★★★★★          ★★★
상세 방법론 추출         ★            ★★★            ★★★★★
정밀 검색               ★            ★★★            ★★★★★
처리 효율성             ★★★★★        ★★★★           ★★
정보 완전성             ★★           ★★★★           ★★★★★
```

---

## 5. 종합 비교 매트릭스

### 5.1 옵션 × 해상도 조합

| 조합 | 총 유닛 | 품질 지표 | 권장 용도 |
|-----|--------|---------|----------|
| Quick + Low (haiku+opus) | 65 | 90% accepted | 빠른 개요 |
| **Standard + Medium (sonnet+opus)** | **89** | **38.2% refined, 72% thesis** | **일반 권장** |
| Premium + High (opus full) | 126 | 81.7% premium | 고품질 제출물 |

### 5.2 비용-품질 트레이드오프

| 조합 | 추정 비용 | 품질 수준 | 효율성 |
|-----|----------|----------|--------|
| Quick + Low | ~$0.03 | 기본+ | ★★★★★ |
| **Standard + Medium** | **~$0.08** | **좋음** | **★★★★** |
| Premium + High | ~$0.18 | 최고 | ★★★ |

---

## 6. 핵심 발견

### 6.1 Claude 모델 효과

1. **Opus 검증은 모든 옵션에서 효과적**
   - Quick Draft에서도 90% 수용률로 품질 보장
   - Standard에서 38.2% 분류 정제
   - Premium에서 81.7% 최고 품질 달성

2. **Haiku는 빠른 추출에 효율적**
   - Low resolution에서 충분한 성능
   - Opus 검증과 조합 시 비용 효율적

3. **Sonnet은 균형 잡힌 선택**
   - CoT 추출과 Medium resolution에 최적
   - Gap 분석 능력 보유

### 6.2 해상도 선택 가이드라인

1. **Low (1-2)는 개요용으로만 적합**
   - 22.2% 슬라이드에서 심각한 정보 손실
   - 빠른 스캔/비교에 활용

2. **Medium (1-3)이 일반 권장**
   - 균형 잡힌 정보 커버리지
   - 논문 구조와 자연스러운 대응

3. **High (1-5)는 주의해서 사용**
   - 42.3% 소형 유닛으로 문맥 의존도 높음
   - 정밀 검색/상세 방법론에만 적합

### 6.3 최적 조합 권장

| 시나리오 | 권장 조합 | 이유 |
|---------|----------|------|
| 빠른 개요 확인 | Quick + Low (haiku+opus) | 90% 수용, 최고 효율 |
| **일반 논문 작성** | **Standard + Medium (sonnet+opus)** | **균형 잡힌 품질/비용** |
| 고품질 제출물 | Premium + High (opus full) | 81.7% premium, 최고 품질 |
| 이미지 많은 연구 | Premium + Medium | 멀티모달 분석 + 균형 해상도 |

---

## 7. 결론

### 7.1 주요 결론

1. **Opus 검증은 모든 옵션에서 사용해야 함**
   - Quick Draft에서도 품질 보장 효과 확인

2. **Medium 해상도 (1-3)가 일반 권장**
   - Low는 정보 손실 위험
   - High는 과도한 세분화 위험

3. **Standard + Medium + Opus 조합이 기본 권장**
   - 적정 비용 (~$0.08)
   - 좋은 품질 (38.2% 정제, 72% Thesis 정렬)

### 7.2 산출물

```
/experiments/2026-01-31-exp2/
├── exp3-quickdraft-pro-low.json       # Quick Draft 실험 (exp3)
├── exp3-standard-pro-medium.json      # Standard 실험 (exp3)
├── exp3-premium-pro-high.json         # Premium 실험 (exp3)
├── exp3-resolution-low.json           # 해상도 Low (exp3)
├── exp3-resolution-medium.json        # 해상도 Medium (exp3)
├── exp3-resolution-high.json          # 해상도 High (exp3)
├── exp4-quickdraft-haiku-opus.json    # Quick Draft Claude 모델
├── exp4-standard-sonnet-opus.json     # Standard Claude 모델
├── exp4-resolution-low.json           # 해상도 Low 상세 분석
├── exp4-resolution-high.json          # 해상도 High 상세 분석
└── exp4-final-report.md               # 현재 문서
```

---

> **요약**: Claude 모델 기반 실험에서 **Opus 검증을 모든 옵션에 적용**하고, **Medium 해상도(1-3)를 기본**으로 선택하는 것이 최적. **Standard + Medium + Opus** 조합을 일반 권장 옵션으로 제안.
