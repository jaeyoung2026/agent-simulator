# Clustering Model 이해 문서

> 이 문서는 Moonwriter 서비스의 클러스터링 모델을 분석하고 이해한 내용을 정리한 것입니다.

---

## 1. 클러스터링의 목적

Semantic Unit(의미 단위)들을 **논문 구조**로 조직화하는 과정입니다.

```
[개별 의미 단위들] → 클러스터링 → [5단계 논문 흐름]
                                   ① 배경/동기
                                   ② 방법론
                                   ③ 결과
                                   ④ 해석/논의
                                   ⑤ 기여/향후
```

---

## 2. 두 가지 전략

### 자동 선택 기준 (mode: "auto")

| 유닛 수 | 전략 | 이유 |
|--------|------|------|
| ~500개 | Direct | 컨텍스트 여유, 품질 충분 |
| 500개+ | Distributed | 품질 유지 + 비용 최적화 |

### Direct vs Distributed 비교

| 항목 | Direct | Distributed |
|-----|--------|-------------|
| **처리 방식** | 단일 LLM 호출 | 4단계 분리 처리 |
| **Thesis 추출** | 암시적 | 명시적 (1단계) |
| **Thesis 전파** | 없음 | 모든 단계에 전달 |
| **일관성 검증** | 없음 | 3단계에서 검증 |
| **품질 검증** | 간극만 | 간극 + 논문 품질 |
| **비용 (1000개)** | ~$0.22 | ~$0.07 (70% 절감) |

---

## 3. Thesis-First 패턴 (핵심 아이디어)

Distributed 전략의 핵심은 **Thesis-First**입니다.

> **핵심 주장(Thesis)을 가장 먼저 추출하고, 모든 후속 분석에 전파한다.**

### Thesis 구조

```typescript
interface Thesis {
  mainQuestion: string;   // 연구가 답하려는 핵심 질문 (한 문장)
  mainClaim: string;      // 연구가 주장하는 핵심 내용 (한 문장)
  keyEvidence: string[];  // 주장을 뒷받침하는 핵심 증거 유닛 ID 목록
}
```

### 왜 Thesis-First인가?

1. **일관성**: 모든 클러스터가 하나의 핵심 주장을 향해 수렴
2. **연결성**: 각 클러스터가 Thesis와 어떻게 연결되는지 명시 가능
3. **검증 가능**: Thesis 기준으로 빈틈/약한 연결 감지 가능

---

## 4. Distributed 4단계 상세

### 1단계: Thesis 추출 + 경량 분류 (Flash)

**입력**: 전체 의미 단위 (경량화된 LightweightUnit)

**출력**:
- `thesis`: 핵심 주장
- `clusters`: 클러스터 할당 (유닛 ID 목록)
- `crossReferences`: 유닛 간 참조
- `categoryLinks`: 카테고리 간 연결

**적용 원칙**: 시간적 우선순위, Thesis-First, 5단계

### 2단계: Thesis-Aware 클러스터 분석 (Flash 병렬)

**입력**: thesis + 클러스터별 유닛 원본 (병렬 처리)

**출력**:
- `description`: 클러스터 설명
- `keyInsight`: 핵심 인사이트
- `thesisConnection`: Thesis와의 연결

**적용 원칙**: thesisConnection 가이드, Thesis/이정표

### 3단계: 일관성 검증 + 흐름 통합 (Flash)

**입력**: 2단계 결과 + crossReferences

**검증**: thesisConnection ↔ thesis 연결 일관성

**출력**:
- `flowAnalysis`: 흐름 분석
- `relationAnalysis`: 관계 분석

**적용 원칙**: 일관성 검증 기준, 5단계/관계 유형

### 4단계: 논문 품질 검증 (Pro)

**입력**: 3단계 결과

**검증**: thesis 일관성, 섹션 변환 가능성, 간극

**출력**:
- `gapAnalysis`: 간극 분석
- `qualityIssues`: 품질 이슈

**적용 원칙**: 완성도 체크리스트, 비판적 검토, 정보 손실 감지

---

## 5. 시간적 우선순위 원칙

> 최신 파일의 유닛이 연구의 최신 정보를 반영할 가능성이 높다.

### 규칙

| 규칙 | 설명 |
|-----|------|
| recentFirst | recency가 'recent'인 유닛은 최신 연구 결과를 반영 |
| conflictResolution | 동일 주제에 여러 유닛이 있으면 최신 유닛의 정보를 우선 반영 |
| updateAssumed | 오래된 유닛과 최신 유닛의 내용이 충돌하면 최신 정보가 업데이트된 것으로 간주 |

### 대표 유닛 선정 기준

1. **최신 유닛 우선** (24시간 이상 차이 시)
2. **Primary Role 우선** (PrimaryResult 역할)
3. **keyFindings 많은 순**

---

## 6. thesisConnection 작성 가이드

각 클러스터가 핵심 주장에 **어떻게 기여하는지** 명시합니다.

### Stage별 연결

| Stage | thesisConnection |
|-------|------------------|
| background | 핵심 주장의 필요성/배경 제시 |
| method | 핵심 주장을 검증하기 위한 방법 설명 |
| result | 핵심 주장을 지지하는 증거 제시 |
| interpretation | 핵심 주장의 의미 해석 |
| contribution | 핵심 주장의 기여점 정리 |

### 좋은 예 vs 나쁜 예

| 구분 | 예시 |
|-----|------|
| **좋은 예** | "이 클러스터는 BERT의 효과성을 실험적으로 검증하여 핵심 주장의 근거를 제공한다" |
| **나쁜 예** | "결과를 보여준다" (너무 추상적) |

---

## 7. 품질 보장 메커니즘

### 연결 강도 (connectionStrength)

| 강도 | 정의 | 기준 |
|-----|------|------|
| **strong** | 구체적 데이터/실험 결과/인용으로 연결됨 | 증거 유형 2개 이상 |
| **moderate** | 개념적 연결은 있으나 구체적 증거 부족 | 증거 유형 1개 |
| **weak** | 연결 근거 없음 또는 논리적 비약 존재 | 증거 없음 |

### 증거 유형

| 유형 | 설명 |
|-----|------|
| dataReference | 동일 데이터셋/실험 결과를 참조 |
| logicalChain | A이므로 B라는 명시적 인과 관계 |
| citation | 이전 클러스터 내용을 명시적으로 언급 |
| methodResult | method 클러스터의 절차가 result에서 실행됨 |

### 완성도 체크리스트

| Stage | 필수 요소 |
|-------|----------|
| background | 연구 문제의 중요성, 기존 연구 한계, 본 연구 차별점 |
| method | 방법론 선택 근거, 구체적 절차, 데이터 수집 방법 |
| result | 정량/정성 데이터, 통계적 유의성, 예상 외 결과 언급 |
| interpretation | 결과 해석, 기존 연구 비교, 한계점 인정 |
| contribution | 학문적/실용적 기여, 후속 연구 방향, 핵심 주장 재확인 |

---

## 8. 비판적 검토 원칙 (긍정 편향 방지)

### 반드시 확인할 질문

- 이 클러스터 없이도 Thesis가 성립하는가? → 그렇다면 연결 약함
- 이 연결을 독자가 이해할 수 있는가? → 아니라면 설명 부족
- 주장에 대한 근거가 제시되었는가? → 아니라면 간극 존재

### 품질 판단 기준 (엄격 적용)

| 품질 | 기준 |
|-----|------|
| **high** | 모든 stage 필수 요소 충족 + 모든 연결 strong/moderate |
| **medium** | 일부 필수 요소 미충족 또는 weak 연결 1-2개 존재 |
| **low** | 필수 요소 다수 누락 또는 weak 연결 3개 이상 |

---

## 9. 정보 손실 감지

### 손실 발생 지점

| 단계 | 손실 내용 |
|-----|----------|
| 1단계 | 경량 유닛 변환 시 summary, keyFindings 상세 내용 손실 |
| 2단계 | 병렬 처리로 클러스터 간 교차 참조 맥락 단절 |
| 3단계 | 대표 유닛만 검토하여 비대표 유닛 정보 누락 |

### 완화 전략

- 3단계에서 crossReferences를 재검토하여 누락된 연결 복원
- 4단계에서 원본 유닛 수 대비 반영된 정보량 검증

---

## 10. 단계 간 데이터 보존 규칙

> **중요**: 2단계 LLM은 `unitIds`를 응답에서 생략할 수 있다. 1단계에서 확정된 `unitIds`를 신뢰 소스로 사용한다.

| 필드 | 신뢰 소스 | 이유 |
|-----|---------|------|
| `unitIds` | **1단계** | 클러스터 멤버십은 1단계에서 확정 |
| `description` | 2단계 | 상세 분석 결과 |
| `keyInsight` | 2단계 | Thesis-Aware 인사이트 |
| `thesisConnection` | 2단계 (3단계 강화) | 핵심 주장 연결 |

---

## 11. 핵심 통찰

### Semantic Model과의 연결

| Semantic Model | Clustering Model |
|---------------|------------------|
| Semantic Unit 추출 | Unit을 클러스터로 그룹화 |
| Category (구조적 위치) | Stage (5단계)에 매핑 |
| Role (기능적 역할) | 대표 유닛 선정 기준 |
| Relation (미정의) | 클러스터 간 관계 분석 |

### 설계 철학

1. **Thesis-First**: 핵심 주장을 먼저 추출하고 모든 분석에 전파
2. **시간적 우선순위**: 최신 정보를 우선 반영
3. **품질 보장**: 다층적 검증 (일관성, 완성도, 비판적 검토)
4. **비용 최적화**: Flash 최대 활용, Pro는 검증에만

---

## 12. 코드-문서 동기화

| 문서 | 동기화 대상 코드 |
|------|-----------------|
| `clustering-principles.md` | `lib/clustering-principles.ts` |
| `clustering-strategy.md` | `lib/clustering-strategies.ts` |

수정 시 반드시 문서와 코드를 함께 수정해야 합니다.
