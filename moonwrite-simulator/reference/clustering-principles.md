# Clustering Principles: 클러스터링 분석 원칙

Moonwriter의 클러스터링 분석에 적용되는 원칙을 정의합니다.

> 이 파일은 `lib/clustering-principles.ts`와 동기화 유지

---

## 1. 시간적 우선순위 (Temporal Priority)

### 원칙

> 최신 파일의 유닛이 연구의 최신 정보를 반영할 가능성이 높다.

### 규칙

| 규칙 | 설명 |
|-----|------|
| **recentFirst** | recency가 'recent'인 유닛은 최신 연구 결과를 반영 |
| **conflictResolution** | 동일 주제에 여러 유닛이 있으면 최신 유닛의 정보를 우선 반영 |
| **updateAssumed** | 오래된 유닛과 최신 유닛의 내용이 충돌하면 최신 정보가 업데이트된 것으로 간주 |

### 적용 전략

- Direct: O
- Distributed: O (모든 단계)

---

## 2. Thesis-First 원칙

### 원칙

> 핵심 주장(Thesis)을 가장 먼저 추출하고, 모든 후속 분석에 전파한다.

### 구조

```typescript
interface Thesis {
  mainQuestion: string;   // 연구가 답하려는 핵심 질문 (한 문장)
  mainClaim: string;      // 연구가 주장하는 핵심 내용 (한 문장)
  keyEvidence: string[];  // 주장을 뒷받침하는 핵심 증거 유닛 ID 목록
}
```

### 적용 전략

- Direct: X (한 번에 처리하므로 별도 단계 불필요)
- Distributed: O (1단계에서 추출, 2-3단계에 전파)

---

## 3. thesisConnection 작성 가이드

### 원칙

> 각 클러스터가 핵심 주장에 어떻게 기여하는지 명시한다.

### 작성 기준

핵심 주장을 지지하는 관점에서 **구체적으로** 작성

| 구분 | 예시 |
|-----|------|
| **좋은 예** | "이 클러스터는 BERT의 효과성을 실험적으로 검증하여 핵심 주장의 근거를 제공한다" |
| **나쁜 예** | "결과를 보여준다" (너무 추상적) |

### Stage별 연결

| Stage | thesisConnection |
|-------|------------------|
| background | 핵심 주장의 필요성/배경 제시 |
| method | 핵심 주장을 검증하기 위한 방법 설명 |
| result | 핵심 주장을 지지하는 증거 제시 |
| interpretation | 핵심 주장의 의미 해석 |
| contribution | 핵심 주장의 기여점 정리 |

### 적용 전략

- Direct: X (자동 생성)
- Distributed: O (2단계에서 Thesis 컨텍스트와 함께 생성)

---

## 4. 일관성 검증 기준

### 연결 강도 (connectionStrength) - 증거 기반 판단

| 강도 | 정의 | 조치 |
|-----|------|------|
| **strong** | 구체적 데이터/실험 결과/인용으로 연결됨 | 유지 |
| **moderate** | 개념적 연결은 있으나 구체적 증거 부족 | 선택적 강화 |
| **weak** | 연결 근거 없음 또는 논리적 비약 존재 | 강화 제안 생성 |

### 서사 일관성 (narrativeCoherence)

| 상태 | 정의 |
|-----|------|
| **coherent** | 각 클러스터가 Thesis를 향해 수렴하며 빈틈 없음 |
| **fragmented** | 클러스터 간 단절 또는 Thesis와 무관한 내용 존재 |

### 적용 전략

- Direct: X (단일 호출이라 자체 일관성)
- Distributed: O (3단계에서 검증)

---

## 5. 완성도 체크리스트

### 원칙

> 각 stage는 필수 요소를 포함해야 논문으로 완성될 수 있다.

### Stage별 필수 요소

| Stage | 핵심 질문 | 필수 요소 |
|-------|----------|----------|
| **background** | 왜 이 연구가 필요한가? | 연구 문제의 중요성, 기존 연구 한계, 본 연구 차별점 |
| **method** | 어떻게 수행했는가? | 방법론 선택 근거, 구체적 절차, 데이터 수집 방법 |
| **result** | 무엇을 발견했는가? | 정량/정성 데이터, 통계적 유의성, 예상 외 결과 언급 |
| **interpretation** | 결과가 무엇을 의미하는가? | 결과 해석, 기존 연구 비교, 한계점 인정 |
| **contribution** | 왜 중요한가? | 학문적/실용적 기여, 후속 연구 방향, 핵심 주장 재확인 |

### 적용 전략

- Direct: X
- Distributed: O (4단계 Pro 검증에서 적용)

---

## 6. 연결 증거 기준

### 원칙

> 연결 강도는 주관적 판단이 아닌 증거 유무로 결정한다.

### 증거 유형

| 유형 | 설명 |
|-----|------|
| **dataReference** | 동일 데이터셋/실험 결과를 참조 |
| **logicalChain** | A이므로 B라는 명시적 인과 관계 |
| **citation** | 이전 클러스터 내용을 명시적으로 언급 |
| **methodResult** | method 클러스터의 절차가 result에서 실행됨 |

### 판단 기준

| 강도 | 기준 |
|-----|------|
| **strong** | 위 증거 유형 중 2개 이상 존재 |
| **moderate** | 위 증거 유형 중 1개 존재 |
| **weak** | 증거 없이 암묵적 연결만 가정 |

### 적용 전략

- Direct: X
- Distributed: O (3단계 일관성 검증에서 적용)

---

## 7. 비판적 검토 원칙 (긍정 편향 방지)

### 원칙

> 검증 단계에서는 긍정 편향을 경계하고 엄격하게 평가한다.

### 경계해야 할 편향

- 자신이 생성한 내용을 과대평가하는 경향 경계
- "대체로 좋음"이라는 모호한 평가 지양
- 문제가 없어 보여도 빠진 것이 없는지 재확인

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

### 적용 전략

- Direct: X
- Distributed: O (4단계 Pro 검증에서 적용)

---

## 8. 정보 손실 감지 원칙

### 원칙

> 분산 처리 과정에서 발생하는 정보 손실을 적극적으로 감지한다.

### 손실 발생 지점

| 단계 | 손실 내용 |
|-----|----------|
| **1단계** | 경량 유닛 변환 시 summary, keyFindings 상세 내용 손실 |
| **2단계** | 병렬 처리로 클러스터 간 교차 참조 맥락 단절 |
| **3단계** | 대표 유닛만 검토하여 비대표 유닛 정보 누락 |

### 감지 질문

- 원본 유닛의 핵심 데이터가 최종 결과에 반영되었는가?
- 클러스터 A와 B의 관계가 2단계 병렬 처리에서 놓쳤을 가능성은?
- 대표 유닛 외의 유닛들이 가진 고유 정보가 누락되지 않았는가?

### 완화 전략

- 3단계에서 crossReferences를 재검토하여 누락된 연결 복원
- 4단계에서 원본 유닛 수 대비 반영된 정보량 검증

### 적용 전략

- Direct: X (단일 호출이라 손실 없음)
- Distributed: O (4단계 Pro 검증에서 적용)

---

## 9. 전략별 원칙 적용

### Direct 전략

```
단일 호출
   │
   ├── 시간적 우선순위 ✓
   ├── 5단계/관계 유형 (semantic-principles) ✓
   └── Thesis/이정표 (writing-principles) ✓
```

### Distributed 전략

```
1단계: Thesis 추출 + 분류
   ├── 시간적 우선순위 ✓
   ├── Thesis-First ✓
   └── 5단계 (semantic-principles) ✓
       │
       ▼
2단계: 클러스터별 상세 분석 (병렬)
   ├── thesisConnection 가이드 ✓
   └── Thesis/이정표 (writing-principles) ✓
       │
       ▼
3단계: 일관성 검증 + 통합
   ├── 일관성 검증 기준 ✓ (연결 증거 기준 포함)
   └── 5단계/관계 유형 (semantic-principles) ✓
       │
       ▼
4단계: 품질 검증 (Pro) - 엄격 적용
   ├── 완성도 체크리스트 ✓
   ├── 비판적 검토 원칙 ✓
   └── 정보 손실 감지 ✓
```

---

## 10. 구현 현황

### 원칙 정의

| 파일 | 역할 |
|------|------|
| `lib/clustering-principles.ts` | 원칙 상수 및 프롬프트 함수 |

### 함수 매핑

| 함수 | 적용 위치 |
|------|----------|
| `getTemporalPriorityPrinciples()` | Direct, Distributed 1단계 |
| `getThesisFirstPrinciples()` | Distributed 1단계 |
| `getThesisConnectionGuide()` | Distributed 2단계 |
| `getConsistencyCriteria()` | Distributed 3단계 |
| `getCompletenessChecklist()` | Distributed 4단계 |
| `getCriticalReviewPrinciples()` | Distributed 4단계 |
| `getInformationLossDetection()` | Distributed 4단계 |
| `getDirectStrategyPrinciples()` | Direct 전략 전체 |
| `getDistributedStep1Principles()` | Distributed 1단계 |
| `getDistributedStep2Principles()` | Distributed 2단계 |
| `getDistributedStep3Principles()` | Distributed 3단계 |
| `getDistributedStep4Principles()` | Distributed 4단계 |

---

## 11. 옵션별 원칙 적용

### Quick Draft (Direct 전략)

| 원칙 | 적용 |
|------|------|
| 시간적 우선순위 | O |
| Thesis-First | X (한 번에 처리) |
| thesisConnection | X (자동 생성) |
| 일관성 검증 | X |
| 완성도 체크리스트 | X |

### Standard (Distributed 전략)

| 원칙 | 적용 |
|------|------|
| 시간적 우선순위 | O |
| Thesis-First | O (1단계) |
| thesisConnection | O (2단계) |
| 일관성 검증 | O (3단계) |
| 완성도 체크리스트 | O (4단계) |
| 역개요 검증 | **X** |

### Premium (Distributed 전략 + 추가 기능)

| 원칙 | 적용 |
|------|------|
| 시간적 우선순위 | O |
| Thesis-First | O (1단계) |
| thesisConnection | O (2단계) |
| 일관성 검증 | O (3단계) |
| 완성도 체크리스트 | O (4단계) |
| **역개요 검증** | **O** |
| **Self-Critique** | **O** |
| **정보 손실 감지** | **O** |

---

## 12. 핵심 원칙

> 클러스터링 원칙은 **옵션에 따라 선택적으로 적용**된다.
>
> - **Quick Draft**: Direct 전략, 최소 원칙 적용
> - **Standard**: Distributed 전략, 핵심 원칙 적용
> - **Premium**: Distributed 전략 + 역개요 검증, Self-Critique, 정보 손실 감지
>
> 원칙 수정 시 `lib/clustering-principles.ts`와 이 문서를 함께 수정하라.
