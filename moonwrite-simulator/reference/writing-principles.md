# Writing Principles: 연구 중 집필 원칙

Moonwriter의 논문 생성에 적용되는 집필 원칙을 정의합니다.

## 1. 핵심 철학: 연구 중 집필 (Writing as you Research)

### 패러다임 전환

> 집필을 연구의 최종 기록 활동이 아닌, **연구 과정의 복잡성을 관리하고 논리적 간극을 발견하는 인지적 도구**로 정의한다.

### 주요 이점

- **집필 부하 분산**: 연구 종료 후 막대한 집필 부담을 연구 과정에 분산
- **데이터 망각 방지**: 연구 시점의 맥락과 통찰을 즉시 기록
- **연구 질문 정교화**: 글로 표현하는 과정에서 논리적 허점 발견
- **정시 완료 가능성 향상**: 점진적 진행으로 예측 가능한 완성

---

## 2. 기초 원칙 (Foundation Principles)

### 2.1 핵심 주장 확립 (Thesis Statement)

> 연구의 목적을 **명확한 하나의 선언적 문장**으로 작성하여 모든 단락의 이정표로 삼는다.

**정의:**
- Working Title: 연구 전체를 관통하는 핵심 질문 또는 주장
- 모든 섹션은 이 핵심 주장을 지지하거나 답변하는 방향으로 작성

**예시:**
```
[약한 주장] BERT 모델을 사용한 감성 분석 연구
[강한 주장] 사전학습된 BERT 모델은 도메인 특화 fine-tuning을 통해
           기존 감성 분석 방법보다 15% 높은 정확도를 달성할 수 있다.
```

---

### 2.2 원자적 기록 (Atomic Documentation)

> 연구의 각 시점에서 발생하는 지식 조각들을 **논문의 구성 요소로 즉시 변환**한다.

**방법론 기록 원칙 (To-clause):**
- 수행한 모든 과정에 "~하기 위해"라는 목적절 사용
- 연구의 당위성을 명시적으로 부여

**예시:**
```
[Before] 데이터를 8:2로 분할하였다.
[After]  모델의 일반화 성능을 평가하기 위해, 데이터를 학습(80%)과
         테스트(20%)로 분할하였다.
```

**결과 기록 원칙 (Knowledge Atoms):**
- 방법론과 1:1 대응하는 결과값을 객관적 사실 단위로 기록
- 해석 없이 순수한 측정값/관찰값만 기록

---

### 2.3 플레이스홀더 (Placeholder)

> 아직 데이터가 확보되지 않은 부분은 **작성 예정 사양**을 명시하여 다음 연구 방향을 가이드한다.

**정의:**
- 빈 섹션을 방치하지 않고, 필요한 내용을 명세로 기록
- 연구자가 수행해야 할 실험이나 분석을 구체적으로 지시

**예시:**
```
### 3.2 Ablation Study
[작성 예정: attention 메커니즘 제거 시 성능 변화 측정 필요.
 예상 실험: base 모델 vs no-attention 모델 비교]
```

---

## 3. 구조 원칙 (Structure Principles)

### 3.1 골격 드래프팅 (Skeleton Drafting)

> 논문의 각 문단에서 **첫 문장(주제문)**과 **마지막 문장(결론문)**만 먼저 작성하여 전체 뼈대를 구축한다.

**구성:**
- **주제문 (Topic Sentence)**: 문단의 핵심 주장을 선언
- **결론문 (Closing Sentence)**: 문단의 의미를 정리하거나 다음으로 연결

**예시:**
```
[주제문]  제안된 방법은 모든 평가 지표에서 baseline을 상회하였다.
[본문]    ... (나중에 채움) ...
[결론문]  이러한 성능 향상은 attention 메커니즘의 효과를 입증한다.
```

---

### 3.2 모듈형 집필 (Modular Writing)

> 각 섹션을 **독립적이고 재사용 가능한 단위**로 작성한다.

**원칙:**
- **단일 목적 (Single-purpose)**: 각 모듈은 하나의 핵심 아이디어만 다룸
- **의존성 최소화**: 다른 섹션 내용을 직접 참조하지 않음
- **교체 가능성**: 연구 방향 변경 시 해당 모듈만 업데이트

**적용:**
- "앞서 언급한" 대신 핵심 내용을 간략히 재진술
- 각 섹션만 읽어도 맥락 이해 가능

---

## 4. 연결 원칙 (Connection Principles)

### 4.1 이정표 (Signpost)

> 각 섹션의 시작부에서 **전체 연구 목표와의 연결성**을 명시한다.

**역할:**
- 독자에게 "지금 어디에 있고, 왜 여기에 있는지" 안내
- 서사의 연속성 확보

**적용 위치:** Introduction, Methods, Results, Discussion 각 섹션 첫 문단

**예시:**
```
[Before] 본 연구에서는 BERT 모델을 사용하였다.
[After]  앞서 제기한 "감성 분석 정확도 향상" 문제를 해결하기 위해,
         본 연구에서는 BERT 모델을 사용하였다.
```

---

### 4.2 글루 문장 (Glue Sentence)

> 섹션의 끝에서 **다음 섹션으로 연결**하는 전환 문장을 삽입한다.

**역할:**
- 분절적으로 작성된 모듈들을 하나의 서사로 연결
- 독자의 인지적 부하 감소

**적용 위치:** 각 섹션 마지막 문단

**예시:**
```
[Before] F1 스코어는 0.85로 측정되었다.
[After]  F1 스코어는 0.85로 측정되었다. 이러한 결과가 의미하는 바와
         한계점에 대해서는 다음 절에서 논의한다.
```

---

## 5. 검증 원칙 (Verification Principles)

### 5.1 역개요 (Reverse Outline)

> 작성된 드래프트에서 **핵심 주장만 추출**하여 초기 설계된 논리 경로에서 이탈하지 않았는지 점검한다.

**방법:**
1. 각 문단의 주제문만 추출
2. 추출된 문장들을 순서대로 나열
3. 핵심 주장(Thesis)과의 연결성 확인
4. 논리적 비약이나 불필요한 반복 발견

---

## 6. 섹션별 적용 가이드

| 섹션 | 이정표 | 주제문 | 글루 문장 | To-clause |
|------|--------|--------|-----------|-----------|
| Abstract | - | O | - | - |
| Introduction | O | O | O | - |
| Methods | O | O | O | O |
| Results | O | O | O | - |
| Discussion | O | O | O | - |
| Conclusion | O | O | - | - |

---

## 7. 옵션별 Writing Principles 적용

### Standard (3개 항목)

| 항목 | 설명 |
|------|------|
| **thesis_clarity** | 핵심 주장이 명확히 제시되었는가 |
| **evidence_integration** | 증거가 적절히 통합되었는가 |
| **logical_flow** | 논리적 흐름이 자연스러운가 |

### Premium (6개 항목)

| 항목 | 설명 |
|------|------|
| **thesis_clarity** | 핵심 주장이 명확히 제시되었는가 |
| **evidence_integration** | 증거가 적절히 통합되었는가 |
| **logical_flow** | 논리적 흐름이 자연스러운가 |
| **technical_precision** | 기술 용어가 정확하게 사용되었는가 |
| **reproducibility** | 재현 가능한 수준으로 기술되었는가 |
| **contribution_clarity** | 연구 기여가 명확히 드러나는가 |

---

## 8. 구현 현황

### 원칙 정의 (Single Source of Truth)

- `lib/writing-principles.ts`: 프롬프트에 주입되는 원칙 정의
  - `CORE_PHILOSOPHY`: 핵심 철학
  - `FOUNDATION_PRINCIPLES`: 기초 원칙 (핵심 주장, To-clause, 플레이스홀더)
  - `STRUCTURE_PRINCIPLES`: 구조 원칙 (골격 드래프팅, 모듈형 집필)
  - `CONNECTION_PRINCIPLES`: 연결 원칙 (이정표, 글루 문장)
  - `STANDARD_PRINCIPLES`: Standard 옵션용 (3개 항목)
  - `PREMIUM_PRINCIPLES`: Premium 옵션용 (6개 항목)
  - `getPaperGenerationPrinciples()`: 논문 생성 프롬프트용
  - `getNarrativeArcPrinciples()`: Narrative Arc 분석 프롬프트용

> 주의: 이 문서와 `lib/writing-principles.ts`는 동일한 개념을 정의합니다. 수정 시 양쪽 일관성을 유지하세요.

### API

| 엔드포인트 | 적용 원칙 |
|------------|----------|
| `POST /api/library/cluster` | 핵심 주장 추출, 이정표 연결 (thesisConnection) |
| `POST /api/paper/generate` | 핵심 주장, 골격 드래프팅, 이정표, 글루 문장, 모듈형 집필 |

---

## 9. 핵심 원칙

> Moonwriter의 Writing Principles는 **집필을 인지적 도구로 활용**하여 연구의 논리적 간극을 발견하고, **기초-구조-연결-검증**의 4단계 원칙을 통해 논리적이고 읽기 쉬운 논문 작성을 지원한다.
>
> - **Standard**: 3개 핵심 항목 (thesis_clarity, evidence_integration, logical_flow)
> - **Premium**: 6개 전체 항목 (+ technical_precision, reproducibility, contribution_clarity)
