# Semantic Principles: 의미 단위 분석 원칙

> **삼분법 위치**: 원칙 (Principles)
>
> 이 문서는 AI가 슬라이드를 분석하여 Semantic Unit을 추출할 때 적용해야 하는 **판단 기준과 개념 정의**를 다룹니다.
>
> **동기화**: `lib/semantic-principles.ts`와 동기화 유지

---

## 1. 핵심 철학

### 모델의 목적

> **연구 결과를 분류하는 체계가 아니라, 연구가 '논문으로 자라나는 과정'을 표현하는 개념 모델**

### 핵심 질문

분석 시 항상 다음 질문에 답할 수 있어야 합니다:

1. 이 연구에서 **의미 있는 주장들은 무엇인가?**
2. 각 주장은 논문에서 **어떤 역할**을 수행하는가?
3. 주장들은 **어떤 논증 흐름**으로 연결되는가?
4. 기존 연구는 **어디에서, 어떤 이유로 호출되는가?**

---

## 2. Semantic Unit 정의

### 원칙

> 논문에서 **하나의 문단(paragraph)** 을 구성할 수 있는 **최소 의미 단위의 주장 또는 개념**

### 판단 기준

| O (Semantic Unit) | X (Semantic Unit 아님) |
|-------------------|----------------------|
| 하나의 질문에 대한 하나의 답 | 슬라이드 한 장 |
| 독립적으로 설명 가능한 개념 | 문장 하나 |
| 논문 한 문단으로 확장 가능 | 키워드 나열 |

### 추출 규칙

| 규칙 | 설명 |
|-----|------|
| **분리 우선** | 하나의 슬라이드에 여러 개념이 있으면 별도 Unit으로 분리 |
| **의미 중심** | 물리적 위치가 아닌 의미적 완결성으로 판단 |
| **빈 Unit 회피** | 의미 없는 슬라이드(표지, 목차)는 Unit 생성하지 않음 |

### 적용 위치

- 슬라이드 분석: O (`getSlideAnalysisPrinciples()`)
- 클러스터링: O (Unit 정의 참조)

---

## 3. Category (구조적 위치)

### 원칙

> Semantic Unit의 **논문 내 기본 좌표**. 논문 구조(IMRaD)를 반영하며 **단일 값**이다.

### 정의

| Category | 설명 |
|----------|------|
| `introduction` | 연구 배경, 목적, 동기, 문제 정의 |
| `method` | 연구 방법, 실험 설계, 데이터셋, 모델 구조 |
| `result` | 실험 결과, 성능 비교, 데이터 분석 |
| `discussion` | 결과 해석, 한계점, 시사점 |
| `conclusion` | 결론, 기여점, 향후 과제 |
| `reference` | 참고문헌 |
| `other` | 표지, 목차, 감사 인사 등 |

### 적용 위치

- 슬라이드 분석: O (`CATEGORY_DEFINITIONS`)
- 클러스터링: O (Stage 매핑 참조)

---

## 4. Role (기능적 역할)

### 원칙

> Semantic Unit이 **논문 전체 논증에서 수행하는 기능적 역할**. **복수 선택 가능**하며, 다층적 의미를 표현한다.

### 정의

| Role | 설명 |
|------|------|
| `Background` | 연구 배경/문제 제시 |
| `MethodComponent` | 방법의 구성 요소 |
| `PrimaryResult` | 핵심 실험 결과 |
| `Supporting` | 보조/비교 결과 |
| `Interpretation` | 결과 해석 |
| `LimitationEvidence` | 한계의 근거 |
| `Contribution` | 기여점 명시 |
| `FutureWorkSeed` | 후속 연구 연결점 |

### Category vs Role

| 관점 | Category | Role |
|------|----------|------|
| 질문 | "어디에 위치하는가?" | "무엇을 하는가?" |
| 선택 | 단일 | 복수 가능 |
| 예시 | result | PrimaryResult + Interpretation |

### 적용 위치

- 슬라이드 분석: O (`ROLE_DEFINITIONS`)
- 클러스터링: O (Stage 배치 기준)

---

## 5. Reference Role (외부 참조 역할)

### 원칙

> 외부 논문은 Semantic Unit이 아니다. 항상 **우리 Unit을 설명·정당화·비교하기 위해 호출된 외부 증거**로만 존재한다.

### 정의

| Reference Role | 설명 |
|----------------|------|
| `Background` | 연구 배경 제시 |
| `Comparison` | 성능/방법 비교 |
| `Support` | 주장 강화 |
| `Contrast` | 반례 또는 차별점 |

### 적용 위치

- 슬라이드 분석: O (`REFERENCE_ROLE_DEFINITIONS`)
- 클러스터링: X

---

## 6. Relation Type (논증 관계)

### 원칙

> Semantic Unit 간 관계는 **논증 흐름을 명시적으로 드러내는 장치**다.

### 정의

| Relation Type | 설명 |
|---------------|------|
| `motivates` | A가 B를 동기 부여 |
| `supports` | A가 B를 지지 |
| `evaluates` | A가 B를 평가 |
| `extends` | A가 B를 확장 |
| `contrasts` | A가 B와 대조 |

### 적용 위치

- 슬라이드 분석: X
- 클러스터링: O (`RELATION_DEFINITIONS`, `getClusteringPrinciples()`)

---

## 7. Narrative Arc (논문 흐름)

### 원칙

> 의미 단위를 논문의 논리적 흐름에 맞게 배치하는 **5단계 모델**

### 정의

```
① 배경/동기 → ② 방법론 → ③ 결과 → ④ 해석/논의 → ⑤ 기여/향후
```

| Stage | ID | 설명 |
|-------|-----|------|
| 1단계 | `background` | 연구 배경, 동기, 문제 정의 |
| 2단계 | `method` | 연구 방법, 실험 설계, 데이터 |
| 3단계 | `result` | 실험 결과, 성능 비교 |
| 4단계 | `interpretation` | 결과 해석, 한계, 시사점 |
| 5단계 | `contribution` | 기여점, 향후 과제 |

### 적용 위치

- 슬라이드 분석: X
- 클러스터링: O (`NARRATIVE_STAGES`, `getClusteringPrinciples()`)

> 상세 분석 방법: `contexts/narrative-arc.md`

---

## 8. 적용 위치 요약

| 원칙 | 슬라이드 분석 | 클러스터링 |
|------|-------------|-----------|
| Semantic Unit 정의 | O | O |
| Category | O | O |
| Role | O | O |
| Reference Role | O | X |
| Relation Type | X | O |
| Narrative Arc | X | O |

---

## 9. 구현 현황

### 원칙 정의

| 파일 | 역할 |
|------|------|
| `lib/semantic-principles.ts` | 원칙 상수 및 프롬프트 함수 |

### 함수 매핑

| 함수 | 적용 위치 |
|------|----------|
| `getSlideAnalysisPrinciples()` | 슬라이드 분석 프롬프트 |
| `getClusteringPrinciples()` | 클러스터링 프롬프트 |
| `getAllPrinciples()` | 문서화용 전체 출력 |

### 상수 매핑

| 상수 | 용도 |
|------|------|
| `CORE_PRINCIPLES` | 핵심 철학, 질문 |
| `CATEGORY_DEFINITIONS` | Category 정의 |
| `ROLE_DEFINITIONS` | Role 정의 |
| `REFERENCE_ROLE_DEFINITIONS` | Reference Role 정의 |
| `RELATION_DEFINITIONS` | Relation Type 정의 |
| `NARRATIVE_STAGES` | 5단계 정의 |

---

## 10. 옵션별 이미지 분석 원칙

### Quick Draft

| 분석 요소 | 적용 |
|----------|------|
| 유형 분류 | O |
| 내용 설명 | 간략 (1문장) |
| 연구 역할 | X |
| 텍스트 관계 | X |
| 재현성 정보 | X |

### Standard

| 분석 요소 | 적용 |
|----------|------|
| 유형 분류 | O |
| 내용 설명 | 상세 (2-3문장) |
| 연구 역할 | O |
| 텍스트 관계 | X |
| 재현성 정보 | X |

### Premium

| 분석 요소 | 적용 |
|----------|------|
| 유형 분류 | O |
| 내용 설명 | 심층 (단락) |
| 연구 역할 | O (정량적 분석 포함) |
| 텍스트 관계 | **O** |
| 재현성 정보 | **O** |
| 교차 이미지 분석 | **O** |

---

## 11. 핵심 원칙

> Moonwriter의 Semantic Model은 **연구자의 사고 단위(Semantic Unit)를 중심으로, 구조(Category), 기능(Role), 근거(External Reference), 그리고 논증 흐름(Relation)을 분리해 표현하는 '지속적 집필을 위한 개념 지도'다.**
>
> - **Quick Draft**: 기본 유형 분류
> - **Standard**: 유형 + 역할 분석
> - **Premium**: 심층 분석 + 재현성 정보
>
> 원칙 수정 시 `lib/semantic-principles.ts`와 이 문서를 함께 수정하라.
