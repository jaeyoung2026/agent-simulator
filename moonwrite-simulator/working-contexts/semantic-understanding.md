# Semantic Model 이해 문서

> 이 문서는 Moonwriter 서비스의 Semantic Model을 분석하고 이해한 내용을 정리한 것입니다.

---

## 1. 삼분법 (Tripartite) 패턴

Moonwriter는 **원칙/전략/출력** 삼분법 패턴으로 문서와 코드를 구조화합니다.

| 계층 | 문서 | 코드 | 질문 |
|------|------|------|------|
| **원칙 (Principles)** | `semantic-principles.md` | `lib/semantic-principles.ts` | "무엇이 의미 단위인가?" |
| **전략 (Strategy)** | `semantic-strategy.md` | `app/api/review/slide/route.ts` | "어떻게 추출하는가?" |
| **출력 (Output)** | `semantic-format.md` | `types/semantic.ts` | "어떤 형식으로 저장하는가?" |

이 구조의 장점:
- **관심사 분리**: 개념 정의, 처리 방법, 데이터 형식이 명확히 분리됨
- **동기화 명시**: 각 문서가 어떤 코드와 동기화되어야 하는지 명시
- **독립적 변경**: 원칙을 바꾸지 않고 전략만 변경하거나, 전략을 바꾸지 않고 출력 형식만 변경 가능

---

## 2. 핵심 개념: Semantic Unit

### 정의

> **논문에서 하나의 문단(paragraph)을 구성할 수 있는 최소 의미 단위의 주장 또는 개념**

### 슬라이드 vs 의미 단위

| 기존 접근 | Moonwriter 접근 |
|----------|----------------|
| 슬라이드 1장 = 1개 데이터 | 슬라이드 1장 = N개 의미 단위 |
| 물리적 단위 | 의미적 단위 |
| 위치 기반 | 개념 기반 |

핵심 통찰: **슬라이드는 발표 단위이지만, 논문은 주장 단위**입니다. 하나의 슬라이드에 여러 주장이 있으면 별도의 Semantic Unit으로 분리해야 합니다.

---

## 3. 다층적 분류 체계

Semantic Unit은 두 가지 축으로 분류됩니다:

### Category (구조적 위치) - 단일 선택

"이 단위가 논문의 **어디에** 위치하는가?"

```
introduction → method → result → discussion → conclusion
```

### Role (기능적 역할) - 복수 선택

"이 단위가 **무엇을** 하는가?"

```
Background, MethodComponent, PrimaryResult, Supporting,
Interpretation, LimitationEvidence, Contribution, FutureWorkSeed
```

### 분리의 이유

하나의 result 섹션 내용이 `PrimaryResult + Interpretation` 두 가지 역할을 동시에 수행할 수 있습니다. Category만으로는 이런 복합적 의미를 표현할 수 없습니다.

---

## 4. 데이터 흐름 (파이프라인)

```
[슬라이드]
    ↓ 콘텐츠 추출 (텍스트, 이미지, 테이블)
    ↓ 이미지 필터링 (로고, 장식 제거)
    ↓ Gemini AI 분석
[ExtractedUnit]  ← AI가 생성하는 원시 형태
    ↓ 출처 정보 추가, ID 생성
[SemanticUnit]   ← 저장용 완전한 형태
    ↓ 시간 가중치 추가
[TimeWeightedSemanticUnit]  ← 클러스터링용 형태
```

### 타입 진화의 이유

| 타입 | 추가되는 정보 | 사용 시점 |
|------|-------------|----------|
| ExtractedUnit | - | AI 분석 직후 |
| SemanticUnit | id, sources, relations | DB 저장 |
| TimeWeightedSemanticUnit | recencyScore, generationId | 클러스터링 |

---

## 5. 외부 참조 처리

### 원칙

> 외부 논문은 Semantic Unit이 아니다. 항상 **우리 Unit을 설명·정당화·비교하기 위해 호출된 외부 증거**로만 존재한다.

### Reference Role

| Role | 설명 | 예시 |
|------|------|------|
| Background | 배경 제시 | "Kim et al.이 이 분야를 개척했다" |
| Comparison | 성능/방법 비교 | "기존 방법 대비 15% 향상" |
| Support | 주장 강화 | "Park et al.도 같은 결론을 보고했다" |
| Contrast | 반례/차별점 | "Lee et al.과 달리 우리는..." |

---

## 6. 관계와 논문 흐름

### Relation Type (Unit 간 관계)

| Type | 의미 | 예시 |
|------|------|------|
| motivates | A가 B를 동기 부여 | 문제 정의 → 해결책 제안 |
| supports | A가 B를 지지 | 실험 결과 → 주장 |
| evaluates | A가 B를 평가 | 분석 → 방법론 |
| extends | A가 B를 확장 | 기본 모델 → 개선 모델 |
| contrasts | A가 B와 대조 | 우리 방법 ↔ 기존 방법 |

### Narrative Arc (5단계 흐름)

```
① 배경/동기 → ② 방법론 → ③ 결과 → ④ 해석/논의 → ⑤ 기여/향후
```

클러스터링 시 Semantic Unit들이 이 5단계에 배치되어 논문 구조가 됩니다.

---

## 7. 적용 위치 매트릭스

| 개념 | 슬라이드 분석 | 클러스터링 |
|------|-------------|-----------|
| Semantic Unit 정의 | ✓ | ✓ |
| Category | ✓ | ✓ |
| Role | ✓ | ✓ |
| Reference Role | ✓ | ✗ |
| Relation Type | ✗ | ✓ |
| Narrative Arc | ✗ | ✓ |

슬라이드 분석에서는 개별 Unit을 추출하고, 클러스터링에서는 Unit 간 관계와 전체 흐름을 분석합니다.

---

## 8. 이미지 처리

### 2단계 필터링

1. **프로그래밍 필터** (`lib/image-strategies.ts`):
   - 크기, 비율, 위치, 반복 패턴 검사
   - 로고, 템플릿 장식 제거

2. **AI 필터** (Gemini):
   - 분석 가치 있는 이미지만 선별
   - linkedImages에 포함

### LinkedImage 구조

```typescript
{
  url: "원본 URL",
  imageId: "img-{hash}",      // Supabase 저장 ID
  role: "primary" | "supporting",
  description: "AI 생성 설명"
}
```

---

## 9. 핵심 통찰

### 이 모델이 해결하는 문제

1. **물리적 단위 → 의미적 단위**: 슬라이드가 아닌 주장 단위로 분석
2. **단일 분류 → 다층 분류**: Category + Role로 복합적 의미 표현
3. **독립 분석 → 관계 분석**: Unit 간 관계로 논증 흐름 표현
4. **연구 자료 → 논문 구조**: 5단계 Narrative Arc로 자동 구조화

### 설계 철학

> **Semantic Model은 연구자의 사고 단위(Semantic Unit)를 중심으로, 구조(Category), 기능(Role), 근거(External Reference), 그리고 논증 흐름(Relation)을 분리해 표현하는 '지속적 집필을 위한 개념 지도'다.**

---

## 10. 코드-문서 동기화

| 문서 | 동기화 대상 코드 |
|------|-----------------|
| `semantic-principles.md` | `lib/semantic-principles.ts` |
| `semantic-strategy.md` | `app/api/review/slide/route.ts` |
| `semantic-format.md` | `types/semantic.ts` |

수정 시 반드시 문서와 코드를 함께 수정해야 합니다.
