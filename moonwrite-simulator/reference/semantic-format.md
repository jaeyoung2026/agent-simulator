# Semantic Format: 의미 단위 출력 형식

> **삼분법 위치**: 출력 (Output)
>
> 이 문서는 Semantic Unit 관련 **타입과 인터페이스 구조**를 정의합니다.
>
> **원칙 참조:**
> - Category, Role, Relation 정의: `contexts/semantic-principles.md`
>
> **동기화**: `types/semantic.ts`와 동기화 유지

---

## 1. 개요

### 타입 계층

```
ExtractedUnit (분석 결과)
      ↓ 변환
SemanticUnit (저장용)
      ↓ 확장
TimeWeightedSemanticUnit (클러스터링용)
```

### 구현 위치

| 타입 | 파일 |
|------|------|
| 핵심 타입 | `types/semantic.ts` |
| 재export | `types/generation.ts` |

---

## 2. 기본 열거형

### ContentCategory

논문 구조 내 위치를 나타냅니다.

```typescript
type ContentCategory =
  | 'introduction'    // 연구 배경, 목적, 동기
  | 'method'          // 연구 방법, 실험 설계
  | 'result'          // 실험 결과, 데이터
  | 'discussion'      // 결과 해석, 논의
  | 'conclusion'      // 결론, 향후 과제
  | 'reference'       // 참고문헌
  | 'other';          // 기타 (표지, 목차 등)
```

### SemanticRole

기능적 역할을 나타냅니다 (복수 가능).

```typescript
type SemanticRole =
  | 'Background'         // 연구 배경/문제 제시
  | 'MethodComponent'    // 방법의 구성 요소
  | 'PrimaryResult'      // 핵심 실험 결과
  | 'Supporting'         // 보조/비교 결과
  | 'Interpretation'     // 결과 해석
  | 'LimitationEvidence' // 한계의 근거
  | 'Contribution'       // 기여 명시
  | 'FutureWorkSeed';    // 후속 연구 연결점
```

### ReferenceRole

외부 참조 역할을 나타냅니다.

```typescript
type ReferenceRole = 'Background' | 'Comparison' | 'Support' | 'Contrast';
```

### RelationType

Unit 간 관계 유형을 나타냅니다.

```typescript
type RelationType =
  | 'motivates'    // A가 B를 동기 부여
  | 'supports'     // A가 B를 지지
  | 'evaluates'    // A가 B를 평가
  | 'extends'      // A가 B를 확장
  | 'contrasts';   // A가 B와 대조
```

---

## 3. 보조 인터페이스

### ExternalReference

외부 논문 인용 정보입니다.

```typescript
interface ExternalReference {
  citation: string;           // "Kim et al., 2023"
  context: string;            // 인용 맥락
  referenceRole: ReferenceRole;
}
```

### LinkedImage

Unit에 연결된 이미지입니다.

```typescript
interface LinkedImage {
  url: string;                       // 원본 URL
  imageId?: string;                  // 로컬 저장 ID (img-{hash})
  role: 'primary' | 'supporting';    // 핵심 vs 보조
  description?: string;              // AI 생성 설명
}
```

### SemanticRelation

Unit 간 관계입니다.

```typescript
interface SemanticRelation {
  targetId: string;           // 대상 SemanticUnit ID
  relationType: RelationType;
  description?: string;       // 관계 설명
}
```

### SlideSource

슬라이드 출처 정보입니다.

```typescript
interface SlideSource {
  fileName: string;
  fileId: string;
  slideNumber: number;
  thumbnailUrl?: string;
}
```

### SourcedItem

출처가 포함된 항목입니다.

```typescript
interface SourcedItem {
  content: string;
  source: string;  // "파일명 #슬라이드번호" 형식
}
```

---

## 4. 핵심 인터페이스

### ExtractedUnit

슬라이드 분석 결과입니다. AI가 생성하는 원시 형태입니다.

```typescript
interface ExtractedUnit {
  title: string;                    // 토픽 제목
  category: ContentCategory;        // 연구 단계 분류
  summary: string;                  // 요약
  keyFindings: string[];            // 핵심 발견 (문자열)
  dataPoints: string[];             // 데이터 포인트 (문자열)
  keywords: string[];               // 관련 키워드
  roles: SemanticRole[];            // 기능적 역할
  references: ExternalReference[];  // 외부 논문 인용
  linkedImages?: LinkedImage[];     // 연결된 이미지
}
```

### SemanticUnit

저장용 완전한 의미 단위입니다.

```typescript
interface SemanticUnit {
  id: string;                       // semantic-{timestamp}-{index}
  title: string;
  category: ContentCategory;
  sources: SlideSource[];           // 출처 슬라이드 목록
  summary: string;
  keyFindings: SourcedItem[];       // 출처 포함
  dataPoints: SourcedItem[];        // 출처 포함
  keywords: string[];
  roles: SemanticRole[];
  references: ExternalReference[];
  relations: SemanticRelation[];    // 다른 Unit과의 관계
  linkedImages?: LinkedImage[];
}
```

### TimeWeightedSemanticUnit

시간 가중치가 포함된 Unit입니다. 클러스터링에서 최신성을 반영합니다.

```typescript
interface TimeWeightedSemanticUnit extends SemanticUnit {
  generationId: string;          // 출처 Generation ID
  generationCreatedAt: number;   // Generation 생성 시간
  sourceLastModified: number;    // 소스 파일 최신 수정 시간
  recencyScore: number;          // 최신도 점수 (0-1)
}
```

---

## 5. ID 형식

| 타입 | 형식 | 예시 |
|------|------|------|
| SemanticUnit | `semantic-{timestamp}-{index}` | `semantic-1704067200000-0` |
| LinkedImage | `img-{hash}` | `img-a1b2c3d4` |
| Generation | `generation-{timestamp}-{random}` | `generation-1704067200000-abc123` |

---

## 6. JSON 스키마 (AI 응답용)

슬라이드 분석 API에서 AI가 반환하는 형식:

```json
{
  "topics": [
    {
      "title": "BERT 모델 구조",
      "category": "method",
      "summary": "양방향 인코더 기반의 언어 모델 구조",
      "keyFindings": ["12개 레이어", "768 차원"],
      "dataPoints": ["파라미터 수: 110M"],
      "keywords": ["BERT", "Transformer", "NLP"],
      "roles": ["MethodComponent"],
      "references": [
        {
          "citation": "Devlin et al., 2019",
          "context": "BERT 논문",
          "referenceRole": "Background"
        }
      ],
      "linkedImageIndices": [
        {
          "imageIndex": 0,
          "role": "primary",
          "description": "BERT 아키텍처 다이어그램"
        }
      ]
    }
  ],
  "visualAnalysis": "그래프에서 학습률 0.001에서 최적 성능"
}
```

---

## 7. 핵심 원칙

> Semantic Format은 **분석 결과의 출력 형식**을 정의한다.
>
> - **ExtractedUnit**: AI가 생성하는 원시 형태
> - **SemanticUnit**: 출처 정보가 포함된 저장용 형태
> - **TimeWeightedSemanticUnit**: 시간 가중치가 포함된 클러스터링용 형태
>
> 분석 원칙은 `contexts/semantic-principles.md` 참조.
> 타입 수정 시 `types/semantic.ts`와 이 문서를 함께 수정하라.
