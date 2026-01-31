# Semantic Strategy: 의미 단위 추출 전략

> **삼분법 위치**: 전략 (Strategy)
>
> 이 문서는 슬라이드에서 Semantic Unit을 **어떻게 추출하고 처리하는가**를 다룹니다.
>
> **원칙 참조:**
> - Semantic Unit 정의, Category, Role: `contexts/semantic-principles.md`
> - 이미지 필터링: `contexts/image-principles.md`
>
> **동기화**: `app/api/review/slide/route.ts`와 동기화 유지

---

## 1. 개요

### 옵션별 해상도 (고정)

| 옵션 | 해상도 | Units/Slide | 용도 |
|------|--------|-------------|------|
| **Quick Draft** | Low | 1-2 | 빠른 스크리닝 |
| **Standard** | Medium | 1-3 | 일반 논문 작성 |
| **Premium** | Medium | 1-3 | 고품질 제출물 |

### 처리 흐름

```
슬라이드 → 콘텐츠 추출 → AI 분석 → ExtractedUnit → SemanticUnit
```

### 관련 API

| 엔드포인트 | 기능 |
|------------|------|
| `POST /api/review/slide` | 슬라이드 분석 → ExtractedUnit 추출 |

---

## 2. 슬라이드 분석 파이프라인

### 2.1 콘텐츠 추출

슬라이드에서 세 가지 유형의 콘텐츠를 추출합니다:

| 유형 | 추출 대상 | 처리 방식 |
|------|----------|----------|
| **텍스트** | shape.text.textElements | 문자열 연결 |
| **이미지** | element.image | 필터링 후 Base64 변환 |
| **테이블** | element.table | 마크다운 변환 |

### 2.2 이미지 필터링

이미지는 `lib/image-strategies.ts`의 규칙에 따라 필터링됩니다:

**제외 대상:**
- 회사/기관 로고
- 템플릿 장식 요소
- 슬라이드 번호, 워터마크
- 클립아트, 스톡 이미지

**포함 대상:**
- 실험 결과 그래프/차트
- 시스템 아키텍처 다이어그램
- 연구 방법론 플로우차트
- 데이터 시각화

> 상세: `contexts/image-principles.md`, `contexts/image-strategy.md`

### 2.3 AI 분석 (Gemini)

멀티모달 분석을 통해 ExtractedUnit을 추출합니다.

**모델:** `gemini-3-flash-preview`

**프롬프트 구성:**
1. 이미지 (inlineData)
2. 텍스트 + 테이블 (마크다운)
3. 추출 지침 (JSON 스키마)
4. 원칙 주입 (`getSlideAnalysisPrinciples()`)

**출력 형식:** JSON
```json
{
  "topics": [ExtractedUnit, ...],
  "visualAnalysis": "시각 자료 분석 텍스트"
}
```

---

## 3. 배치 처리

### 3.1 배치 요청

한 번에 여러 슬라이드를 분석할 수 있습니다:

```typescript
POST /api/review/slide
{
  "fileId": "presentation-id",
  "slideIndices": [0, 1, 2, 3, 4]  // 최대 10개
}
```

### 3.2 병렬 처리

각 슬라이드는 병렬로 분석됩니다:

```
slideIndices.map(async (idx) => analyzeOneSlide(...))
```

---

## 4. Rate Limiting

| 설정 | 값 | 설명 |
|------|-----|------|
| `MAX_REQUESTS_PER_MINUTE` | 10 | 분당 최대 배치 요청 |
| `MAX_BATCH_SIZE` | 10 | 한 배치당 최대 슬라이드 |
| `MAX_SLIDE_INDEX` | 50 | 최대 슬라이드 인덱스 |
| `GEMINI_TIMEOUT_MS` | 60000 | API 타임아웃 (60초) |

### 추적 방식

세션별 요청 추적 (메모리 기반):

```typescript
requestTracker = Map<sessionId, { count, windowStart }>
```

---

## 5. 이미지 저장

분석된 이미지는 Supabase에 저장됩니다:

### 저장 흐름

```
이미지 URL → fetchImageAsBase64 → generateImageId → saveImageToDb
```

### 저장 데이터

| 필드 | 설명 |
|------|------|
| `id` | `img-{hash}` |
| `user_id` | 사용자 ID |
| `data` | Base64 데이터 |
| `mime_type` | MIME 타입 |
| `original_url` | 원본 URL |

### LinkedImage 매핑

ExtractedUnit에 `linkedImages` 배열로 연결:

```typescript
{
  url: "원본 URL",
  imageId: "img-xxx",  // 저장된 이미지 ID
  role: "primary" | "supporting",
  description: "AI 생성 설명"
}
```

---

## 6. ExtractedUnit → SemanticUnit 변환

슬라이드 분석 결과(ExtractedUnit)를 저장용 SemanticUnit으로 변환합니다.

### 변환 시 추가되는 필드

| 필드 | 생성 방식 |
|------|----------|
| `id` | `semantic-{timestamp}-{index}` |
| `sources` | 출처 슬라이드 정보 배열 |
| `keyFindings` | SourcedItem으로 변환 (출처 포함) |
| `dataPoints` | SourcedItem으로 변환 (출처 포함) |
| `relations` | 초기값 `[]` (클러스터링 시 생성) |

### 출처 형식

```typescript
{
  fileName: "연구발표.pptx",
  fileId: "abc123",
  slideNumber: 5,
  thumbnailUrl: "..."
}
```

---

## 7. 에러 처리

### 슬라이드 분석 실패

실패한 슬라이드는 빈 결과와 에러 메시지를 반환:

```typescript
{
  slideIndex: 3,
  slideNumber: 4,
  contentSummary: { hasText: false, hasImages: false, hasTables: false },
  units: [],
  visualAnalysis: null,
  error: "분석 실패 메시지"
}
```

### Rate Limit 초과

429 응답과 함께 재시도 정보 제공:

```typescript
{
  error: "Rate limit exceeded",
  resetIn: 45000  // ms
}
```

---

## 8. 구현 현황

### 코드 파일

| 파일 | 역할 |
|------|------|
| `app/api/review/slide/route.ts` | API 구현체 |
| `lib/image-strategies.ts` | 이미지 처리 함수 |
| `lib/semantic-principles.ts` | 원칙 주입 함수 |

### 상수

| 상수 | 값 | 위치 |
|-----|-----|------|
| `MODEL_NAME` | `gemini-3-flash-preview` | `route.ts` |
| `GEMINI_TIMEOUT_MS` | 60000 | `route.ts` |
| `MAX_REQUESTS_PER_MINUTE` | 10 | `route.ts` |
| `MAX_BATCH_SIZE` | 10 | `route.ts` |
| `MAX_SLIDE_INDEX` | 50 | `route.ts` |

### 함수 매핑

| 함수 | 역할 |
|------|------|
| `analyzeOneSlide()` | 단일 슬라이드 분석 |
| `saveImageToDb()` | 이미지 DB 저장 |
| `checkRateLimit()` | Rate limiting 체크 |
| `getSlideAnalysisPrinciples()` | 원칙 프롬프트 생성 |

---

## 9. 핵심 원칙

> 슬라이드 분석 전략은 **멀티모달 AI를 활용한 의미 단위 추출 파이프라인**이다.
>
> - **해상도 고정**: Quick Draft=Low, Standard/Premium=Medium
> - **콘텐츠 추출**: 텍스트, 이미지, 테이블 분리 추출
> - **이미지 필터링**: 의미 있는 이미지만 선별
> - **AI 분석**: Gemini 멀티모달로 ExtractedUnit 추출
> - **배치 처리**: 병렬 분석으로 성능 최적화
>
> 전략 수정 시 `app/api/review/slide/route.ts`와 이 문서를 함께 수정하라.
