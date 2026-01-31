# Clustering Strategy: 클러스터링 처리 전략

Moonwriter에서 의미 단위를 클러스터링하는 처리 전략을 정의합니다.

> **원칙 참조:**
> - 시간적 우선순위, Thesis-First, thesisConnection 가이드: `contexts/clustering-principles.md`
> - 5단계, 관계 유형: `contexts/semantic-model.md`
> - Thesis, 이정표: `contexts/writing-principles.md`
>
> **동기화 대상:** `lib/clustering-strategies.ts`

---

## 1. 개요

### 배경

의미 단위가 많아지면 단일 LLM 호출로 처리 시 품질 저하 위험이 있습니다.

- **정보 손실**: 세부사항 누락 가능
- **관계 분석 부정확**: 카테고리 간 연결 놓침
- **비용 비효율**: 불필요하게 큰 프롬프트

### 핵심 목표

- **정보 손실 없음**: 원본 데이터 100% 활용
- **정확한 관계 분석**: 카테고리 간 연결 보존
- **비용 최적화**: Flash 모델 최대 활용
- **확장성**: 1000개 이상 유닛도 처리 가능

---

## 2. 옵션별 전략 매핑

### 3가지 옵션

| 옵션 | 전략 | 해상도 (고정) | 용도 |
|------|------|--------------|------|
| **Quick Draft** | Direct | Low (1-2 units/slide) | 빠른 스크리닝 |
| **Standard** | Distributed | Medium (1-3 units/slide) | 일반 논문 작성 |
| **Premium** | Distributed | Medium (1-3 units/slide) | 고품질 제출물 |

### Standard vs Premium 기능 차별화

| 기능 | Standard | Premium |
|------|----------|---------|
| Self-Critique | - | O |
| 역개요 검증 | - | O |
| Writing Principles | 3개 | 6개 |
| 플레이스홀더 | 위치만 | 구체적 제안 |
| 이미지 분석 | 유형+역할 | 심층+재현성 |
| Gap 심각도 | - | O |

### 모델 배분 (공통)

| 단계 | 모델 | 역할 |
|------|------|------|
| Step 1-3 | Flash (Sonnet) | 추출, 분류, 검증 |
| Step 4 | Pro (Opus) | 최종 품질 검증 |

> **비용 최적화**: Flash 85-90% + Pro 10-15% → **84-87% 비용 절감**

### Premium 추가 Pro 호출

| 추가 기능 | 모델 | Input | Output |
|----------|------|-------|--------|
| 역개요 검증 | Pro | 2,000 | 1,000 |
| Writing Principles 상세 | Pro | 1,500 | 800 |
| 상세 플레이스홀더 | Flash | 1,000 | 500 |

---

## 3. Direct 전략

### 처리 흐름

```
의미 단위 N개
       ↓
┌──────────────────────────────────┐
│ Flash: 전체 분석                  │
│   - 원칙: 시간적 우선순위         │
│   - 원칙: 5단계/관계 유형         │
│   - 원칙: Thesis/이정표           │
│                                  │
│   출력: ClusterResult            │
└──────────────────────────────────┘
       ↓
┌──────────────────────────────────┐
│ Pro: 간극 검증                    │
└──────────────────────────────────┘
```

### 적용 원칙

| 원칙 | 함수 | 적용 |
|-----|------|------|
| 시간적 우선순위 | `getDirectStrategyPrinciples()` | O |
| 5단계/관계 유형 | `getClusteringPrinciples()` | O |
| Thesis/이정표 | `getNarrativeArcPrinciples()` | O |
| Thesis-First | - | X (한 번에 처리) |
| thesisConnection 가이드 | - | X (자동 생성) |
| 일관성 검증 | - | X (단일 호출) |

---

## 4. Distributed 전략 (Thesis-First 패턴)

### 핵심 아이디어

> **Thesis-First**: 핵심 주장을 먼저 추출하고, 모든 후속 분석에 전파한다.

### 처리 흐름

```
의미 단위 N개
       ↓
┌──────────────────────────────────────────────────┐
│ 1단계: Thesis 추출 + 경량 분류 (Flash)            │
│                                                  │
│   원칙: 시간적 우선순위, Thesis-First, 5단계      │
│                                                  │
│   출력:                                          │
│     - thesis: 핵심 주장                          │
│     - clusters: 클러스터 할당                     │
│     - crossReferences: 유닛 간 참조              │
│     - categoryLinks: 카테고리 간 연결             │
└──────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────┐
│ 2단계: Thesis-Aware 클러스터 분석 (Flash 병렬)    │
│                                                  │
│   원칙: thesisConnection 가이드, Thesis/이정표    │
│                                                  │
│   입력: thesis + 클러스터 유닛 원본               │
│   출력: description, keyInsight, thesisConnection│
└──────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────┐
│ 3단계: 일관성 검증 + 흐름 통합 (Flash)            │
│                                                  │
│   원칙: 일관성 검증 기준, 5단계/관계 유형         │
│                                                  │
│   검증: thesisConnection ↔ thesis 연결           │
│   출력: flowAnalysis, relationAnalysis           │
└──────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────┐
│ 4단계: 논문 품질 검증 (Pro)                       │
│                                                  │
│   검증: thesis 일관성, 섹션 변환 가능성, 간극     │
│   출력: gapAnalysis, qualityIssues               │
└──────────────────────────────────────────────────┘
```

### 단계별 적용 원칙

| 단계 | 원칙 | 함수 |
|-----|------|------|
| 1단계 | 시간적 우선순위, Thesis-First | `getDistributedStep1Principles()` |
| 1단계 | 5단계 | `getClusteringPrinciples()` |
| 2단계 | thesisConnection 가이드 | `getDistributedStep2Principles()` |
| 2단계 | Thesis/이정표 | `getNarrativeArcPrinciples()` |
| 3단계 | 일관성 검증 기준 | `getDistributedStep3Principles()` |
| 3단계 | 5단계/관계 유형 | `getClusteringPrinciples()` |

### 단계 간 데이터 보존

> **중요**: 2단계 LLM은 `unitIds`를 응답에서 생략할 수 있다. 1단계에서 확정된 `unitIds`를 신뢰 소스로 사용한다.

| 필드 | 신뢰 소스 | 이유 |
|-----|---------|------|
| `unitIds` | **1단계** | 클러스터 멤버십은 1단계에서 확정, 2단계는 생략 가능 |
| `description` | 2단계 | 상세 분석 결과 |
| `keyInsight` | 2단계 | Thesis-Aware 인사이트 |
| `thesisConnection` | 2단계 (3단계 강화) | 핵심 주장 연결 |

```typescript
// lib/clustering-strategies.ts - 결과 조합 시 fallback 적용
const cluster = {
  ...
  unitIds: detail.unitIds || originalCluster.unitIds, // 1단계 fallback
  ...
};
```

---

## 5. Direct vs Distributed 비교

| 항목 | Direct | Distributed |
|-----|--------|-------------|
| Thesis 추출 | 암시적 | **명시적 (1단계)** |
| Thesis 전파 | 없음 | **모든 단계에 전달** |
| thesisConnection | 자동 생성 | **가이드 기반 생성** |
| 일관성 검증 | 없음 | **3단계에서 검증** |
| 품질 검증 | 간극만 | **간극 + 논문 품질** |
| 처리 시간 | 빠름 | 상대적으로 느림 |
| 토큰 비용 | 높음 (대규모 시) | **낮음 (최적화)** |

---

## 6. 경량 데이터 구조

### LightweightUnit (1단계용)

```typescript
interface LightweightUnit {
  id: string;
  title: string;
  category: ContentCategory;
  roles: SemanticRole[];
  keywords: string[];          // 최대 5개
  hasReferences: boolean;
  referenceHints: string[];    // 참조 논문 저자명
  keyFindingHint: string;      // keyFindings 첫 항목 요약 (50자)
  recency: "recent" | "older"; // 시간적 우선순위
}
```

---

## 7. 대표 유닛 선정 기준

```typescript
function selectRepresentativeUnits(cluster, units, timestamps) {
  return units
    .filter(u => cluster.unitIds.includes(u.id))
    .sort((a, b) => {
      // 1. 최신 유닛 우선
      const aTime = timestamps.get(a.id) || 0;
      const bTime = timestamps.get(b.id) || 0;
      if (Math.abs(aTime - bTime) > 86400000) return bTime - aTime;

      // 2. Primary Role 우선
      const aHasPrimary = a.roles?.includes("PrimaryResult") ? 1 : 0;
      const bHasPrimary = b.roles?.includes("PrimaryResult") ? 1 : 0;
      if (aHasPrimary !== bHasPrimary) return bHasPrimary - aHasPrimary;

      // 3. keyFindings 많은 순
      return (b.keyFindings?.length || 0) - (a.keyFindings?.length || 0);
    })
    .slice(0, 2);
}
```

---

## 8. 비용 분석

### 1000개 유닛 기준

| 전략 | 단계 | 모델 | 예상 토큰 | 비용 |
|-----|-----|------|----------|------|
| **Distributed** | 1단계 | Flash | ~40K | ~$0.004 |
| | 2단계 (10개 병렬) | Flash × 10 | ~150K | ~$0.015 |
| | 3단계 | Flash | ~30K | ~$0.003 |
| | 4단계 | Pro | ~50K | ~$0.05 |
| | **합계** | | ~270K | **~$0.07** |
| **Direct** | Flash 전체 | Flash | ~200K | ~$0.02 |
| | Pro 검증 | Pro | ~200K | ~$0.20 |
| | **합계** | | ~400K | **~$0.22** |

→ Distributed가 약 **70% 비용 절감** + 품질 향상

---

## 9. 구현 현황

### 코드 파일

| 파일 | 역할 |
|------|------|
| `lib/clustering-strategies.ts` | 전략 구현 |
| `lib/clustering-principles.ts` | 원칙 정의 (프롬프트 주입) |

### API

| 엔드포인트 | 변경사항 |
|------------|---------|
| `POST /api/library/cluster` | mode 파라미터로 전략 선택 |

### 상수

| 상수 | 값 | 위치 |
|-----|-----|------|
| `DEFAULT_DIRECT_THRESHOLD` | 500 | `clustering-strategies.ts` |
| `FLASH_MODEL` | `gemini-3-flash-preview` | `clustering-strategies.ts` |
| `PRO_MODEL` | `gemini-3-pro-preview` | `clustering-strategies.ts` |

---

## 10. 핵심 원칙

> 클러스터링 전략은 **옵션에 따라 처리 방식을 선택**한다.
>
> - **Quick Draft**: Direct 전략, Low 해상도 - 빠른 스크리닝
> - **Standard**: Distributed 전략, Medium 해상도 - 일반 논문 작성
> - **Premium**: Distributed 전략 + 추가 기능, Medium 해상도 - 고품질 제출물
>
> Standard와 Premium은 동일한 모델 배분 (Flash 1-3 + Pro 4)을 사용하되,
> Premium은 역개요 검증, Writing Principles 6개, Self-Critique 등 추가 기능으로 차별화한다.
>
> 전략 수정 시 `lib/clustering-strategies.ts`와 이 문서를 함께 수정하라.
> 원칙 수정 시 `lib/clustering-principles.ts`와 `contexts/clustering-principles.md`를 함께 수정하라.
