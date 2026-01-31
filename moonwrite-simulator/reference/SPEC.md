# Moonwriter

> **연구 리포트가 논문 초안이 됩니다. 슬라이드를 올리면 논문 구조가 보입니다.**

## 프로젝트 소개

**Moonwriter**는 정기 연구 리포트(슬라이드 형식)를 논문 구조로 변환하는 도구입니다. 이미 만든 연구 리포트를 활용해 논문의 진척도를 확인하고, 부족한 부분을 발견합니다.

## 관련 문서

| 문서 | 설명 |
|------|------|
| [AI Engineering Methodology](./contexts/ai-engineering-methodology.md) | **삼분법 (원칙/전략/출력) 기반 AI 엔지니어링 방법론** |
| [Tech Spec](./TECH.md) | 기술 스택, 아키텍처, API, 환경 변수 |
| [Semantic Model](./contexts/semantic-model.md) | 의미 단위 기반 논문 구조화 모델 (SemanticUnit, Role, Reference, Relation) |
| [Writing Principles](./contexts/writing-principles.md) | 연구와 집필의 역동적 통합 원칙 |
| [Clustering Principles](./contexts/clustering-principles.md) | 클러스터링 분석 원칙 (시간적 우선순위, Thesis-First, thesisConnection) |
| [Clustering Strategy](./contexts/clustering-strategy.md) | 클러스터링 처리 전략 (Direct vs Distributed) |
| [Cluster Result Format](./contexts/cluster-result-format.md) | 분석 결과 출력 형식 (ClusterResult, Gap 등) |
| [Image Principles](./contexts/image-principles.md) | 이미지 필터링 규칙, 역할 정의, AI 지침 |
| [Image Strategy](./contexts/image-strategy.md) | 이미지 처리 파이프라인, 저장/집계 전략 |

## 해결하는 문제

**연구 리포트가 논문 초안이 됩니다. 슬라이드를 올리면 논문 구조가 보입니다.**

기존 방식의 문제:
- 연구 종료 후 막대한 집필 부담
- 시간이 지나면 맥락과 통찰을 잊어버림
- 논문 작성 시점에서야 논리적 허점 발견

Moonwriter는 이미 만든 슬라이드를 분석하여 논문 구조로 변환합니다. 추가 작업 없이 **얼마나 진척되어 있고, 무엇을 더 해야 하는지** 바로 보여줍니다.

## 핵심 가치

- **이미 만든 자료 활용**: 정기 연구 리포트가 그대로 논문의 재료가 됩니다
- **진척도 시각화**: 5단계 논문 흐름에서 어디가 채워졌고 어디가 비어있는지 확인
- **부족한 부분 발견**: 논리적 비약이나 근거 부족 구간을 자동 식별

## 3가지 논문 생성 옵션

| 옵션 | 전략 | 해상도 (고정) | 용도 |
|------|------|--------------|------|
| **Quick Draft** | Direct | Low (1-2 units/slide) | 빠른 스크리닝 |
| **Standard** | Distributed | Medium (1-3 units/slide) | 일반 논문 작성 (권장) |
| **Premium** | Distributed | Medium (1-3 units/slide) | 고품질 제출물 |

### 옵션별 기능 비교

| 기능 | Quick Draft | Standard | Premium |
|------|-------------|----------|---------|
| Thesis 추출 | 자동 | CoT | 상세 + Sub-claims |
| thesisConnection | - | O | O (강도 포함) |
| 이미지 분석 | 유형만 | 유형+역할 | 심층+재현성 |
| Gap 분석 | - | 기본 (4) | 상세 (5+심각도) |
| Writing Principles | 최소 | 3개 | 6개 |
| 역개요 검증 | - | - | O |
| Self-Critique | - | - | O |
| 플레이스홀더 | - | 위치만 | 구체적 제안 |

### 모델 배분 전략

| 단계 | 모델 | 역할 |
|------|------|------|
| Step 1-3 | Flash (Sonnet) | 추출, 분류, 검증 |
| Step 4 | Pro (Opus) | 최종 품질 검증 |

> **비용 최적화**: Flash 85-90% + Pro 10-15%로 전체 Pro 대비 **84-87% 비용 절감**

## 이미지 분석 전략

슬라이드의 시각 자료(그래프, 다이어그램, 표 등)는 Gemini 멀티모달 API로 분석합니다.

**목표**: Narrative Arc 분석 결과 품질과 논문 초안 품질 향상

### 2단계 이미지 필터링

의미 없는 이미지(로고, 템플릿 장식)를 제거하기 위해 2단계 필터링을 적용합니다:

**1단계 - 프로그래밍 필터** (`lib/image-strategies.ts`):
- 크기 필터: 150pt 미만 이미지 제외
- 비율 필터: 가로/세로 비율 6:1 초과 제외 (배너/구분선)
- 위치 필터: 모서리/헤더/푸터 위치의 작은 이미지 제외
- 반복 필터: 30% 이상 슬라이드에 등장하는 이미지 제외 (로고)

**2단계 - Gemini AI 필터**:
- 프롬프트에서 의미 있는 이미지 기준 제공
- AI가 분석 가치 있는 이미지만 선별하여 linkedImages에 포함

### 이미지-의미 단위 연결 (linkedImages)

분석된 이미지는 관련 의미 단위에 연결됩니다:

```typescript
interface LinkedImage {
  url: string;                        // 원본 이미지 URL
  imageId?: string;                   // Supabase 저장 ID (img-{hash})
  role: 'primary' | 'supporting';     // 핵심 vs 보조
  description?: string;               // AI 생성 설명
}
```

### 이미지 저장

Google URL 만료에 대비해 이미지를 Supabase에 저장합니다:

- `lib/generationStore.ts`: 이미지 저장 함수 포함
- `components/StoredImage.tsx`: 저장된 이미지 우선 로드 컴포넌트
- 썸네일 자동 생성 (200px)
- 분석 완료 후 백그라운드에서 이미지 저장

## 서비스 핵심 개념

서비스는 Write-View 패턴으로 설계되어 있습니다.

| 단계 | 역할 | 설명 |
|------|------|------|
| **분석 (Write)** | 데이터 생성 | 기존 슬라이드를 의미 기준으로 분석하여 의미 단위로 변환 |
| **라이브러리 (View)** | 데이터 관리 | 생성된 의미 단위 검색, 필터, 편집, 삭제 |
| **논문 (Write)** | 콘텐츠 생성 | 의미 단위를 클러스터링하고 논문 초안 생성 |

**핵심 기술**: 데이터를 적절히 처리해서 Supabase DB에 저장하고 활용하는 것이 서비스의 핵심입니다.

### 프로젝트 구조

연구 데이터를 프로젝트 단위로 관리합니다.

```
projects
  ├── 1:N ──→ generations (분석 데이터)
  └── 1:N ──→ narrative_arcs (클러스터링 결과)
```

- **프로젝트**: 연구 주제별로 분석 데이터를 그룹화
- **프로젝트 선택**: 사이드바에서 프로젝트를 선택하면 해당 프로젝트 데이터만 표시
- **전체 보기**: 프로젝트 미선택 시 모든 데이터 표시
- **하위 호환성**: 기존 데이터는 프로젝트 없이도 사용 가능 (project_id nullable)

## 주요 기능

- 다중 데이터 소스 지원 (어댑터 패턴)
- Google 드라이브 연동 (폴더 구조 지원)
- 슬라이드 분석 파이프라인
- 분석 결과 Supabase DB 저장
- 배치 API로 병렬 분석 (10개 슬라이드 동시 처리)

## 데이터 소스 아키텍처

외부 데이터 소스를 추상화하여 새로운 소스(Notion, PDF 등)를 쉽게 추가할 수 있습니다.

### 어댑터 패턴

```
lib/datasources/
├── types.ts           # DataSourceAdapter 인터페이스
├── registry.ts        # 어댑터 레지스트리 (싱글톤)
├── index.ts           # 모듈 진입점 (자동 등록)
└── google-slides/
    └── adapter.ts     # 구글 슬라이드 어댑터
```

### 핵심 인터페이스

```typescript
interface DataSourceAdapter {
  type: DataSourceType;           // 'google-slides' | 'notion' | 'pdf' | 'markdown'
  displayName: string;            // 표시명
  browse(auth, options): Promise<BrowseResult>;      // 파일/폴더 목록
  extract(auth, itemId, options): Promise<ContentExtraction>;  // 콘텐츠 추출
  extractPages(auth, itemId, pageIndices): Promise<ContentPage[]>;  // 특정 페이지
  isAvailable(auth): boolean;     // 인증 가능 여부
}
```

### 통합 API

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /api/sources/browse?source={type}&folderId={id}` | 파일/폴더 목록 조회 |
| `GET /api/sources/{sourceType}/{itemId}` | 콘텐츠 추출 |

### 기존 API 호환성

기존 API(`/api/slides/*`, `/api/drive/*`)는 내부적으로 어댑터를 사용하여 동작합니다.

## 사용자 경험 흐름

분석과 라이브러리가 분리된 경험으로 설계되어 있습니다.

### 1단계: 분석 (Write)
슬라이드에서 의미를 분석하는 단계입니다.

- `/generation` 페이지에서 Google 드라이브 파일 선택
- 선택한 파일들의 슬라이드를 분석하여 의미 단위로 변환
- 완료 시 선택 UI 표시: 라이브러리 / 새 분석

### 2단계: 라이브러리 (View)
생성된 의미 단위를 관리하는 단계입니다.

- `/library` 페이지에서 의미 단위 라이브러리 조회
- **보기 모드**: 분석별 / 의미 단위별 전환
- **검색/필터**: 키워드 검색, 카테고리 필터링
- **편집**: 제목, 카테고리, 요약, 키워드, 역할 수정
- **삭제**: 개별 의미 단위 또는 분석 데이터 전체 삭제

### 3단계: 논문 (Write)
의미 단위를 활용하여 논문을 작성하는 단계입니다.

- `/paper` 페이지에서 논문 흐름 분석 및 초안 생성
- **클러스터링**: AI가 의미 단위를 5단계 논문 흐름으로 그룹화
- **논문 초안 생성**: 클러스터 기반 학술 논문 Markdown 생성
- **내보내기**: 복사 또는 Markdown 파일 다운로드

### 페이지 구조

| 경로 | 역할 | 설명 |
|------|------|------|
| `/dashboard` | 연구 현황 | 논문 흐름 시각화, 통계 |
| `/generation` | 분석 (Write) | 파일 선택 → 분석 → 저장 |
| `/library` | 라이브러리 (View) | 의미 단위 검색, 필터, 편집, 삭제 |
| `/paper` | 논문 (Write) | 클러스터링 → 논문 초안 생성 |
| `/projects` | 프로젝트 관리 | 프로젝트 생성, 편집, 삭제 |

### 레이아웃 구조

모든 페이지는 **사이드바 + 본문** 레이아웃을 사용합니다:

**사이드바 (왼쪽 고정, 240px)**:
- 로고 (Moonwriter)
- 프로젝트 선택기: 현재 프로젝트 표시, 클릭하여 프로젝트 전환/생성
- 세로 네비게이션: 현황, 분석, 라이브러리, 논문, 프로젝트 관리
- 현재 페이지 하이라이트
- 하단: 사용자 이메일, 로그아웃 버튼

### 대시보드 구성

대시보드는 논문 흐름 중심으로 연구 현황을 시각화합니다.

**기존 사용자:**
- **논문 흐름**: 최근 분석의 의미 단위를 5단계로 분류하여 표시
  - 배경/동기 → 방법론 → 결과 → 해석/논의 → 기여/향후
- Role 기반 분류 우선, Category로 폴백
- 각 단계별 의미 단위 목록 (최대 3개, 나머지는 "+N개 더")
- **통계**: 총 분석 개수, 의미 단위 개수

**신규 사용자:**
- 서비스 소개 및 5단계 흐름 설명
- 분석 시작은 사이드바 "분석" 메뉴 안내

## 의미 단위 (Semantic Unit)

> 상세 내용: [Semantic Model](./contexts/semantic-model.md)

각 슬라이드에서 **개별 의미 단위**를 분석합니다. 하나의 슬라이드에서 여러 의미 단위가 분석될 수 있습니다.

### 핵심 필드

| 필드 | 설명 |
|------|------|
| `title` | 제목 |
| `category` | 콘텐츠 분류 (introduction, method, result, discussion, conclusion, reference, other) |
| `roles` | 기능적 역할 (Background, MethodComponent, PrimaryResult 등) |
| `references` | 외부 논문 인용 정보 |
| `relations` | 다른 의미 단위와의 관계 |
| `linkedImages` | 연결된 이미지 (url, role, description) |

### 카테고리 분류 기준

- **introduction**: 연구 배경, 목적, 동기, 문제 정의
- **method**: 연구 방법, 실험 설계, 데이터셋, 모델 구조
- **result**: 실험 결과, 성능 비교, 데이터 분석
- **discussion**: 결과 해석, 한계점, 시사점
- **conclusion**: 결론, 기여점, 향후 과제
- **reference**: 참고문헌
- **other**: 표지, 목차, 감사 인사 등

## 논문 흐름 분석

> 상세 내용: [Semantic Model - Narrative Arc](./contexts/semantic-model.md#7-논문-흐름-분석-narrative-arc)

`/paper` 페이지에서 의미 단위들을 논문의 논리적 흐름에 맞게 분석합니다.

### 5단계 흐름

```
① 배경/동기 → ② 방법론 → ③ 결과 → ④ 해석/논의 → ⑤ 기여/향후
```

### AI 클러스터링 (2단계 분석)

**1단계 - Flash 모델**: 클러스터링, 흐름 분석, 관계 분석
- 의미 단위들을 의미적으로 그룹화하여 클러스터 생성
- 각 클러스터에 대표 제목, 설명, 핵심 인사이트 부여
- Role 기반으로 논문 단계에 배치
- 단계 간 연결 강도(strong/moderate/weak) 분석
- 클러스터 간 관계(supports, extends, contrasts 등) 분석

**2단계 - Pro 모델**: 간극 분석 검증
- 논리적 비약, 근거 부족, 방법론 정당화 검증
- major/minor 간극 식별 및 보완 제안

### 대규모 의미 단위 처리 (300개+)

> 상세 내용: [Large-Scale Clustering](./contexts/large-scale-clustering.md)

의미 단위가 300개를 초과하면 **분류-분석 분리 패턴**을 적용:

| 단계 | 모델 | 역할 |
|-----|------|------|
| 1단계 | Flash | 경량 분류 + 관계 맵 추출 (교차 참조, 카테고리 연결) |
| 2단계 | Flash (병렬) | 클러스터별 상세 분석 (원본 데이터 100% 활용) |
| 3단계 | Flash | 전체 흐름 통합 (관계 맵 + 대표 유닛 기반) |
| 4단계 | Pro | 간극 검증 (기존 로직 동일) |

**핵심 원칙**:
- 정보 손실 없음: 원본 데이터 100% 활용
- 정확한 관계 분석: 1단계에서 카테고리 간 연결 사전 추출
- 비용 최적화: Flash 최대 활용, Pro는 검증에만 사용

### 논문 초안 생성

클러스터링 완료 후 "논문 초안 생성" 버튼으로 학술 논문 초안을 Markdown 형식으로 생성합니다.

- Abstract, Introduction, Methods, Results, Discussion, Conclusion, References 섹션 자동 구성
- 의미 단위의 keyFindings, dataPoints를 활용한 구체적 내용
- 참고문헌 인용 자동 삽입
- **탭 전환**: 분석 결과 / 논문 초안 탭으로 자유롭게 전환
- **뷰 전환**: 미리보기(렌더링된 마크다운) / 원본(마크다운 텍스트) 전환
- **내보내기**: 복사, Markdown 다운로드, LaTeX 다운로드
- **공유**: 공개 링크 생성으로 분석 결과 공유 (읽기 전용)
- **자동 저장**: 분석 결과와 논문 초안이 브라우저에 저장되어 새로고침 후에도 복원

### API

| 엔드포인트 | 기능 |
|-----------|------|
| `POST /api/library/cluster` | 의미 단위 클러스터링 |
| `POST /api/paper/generate` | 논문 초안 생성 |
| `POST /api/share/create` | 공유 링크 생성 (인증 필요) |
| `GET /api/share/[token]` | 공유 데이터 조회 (인증 불필요) |
| `DELETE /api/share/[token]` | 공유 삭제 (인증 필요) |

### 공유 기능

분석 결과(Narrative Arc)를 공개 링크로 공유할 수 있습니다.

**공유 링크 생성**:
- 분석 결과 화면에서 "공유" 버튼 클릭
- 32자 고유 토큰 기반 URL 생성
- 선택적 만료 기간 설정

**공유 페이지** (`/share/[token]`):
- 로그인 없이 접근 가능 (읽기 전용)
- 분석 결과와 논문 초안 탭 제공
- 조회수 추적

**보안**:
- RLS(Row Level Security)로 소유자만 공유 관리
- Service Role 키를 통한 공개 조회 지원
- Supabase RPC 함수로 안전한 데이터 접근

## 성능 최적화

### 캐싱
슬라이드 콘텐츠(텍스트, 이미지 URL, 표)의 SHA-256 해시를 키로 사용하여 분석 결과를 Supabase DB에 저장합니다. 동일한 슬라이드를 다시 열면 LLM 호출 없이 즉시 결과를 표시합니다. 슬라이드 내용이 변경되면 해시가 달라져 새로 분석합니다.

### 배치 처리
클라이언트에서 캐시 미스인 슬라이드들을 10개씩 묶어서 한 번의 API 요청으로 전송합니다. 서버에서는 Promise.all로 Gemini API를 병렬 호출하여 처리 시간을 단축합니다.

### 파일 병렬 처리
여러 파일 분석 시 최대 3개 파일을 동시에 처리합니다. 제한된 동시성(Concurrency Pool) 패턴으로 Rate Limit 회피와 성능 향상을 동시에 달성합니다.

### 분석 데이터 저장
분석 데이터는 Supabase PostgreSQL에 6개 테이블로 저장됩니다.

- **projects**: 연구 프로젝트
  - `name`: 프로젝트 이름
  - `description`: 프로젝트 설명
  - `settings`: 프로젝트별 설정 (JSON)
- **generations**: 의미 단위 기반 분석 데이터
  - `project_id`: 연결된 프로젝트 ID (nullable)
  - `semantic_units`: 개별 의미 단위 목록 (linkedImages 포함)
  - `source_files`: 원본 파일 정보
  - `stats`: 통계 (슬라이드 수, 의미 단위 수, 카테고리별 분포)
- **slide_cache**: 슬라이드 콘텐츠 해시 기반 분석 결과 캐시
- **narrative_arcs**: Narrative Arc 분석 결과 및 논문 초안
  - `project_id`: 연결된 프로젝트 ID (nullable)
  - `cluster_result`: 클러스터링 결과 (클러스터별 이미지 포함)
  - `paper_markdown`: 생성된 논문 초안
  - `unit_ids_hash`: 의미 단위 ID 해시 (빠른 조회용)
- **shares**: 공유 링크 관리
  - `token`: 32자 고유 토큰
  - `narrative_arc_id`: 공유 대상 Narrative Arc ID
  - `expires_at`: 만료 시간 (선택)
  - `view_count`: 조회수
- **images**: 저장된 이미지
  - `id`: 이미지 ID (img-{URL해시})
  - `data`: Base64 인코딩된 이미지 데이터
  - `thumbnail_data`: 200px 썸네일 데이터
  - `original_url`: 원본 URL
  - `mime_type`: 이미지 MIME 타입

### 인증
Supabase Auth를 사용하여 Google OAuth 로그인을 지원합니다. Google Drive/Presentations API 접근을 위한 추가 스코프도 처리합니다.

## 디자인 원칙

학술 도구로서의 성격을 반영한 절제된 디자인을 적용합니다.

- **색상**: 차분한 stone 계열 팔레트, 최소한의 강조색
- **타이포그래피**: 명확한 계층 구조, 읽기 쉬운 폰트
- **레이아웃**: 정보 밀도를 높이되 여백으로 가독성 확보
- **인터랙션**: 미묘한 테두리 변화, 그림자 최소화
- **아이콘**: 이모지 대신 텍스트 레이블 사용

## 운영 모니터링

### 토큰 사용량 추적
Gemini API 호출 시 토큰 사용량을 자동으로 기록합니다. `lib/tokenUsage.ts` 모듈에서 세션별, 일별 통계를 제공합니다.

조회 API: `GET /api/usage?type=total|daily|session|recent`
