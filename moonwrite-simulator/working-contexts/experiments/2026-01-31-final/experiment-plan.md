# 최종 실험 계획: 논문 생성 옵션 비교

> **목적**: Moonwriter의 3가지 논문 생성 옵션을 설계하고 검증한다.
> **수행자**: 이 문서를 읽는 새로운 에이전트가 독립적으로 실험을 수행할 수 있도록 상세히 작성함.

---

## 1. 배경 요약

### 1.1 Moonwriter 개요

Moonwriter는 **연구 발표 슬라이드를 논문 초안으로 변환**하는 도구이다.

```
[슬라이드] → 의미 추출 → 클러스터링 → 논문 초안 생성
```

핵심 철학: **연구 중 집필 (Writing as you Research)** - 집필을 연구 과정의 인지적 도구로 활용

### 1.2 이전 실험 결과

#### 실험 1: 의미 추출 & 클러스터링 방법 비교

| 방법 | 의미 추출 | 클러스터링 | 비용 | 품질 |
|-----|----------|-----------|------|------|
| Baseline | LLM 직접 | LLM 직접 | 중간 | 좋음 |
| 임베딩 기반 | 유사도 그룹화 | 키워드 매칭 | **낮음** | 보통 |
| Proposition | 원자적 명제 | 명제별 LLM | 높음 | 좋음 |
| **하이브리드** | LLM 추출 | 키워드 초벌 + LLM 정제 | **중간-낮음** | **매우 좋음** |

**결론**: 하이브리드 방식이 44% 비용 절감 + 품질 유지

#### 실험 2: 관점 & 프롬프트 전략 비교

| 관점 | 분류 균형 | 적합도 |
|-----|----------|--------|
| 논문 구조 (IMRaD) | Method 60% 편중 | ★★★ |
| **연구 스토리** | 균형 분포 | **★★★★★** |
| 문제-해결 | 균형 분포 | ★★★★ |

| 프롬프트 전략 | 정확도 | 비용 |
|-------------|--------|------|
| Direct | 80% | 낮음 |
| **Chain-of-Thought** | 100% | 중간 |
| Self-Critique | 100% | 높음 |

**결론**: 연구 스토리 관점 + Chain-of-Thought 프롬프트가 최적

### 1.3 Writing Principles (논문 생성 원칙)

논문 생성 시 적용되는 핵심 원칙 (`reference/writing-principles.md`):

| 원칙 | 설명 | 적용 |
|-----|------|------|
| **핵심 주장 (Thesis)** | 모든 섹션의 이정표가 되는 하나의 선언적 문장 | 클러스터링 시 추출 |
| **원자적 기록 (To-clause)** | 모든 방법에 "~하기 위해" 목적절 추가 | Method 섹션 생성 시 |
| **플레이스홀더** | 빈 섹션에 작성 예정 사양 명시 | Gap 발견 시 |
| **골격 드래프팅** | 주제문 + 결론문만 먼저 작성 | 초안 구조화 시 |
| **이정표 (Signpost)** | 각 섹션에서 전체 목표와의 연결성 명시 | 섹션 시작부 |
| **글루 문장** | 섹션 끝에서 다음 섹션으로 연결 | 섹션 끝부분 |

---

## 2. 실험 목표

### 2.1 검증할 가설

1. **의미 추출 × 클러스터링 조합**이 논문 품질에 영향을 미친다
2. **3가지 옵션**으로 비용-품질 트레이드오프를 제공할 수 있다
3. **Writing Principles 적용 수준**에 따라 논문 완성도가 달라진다

### 2.2 3가지 논문 생성 옵션 설계

| 옵션 | 이름 | 대상 사용자 | 비용 | 품질 |
|-----|------|-----------|------|------|
| **옵션 1** | Quick Draft | 빠른 확인 필요 | 낮음 | 기본 |
| **옵션 2** | Standard | 일반 사용 | 중간 | 좋음 |
| **옵션 3** | Premium | 최고 품질 필요 | 높음 | 최고 |

---

## 3. 옵션별 상세 설계

### 3.1 옵션 1: Quick Draft (빠른 초안)

**목적**: 빠르게 논문 구조를 확인하고 싶은 사용자

#### 의미 추출 전략
```
방식: Direct LLM 추출 (단일 호출)
프롬프트: "이 슬라이드의 핵심 주장을 추출하세요"
출력: 슬라이드당 1개 SemanticUnit
```

#### 클러스터링 전략
```
방식: 키워드 기반 분류 (LLM 없음)
분류 기준: 연구 스토리 5단계
  - Planning: 계획, 목표, 로드맵
  - Setup: 설치, 환경, 설정
  - Execution: 구현, 개발, 실험
  - Finding: 결과, 데이터, 발견
  - Interpretation: 분석, 해석, 의미
```

#### Writing Principles 적용
```
적용 수준: 최소
- Thesis: 자동 추출 (첫 문장)
- 골격 드래프팅: 주제문만
- 이정표/글루 문장: 미적용
- 플레이스홀더: 빈 섹션 표시만
```

#### 예상 성능
- LLM 호출: 슬라이드 수 + 1회 (논문 생성)
- 처리 시간: 빠름
- 비용: 낮음
- 품질: 기본 (구조 확인용)

---

### 3.2 옵션 2: Standard (표준)

**목적**: 균형 잡힌 품질과 비용을 원하는 일반 사용자

#### 의미 추출 전략
```
방식: Chain-of-Thought LLM 추출
프롬프트:
  1. "이 슬라이드의 핵심 내용을 요약하세요"
  2. "연구 진행 과정에서 어떤 역할인지 설명하세요"
  3. "SemanticUnit으로 구조화하세요"
출력: 슬라이드당 1~3개 SemanticUnit
```

#### 클러스터링 전략
```
방식: 하이브리드 (키워드 초벌 + LLM 정제)
1단계: 키워드로 신뢰도 High/Low 분류
2단계: Low 신뢰도만 LLM 정밀 분류
분류 기준: 연구 스토리 5단계 + Thesis 연결
```

#### Writing Principles 적용
```
적용 수준: 표준
- Thesis: CoT로 추출 (핵심 질문 + 핵심 주장)
- 골격 드래프팅: 주제문 + 결론문
- 이정표: 각 섹션 시작부에 적용
- 글루 문장: 각 섹션 끝에 적용
- 플레이스홀더: Gap 분석 결과 반영
```

#### 예상 성능
- LLM 호출: 슬라이드 수 × 1.5 + 클러스터 수 × 0.5 + 1회
- 처리 시간: 중간
- 비용: 중간
- 품질: 좋음 (일반 사용 권장)

---

### 3.3 옵션 3: Premium (프리미엄)

**목적**: 최고 품질의 논문 초안이 필요한 사용자

#### 의미 추출 전략
```
방식: Self-Critique LLM 추출 + 이미지 분석
프롬프트:
  1. CoT로 SemanticUnit 추출
  2. "이 분류가 적절한지 검토하세요"
  3. "다른 해석 가능성이 있다면?"
  4. 최종 확정
이미지: 멀티모달 분석으로 그래프/다이어그램 해석
출력: 슬라이드당 1~5개 SemanticUnit (상세)
```

#### 클러스터링 전략
```
방식: Thesis-First Distributed (4단계)
1단계: Thesis 추출 + 경량 분류
2단계: Thesis-Aware 클러스터 분석 (병렬)
3단계: 일관성 검증 + 흐름 통합
4단계: Pro 모델로 품질 검증

분류 기준: 연구 스토리 + 논문 구조 통합
  - Planning/Intro: 배경, 동기, 목표
  - Setup/Method: 환경, 도구, 방법론
  - Execution/Method: 구현, 실험 과정
  - Finding/Result: 데이터, 결과
  - Interpretation/Discussion: 해석, 한계, 의의
```

#### Writing Principles 적용
```
적용 수준: 전체
- Thesis: 핵심 질문 + 핵심 주장 + 핵심 증거 연결
- 원자적 기록: To-clause 자동 생성
- 골격 드래프팅: 주제문 + 본문 개요 + 결론문
- 이정표: 모든 섹션에 적용 + Thesis 연결
- 글루 문장: 모든 섹션 전환에 적용
- 플레이스홀더: Gap 분석 + 구체적 제안
- 역개요: 생성 후 Thesis 연결성 검증
```

#### 예상 성능
- LLM 호출: 슬라이드 수 × 3 + 클러스터 수 × 2 + Pro 검증 1회
- 처리 시간: 느림
- 비용: 높음
- 품질: 최고 (중요 문서용)

---

## 4. 실험 설계

### 4.1 실험 데이터

```
위치: /working-contexts/experiments/2026-01-31-exp2/
파일:
  - all-slides-data.json: 275개 슬라이드 (텍스트 + 이미지 메타데이터)
  - images/: 132개 필터링된 이미지
  - samples.json: 실험용 샘플 20개
```

### 4.2 실험 절차

#### 4.2.1 샘플 선정
```
기존 samples.json 사용 (20개 슬라이드)
- 이미지 포함 슬라이드: 12개
- 텍스트만 슬라이드: 8개
- 다양한 파일에서 선정
```

#### 4.2.2 각 옵션 실행
```
옵션 1 (Quick Draft):
  1. Direct 추출로 SemanticUnit 생성
  2. 키워드 분류로 5단계 배치
  3. 최소 Writing Principles로 논문 생성

옵션 2 (Standard):
  1. CoT 추출로 SemanticUnit 생성
  2. 하이브리드 분류로 5단계 배치
  3. 표준 Writing Principles로 논문 생성

옵션 3 (Premium):
  1. Self-Critique + 이미지 분석으로 SemanticUnit 생성
  2. Thesis-First Distributed로 5단계 배치
  3. 전체 Writing Principles로 논문 생성
```

#### 4.2.3 결과 비교
```
평가 기준:
1. 추출 품질: SemanticUnit 수, 완결성, 중복/누락
2. 분류 정확도: 5단계 배치 적절성
3. 논문 완성도:
   - Thesis 명확성
   - 섹션 구조
   - 이정표/글루 문장 자연스러움
   - 플레이스홀더 구체성
4. 비용: LLM 호출 수, 토큰 사용량
5. 시간: 처리 소요 시간
```

### 4.3 평가 매트릭스

| 평가 항목 | Quick Draft | Standard | Premium |
|----------|-------------|----------|---------|
| **추출 품질** | ? | ? | ? |
| **분류 정확도** | ? | ? | ? |
| **Thesis 명확성** | ? | ? | ? |
| **섹션 구조** | ? | ? | ? |
| **연결 자연스러움** | ? | ? | ? |
| **LLM 호출 수** | ? | ? | ? |
| **처리 시간** | ? | ? | ? |

---

## 5. 구현 가이드

### 5.1 옵션 1: Quick Draft 구현

#### 의미 추출 프롬프트
```
당신은 연구 슬라이드 분석 전문가입니다.

다음 슬라이드에서 핵심 주장을 추출하세요.

슬라이드 내용:
{slide_content}

{image_description if exists}

출력 형식 (JSON):
{
  "title": "핵심 주장 제목",
  "category": "planning|setup|execution|finding|interpretation",
  "summary": "한 문장 요약",
  "keywords": ["키워드1", "키워드2"]
}
```

#### 키워드 분류 규칙
```python
CATEGORY_KEYWORDS = {
    "planning": ["계획", "목표", "로드맵", "예정", "할 일", "plan", "goal"],
    "setup": ["설치", "환경", "설정", "구축", "install", "setup", "config"],
    "execution": ["구현", "개발", "실험", "진행", "implement", "develop"],
    "finding": ["결과", "데이터", "발견", "측정", "result", "data", "found"],
    "interpretation": ["분석", "해석", "의미", "한계", "analysis", "meaning"]
}

def classify_by_keywords(text):
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text.lower():
                scores[cat] += 1
    return max(scores, key=scores.get)
```

#### 논문 생성 프롬프트
```
다음 SemanticUnit들을 논문 초안으로 변환하세요.

Units:
{semantic_units_json}

논문 구조:
1. Abstract (1문단)
2. Introduction (Planning 단계 내용)
3. Methods (Setup + Execution 단계 내용)
4. Results (Finding 단계 내용)
5. Discussion (Interpretation 단계 내용)
6. Conclusion

각 섹션은 주제문으로 시작하세요.
빈 섹션은 [작성 예정]으로 표시하세요.
```

---

### 5.2 옵션 2: Standard 구현

#### 의미 추출 프롬프트 (Chain-of-Thought)
```
당신은 연구 슬라이드 분석 전문가입니다.

다음 슬라이드를 단계적으로 분석하세요.

슬라이드 내용:
{slide_content}

{image_description if exists}

## 1단계: 핵심 내용 요약
이 슬라이드가 전달하는 핵심 내용을 1-2문장으로 요약하세요.

## 2단계: 연구 과정에서의 역할
이 내용이 연구 진행 과정에서 어떤 역할을 하는지 설명하세요.
- Planning: 앞으로 할 계획
- Setup: 환경/도구 준비
- Execution: 실제 수행/구현
- Finding: 결과/데이터 발견
- Interpretation: 의미 해석/분석

## 3단계: SemanticUnit 구조화
위 분석을 바탕으로 SemanticUnit을 생성하세요.

출력 형식 (JSON):
{
  "reasoning": "분류 근거",
  "units": [
    {
      "title": "제목",
      "category": "카테고리",
      "summary": "요약",
      "role": "역할 (Background/MethodComponent/PrimaryResult/Interpretation 등)",
      "keywords": ["키워드"]
    }
  ]
}
```

#### 하이브리드 분류
```python
def hybrid_classify(units):
    high_confidence = []
    low_confidence = []

    for unit in units:
        # 키워드로 초벌 분류
        category, confidence = classify_with_confidence(unit)
        unit["initial_category"] = category

        if confidence == "high":
            unit["final_category"] = category
            high_confidence.append(unit)
        else:
            low_confidence.append(unit)

    # Low confidence만 LLM으로 정밀 분류
    if low_confidence:
        refined = llm_refine_classification(low_confidence)
        for unit in refined:
            unit["final_category"] = unit["refined_category"]

    return high_confidence + low_confidence
```

#### 논문 생성 프롬프트 (표준 Writing Principles)
```
다음 SemanticUnit들을 학술 논문 초안으로 변환하세요.

## Thesis (핵심 주장)
{thesis}

## Units by Stage
{units_by_stage_json}

## 적용할 Writing Principles

1. **이정표 (Signpost)**: 각 섹션 시작에서 Thesis와의 연결성 명시
   예: "앞서 제기한 [문제]를 해결하기 위해..."

2. **글루 문장 (Glue Sentence)**: 각 섹션 끝에서 다음 섹션으로 연결
   예: "이 결과가 의미하는 바는 다음 절에서 논의한다."

3. **골격 드래프팅**: 각 문단은 주제문 + 결론문 구조

4. **플레이스홀더**: 빈 내용은 [작성 예정: 구체적 내용] 형식

## 논문 구조
1. Abstract
2. Introduction (이정표로 시작)
3. Methods (To-clause 적용: "~하기 위해")
4. Results
5. Discussion
6. Conclusion (Thesis 재확인)

출력: Markdown 형식
```

---

### 5.3 옵션 3: Premium 구현

#### 의미 추출 프롬프트 (Self-Critique + 이미지)
```
당신은 연구 슬라이드 분석 전문가입니다.

다음 슬라이드를 심층 분석하세요.

슬라이드 내용:
{slide_content}

이미지:
{images - 실제 이미지 파일 첨부}

## 1단계: Chain-of-Thought 분석
[Standard와 동일한 3단계 분석]

## 2단계: 이미지 분석 (해당 시)
이미지가 있다면:
- 이미지 유형 (그래프/다이어그램/스크린샷/표)
- 이미지가 전달하는 핵심 정보
- 이미지가 분류에 미치는 영향

## 3단계: Self-Critique
위 분석이 적절한지 검토하세요:
- 다른 카테고리가 더 적절할 수 있는 이유가 있는가?
- 하나의 슬라이드에서 여러 SemanticUnit을 추출해야 하는가?
- 누락된 정보가 있는가?

## 4단계: 최종 확정
검토를 반영한 최종 SemanticUnit을 생성하세요.

출력 형식 (JSON):
{
  "initial_analysis": {...},
  "self_critique": "검토 내용",
  "final_units": [...]
}
```

#### Thesis-First Distributed 분류
```
[기존 clustering-strategy.md 참조]

1단계 (Flash): Thesis 추출 + 경량 분류
2단계 (Flash 병렬): Thesis-Aware 클러스터 분석
3단계 (Flash): 일관성 검증 + 흐름 통합
4단계 (Pro): 품질 검증
```

#### 논문 생성 프롬프트 (전체 Writing Principles)
```
다음은 고품질 학술 논문 생성 요청입니다.

## Thesis
{thesis - 핵심 질문 + 핵심 주장 + 핵심 증거}

## Clustered Units
{clusters_with_thesis_connection}

## Gap Analysis
{gap_analysis}

## 전체 Writing Principles 적용

1. **핵심 주장 (Thesis)**: 모든 섹션이 Thesis를 지지하는 방향으로 작성

2. **원자적 기록 (To-clause)**:
   Methods의 모든 문장에 "~하기 위해" 목적절 추가

3. **골격 드래프팅**:
   - 주제문: 문단의 핵심 주장 선언
   - 본문 개요: 지지 내용 요약
   - 결론문: 의미 정리 또는 다음 연결

4. **이정표 (Signpost)**:
   각 섹션 첫 문단에서 "앞서 제기한 [Thesis 요약]을 [이 섹션의 역할]하기 위해..."

5. **글루 문장 (Glue Sentence)**:
   각 섹션 마지막에 "이러한 [현재 섹션 결론]은 [다음 섹션]에서 논의할 [내용]의 기반이 된다."

6. **플레이스홀더**:
   Gap Analysis 결과를 반영하여 구체적 작성 예정 사양 명시
   예: "[작성 예정: attention 메커니즘 제거 시 성능 변화 측정. 예상 실험: base vs no-attention]"

7. **역개요 검증**:
   생성 후 각 문단의 주제문만 추출하여 Thesis 연결성 확인

## 논문 구조
[상세 구조 with 이정표/글루 문장 위치 명시]

출력: Markdown 형식 (고품질)
```

---

## 6. 실험 실행 가이드

### 6.1 사전 준비

```bash
# 데이터 확인
ls /working-contexts/experiments/2026-01-31-exp2/
# 예상 출력: all-slides-data.json, images/, samples.json

# 샘플 확인
cat /working-contexts/experiments/2026-01-31-exp2/samples.json | head -50
```

### 6.2 실험 실행

```
각 옵션에 대해:

1. samples.json에서 20개 슬라이드 로드
2. 해당 옵션의 의미 추출 전략 적용
3. 해당 옵션의 클러스터링 전략 적용
4. 해당 옵션의 Writing Principles로 논문 생성
5. 결과 저장:
   - result-option1-extraction.json
   - result-option1-clustering.json
   - result-option1-paper.md
   - (옵션 2, 3도 동일)
```

### 6.3 결과 비교

```
비교 문서 생성:
- comparison-extraction.md: 3가지 옵션의 추출 결과 비교
- comparison-clustering.md: 3가지 옵션의 분류 결과 비교
- comparison-paper.md: 3가지 옵션의 논문 품질 비교
- final-report.md: 최종 권장 사항
```

---

## 7. 예상 결과

### 7.1 옵션별 특성 예측

| 옵션 | 추출 단위 | 분류 정확도 | 논문 완성도 | 비용 |
|-----|----------|------------|------------|------|
| Quick Draft | 슬라이드당 1개 | 70-80% | 기본 구조 | 1x |
| Standard | 슬라이드당 1-3개 | 90-95% | 좋은 품질 | 2-3x |
| Premium | 슬라이드당 1-5개 | 95-100% | 최고 품질 | 5-7x |

### 7.2 사용 시나리오

| 시나리오 | 권장 옵션 |
|---------|----------|
| 빠르게 논문 구조 확인 | Quick Draft |
| 일반적인 논문 초안 작성 | Standard |
| 중요한 학회/저널 제출용 | Premium |
| 비용 제약이 있는 경우 | Quick Draft → 필요시 Premium |

---

## 8. 산출물

실험 완료 후 생성될 파일:

```
/working-contexts/experiments/2026-01-31-final/
├── experiment-plan.md          # 현재 문서
├── result-option1-extraction.json
├── result-option1-clustering.json
├── result-option1-paper.md
├── result-option2-extraction.json
├── result-option2-clustering.json
├── result-option2-paper.md
├── result-option3-extraction.json
├── result-option3-clustering.json
├── result-option3-paper.md
├── comparison-extraction.md
├── comparison-clustering.md
├── comparison-paper.md
└── final-report.md             # 최종 권장 사항
```

---

## 9. 시작 명령

이 문서를 읽은 에이전트는 다음 명령으로 실험을 시작하세요:

```
"최종 실험 시작"
```

실험 완료 후 final-report.md에 결과를 정리하세요.
