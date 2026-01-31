# 의미 단위 추출(Semantic Segmentation / Text Chunking) 방법론 조사

> 문서나 텍스트에서 의미 있는 단위를 추출하는 다양한 방법론에 대한 종합 조사 보고서

---

## 목차

1. [개요](#개요)
2. [규칙 기반 방법](#1-규칙-기반-방법)
3. [임베딩 기반 방법](#2-임베딩-기반-방법)
4. [LLM 기반 방법](#3-llm-기반-방법)
5. [하이브리드 방법](#4-하이브리드-방법)
6. [RAG 프레임워크별 Chunking 방법](#5-rag-프레임워크별-chunking-방법)
7. [실무 권장사항](#6-실무-권장사항)
8. [도구 및 라이브러리 요약](#7-도구-및-라이브러리-요약)
9. [참고문헌](#8-참고문헌)

---

## 개요

Text Chunking(텍스트 청킹)은 대용량 문서를 LLM이나 검색 시스템에서 효율적으로 처리할 수 있는 작은 단위로 분할하는 과정이다. RAG(Retrieval-Augmented Generation) 시스템에서 청킹의 품질은 검색 정확도와 최종 응답 품질에 직접적인 영향을 미친다.

### 핵심 고려사항
- **의미적 일관성**: 분할된 청크가 독립적으로 의미를 가져야 함
- **크기 최적화**: 너무 작으면 문맥 손실, 너무 크면 검색 정확도 저하
- **비용 효율성**: 임베딩 및 처리 비용과 품질 간의 균형

---

## 1. 규칙 기반 방법

### 1.1 Fixed-Size Chunking (고정 크기 청킹)

**핵심 아이디어**
- 텍스트를 일정한 문자 수 또는 토큰 수로 균등하게 분할
- 가장 단순하고 구현이 쉬운 방식

**장점**
- 구현이 매우 간단
- 예측 가능한 청크 크기
- 계산 비용이 낮음
- 일관된 처리 시간

**단점**
- 문장/단락 중간에서 분할될 수 있음
- 의미적 경계를 무시함
- 문맥 손실 발생 가능

**관련 도구**
- LangChain `CharacterTextSplitter`
- 직접 구현 (Python 문자열 슬라이싱)

---

### 1.2 Sentence-Based Chunking (문장 기반 청킹)

**핵심 아이디어**
- 문장 경계를 기준으로 텍스트 분할
- NLP 라이브러리의 문장 분리기 활용

**장점**
- 문법적으로 완전한 단위 유지
- 가독성이 좋음
- 규칙 기반으로 일관성 있음

**단점**
- 약어(U.K., Dr. 등) 처리 문제
- 문장 길이 변동성이 큼
- 비라틴 문자(한국어 등) 처리 어려움
- 구두점 없는 텍스트에서 성능 저하

**관련 도구/라이브러리**
| 도구 | 설명 |
|------|------|
| **spaCy Sentencizer** | 규칙 기반 문장 분리, 커스터마이즈 가능 |
| **spaCy SentenceRecognizer** | 통계 기반 문장 경계 탐지, 파서보다 빠름 |
| **NLTK punkt** | 사전 훈련된 문장 토크나이저 |
| **Stanza** | Stanford NLP의 문장 분할기 |

---

### 1.3 Paragraph/Structure-Based Chunking (단락/구조 기반 청킹)

**핵심 아이디어**
- 문서의 자연스러운 구조(단락, 제목, 섹션)를 기준으로 분할
- 마크다운, HTML 등 구조화된 문서에 적합

**장점**
- 저자의 의도된 구조 보존
- 주제별 일관성 유지
- 메타데이터 활용 가능

**단점**
- 단락 크기의 높은 변동성
- 구조가 없는 문서에는 적용 불가
- 긴 단락 처리 필요

**관련 도구**
- LangChain `HTMLHeaderTextSplitter`
- LangChain `MarkdownHeaderTextSplitter`
- Docling `HierarchicalChunker`

---

### 1.4 Recursive Character Text Splitting (재귀적 문자 분할)

**핵심 아이디어**
- 계층적 구분자 목록(`["\n\n", "\n", " ", ""]`)을 순차적으로 적용
- 단락 → 문장 → 단어 순서로 분할 시도
- 의미적으로 관련된 텍스트를 최대한 함께 유지

**장점**
- 고정 크기와 구조 기반의 절충안
- LangChain에서 가장 권장되는 방식
- 유연한 파라미터 조정 가능
- 다양한 문서 유형에 적용 가능

**단점**
- 최적 파라미터 튜닝 필요
- 완전한 의미적 일관성 보장 안 됨

**관련 도구**
- LangChain `RecursiveCharacterTextSplitter` (가장 권장)
- 권장 설정: 400-512 토큰, 10-20% 오버랩

---

### 1.5 Sliding Window Chunking (슬라이딩 윈도우 청킹)

**핵심 아이디어**
- 고정 크기 청킹에 오버랩(중첩) 적용
- 연속 청크 간 일부 내용 공유로 문맥 연속성 보장

**장점**
- 청크 경계에서의 정보 손실 방지
- 문맥 연속성 유지
- 검색 정확도 향상

**단점**
- 저장 공간 증가 (중복 데이터)
- 처리 비용 증가
- 중복 결과 처리 필요

**권장 설정**
- 오버랩: 청크 크기의 10-20%
- 500 토큰 청크 → 50-100 토큰 오버랩

---

## 2. 임베딩 기반 방법

### 2.1 Semantic Chunking (의미론적 청킹)

**핵심 아이디어**
- 임베딩 유사도를 기반으로 문장/단락 간 의미적 거리 측정
- 유사도가 급격히 감소하는 지점에서 청크 경계 설정
- Greg Kamradt가 처음 제안한 5단계 청킹 중 가장 진보된 방식

**작동 방식**
1. 문서를 문장 단위로 분할
2. 각 문장과 인접 문장들을 그룹화하여 임베딩 생성
3. 연속 그룹 간 의미적 거리 계산
4. 임계값 초과 시 청크 경계 설정

**장점**
- 의미적으로 일관된 청크 생성
- 주제 전환 지점을 자동 탐지
- 문서 구조에 의존하지 않음

**단점**
- 임베딩 비용 발생
- 처리 속도가 느림
- 최적 임계값 튜닝 필요
- Vectara 2024 연구: 실제 문서에서 고정 크기 대비 뚜렷한 이점 없는 경우도 있음

**관련 도구**
| 도구 | 설명 |
|------|------|
| LlamaIndex `SemanticSplitterNodeParser` | buffer_size, breakpoint_percentile_threshold 파라미터 |
| LlamaIndex `SemanticDoubleMergingSplitterNodeParser` | 이중 병합 방식 |
| `semchunk` | 경량 고속 의미적 청킹 라이브러리, 85% 더 빠름 |
| `semantic-split` | SentenceTransformers + spaCy 기반 |

**핵심 파라미터 (LlamaIndex)**
```python
from llama_index.core.node_parser import SemanticSplitterNodeParser

splitter = SemanticSplitterNodeParser(
    buffer_size=1,  # 유사도 계산 시 포함할 인접 문장 수
    breakpoint_percentile_threshold=95,  # 경계 결정 임계값
    embed_model=embed_model
)
```

---

### 2.2 Max-Min Semantic Chunking

**핵심 아이디어**
- 의미적 유사도와 Max-Min 알고리즘을 결합
- 청크 내 일관성 최대화, 청크 간 차이 최대화

**장점**
- 전통적 방법보다 의미적 일관성 향상
- 적응적 경계 설정

**단점**
- 계산 복잡도 높음
- 구현 복잡성

**관련 논문**
- "Max–Min semantic chunking of documents for RAG application" (2025, Springer)

---

### 2.3 Late Chunking (지연 청킹)

**핵심 아이디어**
- 먼저 전체 문서를 긴 컨텍스트 임베딩 모델로 인코딩
- 토큰 임베딩 시퀀스를 생성한 후 청크로 분할
- 각 청크 임베딩은 토큰 임베딩의 평균 풀링으로 생성

**장점**
- 청크가 전체 문서 컨텍스트 정보 포함
- 기존 방식 대비 의미적 풍부함 향상

**단점**
- 긴 컨텍스트 임베딩 모델 필요
- 계산 비용 높음

**관련 논문**
- "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models" (arXiv, 2024)

---

## 3. LLM 기반 방법

### 3.1 Agentic Chunking (에이전트 청킹)

**핵심 아이디어**
- AI 에이전트가 동적으로 최적의 분할 방식 결정
- 콘텐츠 분석을 통해 실시간으로 청킹 전략 조정
- 메타데이터 자동 생성 및 추가

**장점**
- 문서 유형에 따른 적응적 처리
- 풍부한 메타데이터 생성
- 높은 검색 정확도

**단점**
- LLM API 비용 발생
- 처리 속도 느림
- 결과의 비결정성

**관련 도구**
- IBM watsonx.ai + LangChain 통합
- Chonkie `AgenticChunker`

---

### 3.2 Contextual Retrieval (Anthropic, 2024)

**핵심 아이디어**
- 각 청크에 문서 전체 맥락을 설명하는 접두사 추가
- Claude를 사용하여 청크별 맥락 설명 생성
- BM25와 임베딩을 결합한 하이브리드 검색

**작동 방식**
1. 문서를 일반적인 방식으로 청킹
2. 각 청크에 대해 LLM으로 맥락 설명 생성
3. 맥락 설명을 청크 앞에 추가
4. 확장된 청크를 임베딩 및 BM25 인덱싱

**장점**
- 검색 실패율 35% 감소 (5.7% → 3.7%)
- Contextual BM25 결합 시 49% 오류 감소
- 리랭킹 추가 시 67% 정확도 향상 (오류율 1.9%)
- Prompt Caching으로 비용 절감 ($1.02/백만 청크)

**단점**
- 초기 처리 비용 높음
- 문서 업데이트 시 재처리 필요

**관련 자료**
- [Anthropic Contextual Retrieval 공식 문서](https://www.anthropic.com/news/contextual-retrieval)

---

### 3.3 Proposition-Based Chunking (Dense X Retrieval)

**핵심 아이디어**
- 텍스트를 원자적(atomic), 자기완결적(self-contained) 명제로 분해
- 각 명제는 독립적인 사실을 표현
- 대명사를 완전한 엔티티 이름으로 대체

**명제의 특성**
- **Unique**: 텍스트 내 고유한 의미 단위
- **Atomic**: 더 이상 분할 불가능
- **Self-contained**: 해석에 필요한 모든 컨텍스트 포함

**장점**
- Recall@5 17-25% 상대적 향상 (EntityQuestions)
- LLM 응답 품질 향상 (관련 정보 밀도 최대화)
- 희귀 엔티티 질문에서 특히 효과적
- 최적 범위: 100-200 단어 ≈ 10 명제 ≈ 5 문장

**단점**
- 대규모 코퍼스에서 비용 매우 높음
- LLM 호출 필요
- 처리 시간 길음

**관련 논문/도구**
- "Dense X Retrieval: What Retrieval Granularity Should We Use?" (EMNLP 2024)
- LlamaIndex Dense X Retrieval LlamaPack
- Propositionizer (Flan-T5-large 파인튜닝 모델)

---

### 3.4 Meta-Chunking

**핵심 아이디어**
- LLM의 논리적 인식 능력을 활용하여 문서를 논리적으로 일관된 청크로 분할
- 청크 크기의 유연한 가변성 허용
- 아이디어의 완전성과 독립성 유지

**장점**
- 논리적 무결성 보존
- 동적 크기 조정
- 의미적 완결성 보장

**단점**
- LLM 추론 비용
- 처리 속도

**관련 논문**
- "Meta-Chunking: Learning Text Segmentation and Semantic Completion via Logical Perception" (2024)

---

### 3.5 PPL (Perplexity) Chunking

**핵심 아이디어**
- 각 문장의 perplexity를 컨텍스트 기반으로 계산
- PPL 분포 특성 분석으로 청크 경계 식별
- 작은 언어 모델로도 효과적 수행 가능

**장점**
- 모델 규모 의존성 감소
- 작은 LM으로도 적용 가능
- 효율적인 처리

**단점**
- PPL 임계값 튜닝 필요
- 특수 문서 유형에서 성능 변동

---

## 4. 하이브리드 방법

### 4.1 S² Chunking (Spatial-Semantic Hybrid)

**핵심 아이디어**
- 의미 분석과 공간 레이아웃 정보 통합
- 바운딩 박스(bbox) 좌표로 공간 관계 캡처
- 그래프 기반 모델 + 스펙트럴 클러스터링

**장점**
- 문서 레이아웃 인식
- 표, 그림과 캡션 연결 유지
- 다양한 문서 유형에 적응

**단점**
- 레이아웃 정보 추출 필요
- 구현 복잡성

**관련 논문**
- "S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis" (2025)

---

### 4.2 Hierarchical Text Segmentation + Clustering

**핵심 아이디어**
- 계층적 텍스트 분할과 클러스터링 결합
- 세그먼트 수준 + 클러스터 수준 벡터 표현 활용
- 추론 시 두 수준의 표현 모두 활용

**장점**
- NarrativeQA, QuALITY, QASPER에서 성능 향상
- 의미적으로 더 일관된 청크 생성

**관련 논문**
- "Enhancing Retrieval Augmented Generation with Hierarchical Text Segmentation Chunking" (2025)

---

### 4.3 Docling Hybrid Chunker

**핵심 아이디어**
- 문서 기반 계층적 청킹 위에 토큰화 인식 정제 적용
- 문서 구조 정보와 토큰 제한 결합

**장점**
- 문서 메타데이터 보존 (헤더, 캡션 등)
- 토큰 제한 준수
- 오픈소스, 로컬 실행 가능

**관련 도구**
- Docling `HybridChunker`

---

### 4.4 BM25 + Semantic Embedding + RRF

**핵심 아이디어**
- 키워드 검색(BM25)과 의미 검색(임베딩) 결합
- Reciprocal Rank Fusion(RRF)으로 결과 통합
- 레이어드 아키텍처: BM25 → 지능형 청킹 → 임베딩 → 하이브리드 퓨전

**장점**
- 정확한 키워드 매칭 + 의미적 유사성
- 상호 보완적 강점 활용
- 검색 정확도 극대화

---

## 5. RAG 프레임워크별 Chunking 방법

### 5.1 LangChain Text Splitters

| Splitter | 설명 | 용도 |
|----------|------|------|
| `CharacterTextSplitter` | 단일 구분자 기반 분할 | 간단한 텍스트 |
| `RecursiveCharacterTextSplitter` | 계층적 구분자 순차 적용 (권장) | 범용 |
| `HTMLHeaderTextSplitter` | HTML 헤더 계층 기반 | HTML 문서 |
| `MarkdownHeaderTextSplitter` | 마크다운 헤더 기반 | 마크다운 문서 |
| `PythonCodeTextSplitter` | Python 코드 구조 인식 | 코드 문서 |
| `SemanticChunker` (실험적) | 임베딩 유사도 기반 | 의미적 분할 필요 시 |

**핵심 파라미터**
- `chunk_size`: 최대 청크 크기
- `chunk_overlap`: 청크 간 중첩 크기
- `separators`: 분할 구분자 목록

---

### 5.2 LlamaIndex Node Parsers

| Parser | 설명 |
|--------|------|
| `SentenceSplitter` | 문장 경계 기반 분할 |
| `TokenTextSplitter` | 토큰 수 기반 분할 |
| `SemanticSplitterNodeParser` | 임베딩 유사도 기반 적응적 분할 |
| `SemanticDoubleMergingSplitterNodeParser` | 이중 병합 의미론적 분할 |
| `HierarchicalNodeParser` | 계층적 노드 생성 |

**SemanticSplitterNodeParser 예시**
```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model
)
nodes = splitter.get_nodes_from_documents(documents)
```

---

### 5.3 Unstructured.io

**청킹 전략**
| 전략 | 설명 |
|------|------|
| `basic` | 순차적 요소 결합, max_characters 준수 |
| `by_title` | 섹션 경계 보존 |
| `by_page` | 페이지 경계 보존 |

**핵심 파라미터**
- `max_characters` (기본값 500): 하드 최대 크기
- `new_after_n_chars`: 소프트 최대 크기
- `pdf_infer_table_structure=True`: 표 구조 추론
- `strategy="hi_res"`: 고해상도 레이아웃 분석

**특징**
- 레이아웃 인식 파싱
- 표 구조 보존
- 다양한 문서 형식 지원 (PDF, DOCX, HTML 등)

---

## 6. 실무 권장사항

### 6.1 청크 크기 가이드

| 청크 크기 | 적합한 용도 |
|-----------|-------------|
| 256-512 토큰 | 사실 기반 질문, 정의, 정책 문서, 고객 지원 |
| 512-1024 토큰 | 분석적 질문, 연구 논문, 대화형 AI |
| 페이지 단위 | 일관된 성능 필요 시 (NVIDIA 벤치마크 우승) |

### 6.2 문서 유형별 권장 전략

| 문서 유형 | 권장 전략 |
|-----------|-----------|
| 구조화된 문서 (보고서, 기사) | 의미론적/재귀적 청킹 |
| 코드/기술 문서 | 언어별 재귀적 청킹 |
| 혼합/비구조화 콘텐츠 | AI 기반/컨텍스트 강화 청킹 |
| PDF (복잡한 레이아웃) | Unstructured.io, Docling |
| 법률/계약 문서 | 슬라이딩 윈도우, 의미론적 청킹 |

### 6.3 시작점 권장 설정

```
- 방법: RecursiveCharacterTextSplitter
- 청크 크기: 400-512 토큰
- 오버랩: 10-20% (50-100 토큰)
```

의미론적/페이지 수준 청킹은 메트릭이 추가 성능과 비용을 정당화할 때만 이동

### 6.4 평가 지표

- **Hit Rate**: 관련 청크 검색 성공률
- **Precision@K**: 상위 K 결과 중 관련 결과 비율
- **Recall@K**: 전체 관련 청크 중 검색된 비율
- **Context Relevancy**: 검색된 컨텍스트의 질문 관련성

---

## 7. 도구 및 라이브러리 요약

### 7.1 프레임워크

| 이름 | 설명 | 링크 |
|------|------|------|
| **LangChain** | RAG 파이프라인 프레임워크, 다양한 TextSplitter 제공 | [LangChain Docs](https://python.langchain.com/) |
| **LlamaIndex** | 데이터 프레임워크, NodeParser 모듈 | [LlamaIndex Docs](https://docs.llamaindex.ai/) |
| **Unstructured.io** | 문서 변환/청킹 플랫폼 | [Unstructured GitHub](https://github.com/Unstructured-IO/unstructured) |
| **Docling** | IBM 오픈소스 문서 변환 도구 | [Docling Docs](https://docling-project.github.io/docling/) |

### 7.2 전문 라이브러리

| 이름 | 특징 | 설치 |
|------|------|------|
| **Chonkie** | 경량 고속, 32+ 통합, SIMD 가속 | `pip install chonkie` |
| **semchunk** | 빠른 의미론적 청킹, 프로덕션 검증 | `pip install semchunk` |
| **ai-chunking** | 마크다운 중심, 계층적/의미적 전략 | `pip install ai-chunking` |
| **chunkipy** | 토큰화 인식, 스마트 오버랩 | `pip install chunkipy` |
| **semantic-split** | SentenceTransformers + spaCy | `pip install semantic-split` |

### 7.3 NLP 기반 도구

| 이름 | 용도 |
|------|------|
| **spaCy** | 문장 분할, 의존성 파싱 |
| **NLTK** | Punkt 토크나이저, 문장 분할 |
| **Stanza** | Stanford NLP, 다국어 지원 |

---

## 8. 참고문헌

### 논문
- [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648) (EMNLP 2024)
- [Meta-Chunking: Learning Text Segmentation and Semantic Completion via Logical Perception](https://arxiv.org/html/2410.12788v3)
- [S2 Chunking: A Hybrid Framework for Document Segmentation](https://arxiv.org/html/2501.05485v1)
- [Late Chunking: Contextual Chunk Embeddings](https://arxiv.org/html/2409.04701v3)
- [Max–Min semantic chunking of documents for RAG](https://link.springer.com/article/10.1007/s10791-025-09638-7)
- [Enhancing RAG with Hierarchical Text Segmentation](https://arxiv.org/html/2507.09935v1)

### 블로그/가이드
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Pinecone Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Best Chunking Strategies for RAG in 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [LangChain Text Splitters Reference](https://reference.langchain.com/python/langchain_text_splitters/)
- [LlamaIndex Semantic Chunker](https://developers.llamaindex.ai/python/examples/node_parsers/semantic_chunking/)
- [Weaviate Chunking Strategies](https://weaviate.io/blog/chunking-strategies-for-rag)
- [DataCamp Chunking Strategies](https://www.datacamp.com/blog/chunking-strategies)

### 벤치마크
- NVIDIA 2024 Chunking Benchmark: 페이지 수준 청킹 0.648 정확도 달성
- Chroma Research: LLMSemanticChunker 0.919 리콜
- Vectara 2024: "Is Semantic Chunking Worth It?" 연구

---

*문서 작성일: 2026-01-31*
*최신 방법론 포함하여 업데이트됨*
