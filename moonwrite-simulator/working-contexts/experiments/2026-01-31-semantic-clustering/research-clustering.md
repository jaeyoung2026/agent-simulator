# 텍스트/문서 클러스터링 방법론 연구

> **목표**: 의미 단위들을 논문 구조(5단계: 배경 → 방법 → 결과 → 해석 → 기여)로 그룹화하는 다양한 방법 조사
>
> **작성일**: 2026-01-31

---

## 목차

1. [전통적 클러스터링](#1-전통적-클러스터링)
2. [임베딩 기반 클러스터링](#2-임베딩-기반-클러스터링)
3. [LLM 기반 분류](#3-llm-기반-분류)
4. [토픽 모델링](#4-토픽-모델링)
5. [하이브리드 방법](#5-하이브리드-방법)
6. [학술 문서 구조화 관련 연구](#6-학술-문서-구조화-관련-연구)
7. [방법론 비교 요약](#7-방법론-비교-요약)
8. [권장 접근법](#8-권장-접근법)

---

## 1. 전통적 클러스터링

### 1.1 K-Means

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 데이터를 K개의 클러스터로 분할, 각 클러스터의 중심점(centroid)과의 거리를 최소화하는 반복 알고리즘 |
| **장점** | - 단순하고 구현이 쉬움<br>- 대규모 데이터셋에서 빠른 처리 속도<br>- 구형(spherical) 클러스터에 효과적<br>- LLM 임베딩과 조합 시 가장 효율적 (평균 순위 2.4) |
| **단점** | - 클러스터 수 K를 사전에 지정해야 함<br>- 비구형 클러스터에 부적합<br>- 이상치(outlier)에 민감<br>- 클러스터 크기가 불균등할 때 성능 저하 |
| **도구/라이브러리** | scikit-learn (`KMeans`), faiss, scipy |
| **관련 논문** | Lloyd, S. (1982). "Least squares quantization in PCM" |

### 1.2 Hierarchical Clustering (계층적 클러스터링)

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 덴드로그램(dendrogram)을 통해 데이터의 계층적 구조를 표현, 병합(agglomerative) 또는 분할(divisive) 방식 |
| **장점** | - 클러스터 수 사전 지정 불필요<br>- 데이터의 계층적 관계 시각화 가능<br>- 다양한 연결 기준(single, complete, average, ward) 선택 가능<br>- 구형/타원형 클러스터에 적합 |
| **단점** | - 계산 복잡도 O(n²) ~ O(n³)로 대규모 데이터에 부적합<br>- 한번 병합/분할되면 되돌릴 수 없음<br>- 거리 측정 방법과 연결 기준 선택에 민감 |
| **도구/라이브러리** | scikit-learn (`AgglomerativeClustering`), scipy (`hierarchy`), fastcluster |
| **관련 논문** | Ward, J. H. (1963). "Hierarchical Grouping to Optimize an Objective Function" |

### 1.3 DBSCAN (Density-Based Spatial Clustering)

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 밀도 기반 클러스터링, 고밀도 영역을 클러스터로 인식하고 저밀도 영역을 노이즈로 처리 |
| **장점** | - 클러스터 수 사전 지정 불필요<br>- 임의의 형태(비구형) 클러스터 탐지 가능<br>- 노이즈/이상치 자동 식별<br>- 제품 리뷰 클러스터링에서 99.80% 정확도 달성 |
| **단점** | - ε(epsilon)과 MinPts 파라미터 튜닝 필요<br>- 밀도가 다른 클러스터에서 성능 저하<br>- 고차원 데이터에서 "차원의 저주" 문제 |
| **도구/라이브러리** | scikit-learn (`DBSCAN`), pyclustering |
| **관련 논문** | Ester et al. (1996). "A Density-Based Algorithm for Discovering Clusters" |

### 1.4 HDBSCAN (Hierarchical DBSCAN)

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | DBSCAN의 확장, 다양한 밀도의 클러스터를 처리하고 계층적 구조를 제공 |
| **장점** | - DBSCAN의 모든 장점 상속<br>- 가변 밀도 클러스터 처리 가능<br>- 더 직관적인 하이퍼파라미터<br>- BERTopic의 기본 클러스터링 알고리즘으로 채택 |
| **단점** | - DBSCAN보다 계산 비용이 높음<br>- min_cluster_size 파라미터 튜닝 필요 |
| **도구/라이브러리** | hdbscan, scikit-learn (v1.3+) |
| **관련 논문** | Campello et al. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates" |

---

## 2. 임베딩 기반 클러스터링

### 2.1 Sentence-BERT (SBERT) + 클러스터링

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | Siamese/Triplet 네트워크 구조로 BERT를 수정하여 의미적으로 유사한 문장이 가까운 벡터 공간에 위치하도록 학습 |
| **장점** | - 의미적 유사도 기반 클러스터링 가능<br>- BERT 대비 65시간 → 5초로 속도 대폭 향상<br>- STS 벤치마크에서 InferSent 대비 11.7점 향상<br>- GPU에서 초당 2,042 문장 처리 가능 |
| **단점** | - 긴 텍스트(토큰 제한 초과) 처리에 제한<br>- 도메인 특화 성능을 위해 파인튜닝 필요 |
| **도구/라이브러리** | sentence-transformers, Hugging Face Transformers |
| **관련 논문** | [Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://arxiv.org/abs/1908.10084) |

**주요 Pre-trained 모델 비교:**

| 모델 | 차원 | 레이어 | 파라미터 | 특징 |
|------|------|--------|----------|------|
| **all-mpnet-base-v2** | 768 | 12 | 110M | 최고 품질, STS-B 87-88% |
| **all-MiniLM-L6-v2** | 384 | 6 | 22M | 5배 빠름, STS-B 84-85% |
| **all-MiniLM-L12-v2** | 384 | 12 | - | 품질/속도 균형 |

### 2.2 LLM 임베딩 + 클러스터링

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | GPT, Claude 등 대형 언어 모델의 임베딩을 추출하여 클러스터링에 활용 |
| **장점** | - 구조화된 언어의 미묘한 차이 포착에 우수<br>- 클러스터 순도(purity) 향상<br>- Silhouette 점수 개선<br>- GPT-3.5 Turbo가 5개 중 3개 메트릭에서 최고 성능 |
| **단점** | - API 비용 발생<br>- 처리 속도가 느림<br>- 임베딩 추출 제한 (일부 모델) |
| **도구/라이브러리** | OpenAI API (`text-embedding-ada-002`, `text-embedding-3-small/large`), Anthropic API, Cohere Embed |
| **관련 논문** | [Viswanathan et al. (2024). "Text Clustering with Large Language Model Embeddings"](https://arxiv.org/abs/2403.15112) |

### 2.3 도메인 특화 BERT 모델

| 모델 | 도메인 | 학습 데이터 | 특징 |
|------|--------|-------------|------|
| **SciBERT** | 과학 전반 | Semantic Scholar 1.14M 논문 (3.1B 토큰) | 과학 텍스트 분류에서 87% 정확도, BERT 대비 일관된 성능 향상 |
| **PubMedBERT** | 생물의학 | PubMed 논문 | 생의학 NLP 벤치마크에서 BERT 능가 |
| **BioBERT** | 생물의학 | PubMed + PMC | 생물의학 텍스트 마이닝에 최적화 |
| **LegalBERT** | 법률 | 법률 문서 | 법률 도메인 특화 |

**참고 논문**: [Beltagy et al. (2019). "SciBERT: A Pretrained Language Model for Scientific Text"](https://arxiv.org/abs/1903.10676)

---

## 3. LLM 기반 분류

### 3.1 Zero-Shot Classification

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 사전 학습된 LLM에게 레이블 없이 분류 작업을 수행하도록 프롬프트 |
| **장점** | - 학습 데이터 불필요<br>- 빠른 프로토타이핑 가능<br>- 새로운 카테고리에 유연하게 대응 |
| **단점** | - 전문 도메인에서 성능 저하<br>- 파인튜닝된 모델 대비 낮은 정확도<br>- 일관성 문제 (같은 입력에 다른 출력) |
| **도구/라이브러리** | OpenAI API, Anthropic Claude API, Hugging Face (`zero-shot-classification` pipeline) |
| **성능 참고** | GPT-4가 13개 병리 분류 작업 중 대부분에서 지도학습 모델과 동등하거나 우수한 성능 |

**프롬프트 예시:**
```
다음 문장이 논문의 어느 섹션에 해당하는지 분류하세요:
- 배경 (Background)
- 방법 (Methods)
- 결과 (Results)
- 해석 (Discussion)
- 기여 (Contribution)

문장: "{text}"
```

### 3.2 Few-Shot Classification

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 소수의 레이블된 예시를 프롬프트에 포함하여 분류 성능 향상 |
| **장점** | - Zero-shot 대비 12.6% 정확도 향상 (0.60 → 0.67)<br>- 최소한의 레이블 데이터로 성능 개선<br>- 도메인 적응 용이 |
| **단점** | - 예시 선택에 따른 성능 변동<br>- 컨텍스트 윈도우 제한<br>- 토큰 비용 증가 |
| **도구/라이브러리** | LangChain, OpenAI API, scikit-llm |
| **관련 논문** | [Chae & Davidson (2025). "Large Language Models for Text Classification: From Zero-Shot Learning to Instruction-Tuning"](https://journals.sagepub.com/doi/10.1177/00491241251325243) |

### 3.3 Chain-of-Thought (CoT) Prompting

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | "Let's think step by step" 등의 프롬프트로 단계별 추론을 유도하여 분류 성능 향상 |
| **장점** | - 복잡한 추론이 필요한 분류에 효과적<br>- Zero-shot + CoT로 13.44% 정확도 향상<br>- Few-shot + CoT로 추가 3.7% 향상<br>- GPT-4o + Few-shot + CoT: F1 84.54% 달성 |
| **단점** | - 토큰 사용량 증가로 비용 상승<br>- 응답 시간 증가<br>- 단순 분류 작업에는 과도할 수 있음 |
| **도구/라이브러리** | LangChain, DSPy, Guidance |
| **관련 논문** | [Kojima et al. (2022). "Large Language Models are Zero-Shot Reasoners"](https://arxiv.org/abs/2205.11916) |

**CoT 프롬프트 예시:**
```
다음 문장을 논문 섹션으로 분류하세요. 단계별로 생각해봅시다:

1. 먼저 문장의 핵심 내용을 파악합니다
2. 각 섹션의 특성과 비교합니다:
   - 배경: 연구 동기, 기존 연구, 문제 정의
   - 방법: 실험 설계, 데이터, 알고리즘
   - 결과: 수치, 측정값, 관찰
   - 해석: 의미 분석, 비교, 한계
   - 기여: 새로운 발견, 영향, 미래 방향
3. 가장 적합한 섹션을 선택합니다

문장: "{text}"
```

### 3.4 Many-Shot In-Context Learning

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 확장된 컨텍스트 윈도우(100K+ 토큰)를 활용하여 수백~수천 개의 예시로 학습 |
| **장점** | - 파인튜닝 없이 준-지도학습 수준 성능<br>- 작업 명세가 더 정확해짐<br>- LLM의 다재다능성 향상 |
| **단점** | - 매우 긴 컨텍스트 필요<br>- 비용 급증<br>- 긴 컨텍스트에서의 "lost in the middle" 문제 |
| **관련 논문** | [Agarwal et al. (2024). "Many-Shot In-Context Learning"](https://arxiv.org/abs/2404.11018) |

---

## 4. 토픽 모델링

### 4.1 LDA (Latent Dirichlet Allocation)

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 각 문서는 토픽들의 혼합, 각 토픽은 단어들의 확률 분포라는 생성 모델 가정 |
| **장점** | - 사전 학습 데이터 불필요<br>- 해석 가능한 토픽 생성<br>- 시간에 따른 토픽 변화 분석 가능<br>- 감성 분석과 결합 가능 |
| **단점** | - 토픽 수 K 사전 지정 필요<br>- Bag-of-Words 가정 (단어 순서 무시)<br>- 짧은 문서에서 데이터 희소성 문제<br>- 토픽 간 상관관계 모델링 불가 |
| **도구/라이브러리** | gensim, scikit-learn (`LatentDirichletAllocation`), MALLET, pyLDAvis |
| **관련 논문** | [Blei, Ng & Jordan (2003). "Latent Dirichlet Allocation"](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) |

### 4.2 BERTopic

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | BERT 임베딩 → UMAP 차원 축소 → HDBSCAN 클러스터링 → c-TF-IDF 토픽 표현 |
| **장점** | - 문맥 기반 의미 이해<br>- 모듈식 구조 (각 단계 커스터마이즈 가능)<br>- 이상치 자동 처리<br>- LLM 통합으로 토픽 레이블 자동 생성<br>- 지도학습 모드 지원 |
| **단점** | - LDA 대비 계산 비용 높음<br>- UMAP/HDBSCAN 하이퍼파라미터 튜닝 필요<br>- 작은 데이터셋에서 불안정할 수 있음 |
| **도구/라이브러리** | bertopic, 공식 문서: [bertopic.com](https://bertopic.com) |
| **관련 논문** | [Grootendorst (2022). "BERTopic: Neural topic modeling with a class-based TF-IDF procedure"](https://arxiv.org/abs/2203.05794) |

**BERTopic 파이프라인:**
```
Documents → Embeddings → Dimensionality Reduction → Clustering → Topic Representation
             (BERT)         (UMAP)                  (HDBSCAN)     (c-TF-IDF)
```

### 4.3 Top2Vec

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | Doc2Vec 임베딩 → UMAP → HDBSCAN, 토픽 수 자동 결정 |
| **장점** | - 토픽 수 자동 결정<br>- 단어와 문서가 같은 공간에 임베딩<br>- 의미적 검색 지원 |
| **단점** | - BERTopic 대비 덜 모듈화됨<br>- 커스터마이즈 제한적 |
| **도구/라이브러리** | top2vec |
| **관련 논문** | Angelov, D. (2020). "Top2Vec: Distributed Representations of Topics" |

---

## 5. 하이브리드 방법

### 5.1 LDA + BERT + Autoencoder

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | LDA의 토픽 분포, BERT의 문맥 임베딩, Autoencoder의 차원 축소를 결합 |
| **장점** | - 각 방법의 장점 결합<br>- BERT 단독 대비 향상된 클러스터링 품질<br>- TF-IDF의 중요도 평가 + BERT의 문맥 이해 |
| **단점** | - 복잡한 파이프라인<br>- 각 컴포넌트별 튜닝 필요<br>- 학습 시간 증가 |
| **관련 논문** | [MDPI (2025). "Analysis of Short Texts Using Intelligent Clustering Methods"](https://www.mdpi.com/1999-4893/18/5/289) |

### 5.2 ClusterFusion

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 임베딩 기반 클러스터링 + LLM을 보조 모듈로 활용하여 표현 강화 및 경계 조정 |
| **장점** | - 도메인 특화 데이터셋에서 큰 성능 향상<br>- OpenAI Codex에서 44.5% → 66.0% 정확도 (48% 상대적 향상)<br>- 토픽 요약 + 토픽 할당 2단계 접근 |
| **단점** | - LLM API 비용<br>- 복잡한 구현 |
| **관련 논문** | [Zhang et al. (2024). "ClusterFusion: Hybrid Clustering with Embedding Guidance and LLM Adaptation"](https://arxiv.org/html/2512.04350v1) |

### 5.3 LLMEdgeRefine

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | K-means로 초기 클러스터링 후, LLM을 사용하여 경계(edge) 포인트 정제 |
| **장점** | - 클러스터 경계의 모호한 샘플 처리 개선<br>- K-means의 효율성 + LLM의 세밀한 판단 결합 |
| **단점** | - 경계 임계값 설정 필요<br>- LLM 호출 비용 |
| **관련 논문** | [EMNLP 2024. "LLMEdgeRefine: Enhancing Text Clustering with LLM-based approaches"](https://aclanthology.org/2024.emnlp-main.1025.pdf) |

### 5.4 Text Clustering as Classification with LLMs

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | LLM 자체를 클러스터링 코어로 사용: (1) 토픽 요약 → (2) 토픽 할당 |
| **장점** | - 임베딩 기반 방법과 달리 직접적인 의미 이해<br>- 인간 해석 가능한 클러스터 생성 |
| **단점** | - 대규모 데이터셋에서 비용 문제<br>- 일관성 유지 어려움 |
| **관련 논문** | [ACM (2024). "Text Clustering as Classification with LLMs"](https://dl.acm.org/doi/pdf/10.1145/3767695.3769519) |

---

## 6. 학술 문서 구조화 관련 연구

### 6.1 IMRaD 구조 분류

**IMRaD**: Introduction, Methods, Results, and Discussion

| 연구 | 방법 | 성능 |
|------|------|------|
| Ribeiro et al. (2018) | SVM 분류기 | 81.30% 정확도 (PubMed Central 129개 논문) |
| HAN (Hierarchical Attention Network) | 섹션 제목 특징 + 계층적 어텐션 | 문서 구조 식별 향상 |
| DANN (Deep Attentive Neural Network) | Bi-RNN + Attention | 104개 주제 분류, F1 0.50-0.95 |

**관련 연구**: [Ribeiro & Yao (2018). "Discovering IMRaD Structure with Different Classifiers"](https://www.semanticscholar.org/paper/Discovering-IMRaD-Structure-with-Different-Ribeiro-Yao/be2ef84f950edf665924cbb7d24545eeb319dffd)

### 6.2 Argumentative Zoning

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 문장 수준에서 수사적 역할(비판, 지지, 방법 비교, 결과, 목표 등) 레이블링 |
| **장점** | - 세밀한 문서 구조 분석<br>- 문헌 리뷰 자동화 지원<br>- 정보 검색 향상 |
| **도구** | SciBERT, BioBERT를 활용한 시퀀스 레이블링 |

### 6.3 Scholarly Argumentation Mining (SAM)

**두 가지 하위 작업:**
1. **ADUR (Argumentative Discourse Unit Recognition)**: 논증 단위 인식
2. **ARE (Argumentative Relation Extraction)**: 논증 관계 추출

| 모델 | 특징 |
|------|------|
| LSTM/GRU + CRF | BIO 태깅 스킴으로 논증 단위 인코딩 |
| BERT variants | ELMo, Flair, BERT, BioBERT, SciBERT 등 활용 |
| Transfer Learning | 담화 파싱을 보조 작업으로 활용 |

**관련 연구**: [Argument Mining Workshop 2024](https://argmining-org.github.io/2024/)

### 6.4 Scientific Discourse Tagging

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 절(clause) 수준에서 담화 유형을 시퀀스 태깅 작업으로 모델링 |
| **접근법** | 풍부한 문맥 표현 학습 (Contextualized Deep Representation Learning) |
| **적용** | 과학 논문의 담화 구조 분석 |

---

## 7. 방법론 비교 요약

### 7.1 전반적 비교표

| 방법 | 데이터 요구량 | 계산 비용 | 해석 가능성 | 도메인 적응 | 정확도 |
|------|--------------|----------|------------|------------|--------|
| K-Means | 낮음 | 낮음 | 중간 | 낮음 | 중간 |
| HDBSCAN | 낮음 | 중간 | 중간 | 낮음 | 중간-높음 |
| SBERT + 클러스터링 | 중간 | 중간 | 중간 | 중간 | 높음 |
| LLM 임베딩 | 낮음 | 높음 | 높음 | 높음 | 높음 |
| Zero-Shot LLM | 없음 | 높음 | 높음 | 중간 | 중간 |
| Few-Shot LLM | 매우 낮음 | 높음 | 높음 | 높음 | 높음 |
| CoT + LLM | 매우 낮음 | 매우 높음 | 매우 높음 | 높음 | 매우 높음 |
| LDA | 중간 | 중간 | 높음 | 낮음 | 중간 |
| BERTopic | 중간 | 중간-높음 | 높음 | 높음 | 높음 |
| 하이브리드 | 중간 | 높음 | 높음 | 매우 높음 | 매우 높음 |

### 7.2 논문 구조 분류 특화 비교

| 방법 | 5단계 구조 적합성 | 장점 | 단점 |
|------|-----------------|------|------|
| **LLM Zero/Few-Shot** | 매우 높음 | 구조 정의만으로 즉시 적용, 유연한 카테고리 | 일관성, 비용 |
| **SciBERT + 분류기** | 높음 | 과학 도메인 최적화, 안정적 | 학습 데이터 필요 |
| **BERTopic (지도)** | 중간-높음 | 토픽과 구조 동시 파악 | 카테고리 수 제어 어려움 |
| **SBERT + K-Means** | 중간 | 빠름, 단순 | 경계 불명확 |
| **하이브리드 (임베딩 + LLM)** | 매우 높음 | 최고 정확도, 경계 정제 | 복잡성, 비용 |

---

## 8. 권장 접근법

### 8.1 논문 구조 5단계 분류를 위한 권장 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: 임베딩 생성                      │
├─────────────────────────────────────────────────────────────┤
│  Option A: SciBERT (과학 도메인 최적화)                      │
│  Option B: all-mpnet-base-v2 (범용, 고품질)                  │
│  Option C: OpenAI text-embedding-3-large (최고 품질, 유료)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Phase 2: 초기 클러스터링                   │
├─────────────────────────────────────────────────────────────┤
│  K-Means (K=5, 빠름) 또는 HDBSCAN (자동 클러스터 수)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Phase 3: LLM 정제                         │
├─────────────────────────────────────────────────────────────┤
│  Few-Shot + CoT 프롬프팅으로 경계 샘플 재분류                 │
│  또는 클러스터 → 섹션 매핑 검증                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 구현 예시 코드 (Python)

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from bertopic import BERTopic
import openai

# 1. 임베딩 생성
model = SentenceTransformer('allenai/scibert_scivocab_uncased')
# 또는 model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(sentences)

# 2. 클러스터링
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# 3. LLM으로 클러스터 레이블 매핑
def classify_with_llm(text, examples):
    prompt = f"""
    논문 문장을 다음 5가지 섹션 중 하나로 분류하세요:
    - 배경: 연구 동기, 기존 연구, 문제 정의
    - 방법: 실험 설계, 데이터, 알고리즘
    - 결과: 수치, 측정값, 관찰
    - 해석: 의미 분석, 비교, 한계
    - 기여: 새로운 발견, 영향, 미래 방향

    예시:
    {examples}

    분류할 문장: "{text}"

    단계별로 생각하고 가장 적합한 섹션을 선택하세요.
    """
    # API 호출
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. BERTopic 대안 (지도학습 모드)
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(sentences, embeddings)
```

### 8.3 선택 가이드

| 상황 | 권장 방법 |
|------|----------|
| **빠른 프로토타입** | Zero-Shot LLM 분류 |
| **비용 최소화** | SBERT + K-Means |
| **최고 정확도** | 하이브리드 (SciBERT 임베딩 + LLM 정제) |
| **대규모 데이터** | BERTopic 또는 SBERT + HDBSCAN |
| **레이블 데이터 있음** | SciBERT 파인튜닝 |
| **해석 가능성 중요** | Few-Shot + CoT |

---

## 참고 문헌 및 리소스

### 핵심 논문

1. [Sentence-BERT (2019)](https://arxiv.org/abs/1908.10084) - 문장 임베딩의 표준
2. [BERTopic (2022)](https://arxiv.org/abs/2203.05794) - 신경망 토픽 모델링
3. [SciBERT (2019)](https://arxiv.org/abs/1903.10676) - 과학 텍스트 특화 BERT
4. [Text Clustering with LLM Embeddings (2024)](https://arxiv.org/abs/2403.15112) - LLM 임베딩 클러스터링
5. [ClusterFusion (2024)](https://arxiv.org/html/2512.04350v1) - 하이브리드 클러스터링

### 도구 및 라이브러리

| 도구 | 용도 | 링크 |
|------|------|------|
| sentence-transformers | 문장 임베딩 | https://sbert.net |
| BERTopic | 토픽 모델링 | https://bertopic.com |
| scikit-learn | 전통적 클러스터링 | https://scikit-learn.org |
| hdbscan | 밀도 기반 클러스터링 | https://hdbscan.readthedocs.io |
| gensim | LDA 토픽 모델링 | https://radimrehurek.com/gensim |
| LangChain | LLM 체인 | https://langchain.com |
| scikit-llm | LLM 분류 | https://github.com/iryna-kondr/scikit-llm |

### 최신 연구 동향 (2024-2025)

- [Human-interpretable clustering with LLMs (Royal Society, 2025)](https://royalsocietypublishing.org/rsos/article/12/1/241692/92905/Human-interpretable-clustering-of-short-text-using)
- [A Comprehensive Survey on Deep Clustering (ACM, 2024)](https://dl.acm.org/doi/10.1145/3689036)
- [LLMs for Text Classification: Zero-Shot to Instruction-Tuning (SAGE, 2025)](https://journals.sagepub.com/doi/10.1177/00491241251325243)

---

*이 문서는 2026년 1월 31일 기준 최신 연구를 반영하여 작성되었습니다.*
