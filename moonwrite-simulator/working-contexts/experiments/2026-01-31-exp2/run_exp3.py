#!/usr/bin/env python3
"""
Experiment 3: Standard + Pro (Medium Resolution)
- Model: Flash CoT extraction + Pro classification refinement
- Semantic segmentation: 1-3 per slide
- Image analysis: Standard (type + role)
- Writing Principles: Standard
"""

import json
import os
import time
from pathlib import Path
import google.generativeai as genai
from PIL import Image

# API 설정
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 모델 설정
flash_model = genai.GenerativeModel('gemini-2.0-flash-001')
pro_model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')

# 실험 조건
CONDITION = "Standard + Pro (Medium)"
MODEL = "flash + pro"
RESOLUTION = "1-3 units/slide"

# Writing Principles (Standard)
WRITING_PRINCIPLES = """
## Writing Principles (Standard)
1. 학술적 정확성: 연구 내용을 정확하게 전달
2. 명확성: 핵심 개념을 명확히 설명
3. 논리적 구조: 연구의 흐름을 논리적으로 구성
4. 기술적 정밀성: 수식, 알고리즘, 방법론을 정확히 기술
"""

# 카테고리 분류 체계
CATEGORIES = {
    "Background": "연구 배경, 동기, 문제 정의",
    "RelatedWork": "관련 연구, 선행 연구 분석",
    "Method": "제안 방법론, 알고리즘, 모델 설계",
    "Implementation": "구현 세부사항, 코드, 환경 설정",
    "Experiment": "실험 설계, 실험 환경",
    "Result": "실험 결과, 성능 분석",
    "Discussion": "논의, 한계점, 향후 연구",
    "Contribution": "핵심 기여, 연구 성과"
}


def analyze_image(image_path):
    """이미지 유형 및 역할 분석 (Standard)"""
    try:
        img = Image.open(image_path)

        prompt = """이미지를 분석하여 다음 정보를 JSON으로 반환:
{
    "type": "diagram|chart|screenshot|equation|table|photo|other",
    "role": "illustration|result|architecture|data|process|other",
    "brief_description": "간략한 설명 (1문장)"
}"""

        response = flash_model.generate_content([prompt, img])
        text = response.text.strip()

        # JSON 추출
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text)
    except Exception as e:
        return {"type": "unknown", "role": "unknown", "brief_description": f"Error: {str(e)[:50]}"}


def extract_semantic_units_cot(slide, images_info):
    """CoT 3단계 방식으로 SemanticUnit 추출 (Flash)"""

    content = slide.get('content', '')
    filename = slide.get('filename', '')
    slide_num = slide.get('slide_number', 0)

    # 이미지 정보 포함
    image_context = ""
    if images_info:
        image_context = "\n\n[이미지 정보]\n"
        for img_info in images_info:
            image_context += f"- {img_info.get('type', 'unknown')} ({img_info.get('role', 'unknown')}): {img_info.get('brief_description', '')}\n"

    prompt = f"""
{WRITING_PRINCIPLES}

당신은 연구 슬라이드를 분석하여 의미 단위(Semantic Unit)를 추출하는 전문가입니다.

## 슬라이드 정보
- 파일: {filename}
- 슬라이드 번호: {slide_num}
- 내용:
{content}
{image_context}

## 작업: Chain-of-Thought 3단계 분석

### Step 1: 관찰 (Observation)
슬라이드의 주요 요소를 나열하세요:
- 제목/헤더
- 텍스트 내용
- 수식/코드
- 이미지/다이어그램

### Step 2: 해석 (Interpretation)
각 요소가 전달하는 의미를 분석하세요:
- 이 슬라이드의 핵심 메시지는?
- 연구의 어떤 부분을 설명하는가?
- 어떤 논리적 흐름을 따르는가?

### Step 3: 구조화 (Structuring)
1-3개의 SemanticUnit으로 정리하세요.

## 출력 형식 (JSON)
```json
{{
    "cot_reasoning": {{
        "observation": "관찰 내용",
        "interpretation": "해석 내용",
        "structuring": "구조화 결정 근거"
    }},
    "units": [
        {{
            "id": "unit_1",
            "content": "의미 단위의 핵심 내용",
            "initial_category": "카테고리 (Background/RelatedWork/Method/Implementation/Experiment/Result/Discussion/Contribution)",
            "confidence": 0.0-1.0,
            "evidence": "분류 근거"
        }}
    ]
}}
```

주의: 슬라이드당 1-3개의 unit만 추출하세요. 과도한 세분화를 피하세요.
"""

    try:
        response = flash_model.generate_content(prompt)
        text = response.text.strip()

        # JSON 추출
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text)
    except Exception as e:
        return {
            "cot_reasoning": {"error": str(e)},
            "units": [{
                "id": "unit_1",
                "content": content[:200] if content else "Unknown",
                "initial_category": "Discussion",
                "confidence": 0.3,
                "evidence": "Fallback extraction"
            }]
        }


def refine_classification_pro(units_batch, context):
    """Pro 모델로 분류 정제"""

    units_json = json.dumps(units_batch, ensure_ascii=False, indent=2)

    prompt = f"""
당신은 연구 문서 분류 전문가입니다. Flash 모델이 초기 분류한 SemanticUnit들을 검토하고 정제해주세요.

## 연구 컨텍스트
{context}

## 카테고리 정의
- Background: 연구 배경, 동기, 문제 정의
- RelatedWork: 관련 연구, 선행 연구 분석
- Method: 제안 방법론, 알고리즘, 모델 설계
- Implementation: 구현 세부사항, 코드, 환경 설정
- Experiment: 실험 설계, 실험 환경
- Result: 실험 결과, 성능 분석
- Discussion: 논의, 한계점, 향후 연구
- Contribution: 핵심 기여, 연구 성과

## 초기 분류 결과
{units_json}

## 작업
1. 각 unit의 분류가 적절한지 검토
2. 필요시 카테고리 수정
3. 낮은 confidence(<0.7)인 항목 특별 검토
4. 분류 근거 보강

## 출력 형식 (JSON)
```json
{{
    "refined_units": [
        {{
            "id": "원래 id",
            "content": "원래 content",
            "initial_category": "원래 분류",
            "refined_category": "정제된 분류",
            "was_refined": true/false,
            "refinement_reason": "수정 이유 (수정된 경우)",
            "final_confidence": 0.0-1.0
        }}
    ],
    "refinement_summary": {{
        "total_reviewed": N,
        "refined_count": N,
        "low_confidence_resolved": N
    }}
}}
```
"""

    try:
        response = pro_model.generate_content(prompt)
        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text)
    except Exception as e:
        return {
            "refined_units": units_batch,
            "refinement_summary": {
                "total_reviewed": len(units_batch),
                "refined_count": 0,
                "low_confidence_resolved": 0,
                "error": str(e)
            }
        }


def analyze_gaps_pro(all_units, thesis_context):
    """Pro 모델로 Gap 분석"""

    # 카테고리별 분포 계산
    category_counts = {}
    for unit in all_units:
        cat = unit.get('refined_category', unit.get('initial_category', 'Unknown'))
        category_counts[cat] = category_counts.get(cat, 0) + 1

    units_summary = json.dumps(category_counts, ensure_ascii=False)
    sample_units = all_units[:10] if len(all_units) > 10 else all_units
    sample_json = json.dumps(sample_units, ensure_ascii=False, indent=2)

    prompt = f"""
당신은 학술 논문 구조 분석 전문가입니다.

## 연구 컨텍스트
{thesis_context}

## 현재 SemanticUnit 분포
{units_summary}

## 샘플 Units
{sample_json}

## 작업: Gap 분석
논문 작성을 위해 부족한 부분을 식별하세요.

## 출력 형식 (JSON)
```json
{{
    "structural_gaps": [
        {{
            "category": "부족한 카테고리",
            "severity": "high/medium/low",
            "description": "어떤 내용이 부족한지",
            "suggestion": "보완 방안"
        }}
    ],
    "content_gaps": [
        {{
            "topic": "부족한 주제",
            "current_coverage": "현재 상태",
            "needed": "필요한 내용"
        }}
    ],
    "thesis_readiness": {{
        "score": 0-100,
        "assessment": "전체 평가",
        "priority_actions": ["우선 조치 1", "우선 조치 2"]
    }}
}}
```
"""

    try:
        response = pro_model.generate_content(prompt)
        text = response.text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text)
    except Exception as e:
        return {
            "structural_gaps": [],
            "content_gaps": [],
            "thesis_readiness": {
                "score": 0,
                "assessment": f"Error: {str(e)}",
                "priority_actions": []
            }
        }


def main():
    # 샘플 데이터 로드
    with open('/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-final/samples-extended.json', 'r') as f:
        samples = json.load(f)

    print(f"=== Experiment 3: {CONDITION} ===")
    print(f"총 샘플 수: {len(samples)}")

    all_units = []
    sample_extractions = []
    pro_refinement_stats = {
        "classification_refined": 0,
        "gaps_identified": 0
    }

    # 1단계: Flash로 SemanticUnit 추출
    print("\n[1단계] Flash CoT로 SemanticUnit 추출...")
    for i, slide in enumerate(samples):
        print(f"  Processing slide {i+1}/{len(samples)}", end="\r")

        # 이미지 분석
        images_info = []
        for img in slide.get('images', []):
            img_path = img.get('path', '')
            if img_path and os.path.exists(img_path):
                img_info = analyze_image(img_path)
                img_info['filename'] = img.get('filename', '')
                images_info.append(img_info)

        # CoT 추출
        extraction = extract_semantic_units_cot(slide, images_info)

        # 메타데이터 추가
        for unit in extraction.get('units', []):
            unit['source_file'] = slide.get('filename', '')
            unit['source_slide'] = slide.get('slide_number', 0)
            unit['images_count'] = len(images_info)
            all_units.append(unit)

        # 샘플 저장 (최대 10개)
        if len(sample_extractions) < 10:
            sample_extractions.append({
                "slide": {
                    "filename": slide.get('filename', ''),
                    "slide_number": slide.get('slide_number', 0),
                    "content_preview": slide.get('content', '')[:150] + "..." if len(slide.get('content', '')) > 150 else slide.get('content', '')
                },
                "extraction": extraction
            })

        # Rate limiting
        time.sleep(0.3)

    print(f"\n  총 추출된 units: {len(all_units)}")

    # 2단계: Pro로 분류 정제
    print("\n[2단계] Pro로 분류 정제...")

    # 배치로 나누어 처리
    batch_size = 15
    refined_units = []

    thesis_context = """
    연구 주제: Heat2Torque - 로봇 모터의 열 관리를 위한 시뮬레이션 프레임워크
    핵심 내용:
    - 모터 발열 시뮬레이션 (열 모델)
    - 토크 제한 예측 및 관리
    - 강화학습 기반 열 인식 제어
    - 장기 운용 안정성 향상
    """

    for i in range(0, len(all_units), batch_size):
        batch = all_units[i:i+batch_size]
        print(f"  Refining batch {i//batch_size + 1}/{(len(all_units)-1)//batch_size + 1}", end="\r")

        result = refine_classification_pro(batch, thesis_context)

        if 'refined_units' in result:
            for unit in result['refined_units']:
                if unit.get('was_refined', False):
                    pro_refinement_stats['classification_refined'] += 1
                refined_units.append(unit)

        time.sleep(1)  # Pro 모델 rate limiting

    print(f"\n  정제된 units: {len(refined_units)}")
    print(f"  분류 수정됨: {pro_refinement_stats['classification_refined']}")

    # 3단계: Gap 분석
    print("\n[3단계] Pro로 Gap 분석...")
    gap_analysis = analyze_gaps_pro(refined_units, thesis_context)

    pro_refinement_stats['gaps_identified'] = (
        len(gap_analysis.get('structural_gaps', [])) +
        len(gap_analysis.get('content_gaps', []))
    )
    print(f"  식별된 gaps: {pro_refinement_stats['gaps_identified']}")

    # 카테고리 분포 계산
    category_distribution = {}
    for unit in refined_units:
        cat = unit.get('refined_category', unit.get('initial_category', 'Unknown'))
        category_distribution[cat] = category_distribution.get(cat, 0) + 1

    # 결과 정리
    result = {
        "condition": CONDITION,
        "model": MODEL,
        "resolution": RESOLUTION,
        "total_units": len(refined_units),
        "avg_units_per_slide": round(len(refined_units) / len(samples), 2) if samples else 0,
        "sample_extractions": sample_extractions,
        "category_distribution": category_distribution,
        "pro_refinement_stats": pro_refinement_stats,
        "thesis": {
            "gap_analysis": gap_analysis,
            "readiness_score": gap_analysis.get('thesis_readiness', {}).get('score', 0)
        }
    }

    # 결과 저장
    output_path = '/Users/jaeyoungkang/workspace/moonwrite-simulator/working-contexts/experiments/2026-01-31-exp2/exp3-standard-pro-medium.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 결과 저장 완료: {output_path} ===")
    print(f"\n=== 실험 결과 요약 ===")
    print(f"조건: {CONDITION}")
    print(f"모델: {MODEL}")
    print(f"해상도: {RESOLUTION}")
    print(f"총 units: {result['total_units']}")
    print(f"슬라이드당 평균 units: {result['avg_units_per_slide']}")
    print(f"카테고리 분포: {json.dumps(category_distribution, indent=2)}")
    print(f"Pro 정제 통계:")
    print(f"  - 분류 수정: {pro_refinement_stats['classification_refined']}")
    print(f"  - Gap 식별: {pro_refinement_stats['gaps_identified']}")
    print(f"논문 준비도 점수: {result['thesis']['readiness_score']}")

    return result


if __name__ == "__main__":
    main()
