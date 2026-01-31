# Weekly Report 관점 슬라이드 분류 결과

> 분석 일시: 2026-01-31
> 분류 기준: 완료(Done), 진행중(InProgress), 계획(Planned), 이슈(Issue), 학습(Learning)

---

## 분류 요약

| 카테고리 | 슬라이드 수 |
|---------|-----------|
| 완료(Done) | 5 |
| 진행중(InProgress) | 3 |
| 계획(Planned) | 0 |
| 이슈(Issue) | 3 |
| 학습(Learning) | 9 |

---

## 상세 분류

### 1. WR 1_14.pptx - Slide 10
- **카테고리**: Learning (학습)
- **분류 근거**: 2저항 열 모델의 Optimization 방법론(MSE, Adam Optimizer, 로그 스케일 학습)에 대한 기술적 설명
- **신뢰도**: High

---

### 2. WR 9_23.pptx - Slide 37
- **카테고리**: Learning (학습)
- **분류 근거**: FEMNet(Failure Estimation Model + Modulation Model) 아키텍처 구조를 설명하는 학습 자료
- **신뢰도**: High

---

### 3. WR 8_5.pptx - Slide 7
- **카테고리**: Learning (학습)
- **분류 근거**: 휴머노이드 로봇의 전기 모터 열 제어(수냉, two-resister model) 논문/기술 내용 정리
- **신뢰도**: High

---

### 4. WR 07_14 - EunwooSong.pptx - Slide 14
- **카테고리**: Learning (학습)
- **분류 근거**: 모터의 입력 전력과 출력 전력 수식에 대한 이론적 설명
- **신뢰도**: High

---

### 5. WR 07_01 - EunwooSong.pptx - Slide 17
- **카테고리**: Done (완료)
- **분류 근거**: toddlerbot 모듈의 경로 문제를 pathlib를 이용해 절대경로로 수정 완료한 내용
- **신뢰도**: High

---

### 6. WR 07_01 - EunwooSong.pptx - Slide 8
- **카테고리**: Done (완료)
- **분류 근거**: MuJoCo 설치 테스트를 완료하고 실행 결과(qpos 출력)를 확인한 상태
- **신뢰도**: High

---

### 7. WR 9_23.pptx - Slide 53
- **카테고리**: InProgress (진행중)
- **분류 근거**: Thermal Model의 Online Learning 방법론 구현 중 - 하이퍼파라미터(Dt_data, N_seq, N_batch) 설정 단계
- **신뢰도**: Medium

---

### 8. WR 07_01 - EunwooSong.pptx - Slide 18
- **카테고리**: Done (완료)
- **분류 근거**: toddlerbot/utils/file_utils.py의 경로 문제를 수정한 코드 구현 완료
- **신뢰도**: High

---

### 9. WR 9_23.pptx - Slide 46
- **카테고리**: Learning (학습)
- **분류 근거**: Thermal Controller의 Motor Core Temperature 제어 수식에 대한 이론적 설명
- **신뢰도**: High

---

### 10. WR 07_14 - EunwooSong.pptx - Slide 17
- **카테고리**: Issue (이슈)
- **분류 근거**: 모터 과열로 인한 토크 감소(derating) 문제 - KOLLMORGEN 사례로 문제점 제시
- **신뢰도**: High

---

### 11. WR 1_14.pptx - Slide 14
- **카테고리**: Issue (이슈)
- **분류 근거**: 2축 모델 측정 결과의 Limitation - 시뮬레이션(sim)과 실측(obs) 간 오차 발생 문제
- **신뢰도**: High

---

### 12. WR 9_2.pptx - Slide 11
- **카테고리**: Done (완료)
- **분류 근거**: HeatState 결과 그래프(st_torque) 측정 완료 - right_knee_actuator 토크 데이터 수집
- **신뢰도**: High

---

### 13. WR 07_14 - EunwooSong.pptx - Slide 5
- **카테고리**: Issue (이슈)
- **분류 근거**: ToddlerBot walking RL 실행 시 servo motor overload로 19분 후 Fall Count 증가 문제 제기
- **신뢰도**: High

---

### 14. WR 9_23.pptx - Slide 28
- **카테고리**: InProgress (진행중)
- **분류 근거**: Data Collection 진행 중 - 12개 actuator 손상 시나리오별 20000개 trajectory 생성 작업
- **신뢰도**: Medium

---

### 15. WR 07_01 - EunwooSong.pptx - Slide 12
- **카테고리**: Done (완료)
- **분류 근거**: MuJoCo 환경 렌더링 샘플 코드 작성 및 테스트 완료
- **신뢰도**: High

---

### 16. WR 07_14 - EunwooSong.pptx - Slide 13
- **카테고리**: Learning (학습)
- **분류 근거**: 모터 발열량 계산 원리(입력 전력 - 기계적 출력) 및 Mujoco에서의 Torque/velocity 추출 방법 설명
- **신뢰도**: High

---

### 17. WR 9_23.pptx - Slide 51
- **카테고리**: Learning (학습)
- **분류 근거**: Proposed Thermal Model의 수식(P1~P5 파라미터 학습) 설명 - 2저항 열모델 확장 이론
- **신뢰도**: High

---

### 18. WR 9_2.pptx - Slide 6
- **카테고리**: InProgress (진행중)
- **분류 근거**: Torque limited 조건(1000~1389 steps)과 Normal 조건(0~500 steps) 비교 실험 진행 중
- **신뢰도**: Medium

---

### 19. WR 07_14 - EunwooSong.pptx - Slide 26
- **카테고리**: Learning (학습)
- **분류 근거**: DreamFLEX 논문 리뷰 - 험지에서의 결함(잠긴 관절, 약화된 모터) 대응 학습 방법
- **신뢰도**: High

---

### 20. WR 07_29.pptx - Slide 9
- **카테고리**: Done (완료)
- **분류 근거**: RTX 5080에서 10시간 학습 완료 - 5,120,000 steps와 302,080,000 steps 결과 비교
- **신뢰도**: High

---

## 카테고리별 슬라이드 목록

### Done (완료) - 6건
| 파일명 | 슬라이드 | 내용 요약 |
|--------|---------|----------|
| WR 07_01 - EunwooSong.pptx | 17 | toddlerbot 모듈 경로 문제 수정 |
| WR 07_01 - EunwooSong.pptx | 8 | MuJoCo 설치 테스트 완료 |
| WR 07_01 - EunwooSong.pptx | 18 | file_utils.py 경로 코드 수정 |
| WR 9_2.pptx | 11 | HeatState 결과 측정 |
| WR 07_01 - EunwooSong.pptx | 12 | MuJoCo 렌더링 샘플 코드 |
| WR 07_29.pptx | 9 | RTX 5080 학습 결과 |

### InProgress (진행중) - 3건
| 파일명 | 슬라이드 | 내용 요약 |
|--------|---------|----------|
| WR 9_23.pptx | 53 | Thermal Model Online Learning |
| WR 9_23.pptx | 28 | Data Collection (12 시나리오) |
| WR 9_2.pptx | 6 | Torque limited 비교 실험 |

### Issue (이슈) - 3건
| 파일명 | 슬라이드 | 내용 요약 |
|--------|---------|----------|
| WR 07_14 - EunwooSong.pptx | 17 | 모터 과열 토크 감소 문제 |
| WR 1_14.pptx | 14 | 2축 모델 측정 오차 |
| WR 07_14 - EunwooSong.pptx | 5 | Servo motor overload Fall |

### Learning (학습) - 8건
| 파일명 | 슬라이드 | 내용 요약 |
|--------|---------|----------|
| WR 1_14.pptx | 10 | Optimization (MSE, Adam) |
| WR 9_23.pptx | 37 | FEMNet 아키텍처 |
| WR 8_5.pptx | 7 | 모터 열 제어 (수냉) |
| WR 07_14 - EunwooSong.pptx | 14 | 입력/출력 전력 수식 |
| WR 9_23.pptx | 46 | Thermal Controller 수식 |
| WR 07_14 - EunwooSong.pptx | 13 | 모터 발열량 계산 |
| WR 9_23.pptx | 51 | Proposed Thermal Model |
| WR 07_14 - EunwooSong.pptx | 26 | DreamFLEX 논문 리뷰 |

### Planned (계획) - 0건
> 명시적으로 "예정", "계획", "TODO" 등의 표현이 있는 슬라이드가 없음

---

## 분류 방법론

1. **Done (완료)**: 코드 수정/구현 완료, 테스트 결과 확인, 실험 결과 도출 등 완결된 작업
2. **InProgress (진행중)**: 실험 진행 중, 데이터 수집 중, 파라미터 튜닝 중 등 현재 진행 상태
3. **Planned (계획)**: "예정", "TODO", "다음에" 등 미래 계획 명시
4. **Issue (이슈)**: 문제점, 한계(Limitation), 오류, 블로커 등 해결 필요 사항
5. **Learning (학습)**: 논문 리뷰, 이론/수식 설명, 아키텍처 설명, 참고 자료 정리

---

*분류 신뢰도 기준*
- **High**: 텍스트 내용과 이미지가 분류 기준에 명확히 부합
- **Medium**: 맥락상 추론이 필요하거나 여러 카테고리에 걸칠 수 있음
- **Low**: 정보 부족으로 분류 근거가 약함
