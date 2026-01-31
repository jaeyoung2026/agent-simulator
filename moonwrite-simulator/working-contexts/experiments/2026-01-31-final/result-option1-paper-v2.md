# Thermal-Aware Fault-Tolerant Walking Control for ToddlerBot

> **Quick Draft** - 빠른 구조 확인용

## Abstract

본 연구는 ToddlerBot 로봇의 열 관리를 통한 장기 안정성 향상 방법을 제안한다. 모터 과열로 인한 토크 감소 문제를 해결하기 위해 Two-resistor 열 모델과 FEMNet 기반 결함 상태 추론을 적용하였다.

## 1. Introduction

ToddlerBot의 walking RL 정책 실행 시, servo motor의 overload로 인한 토크 감소와 기능 정지로 19분 이후 Fall Count가 증가하는 문제가 발생한다.

로봇이 특정 task를 잘 수행하는 것도 중요하지만, 장기적인 안정성 또한 고려되어야 한다.

[작성 예정: 연구 동기와 기존 연구의 한계점 상세 기술]

## 2. Methods

### 2.1 Two-Resistor Thermal Model

모터의 발열량은 입력 전력에서 기계적인 출력을 빼어 구한다. 권선에서 발생하는 열손실이 주를 이룸.

![열 모델 다이어그램](이미지: 2저항 열모델 회로도)

### 2.2 FEMNet (Fault Estimation Network)

teacher-student 프레임워크인 Actuator Degradation Adaptation Transformer를 제안한다.

### 2.3 Online Learning

열 파라미터의 업데이트 필요성:
- 부착된 금속 부품으로의 열 방출
- 주변 온도 오차
- 모터의 열화 또는 손상

### 2.4 Thermal Controller

![Thermal Controller 수식](이미지: 제어기 수식)

[작성 예정: 제어 알고리즘 상세 설명]

## 3. Results

### 3.1 학습 결과

RTX 5080에서 3시간 43분 학습 완료.

![학습 결과](이미지: 학습 곡선 그래프)

### 3.2 성능 비교

Heat2Torque JAX 코드 속도 비교 결과 제시.

![성능 비교](이미지: 비교 차트)

[작성 예정: 정량적 성능 비교 결과]

## 4. Discussion

단순히 특정 이상 온도로 올라가면 동작을 멈추게 할 수 있지만, 냉각이 될 때까지 Agent가 멈춰야하는 문제점이 있다. 본 연구는 overload로 인해 예상한 토크가 부족한 상황을 극복하는 것에 초점.

[작성 예정: 한계점 및 향후 연구]

## 5. Conclusion

실시간 발열 인식 및 예측 프레임워크 개발과 발열 인식을 통한 장기 동작 계획을 제안하였다.

[작성 예정: 주요 기여 요약]

## References

[작성 예정]

---

**Writing Principles 적용**: 최소 (주제문만)
**이미지 처리**: 기본 유형 분류만 적용
