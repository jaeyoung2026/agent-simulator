# Thermal-Aware Adaptive Locomotion Control for Humanoid Robots: An Integrated Approach to Motor Temperature Management and Fault-Tolerant Walking

---

## Abstract

휴머노이드 로봇의 전기 모터는 지속적인 고출력 동작 시 열 축적으로 인한 성능 저하와 고장 위험에 노출된다. 본 논문은 확장된 열 모델과 온라인 학습 기반 열 제어기를 통해 모터의 코어 온도를 정확히 예측하고 제어하는 방법을 제안한다. 또한, FEMNet(Failure Estimation & Modulation Network) 기반 고장 인식 및 적응 시스템을 통해 액추에이터 고장 상황에서도 안정적인 보행을 유지할 수 있음을 보인다. 제안된 방법은 기존 2저항 열 모델을 P1-P5 파라미터로 확장하여 비선형 열 특성을 모델링하고, MSE 기반 제어 손실 함수를 통해 안전한 온도 범위 내에서 최대 토크를 활용한다.

**Keywords**: Humanoid Robot, Thermal Control, Motor Temperature, Fault-Tolerant Locomotion, Online Learning, FEMNet

---

## 1. Introduction

**[주제문]** 휴머노이드 로봇의 안정적인 장시간 운용을 위해서는 전기 모터의 열 관리와 고장 상황에 대한 적응 능력이 필수적이다.

### 1.1 Research Background and Motivation

휴머노이드 로봇의 전기 모터는 최대 성능을 발휘하기 위해 높은 전류를 사용하지만, 이는 필연적으로 열 발생을 동반한다. 전기 모터의 최대 성능을 안전하게 활용하기 위해서는 효과적인 열 관리 시스템이 필요하다. 수냉을 적용한 모터(예: ECpowermax30 200W 48V water-cooling)는 그래프에서 파란 점선으로 표시된 바와 같이 장시간 높은 전류를 유지할 수 있어 지속적인 고출력 운용이 가능해진다. 센서 정보로부터 모터의 권선 온도와 코어 온도를 2저항 열 모델(two-resistor thermal model)을 사용하여 계산하는 것이 일반적인 접근법이다.

자연계에서 다리를 다친 동물이 절뚝거리며 이동하는 것처럼, 로봇도 모터에 문제가 생기더라도 계속 동작할 수 있어야 한다. 이러한 생체 모방적 적응 능력은 실제 환경에서 로봇의 실용성을 크게 향상시킬 수 있다.

### 1.2 Problem Definition

ToddlerBot의 walking RL 정책 실행 실험에서 서보 모터의 과부하(overload)로 인한 토크 감소와 기능 정지가 발생하여 **19분 이후 낙상 횟수(Fall Count)가 급격히 증가**하는 현상이 관찰되었다. 이는 누적되는 열 스트레스에 대한 대응 방안이 필요함을 명확히 보여준다.

모터가 과열되면 최대 토크가 감소하는 **디레이팅(derating)** 현상이 발생한다. KOLLMORGEN 모터의 사례에서 보듯이, 온도 상승에 따라 허용 가능한 최대 토크가 점진적으로 감소하여 로봇의 동작 능력이 저하된다. 따라서 다음과 같은 핵심 질문이 제기된다:

> **핵심 질문**: 휴머노이드 로봇의 전기 모터에서 발생하는 열 문제를 어떻게 효과적으로 관리하고, 고장 상황에서도 안정적인 보행을 유지할 수 있는가?

### 1.3 Physical Background

모터의 발열량을 이해하기 위해서는 전력 관계를 파악해야 한다. 모터에 공급되는 입력 전력과 기계적 출력 전력의 관계는 다음과 같다:

- **입력 전력**: $P_{in} = V \times I$ (전압 $\times$ 전류)
- **출력 전력**: $P_{out} = \tau \times \omega$ (토크 $\times$ 각속도)
- **손실 전력**: $P_{loss} = P_{in} - P_{out}$

모터의 발열량은 입력 전력에서 기계적 출력을 빼서 구하며($Q = P_{in} - P_{out}$), 권선에서 발생하는 열손실($I^2R$)이 주요 요인이다. MuJoCo 시뮬레이션에서 토크(Torque)와 각속도(velocity)를 획득하여 출력 전력을 계산할 수 있다.

**[결론문]** 이러한 배경에서 본 연구는 확장된 열 모델, 온라인 학습 기반 파라미터 적응, 그리고 고장 인식 적응 제어를 통합한 시스템을 제안한다.

*앞서 제기한 열 관리 및 고장 적응 문제를 해결하기 위해, 다음 섹션에서는 관련 선행 연구를 검토한다.*

---

## 2. Related Work

**[주제문]** 본 섹션에서는 로봇의 고장 인식 및 적응적 보행 제어에 관한 선행 연구를 검토한다.

### 2.1 Fault-Aware Locomotion Control

앞서 제기한 고장 상황에서의 적응 문제를 해결하기 위해, 여러 연구들이 결함 인식 로코모션을 탐구해왔다. **DreamFLEX**는 험지에서의 이상 상황에 대한 결함을 다음과 같이 정의하고 학습한다:

1. **잠긴 관절(Locked Joint)**: 외부 충격으로 특정 관절이 고정되거나 움직임 범위가 제한됨
2. **약화된 모터(Weakened Motor)**: 과열, 전력 부족 등으로 정상적인 토크 발생이 제한됨

이러한 결함 시나리오를 학습하여 sim2real gap을 최소화하는 것이 목표이다. DreamFLEX에서는 결함이 생긴 관절을 빨간색 원으로 표시하여 어떤 관절에 문제가 있는지 직관적으로 파악할 수 있게 한다.

### 2.2 Thermal Modeling for Electric Motors

**[플레이스홀더: 선행 열 모델링 연구 검토 필요]**

> 2저항 모델의 기원, 다중 저항 모델, 신경망 기반 열 추정 방법 등 열 모델링 관련 선행 연구에 대한 체계적인 리뷰가 필요하다. 특히 다음 내용을 보완해야 한다:
> - 전통적인 열 등가 회로 모델의 발전 과정
> - Lumped parameter 모델과 FEM 기반 모델의 비교
> - 데이터 기반 열 추정 방법론의 최근 동향

**[결론문]** 기존 연구들은 고장 인식 또는 열 모델링을 개별적으로 다루었으나, 본 연구는 이를 통합하여 열 인식 적응적 보행 제어를 구현한다.

*앞서 검토한 선행 연구의 한계를 극복하기 위해, 다음 섹션에서는 제안하는 방법론을 상세히 설명한다.*

---

## 3. Methodology

**[주제문]** 본 섹션에서는 FEMNet 기반 고장 인식 시스템, 확장된 열 모델, 온라인 학습 알고리즘, 그리고 열 제어기를 포함한 제안 방법론을 설명한다.

### 3.1 FEMNet: Failure Estimation & Modulation Network

앞서 제기한 고장 인식 문제를 해결하기 위해, FEMNet(Failure Estimation & Modulation Network)을 제안한다. FEMNet은 두 개의 주요 모듈로 구성된다:

#### 3.1.1 Failure Estimation Model

Encoder Network($512 \times 256 \times 128$)는 관측 이력($\mathbf{o}_t^H$)에서 다음을 추출한다:
- $\mathbf{v}_t$: 몸체 선속도 (Body linear velocity)
- $\mathbf{f}_t$: 관절 고장 벡터 (Joint fault vector)
- $\mathbf{z}_t$: 컨텍스트 벡터 (Context vector)

#### 3.1.2 Modulation Model

Modulation Model($64 \times 64$)은 변조 파라미터($\gamma_1, \gamma_2$)를 학습하여 컨텍스트를 변조된 컨텍스트($\tilde{\mathbf{z}}_t$)로 변환한다:

$$\tilde{\mathbf{z}}_t = \gamma_1 \cdot \mathbf{z}_t + \gamma_2$$

Decoder Network($64 \times 128$)는 변조된 컨텍스트를 사용하여 다음 관측($\mathbf{o}_{t+1}$)을 예측한다.

### 3.2 Extended Thermal Model

앞서 언급한 2저항 열 모델의 한계를 극복하기 위해, 추가 파라미터 $P_1 \sim P_5$를 도입한 확장 열 모델을 제안한다.

#### 3.2.1 Thermal Differential Equations

코어 온도($c_1$)와 하우징 온도($c_2$)의 시간 변화율을 나타내는 미분 방정식:

$$\dot{c}_1 = W_1 \exp(P_1) \tau^2 - \frac{c_1 - c_2}{W_2 \exp(P_2)}$$

$$\dot{c}_2 = \frac{c_1 - c_2}{W_3 \exp(P_3)} - \frac{c_2 - W_5(1 + P_5)}{W_4 \exp(P_4)}$$

여기서 $\exp(P_i)$ 형태로 파라미터가 적용되며, 모든 $P_i = 0$이면 기존 2저항 열 모델과 동일해진다. 이 확장을 통해 비선형 열 특성을 더 정확하게 모델링할 수 있다.

#### 3.2.2 Parameter Notation

2저항 열 모델의 파라미터:
- $R_{th1}, R_{th2}$: 열 저항 (Thermal resistance)
- $C_{th1}, C_{th2}$: 열 용량 (Thermal capacitance)
- $Q$: 열 발생률 (Heat generation rate)

### 3.3 Parameter Optimization

실제 측정된 하우징 온도와 2저항 열 모델에서 계산된 온도 사이의 MSE를 손실 함수로 사용하여 Adam Optimizer로 학습한다:

$$\mathcal{L}_{thermal} = \text{MSE}(T_{housing}^{measured}, T_{housing}^{predicted})$$

열 저항($R_{th1}, R_{th2}$), 열 용량($C_{th1}, C_{th2}$), 열 발생률($Q$)이 **항상 양수가 되도록 로그 스케일에서 학습**을 진행한다.

### 3.4 Online Learning Algorithm

실시간으로 수집되는 데이터를 사용하여 열 모델 파라미터를 지속적으로 업데이트한다. 온라인 학습의 주요 하이퍼파라미터:

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| $\Delta t_{data}$ | 1.0 sec | 데이터 샘플링 간격 |
| $N_{seq}$ | 30 step | 시퀀스 길이 |
| $N_{batch}$ | 10 seq | 배치 크기 |

배치 단위로 손실 함수를 계산하고 그래디언트 하강법으로 파라미터를 조정하여 실제 환경 변화에 적응한다.

### 3.5 Thermal Controller

앞서 제안한 열 모델을 기반으로, 안전한 온도 범위 내에서 최대 성능을 발휘하도록 하는 열 제어기를 설계한다.

#### 3.5.1 Control Loss Function

열 제어기의 손실 함수 $L_{control}$은 두 부분으로 구성된다:

$$L_{control} = \text{MSE}(c_{1,[k+1,k+N_{control}-1]}, c_{1,[k+1,k+N_{control}-1]}^{max}) + W_{control} \text{MSE}(\mathbf{0}, f_{[k,k+N_{control}-1]}^{limit})$$

- 첫 번째 항: 예측된 코어 온도($c_1$)와 최대 허용 온도($c_1^{max}$) 간의 MSE
- 두 번째 항: 토크 제한($f^{limit}$) 준수를 위한 가중치($W_{control}$)가 적용된 MSE

#### 3.5.2 Control Strategy

열 제어기는 $N_{control}$ 스텝 앞의 코어 온도를 예측하고, 예측 온도가 최대 허용 온도를 초과하지 않도록 현재 토크를 조절한다. 이를 통해 모터 손상을 방지하면서 최대 성능을 유지한다.

### 3.6 Data Collection for Training

12개 액추에이터에 대해 각각의 손상 시나리오($\tau^i$: $i$번째 액추에이터가 손상된 시나리오)를 고려한다.

#### 3.6.1 Trajectory Generation

학습된 teacher policy에서 각 시나리오별로 $N = 20,000$개의 trajectory를 생성한다. 각 trajectory($traj_n^i$)는 다음으로 구성된다:
- $\mathbf{s}$: 상태 정보 (로봇의 현재 자세와 속도)
- $\mathbf{d}$: 손상 정보 (어떤 액추에이터에 어떤 유형의 고장이 발생했는지)
- 추가 관측값

발열량 계산에 필요한 모터 파라미터(효율, 토크 상수, 권선 저항 등)는 maxon motor의 key information에서 참조하여 실제 하드웨어 스펙을 기반으로 정확한 열 모델링을 수행한다.

**[결론문]** 제안된 방법론은 고장 인식(FEMNet), 열 모델링(확장 열 모델), 온라인 학습, 열 제어기를 유기적으로 통합하여 열 인식 적응적 보행 제어를 구현한다.

*앞서 설명한 방법론의 효과를 검증하기 위해, 다음 섹션에서는 실험 설정과 구현 세부사항을 설명한다.*

---

## 4. Experiments

**[주제문]** 본 섹션에서는 제안된 방법론을 검증하기 위한 시뮬레이션 환경 구축과 구현 세부사항을 설명한다.

### 4.1 Simulation Environment Setup

#### 4.1.1 MuJoCo Installation and Verification

mujoco_py를 import하고 humanoid.xml 모델을 로드하여 MjSim 객체를 생성한다. `sim.step()`으로 시뮬레이션을 진행하고 `sim.data.qpos`로 관절 위치를 확인하여 설치가 정상적으로 완료되었는지 검증한다.

```python
import mujoco_py
import os
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
sim.step()
print(sim.data.qpos)
```

#### 4.1.2 Environment Rendering

gym 환경을 생성하고 `render_mode='rgb_array'`로 설정하여 화면을 numpy 배열로 받는다. IPython.display를 사용하여 Jupyter 환경에서 실시간으로 시뮬레이션 화면을 시각화할 수 있다.

### 4.2 Implementation Issues and Solutions

#### 4.2.1 ToddlerBot Path Resolution

ToddlerBot 모듈에서 `os.path`의 상대경로가 올바르게 동작하지 않는 문제가 있었다. `pathlib`의 Path 클래스를 이용하여 절대경로로 코드를 수정함으로써 `descriptions/assemblies/` 디렉토리 접근 문제를 해결했다.

```python
from pathlib import Path

class Robot:
    def __init__(self, robot_name: str):
        self.root_path = Path(__file__).parent.parent
        self.root_path = self.root_path / "descriptions" / self.name
        self.config_path = self.root_path / "config.json"
```

#### 4.2.2 Robot File Path Utility

`find_robot_file_path` 함수는 로봇 이름과 파일 확장자(`.urdf`)를 받아 해당 파일의 절대 경로를 반환한다. `descriptions/{robot_name}/` 디렉토리를 먼저 확인하고, 없으면 `descriptions/assemblies/` 디렉토리를 확인한다.

### 4.3 Experimental Setup Details

**[플레이스홀더: 실험 설정 상세화 필요]**

> 다음 정보를 보완해야 한다:
> - 사용된 로봇 플랫폼 (ToddlerBot 사양)
> - 센서 종류 및 사양 (온도 센서, 전류 센서 등)
> - 데이터 샘플링 주파수
> - 실험 프로토콜 및 반복 횟수

**[결론문]** 시뮬레이션 환경이 성공적으로 구축되었으며, 구현 과정에서 발생한 경로 문제들이 해결되었다.

*앞서 설명한 실험 환경에서 수행된 결과를 다음 섹션에서 분석한다.*

---

## 5. Results and Discussion

**[주제문]** 본 섹션에서는 제안된 열 모델과 제어기의 성능을 검증하고, 토크 제한이 보행 안정성에 미치는 영향을 분석한다.

### 5.1 Thermal State Estimation Results

#### 5.1.1 HeatState Module Validation

HeatState 모듈의 실험 결과를 통해 열 상태 추정의 정확도를 검증했다. 그래프는 시간에 따른 열 상태 변화와 예측 정확도를 보여준다. HeatState 모듈의 온도 추정값과 실제 측정값을 비교한 결과, 두 값의 높은 일치도를 통해 열 모델의 정확도를 정량적으로 확인할 수 있었다.

#### 5.1.2 Limitations of Two-Resistor Model

2축(2저항) 열 모델의 측정 결과 그래프에서 모델 예측과 실제 측정값 사이에 괴리가 있음을 확인했다. 이러한 한계점은 다음과 같은 개선 방향을 시사한다:
1. 추가 열 경로 고려
2. 비선형 열 특성 반영
3. 온라인 파라미터 적응

이러한 개선이 제안된 확장 열 모델에 반영되었다.

### 5.2 Torque Control Effect Analysis

#### 5.2.1 Normal vs. Torque-Limited Comparison

정상 상태(0~500 스텝)와 토크 제한 상태(1000~1389 스텝)에서의 로봇 동작을 비교했다. 토크 제한이 보행 패턴과 안정성에 미치는 영향을 시각적으로 확인할 수 있었다.

#### 5.2.2 Walking Stability Analysis

정상 상태에서의 안정적인 보행과 토크 제한 상태에서의 불안정한 보행을 비교한 결과, 토크 제한 시 로봇이 균형을 유지하기 어려워지며, 이를 극복하기 위한 적응적 제어 전략이 필요함을 확인했다.

### 5.3 Training Progress

#### 5.3.1 Learning Performance

RTX 5080 GPU에서 10시간 학습을 진행한 결과, 5,120,000 스텝(초기)과 302,080,000 스텝(후기)의 성능을 비교하여 학습이 진행됨에 따라 정책의 성능이 향상됨을 확인했다.

#### 5.3.2 Learning Curve Analysis

초기 학습 상태와 충분한 학습 후의 성능을 비교하면, 학습이 진행됨에 따라 정책이 더 안정적이고 효율적인 행동을 생성함을 확인할 수 있다.

### 5.4 Quantitative Evaluation

**[플레이스홀더: 정량적 비교 결과 추가 필요]**

> 다음 정량적 지표를 추가해야 한다:
> - 기존 방법 대비 온도 추정 오차 감소율 (RMSE, MAE)
> - 보행 안정성 향상 지표 (낙상 횟수, 보행 거리)
> - 토크 효율성 비교

### 5.5 Real Robot Validation

**[플레이스홀더: 실제 로봇 실험 결과 추가 필요]**

> 다음 실제 로봇 실험 결과를 추가해야 한다:
> - sim2real 전이 성능
> - 실제 환경에서의 열 제어 효과
> - 장시간 운용 테스트 결과

**[결론문]** 실험 결과는 제안된 열 모델과 제어기가 효과적으로 온도를 추정하고 제어할 수 있음을 보여주며, 향후 실제 로봇에서의 검증이 필요하다.

*앞서 분석한 결과를 바탕으로, 다음 섹션에서 연구의 결론과 향후 방향을 제시한다.*

---

## 6. Conclusion

**[주제문]** 본 연구는 휴머노이드 로봇의 열 인식 적응적 보행 제어를 위한 통합 시스템을 제안하였다.

### 6.1 Summary of Contributions

본 논문의 주요 기여는 다음과 같다:

1. **확장된 열 모델**: 기존 2저항 열 모델을 $P_1 \sim P_5$ 파라미터로 확장하여 비선형 열 특성을 모델링
2. **온라인 학습**: 실시간 데이터를 활용한 열 모델 파라미터의 지속적 업데이트
3. **MSE 기반 열 제어기**: 안전한 온도 범위 내에서 최대 토크를 활용하는 제어 전략
4. **FEMNet**: 고장 상황 인식 및 적응적 보행 제어를 위한 신경망 아키텍처

### 6.2 Limitations and Future Work

본 연구의 한계점과 향후 연구 방향:

- 실제 로봇에서의 검증 필요
- 다양한 환경 조건(온도, 습도)에서의 성능 평가
- 더 복잡한 고장 시나리오에 대한 일반화

### 6.3 Concluding Remarks

**핵심 주장**: 확장된 열 모델과 온라인 학습 기반 열 제어기를 통해 모터의 코어 온도를 정확히 예측하고 제어할 수 있으며, FEMNet 기반 고장 인식 및 적응 시스템을 통해 액추에이터 고장 상황에서도 안정적인 보행을 유지할 수 있다.

---

## References

[참고문헌 추가 필요]

---

## Appendix

### A. Thermal Model Derivation

확장된 열 모델의 상세 유도 과정

### B. FEMNet Architecture Details

FEMNet의 레이어별 상세 구조

### C. Hyperparameter Settings

실험에 사용된 전체 하이퍼파라미터 목록

---

*Generated with Moonwriter Standard Option (Option 2)*
*Writing Principles Applied: Thesis CoT Extraction, Skeleton Drafting, Milestone Markers, Glue Sentences, Gap Placeholders*
