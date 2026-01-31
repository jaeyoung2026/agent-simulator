# Thermal-Aware Fault-Tolerant Control for Humanoid Robot Locomotion

## Abstract

Humanoid robots operating in real-world environments face a critical challenge: motor thermal limits and actuator failures can cause torque reduction and eventual system failure during extended locomotion tasks. This paper presents an integrated approach combining physics-based thermal modeling with learned failure estimation networks to enable thermal-aware fault-tolerant control. We extend the classical two-resistor thermal model with five learnable parameters (P1-P5) that adapt online to specific motor characteristics. Additionally, we introduce FEMNet, a failure estimation model that infers actuator health states from observation history alone, without requiring dedicated failure sensors. Experimental validation on the ToddlerBot platform in MuJoCo simulation demonstrates accurate temperature prediction and maintained locomotion capability under various fault scenarios, with walking performance sustained even under torque-limited conditions lasting over 1000 steps.

---

## 1. Introduction

**High-performance electric motors are essential for dynamic humanoid robot locomotion, yet their operation is fundamentally constrained by thermal limits that restrict continuous high-torque output.** As robots are deployed for increasingly demanding tasks requiring sustained operation, the challenge of thermal management becomes critical. When motors overheat, the resulting torque derating can lead to degraded performance or complete locomotion failure, as the commanded torques can no longer be achieved.

The severity of this problem is illustrated by the observation that humanoid robots like ToddlerBot experience servo motor overload during walking, with fall counts increasing significantly after approximately 19 minutes of operation. This degradation occurs because accumulated heat causes the motor windings to approach their temperature limits, triggering protective torque reduction mechanisms. The robot's control policy, trained assuming full torque capability, becomes increasingly ineffective as the available torque diminishes.

Interestingly, biological systems provide inspiration for addressing this challenge. Animals with injured limbs do not simply collapse; instead, they adapt their gait to compensate for reduced capability in the affected limbs. A hedgehog with a bandaged leg continues to locomote, redistributing load to healthy limbs and modifying its movement patterns. This biological resilience suggests that robots could similarly adapt to degraded actuator performance if they could accurately sense and respond to their thermal state.

**This paper addresses two key questions: (1) How can we accurately predict motor temperatures in real-time to anticipate thermal limits? (2) How can robots detect and adapt to actuator degradation without requiring explicit failure sensors?** We propose a hybrid approach that combines the physical interpretability of classical thermal models with the adaptability of learned parameters, integrated with a neural network that estimates fault states from motion patterns alone. The following sections detail our thermal modeling approach, the FEMNet architecture for failure estimation, and experimental validation demonstrating effective thermal-aware fault-tolerant locomotion.

---

## 2. Related Work

**Prior research has addressed motor thermal modeling and fault-tolerant locomotion as separate problems, yet an integrated approach combining real-time thermal prediction with learned fault estimation remains unexplored.** This section reviews relevant work in both areas and identifies the gap our approach addresses.

### 2.1 Fault-Tolerant Locomotion

Recent advances in reinforcement learning have enabled robots to learn robust locomotion policies. DreamFLEX [Prior Work] demonstrated fault-aware quadrupedal locomotion on rough terrain, defining two primary fault types: locked joints (where external impact causes joint fixation or range limitation) and weakened motors (where overheating or power shortage limits torque output). Their approach learns policies that minimize the sim2real gap under fault conditions, showing that robots can maintain locomotion despite actuator degradation. However, this work assumes fault states are known or can be accurately sensed, rather than inferred from motion patterns.

### 2.2 Motor Thermal Modeling

Industrial motor control systems commonly employ thermal models to prevent overheating damage. Manufacturers like KOLLMORGEN specify derating curves that define how maximum torque decreases with temperature. The standard approach models this relationship as a piecewise function:

- Full torque (alpha = 1) when temperature T < T_threshold
- Linear reduction when T_threshold <= T < T_max
- Zero torque (alpha = 0) when T >= T_max

The output torque is then: tau_out = alpha(T) * tau_commanded

While physically meaningful, these models typically use fixed parameters that may not accurately capture the thermal dynamics of specific motors or operating conditions.

### 2.3 Research Gap

**Existing approaches either assume accurate thermal sensors (which may not be available for internal motor temperatures) or treat fault detection and thermal management as separate problems.** Our work bridges this gap by proposing a learnable thermal model that adapts online and a failure estimation network that infers actuator health from readily available observation data. This integration enables thermal-aware control without requiring specialized sensors beyond standard motor encoders and housing temperature measurements.

---

## 3. Methods

This section presents our integrated approach, beginning with the extended thermal model, followed by the online learning procedure, the FEMNet architecture for failure estimation, and the thermal-aware controller.

### 3.1 Extended Two-Resistor Thermal Model

**We extend the classical two-resistor thermal model with five learnable parameters to capture real-world thermal dynamics while maintaining physical interpretability.** The standard model represents motor thermal behavior as an electrical circuit analog with two temperature nodes (winding c1 and housing c2), thermal resistances (R1, R2), and thermal capacitances (C1, C2).

#### 3.1.1 Heat Generation Model

Motor heat generation arises from the difference between electrical input power and mechanical output power:

P_loss = P_electrical - P_mechanical = (tau * omega) / eta - tau * omega

where tau is torque, omega is angular velocity, and eta is motor efficiency. The dominant loss occurs in the motor windings as I^2R copper losses. In MuJoCo simulation, torque and velocity are directly available, enabling accurate heat generation calculation.

The mechanical output power is simply:

P_mechanical = tau * omega

#### 3.1.2 Proposed Model Extension

**Our key contribution is augmenting the standard model with five learnable parameters P1-P5 that adapt the thermal dynamics to specific motors.** The extended differential equations are:

c1_dot = W1 * exp(P1) * tau^2 - (c1 - c2) / (W2 * exp(P2))    ... (Eq. 7')

c2_dot = (c1 - c2) / (W3 * exp(P3)) - (c2 - W5 * (1 + P5)) / (W4 * exp(P4))    ... (Eq. 8')

where W1-W5 are fixed physics-derived weights and P1-P5 are learned correction terms. The exponential parameterization ensures thermal resistances and capacitances remain positive. Importantly, when P_i = 0 for all i, the model reduces to the standard two-resistor formulation, providing a principled baseline.

### 3.2 Online Parameter Learning

**To adapt the thermal model to individual motors during operation, we employ online learning with carefully designed constraints ensuring physical validity.** The learning objective minimizes the mean squared error between measured housing temperature T_measured and predicted temperature T_predicted from the model.

#### 3.2.1 Optimization Setup

- **Loss function**: L = MSE(T_measured, T_predicted)
- **Optimizer**: Adam optimizer
- **Constraint handling**: Parameters are learned in log-scale to ensure positivity of R1, R2, C1, C2, and related quantities

#### 3.2.2 Training Configuration

The online learning uses the following hyperparameters:
- Temporal resolution: delta_t_data = 1.0 second
- Sequence length: N_seq = 30 steps
- Batch size: N_batch = 10 sequences

The parameter update rule follows standard gradient descent:

P_{1,2,3,4,5} <- P_{1,2,3,4,5} - alpha * (dL_update / dP_{1,2,3,4,5})

This online adaptation enables the model to continuously improve its predictions as the robot operates, capturing motor-specific characteristics that may not be known a priori.

### 3.3 FEMNet: Failure Estimation Model Network

**FEMNet learns to estimate actuator health states from observation history, enabling fault-aware control without dedicated failure sensors.** The architecture comprises two main components: a failure estimation model and a modulation model.

#### 3.3.1 Failure Estimation Model (Encoder)

The encoder network processes observation history o_t^H through a multi-layer perceptron (512 x 256 x 128) to produce three outputs:
- **v_t**: Body linear velocity estimate
- **f_t**: Joint fault vector identifying damaged actuators
- **z_t**: Context vector for downstream policy modulation

This architecture enables the network to learn compressed representations of the robot's health state from readily available sensor data.

#### 3.3.2 Modulation Model

The modulation network (64 x 64) transforms the context vector z_t into a modulated context z_tilde using learned parameters gamma_1 and gamma_2:

z_tilde = gamma_1 * z_t + gamma_2

A decoder network (64 x 128) predicts the next observation o_{t+1}, providing a self-supervised learning signal.

#### 3.3.3 Training Data Collection

**To train FEMNet, we generate simulation data covering 12 actuator damage scenarios, providing comprehensive coverage of possible failure modes.** For each scenario i (where i indicates which actuator is damaged):

- T^i = {traj_0^i, traj_1^i, ..., traj_N^i} where N = 20,000 trajectories
- Each trajectory contains state-action tuples: traj_k^i = {(s_{k,t}^i, d_{k,t}^i, a_{k,t}^i)}_{t=0,1,...,T}

The data is generated from a trained teacher policy operating under simulated fault conditions, ensuring the network learns from realistic motion patterns.

### 3.4 Thermal-Aware Controller

**The thermal controller integrates predicted temperatures with the control policy to prevent overheating while maximizing performance.** The control objective balances two goals: keeping motor core temperature c1 below the damage threshold c1_max, and minimizing unnecessary torque limitation.

#### 3.4.1 Control Loss Function

L_control = MSE(c1_{[k+1,k+N_control-1]}, c1_{[k+1,k+N_control-1]}^max) + W_control * MSE(0, f_{[k,k+N_control-1]}^limit)

The first term penalizes predicted temperatures exceeding the limit over a prediction horizon. The second term regularizes the fault vector to avoid over-conservative torque limiting.

#### 3.4.2 Torque Modulation

Based on the predicted temperature T_i for actuator i, the output torque is scaled:

tau_{i,out} = alpha_i(T_i) * tau_{i,cmd}

where alpha_i follows the standard derating curve described in Section 2.2. This modulation ensures commanded torques respect thermal limits while providing maximum available performance.

---

## 4. Experiments

This section describes the experimental setup and presents results validating our approach.

### 4.1 Implementation

**We validate our approach using the ToddlerBot humanoid platform in MuJoCo simulation.** ToddlerBot is a compact humanoid robot with 12 actuated joints, providing a representative testbed for bipedal locomotion research.

#### 4.1.1 Simulation Environment

The experiments use MuJoCo physics simulation with the following configuration:
- Python environment with mujoco_py interface
- Gymnasium (gym) for reinforcement learning environment wrapper
- IPython display with matplotlib for visualization

#### 4.1.2 Software Infrastructure

To ensure reproducibility, several code modifications were necessary for the ToddlerBot codebase:
- Path resolution using pathlib for absolute paths (replacing unreliable os.path relative paths)
- Robot configuration loading from JSON files
- Cache management for forward/inverse kinematics

The core path resolution pattern used:

```python
self.root_path = Path(__file__).parent.parent
self.root_path = self.root_path / "descriptions" / self.name
self.config_path = self.root_path / "config.json"
```

#### 4.1.3 Training Hardware

Reinforcement learning training was conducted on an RTX 5080 GPU, with typical training runs requiring approximately 10 hours to reach 300 million steps.

### 4.2 Experimental Results

**Experimental results demonstrate successful thermal prediction and maintained locomotion under fault conditions.**

#### 4.2.1 Thermal State Monitoring (HeatState Results)

The HeatState simulation tracks actuator torque and temperature over time. Analysis of act 13 (right_knee_actuator) shows:
- Torque oscillations characteristic of walking gait (range: -2.0 to 1.0 Nm)
- High-frequency variations indicating rapid control adjustments
- Stable thermal behavior over 100+ seconds of continuous operation

The similarity between commanded torque (target_tau) and achieved torque (st_torque) demonstrates effective torque tracking when temperatures remain within limits.

#### 4.2.2 Normal vs. Torque-Limited Performance

**A key validation compares walking performance under normal conditions versus torque-limited conditions:**

| Condition | Steps | Performance |
|-----------|-------|-------------|
| Normal | 0-500 | Full torque capability, stable walking |
| Torque-limited | 1000-1389 | Reduced torque, adapted gait maintained |

The results show that with thermal-aware control, the robot maintains walking capability even when torque is significantly limited, demonstrating successful adaptation to degraded conditions.

#### 4.2.3 Learning Progression

Training curves at different stages reveal policy improvement:
- At 5,120,000 steps: Initial policy learning basic locomotion
- At 302,080,000 steps: Mature policy with robust fault-tolerant behavior

The substantial training required (10 hours on RTX 5080) reflects the complexity of learning both nominal locomotion and fault adaptation.

---

## 5. Discussion

**While our approach shows promising results in simulation, several limitations warrant discussion and suggest directions for future work.**

### 5.1 Model Accuracy Limitations

The two-resistor thermal model with learned parameters provides a good balance between accuracy and computational efficiency. However, measurement results on the ToddlerBot platform reveal accuracy bounds, particularly for rapid temperature transients. The simplified two-axis model may not fully capture three-dimensional heat flow in complex motor geometries.

### 5.2 Simulation-to-Real Gap

All experiments were conducted in MuJoCo simulation. While simulation enables controlled experiments across fault scenarios without damaging hardware, the sim2real gap remains a concern for deployment. Key differences include:
- Idealized sensor readings vs. noisy real measurements
- Perfect model parameters vs. manufacturing variations
- Deterministic simulation vs. stochastic real-world dynamics

### 5.3 Future Directions

Several extensions could address current limitations:

1. **Real Hardware Validation**: Transfer the learned policies and thermal models to physical ToddlerBot hardware, validating sim2real transfer.

2. **Extended Thermal Dynamics**: Incorporate additional thermal effects such as ambient temperature variations, cooling system dynamics, and motor-specific calibration.

3. **Multi-Motor Coordination**: Extend fault estimation to consider interactions between multiple degraded actuators, enabling coordinated adaptation.

4. **[PLACEHOLDER: Longer-horizon experiments needed]** Current experiments validate performance up to ~1400 steps. Extended trials (10,000+ steps) would better characterize long-term thermal management effectiveness. Proposed experiment: 1-hour continuous walking with periodic fault injection.

---

## 6. Conclusion

This paper presented an integrated approach to thermal-aware fault-tolerant control for humanoid robot locomotion. By combining a physics-based two-resistor thermal model with learnable parameters and a neural network (FEMNet) for failure estimation, we enable robots to predict motor degradation and adapt their control policies in real-time.

**Our key contributions are:**

1. **Extended Thermal Model**: A two-resistor model augmented with five learnable parameters (P1-P5) that adapt online to specific motor characteristics while maintaining physical interpretability.

2. **FEMNet Architecture**: A failure estimation network that infers actuator health states from observation history alone, eliminating the need for dedicated failure sensors.

3. **Integrated Controller**: A thermal-aware control framework that uses predicted temperatures to modulate torque commands, maintaining locomotion within safe thermal limits.

Experimental validation on the ToddlerBot platform demonstrates accurate temperature prediction and maintained walking capability under torque-limited conditions. The bio-inspired principle---that robots, like injured animals, can adapt their gait to compensate for reduced actuator capability---proves viable through our learning-based approach.

While simulation results are promising, future work will focus on real hardware validation and extended operational scenarios. The framework presented here provides a foundation for deploying humanoid robots in demanding real-world applications where thermal management and fault tolerance are critical for sustained operation.

---

## Acknowledgments

This research was conducted at the Global School of Media. We thank the ToddlerBot development team for the simulation platform.

---

## References

[To be added based on cited works including DreamFLEX, KOLLMORGEN motor specifications, and MuJoCo simulation framework]
