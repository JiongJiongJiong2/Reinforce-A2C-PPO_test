# AI4023 Deep Reinforcement Learning - Project 2 Rules
**REINFORCE + A2C + PPO Implementation (2026 Spring)**

## 项目目标
- 严格按照 PDF 要求实现 **REINFORCE**、**n-step A2C** 和 **PPO-Clip**。
- 必须在 `CartPole-v0` 和 `Pendulum-v0` 上成功训练并报告对应指标。
- **最优先目标**：代码能正常运行、通过 autograder 检查、生成要求的 plots 和数值报告。
- 完成题目要求即可，**不要添加任何额外优化、trick 或修改**。
- 每一次修改代码前必须说明要修改哪个文件、哪个函数，以及修改理由。
- 涉及环境配置、依赖安装、运行命令等操作，必须先询问确认后再执行。

## 核心公式（必须严格按照以下公式实现）

### 1. REINFORCE (Policy Gradient)
$$
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{l=1}^{N} \left( \sum_{t=0}^{T} \gamma^{t} \nabla_{\theta} \log \pi_{\theta}(a_{t} | s_{t}) \cdot G_t \right)
$$
其中 $G_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ （Monte Carlo Return）

### 2. A2C (Advantage Actor-Critic) - n-step
$$
A_t = G_t^{(n)} - V(s_t)
$$
其中 n-step return：
$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n})
$$
（如果到达 terminal 状态，后续价值为 0）

### 3. PPO-Clip
$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
$$
其中：
- $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$
- $\hat{A}_t$ 使用 Monte Carlo 估计（Algorithm 2）
- Value Loss：$\mathbb{E}_t[(V_\phi(s) - \hat{V}(s))^2]$

**所有公式中的 `.detach()` 使用必须严格遵守**，防止梯度泄漏。

## 文件实现要求

### 1. `component/network.py`（重要）
- `GaussianPolicyNet.forward()`
- `CategoricalPolicyNet.forward()`
- `GaussianActorCriticNet.forward()`
- `CategoricalActorCriticNet.forward()`

**要求**：严格按照 PDF 和代码注释中的网络结构与输出格式实现，不要修改网络架构。

### 2. `agent/REINFORCE_agent.py`
- `REINFORCEAgent.step()`（20 points 部分）

### 3. `agent/A2C_agent.py`
- `A2CAgent.step()`（n-step 版本）

### 4. `agent/BaseAgent.py`
- `eval_episode()`（用于评估策略）

### 5. `ppo.py`
- 完整实现 **PPO-Clip**（Algorithm 1 + Algorithm 2）
- 必须使用 Monte Carlo 优势估计
- 严格按照推荐超参数或 PDF 中提供的默认值（除非完全跑不通）

## 实现原则（重要！）

1. **最优先**：代码能跑通，能生成要求的训练曲线和最终评估 return。
2. **严格遵守** PDF 中指定的函数签名、接口和文件结构，不要擅自修改已有框架。
3. **先实现顺序**：
   - 先完成 REINFORCE（CartPole）并验证
   - 再实现 n-step A2C
   - 最后实现 PPO（重点在 PPO-Clip）
4. **网络保持最小改动**：只实现 forward 函数，不要增加层数或改动结构。
5. **超参数**：优先使用 `main.py` 中已给的默认值，或 PDF 推荐值。不要随意大幅调整。
6. **随机种子**：使用 `set_seed()` 保证结果可复现，多跑几次不同种子验证稳定性。
7. **报告内容**：严格按照 PDF 要求报告 training return、policy loss、value loss（A2C）、final evaluation return 等。

## 调试建议
- 先在 `CartPole-v0` 上快速验证（更容易收敛）。
- 观察 policy loss 是否正常、rewards 是否上升。
- 使用 `print` 临时输出关键变量（如 log_prob、advantage、ratio、loss 等）帮助调试。
- PPO 部分重点检查 `.detach()` 是否正确使用，以及 minibatch 更新逻辑。

## 禁止事项
- **不要**修改 PDF 中没有要求修改的文件或函数。
- **不要**添加 entropy bonus、GAE、PPO-Penalty 等未要求的技巧。
- **不要**大幅修改网络结构或加入复杂组件（CNN、RNN 等）。
- **不要**修改 `main.py` 中训练和绘图的主体逻辑。
- **不要**自行添加 early stopping、learning rate decay 等优化手段。
- PPO 只实现 **PPO-Clip** 版本（PDF 已明确说明）。

---

**使用提示**：  
当你需要我帮你写具体函数时，请直接告诉我**文件名 + 函数名**（例如：“写 network.py 中的 CategoricalActorCriticNet.forward()” 或 “写 A2C_agent.py 中的 step()”），我会给你**可直接复制粘贴的代码 + 详细说明**。

现在这个规则文档已经准备好了，你可以直接复制使用。

---

**确认一下**：这个规则是否符合你的预期？需要我再调整某些部分吗？（比如增加更多公式细节或修改顺序）