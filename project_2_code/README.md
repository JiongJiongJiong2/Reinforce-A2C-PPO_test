# AI4023 Deep Reinforcement Learning - Project 2

**REINFORCE + A2C + PPO Implementation**

作者: 2230026047

---

## 目录结构

```
project_2_code/
├── section_1/                 # REINFORCE 和 A2C 实现
│   ├── main.py               # 主程序入口
│   ├── agent/                # Agent 实现
│   │   ├── BaseAgent.py      # 基础 Agent (eval_episode)
│   │   ├── REINFORCE_agent.py # REINFORCE Agent
│   │   └── A2C_agent.py      # A2C Agent
│   ├── component/            # 网络和环境组件
│   │   ├── network.py        # 策略网络
│   │   └── envs.py           # 环境封装
│   ├── utils/                # 工具函数
│   └── plot.py               # 绘图函数
│
└── section_2/                 # PPO 实现
    └── ppo.py                # PPO-Clip 算法
```

---

## 环境配置

### 依赖安装

```bash
pip install torch numpy matplotlib gymnasium tensorboardX
```

### 验证环境

```bash
python -c "import torch; import gymnasium; print('环境配置成功')"
```

---

## Section 1: REINFORCE 和 A2C

### 1.1 运行 CartPole-v0 实验

**修改 `main.py` 第 126 行：**
```python
game = 'CartPole-v0'
# game = 'Pendulum-v0'
```

**运行命令（输出重定向到文件）：**

Windows CMD:
```bash
cd project_2_code\section_1
python main.py > output_cartpole.txt 2>&1
```

Windows PowerShell:
```powershell
cd project_2_code\section_1
python main.py 2>&1 | Tee-Object -FilePath output_cartpole.txt
```

Linux/Mac:
```bash
cd project_2_code/section_1
python main.py 2>&1 | tee output_cartpole.txt
```

### 1.2 运行 Pendulum-v0 实验

**修改 `main.py` 第 126-127 行：**
```python
# game = 'CartPole-v0'
game = 'Pendulum-v0'
```

**运行命令：**
```bash
cd project_2_code\section_1
python main.py > output_pendulum.txt 2>&1
```

### 1.3 输出文件说明

运行后会在以下目录生成文件：

| 目录/文件 | 说明 |
|-----------|------|
| `data/REINFORCEAgent_CartPole-v0_<time>/` | REINFORCE 训练数据 |
| `data/A2CAgent_CartPole-v0_<time>/` | A2C 训练数据 |
| `data/*/return*.txt` | 每个 episode 的 return |
| `data/*/policy_loss*.txt` | 每个 episode 的 policy loss |
| `data/*/value_loss*.txt` | A2C 的 value loss |
| `log/` | 控制台日志 |
| `tf_log/` | TensorBoard 日志 |
| `*.png` | 训练曲线图 |

### 1.4 使用 TensorBoard 查看训练曲线

```bash
tensorboard --logdir=tf_log
```

然后在浏览器打开 `http://localhost:6006`

---

## Section 2: PPO-Clip

### 2.1 运行 PPO 在 CartPole-v1

**运行命令：**
```bash
cd project_2_code\section_2
python ppo.py --seed 0 --algorithm PPO --figure CartPole_PPO_output.png
```

**输出重定向：**
```bash
cd project_2_code\section_2
python ppo.py --seed 0 --algorithm PPO > output_ppo.txt 2>&1
```

### 2.2 多种子实验

为了验证结果稳定性，可以运行多种子实验：

```bash
cd project_2_code\section_2
python ppo.py --seed 0 --algorithm PPO --figure CartPole_PPO_seed0.png
python ppo.py --seed 1 --algorithm PPO --figure CartPole_PPO_seed1.png
python ppo.py --seed 2 --algorithm PPO --figure CartPole_PPO_seed2.png
```

### 2.3 PPO 超参数

在 `ppo.py` 中定义的超参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `N` | 1024 | batch size |
| `M` | 32 | 更新迭代次数 |
| `MINIBATCH_SIZE` | 128 | minibatch 大小 |
| `GAMMA` | 0.99 | 折扣因子 |
| `LR` | 1e-4 | 学习率 |
| `EPS` | 0.1 | PPO clip 参数 |

---

## 实验流程总结

### 完整实验步骤

1. **安装依赖**
   ```bash
   pip install torch numpy matplotlib gymnasium tensorboardX
   ```

2. **Section 1: CartPole-v0**
   ```bash
   cd project_2_code\section_1
   # 确保 main.py 中 game = 'CartPole-v0'
   python main.py > output_cartpole.txt 2>&1
   ```

3. **Section 1: Pendulum-v0**
   ```bash
   cd project_2_code\section_1
   # 修改 main.py 中 game = 'Pendulum-v0'
   python main.py > output_pendulum.txt 2>&1
   ```

4. **Section 2: PPO**
   ```bash
   cd project_2_code\section_2
   python ppo.py --seed 0 --algorithm PPO > output_ppo.txt 2>&1
   ```

5. **查看结果**
   - 训练曲线图: `data/*.png` 或 `CartPole_PPO_output.png`
   - TensorBoard: `tensorboard --logdir=tf_log`

---

## 报告所需数据

### REINFORCE (CartPole-v0)
- 训练 return 曲线: `data/REINFORCEAgent_CartPole-v0_*/return*.txt`
- Policy loss 曲线: `data/REINFORCEAgent_CartPole-v0_*/policy_loss*.txt`
- 最终评估 return: 查看输出文件末尾的 `episodic_return_test`

### A2C (CartPole-v0)
- 训练 return 曲线: `data/A2CAgent_CartPole-v0_*/return*.txt`
- Policy loss 曲线: `data/A2CAgent_CartPole-v0_*/policy_loss*.txt`
- Value loss 曲线: `data/A2CAgent_CartPole-v0_*/value_loss*.txt`
- 最终评估 return: 查看输出文件末尾的 `episodic_return_test`

### PPO (CartPole-v1)
- 训练曲线图: `CartPole_PPO_output.png`
- 最大 running average: 查看图片标题

---

## 注意事项

1. **CartPole-v0 vs CartPole-v1**: gymnasium 中 `CartPole-v0` 已弃用，建议使用 `CartPole-v1`。代码中已兼容处理。

2. **训练时间**:
   - REINFORCE CartPole: ~2e5 steps, 约 5-10 分钟
   - A2C CartPole: ~2e5 steps, 约 5-10 分钟
   - PPO CartPole: 2000 episodes, 约 10-20 分钟

3. **输出重定向**: 使用 `> output.txt 2>&1` 可以将所有输出保存到文件，避免终端刷屏。

4. **种子设置**: 为了可复现性，代码中设置了随机种子。可修改 `main.py` 中的 `set_seed()` 参数。