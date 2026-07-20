# Reinforcement Learning Lab

Collection of reinforcement learning algorithms implemented from scratch using [Gymnasium](https://gymnasium.farama.org/).

## Projects

| Project | Algorithms | Environment | Key Result |
|---|---|---|---|
| Cliff Walking | SARSA, Q-Learning | `CliffWalking-v1` | See below |

---

## Cliff Walking: SARSA vs Q-Learning

A classic reinforcement learning problem used to demonstrate the difference between **on-policy** and **off-policy** learning.

The agent must navigate a 4x12 grid from start to goal, avoiding a cliff along the bottom edge. Falling off the cliff gives a reward of -100 and resets the agent to start. Every other step gives -1 reward.

### Algorithms

**SARSA (on-policy)**
```
Q[s, a] += alpha * (reward + gamma * Q[s', a'] - Q[s, a])
```
Updates using the action actually taken by the current policy (including exploration).

**Q-Learning (off-policy)**
```
Q[s, a] += alpha * (reward + gamma * max(Q[s']) - Q[s, a])
```
Updates using the best possible next action, regardless of what the policy actually does.

### Results

| Algorithm | Final Reward | Episode Length | Path Learned |
|---|---|---|---|
| SARSA | -17 | 17 | Safe path, one row away from the cliff |
| Q-Learning | -13 | 13 | Optimal path, directly along the cliff edge |

### Key Insight

SARSA learns a **safer** policy because its updates account for the exploration (epsilon-greedy) that happens during training — it "knows" it might randomly step off the cliff, so it stays away from the edge. Q-learning learns the **optimal** policy assuming the agent will always act greedily once trained, so it's willing to walk right along the cliff edge for a shorter path.

### Setup

```bash
pip install -r requirements.txt
```

### Files
- `Cliff_Walking_SARSA.ipynb` — SARSA implementation and training
- `Cliff_Walking_Q_learning.ipynb` — Q-learning implementation and training

---

*More RL algorithms (DQN, Policy Gradient, etc.) will be added here over time.*
