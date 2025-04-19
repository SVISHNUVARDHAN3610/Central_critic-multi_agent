# 🤖 Central Critic Multi-Agent in Reinforcement Learning

A project that explores the power of centralized training and decentralized execution in multi-agent reinforcement learning using a Central Critic architecture. Implemented with MADDPG (Multi-Agent Deep Deterministic Policy Gradient), this system enables agents to collaborate or compete efficiently in complex environments.

---

## 📖 Project Description

This project implements a **Central Critic architecture** within a **Multi-Agent Reinforcement Learning (MARL)** framework. Each agent independently learns its policy (actor), while a shared **centralized critic** evaluates their actions using the global state and all agent actions. This architecture helps stabilize learning in environments where agent policies are continuously changing (non-stationary).

Using the **MADDPG algorithm**, agents are trained in simulated environments like **Multi-Agent Particle Environment (MPE)** and **PyBullet**, allowing for experimentation with both cooperative and competitive tasks.

---

## ⚙️ Key Features

- ✅ Centralized training with decentralized execution
- 🎯 Implements Multi-Agent DDPG (MADDPG)
- 🌍 Compatible with MPE and PyBullet environments
- 📈 Evaluation and visualization of agent behavior
- 🧠 Solves cooperative and adversarial scenarios

---

## 🛠️ Technologies Used

- Python
- PyTorch / TensorFlow
- OpenAI Gym
- Multi-Agent Particle Environment (MPE)
- PyBullet
- NumPy, Matplotlib

---

