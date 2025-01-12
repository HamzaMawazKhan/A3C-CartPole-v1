# A3C Algorithm Implementation in CartPole-v1 Environment

This repository contains an implementation of the **Asynchronous Advantage Actor-Critic (A3C)** algorithm to train an agent in the CartPole-v1 environment using Python. The project demonstrates reinforcement learning techniques with parallel training, neural network-based policy learning, and performance evaluation.

## Project Overview

The A3C algorithm is a reinforcement learning technique that employs two neural networks:

- **Actor**: Predicts action probabilities for the agent.
- **Critic**: Estimates the value of the current state to guide the actor.

The algorithm supports **parallel training** through multiple threads. Each thread interacts with the environment independently and periodically syncs its local networks with shared global networks.

### Algorithm Workflow

1. **Initialization**:
   - Create global actor-critic networks and shared optimizers.

2. **Parallel Worker Threads**:
   - Each thread interacts with the environment to collect actions, states, and rewards.
   - Compute discounted rewards and calculate losses:
     - **Policy Loss**: Guides updates to the actor network.
     - **Value Loss**: Guides updates to the critic network.
   - Synchronize local networks with global networks by applying computed gradients.

3. **Evaluation**:
   - Test the trained global actor using a greedy policy to evaluate its performance.

### Key Features

- **Parallelism**: Accelerates training by sampling diverse experiences across threads.
- **Stability**: Normalized rewards and smooth loss functions improve convergence.
- **Efficiency**: Shared global parameters minimize redundant computations.

## CartPole-v1 Environment

The CartPole-v1 environment is a classic reinforcement learning task with the following characteristics:

- **State Space**: A 4-dimensional vector representing:
  - Cart position
  - Cart velocity
  - Pole angle
  - Pole angular velocity

- **Action Space**: Two discrete actions—move the cart left or right.

- **Goal**: Keep the pole upright and the cart within track boundaries for as long as possible.

- **Reward**: +1 for every time step the pole remains balanced.

- **Termination**:
  - Pole angle exceeds a threshold.
  - Cart moves out of bounds.
  - Maximum time steps are reached.

## Results

### Training Performance

During training, the agent’s performance improves steadily:
- Initially, rewards are low and inconsistent due to random exploration.
- Over time, as the policy improves, the rewards stabilize and approach the maximum value (500).

### Evaluation Performance

After training, the global actor network achieves the maximum reward (500) consistently during evaluation episodes, indicating that the agent has learned the optimal policy for the CartPole-v1 environment.

