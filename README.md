# Reinforcement Learning Algorithms

This repository contains implementations of various reinforcement learning algorithms. The goal is to provide clean, educational implementations of RL algorithms using modern deep learning frameworks.

## Structure

- `Bandits/` - Contains Multi-Armed Bandit algorithm implementations
  - UCB (Upper Confidence Bound) algorithm
  
- `Policy Gradients/` - Contains Policy Gradient algorithm implementations
  - `Reinforce/` - REINFORCE algorithm implementation
    - Implementation uses JAX/Flax for the CartPole environment
    - Includes both Simple and MOE (Mixture of Experts) variants

## Getting Started

Each algorithm directory contains its own README with specific implementation details and usage instructions.

## Dependencies

The implementations primarily use:
- JAX for high-performance computing
- Flax (Neural Networks library for JAX)
- Gymnax for RL environments
- Optax for optimization

## Contributing

Feel free to contribute additional RL algorithm implementations or improvements to existing ones.
