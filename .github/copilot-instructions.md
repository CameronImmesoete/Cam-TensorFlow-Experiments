# Copilot Instructions

> Base instructions: [CameronImmesoete/.github/.github/copilot-instructions.md@1f79bfb](https://github.com/CameronImmesoete/.github/blob/1f79bfb3e9eee277d05ecdd3332220204cb0f38b/.github/copilot-instructions.md)

## Repository-Specific Guidelines

This is a TensorFlow image classification experiments project.

- Use TensorFlow 2 patterns (Keras API, not tf.compat.v1)
- Model architecture: verify layer shapes and activation functions
- Image preprocessing: consistent normalization and augmentation
- Training reproducibility: configurable random seeds
- GPU memory management: use tf.data pipelines with prefetch
