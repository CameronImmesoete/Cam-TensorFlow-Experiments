# Code Review Standards

> Base review standards: [CameronImmesoete/.github/.github/copilot-review-skill.md@1f79bfb](https://github.com/CameronImmesoete/.github/blob/1f79bfb3e9eee277d05ecdd3332220204cb0f38b/.github/copilot-review-skill.md)

## Repository-Specific Review Criteria

### TensorFlow Patterns
- TF2 Keras API used (no legacy tf.compat.v1 patterns)
- Model layer shapes are compatible (input/output dimensions match)
- Activation functions appropriate per layer type
- Loss function matches the classification task

### Training Pipeline
- Data preprocessing is consistent (same normalization for train/test)
- Train/validation/test splits are properly separated (no data leakage)
- Random seeds configurable for reproducibility
- GPU memory managed via tf.data (batching, prefetch, cache)
