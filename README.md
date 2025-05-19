Toxicity-Aware Dialogue Summarization using PPO and LoRA (FLAN-T5)
Tools: PyTorch, Hugging Face (Transformers, TRL, PEFT), PPO, Evaluate, LoRA, bfloat16

Completed as part of the AWS Generative AI with LLMs specialization, this project involved building a detoxification pipeline for dialogue summarization using flan-t5-base, fine-tuned with LoRA (0.59% of 570M params trainable) and optimized using PPO (Reinforcement Learning).

Processed 12k+ dialogue samples from the Hugging Face dialogsum dataset.

Used facebook/roberta-hate-speech-dynabench as a reward model to reduce toxicity.

Achieved a 39.7% reduction in average toxicity (from 0.278 to 0.167) on generated summaries.

Integrated dynamic sampling with LengthSampler and deployed training with multi-GPU support in bfloat16.

