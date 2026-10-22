# 量化加速基准 (ngl=99, pp=512, tg=128, reps=3)

| 模型 | 类型 | 体积 | prompt t/s | decode t/s | decode 加速比 |
|------|------|------|-----------|-----------|--------------|
| qwen25-3b-ft-q4_k_m.gguf | qwen2 3B Q4_K - Medium | 1.92 GB | 6495.5 | 107.2 | 3.63x |
| qwen25-3b-ft-q8_0.gguf | qwen2 3B Q8_0 | 3.28 GB | 6424.3 | 66.5 | 2.25x |
| qwen25-3b-ft-f16.gguf | qwen2 3B F16 | 6.17 GB | 4208.8 | 29.6 | 1.00x |
