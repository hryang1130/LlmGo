# 量化加速基准 (ngl=99, pp=512, tg=128, reps=3)

| 模型 | 类型 | 体积 | prompt t/s | decode t/s | decode 加速比 |
|------|------|------|-----------|-----------|--------------|
| qwen25-3b-base-q4_k_m.gguf | qwen2 3B Q4_K - Medium | 1.92 GB | 6438.7 | 108.4 | 8.03x |
| qwen25-3b-base-q8_0.gguf | qwen2 3B Q8_0 | 3.28 GB | 6528.2 | 66.6 | 4.94x |
| qwen25-3b-base-f16.gguf | qwen2 3B F16 | 6.17 GB | 237.9 | 13.5 | 1.00x |
