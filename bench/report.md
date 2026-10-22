# Qwen2.5-3B QLoRA 微调 + 量化加速报告

硬件: RTX 4060 Ti 8GB / i5-13400F / 32GB RAM · llama.cpp b10909 CUDA 12.4 · ngl=99 (全部 offload)
基准: prompt 512 tok / decode 128 tok, 3 次取平均

## 训练信息

- 基座: Qwen/Qwen2.5-3B-Instruct (3.09B 参数)
- 数据: shibing624/alpaca-zh, 1200 条中文指令
- 方法: QLoRA (NF4 4bit + double quant, LoRA r=16 α=32, 全线性层)
- 训练: 1 epoch / seq 768 / batch 1×8 accum / lr 2e-4 cosine / paged_adamw_8bit
- 耗时: 29 分钟 (RTX 4060 Ti), 显存占用约 5-6GB
- loss: 1.78 → 1.42

## 量化加速对比 (微调后模型, RTX 4060 Ti)

| 模型 | 类型 | 体积 | prompt t/s | decode t/s | decode 加速比 |
|------|------|------|-----------|-----------|--------------|
| qwen25-3b-ft-q4_k_m.gguf | qwen2 3B Q4_K - Medium | 1.92 GB | 6495.5 | 107.2 | 3.63x |
| qwen25-3b-ft-q8_0.gguf | qwen2 3B Q8_0 | 3.28 GB | 6424.3 | 66.5 | 2.25x |
| qwen25-3b-ft-f16.gguf | qwen2 3B F16 | 6.17 GB | 4208.8 | 29.6 | 1.00x |

基座模型同配置测试 (bench/base/bench.md): F16 在 8GB 卡上仅 13.5 tok/s (权重+KV 挤爆显存被挤出),
Q4_K_M 达 108.4 tok/s (**8.03x**) —— 显存越紧张, 量化收益越大。

## 结论

1. **decode 是显存带宽瓶颈**: Q4_K_M 体积仅为 F16 的 31%, decode 快 3.6~8 倍
2. **8GB 显存上 F16 3B 无法完整驻留**, 量化不只是加速, 更是"能不能跑"的问题
3. Q4_K_M 精度损失对指令跟随几乎无感 (实测回答连贯, 87-97 tok/s 实际生成速度)

## 产物

- `models/qwen25-3b-lora/` — LoRA 适配器 (144 MB)
- `models/qwen25-3b-merged/` — 合并后 FP16 模型 (HF 格式)
- `models/qwen25-3b-ft-{f16,q8_0,q4_k_m}.gguf` — 三档量化
- `bench/base/bench.md` / `bench/finetuned/bench.md` — llama-bench 原始结果
