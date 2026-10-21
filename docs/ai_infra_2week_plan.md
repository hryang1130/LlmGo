# AI Infra 两周冲刺计划（80–85 小时）

> 目标不是"把课刷完"，而是用两周产出三份**可复现、能讲清楚**的实验报告。
>
> 时间假设：工作日每天 6 小时，周末每天 8 小时，两周合计约 84 小时。
> 前置条件：已经跑通过本仓库的 train → export → quantize → deploy 链路。

---

## 0. 使用说明

- 每天分三个 block：上午读、下午动手、晚上写。
- **每天必须有数字产出**，没有跑出结果的"学习"不算完成。
- 阅读时间不超过当天的 30%，剩下时间用来做实验和写报告。
- 所有实验记录写进 `out/reports/` 或 `out/lab/`，最后直接汇总成报告。
- 文中的链接以官方地址为准；如果路径变动，在对应仓库或站点里搜标题即可。
- 本计划配套的 Q4_K 数值详解见 [q4_k_m_quantization.md](./q4_k_m_quantization.md)。
- 开工前先跑 `make check`；如果 Python 依赖不在 `PATH`，先设置
  `set PYTHON_CMD=D:\envs\llm_go\Scripts\python.exe`（改成你自己的虚拟环境）。

---

## 1. 两周验收目标

### 交付物 1：`out/quant_report.md`

必须包含：

- Q4_K_M 的 BPW 推导（为什么是 4.99 而不是 4.0）；
- Q8_0 / Q4_K_S / Q4_K_M / IQ4_XS 的体积、BPW、PPL、解码速度；
- imatrix 开/关 A/B；
- 对"绝对 PPL 为什么这么高"的解释，以及相对指标的使用方式；
- 至少一个自己实现的量化/反量化实验。

验收问题：你能不看资料画出 `block_q4_K` 的布局，并手算一个子块的误差吗？

### 交付物 2：`out/serving_report.md`

必须包含：

- 并发 1/2/4/8/16 下的 TTFT、TPOT、吞吐、RSS、P99；
- 吞吐平台期出现的位置和原因；
- 投机解码的接收率、加速比，以及理论值与实测值的差距；
- 一次 roofline 分析：decode 为什么是 memory-bound。

验收问题：你能解释"为什么并发上去后吞吐不再涨，而延迟继续涨"吗？

### 交付物 3：Agent Runtime demo + `out/agent_report.md`

必须包含：

- 工具调用循环（至少一个工具）；
- 超时、重试、幂等；
- tracing（每次请求的 span / 日志）；
- tokens/task、每任务成本、成功率、P99。

验收问题：工具超时、重复请求、上游 500 时，你的 runtime 分别怎么处理？

---

## 1.5 两周总览

| 天 | 主题 | 主要产出 |
|---|---|---|
| D1 | 量化基础 | 粒度 vs MSE 误差表 |
| D2 | Q4_K 源码 + Python 实现 | `q4k_ref.py` + BPW/误差表 |
| D3 | PTQ 方法与 importance weighting | 加权 vs 不加权对比 |
| D4 | 正确的 benchmark | `quant_results.csv` + 报告草稿 |
| D5 | KV cache 量化 | KV 前后速度/内存对比 |
| D6 | 写量化报告 | `out/quant_report.md` |
| D7 | 缓冲 / 可选 QAT | 补漏或 QAT 记录 |
| D8 | Roofline + llama-bench | 带宽/算力利用率表 |
| D9 | PagedAttention / 调度 | 调度流程图 |
| D10 | 并发压测 | 并发–吞吐/P99 曲线 |
| D11 | 投机解码 | 接收率 + 加速比表 |
| D12 | Agent Runtime | demo + `agent_report.md` 草稿 |
| D13 | 成本与路由 | 每任务成本表 |
| D14 | 收尾 + 模拟面试 | 三份报告 + 5 分钟讲述 |

---

## 2. 每日节奏

| 时段 | 时长 | 内容 |
|---|---:|---|
| 上午 | 2h | 读课程/论文/源码，只读当天要用的部分 |
| 下午 | 3–4h | 动手实验，必须留下命令、数据、图表 |
| 晚上 | 1h | 写笔记 + 更新报告 |

连续学习超过 90 分钟效率会下降，建议每 90 分钟休息 15 分钟。

每天结束前写 5 行：

```markdown
## D<N>
- 今天验证了：
- 与预期不符：
- 明天要补：
- 新增文件：
- 一个新问题：
```

---

## 3. Week 1：量化（约 41 小时）

### D1：量化基础与手工实现（6h）

**上午（2h）**

- 看 MIT 6.5940 的量化相关讲座（2 讲左右）：
  [课程主页](https://hanlab.mit.edu/courses/2024-fall-65940) ·
  [YouTube 搜索](https://www.youtube.com/results?search_query=MIT+6.5940+TinyML) ·
  [B 站搜索](https://search.bilibili.com/all?keyword=MIT%206.5940)
- 整理分类：PTQ vs QAT、weight-only vs weight+activation、per-tensor / per-channel / per-group、对称 vs 非对称。

**下午（3h）**

- 从 `models/tinyllm-hf/model.safetensors` 里取一个真实权重矩阵；
- 用 PyTorch 实现三种粒度：

```python
# per-tensor
scale = w.abs().max() / 7
q = (w / scale).round().clamp(-8, 7)
w_hat = q * scale

# per-channel（沿输出维度）
scale = w.abs().amax(dim=1, keepdim=True) / 7
q = (w / scale).round().clamp(-8, 7)
w_hat = q * scale

# per-group（每 32 或 128 个元素一组）
```

- 对每种粒度计算 MSE、最大误差、压缩后的 bit/权重。

**晚上（1h）**

- 写 `out/notes/d1_quant_basics.md`。

**产出**：一张误差表（粒度 vs MSE vs bit/权重）。

**验收**：为什么 per-group 通常比 per-channel 更适合 LLM 权重？outlier 在其中扮演什么角色？

---

### D2：读 Q4_K 源码 + Python 参考实现（6h）

**上午（2h）**

- 读 llama.cpp 的量化源码：
  [ggml-quants.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c) ·
  [llama-quant.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp) ·
  [quantize 工具说明](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize)
- 对照本仓库的 [q4_k_m_quantization.md](./q4_k_m_quantization.md)，确认：
  - `d` / `dmin` 的含义；
  - 8 个 6-bit scale/min 的打包方式；
  - `qs` 的 nibble 交错顺序；
  - `dequantize_row_q4_K` 里的减号。

**下午（4h）**

- 用 Python 实现 `quantize_row_q4_K_ref` 和 `dequantize_row_q4_K`；
- 从 F16 GGUF 或 safetensors 读一个真实张量，量化后再反量化，计算 MSE 与 BPW；
- 工具：
  [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py) ·
  [GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

**晚上（1h）**

- 写 `out/notes/d2_q4k_ref.md`，附上你手算的一个子块。

**产出**：`out/lab/q4k_ref.py` + 误差表。

**验收**：你的 BPW 是否等于 4.5？误差量级是否和 llama.cpp 日志一致？

---

### D3：PTQ 方法与 importance weighting（6h）

**上午（2h）**

读三篇方法的核心章节（只读方法和实验设置，不逐字读）：

- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)

可选补充：[LLM.int8()](https://arxiv.org/abs/2208.07339)。

**下午（3h）**

- 实现一个简化版 importance-weighted 量化：
  - 用一小段校准文本跑前向，统计每列激活的 L2 范数；
  - 把 importance 作为权重，最小化 `Σ importance_i × (w_i - w_hat_i)^2`；
  - 对比"不加权"和"加权"的 MSE。
- 这一步是在模拟 imatrix 的思想，不要求复现官方实现。

**晚上（1h）**

- 记录：哪些张量最受益？为什么？

**产出**：加权 vs 不加权的误差对比表。

**验收**：imatrix 为什么能改善 4-bit 量化？它改变了什么，没有改变什么？

---

### D4：搭正确的 benchmark（6h）

**上午（1h）**

- 准备一份**固定不变**的留出语料，至少能切出 5 个 512-token chunk；
- 可以用现有 [data/eval.txt](../data/eval.txt) 扩写，或使用：
  [Project Gutenberg 中文](https://www.gutenberg.org/browse/languages/zh) ·
  [THUCNews](http://thuctc.thunlp.org/) ·
  [CLUE](https://github.com/CLUEbenchmark/CLUE) ·
  [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)

注意：本项目的绝对 PPL 约为 2000，这是小模型 + 小语料的泛化问题，不是量化造成的。**只看相对变化，不看绝对值。**

**下午（4h）**

1. 先跑不带 imatrix 的一组：

```powershell
python pipeline/quantize_sweep.py --llama-cpp D:/tools/llama.cpp `
  --schemes Q8_0,Q4_K_S,Q4_K_M,IQ4_XS --imatrix none --force
```

2. 把产物改名保存，避免下一轮覆盖：

```powershell
Rename-Item models\tinyllm-q4_k_m.gguf models\tinyllm-q4_k_m_noimat.gguf
```

3. 再跑带 imatrix 的一组：

```powershell
python pipeline/quantize_sweep.py --llama-cpp D:/tools/llama.cpp `
  --schemes Q8_0,Q4_K_S,Q4_K_M,IQ4_XS --imatrix auto --force
```

4. 显式指定模型列表做评测（不要用默认 glob，否则会把 target/SFT 模型混进来）：

```powershell
python pipeline/eval.py --llama-cpp D:/tools/llama.cpp `
  --models models/tinyllm-f16.gguf models/tinyllm-q8_0.gguf `
           models/tinyllm-q4_k_s.gguf models/tinyllm-q4_k_m.gguf `
           models/tinyllm-iq4_xs.gguf `
  --rounds 3 --kv-quant
```

相关脚本：[quantize_sweep.py](../pipeline/quantize_sweep.py) ·
[eval.py](../pipeline/eval.py)

**晚上（1h）**

- 把结果整理成 `out/lab/quant_results.csv`；
- 列建议：`scheme,size_mb,bpw,ppl,ppl_delta,decode_tok_s,ttft_ms,rss_mb,imatrix,notes`。

**产出**：CSV + `out/quant_report.md` 草稿。

**验收**：你能解释为什么 4-bit 文件不是 F16 的 1/4 吗？

---

### D5：KV cache 量化与激活量化（5h）

**上午（2h）**

- 复习 [LLM.int8()](https://arxiv.org/abs/2208.07339) 和 [SmoothQuant](https://arxiv.org/abs/2211.10438) 的核心思想；
- 读 llama.cpp server 的缓存参数说明：
  [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)

**下午（2h）**

- 跑一次带 `--kv-quant` 的评测（D4 的命令已经包含）；
- 对比 KV q8_0 前后的解码速度、RSS、PPL；
- 如果支持，再试 `--cache-type-k q8_0 --cache-type-v q8_0`。

**晚上（1h）**

- 记录：KV 量化省的是内存还是带宽？为什么 CPU 上速度可能反而下降？

**产出**：KV 量化对比表。

**验收**：KV cache 大小怎么估算？与层数、上下文长度、KV 头数、位宽是什么关系？

---

### D6：写量化报告 + 5 分钟讲述（8h）

**上午（2h）**

- 精读 [GPTQ](https://arxiv.org/abs/2210.17323) 和 [AWQ](https://arxiv.org/abs/2306.00978)；
- 各写 5 条要点：假设、近似、校准数据、误差、部署代价。

**下午（4h）**

- 完成 `out/quant_report.md`：
  - Q4_K_M 的 BPW 推导；
  - 四种方案对比；
  - imatrix A/B；
  - 自己的 Python 量化实验；
  - 对 PPL 绝对值的解释。
- 画两张图：体积 vs BPW、PPL delta vs 方案。

**晚上（2h）**

- 准备 5 分钟讲述稿，录音一次，检查是否能在 5 分钟内讲完。

**产出**：最终版 `out/quant_report.md`。

---

### D7：缓冲 / 可选 QAT（4h）

二选一：

- 补前六天没做完的实验；
- 做一次 fake-quant + 少量微调：在 tiny 模型上插入量化/反量化，训练几十步，观察 loss 变化。

如果前面进度正常，建议休息半天，避免第二周崩盘。

---

## 4. Week 2：推理系统 + Agent Runtime（约 43 小时）

### D8：Roofline 与 llama-bench（6h）

**上午（2h）**

- 复习 roofline 模型与内存层次：
  [Roofline 论文](https://dl.acm.org/doi/10.1145/1498765.1498785) ·
  [CS149](https://gfxcourses.stanford.edu/cs149/) ·
  [CS336](https://stanford-cs336.github.io/spring2025/)
- 记住两个判断：算术强度低 → memory-bound；算术强度高 → compute-bound。

**下午（3h）**

- 用 llama-bench 测 prefill（pp）和 decode（tg）：

```powershell
D:\tools\llama.cpp\bin\llama-bench.exe -m models\tinyllm-f16.gguf -p 512 -n 128
D:\tools\llama.cpp\bin\llama-bench.exe -m models\tinyllm-q4_k_m.gguf -p 512 -n 128
```

工具说明：[llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)

**计算**：

- decode 有效带宽 ≈ `模型字节数 × tok/s`；
- prefill 有效算力 ≈ `2 × 参数量 × prompt_tokens / 时间`；
- 和机器的理论内存带宽 / 峰值算力对比，算出利用率。

**晚上（1h）**

- 写 `out/notes/d8_roofline.md`。

**产出**：F16 vs Q4 的 pp/tg 和带宽/算力利用率表。

**验收**：为什么 decode 阶段加 GPU 算力往往没用，而加带宽有用？

---

### D9：PagedAttention / continuous batching（6h）

**上午（2h）**

- 读 [PagedAttention / vLLM 论文](https://arxiv.org/abs/2309.06180)；
- 看 [vLLM 文档](https://docs.vllm.ai/) 里的 scheduler、KV cache、prefix caching 章节；
- 可选：[SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)。

**下午（3h）**

- 画一张图：请求 → scheduler → block manager → KV cache → worker；
- 把概念映射到 llama-server 的 `--parallel` slot：一个 slot 对应一条序列，slot 之间共享 batch。
- 如果有 GPU，安装 vLLM 并跑一个 0.5B 模型；没有 GPU 就跳过安装，只做概念映射。

**晚上（1h）**

- 写 `out/notes/d9_scheduler.md`。

**产出**：调度流程图 + 一页术语表。

**验收**：static batching 和 continuous batching 的区别是什么？KV cache 碎片从哪来？

---

### D10：并发压测（6h）

**上午（1h）**

- 用附录 A 的脚本（或自己写）做压测；
- 统一 prompt、`n_predict`、temperature=0，保证可比。

**下午（4h）**

- 启动 llama-server：

```powershell
D:\tools\llama.cpp\bin\llama-server.exe -m models\tinyllm-q4_k_m.gguf `
  --host 127.0.0.1 --port 8081 --ctx-size 512 --parallel 4
```

- 依次跑 concurrency = 1 / 2 / 4 / 8 / 16；
- 记录：TTFT、TPOT、总吞吐、P99、服务端 RSS；
- 再换 `--parallel 1 / 2 / 8`，观察 slot 数的影响。

相关文档：[llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)

**晚上（1h）**

- 画并发–吞吐曲线和并发–P99 曲线。

**产出**：`out/lab/load_results.csv` + `out/serving_report.md` 草稿。

**验收**：吞吐平台期出现在哪里？此时瓶颈是算力、带宽还是调度？

---

### D11：投机解码（6h）

**上午（2h）**

- 读 [Leviathan et al.](https://arxiv.org/abs/2211.17192) 和 [Chen et al.](https://arxiv.org/abs/2302.01318)；
- 复习本仓库的 [spec_decode.py](../pipeline/spec_decode.py)。

**下午（3h）**

- 跑投机解码对比：

```powershell
python pipeline/spec_decode.py --llama-cpp D:/tools/llama.cpp `
  --draft models/tinyllm-q4_k_m.gguf `
  --target models/tinyllm-target-q4_k_m.gguf `
  --draft-max 8 --rounds 3
```

- 确认 llama-server 启动时带了 `--metrics`；
- 用 `curl http://127.0.0.1:PORT/metrics` 查 `spec_decode` / `draft` 相关计数器；
- 调整 `draft_max`、`draft_min`、`draft_p_min`，记录接收率与加速比。

**晚上（1h）**

- 计算理论加速上限 `(1-α^(γ+1))/(1-α)`，和实测对比。

**产出**：接收率 + 加速比表。

**验收**：为什么实测加速比远低于理论上限？

---

### D12：最小 Agent Runtime（7h）

**上午（1h）**

- 设计状态机：`received → planning → tool_call → observing → final`；
- 定义失败路径：工具超时、工具报错、上游 500、重复请求。

**下午（5h）**

- 在现有 Go 网关上实现：
  - 至少一个工具（如 `time`、`calculator` 或只读文件查询）；
  - 工具调用循环（最多 N 轮）；
  - 超时与重试（带退避）；
  - 幂等键（同一请求重复到达不重复执行工具）；
  - tracing：每个请求一个 trace id，记录每步耗时；
  - 成本统计：prompt tokens、completion tokens、估算费用。

参考：
[OpenTelemetry Go](https://opentelemetry.io/docs/languages/go/) ·
[Temporal](https://docs.temporal.io/) ·
[Firecracker](https://firecracker-microvm.github.io/) ·
[gVisor](https://gvisor.dev/)

**晚上（1h）**

- 用 3 个用例测试：正常、工具超时、重复请求。

**产出**：可运行的 demo + `out/agent_report.md` 草稿。

**验收**：重复请求会导致工具被执行两次吗？你的幂等键放在哪一层？

---

### D13：成本与路由（6h）

**上午（1h）**

- 定义成本指标：`tokens/task`、`¥/task`、`¥/successful_task`；
- 明确输入 token、输出 token、缓存命中的计价方式。

**下午（4h）**

- 做三组 A/B：
  1. 全部走 target vs 简单请求走 draft；
  2. 有缓存 vs 无缓存；
  3. Q4_K_M vs Q8_0（质量换成本）。
- 记录每组的成功率、P99、每任务成本。

**晚上（1h）**

- 画成本–质量–延迟三角图（至少一张表）。

**产出**：成本对比表。

**验收**：哪类请求适合路由到小模型？路由错误会带来什么代价？

---

### D14：收尾与模拟面试（6h）

**上午（3h）**

- 完成三份报告，确保每个数字都能追溯到命令和日志；
- 检查所有图表都有单位、样本量、环境说明。

**下午（2h）**

模拟回答这 5 个问题：

1. Q4_K_M 的 4.99 BPW 是怎么来的？
2. 为什么量化后 PPL 变化很小/很大？你怎么判断是否可信？
3. 并发上去后吞吐为什么不再涨？瓶颈怎么定位？
4. 投机解码的接收率怎么测？为什么加速比低于理论值？
5. Agent Runtime 怎么保证超时、重试、幂等和成本可控？

**晚上（1h）**

- 写下一步计划：继续深挖 kernel、编译，还是转向 Agent 产品。

**产出**：三份最终报告 + 5 分钟讲述稿。

---

## 5. 命令速查

| 目的 | 命令 |
|---|---|
| 训练 draft | `make train` |
| 训练 target | `make train-target` |
| 导出 + 量化 | `make export` / `make quantize` |
| 多方案量化 | `python pipeline/quantize_sweep.py --llama-cpp D:/tools/llama.cpp --schemes Q8_0,Q4_K_M,Q4_K_S,IQ4_XS --imatrix auto --force` |
| 评测 | `python pipeline/eval.py --llama-cpp D:/tools/llama.cpp --models <显式列表> --rounds 3 --kv-quant` |
| 投机解码 | `python pipeline/spec_decode.py --llama-cpp D:/tools/llama.cpp --draft <draft> --target <target>` |
| 原始性能 | `llama-bench -m <model> -p 512 -n 128` |
| 查看指标 | `curl http://127.0.0.1:8081/metrics` |

---

## 6. 阅读清单

### 6.1 必读（按顺序）

| 主题 | 材料 | 时间 | 链接 |
|---|---|---:|---|
| 量化全景 | MIT 6.5940 量化讲座 | 3h | [课程](https://hanlab.mit.edu/courses/2024-fall-65940) |
| 量化综述 | A Survey of Quantization Methods | 1h | [arXiv](https://arxiv.org/abs/2103.13630) |
| 4-bit 精度 | The Case for 4-bit Precision | 1h | [arXiv](https://arxiv.org/abs/2212.09720) |
| GPTQ | GPTQ 论文 | 1.5h | [arXiv](https://arxiv.org/abs/2210.17323) |
| AWQ | AWQ 论文 | 1.5h | [arXiv](https://arxiv.org/abs/2306.00978) |
| 激活量化 | SmoothQuant | 1h | [arXiv](https://arxiv.org/abs/2211.10438) |
| 服务调度 | PagedAttention / vLLM | 2h | [arXiv](https://arxiv.org/abs/2309.06180) |
| 投机解码 | Leviathan et al. | 1h | [arXiv](https://arxiv.org/abs/2211.17192) |
| 投机采样 | Chen et al. | 1h | [arXiv](https://arxiv.org/abs/2302.01318) |

### 6.2 选读

| 主题 | 材料 | 链接 |
|---|---|---|
| LLM 全栈 | Stanford CS336 | [课程](https://stanford-cs336.github.io/spring2025/) |
| 编译部署 | MLC（有中文版） | [mlc.ai](https://mlc.ai/zh/) |
| 系统底子 | CMU 10-414/714 | [dlsyscourse](https://dlsyscourse.org/) |
| 并行计算 | Stanford CS149 | [课程](https://gfxcourses.stanford.edu/cs149/) |
| 连续批处理 | Orca (OSDI'22) | [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu) |
| FlashAttention | Dao et al. | [arXiv](https://arxiv.org/abs/2205.14135) |
| chunked prefill | Sarathi-Serve | [arXiv](https://arxiv.org/abs/2403.02310) |
| PD 分离 | DistServe | [arXiv](https://arxiv.org/abs/2401.09670) |
| 训练系统 | Megatron-LM | [arXiv](https://arxiv.org/abs/1909.08053) |
| 显存优化 | ZeRO | [arXiv](https://arxiv.org/abs/1910.02054) |

---

## 7. 工具箱

| 类别 | 工具 | 链接 |
|---|---|---|
| 推理运行时 | llama.cpp | [GitHub](https://github.com/ggml-org/llama.cpp) |
| 高吞吐服务 | vLLM | [GitHub](https://github.com/vllm-project/vllm) · [文档](https://docs.vllm.ai/) |
| 结构化生成 | SGLang | [GitHub](https://github.com/sgl-project/sglang) · [文档](https://docs.sglang.ai/) |
| NVIDIA 部署 | TensorRT-LLM | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| 量化工具 | AutoAWQ | [GitHub](https://github.com/casper-hansen/AutoAWQ) |
| 量化工具 | AutoGPTQ | [GitHub](https://github.com/AutoGPTQ/AutoGPTQ) |
| 量化工具 | llm-compressor | [GitHub](https://github.com/vllm-project/llm-compressor) |
| HF 量化文档 | Transformers quantization | [文档](https://huggingface.co/docs/transformers/main/en/quantization/overview) |
| 性能分析 | Nsight Systems | [NVIDIA](https://developer.nvidia.com/nsight-systems) |
| 性能分析 | Nsight Compute | [NVIDIA](https://developer.nvidia.com/nsight-compute) |
| Agent 追踪 | OpenTelemetry Go | [文档](https://opentelemetry.io/docs/languages/go/) |
| 持久化工作流 | Temporal | [文档](https://docs.temporal.io/) |
| 沙箱 | Firecracker | [官网](https://firecracker-microvm.github.io/) |
| 沙箱 | gVisor | [官网](https://gvisor.dev/) |
| 平台 | Kubernetes | [文档](https://kubernetes.io/docs/) |
| 平台 | Ray | [文档](https://docs.ray.io/en/latest/) |
| 平台 | KServe | [文档](https://kserve.github.io/website/) |

---

## 8. 指标定义

| 指标 | 定义 | 注意 |
|---|---|---|
| BPW | bits per weight，平均每个权重占多少 bit | 不等于名义位宽，含 scale/min 元数据 |
| PPL | `exp(平均交叉熵)` | 小模型/小语料的绝对值不可信，只看相对变化 |
| TTFT | 从发出请求到收到第一个 token 的时间 | 包含排队 + prefill |
| TPOT | `(总时间 - TTFT) / (生成 token 数 - 1)` | 反映逐 token 解码速度 |
| 吞吐 | 总生成 token 数 / 总墙钟时间 | 并发越高越接近硬件上限 |
| 接收率 α | 被 target 接受的 draft token / draft 总 token | 投机解码的核心指标 |
| P99 | 99% 请求的延迟上界 | 比平均值更能反映用户体验 |
| 每任务成本 | `(输入 tokens × 单价 + 输出 tokens × 单价) / 成功任务数` | 失败重试也要计入 |

---

## 9. 降级方案

### 只有 40 小时

砍掉 D3、D7、D13，保留：

- D1–D2：量化原理 + Q4_K 实现；
- D4：benchmark；
- D8–D9：roofline + 调度概念；
- D10：并发压测；
- D14：写报告。

### 没有 GPU

- 跳过 vLLM 实测，用 llama-server 做并发实验；
- 重点放在 CPU 上的 memory-bound 分析；
- 报告里明确写"本实验在 CPU 上完成"。

### 没有时间做 Agent

- 只做工具调用 + 超时/重试 + 成本统计；
- tracing 用简单 JSON 日志代替 OpenTelemetry。

---

## 10. 最终讲述提纲

用 5 分钟讲清楚这 5 件事：

1. **问题**：为什么模型需要量化？4 bit 到底省了什么？
2. **方法**：Q4_K 的三层结构，`d`、`dmin`、`sc`、`m`、`q` 各是什么。
3. **证据**：你的体积/BPW/PPL/速度表，imatrix A/B 的结果。
4. **系统**：并发曲线、瓶颈、投机解码接收率。
5. **落地**：Agent Runtime 的可靠性设计和每任务成本。

---

## 附录 A：并发压测脚本

保存为 `out/lab/load_test.py`。它用流式接口近似统计 TTFT/TPOT/吞吐：

```python
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = "人工智能是计算机科学的一个分支，"


def one(url: str, n_predict: int):
    body = json.dumps({
        "prompt": PROMPT,
        "n_predict": n_predict,
        "temperature": 0.0,
        "stream": True,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if data.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
    total = time.perf_counter() - t0
    return ttft, total, chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081/completion")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--n-predict", type=int, default=128)
    args = ap.parse_args()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        rows = list(ex.map(
            lambda _: one(args.url, args.n_predict), range(args.requests)))
    wall = time.perf_counter() - t0

    ttfts = [r[0] for r in rows if r[0] is not None]
    totals = [r[1] for r in rows]
    chunks = sum(r[2] for r in rows)
    tpots = [(t - f) / max(n - 1, 1)
             for f, t, n in rows if f is not None]
    p99 = sorted(totals)[max(0, int(len(totals) * 0.99) - 1)]

    print(f"concurrency={args.concurrency} requests={args.requests}")
    print(f"throughput={chunks / wall:.1f} tok/s  wall={wall:.2f}s")
    print(f"TTFT median={statistics.median(ttfts) * 1000:.0f} ms")
    print(f"TPOT median={statistics.median(tpots) * 1000:.1f} ms")
    print(f"latency p50={statistics.median(totals):.2f}s p99={p99:.2f}s")


if __name__ == "__main__":
    main()
```

说明：`chunks` 是流式 chunk 数，不一定等于 token 数；做相对比较足够，若要求精确 token 数，读响应里的 `timings.predicted_n`。

---

## 附录 B：结果 CSV 建议字段

```text
scheme,size_mb,bpw,ppl,ppl_delta,decode_tok_s,ttft_ms,rss_mb,imatrix,notes
```

并发实验建议字段：

```text
concurrency,requests,n_predict,ttft_ms,tpot_ms,throughput_tok_s,p50_s,p99_s,rss_mb
```

Agent 实验建议字段：

```text
task_id,success,rounds,prompt_tokens,completion_tokens,cost,latency_s,retries,trace_id
```

---

## 附录 C：链接维护

- 课程和论文链接以官方页面为准；
- llama.cpp 的源码路径可能随版本调整，404 时在仓库内搜文件名（如 `ggml-quants.c`）；
- vLLM / SGLang / TensorRT-LLM 的文档更新很快，以 `docs` 目录为准；
- 所有实验都要记录 llama.cpp 的 build 号（`llama-server --version`），否则数字不可复现。
