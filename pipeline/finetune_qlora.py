#!/usr/bin/env python
"""
Qwen2.5 QLoRA 微调 + 权重合并

用法:
    # 4bit QLoRA 微调 (默认)
    python pipeline/finetune_qlora.py

    # 只合并已有 adapter
    python pipeline/finetune_qlora.py --merge-only

    # 显存不够时改用 bf16 LoRA
    python pipeline/finetune_qlora.py --no-4bit --seq-len 512

产物:
    models/qwen25-3b-lora/      LoRA 适配器
    models/qwen25-3b-merged/   合并后的 FP16 模型 (供 GGUF 导出)
"""
import argparse
import os
import sys
from pathlib import Path

# HF 官方域名在部分网络下不通, 默认走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATASET = "shibing624/alpaca-zh"
DEFAULT_ADAPTER = ROOT / "models" / "qwen25-3b-lora"
DEFAULT_MERGED = ROOT / "models" / "qwen25-3b-merged"
LOCAL_ALPACA = ROOT / "data" / "alpaca-zh" / "alpaca_gpt4_data_zh.json"


def build_texts(example):
    """alpaca 格式 -> (prompt, response)"""
    instruction = (example.get("instruction") or "").strip()
    extra = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()
    prompt = instruction if not extra else f"{instruction}\n\n{extra}"
    return prompt, output


def tokenize_example(tok, example, seq_len):
    prompt, response = build_texts(example)
    if not prompt or not response:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    msgs = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    resp_ids = tok(response + tok.eos_token, add_special_tokens=False)["input_ids"]

    ids = (prompt_ids + resp_ids)[:seq_len]
    labels = ([-100] * len(prompt_ids) + resp_ids)[:seq_len]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def load_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def do_train(args):
    print(f"[finetune] base model : {args.model}")
    print(f"[finetune] dataset    : {args.dataset}")
    print(f"[finetune] 4bit       : {not args.no_4bit}")

    tok = load_tokenizer(args.model)

    if not torch.cuda.is_available():
        raise SystemExit("[finetune] 未检测到 CUDA, 请先安装 CUDA 版 torch")

    quant_cfg = None
    if not args.no_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"[finetune] 加载数据集 {args.dataset} ...")
    if args.dataset == DEFAULT_DATASET and LOCAL_ALPACA.exists():
        print(f"[finetune] 使用本地文件: {LOCAL_ALPACA}")
        ds = load_dataset("json", data_files=str(LOCAL_ALPACA), split="train")
    else:
        ds = load_dataset(args.dataset, split="train")
    if args.samples and args.samples < len(ds):
        ds = ds.select(range(args.samples))
    ds = ds.map(lambda ex: tokenize_example(tok, ex, args.seq_len),
                remove_columns=ds.column_names, desc="tokenize")
    ds = ds.filter(lambda ex: len(ex["input_ids"]) > 8)
    print(f"[finetune] 样本数: {len(ds)}")

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)

    targs = TrainingArguments(
        output_dir=str(ROOT / "models" / "qwen25-3b-qlora-ckpt"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    args.adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.adapter))
    tok.save_pretrained(str(args.adapter))
    print(f"[finetune] LoRA 适配器已保存: {args.adapter}")


def do_merge(args):
    print(f"[merge] 合并 {args.adapter} -> {args.merged}")
    tok = load_tokenizer(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), device_map="cpu")
    model = model.merge_and_unload()
    args.merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.merged), safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(str(args.merged))
    print(f"[merge] 完成: {args.merged}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--samples", type=int, default=1200, help="训练样本数, 0=全部")
    p.add_argument("--seq-len", type=int, default=768)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--no-4bit", action="store_true", help="改用 bf16 LoRA (更吃显存)")
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--no-merge", action="store_true")
    args = p.parse_args()

    if args.samples == 0:
        args.samples = None

    if not args.merge_only:
        do_train(args)
    if not args.no_merge:
        do_merge(args)


if __name__ == "__main__":
    sys.exit(main())
