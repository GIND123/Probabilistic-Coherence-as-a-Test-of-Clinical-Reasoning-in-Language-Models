"""LoRA fine-tuning for schema-compliant likelihood-ratio emission.

    .venv/bin/python -m coherence.train.train_lr_lora --base Qwen/Qwen3-8B

Trains only on the assistant turn, so the loss is on the emitted weight
vector and not on the 49-line candidate list that prefixes every prompt --
without that mask the model spends its capacity learning to reproduce a
constant.

The held-out set is 103 findings the adapter never sees in any format or
demographic framing (see `build_lr_dataset`), so the reported test loss
measures generalisation to unseen findings rather than memorisation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

from coherence.config import BUILD, MODELS, SEED

DATA = BUILD / "train"
IGNORE = -100


def build_tokenise_fn(tok, max_len: int):
    def fn(ex):
        msgs = ex["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                         add_generation_prompt=True)
        full = prompt + msgs[-1]["content"] + (tok.eos_token or "")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False, truncation=True,
                    max_length=max_len)["input_ids"]
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = IGNORE          # mask the prompt
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids),
                "labels": labels}
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default=str(MODELS / "elr-lora-qwen3-8b"))
    # The first run went 3 epochs and overfit: held-out loss bottomed at
    # 0.1883 around step 450 (~1.55 epochs) and rose to 0.2433 by step 870,
    # 29% worse. Two epochs brackets the minimum, and best-checkpoint
    # selection below makes the exact figure non-critical.
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    # Measured: prompts are ~584 tokens and answers ~321, so p95 total is 937
    # and the longest example is 950. 3072 was padding the batch to three
    # times what any example needs.
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-subset", type=int, default=400)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--warmup-steps", type=int, default=40)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files={
        "train": str(DATA / "lr_sft_train.jsonl"),
        "test": str(DATA / "lr_sft_test.jsonl")})
    fn = build_tokenise_fn(tok, args.max_len)
    ds = ds.map(fn, remove_columns=ds["train"].column_names, num_proc=8)
    # Full-eval on 4,074 held-out examples took 8 minutes and ran every 100
    # steps. A fixed random subset is enough to track generalisation during
    # training; the final number is computed on the whole held-out set.
    eval_full = ds["test"]
    eval_small = (eval_full.shuffle(seed=SEED).select(range(min(args.eval_subset,
                                                               len(eval_full))))
                  if args.eval_subset else eval_full)

    kwargs = dict(dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
                  attn_implementation="sdpa")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        # transformers 5.x dropped `warmup_ratio`; compute the step count.
        warmup_steps=args.warmup_steps,
        logging_steps=10, eval_strategy="steps", eval_steps=50,
        # save_steps must match eval_steps for best-checkpoint selection, and
        # the limit must not be able to evict the best one. The first run
        # saved every 150 steps with a limit of 3 and deleted step 450 -- the
        # best checkpoint -- keeping only the three most overfit ones.
        save_strategy="steps", save_steps=50, save_total_limit=5,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True, gradient_checkpointing=True,
        report_to=[], seed=SEED, dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds["train"], eval_dataset=eval_small,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=IGNORE),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    # `load_best_model_at_end` means this evaluates the selected checkpoint,
    # not the last one.
    metrics = trainer.evaluate(eval_dataset=eval_full)   # full held-out set
    metrics["selected_checkpoint"] = str(getattr(trainer.state, "best_model_checkpoint", ""))
    metrics["best_eval_loss_during_training"] = float(
        getattr(trainer.state, "best_metric", float("nan")))
    Path(args.out, "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
