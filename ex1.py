#!/usr/bin/env python
"""
ex1.py - Fine-tune BERT‑base on GLUE MRPC for paraphrase detection.

This script follows the interface required by ANLP Exercise 1.
You can use it both for training and for generating predictions.

Example usage
-------------
Training (with subset of the data)
    python ex1.py \
        --do_train \
        --max_train_samples 2000 \
        --max_eval_samples 800 \
        --num_train_epochs 3 \
        --lr 2e-5 \
        --batch_size 32

Full‑data training with different hyper‑parameters
    python ex1.py --do_train --num_train_epochs 4 --lr 3e-5 --batch_size 16

Prediction (after training)
    python ex1.py --do_predict --model_path outputs --max_predict_samples -1
"""

import argparse
import os
import random
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------
# Argument parsing utilities
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine‑tune BERT on MRPC")

    # Data slicing
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)

    # Training hyper‑parameters
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32)

    # Execution flags
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_predict", action="store_true", help="Run prediction")

    # Misc / paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Checkpoint path (for prediction) or output dir (for training).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")

    return parser.parse_args()


# --------------
# Helper utils
# --------------

def set_seed(seed: int):
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_datasets(tokenizer, args):
    """Load GLUE‑MRPC and tokenize with dynamic padding."""
    raw = load_dataset("glue", "mrpc")

    def tokenize_fn(batch):
        tok = tokenizer(batch["sentence1"], batch["sentence2"], truncation=True)
        # Trainer expects the label column to be named **labels**
        tok["labels"] = batch["label"]
        return tok

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=raw["train"].column_names)

    # Respect max_*_samples arguments
    if args.max_train_samples != -1:
        tokenized["train"] = tokenized["train"].select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        tokenized["validation"] = tokenized["validation"].select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        tokenized["test"] = tokenized["test"].select(range(args.max_predict_samples))

    return tokenized, raw


# --------------
# Main routine
# --------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Configure Weights & Biases automatically if available.
    os.environ.setdefault("WANDB_PROJECT", "anlp_ex1_mrpc")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    datasets, raw_ds = preprocess_datasets(tokenizer, args)

    # Metric definition
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # Finds the longest sequence in that batch only, and pads every other example up to that same length.
    data_collator = DataCollatorWithPadding(tokenizer)

    # Create output directory
    run_name = f"lr{args.lr}_bs{args.batch_size}_ep{args.num_train_epochs}"
    run_dir = Path(args.output_dir) / run_name  # 1 folder per config

    # --------------------
    # Training phase
    # --------------------
    if args.do_train:
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )

        # Trainer arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=10,
            weight_decay=0.01,
            report_to=["wandb"],
            run_name=run_name,
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Final validation accuracy
        val_metrics = trainer.evaluate()
        val_acc = val_metrics["eval_accuracy"]
        print(f"Validation accuracy: {val_acc:.4f}")

        # Append to res.txt in required format
        with open("res.txt", "a", encoding="utf-8") as fh:
            fh.write(
                f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {val_acc:.4f}\n"
            )

        # Persist final model for later prediction
        trainer.save_model(run_dir)

    # --------------------
    # Prediction phase
    # --------------------
    if args.do_predict:
        if not args.model_path:
            raise ValueError("--model_path must be supplied when --do_predict is set.")

        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
        preds = trainer.predict(datasets["test"]) # model.eval() happens inside
        pred_labels = np.argmax(preds.predictions, axis=-1)

        # Calulate and print accuracy
        test_acc = accuracy.compute(predictions=pred_labels, references=raw_ds["test"]["label"])["accuracy"]
        print(f"Test accuracy: {test_acc:.4f}")

        # ----------------------------------------------------------------
        # Write predictions.txt in required format
        # <sentence1>###<sentence2>###<pred_label>
        # ----------------------------------------------------------------
        out_file = Path("predictions.txt")
        with out_file.open("w", encoding="utf-8") as fh:
            for ex, lbl in zip(raw_ds["test"], pred_labels):
                s1 = ex["sentence1"].replace("\n", " ").strip()
                s2 = ex["sentence2"].replace("\n", " ").strip()
                fh.write(f"{s1}###{s2}###{int(lbl)}\n")
        print(f"Wrote predictions to {out_file.resolve()}")



if __name__ == "__main__":
    main()
