import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


def load_model_for_training(model_name, num_labels, id2label, label2id, cache_dir=None):
    """Load base BERT weights without using transformers' blocked torch.load path."""
    local_dir = snapshot_download(model_name, cache_dir=cache_dir)
    weights_path = os.path.join(local_dir, "pytorch_model.bin")

    config = AutoConfig.from_pretrained(
        local_dir,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_config(config)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=False)

    print(
        f"Loaded encoder weights. Missing keys: {len(incompatible.missing_keys)}, "
        f"unexpected keys: {len(incompatible.unexpected_keys)}"
    )

    return local_dir, model

# ──────────────────────────── Config ────────────────────────────
MODEL_NAME = "nlpaueb/bert-base-greek-uncased-v1"
TRAIN_FILE = "ELCardioCCTrainSet/trainELCardioCC2026.jsonl"
LABEL_FILE = "ELCardioCCTrainSet/labelset.txt"
OUTPUT_DIR = "output"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 20
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42
VAL_RATIO = 0.15

# ──────────────────────────── GPU check ─────────────────────────
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected — training will be slow on CPU")

# ──────────────────────────── Seed ──────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ──────────────────────────── Labels ────────────────────────────
def load_labels(path):
    with open(path) as f:
        labels = [line.strip() for line in f if line.strip()]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label

labels, label2id, id2label = load_labels(LABEL_FILE)
NUM_LABELS = len(labels)
print(f"Loaded {NUM_LABELS} labels")

# ──────────────────────────── Data ──────────────────────────────
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def make_multihot(doc_level_annotations, label2id, num_labels):
    vec = np.zeros(num_labels, dtype=np.float32)
    for group in doc_level_annotations:
        for code in group:
            if code in label2id:
                vec[label2id[code]] = 1.0
    return vec

class ICDDataset(Dataset):
    def __init__(self, samples, tokenizer, label2id, num_labels, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = num_labels
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = make_multihot(
            sample["document_level_annotations"], self.label2id, self.num_labels
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

# Module-level storage for validation group annotations (set in main())
_val_group_annotations = []


def compute_group_metrics_single(predictions, ground_truth):
    """Compute TP, FP, FN for a single document using group-level evaluation.

    A GT group is TP if at least one of its codes appears in predictions.
    FP = predicted codes not belonging to ANY ground truth group.
    FN = GT groups with no matching prediction.
    """
    all_gt_codes = set()
    for group in ground_truth:
        for code in group:
            all_gt_codes.add(code)

    pred_set = set(predictions)

    tp = sum(1 for group in ground_truth if any(c in pred_set for c in group))
    fn = len(ground_truth) - tp
    fp = sum(1 for p in pred_set if p not in all_gt_codes)

    return tp, fp, fn


def compute_metrics_for_trainer(eval_pred):
    """Compute group-level micro-averaged P/R/F1 over the validation set."""
    logits, label_ids = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds_binary = (probs >= 0.5).astype(int)

    total_tp, total_fp, total_fn = 0, 0, 0

    for i in range(len(preds_binary)):
        pred_codes = [id2label[j] for j in range(NUM_LABELS) if preds_binary[i][j] == 1]
        gt_groups = _val_group_annotations[i]
        tp, fp, fn = compute_group_metrics_single(pred_codes, gt_groups)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

# ──────────────────────────── Main ──────────────────────────────
def main():
    # Load data
    all_samples = load_jsonl(TRAIN_FILE)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - VAL_RATIO))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Store val group annotations for group-level metrics in compute_metrics_for_trainer
    global _val_group_annotations
    _val_group_annotations = [s["document_level_annotations"] for s in val_samples]

    # Save val samples for later group-level eval
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "val_samples.jsonl"), "w") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Tokenizer & datasets
    local_model_path, model = load_model_for_training(
        MODEL_NAME, NUM_LABELS, id2label, label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    train_dataset = ICDDataset(train_samples, tokenizer, label2id, NUM_LABELS, MAX_LENGTH)
    val_dataset = ICDDataset(val_samples, tokenizer, label2id, NUM_LABELS, MAX_LENGTH)

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    trainer.train()

    # Save best model & tokenizer
    best_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nBest model saved to {best_dir}")

    # Final eval on val set
    results = trainer.evaluate()
    print(f"\nValidation results: {results}")


if __name__ == "__main__":
    main()
