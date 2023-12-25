import argparse
import evaluate
import logging
import numpy as np
import os
import torch
import datasets
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    SchedulerType,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Keyphrase Extraction task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="Number of preprocessing workers.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed before initializing model.
    set_seed(args.seed)

    # Labels
    label_list = ["B", "I", "O"]
    lbl2idx = {"B": 0, "I": 1, "O": 2}
    idx2label = {0: "B", 1: "I", 2: "O"}

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path, num_labels=3, id2label=idx2label, label2id=lbl2idx
    )

    # Tokenizer
    if args.model_name_or_path == "bloomberg/KBIR" or "roberta-base":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Dataset parameters
    dataset_1_full_name = "midas/inspec"  # scientific articles: train 1000, validation 500, test 500 (avg 10 per doc)
    dataset_2_full_name = "midas/kpcrowd"  # news: train 450, test 50 (avg 46 per doc)

    dataset_subset = "extraction"
    dataset_document_column = "document"
    dataset_biotags_column = "doc_bio_tags"

    # Load dataset
    dataset_1 = load_dataset(dataset_1_full_name, dataset_subset)
    dataset_2 = load_dataset(dataset_2_full_name, dataset_subset)

    # Dataset preprocessing
    def preprocess_fuction(all_samples_per_split):
        tokenized_samples = tokenizer.batch_encode_plus(
            all_samples_per_split[dataset_document_column],
            padding="max_length" if args.pad_to_max_length else False,
            truncation=True,
            is_split_into_words=True,
            max_length=args.max_length,
        )
        total_adjusted_labels = []
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split[dataset_biotags_column][k]
            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:
                if wid is None:
                    adjusted_label_ids.append(lbl2idx["O"])
                elif wid != prev_wid:
                    i = i + 1
                    adjusted_label_ids.append(lbl2idx[existing_label_ids[i]])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(
                        lbl2idx[f"{'I' if existing_label_ids[i] == 'B' else existing_label_ids[i]}"]
                    )

            total_adjusted_labels.append(adjusted_label_ids)
        tokenized_samples["labels"] = total_adjusted_labels
        return tokenized_samples

    # Preprocess dataset
    tokenized_dataset_1 = dataset_1.map(preprocess_fuction, batched=True, remove_columns="id")
    tokenized_dataset_2 = dataset_2.map(preprocess_fuction, batched=True, remove_columns="id")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Evaluation metrics
    seqeval = evaluate.load("seqeval")

    # labels = [label_list[i] for i in dataset["train"][0]["doc_bio_tags"]]
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    train_dataset_1 = tokenized_dataset_1["train"]
    train_dataset_2 = tokenized_dataset_2["train"]

    validation_dataset_1 = tokenized_dataset_1["validation"]
    validation_dataset_2 = tokenized_dataset_2["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_1,  # train_dataset_2
        eval_dataset=validation_dataset_1,  # validation_dataset_2
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()

    # Evaluation
    results = {}
    logger.info("*** Evaluate ***")
    if isinstance(validation_dataset_1, dict):
        metrics = {}
        for eval_ds_name, eval_ds in validation_dataset_1.items():
            dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
            metrics.update(dataset_metrics)
    else:
        metrics = trainer.evaluate(metric_key_prefix="eval")
    metrics["eval_samples"] = len(validation_dataset_1)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": args.model_name_or_path, "tasks": "keyword_extraction"}
    trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
