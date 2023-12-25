import torch
import numpy as np
from tqdm import tqdm
import json
from utils import get_prompt, get_bnb_config
import argparse
import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from dataclasses import dataclass, field
import transformers
import torch
import copy
from typing import Dict, Sequence
from torch.nn.utils.rnn import pad_sequence
import wandb

IGNORE_INDEX = -100

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Preprocess inputs
        preprocessed_inputs = []
        for example in instances:
            input = ""
            for i in example['input']:
                input += (i.lower()+", ")
            preprocessed_inputs.append(input[:-2])
        
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{get_prompt(i)}" for i in preprocessed_inputs]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict
    
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def create_peft_model(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        prepare_model_for_kbit_training,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    for name, module in model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    model.print_trainable_parameters()
    return model

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./Taiwan-LLM-7B-v2.0-chat/",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--train_files",
        type=str,
        default="",
        required=True,
        help="Path to train data."
    )
    args = parser.parse_args()

    
    # init wandb
    run = wandb.init(
        project="adl-final",
        name="7th-training (for report)",
        config={
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "fp16": True,
            "optim": "paged_adamw_8bit",
            "lr_scheduler_type": "cosine",
            "num_train_epochs": 1, 
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout":0.1,
            "lora_bias": "none",
            "lora_target_module": "q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj"
        }
    )    

    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map={"":0}
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model = create_peft_model(model)

    # device = "cuda:0"
    # model = model.to(device)
    # Define training args
    # output_dir = "./results"
    training_args = TrainingArguments(
        # output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,
        learning_rate=1e-4,
        num_train_epochs=1,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        # logging strategies
        # logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        report_to="wandb",
        output_dir="./outputs",
    )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=2048,
        target_max_len=2048,
        train_on_source=False,
        predict_with_generate=False,
    )
    data_files = {}
    data_files["train"] = args.train_files
    extension = "json"
    dataset = load_dataset(extension, data_files=data_files)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator,
    )

    # Start training
    trainer.train()
    wandb.finish()
    # trainer.save_model('./adapter_checkpoint')