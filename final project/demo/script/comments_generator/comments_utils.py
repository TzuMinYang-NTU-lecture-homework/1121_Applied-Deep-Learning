from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    return f"Please make a sentence to describe the personality of the reader who likes to read books related to '{instruction}' using at most two sentences without using below words: {instruction}. ANSWER:"

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    return quantization_config
