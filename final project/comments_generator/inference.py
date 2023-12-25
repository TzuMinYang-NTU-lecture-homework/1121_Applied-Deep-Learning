from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_prompt, get_bnb_config
from peft import PeftModel
import argparse
import torch
import json

'''
usage: python inference.py \
--base_model_path <base model> \
--peft_path <adapter> \
--test_data_path <input file> \
--output_data_path <output file>
'''

def parse_data():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of llama-2 model."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        default="",
        required=True,
        help="Path to output data."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse input arguments
    args = parse_data()

    # load data
    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    # load model and tokenizer
    bnb_config = get_bnb_config()
    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    # else:
    #     model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
    #     revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         revision=revision,
    #         torch_dtype=torch.bfloat16,
    #         quantization_config=bnb_config
    #     )
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         model_name,
    #         revision=revision,
    #     )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # load lora 
    model = PeftModel.from_pretrained(model, args.peft_path)

    # generate outputs
    preprocessed_inputs = []
    labels = []
    for example in data:
        input = ""
        for i in example['input']:
            input += (i.lower()+", ")
        preprocessed_inputs.append(input[:-2])
        labels.append(example['output'])

    inputs = [get_prompt(i) for i in preprocessed_inputs]
    ids = [i for i in range(len(data))]
    outputs = []
    for i in range(len(data)):
        input = tokenizer(inputs[i], return_tensors="pt").to("cuda:0")
        output = model.generate(
            **input, 
            max_new_tokens=256,
            # num_beams=3,
            # top_k=50,
            top_p=0.1
        )
        # print(tokenizer.decode(output[0], skip_special_tokens=True))
        outputs.append(tokenizer.decode(output[0], skip_special_tokens=True).split("ANSWER: ")[-1])
        print("index", i, "\n", outputs[i], "\n", "len=", len(outputs[i].split()))

    data = [{"id": index, "input": input, "output": output, "label": label} for index, input, output, label in zip(ids, preprocessed_inputs, outputs, labels)]
    with open(args.output_data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)