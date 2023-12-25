
### Goal
By giving some keywords about the types of books a person likes to read to generate sentences describing the person's personality
- input: some keywords about books
- output: sentences describing the reader's personality

### Establish enviroment
* must run in linux
* install `conda`
* install packges in `requirements.txt`
    ```bash
    conda env create --name ADL_final_comments python=3.10
    conda activate ADL_final_comments
    pip install -r requirements.txt
    ```

* Download [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) as base model and store as `Llama-2-7b-chat-hf`
    * remove or rename `generation_config.json`

### Train, validation, test data preparation
* You need to fill in your `OPENAI API KEY` in 6th line in `data_generate.py` to run it. 

* output file: (total 10000 data)
    * training file: `./data/train.json` (8000 data)
    * validation file: `./data/valid.json` (1000 data)
    * testing file: `./data/test.json` (1000 data)

```bash
cd data
python data_generate.py
cd ..
```
### How to train

* base model directory: `Llama-2-7b-chat-hf`
* input traning file: `./data/train.json`
* output peft model directory: `checkpoint-1000`
```bash
python train.py \
--base_model_path Llama-2-7b-chat-hf \
--train_files ./data/train.json
```

### How to predict
* base model directory: `Llama-2-7b-chat-hf`
* peft model directory: `checkpoint-1000`
* input testing file: `./data/train.json`
* output file: `./data/output.json`
```bash
python inference.py \
--base_model_path Llama-2-7b-chat-hf \
--peft_path checkpoint-1000 \
--test_data_path ./data/train.json \
--output_data_path ./data/output.json
```