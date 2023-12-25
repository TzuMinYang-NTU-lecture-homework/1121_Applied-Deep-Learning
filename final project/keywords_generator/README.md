### Goal
By giving book catalog to generate some keywords about the types of books a person likes to read
- input: book catalog
- output: several keywords about catalog 

### Establish enviroment
* must run in Windows
* install `conda`
* install packges in `environment.yml`
    ```bash
    conda env create --name ADL_final_keywords -f environment.yml
    conda activate ADL_final_keywords
    ```

* Download [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) as base model and store as `distilbert-base-uncased`

### How to train
* script file: `keyphrase_extraction_finetune.py`
* base model directory: `distilbert-base-uncased`
* output model directory: `distilbert_keyphrase_extraction`
```bash
bash keyphrase_extraction_finetune.sh
```

### How to predict
* script file: `generate_keywords.py`
* model directory: `distilbert_keyphrase_extraction`
* input file: `book_titles.json`
* output file: `output.json`
```bash
bash run.sh
```