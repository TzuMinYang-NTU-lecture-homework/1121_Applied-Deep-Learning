### Reproduce

#### Download 
* clone or download this repository
* If you never done this, `cd ../keywords_generator`
  * Download [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) as base model and store as `distilbert-base-uncased`
  * `cd ../demo`
* If you never done this, `cd ../comments_generator`
  * Download [NousResearch/Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) as base model and store as `Llama-2-7b-chat-hf`
      * remove or rename `generation_config.json`
  * `cd ../demo`

#### Establish enviroment
* must run in linux
* install conda
* install google-chrome browser
* install packges in `requirements.txt`
    ```bash
    conda create --name ADL_HW3_final_project python=3.10
    conda activate ADL_final_project
    pip install -r ./script/requirements.txt
    ```
* install `en_core_web_sm`
    ```bash
    python -m spacy download en_core_web_sm
    ```


#### Run demo
```bash
gradio ./script/frontend.py
```