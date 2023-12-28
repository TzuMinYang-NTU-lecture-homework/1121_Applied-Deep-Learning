### Goal
Integrate two modules and frontend as a web UI.

#### Establish environment
* must run on Linux
* clone or download this repository
* If you have never done this
  * `cd ../keywords_generator`
  * do the things in the README.md
  * `cd ../demo`
* If you have never done this
  * `cd ../comments_generator`
  * do the things in the README.md
  * `cd ../demo`
* install conda
* install google-chrome browser
* install packages in `requirements.txt`
    ```bash
    conda create --name ADL_HW3_final_project python=3.10
    conda activate ADL_final_project
    pip install -r ./script/requirements.txt
    ```
* install `en_core_web_sm`
    ```bash
    python -m spacy download en_core_web_sm
    ```


#### How to run
```bash
gradio ./script/frontend.py
```
