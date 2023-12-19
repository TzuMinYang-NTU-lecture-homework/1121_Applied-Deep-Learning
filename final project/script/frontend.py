import gradio as gr
from wordcloud_generator import generate_wordcloud
from keywords_generator.keywords_generator import generate_keywords
#from comments_generator import generate_comments
from comments_generator.comments_utils import get_bnb_config, get_prompt
from peft import PeftModel
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForCausalLM
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image

# load examples
with open("./example/examples.txt", "r") as f:
    book_examples = f.readlines()
book_examples = [book_example.strip() for book_example in book_examples]

# without this, inputs only show "example{i}"
def process_examples(example_name):
    if example_name == "example1":
        return "\n".join(book_examples[: 33])
    elif example_name == "example2":
        return "\n".join(book_examples[33: 66])
    elif example_name == "example3":
        return "\n".join(book_examples[66:])
    
# prepare keywords model and tokenizer
keywords_model_path = "./script/keywords_generator/distilbert_keyphrase_extraction_v2"
keywords_model = AutoModelForTokenClassification.from_pretrained(keywords_model_path)
keywords_tokenizer = AutoTokenizer.from_pretrained(keywords_model_path)

# prepare comments model and tokenizer
comments_model_path = "./script/comments_generator/Llama-2-7b-chat-hf"
bnb_config = get_bnb_config()
comments_model = AutoModelForCausalLM.from_pretrained(comments_model_path, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
comments_tokenizer = AutoTokenizer.from_pretrained(comments_model_path)

if comments_tokenizer.pad_token_id is None:
    comments_tokenizer.pad_token_id = comments_tokenizer.eos_token_id

comments_peth_model_path = "./script/comments_generator/checkpoint-2000"
comments_model = PeftModel.from_pretrained(comments_model, comments_peth_model_path)

# prepare wordcloud mask
mask_path = "./data/book.jpg"
mask = np.array(Image.open(mask_path))

use_keywords_num = 7

def inference(book_titles):
    if book_titles == "":
        return [None, None, None]
    
    # keywords
    print("================================================ Generating keywords =============================================================")
    keywords_dict = {}
    for book_title in book_titles.split("\n"):
        new_keywords_dict = generate_keywords(keywords_model, keywords_tokenizer, [book_title])
        for key, value in new_keywords_dict.items():
            if key in keywords_dict:
                keywords_dict[key] += value
            else:
                keywords_dict[key] = value
    print(f"final keyword dict is {keywords_dict}")
    keywords = ", ".join([keyword for keyword, _ in keywords_dict.items()][: use_keywords_num])
    
    # comments
    print("================================================ Generating comments =============================================================")
    # 因為只有幾行而已，所以我直接寫過來沒有call另一個檔案的function了
    tokenized_keywords = comments_tokenizer(get_prompt(keywords), return_tensors="pt").to("cuda:0")
    comments = comments_model.generate(**tokenized_keywords, max_new_tokens=1024) # 在這邊設 max_new_tokens 它會warning
    comments = comments_tokenizer.decode(comments[0], skip_special_tokens=True).split("ANSWER: ")[-1]
    comments = ". \n".join(comments.split(". ")) # 每句講完換行不然太長
    print(f"comments are below: \n{comments}")
    
    # wordcloud
    print("================================================ Generating wordcloud =============================================================")
    wordcloud = WordCloud(width=3000, height=2000, max_words=2000, random_state=1, 
                          background_color='white', colormap='Set2', collocations=False, 
                          mask=mask, collocation_threshold=0).generate_from_frequencies(keywords_dict)
    #, contour_width=3, contour_color='black'
    wordcloud = wordcloud.to_image()
    print("Sucess to generate wordcloud.")
    print("======================================================== Done =====================================================================")
    
    return [keywords, comments, wordcloud]

storj_theme = gr.Theme.from_hub("bethecloud/storj_theme")
bibliography = gr.Textbox(label="Your bibliography", info="Please enter your bibliography, each book title for one line. Max input is 50 books", max_lines=10000, interactive=True, lines=25)

with gr.Blocks(theme=storj_theme) as demo:
    with gr.Row():
        # title and description
        gr.HTML("""
            <h1 style="text-align: center;">  Books Wrapped </h1>
            <p style="text-align: center;"> This is a book version Spotify Wrapped application. </p>
            """)
    with gr.Row():
        gr.Examples(["example1", "example2", "example3"], inputs=bibliography, outputs=bibliography, fn=process_examples, run_on_click=True) # !!! 用cache example會error，待解決
    with gr.Row():
        # columns width will be 2: 3
        with gr.Column(scale=2):
            bibliography.render()
            with gr.Row():
                clear =  gr.Button("Clear")
                run = gr.Button("Run analysis", variant="primary")
        with gr.Column(scale=3):
            comments = gr.Textbox(label="Your comments", info="It reflects what kind of people you are.", interactive=False)
            appellation = gr.Textbox(label="Your keywords", info="It gives you some keywords about you.", interactive=False)
            wordcloud = gr.Image(label="Your wordcloud of keywords")
    
    run_event = run.click(fn=inference, inputs=bibliography, outputs=[appellation, comments, wordcloud])
    clear.click(lambda: None, outputs=bibliography)


if __name__ == "__main__":
    demo.launch(show_api=False) # , share=True