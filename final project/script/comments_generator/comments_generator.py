from comments_generator.comments_utils import get_prompt
import json

def generate_comments(keywords, model, tokenizer):
    # load keywords
    with open(args.test_keywords_path, "r") as f:
        keywords = json.load(f)

    # generate commentss
    preprocessed_inputs = []
    for example in keywords:
        input = ""
        for i in example['input']:
            input += (i.lower()+", ")
        preprocessed_inputs.append(input[:-2])

    inputs = [get_prompt(i) for i in preprocessed_inputs]
    ids = [i for i in range(len(keywords))]
    commentss = []
    for i in range(len(keywords)):
        tokenized_keywords = tokenizer(get_prompt(keywords), return_tensors="pt").to("cuda:0")
        comments = model.generate(**tokenized_keywords, max_new_tokens=1024)
        # print(tokenizer.decode(comments[0], skip_special_tokens=True))
        commentss.append(tokenizer.decode(comments[0], skip_special_tokens=True).split("ANSWER: ")[-1])
        print("index ", i, ": ", commentss[i])

    keywords = [{"id": index, "input": input, "comments": comments} for index, input, comments in zip(ids, preprocessed_inputs, commentss)]
    with open(args.comments_keywords_path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)