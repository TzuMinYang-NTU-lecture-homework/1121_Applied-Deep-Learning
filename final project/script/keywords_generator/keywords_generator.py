import argparse
import json
import numpy as np
import os
import re
import spacy
import torch
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


# List of book keywords
book_genre = [
    "Fiction",
    "Non-fiction",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Fantasy",
    "Historical Fiction",
    "Biography",
    "Autobiography",
    "Self-Help",
    "Thriller",
    "Horror",
    "Poetry",
    "Adventure",
    "Travel",
    "Science",
    "Philosophy",
    "Psychology",
    "Business",
    "Economics",
    "Politics",
    "Memoir",
    "Young Adult",
    "Children's Literature",
    "Classic",
    "Crime",
    "Comedy",
    "Drama",
    "Suspense",
    "Supernatural",
    "Cultural",
    "Inspirational",
    "Art",
    "Music",
    "Cooking",
    "Gardening",
    "Parenting",
    "Education",
    "Fitness",
    "Technology",
    "Programming",
    "History",
    "Environmental",
    "Mythology",
    "Religion",
    "Sports",
    "Wellness",
    "Sociology",
    "Linguistics",
    "True Crime",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract keyphrases from book abstracts.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--book_titles_path",
        type=str,
        help="Path to the book titles file.",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to store the extracted keyphrases.",
        required=True,
    )
    args = parser.parse_args()
    return args


def generate_keywords(model, tokenizer, book_titles):
    keyphrases_dicts = []
    fail_book = []
    #ouput_list = []
    # Extract the book abstracts
    idx = 0
    while idx < len(book_titles):
        book_title = book_titles[idx]

        # Handle exceptions
        try:
            # Open the Chrome browser
            options = Options()
            options.add_experimental_option("detach", True)
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options,
            )
            # Search the book title on Google
            print(f'{book_title}: Searching on Google...')
            search_title = "Amazon+book+" + book_title.replace(" ", "+")
            driver.get(f"https://www.google.com.tw/search?q={search_title}")

            # Click the first Amazon link on Google search result
            links = driver.find_elements("xpath", "//a[@href]")
            for link in links:
                # check if the link is an Amazon link and the first word in the book title is in the link
                if "amazon.com" in link.get_attribute("href") and "/dp/" in link.get_attribute("href"):
                    link.click()
                    break

            # Expand the book abstract if it is collapsed
            try:
                driver.find_element(
                    "xpath", "//div[@id='bookDescription_feature_div']//span[@class='a-expander-prompt']"
                ).click()
            except:
                pass
            # Get the book abstract
            book_abstract = driver.find_element(
                "xpath",
                "//div[@id='bookDescription_feature_div']//div[@data-a-expander-name='book_description_expander']//div",
            )
        except:
            print(f"{book_title}: Abstract not found.")
            print("-------------------------------------------------------------")
            fail_book.append(book_title)
            # Close the Chrome browser and try again
            driver.quit()
            idx += 1
            continue

        # Process the book abstract
        book_abstract = (
            book_abstract.text.replace("\n", ". ")
            .replace(". . . ", ". ")
            .replace(". . ", ". ")
            .replace("..", ".")
            .replace("  ", " ")
            .strip()
        )
        # Delete the special characters except for the period and dot
        book_abstract = re.sub("[^a-zA-Z0-9.,\- ]", "", book_abstract)
        #print(book_abstract)
        print(f"{book_title}: Success to find abstract.")

        # Close the Chrome browser
        driver.quit()
        idx += 1

        # Truncate book abstract to 256 tokens
        book_abstract = " ".join(book_abstract.split(" ")[:256])

        # NER model to find person names
        NER = spacy.load("en_core_web_sm")
        doc = NER(book_abstract)

        # Construct a string of person names
        person_names_str = ""
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                person_names_str += ent.text + ", "

        book_abstract = book_abstract.lower()

        # Load the keyphrase extraction model
        #model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        #tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        inputs = tokenizer(book_abstract, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        # Find tokens with the highest logits in B
        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
        keyphrases_score = []
        for i, token in enumerate(predicted_token_class):
            if token == "B":
                keyphrases_score.append(logits[0, i, 0].item())

        # Define post_process functions
        def concat_tokens_by_tag(keyphrases):
            keyphrase_tokens = []
            for id, label in keyphrases:
                if label == "B":
                    keyphrase_tokens.append([id])
                elif label == "I":
                    if len(keyphrase_tokens) > 0:
                        keyphrase_tokens[len(keyphrase_tokens) - 1].append(id)
            return keyphrase_tokens

        def extract_keyphrases(example, predictions, tokenizer, keyphrases_score, index=0):
            idx2label = {0: "B", 1: "I", 2: "O"}
            keyphrases_list = [
                (id, idx2label[label])
                for id, label in zip(np.array(example["input_ids"]).squeeze().tolist(), predictions[index].tolist())
                if idx2label[label] in ["B", "I"]
            ]

            processed_keyphrases = concat_tokens_by_tag(keyphrases_list)
            extracted_kps = tokenizer.batch_decode(
                processed_keyphrases,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            keyphrases = [kp.strip() for kp in extracted_kps]

            # Delete keyphrases are the same with the book title
            for i in range(len(keyphrases)):
                if keyphrases[i] == book_title.lower():
                    keyphrases[i] = ""
                    keyphrases_score[i] = 0
            # Delete repeated keyphrases
            for i in range(len(keyphrases)):
                for j in range(i + 1, len(keyphrases)):
                    if (
                        keyphrases[i].split(" ")[0] == keyphrases[j].split(" ")[0]
                        or keyphrases[i].split(" ")[-1] == keyphrases[j].split(" ")[-1]
                    ):
                        keyphrases[j] = ""
                        keyphrases_score[j] = 0
            # Delete person names
            for i in range(len(keyphrases)):
                for word in keyphrases[i].split(" "):
                    if word in person_names_str.lower():
                        keyphrases[i] = ""
                        keyphrases_score[i] = 0
                        break
            # Delete keyphrases with more than 2 words
            for i in range(len(keyphrases)):
                if len(keyphrases[i].split(" ")) > 2:
                    keyphrases[i] = ""
                    keyphrases_score[i] = 0

            keyphrases = [kp for kp in keyphrases if kp != ""]
            keyphrases_score = [score for score in keyphrases_score if score != 0]
            # keyphrases_score = torch.softmax(torch.tensor(keyphrases_score), dim=0).tolist()

            return keyphrases, keyphrases_score

        keyphrases, scores = extract_keyphrases(inputs, predictions, tokenizer, keyphrases_score)
        if len(keyphrases) == 0:
            print(f"{book_title}: No keyphrases extracted.")
            print("-------------------------------------------------------------")
            continue
        # Sort keyphrases by score
        scores, keyphrases = zip(*sorted(zip(scores, keyphrases), reverse=True))
        # Show scores to 4 decimal places
        scores = [round(score, 4) for score in scores]
        keyphrases_dict = dict(zip(keyphrases, scores))
        keyphrases_dicts.append(keyphrases_dict)

        # Write the extracted keyphrases to the output file
        #output_dict = {
        #    "book_title": book_title,
        #    "book_abstract": book_abstract,
        #    "keyphrases": keyphrases_dict,
        #}
        #ouput_list.append(output_dict)
    #json.dump(ouput_list, output_file)

    """keywords Score Calculation"""
    # convert book keywords to lowercase string
    book_genre_str = str(book_genre).lower()
    # check if keyphrase is in book keywords
    genre_related_keywords = {}
    other_keywords = {}
    for i in range(len(keyphrases_dicts)):
        for keyphrase, score in keyphrases_dicts[i].items():
            for word in keyphrase.split(" "):
                if word in book_genre_str:
                    genre_related_keywords[keyphrase] = score
                    break
            other_keywords[keyphrase] = score

    # sort the keyphrases by score (from low to high)
    genre_related_keywords = sorted(genre_related_keywords.items(), key=lambda x: x[1])
    other_keywords = sorted(other_keywords.items(), key=lambda x: x[1])
    genre_related_keywords = [{k: v} for k, v in genre_related_keywords]
    other_keywords = [{k: v} for k, v in other_keywords]
    # rescale the score so the lowest score is 10, the second lowest score is 11, and so on
    for i in range(len(genre_related_keywords)):
        tag, score = list(genre_related_keywords[i].items())[0]
        genre_related_keywords[i] = {tag: (10 + i) * 2}
    for i in range(len(other_keywords)):
        tag, score = list(other_keywords[i].items())[0]
        other_keywords[i] = {tag: int(score * 2)}
    # sort the keyphrases by score (from high to low)
    genre_related_keywords = sorted(genre_related_keywords, key=lambda x: float(list(x.values())[0]), reverse=True)
    other_keywords = sorted(other_keywords, key=lambda x: float(list(x.values())[0]), reverse=True)
    # merge the two lists
    merged_keywords = genre_related_keywords + other_keywords
    # delete repeated keyphrases
    result = {}
    for i in range(len(merged_keywords)):
        tag, score = list(merged_keywords[i].items())[0]
        if tag.title() not in result:
            result[tag.title()] = score
            
    print(f"{book_title} : dict result is {result}")
    print("-------------------------------------------------------------")
    return result
