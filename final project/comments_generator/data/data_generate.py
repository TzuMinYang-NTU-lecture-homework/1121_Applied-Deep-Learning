import openai
import json
import random

# 設定您的 OpenAI API 密鑰
openai.api_key = 'YOUR API KEY'
book_tags = [
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
    "True Crime"
    "corporate strategies",
    "cognitive biases",
    "stock market",
    "midnight library",
    "glaciologist",
    "nineteenthcentury psychology",
    "alfred adler",
    "philosopher",
    "selfcare",
    "happiness",
    "social commentary",
    "societal expectations",
    "pride",
    "humor",
    "societal constraints",
    "middleearth",
    "magic ring",
    "lord of the rings",
    "fairy tale",
    "dwarves",
    "gandalf",
    "frightening creature",
    "adventurous tale",
    "wizard",
    "remote tropical island",
    "cannibals",
    "trinidad",
    "english",
    "hunger games",
    "capitol",
    "desert planet arrakis",
    "dune",
    "mysticism",
    "environmentalism",
    "Pop culture",
    "Factoids",
    "Cover to cover",
    "Friends and family",
    "Pass time",
    "Relax",
    "Connect with others",
    "Learn",
    "Engaging",
    "Laugh",
    "2020 presidential election",
    "Donald Trump",
    "Elected Republican officials",
    "Oath to the Constitution",
    "Ignored court rulings",
    "Overturning a lawful election",
    "Violent attack on the Capitol",
    "Liz Cheney",
    "Republican officials",
    "Congressional Select Committee",
    "Investigation",
    "Stolen election lie",
    "Constitutional framework",
    "Risks",
    "Oath and Honor",
    "Perilous moment",
    "Violet Sorrengail",
    "Scribe Quadrant",
    "Dragon riders",
    "Commanding general",
    "Elite of Navarre",
    "Fragile humans",
    "Dragon bonding",
    "Death",
    "Cadets",
    "Xaden Riorson",
    "Wingleader",
    "Riders Quadrant",
    "War",
    "Wits",
    "Kingdom's protective wards",
    "Death toll",
    "Terrible secret",
    "Basgiath War College",
    "Graduate or die",
    "Leadership",
    "Friends, enemies, lovers",
    "Sunrise",
    "Pottstown, Pennsylvania",
    "Skeleton",
    "Chicken Hill",
    "Immigrant Jews",
    "African Americans",
    "Secrets",
    "Moshe Ludlow",
    "Chona Ludlow",
    "Integrated theater",
    "Heaven & Earth Grocery Store",
    "State intervention",
    "Deaf boy",
    "Nate Timblin",
    "Black janitor",
    "Margins of white, Christian America",
    "Struggle",
    "Survival",
    "Overlapping stories",
    "Truth",
    "Town's white establishment",
    "Love",
    "Community",
    "The Heaven & Earth Grocery Store",
    "James McBride",
    "Compassionate"
]
out = []
generated_count = 0  # 记录成功生成的数据数量

while generated_count < 10000:
    n = random.randint(1, 10)
    random_tags = random.sample(book_tags, n)

    # 檢查是否已經存在於之前的迭代中
    while any(item['input'] == random_tags for item in out):
        n = random.randint(1, 10)
        random_tags = random.sample(book_tags, n)
    
    # 定義對話內容
    dialogue_history = [
        {"role": "system", "content": "You are a personality trait assistant."},
        {"role": "user", "content": "Describe my personality based on the keyword 'curious'."},
        {"role": "assistant", "content": "you have a thirst for knowledge, enjoy learning new things, and are open-minded."},
        {"role": "user", "content": f"Describe my personality based on the keyword '{', '.join(random_tags)}. Omit any introductory phrases like 'Based on' and please answer in 'Your personality is...' format and don't answer with the words in keywords."}
    ]

    try:
        # 与 ChatGPT 进行互动
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=dialogue_history,
            max_tokens=300  # 可以根据需要调整
        )

        # 提取生成的文本
        generated_text = response['choices'][0]['message']['content']

        # 将生成的文本保存为.json文件
        output_json = {
            "input": random_tags,
            "output": generated_text
        }
        out.append(output_json)
        generated_count += 1

        # 每成功获取一个数据就保存一次
        with open('output.json', 'w', encoding='utf-8') as json_file:
            json.dump(out, json_file, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"An error occurred: {e}")
        # 在异常发生时，你可以选择记录错误信息或者进行其他处理
        continue

# 划分数据集
train_data = out[:8000]
valid_data = out[8000:9000]
test_data = out[9000:]

# 保存数据到不同的 JSON 文件
with open('train.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_data, json_file, indent=2, ensure_ascii=False)

with open('valid.json', 'w', encoding='utf-8') as json_file:
    json.dump(valid_data, json_file, indent=2, ensure_ascii=False)

with open('test.json', 'w', encoding='utf-8') as json_file:
    json.dump(test_data, json_file, indent=2, ensure_ascii=False)

