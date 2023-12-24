from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image

    
def generate_wordcloud(keywords_dict, mask):
    # text = open(tagfile, "r", encoding="utf-8").read()

    wordcloud = WordCloud(width=3000, height=2000, max_words=2000, random_state=1, 
                          background_color='white', colormap='Set2', collocations=False, 
                          mask=mask, collocation_threshold=0).generate_from_frequencies(keywords_dict)
    #, contour_width=3, contour_color='black'
    wordcloud = wordcloud.to_image()
    return wordcloud