# 2023 FALL ADL final project -- books wrapped

* Team: `25`

* Members: `梁舜勛`, `楊子民`, `楊瀚博`, `羅晴`, `李語婕`

### Introduction
* like Spotify's annoual review named Spotify wrapped, this is a English books' version annoual review named `books wrapped``
* use web crawler to catch books' abstract, generate keywords from books' abstract, plot a wordcloud of keywords, finally generate your own personality form keywords 

![Alt text](./demo/data/demo_result.png)

### Running logic
1. input book catalog
   * just need book titles
   * every book title seperated by `\n`
2. use web crawler to search `amazon book {book title}` on Google, click the first result except the advertises, and catch the abstract from amazon book
3. input abstract to `keywords_generator (module1)`, generate logits of keywords, and convert logits to scores
4. input top 7 of keywords scores to `comments_generator (module2)`, and generate personality
5. use all keywords scores to plot the wordcloud of all keywords


#### Function of every directory
* `keywords_generator`: the keywords generator's data preparation, training and inference
* `comments_generator`: the comments generator's training and inference
* `demo`: integrate the fine-tune modules and front end, run a web UI