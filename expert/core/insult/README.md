# Deviant patterns in the speaker's speech
Module to detect the presence of profanity, insults and sarcasm in text.

### About

The module analyzes a passage of text and determines whether or not it contains sarcasm or insults. The main class, which is called Linear_insult_pred and located in the <b>insult_detector.py</b>. that class gets the language that will be used in the arguments. By default it is "eng", but it is also possible to specify "rus". A <b>predict<b> function of the class is called for direct analysis. It takes a sentence or the text itself as an argument. As a response it returns a dictionary with two fields - "verdict", which shows whether something is detected in the text or not, and "html", which contains the html code obtained from eli5, which highlights the words that contribute the most to the definition.


```
{'verdict': 'Текст с оскорблениями', 'html': 'some html code here'}
```
Classes of the prediction:

<li>"Текст без оскорблений и сарказма"</li>
<li>"Текст с оскорблениями"</li>
<li>"Текст с сарказмом"</li>

### It is VERY IMPORTANT to download weights for spacy in the terminal in advance

python -m spacy download ru_core_news_lg


### The example of the usage

```python
from expert.core.insult import contradiction_analysis

if __name__ == "__main__":
    ext = Linear_insult_pred(lang="rus")

    print(
        ext.predict(
            "Я работаю 40 часов в неделю, для того чтобы оставаться бедным"
        )["verdict"]
    )

    print(ext.predict("Ну ты и тварь, конечно.")["verdict"])
    print(ext.predict("Ушлёпок. Я до тебя ещё доберусь.")["verdict"])
    print(ext.predict("А тут мы можем наблюдать восходящий тренд.")["verdict"])
```
