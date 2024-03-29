# Deviant Patterns in the Speaker's Speech

This module analyzes a passage of text and determines whether or not it contains sarcasm or insults. The main class is called `InsultDetector` and is located in `insult_detector.py`. This class receives the language as an argument, with "en" being the default value. However, it is also possible to specify "ru". A `predict` function of the class is called for direct analysis. It takes a sentence or the text itself as an argument. As a response, it returns a dictionary with three fields: "verdict", which shows whether something is detected in the text or not, "probability", float prediction probability. Еhat is, how confident the model is in referring to this particular class and "html", which contains the HTML code obtained from eli5 that highlights the words that contribute the most to the definition.

```
{'verdict': 'Text with insults', 'probability':0.78, 'html': 'some HTML code here'}
```

Classes of prediction:

- "Text without insults and sarcasm"
- "Text with insults"
- "Text with sarcasm"

### It is VERY IMPORTANT to download weights for Spacy in advance using the following command in your terminal:

```python
python -m spacy download ru_core_news_lg
```

### The example of usage:

```python
if __name__ == "__main__":
    ext = InsultDetector(
        video_path="./EXPERT_NEW/text_agressive/models/transcription.json",
        transcription_path="./EXPERT_NEW/text_agressive/models/transcription.json",
        lang="ru",
    )

    print(
        ext.predict("Я работаю 40 часов в неделю, для того чтобы оставаться бедным")[
            "probability"
        ]
    )

    print(ext.predict("Ну ты и тварь, конечно.")["verdict"])
    print(ext.predict("Ушлёпок. Я до тебя ещё доберусь.")["verdict"])
    print(ext.predict("А тут мы можем наблюдать восходящий тренд.")["verdict"])
    ext.get_insult()
```
