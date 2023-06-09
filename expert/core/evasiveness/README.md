# Evasiveness analysis
Module for detecting the number of evasive answers in speech.

### About

The module detects the number of evasive answers in speech.
The main class called EvasivenessDetector is located in the <b>evasiveness_analysis.py</b>.
This class gets as input the path to the video and the results of feature extraction.
The executable function of the EvasivenessDetector is get_evasiveness.
It returns the path to the report, which contains lists with information about evasiveness for every speaker.\
The first list includes summary for each respondent's responses and has the following structure:
```
[{"resp": speaker's name, "total": total number of questions, "evasive": number of evasive answers,
        "not evasive": number of not evasive answers, "neutral": number of neutral answers}]
```
Index numbers of the following lists correspond to a respondents numbers + 1. Every list consisted of
        dicts with the following structure:
```
{resp: str, question: str, answer: str, label: str, model_conf: float, pred_answer: str}
```

### The example of the usage

```python
detector = EvasivenessDetector(
        video_path="test_video.mp4",
        transcription_path="temp/test_video/transcription.json",
        diarization_path="temp/test_video/diarization.json",
        lang="ru",
        device=torch.device("cuda")
)
detector.get_evasiveness()
```
*Result:* 'temp/test_video/evasiveness.json'

### Models
**Question-Answer model**\
The QA model generates an answer from a context.
The speaker's response is passed to the model as the context.
The model returns probable answer and a confidence in the answer found.
If the model recognizes the response in the context with sufficient confidence,
the response is considered not evasive.
The threshold values of this confidence are obtained empirically.\
[link for english model](https://huggingface.co/deepset/roberta-base-squad2),\
[link for russian model](https://huggingface.co/mrm8488/bert-multi-cased-finetuned-xquadv1)

**EvasivenessModels**\
The model for determining evasiveness is a DistilBert, fine-tuned for
the classification task on a dataset consisting of 1000 examples
(question-answer) with an assigned label for each pair: evasive/non-evasive.
A question and an answer are passed to the model and the model returns a predicted
label: LABEL_0 - not evasive, LABEL_1 - evasive.\
[link for english model](https://huggingface.co/alenaa/evasiveness),\
[link for russian model](https://huggingface.co/alenaa/ru_evasiveness)\

**EvasivenessClassificator**\
Evasiveness classificator is a decision tree model, which get a predicted label
from Evasiveness model and confidence from QA model and then returns final result:
evasive/not evasive.\
english model - **rf_evas_model.pickle** \
russian model - **rf_evas_model_ru.pickle**
