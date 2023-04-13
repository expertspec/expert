# Evasiveness analysis
Module for detecting the number of evasive answers in speech.

### About

The module detecting the number of evasive answers in speech. The main class called EvasivenessDetector is located in the <b>evasiveness_analysis.py</b>. This class gets as input the path to the video and the results of feature extraction. 
The executable function of the EvasivenessDetector is get_evasiveness. It returns the path to the report, which contains lists with information about evasiveness for every speaker.\
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
The module is based on the idea of using a QA model that generates an answer from the context. The speaker's response is passed to the model as a context. If the model recognizes the response in the context with sufficient confidence, the response is considered not evasive. The threshold values of this confidence are obtained empirically.\
[link for english model](https://huggingface.co/deepset/roberta-base-squad2),
[link for russian model](https://huggingface.co/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru)

**Question-Statement model**\
The QS model is used to detect questions in dialogues. A sentence is considered as a question if it is detected as such by the model or if it has a question mark at the end. An answer to the question is the phrase(of another speaker) that follows the question.\
[link for english model](https://huggingface.co/shahrukhx01/question-vs-statement-classifier)