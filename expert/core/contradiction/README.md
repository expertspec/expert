# Inconsistency analysis
Module for classifying a pair of texts as entailment, contradiction, neutral.

### About

The module analyzes the text obtained from the speech to text and entered hypothesis. The main class called ContradictionDetector is located in the <b>contradiction_analysis.py</b>. This class gets as input the path to the transcription result and to the video. Also, optionally it's possible to specify the language of the video (default is for "en", for the russian you can set "ru"). 

<i> Additionally, the intervals (the chosen_intervals argument) are also passed in as sequence numbers (list) (duration of the interval is equal to 1 minute as default). If nothing is passed, the whole text will be analyzed </i>

The executable function of the ContradictionDetector is get_contradiction, which takes the entered text for analysis and returns the path to the json file with results.
The example of the json file:
```
[ {"time_interval": "0:00:00-0:00:03", "text": "The higher education system, I am sure, should reduce this gap.", "predict": 2.0}]
```
Classes of the prediction:
  
<li> 0 - entailment </li>
<li> 1 - contradiction </li>
<li> 2 - neutral </li>


### The example of the usage

```python
from expert.core.contradiction import contradiction_analysis
contr_det = contradiction_analysis.ContradictionDetector(transcription_path = 'transcription.json',
                                                        video_path='example.mp4',
                                                        output_dir='./check',
                                                        lang='ru')

contr_det.get_contradiction("Сравнение происходит с предложениями разделенными по паузам.")
```
*Result:* './check/contradiction.json'


