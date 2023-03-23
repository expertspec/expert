# Confidence analysis
Module for evaluating the level of confidence with non-verbal (facial micro-expressions and audio features) features analysis.

### About

The module evaluates the level of confidence with analysis of the non-verbal features of the speaker's from the picked video. The main class called ConfidenceDetector is located in the <b>confidence_analysis.py</b>. This class gets as input the path to the video and the results of feature extraction. Optionally, it is possible to specify the speaker with the setting of the specified image.
The executable function of the ContradictionDetector is get_confidence. It returns the path to the report, which contains the list with records in the next form:
```
{'time_sec' : time, 'confidence' : "No face" (in case, if there no face (couldn't be detected) in the fragment) | float (the level of confidence)}
```

### The example of the usage

```python
from expert.core.confidence import confidence_analysis
conf_det = confidence_analysis.ConfidenceDetector(video_path=video_path,
                                                    features_path="./videos/temps/report.json",
                                                    face_image="./videos/temps/0.jpg",
                                                    path_to_diarization="./videos/temps/diarization.json",
                                                    device = torch.device("cuda"),
                                                    output_dir="./check")

Confidence.get_confidence()
```
*Result:* './check/confidence.json'

### Weights
Here are two options for the models(different configs): only micro-expressions (landmarks_transformer) and with audio features (recommended and set as default) (landmarks_audio_transformer).

- The weights of the model should be localted in the `weights/` 