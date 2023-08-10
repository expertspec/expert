import os

import pytest

from expert.core.insult import InsultDetector


@pytest.fixture()
def insult_detector():
    insult_detector = InsultDetector(
        video_path="./test/unit/data/test_insult_data/transcription.json",
        transcription_path="./test/unit/data/test_insult_data/transcription.json",
        lang="ru",
    )

    return insult_detector


def test_remove_stop_and_make_lemma(insult_detector):
    # Test with Russian text containing stop words
    assert (
        type(
            insult_detector.remove_stop_and_make_lemma(
                "Я работаю 40 часов в неделю, для того чтобы оставаться бедным"
            )
        )
        is str
    )
    # Empty
    assert insult_detector.remove_stop_and_make_lemma("") == ""


def test_predict(insult_detector):
    # Test with Russian text that contains insults
    assert (
        insult_detector.predict("Ну ты и тварь, конечно.")["verdict"]
        == "Text with insults"
    )

    assert (
        insult_detector.predict(
            "Я работаю 40 часов в неделю, для того чтобы оставаться бедным"
        )
        == "Text with sarcasm"
    )

    # Test with Russian text without insults
    assert (
        insult_detector.predict("А тут мы можем наблюдать восходящий тренд.")["verdict"]
        == "Text without insults and sarcasm"
    )
    # Empty
    assert insult_detector.predict("") != None


def test_get_insult(insult_detector):
    insult_detector.get_insult()
    # Test if the output directory for results is created
    assert os.path.exists(insult_detector.output_dir) == True
    # Test if the output file is created
    assert os.path.isfile(insult_detector.output_dir) == True
