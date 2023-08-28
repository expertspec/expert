import pytest

from expert.core.terms_extractor.get_terms import TermsExtractor


@pytest.fixture()
def ru_text():
    ru_text = "В этом тексте содержится один термин - паттерн"

    return ru_text


@pytest.fixture()
def en_text():
    en_text = "This text contains one term - oscillation"

    return en_text


def test_ru_text_term_extractor(ru_text):
    extractor = TermsExtractor(ru_text, lang="ru")
    terms = extractor.extract_terms()

    assert len(terms) == 1
    assert terms[0]["term"] == "паттерн"
    assert terms[0]["start_pos"] == 39
    assert terms[0]["end_pos"] == 46


def test_en_text_term_extractor(en_text):
    extractor = TermsExtractor(en_text, lang="en")
    terms = extractor.extract_terms()

    assert len(terms) == 1
    assert terms[0]["term"] == "oscillation"
    assert terms[0]["start_pos"] == 30
    assert terms[0]["end_pos"] == 41
