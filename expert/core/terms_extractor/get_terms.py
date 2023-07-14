from typing import Dict, List

from expert.core.terms_extractor.app.extractor import find_terms_in_text
from expert.core.terms_extractor.app.util import Dictionary, Language
from expert.core.terms_extractor.app.work_with_terms import load_terms


class TermsExtractor():
    """
    Class for extracting terms from text and returning information about found terms.
    """

    def __init__(self, text_or_filepath: str, lang: str = 'en'):
        """
        Initializing the TermsExtractor Object.

        Args:
        - text_or_filepath (str): Text or path to a file with text in which to find terms.
        - lang (str): The language of the source text in which to search for terms. Default 'en'.

        Raises:
        - ValueError: If text_or_filepath is an empty string.
        - ValueError: If an unsupported language is specified. 'en' and 'ru' supported.
        """
        if not text_or_filepath:
            raise ValueError("text_or_filepath cannot be an empty string. Expected text or .txt file path.")

        if text_or_filepath.endswith('.txt'):
            with open(text_or_filepath, 'r') as file:
                self.text = file.read()
        else:
            self.text = text_or_filepath

        try:
            language = Language(lang)
        except ValueError:
            raise ValueError("Unsupported language. Please choose either 'en' or 'ru'.")

        self.language = language

    def extract_terms(self, severity=2) -> List[Dict]:
        """
        Finds terms from the given dictionary in the text and returns information about the found terms.

        Args:
        - severity (int): 'Severity' level. The value 1 uses a larger vocabulary of terms, which may result in
                        to a lot of highlighting common words instead of terms.

        Returns:
        - found_terms (List[Dict]): List of dictionaries representing the found terms in the text.
        Each dictionary contains the following information about the found term:
            - "term" (str): The original non-preprocessed term from the text.
            - "normal_term" (str): Normalized term.
            - "start_pos" (int): The starting position of the term in the source text.
            - "end_pos" (int): The end position of the term in the source text.
            - "themes" (List[str]): List of topics the term refers to.

        Example:
        text = "This text contains the word pattern, which is a term."

        terms_extractor = TermsExtractor(text, lang='en')
        found_terms = terms_extractor.extract_terms(severity=2)
        for term in found_terms:
            print(term)

        >>  {
                "term": "pattern",
                "normal_term": "pattern",
                "start_pos": 27,
                "end_pos": 34,
                "themes": ["technical", "IT", "medical"]
            }
        """
        dictionary = self._switch_dictionary(severity)

        terms_index = load_terms(dictionary)
        found_terms = find_terms_in_text(self.text, terms_index, self.language)
        return found_terms

    def _switch_dictionary(self, severity: int):
        """
        Returns a dictionary based on language and severity level.

        Args:
        - severity (int): severity level. A value greater than 1 corresponds to the dictionary version EN3_5 or RU3_5,
                        which are more stripped-down versions of the dictionary, otherwise
                        a larger version of the EN4 or RU4 dictionary is used.

        Returns:
        - dictionary (Dictionary): Dictionary object corresponding to language and severity level.

        """
        if self.language == Language('en'):
            if severity > 1:
                dictionary = Dictionary.EN3_5
            else:
                dictionary = Dictionary.EN4
        elif self.language == Language('ru'):
            if severity > 1:
                dictionary = Dictionary.RU3_5
            else:
                dictionary = Dictionary.RU4

        return dictionary
