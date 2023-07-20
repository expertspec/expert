from typing import Dict, List

from expert.core.terms_extractor.tools.text_preprocessing import preproces_text
from expert.core.terms_extractor.tools.util import Language


def find_terms_in_text(text: str, terms: Dict[str, List[str]], lang=Language.EN) -> List[Dict]:
    """
    Finds terms from the given dictionary in the text and returns information about the found terms.

    Args:
    - text (str): The source text in which to find the terms.
    - terms (Dict[str, List[str]]): Dictionary of terms, where key is a normalized term and value is a list of topics,
                                    to which the term refers.
    - lang (Language): Source text language. The default is Language.EN.

    Returns:
    - found_terms (List[Dict]): A list of dictionaries representing found terms in the text.
      Each dictionary contains the following information about the found term:
        - "term" (str): Original term from source text.
        - "normal_term" (str): Normalized term.
        - "start_pos" (int): The starting position of the term in the source text.
        - "end_pos" (int): The end position of the term in the source text.
        - "themes" (List[str]): List of topics the term refers to.

    """

    tokens = preproces_text(text)
    found_terms = []
    for token, start, end, original_token in tokens:
        if token in terms:
            theme = terms[token]
            found_terms.append({
                "term": original_token,
                "normal_term": token,
                "start_pos": start,
                "end_pos": end,
                "themes": theme,
            })

    return found_terms
