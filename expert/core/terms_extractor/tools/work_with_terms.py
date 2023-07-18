import pickle
from typing import Dict, List

from expert.core.terms_extractor.tools.util import Dictionary


def load_terms_from_file(terms_path: str) -> Dict[str, List[str]]:
    """
    Loads a dictionary from a file with the specified path.

    Args:
    - terms_path (str): Path to the file containing the dictionary.

    Returns:
    - term_index (Dict[str, List[str]]): Loaded dictionary where keys are strings,
    and values ​​are lists of strings.
    """
    with open(terms_path, "rb") as f:
        term_index = pickle.load(f)

    return term_index


def load_terms(dict_name: Dictionary) -> Dict[str, List[str]]:
    """
    Loads a dictionary based on the specified dictionary name.

    Args:
    - dict_name (Dictionary): Dictionary name specified using the Dictionary enum.

    Returns:
    - term_index (Dict[str, List[str]]): Loaded dictionary where terms are keys,
    and values ​​- the category of the term.

    Example:
    term_index = load_terms(Dictionary.EN4)
    print(term_index)
    """
    dict_map = {
        Dictionary.EN4: "./expert/core/terms_extractor/data/en_term_index.pickle",
        Dictionary.EN3_5: "./expert/core/terms_extractor/data/en_term_index_3.5.pickle",
        Dictionary.RU4: "./expert/core/terms_extractor/data/ru_term_index_4.pickle",
        Dictionary.RU3_5: "./expert/core/terms_extractor/data/ru_term_index_3.5.pickle",
    }

    term_index = load_terms_from_file(dict_map[dict_name])

    return term_index
