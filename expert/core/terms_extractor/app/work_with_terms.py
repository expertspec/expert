import pickle
from typing import Dict, List

from expert.core.terms_extractor.app.util import Dictionary


def load_terms_from_file(terms_path: str) -> Dict[str, List[str]]:
    """
    Загружает словарь из файла с указанным путем.

    Аргументы:
    - terms_path (str): Путь к файлу, содержащему словарь.

    Возвращает:
    - term_index (Dict[str, List[str]]): Загруженный словарь, где ключами являются строки,
    а значениями - списки строк.
    """
    with open(terms_path, 'rb') as f:
        term_index = pickle.load(f)

    return term_index


def load_terms(dict_name: Dictionary) -> Dict[str, List[str]]:
    """
    Загружает словарь на основе указанного имени словаря.

    Аргументы:
    - dict_name (Dictionary): Имя словаря, указанное с использованием перечисления Dictionary.

    Возвращает:
    - term_index (Dict[str, List[str]]): Загруженный словарь, где ключами являются термины,
    а значениями - категория термина.

    Пример использования:
    term_index = load_terms(Dictionary.EN4)
    print(term_index)
    """
    dict_map = {
        Dictionary.EN4: 'data/en_term_index.pickle',
        Dictionary.EN3_5: 'data/en_term_index_3.5.pickle',
        Dictionary.RU4: 'data/ru_term_index_4.pickle',
        Dictionary.RU3_5: 'data/ru_term_index_3.5.pickle'
    }

    term_index = load_terms_from_file(dict_map[dict_name])

    return term_index


