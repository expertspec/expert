from typing import Dict, List
from app.extractor import find_terms_in_text
from app.work_with_terms import load_terms
from app.util import Dictionary, Language


def extract_terms(text: str, lang: str = 'en') -> List[Dict]:
    """
    Находит термины из заданного словаря в тексте и возвращает информацию о найденных терминах.

    Аргументы:
    - text (str): Исходный текст, в котором нужно найти термины.
    - lang (str): Язык исходного текста в котором нужно искать термины.

    Возвращает:
    - found_terms (List[Dict]): Список словарей, представляющих найденные термины в тексте.
    Каждый словарь содержит следующую информацию о найденном термине:
        - "normal_term" (str): Нормализованный термин.
        - "start_pos" (int): Начальная позиция термина в исходном тексте.
        - "end_pos" (int): Конечная позиция термина в исходном тексте.
        - "themes" (List[str]): Список тем, к которым относится термин.

    Пример использования:
    text = "This text contains the word pattern, which is a term."

    found_terms = extract_terms(text, 'en')
    for term in found_terms:
        print(term)
    
    >>  {
            "term": "pattern",
            "start_pos": 27,
            "end_pos": 34,
            "themes": [technical, IT, medical]
        }
    """
    try:
        language = Language(lang)
    except ValueError:
        raise ValueError("Unsupported language. Please choose either 'en' or 'ru'.")
    
    if language == Language('en'):
        dictionary = Dictionary.EN4
    elif language == Language('ru'):
        dictionary = Dictionary.RU4
        
    terms_index = load_terms(dictionary)
    found_terms = find_terms_in_text(text, terms_index)
    return found_terms