from typing import Dict, List

from expert.core.terms_extractor.app.text_preprocessing import preproces_text
from expert.core.terms_extractor.app.util import Language


def find_terms_in_text(text: str, terms: Dict[str, List[str]], lang=Language.EN) -> List[Dict]:
    """
    Находит термины из заданного словаря в тексте и возвращает информацию о найденных терминах.

    Аргументы:
    - text (str): Исходный текст, в котором нужно найти термины.
    - terms (Dict[str, List[str]]): Словарь терминов, где ключ - нормализованный термин, а значение - список тем,
                                    к которым относится термин.
    - lang (Language): Язык исходного текста. По умолчанию Language.EN.

    Возвращает:
    - found_terms (List[Dict]): Список словарей, представляющих найденные термины в тексте.
      Каждый словарь содержит следующую информацию о найденном термине:
        - "term" (str): Оригинальный термин из исходного текста.
        - "normal_term" (str): Нормализованный термин.
        - "start_pos" (int): Начальная позиция термина в исходном тексте.
        - "end_pos" (int): Конечная позиция термина в исходном тексте.
        - "themes" (List[str]): Список тем, к которым относится термин.

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
