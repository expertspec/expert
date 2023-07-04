from typing import Dict, List

from expert.core.terms_extractor.app.extractor import find_terms_in_text
from expert.core.terms_extractor.app.util import Dictionary, Language
from expert.core.terms_extractor.app.work_with_terms import load_terms


class TermsExtractor():
    """
    Класс для извлечения терминов из текста и возврата информации о найденных терминах.
    """

    def __init__(self, text_or_filepath: str, lang: str = 'en'):
        """
        Инициализация объекта TermsExtractor.

        Аргументы:
        - text_or_filepath (str): Текст или путь к файлу с текстом, в котором нужно найти термины.
        - lang (str): Язык исходного текста, в котором нужно искать термины. По умолчанию 'en'.

        Raises:
        - ValueError: Если text_or_filepath является пустой строкой.
        - ValueError: Если указан неподдерживаемый язык. Поддерживаются 'en' и 'ru'.
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
        Находит термины из заданного словаря в тексте и возвращает информацию о найденных терминах.

        Аргументы:
        - severity (int): Уровень 'строгости'. Значение 1 использует более крупный словарь терминов, что может привести
                        к большому количеству выделения общих слов вместо терминов.

        Возвращает:
        - found_terms (List[Dict]): Список словарей, представляющих найденные термины в тексте.
        Каждый словарь содержит следующую информацию о найденном термине:
            - "term" (str): Исходный не предобработанный термин из текста.
            - "normal_term" (str): Нормализованный термин.
            - "start_pos" (int): Начальная позиция термина в исходном тексте.
            - "end_pos" (int): Конечная позиция термина в исходном тексте.
            - "themes" (List[str]): Список тем, к которым относится термин.

        Пример использования:
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
        Возвращает словарь, основываясь на языке и уровне строгости.

        Аргументы:
        - severity (int): Уровень строгости. Значение больше 1 соответствует версии словаря EN3_5 или RU3_5,
                        что являются более урезанными версиями словаря, в противном случае
                        используется белее крупная версия словаря EN4 или RU4.

        Возвращает:
        - dictionary (Dictionary): Объект словаря, соответствующий языку и уровню строгости.

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
