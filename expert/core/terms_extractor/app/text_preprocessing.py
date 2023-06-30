import re
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple


def get_tokens(text: str) -> List[Tuple[str, int, int]]:
    """
    Извлекает токены из текста и возвращает список кортежей с токенами и их позициями.

    Аргументы:
    - text: Текст, из которого нужно извлечь токены.

    Возвращает:
    - Список кортежей формата ('token', start_pos, end_pos), представляющих токены и соответствующие позиции, начальная и конечная позиция токена в тексте.

    Пример использования:
    >>> text = "Hello, world! This is a sample text."
    >>> result = get_tokens(text)
    >>> print(result)
    [('Hello', 0, 5), (',', 5, 6), ('world', 7, 12), ('!', 12, 13), ('This', 14, 18), ('is', 19, 21), ('a', 22, 23), ('sample', 24, 30), ('text', 31, 35), ('.', 35, 36)]
    """
    tokens_with_positions = [(match.group(), match.start(), match.end()) for match in re.finditer(r'\w+|\S', text)]
    return tokens_with_positions


def prep_tokens(tokens: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """
    Подготавливает токены в списке кортежей для дальнейшей обработки. 

    Аргументы:
    - tokens: Список кортежей формата ('token', start_pos, end_pos), представляющих токены и соответствующие позиции.

    Возвращает:
    - Список кортежей с подготовленными токенами, где каждый элемент представляет собой кортеж вида ('prepared_token', start_pos, end_pos).

    Пример использования:
    >>> tokens = [('Hello', 0, 5), ('world!', 6, 12), ('This', 13, 17), (' is ', 18, 22), ('a', 23, 24), (' sample ', 25, 32), ('text.', 33, 38)]
    >>> result = prep_tokens(tokens)
    >>> print(result)
    [('hello', 0, 5), ('world', 6, 12), ('this', 13, 17), ('is', 18, 22), ('a', 23, 24), ('sample', 25, 32), ('text', 33, 38)]
    """
    tokens = [(token[0].lower(), token[1], token[2]) for token in tokens]
    tokens = [(re.sub(r'[^\w ]', ' ', token[0]), token[1], token[2]) for token in tokens]
    tokens = [(re.sub(r'  +', ' ', token[0]).strip(), token[1], token[2]) for token in tokens]
    tokens = [(token[0], token[1], token[2]) for token in tokens if token[0]]
    return tokens


def create_2grams(tokens: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """
    Создает список 2-грамм на основе исходного списка кортежей.

    Аргументы:
    - tokens: Список кортежей формата ('token', start_pos, end_pos), представляющих токены и соответствующие позиции.

    Возвращает:
    - Список 2-грамм, где каждый элемент представляет собой кортеж вида ('token1 token2', start_pos, end_pos).

    Пример использования:
    >>> tokens = [('token1', 0, 5), ('token2', 6, 12), ('token3', 13, 20)]
    >>> result = create_2grams(tokens)
    >>> print(result)
    
    [('token1 token2', 0, 12), ('token2 token3', 6, 20)]
    """
    ngrams = []
    for i in range(len(tokens) - 1):
        token1 = tokens[i][0]
        token2 = tokens[i + 1][0]
        start_pos = tokens[i][1]
        end_pos = tokens[i + 1][2]
        ngram = ('{} {}'.format(token1, token2), start_pos, end_pos)
        ngrams.append(ngram)
    return ngrams


def create_3grams(tokens: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """
    Создает список 3-грамм на основе исходного списка кортежей.

    Аргументы:
    - tokens: Список кортежей формата ('token', start_pos, end_pos), представляющих токены и соответствующие позиции.

    Возвращает:
    - Список 3-грамм, где каждый элемент представляет собой кортеж вида ('token1 token2 token3', start_pos, end_pos).

    Пример использования:
    >>> tokens = [('token1', 0, 5), ('token2', 6, 12), ('token3', 13, 20), ('token4', 21, 30)]
    >>> result = create_3grams(tokens)
    >>> print(result)
    [('token1 token2 token3', 0, 20), ('token2 token3 token4', 6, 30)]
    """
    ngrams = []
    for i in range(len(tokens) - 2):
        token1 = tokens[i][0]
        token2 = tokens[i + 1][0]
        token3 = tokens[i + 2][0]
        start_pos = tokens[i][1]
        end_pos = tokens[i + 2][2]
        ngram = ('{} {} {}'.format(token1, token2, token3), start_pos, end_pos)
        ngrams.append(ngram)
    return ngrams


def lemm_tokens(tokens: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """
    Производит лемматизацию токенов в списке кортежей.

    Аргументы:
    - tokens: Список кортежей формата ('token', start_pos, end_pos), представляющих токены и соответствующие позиции.

    Возвращает:
    - Список лемматизированных токенов, где каждый элемент представляет собой кортеж вида ('lemmatized_token', start_pos, end_pos).

    Пример использования:
    >>> tokens = [('running', 0, 6), ('dogs', 7, 11), ('quickly', 12, 19)]
    >>> result = lemm_tokens(tokens)
    >>> print(result)
    [('run', 0, 6), ('dog', 7, 11), ('quickly', 12, 19)]
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [(lemmatizer.lemmatize(token[0]), token[1], token[2]) for token in tokens]
    return lemmatized_tokens

def preproces_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Выполняет предобработку текста, включая токенизацию, нормализацию и создание n-грамм.

    Аргументы:
    - text (str): Входной текст, который требуется предобработать.

    Возвращает:
    - tokens (List[Tuple[str, int, int]]): Список токенов с их позициями в исходном тексте.
    Каждый токен представлен в виде кортежа (token, start_pos, end_pos), где:
        - token (str): Токен.
        - start_pos (int): Начальная позиция токена в исходном тексте.
        - end_pos (int): Конечная позиция токена в исходном тексте.

    Пример использования:
    text = "Это пример текста для предобработки."
    tokens = preprocess_text(text)
    for token in tokens:
        print(token)
    """
    tokens = get_tokens(text)
    tokens = prep_tokens(tokens)
    tokens = lemm_tokens(tokens)
    tokens2 = create_2grams(tokens)
    tokens3 = create_3grams(tokens)
    tokens = tokens + tokens2 + tokens3
    
    return tokens