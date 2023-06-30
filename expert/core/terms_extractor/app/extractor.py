from typing import Dict, List
from app.text_preprocessing import preproces_text


def find_terms_in_text(text: str, terms: Dict[str, List[str]]) -> List[Dict]:
    tokens = preproces_text(text)
    found_terms = []
    for token, start, end in tokens:
        if token in terms:
            theme = terms[token]
            found_terms.append({
                "normal_term": token,
                "start_pos": start,
                "end_pos": end,
                "themes": theme,
            })
    
    return found_terms
