# Terms Extractor
Module for extracting terms from text

### About
The module searches for professional terms in the text and extracts them.

To use it, you need to initialize the class <b>TermsExtractor</b> from get_terms.py.

Class <b>TermsExtractor</b> takes arguments:
- text_or_filepath (str): The text or path to the text file in which to find the terms.
- lang (str): The language of the source text in which to search for terms. The default is 'en'.

To search for terms, you need to call function <b>extract_terms</b> of class <b>TermsExtractor</b>, which takes argument:
- severity (int): 'Severity' level. A value of 1 uses a larger vocabulary of terms, which can result in a lot of emphasis on generic words instead of terms. It is recommended to use the default value == 2.

Returns:
- found_terms (List[Dict]): List of dictionaries representing the found terms in the text.

Each dictionary contains the following information about the found term:
- "term" (str): Original non-preprocessed term from the text.
- "normal_term" (str): Normalized term.
- "start_pos" (int): Starting position of the original term in the source text.
- "end_pos" (int): End position of the original term in source text.
- "themes" (List[str]): List of topics to which the term refers.

### The example of the usage

```python
text = "This text contains the word pattern, which is a term."

terms_extractor = TermsExtractor(text, lang='en')
found_terms = terms_extractor.extract_terms(severity=2)
for term in found_terms:
    print(term)

""" Results
>>  {
        "term": "pattern",
        "normal_term": "pattern",
        "start_pos": 27,
        "end_pos": 34,
        "themes": ["technical", "IT", "medical"]
    }
"""
```
