from typing import List, Optional

from pydantic import BaseModel


class Source(BaseModel):
    hash: str
    score_by_report: str
    score_by_source: str
    name: Optional[str]
    author: Optional[str]
    url: Optional[str]


class Service(BaseModel):
    service_name: str
    originality: str
    plagiarism: str
    source: Optional[List]


class Author(BaseModel):
    surname: Optional[str]
    othernames: Optional[str]
    custom_id: Optional[str]


class LoanBlock(BaseModel):
    text: str
    offset: int
    length: int


class SimpleCheckResult(BaseModel):
    filename: str
    plagiarism: str
    services: List[Service]
    author: Optional[Author]
    loan_blocks: Optional[List[LoanBlock]]
