from dataclasses import dataclass
from Text import Text

@dataclass
class Document:
    text: Text
    tokens: list[str]
