from yargy import Parser, rule, and_, or_
from yargy.tokenizer import MorphTokenizer
import gzip
from pprint import pprint
from tqdm import tqdm

from GeneralRule.GeneralRule import GENERAL_RULE

TOKENIZER = MorphTokenizer()

with gzip.open("news.txt.gz", "rt", encoding="utf-8") as file:
    lines = [line for line in file]

parser = Parser(GENERAL_RULE)
persons = []
for line in tqdm(lines):
    if 'родил' not in line:
        continue
    matches = list(parser.findall(line))
    spans = []
    for match in matches:
        spans.append(match.span)
        if match.fact is not None:
            persons.append(match.fact)

parser = Parser(GENERAL_RULE)
for line in tqdm(lines):
    if 'рожден' not in line:
        continue
    matches = list(parser.findall(line))
    spans = []
    for match in matches:
        spans.append(match.span)
        if match.fact is not None:
            persons.append(match.fact)


pprint(persons)
print("Всего записей: ", len(persons))