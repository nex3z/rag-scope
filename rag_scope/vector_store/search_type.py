from enum import Enum


class SearchType(str, Enum):
    SIMILARITY = 'Similarity'
    MMR = 'MMR'
