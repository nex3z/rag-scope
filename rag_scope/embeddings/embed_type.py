from enum import Enum


class EmbedType(str, Enum):
    QUERY = 'Query'
    DOCUMENT = 'Document'
