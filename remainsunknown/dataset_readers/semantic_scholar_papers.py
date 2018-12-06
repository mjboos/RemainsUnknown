from typing import Dict
import json
import logging
from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers import Token
import re

@DatasetReader.register("s2_remains")
class SemanticScholarRemainsReader(DatasetReader):
    """
    FILL IN
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
                 super().__init__(lazy)
                 self._tokenizer = tokenizer or WordTokenizer()
                 self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                description = json.loads(line)
                title = description["title"]
                abstract = description["paperAbstract"]
                yield self.text_to_instance(title, abstract)
    
    #TODO: add textfield that has "remains" sentence
    @overrides
    def text_to_instance(self, title: str, abstract: str) -> Instance:
        tokenized_title = self._tokenizer.tokenize(title)
        title_field = TextField(tokenized_title, self._token_indexers)
        fields = {'title': title_field}
        # only keep part of abstract that is matched
        all_sentences_plus_remains = '(.)* remains [^.]*\.'
        only_remains = '[^.]* remains [^.]*\.'
        p = re.compile(all_sentences_plus_remains)
        p2 = re.compile(only_remains)
        matches = p.search(abstract)
        # TODO: this might be problematic later
        if matches:
            remains = p2.search(abstract)
            # remains sentence is the first sentence
            if remains.group(0) == matches.group(0):
                tokenized_remains = self._tokenizer.tokenize('UNAPPLICABLE')
            else:
                abstract = re.sub('[^.]* remains [^.]*\.', '', matches.group(0))
                tokenized_remains = self._tokenizer.tokenize(remains.group(0))
            remains_field = TextField(tokenized_remains, self._token_indexers)
            fields['remains'] = remains_field
        tokenized_abstract = self._tokenizer.tokenize(abstract)
        abstract_field = TextField(tokenized_abstract, self._token_indexers)
        fields['abstract'] = abstract_field
        return Instance(fields)