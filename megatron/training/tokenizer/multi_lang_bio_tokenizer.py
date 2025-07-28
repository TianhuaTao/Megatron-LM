from typing import Sequence, Tuple, List, Union
from abc import ABC
from abc import abstractmethod
from .tokenizer import MegatronTokenizer
import logging
import itertools
import json
logger = logging.getLogger(__name__)




import numpy



class _MultiLangBioTokenizer(MegatronTokenizer):
    """
    Multi-language (Protein+DNA) Tokenizer based on Residue level tokenizer
    """

    def __init__(self, args):
        name = 'MultiLangBioTokenizer'
        with open(args.vocab_file, 'r') as f:
            vocab_json = json.load(f)
        self._token2id = vocab_json['tokens']
        self._special_token2id = vocab_json['special_tokens']

        # merge two dict
        self._all_token2id = dict()
        self._all_token2id.update(self._token2id)
        self._all_token2id.update(self._special_token2id)
        
        assert len(self._all_token2id) == max(self._all_token2id.values()) + 1, "token2id is not complete"
        
        self._id2token = {v: k for k, v in self._all_token2id.items()}

        # make sure pad, bos, eos, mask are in the vocab
        assert '<pad>' in self._all_token2id, "vocab does not contain <pad>"
        assert '<bos>' in self._all_token2id, "vocab does not contain <bos>"
        assert '<eos>' in self._all_token2id, "vocab does not contain <eos>"
        assert '[MASK]' in self._all_token2id, "vocab does not contain [MASK]"

        self.current_namespace = 'P' # protein as default
        super().__init__(args.vocab_file)
        
    def map_with_namespace(self, token: str) -> str:
        """Map token to full token with namespace
        e.g. when namespace is 'P', token is 'A', the full token is 'P+A'
        """
        return f"{self.current_namespace}+{token}"

    def tokenize(self, text: str) -> numpy.ndarray:
        """Convert text to embedding ids

        Args:
            text (str): The text to convert

        Returns:
            numpy.ndarray: The converted embedding ids
        """
        tokens = []
        in_tag = False
        tag_buffer = []
        for c in text.strip():
            # don't map tokens starting with '<'
            if c == '<' or in_tag:
                # if we are in a tag, we need to check if we are at the end of the tag
                if c == '>':
                    # if we are at the end of the tag, we need to check if we are at the closing tag
                    tag_buffer.append(c)
                    c_with_namespace = ''.join(tag_buffer)
                    
                    # clear the buffer
                    tag_buffer = []
                    in_tag = False
                # go to the closing tag
                else:
                    in_tag = True
                    tag_buffer.append(c)
                    continue
            else:
                # regular token like "A", "C", "G", "T"
                # map to full token with namespace
                c_with_namespace = self.map_with_namespace(c)
            
            if c_with_namespace in self._token2id:
                tokens.append(self._token2id[c_with_namespace])
            else:
                raise ValueError(f"token {c_with_namespace} not in vocab")
        return numpy.array(tokens, dtype=numpy.int16)
    
    def detokenize(self, ids: numpy.ndarray, with_namespace=False) -> str:
        """Convert embedding ids to text

        Args:
            ids (numpy.ndarray): The ids to convert

        Returns:
            str: The converted text
        """
        tokens = [self._id2token[i] for i in ids]
        if with_namespace:
            return ' '.join(tokens)
        else:
            # "P+A" -> "A"
            new_tokens = []
            for token in tokens:
                if '+' in token:
                    # remove the namespace
                    token = token.split('+')[1]
                new_tokens.append(token)

            return ''.join(new_tokens)
            
    @property
    def vocab(self):
        """Dictionary from vocab text token to id token"""
        return self._token2id
        
    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token"""
        return self._id2token
        

    @property
    def vocab_size(self):
        """The vocabulary size"""
        return len(self._all_token2id)


    @property
    def pad(self):
        return self._token2id['<pad>']

    @property
    def eod(self):
        """ The EOD token id is the same as EOS """
        return self.eos

    @property
    def bos(self):
        return self._special_token2id['<bos>']

    @property
    def eos(self):
        return self._special_token2id['<eos>']

    @property
    def mask(self):
        return self._special_token2id['[MASK]']



""" A reference copy of vocabulary

{
    "special_tokens": {
        "<pad>": 0,
        "[MASK]": 1,
        "<bos>": 2,
        "<eos>": 3
    },
    "tokens": {
        "<protein>": 4,
        "</protein>": 5,
        "P+L": 6,
        "P+A": 7,
        "P+G": 8,
        "P+V": 9,
        "P+S": 10,
        "P+E": 11,
        "P+R": 12,
        "P+T": 13,
        "P+I": 14,
        "P+D": 15,
        "P+P": 16,
        "P+K": 17,
        "P+Q": 18,
        "P+N": 19,
        "P+F": 20,
        "P+Y": 21,
        "P+M": 22,
        "P+H": 23,
        "P+W": 24,
        "P+C": 25,
        "P+X": 26,
        "P+B": 27,
        "P+U": 28,
        "P+Z": 29,
        "P+O": 30,
        "P+.": 31,
        "P+-": 32,
        "<dna>": 33,
        "</dna>": 34,
        "<ssu>": 35,
        "</ssu>": 36,
        "D+A": 37,
        "D+C": 38,
        "D+G": 39,
        "D+T": 40
    }
}

"""