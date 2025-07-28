from typing import Sequence, Tuple, List, Union
from abc import ABC
from abc import abstractmethod
from .tokenizer import MegatronTokenizer
import logging
import itertools
import json
logger = logging.getLogger(__name__)




import numpy



class _ProteinTokenizer(MegatronTokenizer):
    """
    Protein Tokenizer based on Residue level tokenizer
    """

    def __init__(self, args):
        name = 'ProteinTokenizer'
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
        assert '[pad]' in self._all_token2id, "vocab does not contain [pad]"
        assert '<bos>' in self._all_token2id, "vocab does not contain <bos>"
        assert '<eos>' in self._all_token2id, "vocab does not contain <eos>"
        assert '[MASK]' in self._all_token2id, "vocab does not contain [MASK]"

        super().__init__(args.vocab_file)


    def tokenize(self, text: str) -> numpy.ndarray:
        """Convert text to embedding ids

        Args:
            text (str): The text to convert

        Returns:
            numpy.ndarray: The converted embedding ids
        """
        tokens = []
        for c in text.strip():
            if c in self._token2id:
                tokens.append(self._token2id[c])
            else:
                raise ValueError(f"token {c} not in vocab")
        return numpy.array(tokens, dtype=numpy.int16)
    
    def detokenize(self, ids: numpy.ndarray) -> str:
        """Convert embedding ids to text

        Args:
            ids (numpy.ndarray): The ids to convert

        Returns:
            str: The converted text
        """
        return ''.join([self._id2token[i] for i in ids])
    
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
        return self._token2id['[pad]']

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
    "tokens":
    {
        "[pad]": 0,
        "L": 1,
        "A": 2,
        "G": 3,
        "V": 4,
        "S": 5,
        "E": 6,
        "R": 7,
        "T": 8,
        "I": 9,
        "D": 10,
        "P": 11,
        "K": 12,
        "Q": 13,
        "N": 14,
        "F": 15,
        "Y": 16,
        "M": 17,
        "H": 18,
        "W": 19,
        "C": 20,
        "X": 21,
        "B": 22,
        "U": 23,
        "Z": 24,
        "O": 25,
        ".": 26,
        "-": 27
    },
    "special_tokens":{
        "[MASK]": 28,
        "<bos>": 29,
        "<eos>": 30
    }
}


"""