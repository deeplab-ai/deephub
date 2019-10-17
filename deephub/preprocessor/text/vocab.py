from __future__ import print_function
from __future__ import absolute_import

import csv
from itertools import count
from collections import Counter

from nltk import sent_tokenize
from tqdm import tqdm


class CSVReader:

    def __init__(self, f, delimiter=',', quotechar='"', skipinitialspace=True):
        """
        Reader for labeled text per row. The label is expected to be the first argument.
        :param f: File like object
        :param delimiter:
        :param quotechar:
        :param skipinitialspace:
        """
        self.f = f

        self.csv_handle = csv.reader(f, quotechar=quotechar, delimiter=delimiter, skipinitialspace=skipinitialspace)

    def __iter__(self):
        return self

    def __next__(self):
        row = next(self.csv_handle)
        return int(row[0])-1, " ".join(row[1:])

    def __len__(self):
        if not self.f.seekable():
            raise IOError("File not seekable")

        previous_position = self.f.tell()
        self.f.seek(0)
        total_lines = sum(1 for _ in self.f)
        self.f.seek(previous_position)
        return total_lines


def extract_term_frequencies(reader, tokenizer):
    """
    Extract term frequencies using a file reader
    :param reader:
    :param Callable tokenizer:
    :rtype: Tuple[Dict[str, int], int]
    :return The term frequencies and the total rows of file
    """

    term_frequencies = Counter()
    rows = 0
    for row_label, row_text in tqdm(reader):
        term_frequencies.update(tokenizer(row_text))
        rows += 1

    return term_frequencies, rows


def extract_vocabulary(term_frequencies, min_term_frequency, max_vocab_size=None, reserved_tokens=None):
    """
    Traverse all tokens from a given text and extract vocabulary
    :param Dict[str, int] term_frequencies: frequencies' dictionary of tokens.
    :param int min_term_frequency: The minimum frequency for a term to be saved in vocabulary
    :param int max_vocab_size: The maximum number of terms to keep in vocab. The least frequent will be truncated.
    :param Optional[Dict[str, int]] reserved_tokens: A list of reserved tokens with their id.
    If None is provided then the default <OOV> and <PAD> will be used.
    :rtype: Dict[str, int]
    """

    if reserved_tokens is None:
        reserved_tokens = {
            '<PAD>': 0,
            '<OOV>': 1,
        }
    # Sort terms by frequency (ascending) and filter out those with small values
    terms = [
        term
        for term, frequency in sorted(term_frequencies.items(), key=lambda tup: tup[1], reverse=True)
        if frequency >= min_term_frequency
    ]

    # Truncate to maximum vocab size
    terms = terms[:max_vocab_size]

    # Word to Id
    word_to_id = {
        term: term_id
        for term, term_id in zip(terms, count(start=max(reserved_tokens.values()) + 1))
    }

    # Add reserved tokens in the index
    if word_to_id:
        word_to_id.update(reserved_tokens)

    return word_to_id


def convert_tokens_to_ids(word_to_id, tokens, max_sent_length=None, unknown_token=None):
    """
    Converts tokens to integers with a predefined given mapping from word_to_id dictionary
    :param Dict[str, int] vocabulary:
    :param List[str] tokens:
    :param Optional[str] unknown_token: The token to use for tokens that are not in the index. If
    None is given then '<OOV>' is used.
    :rtype: List[int]
    """
    if unknown_token is None:
        unknown_token = '<OOV>'

    if max_sent_length is not None:
        tokens = [item for sublist in tokens for item in sublist]

    text_to_id = [
        word_to_id[token] if token in word_to_id else word_to_id[unknown_token]
        for token in tokens
    ]
    if max_sent_length is not None:
        text_to_id = [text_to_id[x:x + max_sent_length] for x in range(0, len(text_to_id), max_sent_length)]

    return text_to_id


def split_and_padd_sentences(row_text, max_sent_length, max_doc_length, tokenizer, pad_token):
    """
    Split and pad the sentences appropriately regarding maximum permitted lengths
    :param str row_text: The row text to pad
    :param Callable tokenizer:
    :param int max_sent_length: The maximum sentence length
    :param int max_doc_length: The maximum documents length (# of sentences)
    :param Optional[str] pad_token: The token to use for tokens that are not in the index. If
    None is given then '<PAD>' is used.
    :return: The whole row text with sentences cut and padded properly
    :rtype List[str]
    """
    tokens = []
    sentences = sent_tokenize(row_text)
    for sent in sentences:
        sent_toks = tokenizer(sent)
        if len(sent_toks) < max_sent_length:
            sent_toks += [pad_token] * (max_sent_length - len(sent_toks))
        tokens += [sent_toks[:max_sent_length]]

    if len(tokens) < max_doc_length:
        tokens += [[pad_token] * max_sent_length for _ in range(max_doc_length - len(tokens))]

    return tokens


def convert_text_to_padded_ids(word_to_id, reader, tokenizer, unknown_token=None, pad_token=None,
                               **kwargs):
    """
    Convert a document into ids based on given vocabulary
    :param Dict[str,int] word_to_id: Word to ids vocabulary to use for ids.
    :param Iterator[str] reader:
    :param Callable tokenizer:
    :param Optional[str] unknown_token: The token to use for tokens that are not in the index. If
    None is given then '<OOV>' is used.
    :param Optional[str] pad_token: The token to use for tokens that are not in the index. If
    None is given then '<PAD>' is used.
    :return:
    """
    assert len(kwargs) == 1
    seq_lengths = kwargs[list(kwargs.keys())[0]]

    if pad_token is None:
        pad_token = '<PAD>'

    max_row_length = seq_lengths['max_row_length']
    max_sent_length = seq_lengths['max_sent_length'] if 'max_sent_length' in seq_lengths else None
    for row_label, row_text in reader:
        tokens = tokenizer(row_text)
        if 'max_sent_length' in seq_lengths:
            tokens = split_and_padd_sentences(row_text, max_sent_length, max_row_length, tokenizer, pad_token)
        else:
            if len(tokens) < max_row_length:
                tokens += [pad_token] * (max_row_length - len(tokens))
        token_ids = convert_tokens_to_ids(word_to_id, tokens, max_sent_length=max_sent_length,
                                          unknown_token=unknown_token)

        token_ids = token_ids[:max_row_length]

        if any(isinstance(el, list) for el in token_ids):
            token_ids = [item for sublist in token_ids for item in sublist]

        yield row_label, token_ids
