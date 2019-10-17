from __future__ import absolute_import
from __future__ import print_function

from collections import Counter
import io
import random

import nltk

from deephub.preprocessor.text.vocab import extract_vocabulary, convert_text_to_padded_ids, CSVReader, \
    extract_term_frequencies, split_and_padd_sentences, convert_tokens_to_ids
from deephub.preprocessor.text.tokenizers import tokenize


# Assure that resources are downloaded
# nltk.download('punkt')


def test_empty():
    vocab = extract_vocabulary(Counter(), 0)
    assert {} == vocab, "Vocabulary of empty file not empty."


def test_vocabulary_extraction(datadir):
    max_vocab_size = 10
    min_term_freq = 1
    with io.open(str(datadir / 'preprocessor' / 'dataset.csv'), 'rt', encoding='utf-8') as f:
        term_frequencies, file_rows = extract_term_frequencies(CSVReader(f), lambda x: tokenize('nltk', x.lower()))

    vocab_toy = extract_vocabulary(term_frequencies, min_term_freq, max_vocab_size=max_vocab_size)

    assert max_vocab_size + 2 == len(vocab_toy), "Vocabulary size is unequal with given max size"
    assert 14 == file_rows, "The file was not read properly. Total rows differ from real."


def test_sentences_split(datadir):
    max_sent_len = 4
    max_doc_len = random.randint(0, 10)
    with io.open(str(datadir / 'preprocessor' / 'dataset.csv'), 'rt', encoding='utf-8') as f:
        reader = CSVReader(f)

        for row_label, row_text in reader:
            tokens = split_and_padd_sentences(row_text, max_sent_len, max_doc_len,
                                              lambda x: tokenize('nltk', x.lower()), '<PAD>')
            assert [len(x) for x in tokens] == [max_sent_len] * len(tokens), \
                "Sentences are not properly padded when max_sent_len is given."
            assert len(tokens) >= max_doc_len, "Documents are not properly padded when row_sent_len is given."


def test_convert_text_to_padded_ids(datadir):
    word_to_id = {'partial': 2, ':)': 3, 'caused': 4, 'global': 5, '<PAD>': 0, 'n94': 8, 'increase': 11, 'mild': 6,
                  'debris': 7, 'skin': 9, 'n30': 10, '<OOV>': 1}
    max_sent_len = 4
    max_doc_len = 10
    seq_lengths = {'max_sent_length': max_sent_len,
                   'max_row_length': max_doc_len}
    with io.open(str(datadir / 'preprocessor' / 'dataset.csv'), 'rt', encoding='utf-8') as f:
        reader = CSVReader(f)
        gen = convert_text_to_padded_ids(word_to_id, reader, lambda x: tokenize('nltk', x.lower()),
                                         seq_lengths=seq_lengths)
        for lbl, text_ids in gen:
            text_ids = [text_ids[x:x + max_sent_len]
                        for x in range(0, len(text_ids), max_sent_len)]
            assert len(text_ids) == max_doc_len, \
                "Documents are not properly padded when row_sent_len is given."

            assert [len(x) for x in text_ids] == [max_sent_len] * len(text_ids), \
                    "Sentences are not properly padded when max_sent_len is given."


def test_convert_text_to_padded_ids_without_max_sent(datadir):
    word_to_id = {'partial': 2, ':)': 3, 'caused': 4, 'global': 5, '<PAD>': 0, 'n94': 8, 'increase': 11, 'mild': 6,
                  'debris': 7, 'skin': 9, 'n30': 10, '<OOV>': 1}
    max_doc_len = 10
    seq_lengths = {'max_row_length': max_doc_len}
    with io.open(str(datadir / 'preprocessor' / 'dataset.csv'), 'rt', encoding='utf-8') as f:
        reader = CSVReader(f)
        gen = convert_text_to_padded_ids(word_to_id, reader, lambda x: tokenize('nltk', x.lower()),
                                         seq_lengths=seq_lengths)
        for lbl, text_ids in gen:
            assert len(text_ids) == max_doc_len, \
                "Documents are not properly padded when row_sent_len is given."


def test_extract_term_frequencies(datadir):
    with io.open(str(datadir / 'preprocessor' / 'dataset.csv'), 'rt', encoding='utf-8') as f:
        reader = CSVReader(f)
        freqs, _ = extract_term_frequencies(reader, lambda x: tokenize('nltk', x.lower()))
    assert 739 == len(freqs), "Actual and calculated length of term-freq structure do not match"
