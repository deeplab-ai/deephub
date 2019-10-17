from __future__ import absolute_import
from __future__ import print_function

import re

import spacy
from nltk import regexp_tokenize  # noqa: F401

_spacy_english_pipeline_g = None


def _spacy_english_pipeline():
    """
    Get (lazily loaded) spacy pipeline
    """
    global _spacy_english_pipeline_g
    if _spacy_english_pipeline_g is None:
        _spacy_english_pipeline_g = spacy.load('en', disable=['ner'])
    return _spacy_english_pipeline_g


def taboola_tokenizer(title):
    """
    Tokenize a document based on taboolas title tokenizer
    :param str title:
    :rtype: List[str]
    """

    # 5000 it / sec
    special_mark = r'(?:[.!?] )'
    max_seq_length = 4
    SPECIAL_MARKS_SEQUENCES = re.compile('(' + special_mark + '{1,' + str(max_seq_length) + '})(' + special_mark + '*)',
                                         re.UNICODE)

    if title is None or title.strip() == "":
        return []

    # split to tokens:
    TOKENIZER = re.compile(r'\w+|[^\w\s]', re.UNICODE)
    tokenized = TOKENIZER.findall(title.lower())

    # truncate any series of ./?/!
    # this is needed since longer sequences might be rare, and thus will be filtered out

    tokenized = ' '.join(tokenized) + ' '
    tokenized = re.sub(SPECIAL_MARKS_SEQUENCES, lambda match: match.groups()[0].replace(' ', '') + ' ',
                       tokenized).strip()
    tknzd = tokenized.split(' ')
    return tknzd


def spacy_tokenizer(line):
    """
    Tokenize sentence based on spacy tokenizer (only text)
    :param str line:
    :rtype: List[str]
    """

    # 40 it/sec
    line = line.replace('\n', ' ').replace('\r', '').replace('\\n', '')

    sentence = _spacy_english_pipeline()(line)
    tknzd = []
    for token in sentence:
        tknzd.append(token.text)
    return tknzd


def wordpunct_tokenizer(line):
    """
    Tokenize sentence based on NLTK's WordPunctTokenizer
    :param srt line:
    :rtype: List[str]
    """
    # 9000 it/sec
    re.sub(r"^\s*(.-)\s*$", "%1", line).replace("\\n", "\n")
    pattern = r'''(?x)          # set flag to allow verbose regexps
            (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
          | \w+(?:-\w+)*        # words with optional internal hyphens
          | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
          | \.\.\.              # ellipsis
          | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
        '''
    tknzd = regexp_tokenize(line, pattern)
    return tknzd


def character_tokenizer(line):
    """
    Tokenize sentence based on NLTK's WordPunctTokenizer
    :param srt line:
    :rtype: List[str]
    """
    tknzd = list(line)
    return tknzd


def tokenize(tokenizer_name, line):
    """
    Tokenize a sentence using an existing tokenizer
    :param str line: The sentence to be tokenized
    :param str tokenizer_name: Supported tokenizers are:
     * 'taboola'
     * 'wordpunct'
     * 'spacy'
    :rtype: List[str]
    """
    supported_tokenizers = {
        'taboola': taboola_tokenizer,
        'nltk': wordpunct_tokenizer,
        'spacy': spacy_tokenizer,
        'char': character_tokenizer
    }

    if tokenizer_name not in supported_tokenizers:
        raise ValueError('Unknown tokenizer "{}"'.format(tokenizer_name))
    return supported_tokenizers[tokenizer_name](line)
