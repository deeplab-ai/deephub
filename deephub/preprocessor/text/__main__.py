from __future__ import absolute_import
from __future__ import print_function

import argparse
import glob
import datetime
import sentencepiece as spc
from collections import Counter


from pathlib import Path
from tqdm import tqdm
from deephub.preprocessor.text.vocab import extract_term_frequencies, CSVReader, extract_vocabulary, \
    convert_text_to_padded_ids
from deephub.preprocessor.text.text_io import write_example_to_tfrecord, save_dict_to_json, file_generator
from deephub.preprocessor.text.tokenizers import tokenize

parser = argparse.ArgumentParser()

parser.add_argument('training-split', type=str, help="File with training dataset that will be "
                                                     "used to create vocabulary")

parser.add_argument('extra-splits', type=str, nargs='*', help="More datasets to extract TFRecrods.""The vocabulary of "
                                                              "training dataset will be used")

parser.add_argument('--min-count-term', default=5, help="Minimum count for term in the dataset",
                    type=int)
parser.add_argument('--tokenizer', default='nltk', required=True, help="Tokenizer to use nltk,spacy,taboola. "
                                                                       "See tokenizers.py.", type=str)
parser.add_argument('--max-vocab-size', default=100000, required=False, help="Maximum vocabulary size", type=int)

parser.add_argument('--max-sent-length', default=None, help="Maximum number of words in a sentence. Used in the case "
                                                            "of HAN.", type=int)

parser.add_argument('--max-row-length', default=15, help="Maximum number of tokens in a training example.", type=int)
parser.add_argument('--train', default='', help="Maximum number of tokens in a training example. Currenlt spc only.",
                    type=str)

parser.add_argument('--pad-token', default="<PAD>", help="Token to be used for padded tokens", type=str)
parser.add_argument('--oov-token', default="<OOV>", help="Token to be used for out of vocabulary tokens", type=str)
parser.add_argument('--output-dir', required=True, help="The output directory to write tf records", type=str)
parser.add_argument('--model-type', required=False, default='unigram', help="Choose from unigram (default), bpe, char,"
                                                                            " or word. The input sentence must be "
                                                                            "pretokenized when using word type. See "
                                                                            "https://github.com/google/sentencepiece",
                    type=str)

args = parser.parse_args()
max_vocab_size = args.max_vocab_size
output_dir = Path(args.output_dir).resolve()

train_splits = getattr(args, 'training-split')
extra_splits = getattr(args, 'extra-splits')


def tokenizer(row):
    return tokenize(args.tokenizer, row.lower())


# Extract vocabulary from training dataset
term_frequencies = Counter()
total_rows = 0
if args.max_sent_length is not None:
    seq_lengths = {'max_sent_length': args.max_sent_length,
                   'max_row_length': args.max_row_length}
else:
    seq_lengths = {'max_row_length': args.max_row_length}
if args.train == 'spc':
    spc.SentencePieceTrainer.Train('--input=' + glob.glob(train_splits)[0] + ' --model_prefix=sentencepiece_model_'
                                   + str(max_vocab_size) + ' --vocab_size=' + str(max_vocab_size) + ' --model_type=' +
                                   args.model_type + ' --pad_id=3 --character_coverage=1.0 --max_sentence_length=4096')
    for pattern in [train_splits] + extra_splits:

        for input_filename in tqdm(glob.glob(pattern)):
            output_filename = (output_dir / Path(input_filename).name).with_suffix('')

            with open(input_filename, 'r') as input_f:
                frows = write_example_to_tfrecord(output_filename=output_filename,
                                                  row_generator=file_generator(CSVReader(input_f)),
                                                  features_type='string_list')
else:
    if args.tokenizer != 'char':
        print("Parsing every file and counting frequencies...")
        for fname in tqdm(glob.glob(train_splits)):
            with open(fname, 'r') as f:
                file_term_frequencies, file_rows = extract_term_frequencies(CSVReader(f), tokenizer)
                term_frequencies += file_term_frequencies
                total_rows += file_rows

        print("Building word vocabulary...")
        word_to_id = extract_vocabulary(term_frequencies,
                                        min_term_frequency=args.min_count_term,
                                        max_vocab_size=args.max_vocab_size)
    else:
        char_vocab = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/|_#$%^&*~`+-=<>()[]{} "
        word_to_id = {}
        for i, c in enumerate(char_vocab):
            word_to_id[c] = i + 2
        word_to_id.update({args.pad_token: 0, args.oov_token: 1})
    # Build word vocab with train and test datasets
    manifest = {
        'train_examples': total_rows,
        'tokenizer': args.tokenizer,
        'vocab_size': len(word_to_id),
        'pad_token': args.pad_token,
        'oov_token': args.oov_token,
        'generated_at': datetime.datetime.now().isoformat()
    }
    manifest.update(seq_lengths)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    save_dict_to_json(manifest, str(output_dir) + '/data_stats.json')
    save_dict_to_json(word_to_id, str(output_dir) + '/vocab.json')
    print("Saved vocabulary. Parsing every file and writing to tfrecords...")

    for pattern in [train_splits] + extra_splits:

        for input_filename in tqdm(glob.glob(pattern)):
            output_filename = (output_dir / Path(input_filename).name).with_suffix('')

            with open(input_filename, 'r') as input_f:
                frows = write_example_to_tfrecord(
                    output_filename=output_filename,
                    row_generator=convert_text_to_padded_ids(
                        word_to_id=word_to_id,
                        reader=CSVReader(input_f),
                        tokenizer=tokenizer,
                        seq_lengths=seq_lengths
                    ))
