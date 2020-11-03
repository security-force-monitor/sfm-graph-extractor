import os
from train import model_fn
from pathlib import Path
import functools
import json
import pickle
from ne_def import correct_position

import tensorflow as tf
import pprint
pp = pprint.PrettyPrinter(indent=2)

package_path, package_init = os.path.split(__file__)
pickle_path = os.path.join(package_path, 'SFM_STARTER')
DATADIR = os.path.join(package_path, 'SFM_STARTER')
PARAMS = os.path.join(package_path, 'results/params.json')
MODELDIR = os.path.join(package_path, 'results/model')

# Predict
with Path(PARAMS).open() as f:
    params = json.load(f)
params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
params['glove'] = str(Path(DATADIR, 'glove.npz'))
estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)

def get_prediction(line):
    predict_inpf = functools.partial(predict_input_fn, line)
    for pred in estimator.predict(predict_inpf):
        pred_tags = pred['tags']
        break
    return pred_tags

def predict_input_fn(line):
    # Words
    words = [w.encode() for w in line.strip().split()]
    nwords = len(words)

    # Chars
    chars = [[c.encode() for c in w] for w in line.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

    # Wrapping in Tensors
    words = tf.constant([words], dtype=tf.string)
    nwords = tf.constant([nwords], dtype=tf.int32)
    chars = tf.constant([chars], dtype=tf.string)
    nchars = tf.constant([lengths], dtype=tf.int32)

    return ((words, nwords), (chars, nchars)), None

def build_pred_dict(test_sentences):
    sentence_pred_tags = {}
    test_count = 1
    for sentence in test_sentences:
        print("--- Progress: ", test_count, "/", len(test_sentences))
        if sentence not in sentence_pred_tags:
            sentence_pred_tags[sentence] = {}

        pred_tags = get_prediction(sentence)
        cur_idx = 0
        entity_start = 0
        entity_end = 0
        prev_tag = None
        words = sentence.strip().split()

        for i in range(len(words)):
            word = words[i]
            tag = pred_tags[i].decode()

            if tag[0] != 'O':
                corrected_start, corrected_end = correct_position(sentence, cur_idx, word)

                if tag[0] == 'B' or \
                    (prev_tag is not None and prev_tag == 'O'):
                    entity_start = corrected_start
                entity_end = corrected_end

                if i + 1 >= len(words) or pred_tags[i + 1].decode()[0] != 'I':
                    sentence_pred_tags[sentence][(entity_start, entity_end)] = tag[2:]

            cur_idx += len(word) + 1
            prev_tag = tag
        test_count += 1

    return sentence_pred_tags

if __name__ == '__main__':
    with open(os.path.join(pickle_path, "dataset_sentences.pickle"),"rb") as pickle_file:
        dataset_sentences = pickle.load(pickle_file)
    with open(os.path.join(pickle_path, "dataset_labels.pickle"),"rb") as pickle_file:
        dataset_labels = pickle.load(pickle_file)

    with open(os.path.join(DATADIR, "test.words.txt")) as test_file:
        test_sentences = test_file.readlines()
    test_sentences = [sentence.strip() for sentence in test_sentences]

    sentence_pred_tags = build_pred_dict(test_sentences)

    with open("sentence_pred_tags.pickle","wb") as pickle_out:
        pickle.dump(sentence_pred_tags, pickle_out)
