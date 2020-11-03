"""GloVe Embeddings + chars conv and max pooling + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tf_metrics import precision, recall, f1
from six.moves import reduce


DATADIR = 'SFM_STARTER'

# Logging
Path('results').mkdir(exist_ok=True)
tf.compat.v1.logging.set_verbosity(logging.INFO)
handlers = [
    # logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),               # (words, nwords)
               ([None, None], [None])),    # (chars, nchars)
              [None])                      # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read vocabs and inputs
    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(params['words'],
                    key_dtype=tf.string,
                    key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                    value_dtype=tf.int64,
                    value_index=tf.lookup.TextFileIndex.LINE_NUMBER),
        num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(params['chars'],
                    key_dtype=tf.string,
                    key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                    value_dtype=tf.int64,
                    value_index=tf.lookup.TextFileIndex.LINE_NUMBER),
        num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)
    variable = tf.compat.v1.get_variable(
        'chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(params=variable, ids=char_ids)
    char_embeddings = tf.keras.layers.Dropout(dropout)(char_embeddings, training=training)

    # Char 1d convolution
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filters'], params['kernel_size'])

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(params=variable, ids=word_ids)

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.keras.layers.Dropout(dropout)(embeddings, training=training)

    # LSTM
    t = tf.transpose(a=embeddings, perm=[1, 0, 2])  # Need time-major
    biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['lstm_size'], return_sequences=True))(t)
    # lstm_cell_fw = tf.keras.layers.LSTM(params['lstm_size'])
    # lstm_cell_bw = tf.keras.layers.LSTM(params['lstm_size'])
    # lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    # output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    # output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    # output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(a=biLSTM, perm=[1, 0, 2])
    output = tf.keras.layers.Dropout(dropout)(output, training=training)

    # CRF
    logits = tf.keras.layers.Dense(num_tags)(output)
    crf_params = tf.compat.v1.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tfa.text.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.lookup.StaticHashTable(
                    tf.lookup.TextFileInitializer(params['tags'],
                        key_dtype=tf.int64,
                        key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                        value_dtype=tf.string,
                        value_index=tf.lookup.TextFileIndex.WHOLE_LINE),
                    "D"
                    )
        pred_strings = reverse_vocab_tags.lookup(tf.cast(pred_ids, dtype=tf.int64))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(params['tags'],
                    key_dtype=tf.string,
                    key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                    value_dtype=tf.int64,
                    value_index=tf.lookup.TextFileIndex.LINE_NUMBER),
            num_oov_buckets=params['num_oov_buckets'])
        # vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tfa.text.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(input_tensor=-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.compat.v1.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.compat.v1.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.compat.v1.train.AdamOptimizer().minimize(
                loss, global_step=tf.compat.v1.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


def masked_conv1d_and_max(t, weights, filters, kernel_size):
    """Applies 1d convolution and a masked max-pooling
    __author__ = Guillaume Genthial

    Parameters
    ----------
    t : tf.Tensor
        A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
    weights : tf.Tensor of tf.bool
        A Tensor of shape [d1, d2, dn-1]
    filters : int
        number of filters
    kernel_size : int
        kernel size for the temporal convolution

    Returns
    -------
    tf.Tensor
        A tensor of shape [d1, d2, dn-1, filters]

    """
    # Get shape and parameters
    shape = tf.shape(input=t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.cast(weights, dtype=tf.float32)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(t)
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(input_tensor=t_conv, axis=-2, keepdims=True)
    t_max = tf.reduce_max(input_tensor=t_conv, axis=-2)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape)

    return t_max


if __name__ == '__main__':
    # Params
    params = {
        'dim_chars': 100,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 15000,
        'filters': 50,
        'kernel_size': 3,
        'lstm_size': 100,
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz'))
    }
    with Path('results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))

    def ftags(name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('valid'), ftags('valid'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.estimator.experimental.stop_if_no_decrease_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=60)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        Path('results/score').mkdir(parents=True, exist_ok=True)
        with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), (_, _)), tags = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'valid', 'test']:
        write_predictions(name)
