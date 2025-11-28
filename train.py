"""Training script for the attentive LSTM question answering model."""

from __future__ import print_function

import codecs
import math
import os
import random
import sys
from datetime import datetime

os.environ.pop("TF_USE_LEGACY_KERAS", None)

import nltk
import numpy as np
import tensorflow.compat.v1 as tf
import yaml
from sklearn.metrics.pairwise import cosine_similarity

from input_wikiqa import load_data
from model import model_qa

tf.disable_v2_behavior()

# Ensure required NLTK data is available
nltk.download('punkt', quiet=True)


def maybe_redirect_stdout(argv):
    """Optionally redirect stdout to a file when running as a script."""
    if len(argv) > 1 and argv[1] == 'y':
        print('outputting in the file now')
        sys.stdout = open('output.txt', 'w')


def chunks(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def change_word_to_id(vocab, sentences, max_sequence_length, not_exist_index):
    """Convert each sentence into a fixed-length list of word indices."""
    current_batch_x_id = []
    length = []
    for sentence in sentences:
        count = 0
        sentence_word_ids = []
        for word in nltk.word_tokenize(sentence):
            if count < max_sequence_length:
                if word in vocab:
                    index = vocab.index(word)
                else:
                    index = not_exist_index
                sentence_word_ids.append(index)
                count += 1
            else:
                break
        length.append(len(sentence_word_ids))
        if len(sentence_word_ids) <= max_sequence_length:
            sentence_word_ids = np.concatenate(
                (sentence_word_ids,
                 np.full((max_sequence_length - len(sentence_word_ids)),
                         not_exist_index)), axis=0)
        current_batch_x_id.append(sentence_word_ids)
    return current_batch_x_id, length


def glove_embeddings(glove_file, embedding_size):
    """
    Return GloVe word vectors in list form.

    Parameters
    ----------
    glove_file : full path to the glove file for word embeddings/vectors.
    """
    vocab = []
    embeddings = []
    with codecs.open(glove_file, encoding='utf-8') as glove_fp:
        for line in glove_fp:
            values = line.strip().split(' ')
            vocab.append(values[0])
            vector = values[1:]
            if len(vector) < embedding_size:
                continue  # skip malformed lines
            if len(vector) > embedding_size:
                vector = vector[:embedding_size]
            embeddings.append([float(x) for x in vector])
    print('loaded glove')
    return vocab, embeddings


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def initialize_embeddings(sess, vocab, embeddings, embedding_size):
    """Initialize the TensorFlow embedding matrix."""
    vocab_size = len(vocab)
    emb = np.asarray(embeddings)
    embedding_matrix = tf.Variable(
        tf.constant(0.0, shape=[vocab_size, embedding_size]),
        trainable=True,
        name="W")
    embedding_placeholder = tf.placeholder(tf.float32,
                                           [vocab_size, embedding_size])
    embedding_init = embedding_matrix.assign(embedding_placeholder)
    sess.run(embedding_init, feed_dict={embedding_placeholder: emb})
    return embedding_matrix


def ensure_model_dir_exists():
    """Create the model directory if it does not already exist."""
    if not os.path.exists('model'):
        os.makedirs('model')


def initialize_uninitialized_variables(sess):
    """Initialize only the variables that are not yet initialized."""
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    if uninitialized_vars:
        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)


def main(argv=None):
    argv = argv or sys.argv
    maybe_redirect_stdout(argv)

    train_data = load_data('WikiQA-train.tsv')
    random.shuffle(train_data)
    test_data = load_data('WikiQA-test.tsv')

    config = load_config('config.yml')
    glove_file = config['glove_file']
    embedding_size = config['embedding_size']
    learning_rate_value = config['learning_rate']
    n_hidden = config['n_hidden']
    margin = config['margin']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    max_sequence_length = config['max_sequence_length']
    dropout_value = config['dropout']
    vocab_file = config['vocab_file']
    embedding_file = config['embedding_file']

    tf.flags.DEFINE_integer("batch_size", batch_size, "Batch Size")
    tf.flags.DEFINE_integer("num_epochs", num_epochs,
                            "Number of training epochs")
    tf.flags.DEFINE_integer("report_error_freq", 10,
                            "How often to log error in the terminal")
    flags = tf.flags.FLAGS
    try:
        flags._parse_flags()
    except AttributeError:
        flags(argv)

    with tf.Session() as sess:
        vocab, embeddings = glove_embeddings(glove_file, embedding_size)
        vocab.append('not_exist')
        not_exist_index = len(vocab) - 1
        embeddings.append(np.zeros(embedding_size))

        embedding_matrix = initialize_embeddings(
            sess, vocab, embeddings, embedding_size)

        x_q = tf.placeholder(tf.int32, [None, max_sequence_length],
                             name='input_question')
        q_seq_length = tf.placeholder(tf.int32, [None],
                                      name='question_seq_length')

        x_a_correct = tf.placeholder(tf.int32, [None, max_sequence_length],
                                     name='input_answer_correct')
        a_correct_seq_length = tf.placeholder(tf.int32, [None],
                                              name='answer_correct_seq_length')

        x_a_wrong = tf.placeholder(tf.int32, [None, max_sequence_length],
                                   name='input_answer_wrong')
        a_wrong_seq_length = tf.placeholder(tf.int32, [None],
                                            name='answer_wrong_seq_length')

        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        input_sentence_vec_q = tf.nn.embedding_lookup(embedding_matrix, x_q)
        input_sentence_vec_a_right = tf.nn.embedding_lookup(
            embedding_matrix, x_a_correct)
        input_sentence_vec_a_wrong = tf.nn.embedding_lookup(
            embedding_matrix, x_a_wrong)

        margin_placeholder = tf.placeholder(tf.float32, name='margin')

        dis_right, output_mean_a_right, output_mean_q_right = model_qa(
            input_sentence_vec_a_right, a_correct_seq_length,
            input_sentence_vec_q, q_seq_length, max_sequence_length,
            dropout_prob, n_hidden, False)
        dis_wrong, _, _ = model_qa(
            input_sentence_vec_a_wrong, a_wrong_seq_length,
            input_sentence_vec_q, q_seq_length, max_sequence_length,
            dropout_prob, n_hidden, True)

        loss = tf.maximum(
            tf.constant(0, dtype=tf.float32),
            margin_placeholder - tf.subtract(1.0, dis_right) +
            tf.subtract(1.0, dis_wrong))

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        initialize_uninitialized_variables(sess)
        ensure_model_dir_exists()

        print('MARGIN - ', margin)
        for epoch in range(1, flags.num_epochs):
            print('---------- EPOCH - ', epoch, ' ----------')

            batch_x = chunks(train_data, flags.batch_size)
            for i in range(
                    0,
                    int(math.ceil(len(train_data) / flags.batch_size) - 1)):
                try:
                    current_batch_x = next(batch_x)
                except StopIteration:
                    break

                x_answers_right = []
                x_answers_wrong = []
                x_questions = []
                for item in current_batch_x:
                    x_questions.append(item[0])
                    x_answers_right.append(item[1][0][0])
                    index_wrong_ans = math.floor(
                        random.random() * len(item[2]))
                    x_answers_wrong.append(
                        item[2][int(index_wrong_ans)][0])

                x_answers_right_id, seq_right = change_word_to_id(
                    vocab, x_answers_right, max_sequence_length,
                    not_exist_index)
                x_answers_wrong_id, seq_wrong = change_word_to_id(
                    vocab, x_answers_wrong, max_sequence_length,
                    not_exist_index)
                x_questions_id, seq_question = change_word_to_id(
                    vocab, x_questions, max_sequence_length, not_exist_index)

                _, loss_val = sess.run(
                    [train_step, loss],
                    feed_dict={
                        x_q: x_questions_id,
                        x_a_correct: x_answers_right_id,
                        x_a_wrong: x_answers_wrong_id,
                        a_correct_seq_length: np.array(seq_right),
                        a_wrong_seq_length: np.array(seq_wrong),
                        q_seq_length: np.array(seq_question),
                        learning_rate: learning_rate_value,
                        margin_placeholder: margin,
                        dropout_prob: dropout_value
                    })
                print('batch - ', i, ', loss - ', loss_val)

            saver = tf.train.Saver()
            saver.save(sess, 'model/my-model-qAndA' + str(datetime.now()))

            y_pred = []
            print('---- TEST ----')
            for i, item in enumerate(test_data):
                total_ans = []
                for pos in item[1]:
                    total_ans.append(pos[0])
                for neg in item[2]:
                    total_ans.append(neg[0])
                question = np.array(item[0])
                question = np.repeat(question, [len(total_ans)], axis=0)

                x_question, seq_question = change_word_to_id(
                    vocab, question, max_sequence_length, not_exist_index)
                x_ans, seq_ans = change_word_to_id(
                    vocab, total_ans, max_sequence_length, not_exist_index)
                sim, a, q = sess.run(
                    [dis_right, output_mean_a_right, output_mean_q_right],
                    feed_dict={
                        x_q: x_question,
                        x_a_correct: x_ans,
                        a_correct_seq_length: seq_ans,
                        q_seq_length: seq_question,
                        dropout_prob: np.float32(1.0)
                    })
                max_cos = -2
                index_max = -1
                for ind, z in enumerate(a):
                    cos_sim = cosine_similarity(
                        np.reshape(z, [1, -1]),
                        np.reshape(q[ind], [1, -1]))[0, 0]
                    if cos_sim > max_cos:
                        max_cos = cos_sim
                        index_max = ind
                if index_max > len(item[1]) - 1:
                    y_pred.append(0.0)
                else:
                    y_pred.append(1.0)
                if i % 20 == 0:
                    print('question - ', i)
                    print(y_pred)
            accuracy = np.mean(y_pred)
            print('\n\n test accuracy - ', accuracy)

            with open(vocab_file, 'w') as f_voc:
                for vocab_item in vocab:
                    f_voc.write(vocab_item.encode('utf-8'))
                    f_voc.write('\n')
            with open(embedding_file, 'w') as f_emb:
                for embedding in embeddings:
                    for value in embedding:
                        f_emb.write(str(value) + ' ')
                    f_emb.write('\n')

            saver = tf.train.Saver()
            saver.save(sess, 'model/my-model-qAndA')


if __name__ == '__main__':
    main()
