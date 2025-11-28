"""Model definition for attentive bi-directional LSTM question answering."""

import os

os.environ.pop("TF_USE_LEGACY_KERAS", None)

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def _bidirectional_lstm(inputs, seq_length, n_hidden, dropout, scope_name,
                        reuse=False):
    """Create a bidirectional LSTM block with dropout."""
    with tf.variable_scope(scope_name, reuse=reuse):
        mask = tf.sequence_mask(seq_length, maxlen=tf.shape(inputs)[1])
        bi_lstm = tf.compat.v1.keras.layers.Bidirectional(
            tf.compat.v1.keras.layers.LSTM(
                n_hidden,
                return_sequences=True,
                name=scope_name + "_lstm"))
        outputs = bi_lstm(inputs, mask=mask)
        outputs = tf.nn.dropout(outputs, keep_prob=dropout)

    return outputs


def model_qa(x_answer, answer_seq_length, x_question, question_seq_length,
             max_document_length, dropout, n_hidden=128, reuse=False):
    """
    Build the attentive QA model.

    x_answer: input tensor for answer sequences.
    answer_seq_length: lengths of answers in the batch.
    x_question: input tensor for question sequences.
    question_seq_length: lengths of questions in the batch.
    n_hidden: number of hidden units in each LSTM cell.
    reuse: whether to reuse variables (shared encoder).
    """
    output_q = _bidirectional_lstm(
        x_question, question_seq_length, n_hidden, dropout, "q_lstm", reuse)
    output_mean_q = tf.reduce_mean(output_q, 1)

    output_a = _bidirectional_lstm(
        x_answer, answer_seq_length, n_hidden, dropout, "a_lstm", reuse)

    # h is the integer value for size of output of lstm (1 x 2h)
    h = 2 * n_hidden

    with tf.variable_scope("model", reuse=reuse):
        w_am = tf.get_variable(
            'w_am', shape=[h, h], initializer=tf.random_normal_initializer())
        w_qm = tf.get_variable(
            'w_qm', shape=[h, h], initializer=tf.random_normal_initializer())
        w_att = tf.get_variable(
            'w_att', shape=[1, h], initializer=tf.random_normal_initializer())

    ans_vec = tf.transpose(tf.reshape(output_a, (-1, h)))  # -> h x bt

    mul_ans = tf.matmul(w_am, ans_vec)

    mul_query = tf.matmul(w_qm, tf.transpose(output_mean_q))
    mul_query = tf.tile(mul_query, [1, max_document_length])

    att_m = tf.tanh(mul_query + mul_ans)

    s_aq = tf.matmul(w_att, att_m)
    s_aq = tf.squeeze(s_aq)
    s_aq = tf.reshape(s_aq, [-1, max_document_length])
    s_aq = tf.nn.softmax(s_aq)

    final_ans_vecs = tf.multiply(output_a, tf.expand_dims(s_aq, -1))
    output_mean_a = tf.reduce_mean(final_ans_vecs, 1)

    cos_distance = tf.losses.cosine_distance(
        tf.nn.l2_normalize(output_mean_q, 1),
        tf.nn.l2_normalize(output_mean_a, 1),
        dim=1,
    )

    return cos_distance, output_mean_a, output_mean_q
