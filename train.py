import tensorflow as tf
import random
from model import *
from input_wikiqa import *
import yaml
import codecs
import os
import numpy as np
import math
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

"""Yield successive n-sized chunks from l."""
def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def change_word_to_id(vocab, sentences, max_sequence_length, not_exist_index):
    current_batch_x_id=[]
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
            #zero padding
        if len(sentence_word_ids) <= max_sequence_length:
            sentence_word_ids = np.concatenate((sentence_word_ids, np.full((max_sequence_length-len(sentence_word_ids)), not_exist_index)), axis=0)
        # print np.shape(sentence_word_ids)
        current_batch_x_id.append(sentence_word_ids)
    return current_batch_x_id, length


def glove_embeddings(glove_dir, dim):
    """
    this function returns glove word vectors in a dict of the form {'word' : <vector>}
    parameters : glove_dir is the directory location of the glove file for word embeddings/vectos
    """
    vocab = []
    embeddings = []
    f = codecs.open(os.path.join(glove_dir, 'glove.6B.' + str(dim) + 'd.txt'), encoding='utf-8')
    for line in f:
        values = line.strip().split(' ')
        vocab.append(values[0])
        embeddings.append(values[1:])
    print 'loaded glove'
    f.close()

    return vocab, embeddings
        
train_data = load_data('WikiQA-train.tsv')
random.shuffle(train_data)
test_data = load_data('WikiQA-test.tsv')

# ==========
#   MODEL
# ==========

#loading parameters from config.yml
with open('config.yml') as config_file:
    config = yaml.load(config_file)
glove_dir = config['glove_dir']
embedding_size = config['embedding_size']
learning_rate_file = config['learning_rate']
n_hidden = config['n_hidden']
margin = config['margin']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
max_sequence_length = config['max_sequence_length']
config_file.close()

#set tf flags
tf.flags.DEFINE_integer("batch_size", batch_size, "Batch Size (default: 16)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("report_error_freq", 10, "How often to log error in the terminal")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()    

#session variable
sess = tf.Session()

#load glove embeddings
vocab, embeddings = glove_embeddings(glove_dir, embedding_size)
vocab.append('not_exist')
not_exist_index = len(vocab) - 1
embeddings.append(np.zeros(embedding_size))

vocab_size = len(vocab)
emb = np.asarray(embeddings)

#populate tf embeddings index
#W is the tensorflow variable that would hold the word vectors according to the id's which we can search 
#using embedding_lookup function
W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=True, name="W")
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
embedding_init = W.assign(embedding_placeholder)
sess.run(embedding_init, feed_dict={embedding_placeholder: emb})

# --------------------------------------------------------------
#                   ------- Inputs ------------
# --------------------------------------------------------------

#input for question
x_q = tf.placeholder(tf.int32, [None, max_sequence_length], name='input_question')
q_seq_length = tf.placeholder(tf.int32, [None], name = 'question_seq_length')

#input for answer - correct
x_a_correct = tf.placeholder(tf.int32, [None, max_sequence_length], name='input_question')
a_correct_seq_length = tf.placeholder(tf.int32, [None], name = 'question_seq_length')

#input for answer - wrong
x_a_wrong = tf.placeholder(tf.int32, [None, max_sequence_length], name='input_question')
a_wrong_seq_length = tf.placeholder(tf.int32, [None], name = 'question_seq_length')

# --------------------------------------------------------------
#              ------- Loss - Cosine Loss ------------
# --------------------------------------------------------------

#looking up embeddings for words in sentences
input_sentence_vec_q = tf.nn.embedding_lookup(W, x_q)
input_sentence_vec_a_right = tf.nn.embedding_lookup(W, x_a_correct)
input_sentence_vec_a_wrong = tf.nn.embedding_lookup(W, x_a_wrong)

#margin
m = tf.placeholder(tf.float32) #read from config file

#cos distances
dis_right,output_mean_a_right, output_mean_q_right = model_qa(input_sentence_vec_a_right, a_correct_seq_length, input_sentence_vec_q, q_seq_length, max_sequence_length,128, False)
dis_wrong,_,_ = model_qa(input_sentence_vec_a_wrong, a_wrong_seq_length, input_sentence_vec_q, q_seq_length,max_sequence_length, 128, True)

# loss calculation
loss = tf.maximum(tf.constant(0, dtype=tf.float32), m - tf.subtract(1.0,dis_right) + tf.subtract(1.0,dis_wrong))
# loss = tf.reduce_mean(losses)

#train
learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#initialize uninitialized variables
uninitialized_vars = []
for var in tf.global_variables():
    try:
        sess.run(var)
    except tf.errors.FailedPreconditionError:
        uninitialized_vars.append(var)
init_new_vars_op = tf.variables_initializer(uninitialized_vars)
sess.run(init_new_vars_op)

# --------------------------------------------------------------
#                   ------- Training ------------
# --------------------------------------------------------------

print 'MARGIN - ', margin
for x in range(1, FLAGS.num_epochs):

    print '---------- EPOCH - ', x, ' ----------'

    batch_x = chunks(train_data, FLAGS.batch_size)
    for i in range(0, int(math.ceil(len(train_data)/FLAGS.batch_size)-1)): #int(math.ceil(len(y_train)/FLAGS.batch_size)-1)
        try:
            current_batch_x = next(batch_x)
        except StopIteration:
            break

        x_answers_right = []
        x_answers_wrong = []
        x_questions = []
        for x in current_batch_x:
            x_questions.append(x[0])
            x_answers_right.append(x[1][0][0])
            index_wrong_ans = math.floor(random.random() * len(x[2]))
            x_answers_wrong.append(x[2][int(index_wrong_ans)][0])



        #get word id list for each word in each sentence of the batch_x
        x_answers_right_id, seq_right = change_word_to_id(vocab, x_answers_right, max_sequence_length, not_exist_index)
        x_answers_wrong_id, seq_wrong = change_word_to_id(vocab, x_answers_wrong, max_sequence_length, not_exist_index)
        x_questions_id, seq_question = change_word_to_id(vocab, x_questions, max_sequence_length, not_exist_index)
        

        _, loss_val = sess.run([train_step, loss], feed_dict={x_q:x_questions_id, x_a_correct:x_answers_right_id, x_a_wrong : x_answers_wrong_id, 
            a_correct_seq_length : np.array(seq_right), a_wrong_seq_length : np.array(seq_wrong), q_seq_length:np.array(seq_question)
            , learning_rate : learning_rate_file, m : margin})
        # if i % FLAGS.report_error_freq == 0:
        print 'batch - ', i, ', loss - ', loss_val    

    # ----- TESTING ------
    #save model
    saver = tf.train.Saver()
    saver.save(sess, 'model/my-model-qAndA' + str(datetime.now()))

    #calculate accuracy
    y_pred = []
    print '---- TEST ----'
    for i, x in enumerate(test_data):
        total_ans = []
        for y in x[1]:
            total_ans.append(y[0])
        for y in x[2]:
            total_ans.append(y[0])
        question = np.array(x[0])
        question = np.repeat(question, [len(total_ans)], axis = 0)

        x_question, seq_question = change_word_to_id(vocab, question, max_sequence_length, not_exist_index)
        x_ans, seq_ans = change_word_to_id(vocab, total_ans, max_sequence_length, not_exist_index)
        sim, a, q = sess.run([dis_right, output_mean_a_right, output_mean_q_right], feed_dict={x_q:x_question, x_a_correct:x_ans, 
                        a_correct_seq_length : seq_ans, q_seq_length:seq_question})
        max_cos = -2
        index_max = -1
        for ind, z in enumerate(a):
            cos_sim = cosine_similarity(np.reshape(z, [1,-1]),np.reshape(q[ind], [1,-1]))[0,0]
            if cos_sim > max_cos:
                max_cos = cos_sim
                index_max = ind
        if index_max > len(x[1])-1:
            y_pred.append(0.0)
        else:
            y_pred.append(1.0)
        if i % 20 == 0:
            print 'question - ', i
    accuracy = np.mean(y_pred)
    print '\n\n test accuracy - ', accuracy

    #     cos_sim = []
    #     max_right = 0
    #     max_wrong = 0
    #     #right
    #     for z in x[1]:
    #         x_ans, seq_ans = change_word_to_id(vocab, z[0], max_sequence_length, not_exist_index)
    #         sim = sess.run(sim_right, feed_dict={x_q:x_question, x_a_correct:x_ans, 
    #                     a_correct_seq_length : np.array(seq_ans), q_seq_length:np.array(seq_question)})
    #         if sim > max_right:
    #             max_right = sim
    #     for y in x[2]:
    #         x_ans, seq_ans = change_word_to_id(vocab, y[0], max_sequence_length, not_exist_index)
    #         sim = sess.run(sim_right, feed_dict={x_q:x_question, x_a_correct:x_ans, 
    #                     a_correct_seq_length : np.array(seq_ans), q_seq_length:np.array(seq_question)})
    #         if sim > max_wrong:
    #             max_wrong = sim
    #     if max_right > max_wrong:
    #         y_pred.append(1.0)
    #     else:
    #         y_pred.append(0.0)
    # accuracy = np.mean(y_pred)
    # print '\n\n test accuracy - ', accuracy

    # f_voc = open(vocab_file, 'w') 
    # f_emb = open(embedding_file, 'w') 
    # for x in vocab:
    #     f_voc.write(x.encode('utf-8'))
    #     f_voc.write('\n').write(str(y) + ' ')
    #     f_emb.write('\n')
    # f_voc.close()
    # f_emb.close()

    saver = tf.train.Saver()
    saver.save(sess, 'model/my-model-qAndA')