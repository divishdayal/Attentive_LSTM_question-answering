import tensorflow as tf

def model_qa(x_answer, answer_seq_length, x_question, question_seq_length, max_document_length, dropout, n_hidden=128, reuse=False):
    '''
    x_answer : input for answer
    answer_seq_length : sequence length for answer
    x_question : input for question
    question_seq_length : sequence length for question
    n_hidden : number of hidden state(memory cell size) in the lstm fw and bw cells
    reuse : to reuse variables or not(shared variables)

    '''

    # --------------------------------------------------------------
    #                   ------- Question ------------
    # --------------------------------------------------------------

    # Define a lstm cell with tensorflow
    #n_hidden is memory cell size
    with tf.variable_scope("q_lstm", reuse=reuse) as scope:
        lstm_cell_q_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        lstm_cell_q_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden)

        #dropout
        lstm_cell_q_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_q_fw, output_keep_prob = dropout)
        lstm_cell_q_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_q_bw, output_keep_prob = dropout)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs_q, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_q_fw,cell_bw=lstm_cell_q_bw, inputs=x_question, 
                                    sequence_length=question_seq_length,
                                    dtype=tf.float32,
                                    )

    #forward backward concatenation as output
    output_fw_q, output_bw_q = outputs_q

    #concatenating fw/bw passes
    output_q = tf.concat([output_fw_q, output_bw_q], axis=-1)

    output_mean_q = tf.reduce_mean(output_q, 1)



    # --------------------------------------------------------------
    #                   ------- Answer ------------
    # --------------------------------------------------------------
    
    with tf.variable_scope("a_lstm", reuse=reuse) as scope:
        lstm_cell_a_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        lstm_cell_a_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden)

        #dropout
        lstm_cell_a_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_a_fw, output_keep_prob = dropout)
        lstm_cell_a_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_a_bw, output_keep_prob = dropout)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs_a, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_a_fw,cell_bw=lstm_cell_a_bw, inputs=x_answer, 
                                    sequence_length=answer_seq_length,
                                    dtype=tf.float32,
                                    )

    #forward backward concatenation as output
    output_fw_a, output_bw_a = outputs_a

    #concatenating fw/bw passes
    output_a = tf.concat([output_fw_a, output_bw_a], axis=-1)

    # --------------------------------------------------------------
    #            ------- Attention and Loss------------
    # --------------------------------------------------------------

    #h is the integer value for size of output of lstm ( 1 x 2h )
    h = 2*lstm_cell_a_fw.output_size

    #weights for attention
    with tf.variable_scope("model", reuse=reuse):
        w_am = tf.get_variable('w_am', shape=[h, h], initializer=tf.random_normal_initializer())
        w_qm = tf.get_variable('w_qm', shape=[h, h], initializer=tf.random_normal_initializer())
        w_att = tf.get_variable('w_att', shape=[1, h], initializer=tf.random_normal_initializer())


    #transposed vecs for ans and ques
    ans_vec =  tf.transpose( tf.reshape(output_a, (-1, h)) ) # -> h x bt where b is batch size and t is sequence length


    mul_ans=tf.matmul(w_am, ans_vec)
    #mul_ans => hxbt

    mul_query=tf.matmul(w_qm, tf.transpose(output_mean_q) )  # -> [h,h] x [b, h]t where t means transpose 
    #mul_query => hxb

    #repeat mul_query
    mul_query = tf.tile(mul_query, [1,max_document_length]) #-> hxbt


    att_m = mul_query + mul_ans
    # att_m => hxbt

    #activation
    att_m = tf.tanh(att_m) # -> hxbt

    s_aq = tf.matmul(w_att, att_m) # -> 1xbt
    s_aq = tf.squeeze(s_aq) # -> bt
    s_aq = tf.reshape(s_aq, [-1, max_document_length]) # bxt

    #applying softmax
    s_aq = tf.nn.softmax(s_aq) # -> bxt

    #answer vectors updated with attention
    final_ans_vecs = tf.multiply( output_a, tf.expand_dims(s_aq, -1) ) # -> txh
    output_mean_a = tf.reduce_mean(final_ans_vecs, 1)

    #cosine distance
    cos_distance = tf.losses.cosine_distance(tf.nn.l2_normalize(output_mean_q, 1), tf.nn.l2_normalize(output_mean_a, 1), dim=1)

    return cos_distance, output_mean_a, output_mean_q