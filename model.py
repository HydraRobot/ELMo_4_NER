'''
ELMo usage example  with pre-computed and cached context independent token representations
'''

import os
import h5py
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings

vocab_file = 'vocab.txt'

# Location of pretrained LM.  Here we use the test fixtures.
datadir = 'kaggle_data'
vocab_file = os.path.join(datadir, 'vocab.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

# Dump the embeddings to a file. Run this once for your dataset.
token_embedding_file = 'kaggle_elmo_token_SMALL.hdf5'
dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)

import tensorflow as tf

tf.reset_default_graph()

import data as dt

#split trainset and validationset
alen=len(dt.X)
val_ratio = 0.1
val_len = int(alen * val_ratio)

tokenized_sentences = dt.X[:-val_len] 
y = dt.y[:-val_len]

tokenized_sentences_val = dt.X[-val_len:] 
y_val = dt.y[-val_len:]

#Creat a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)

#input placeholder to the biLM
token_ids = tf.placeholder('int32', shape=(None, None))
y_label = tf.placeholder('float32', shape=(None, None, 17))

#Build the biLM graph
bilm = BidirectionalLanguageModel(options_file, weight_file, use_character_inputs=False, embedding_weight_file=token_embedding_file)

#Get ops to compute the LM embeddings
embeddings_op = bilm(token_ids)

#Get an op to compute ELMo(weighted average of the internal biLM layers)
elmo_input =  weight_layers('input', embeddings_op, l2_coef=0.0)

hidden_dim = 512
dropout=0.5
#Bidirectional layers
fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,1-dropout)
bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,1-dropout)

##shape(batch_num, length, hs_dim)
(outputs, (fw_st, bw_st))=tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, elmo_input['weighted_op'], dtype=tf.float32,  swap_memory=True)
##shape(batch_num, length, n_tags)
tag_output = tf.layers.dense(tf.concat(outputs, axis=2), dt.n_tags, activation=None)

idx_output=tf.argmax(tag_output, axis=2)

##dt.y shape (batch_num, length) discrete index
##1) to get loss: labels: one-hot logits: one-hot
#tf.nn.softmax_soft_cross_entropy_with_logits
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tag_output, labels=y_label, name=None))

#1 optimizer method 
optimizer = tf.train.AdagradOptimizer(learning_rate=0.005, initial_accumulator_value=0.1)

gvs = optimizer.compute_gradients(loss)

apply_ops=optimizer.apply_gradients(gvs)

tvars = tf.trainable_variables()

acc=tf.metrics.accuracy(labels=tf.argmax(y_label, axis=2), predictions=idx_output)

#output model
data_path = './NER_models'
model_save_name = 'NERModel'
final_model=os.path.join(data_path, model_save_name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())#for tf.metrics
    ids = batcher.batch_sentences(tokenized_sentences)
    ids_val = batcher.batch_sentences(tokenized_sentences_val)
    batch_size = ids.shape[0] // 128     #128 is the max 
    batch_size_val = ids_val.shape[0] // 128
    saver = tf.train.Saver()
    for step in range(1000):
        for i in range(batch_size + 1): 
            if (i < batch_size):
                s_index=i*128
                e_index=(i+1)*128
                ids_i = ids[s_index:e_index]
                y_i = y[s_index:e_index]
                elmo_input_ = sess.run(elmo_input['weighted_op'], feed_dict={token_ids:ids_i})
                loss_, _ = sess.run([loss,  apply_ops] , feed_dict={token_ids:ids_i, y_label: y_i})
                print("loss", loss_)
            else:
                s_index=i*128
                ids_i = ids[s_index:]
                y_i = y[s_index:]
                elmo_input_ = sess.run(elmo_input['weighted_op'], feed_dict={token_ids:ids_i})
                loss_, _ = sess.run([loss,  apply_ops] , feed_dict={token_ids:ids_i, y_label: y_i})
                print("loss", loss_)
        saver.save(sess, final_model)   
        for i in range(batch_size_val + 1):
            if (i < batch_size_val):
                s_index=i*128
                e_index=(i+1)*128
                ids_i = ids_val[s_index:e_index]
                y_i = y_val[s_index:e_index]
                loss_, acc_, = sess.run([loss, acc] , feed_dict={token_ids:ids_i, y_label:y_i})
                print('acc', acc_)
            else:
                s_index=i*128
                ids_i = ids_val[s_index:]
                y_i = y_val[s_index:]
                loss_, acc_, = sess.run([loss, acc] , feed_dict={token_ids:ids_i, y_label:y_i})
                print('acc', acc_)

