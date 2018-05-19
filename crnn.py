# -*- coding: utf-8 -*-
import tensorflow as tf


class CRNN():
	def __init__(self, layers, max_length, n_classes, vocab_size, embedding_size, f1, f2, n_channels):

		self.input_text = tf.placeholder(tf.int32, shape=[None, max_length], name="input_text")
		self.labels = tf.placeholder(tf.int32, shape=[None, n_classes])
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

		l2_loss = tf.constant(0.0)

		self.W_emb = tf.Variable(tf.random_normal([vocab_size, embedding_size]))
		self.text_embedded = tf.nn.embedding_lookup(self.W_emb, self.input_text)

		self.length = self.get_length(self.text_embedded)

		layers = list(map(int, layers.split('-')))
		rnn_cell = tf.nn.rnn_cell.LSTMCell
		cells = [rnn_cell(h, activation=tf.tanh, state_is_tuple=True) for h in layers]
		multi_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
		self.rnn_outputs, _states = tf.nn.bidirectional_dynamic_rnn(multi_cells, multi_cells, self.text_embedded, sequence_length=self.length, dtype=tf.float32)
		self.rnn_outputs = tf.concat(self.rnn_outputs, 2)
		self.rnn_outputs = tf.expand_dims(self.rnn_outputs, -1)
		#(64, 100, 200, 1)
		self.first_pooling = tf.nn.max_pool(self.rnn_outputs, ksize=[1, f1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
		#[batch, in_height, in_width, in_channels] == 64 99 200 1

		# [filter_height, filter_width, in_channels, out_channels]
		W_conv = tf.Variable(tf.truncated_normal([f2, layers[0]*2, 1, n_channels]))
		b_conv = tf.Variable(tf.truncated_normal([n_channels]))
		self.conv = tf.nn.conv2d(self.first_pooling, W_conv, strides=[1, 1, 1, 1], padding='VALID')
		self.conv = tf.nn.relu(tf.nn.bias_add(self.conv, b_conv))
		#(64, 95, 1, 100)
		self.max_pooing = tf.nn.max_pool(self.conv, ksize=[1, max_length-f1-f2+2, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
		self.max_pooing = tf.nn.dropout(self.max_pooing, keep_prob=self.dropout_keep_prob)
		self.max_pooing = tf.squeeze(self.max_pooing, axis=[1, 2])

		self.logits = tf.layers.dense(self.max_pooing, units=n_classes)

		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
		self.optimizer = tf.train.AdamOptimizer()
		self.train = self.optimizer.minimize(self.cost)

		self.predictions = tf.argmax(self.logits, 1, name="predictions")
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 1)), tf.float32))

	@staticmethod
	def get_length(sequence):
		used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
		length = tf.reduce_sum(used, 1)
		length = tf.cast(length, tf.int32)
		return length

