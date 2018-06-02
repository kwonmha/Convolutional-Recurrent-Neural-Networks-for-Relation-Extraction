# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import datetime
import time
from crnn import CRNN
from sklearn.metrics import f1_score
import data_helpers
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_dir", "SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT", "Path of train data")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 100, "Max sentence length in train(98)/test(70) data (Default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_string("word2vec", "GoogleNews-vectors-negative300.bin", "Word2vec file with pre-trained embeddings") #
tf.flags.DEFINE_integer("text_embedding_dim", 300, "Dimensionality of character embedding (Default: 300)")
# tf.flags.DEFINE_integer("position_embedding_dim", 200, "Dimensionality of position embedding (Default: 100)")
# tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (Default: 2,3,4,5)")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (Default: 128)")
tf.flags.DEFINE_string("layers", "100", "Size of rnn output, no (Default: 100")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_string("pooling_type", "max", "pooling method, max or att (Default: max)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")
tf.flags.DEFINE_integer("f1", 2, "f1 filter size (Default : 2)")
tf.flags.DEFINE_integer("f2", 5, "f2 filter size (Default : 5)")
tf.flags.DEFINE_integer("n_channels", 100, "the number of channels-output vector size, nc(Default : 100")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (Default: 100)") # 100 epochs - 11290 steps
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{} = {}".format(attr.upper(), value))
print("")


def train():
	with tf.device('/cpu:0'):
		# x_text, pos1, pos2, y = data_helpers.load_data_and_labels(FLAGS.train_dir)
		x_text, y = data_helpers.load_data_and_labels(FLAGS.train_dir)

	# Build vocabulary
	# Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
	# ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
	# =>
	# [27 39 40 41 42  1 43  0  0 ... 0]
	# dimension = FLAGS.max_sentence_length
	text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
	text_vec = np.array(list(text_vocab_processor.fit_transform(x_text)))
	print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))


	# Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
	# [95 96 97 98 99 100 101 999 999 999 ... 999]
	# =>
	# [11 12 13 14 15  16  21  17  17  17 ...  17]
	# dimension = MAX_SENTENCE_LENGTH
	# pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
	# pos_vocab_processor.fit(pos1 + pos2)
	# pos1_vec = np.array(list(pos_vocab_processor.transform(pos1)))
	# pos2_vec = np.array(list(pos_vocab_processor.transform(pos2)))
	# print("Position Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))

	# x = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])
	x = np.array([list(i) for i in text_vec])

	print("x = {0}".format(x.shape))
	print("y = {0}".format(y.shape))

	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(y)))
	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]

	# Split train/test set
	# TODO: This is very crude, should use cross-validation
	dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
	x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
	# x_dev = np.array(x_dev).transpose((1, 0, 2))
	y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
	print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		model = CRNN(layers=FLAGS.layers, max_length=FLAGS.max_sentence_length, n_classes=y.shape[1], pooling_type=FLAGS.pooling_type,
					 vocab_size=len(text_vocab_processor.vocabulary_), embedding_size=FLAGS.text_embedding_dim,
					 f1=FLAGS.f1, f2=FLAGS.f2, n_channels=FLAGS.n_channels)

		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		# Write vocabulary
		text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))
		# pos_vocab_processor.save(os.path.join(out_dir, "position_vocab"))

		sess.run(tf.global_variables_initializer())

		# Pre-trained word2vec
		if FLAGS.word2vec:
			# initial matrix with random uniform
			initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), FLAGS.text_embedding_dim))
			# load any vectors from the word2vec
			print("Load word2vec file {0}".format(FLAGS.word2vec))
			with open(FLAGS.word2vec, "rb") as f:
				header = f.readline()
				vocab_size, layer1_size = map(int, header.split())
				binary_len = np.dtype('float32').itemsize * layer1_size
				for line in range(vocab_size):
					word = []
					while True:
						ch = f.read(1).decode('latin-1')
						if ch == ' ':
							word = ''.join(word)
							break
						if ch != '\n':
							word.append(ch)
					idx = text_vocab_processor.vocabulary_.get(word)
					if idx != 0:
						initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
					else:
						f.read(binary_len)
			sess.run(model.W_emb.assign(initW))
			print("Success to load pre-trained word2vec model!\n")

		batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

		max_f1 = -1

		for step, batch in enumerate(batches):
			x_batch, y_batch = zip(*batch)

			feed_dict = {model.input_text: x_batch, model.dropout_keep_prob: FLAGS.dropout_keep_prob, model.labels: y_batch}
			# max_pooling, convs = sess.run([model.max_pooing, model.conv], feed_dict=feed_dict)
			_, loss, accuracy = sess.run([model.train, model.cost, model.accuracy], feed_dict=feed_dict)

			# Training log display
			if step % FLAGS.display_every == 0:
				print("step {}:, loss {}, acc {}".format(step, loss, accuracy))

			# Evaluation
			if step % FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				feed_dict = {
					model.input_text: x_dev,
					model.labels: y_dev,
					model.dropout_keep_prob: 1.0
				}
				loss, accuracy, predictions = sess.run(
					[model.cost, model.accuracy, model.predictions], feed_dict)

				f1 = f1_score(np.argmax(y_dev, axis=1), predictions, average="macro")
				print("step {}:, loss {}, acc {}, f1 {}\n".format(step, loss, accuracy, f1))

				# Model checkpoint
				if f1 > max_f1 * 0.99:
					path = saver.save(sess, checkpoint_prefix, global_step=step)
					print("Saved model checkpoint to {}\n".format(path))
					max_f1 = f1

def main(_):
	train()


if __name__ == "__main__":
	tf.app.run()
