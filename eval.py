# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import data_helpers
from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("eval_dir", "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT", "Path of evaluation data")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def eval():
    with tf.device('/cpu:0'):
        x_text, pos1, pos2, y = data_helpers.load_data_and_labels(FLAGS.eval_dir)

    # Map data into vocabulary
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    text_vec = np.array(list(text_vocab_processor.transform(x_text)))

    # Map data into position
    position_path = os.path.join(FLAGS.checkpoint_dir, "..", "position_vocab")
    position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)
    pos1_vec = np.array(list(position_vocab_processor.transform(pos1)))
    pos2_vec = np.array(list(position_vocab_processor.transform(pos2)))

    x_eval = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])
    y_eval = np.argmax(y, axis=1)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()