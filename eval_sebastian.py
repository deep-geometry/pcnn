""" Based on point net evaluation process
downladed from: https://github.com/charlesq34/pointnet
"""

import tensorflow as tf
import numpy as np
import argparse
import socket
import os
import json
from pyhocon import ConfigFactory
from provider import SebastianProvider

from pointcloud_conv_net import Network

import provider


BASE_DIR = '../../../../../'
pv = provider.ClassificationProvider(False)
pv.BASE_DIR = BASE_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 1]')
parser.add_argument('--model_path', default='epoch_250/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--traindir', default="", help="Directory of the train dataset")
parser.add_argument('--testdir', default="", help="Directory of the test dataset")
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--config', type=str, default='pointconv.conf',
                    help='Config to use [default: pointconv]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
conf = ConfigFactory.parse_file('{0}'.format(FLAGS.config))
NUM_POINT = conf.get_list('network.pool_sizes_sigma')[0][0]

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

HOSTNAME = socket.gethostname()

provider = SebastianProvider(traindir=None, testdir=FLAGS.testdir,
                             batch_size=BATCH_SIZE, points_per_patch=NUM_POINT)
TRAIN_FILES = provider.getTrainDataFiles()
TEST_FILES = provider.getTestDataFiles()


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate(num_votes):
    is_training = False

    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
        labels_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 3))
        is_training_pl = tf.placeholder(tf.bool, shape=())
        is_evaluate_pl = tf.placeholder(tf.bool, shape=())

        # Get model and loss
        # pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
        network = Network(conf.get_config('network'))
        pred = network.build_network(pointclouds_pl, is_training_pl,is_evaluate_pl)
        loss = network.cos_loss(pred, labels_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'is_evaluate_pl': is_evaluate_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)


def eval_one_epoch(sess, ops, num_votes=1):

    is_training = False
    is_evaluate = True

    loss_sum = 0
    out_results = []

    out_data = []

    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '----')
        current_data, current_label = provider.loadDataFile(is_train=False)

        current_data = current_data[:, :, :]
        current_label = np.squeeze(current_label)
        print(current_data.shape)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)
        print(num_batches)

        for batch_idx in range(num_batches):

            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            print('start_idx : {0}'.format(start_idx))
            cur_batch_size = current_data[start_idx:end_idx].shape[0]

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx, :],
                         ops['is_training_pl']: is_training,
                         ops['is_evaluate_pl']: is_evaluate}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                          feed_dict=feed_dict)

            for i in range(0, BATCH_SIZE):
                file_name = provider.test_file_list[start_idx+i]
                center_idx = provider.test_ctrs[start_idx+i]
                expected_normal = provider.test_normals[start_idx+i].astype(float)
                expected_normal /= np.linalg.norm(expected_normal)
                predicted_normal = np.array(pred_val[i]).astype(float)
                predicted_normal /= np.linalg.norm(predicted_normal)
                one_minus_cos_loss = float(1.0 - np.abs(np.dot(expected_normal, predicted_normal))) ** 2
                print(predicted_normal, expected_normal, one_minus_cos_loss)
                # infodict = {
                #     'filename': filenames[i],
                #     'expected_normal': target_i,
                #     'predicted_normal': predicted_i,
                #     'one_minus_cos_loss': float(losses[i].data.cpu().numpy()),
                #     'ctr_idx': point_indexes[i],
                # }
                #

                save_dict = {
                    'file_name': file_name,
                    'ctr_idx': int(center_idx),
                    'expected_normal': list(expected_normal),
                    'predicted_normal': list(predicted_normal),
                    'one_minus_cos_loss': one_minus_cos_loss
                }

                out_data.append(save_dict)

    with open("res.json", "w") as json_f:
        json_f.write(json.dumps(out_data))


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=10)
    LOG_FOUT.close()
