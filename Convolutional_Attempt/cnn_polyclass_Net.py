import tensorflow as tf
import datetime
import pickle
import numpy as np
import os
import tflearn
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


meta_information = pickle.load(open("meta.p", "rb"))

n_data_points = 500 * 2		# the 2 represents the x and the f(x)
n_output_layer = meta_information['max_degree']
LR = 1e-3

MODEL_NAME = 'degree_classifier-{}-{}.model'.format(LR, '6conv-basic')


'''
This method is used to turn our degree into a one hot
array to be put into our model
'''
def turn_degree_into_one_hot(q):
    z = np.zeros(n_output_layer)
    z[q] = 1
    return z



def neural_network_model():

    convnet = input_data(shape=[None, n_data_points, 1], name='input')

    q = 2

    convnet = conv_1d(convnet, 32, 2, activation='relu')
    convnet = max_pool_1d(convnet, q)

    convnet = conv_1d(convnet, 64, 2, activation='relu')
    convnet = max_pool_1d(convnet, q)

    convnet = conv_1d(convnet, 32, 2, activation='relu')
    convnet = max_pool_1d(convnet, q)

    convnet = conv_1d(convnet, 64, 2, activation='relu')
    convnet = max_pool_1d(convnet, q)

    convnet = conv_1d(convnet, 32, 2, activation='relu')
    convnet = max_pool_1d(convnet, q)

    convnet = conv_1d(convnet, 64, 2, activation='relu')
    convnet = max_pool_1d(convnet, q)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_output_layer, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!!!')

    return model
    


def train_convolutional_network(model):

    train_data = train_data_set

    train = train_data[:-500]
    test = train_data[-500:]

    # our X is train[0]
    # train[0] is our x and f(x)
    # our Y is train [1]
    # train [1] is the degree of the polynomial

    X = np.array([i[0] for i in train]).reshape(-1, n_data_points, 1)
    Y = np.array([turn_degree_into_one_hot(i[1]) for i in train])

    print(train[0][1])

    test_x = np.array([i[0] for i in test]).reshape(-1, n_data_points, 1)
    test_y = np.array([turn_degree_into_one_hot(i[1]) for i in test])

    model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

if __name__ == '__main__':

    train_data_set = pickle.load(open("train.p", "rb"))
    test_data_set = pickle.load(open("test.p", "rb"))    

    model = neural_network_model()
    train_convolutional_network(model)

