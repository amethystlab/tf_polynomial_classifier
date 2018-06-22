import datetime
import numpy as np
import os
import pickle
import tflearn
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression



LR = 1e-3
META_INFORMATION = pickle.load(open("meta.p", "rb"))
MODEL_NAME = 'degree_classifier-{}-{}.model'.format(LR, '6conv-basic')
N_DATA_POINTS = 500 * 2		# the 2 represents the x and the f(x)
N_OUTPUT_LAYER = META_INFORMATION['max_degree']



'''Creates a neural network

We chose to create a convolutional neuralnetwork. 
The methods in tflearn made this much easier and much cleaner
than if we were to use tensorflow alone
'''
def neural_network_model():

    convnet = input_data(shape=[None, N_DATA_POINTS, 1], name='input')

    convnet = conv_1d(convnet, 32, 2, activation='relu')
    convnet = max_pool_1d(convnet, 2)

    convnet = conv_1d(convnet, 64, 2, activation='relu')
    convnet = max_pool_1d(convnet, 2)

    convnet = conv_1d(convnet, 32, 2, activation='relu')
    convnet = max_pool_1d(convnet, 2)

    convnet = conv_1d(convnet, 64, 2, activation='relu')
    convnet = max_pool_1d(convnet, 2)

    convnet = conv_1d(convnet, 32, 2, activation='relu')
    convnet = max_pool_1d(convnet, 2)

    convnet = conv_1d(convnet, 64, 2, activation='relu')
    convnet = max_pool_1d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, N_OUTPUT_LAYER, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!!!')

    return model



''' Trains our neural network

Uses methods in tflearn to train our network then
saves the network to disk
'''
def train_convolutional_network(model):

    train_data_set = pickle.load(open("train.p", "rb"))
    train_data = train_data_set

    train = train_data[:-500]
    test = train_data[-500:]

    '''
    our X is train[0]
    train[0] is our x and f(x)
    our Y is train [1]
    train[1] is the degree of the polynomial
    train[1] is a one-hot vector
    '''

    X = np.array([i[0] for i in train]).reshape(-1, N_DATA_POINTS, 1)
    Y = np.array([turn_degree_into_one_hot(i[1]) for i in train])

    test_x = np.array([i[0] for i in test]).reshape(-1, N_DATA_POINTS, 1)
    test_y = np.array([turn_degree_into_one_hot(i[1]) for i in test])

    model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x},
            {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)




''' Turns the degree of the polynomial into a one hot 

This method is used to turn our degree into a one hot
array to be put into our model
'''
def turn_degree_into_one_hot(q):
    z = np.zeros(N_OUTPUT_LAYER)
    z[q] = 1
    return z




if __name__ == '__main__':

    model = neural_network_model()
    train_convolutional_network(model)

