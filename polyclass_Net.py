import tensorflow as tf
import datetime
import pickle
import numpy as np
import Data_Creator as dc

# Given: [xi, f(xi)]
# Compute: d

meta_information = pickle.load(open("meta.p", "rb"))

# dan, consider programmatically constructing the layers using something like this:
# layer_sizes = [[800],[500,500],[400],[100]] # nodes for layer 3

n_data_points = 1000 * 2		# the 2 represents the x and the f(x)
n_nodes_hl1 = 800  # nodes for layer 1
n_nodes_hl2 = 500  # nodes for layer 2
n_nodes_hl3 = 400  # nodes for layer 3
n_list_data = meta_information['num_train']  # 100000 currently
n_output_layer = meta_information['max_degree']
num_batches = 10
# arbitrary choice that can be changed
batch_size = int(n_list_data/num_batches)
# batch_size is currently 10000
# we want to make a next batch function that takes a batch size

num_epochs = 1  # epoch = a repetition of data training; chosen arbitrarily

tf.set_random_seed(57)

# (0, n_data_points) as an ordered pair
x = tf.placeholder('float', [None, n_data_points])
y = tf.placeholder('float', [None, n_output_layer])


def neural_network_model(data):

    # (input_data * weights) + biases

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_data_points, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    # creates a variable --> weights = finds normal of the data_size and number of nodes for layer 1
    # biases = finds normal of the nodes for layer 1

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_output_layer])),
                    'biases': tf.Variable(tf.random_normal([n_output_layer]))}
    # creates a variable --> weights = finds normal of the number of nodes for
    # layer 3 and number of classes

    # relu = activation function
    # The activation ops provide different types of nonlinearities for use in neural networks.
    # continuous but not everywhere differentiable functions
    # returns a Tensor

    l1 = tf.add(tf.matmul(data, hidden_1_layer[
                'weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1, 'layer1')

    l2 = tf.add(tf.matmul(l1, hidden_2_layer[
                'weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2, 'layer2')

    l3 = tf.add(tf.matmul(l2, hidden_3_layer[
                'weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3, 'layer3')

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    #(multiplies layer 3 with the weight of the output layer) + the output layer biases

    return output


def make_position_v(q):
    z = np.zeros(n_output_layer)
    z[q] = 1
    return z


def train_neural_network(x):
    predictor = neural_network_model(x)

    #(input_data * weights) + biases
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=predictor, labels=y))
    # minimize our cost
    # softmax: used for multi-class classification

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # the AdamOptimizer was chosen to optimize our data, but why?

    # data_set = pickle.load(open("train.p", "rb"))

    with tf.Session() as sess:
        # begin tf session... we will need to keep this open in order to keep
        # our "machine"
        sess.run(tf.global_variables_initializer())

        print('Testing {} pieces of data {} times for a total of {} pieces of data tested.'.format(
            batch_size, num_batches, batch_size*num_batches))

        for epoch in range(num_epochs):
            epoch_loss = 0

            # resead random number generator here
            dc.seed()

            for ii in range(num_batches):

                current_batch = dc.create_data(
                    batch_size, meta_information['max_degree'])

                epoch_x = np.asarray(
                    [np.hstack((item[0], item[1])).transpose() for item in current_batch])
                epoch_y = np.asarray(
                    [make_position_v(item[2]).transpose() for item in current_batch])

                _, c = sess.run([optimizer, cost], feed_dict={
                                x: epoch_x, y: epoch_y})
                epoch_loss += c
                print('Cost = {:1.8f}'.format(c))
            print('Epoch {}, completed out of {}, with loss {}.'.format(
                epoch+1, num_epochs, epoch_loss))

        correct = tf.equal(tf.argmax(predictor, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        test_d = pickle.load(open("test.p", "rb"))
        test_x = np.asarray([np.hstack((g[0], g[1])).transpose()
                             for g in test_d])
        test_y = np.asarray([make_position_v(g[2]).transpose()
                             for g in test_d])

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

        return predictor, sess

with tf.Session() as sess:
    machine, sess = train_neural_network(x)

    # created a stored "machine" to train the nerual net consistantly
    saver = tf.train.Saver()
    now = datetime.datetime.now()
    base_filename = "polyclass_{}{}{}".format(now.year, now.month, now.day)
    with open(base_filename + '_machine.pickle', 'wb') as file:
        pickle.dump(machine, file, protocol=pickle.HIGHEST_PROTOCOL)
        saver.save(sess, base_filename + '.sess')
