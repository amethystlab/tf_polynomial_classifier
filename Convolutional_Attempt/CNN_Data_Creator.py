import pickle
import sys

import numpy as np

from Polynomial import Polynomial



''' Creates our data

creates and returns an array of data sets
'''
def create_data(num_data_sets, max_degree):

    array_of_data_sets = np.ndarray((num_data_sets), dtype=np.object)

    for i in range(num_data_sets):
        array_of_data_sets[i] = compute_values(max_degree)

    return array_of_data_sets



''' Computes f(x) values

computes the f(x) values for a given function
on a specified range of x values
'''
def compute_values(max_degree):

    poly = Polynomial(max_degree)
    a = np.random.uniform(-10,10)
    b = np.random.uniform(-10,10)

    if b<a:
        c = a
        a = b
        b = c

    x = np.random.uniform(-1, 1, 500)
    x.sort()
    fx = poly.evaluate(x)
    return (np.append(x,fx), poly.degree)



if __name__ == '__main__':

    num_data_sets = 100000
    num_test_sets = 1000

    max_degree = 10

    if len(sys.argv) > 1:
        num_data_sets = int(sys.argv[1])

    print("Creating {} training data sets...".format(num_data_sets))
    train_data_set = create_data(num_data_sets, max_degree)
    pickle.dump(train_data_set, open("train.p", "wb"))

    print("Creating {} test data sets...".format(num_test_sets))
    test_data_set = create_data(num_test_sets, max_degree)
    pickle.dump(test_data_set, open("test.p", "wb"))

    meta_information = {'num_test': num_test_sets,
                        'num_train': num_data_sets, 'max_degree': max_degree}

    pickle.dump(meta_information, open('meta.p', 'wb'))
