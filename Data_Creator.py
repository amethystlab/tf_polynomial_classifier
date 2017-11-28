from Polynomial import Polynomial

import numpy as np
import pickle
import sys

def create_data(num_data_sets, max_degree):

	array_of_data_sets = np.ndarray((num_data_sets), dtype=np.object)

	for i in range(num_data_sets):
		array_of_data_sets[i] = compute_values(max_degree)

	return array_of_data_sets

def compute_values(max_degree):

	poly = Polynomial(max_degree)
	x = np.linspace(-1, 1, num=1000)

	return (x, poly.evaluate(x), poly.degree)

def seed(s = 5757):
	return np.random.seed(s)

if __name__ == '__main__':

	num_data_sets = 100000
	num_test_sets = 1000

	max_degree = 3

	if len(sys.argv) > 1:
		num_data_sets = int(sys.argv[1])

	print("Creating {} training data sets...".format(num_data_sets))
	train_data_set = create_data(num_data_sets, max_degree)
	pickle.dump(train_data_set, open("train.p", "wb"))

	print("Creating {} test data sets...".format(num_test_sets))
	test_data_set = create_data(num_test_sets, max_degree)
	pickle.dump(test_data_set, open("test.p", "wb"))

	meta_information = {'num_test': num_test_sets, 'num_train': num_data_sets, 'max_degree': max_degree}

	pickle.dump(meta_information, open('meta.p','wb'))
