# To run from an interactive shell type vvv
# 				exec(open("./Data_Creator.py").read())

from Polynomial import Polynomial

import numpy as np
import pickle
import sys

def compute_values():
	poly = Polynomial()

	# should probably randomize all the values here...
	# not sure if thats cool though, array might be
	# expecting a certain size
	x = np.linspace(-1, 1, num=1000)

	x_length = len(x)

	f_x = np.zeros(x_length)

	num_coeffs = len(poly.coeffs)
	for p in poly.coeffs:
		coeff = p.coeff
		exponent = p.exponent
		f_x += coeff * pow(x, exponent)

	return (x, f_x, poly.degree)

def create_data(num_data_sets):

	array_of_data_sets = np.ndarray((num_data_sets), dtype=np.object)
	
	for i in range(num_data_sets):
		array_of_data_sets[i] = compute_values()

	return array_of_data_sets


if __name__ == '__main__':

	# default value
	num_data_sets = 1000

	if len(sys.argv) > 0:
		num_data_sets = int(sys.argv[1])

	# test = create_data(1)
	# print(test)

	print("Creating {} data sets...".format(num_data_sets))
	data_set = create_data(num_data_sets)

	print("Pickling data sets...")
	pickle.dump(data_set, open("data_set.p", "wb"))