import numpy as np
import pickle
import pybertini
import random
import sys

def main(n):
	tols = 10 ** (np.linspace(-2, -10, n))
	# preds = [p for p in pybertini.tracking.Predictor]
	import pybertini.tracking as tr
	P = tr.Predictor
	preds = [P.Constant, P.Euler, P.Heun, P.HeunEuler, P.RK4, P.RKCashKarp45, P.RKDormandPrince56, P.RKF45, P.RKNorsett34, P.RKVerner67]
	data_set = create_data(tols, preds)
	pickle.dump(data_set, open('data_set.p','wb'))

def create_data(tracking_tolerances, predictors):

	array_of_data_sets = np.ndarray(tracking_tolerances * predictors, dtype=np.object)
	i = 0
	for t in tracking_tolerances:
		for p in predictors:
			runtime, successcode = compute_values(t, p)
			array_of_data_sets[i] = (t, p, runtime, successcode)
			i += 1

	return array_of_data_sets

def compute_values(tol, pred):
	#todo: compute values
	return random.normalvariate(), pybertini.tracking.SuccessCode.Success


