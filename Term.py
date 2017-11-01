import numpy as np 

class Term:

	def __init__(self, coeff, degree):
		self.coeff = coeff
		self.exponent = np.random.randint(0, degree + 1)

	def __repr__(self):
		string = ""
		string += "{}".format(self.coeff)
		string += "x^"
		string += "{}".format(self.exponent)
		return string