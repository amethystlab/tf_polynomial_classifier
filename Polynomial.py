
import numpy as np 
from Term import Term

class Polynomial:

	# OBJECT CONSTRUCTOR
	def __init__(self):
		# degree of polynomial, any random number 1 - 20 (arbitrary choice)
		# do these all need to be self.degree?
		self.degree = np.random.randint(1, 21)

		# number of coefficients in polynomial,
		# random number 1 - 20 (another arbitrary choice)
		self.num_coeffs = self.degree
		self.coeffs = self.generate_coeffs(self.num_coeffs, self.degree)
		# need a way to combine coeff with exponent, as well as 'x'
		# possibly a 'terms' class that does this, ask Dani what they think

	# WHAT IS TO BE PRINTED TO SCREEN
	def __str__(self):
		string = ""
		string += "Polynomial:\n"
		string += "Degree: {}\n".format(self.degree)
		string += "Coeffs: {}\n".format(self.coeffs)
		return string

	# GENERATE COEFFECIENTS FOR CONSTRUCTOR
	def generate_coeffs(self, num_coeffs, degree):
		terms = np.ndarray((num_coeffs), dtype=np.object)
		# need to figure out how to add a terms object to a python class
		# also need to figure out how i want to deal with exponents

		for i in range(num_coeffs):
			# fill list with random coeffs from 0 to 100
			# (another arbitrary coice)
			coeff = np.random.randint(-100, 101)
			terms[i] = Term(coeff, degree) # need proper syntax

		return terms
	
	# SOLVE f(x) AND RETURN ORDERED PAIRS
	# WOULD PROBABLY BE USED IN A 'GENERATE PAIRS' CLASS
	# def f(x):


if __name__ == '__main__':
	poly = Polynomial()
	print(poly)