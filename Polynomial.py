
import numpy as np 
from Term import Term

class Polynomial:

	# OBJECT CONSTRUCTOR
	def __init__(self):
		# degree of polynomial, any random number 1 - 20 (arbitrary choice)
		self.degree = np.random.randint(1, 21)

		# number of coefficients in polynomial,
		# random number 1 - 20 (another arbitrary choice)
		self.num_coeffs = self.degree
		self.coeffs = self.generate_coeffs(self.num_coeffs, self.degree)
		

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

		for i in range(num_coeffs):			
			# fill list with random coeffs from -100 to 100
			# (another arbitrary coice)
			# coeff = np.random.randint(-100, 101)
			coeff = np.random.uniform(low=-100, high=100, size=1)

			terms[i] = Term(coeff, degree) # need proper syntax

		# make sure the polynomial is the degree it says it is
		terms[0].exponent = degree

		return terms

	def evaluate(self, x):
		f_x = np.zeros(len(x))
		num_coeffs = len(self.coeffs)

		for p in self.coeffs:
			coeff = p.coeff
			exponent = p.exponent
			f_x += coeff * pow(x, exponent)

		return f_x


if __name__ == '__main__':
	poly = Polynomial()
	print(poly)

	# to run in interactive shell type
	# exec(open("./Polynomial.py").read())
	# then you can play with the class