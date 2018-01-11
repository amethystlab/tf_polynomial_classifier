import numpy as np
import matplotlib.pyplot as plt
from Term import Term


class Polynomial:

    # OBJECT CONSTRUCTOR
    def __init__(self, maxdegree=21):
        # degree of polynomial, any random number 1 - 20 (arbitrary choice)
        self.degree = np.random.randint(1, maxdegree)

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

    # CREATES A STRING THAT IS COPY / PASTABLE TO MATLAB
    def print_to_matlab(self):
        pretty = ''
        for c in self.coeffs:
            pretty += '{}*x.^{} + '.format(c.coeff, c.exponent)
        return pretty[:-3]

    # GENERATE COEFFECIENTS FOR CONSTRUCTOR
    def generate_coeffs(self, num_coeffs, degree):
        terms = np.ndarray((num_coeffs), dtype=np.object)

        for i in range(num_coeffs):
            # fill list with random coeffs from -100 to 100
            # (another arbitrary coice)
            # coeff = np.random.randint(-100, 101)
            coeff = np.random.uniform(low=-100, high=100)

            terms[i] = Term(coeff, degree)  # need proper syntax

        # make sure the polynomial is the degree it says it is
        terms[0].exponent = degree

        return terms

    # EVALUATES OUR POLYNOMIAL GIVING US f(x) VALUES
    def evaluate(self, x):
        f_x = np.zeros(len(x))
        num_coeffs = len(self.coeffs)

        for p in self.coeffs:
            coeff = p.coeff
            exponent = p.exponent
            f_x += coeff * pow(x, exponent)

        return f_x

    # GRAPH THE POLYNOMIAL USING PYPLOT IN MATPLOTLIB
    def graph(self, x=np.linspace(-1, 1, num=100)):
        f_x = self.evaluate(x)

        plt.plot(x, f_x)
        plt.xlabel('x-value')
        plt.ylabel('f(x)-value')
        plt.title('Polynomial')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    poly = Polynomial()
    print(poly)
    poly.graph()

    # to run in interactive shell type
    # exec(open("./Polynomial.py").read())
