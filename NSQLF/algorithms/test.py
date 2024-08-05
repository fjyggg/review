import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
from cvxopt import solvers, matrix, spdiag, sqrt, div, exp, spmatrix, log

a = np.arange(9).reshape(3, 3)
print(np.sum(a, 1))