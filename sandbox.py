from numpy import ones, where, array, set_printoptions, asarray, full, float16, float32, cross,\
    dot, NaN, logical_and, isnan, transpose, matmul
from numpy.random import random
from numpy.linalg import norm
from timeit import default_timer
from matplotlib import pyplot as plt
set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
from scipy.optimize import minimize, Bounds, LinearConstraint
from new_geometry_tools import *
from new_scenes import *

