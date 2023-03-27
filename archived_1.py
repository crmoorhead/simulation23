from numpy import ones, where, array, set_printoptions, asarray, full, float16, float32, cross,\
    dot, NaN, logical_and, isnan, transpose, matmul
from numpy.random import random
from timeit import default_timer
from matplotlib import pyplot as plt
set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

"""k = 4
n = 4
m = 4

class GeneratorObject():

    def __init__(self,n,m):
        self.width = n
        self.height = m

    def generate_data(self, k=1, **kwargs):
        if "thresh" in kwargs:
            threshold = kwargs["thresh"]
        else:
            threshold = 0.5
        if "fill" in kwargs:
            fill = kwargs["fill"]
        else:
            fill = 128
        if k == 1:
            value_array = random((self.height, self.width))
        else:
            value_array = random((k, self.height, self.width))
        value_array = where(value_array < threshold, fill, value_array)
        return value_array

    def intersection_params(self, rays, pov, **kwargs):
        return self.generate_data(**kwargs)

data_obs = [GeneratorObject(n,m) for i in range(k)]
rays = data_obs[0].generate_data(3)

def create_dict(objects, rays, pov):
    print("input rays", rays.shape)
    idxs = full(rays.shape[1:], -1, dtype=float16)    # Rays all assigned as not hitting any object
    dists = full(rays.shape[1:], 128, dtype=float16)  # Distances all set to 128 (much higher than scene scope)
    object_num = len(objects)
    ray_dict, idx_dict = {}, {}
    for i in range(object_num):
        dist = objects[i].intersection_params(rays, pov)     # Calc distances until intersection for current object
        idxs, dists = collect_min(idxs, dists, dist, i) # updates closest distances for each ray and retains object
    print("dists", dists.shape)

    for f in range(object_num):
        conditions = asarray(where(idxs == f)).T       # Finds indices where rays hit current object
        if array(conditions).size != 0:                # If there are any hits
            ray_dict[f] = array([rays[:, c[0], c[1]] for c in conditions]) # We assign a subarray of rays for that object
            idx_dict[f] = conditions
        else:
            ray_dict[f] = None
            idx_dict[f] = None

    print("rarray", ray_dict[0].shape)
    print(idx_dict)
    return ray_dict, idx_dict

def collect_min(indices, mins, new_array, index):
    indices = where(new_array < mins, index, indices)
    mins = where(new_array < mins, new_array, mins)
    return indices, mins

rays = create_dict(data_obs, rays, None)
print(rays[1])

from new_geometry_tools import *"""

"""test_tri = Triangle([1,-0.5,0], [1,0.5,0], [1,0,2])
test = array([[0,0,1],[0,1,0],[1,0,0],[0,-1,0]],dtype=float32)
test_tet = Tetrahedron([1,-0.5,0], [1,0.5,0], [1,0,2], [2,0,0])
test_tet.info()"""

# test_plane = Plane([0,0,-1])]

from numpy.random import random
from numpy import array
from numpy.linalg import norm

rays = random((3,3,2))*2
rays /= norm(rays,axis=0)
pov = array([1,2,3], dtype=float)
u = array([10,0,0], dtype=float)
v = array([5,0,10], dtype=float)
anchor = array([-4,3,-2], dtype=float)

print(rays)

def interfunction(rays, pov, u, v, anchor):
    print(rays.shape)
    rshape = rays.shape[1:]
    rays = rays.reshape((3, rays.shape[1] * rays.shape[2])).T
    epsilon = 10 ** -6
    T = pov - anchor
    P = cross(rays, v.reshape((1, 3)))
    S = dot(P, u) + epsilon
    U = dot(P, T)
    U /= S
    if True in (U >= 0) & (U <= 1):
        Q = cross(T, u)
        V = where((U >= 0) & (U <= 1), dot(Q, rays.transpose()), NaN) / S
        t = where((V >= 0) & (V <= 1) & (U + V <= 1), dot(Q, v), NaN) / S
        t = where(t <= 0, NaN, t)
        V = V.reshape(rshape)
        U = U.reshape(rshape)
        t = t.reshape(rshape)
        return U, V, t
    else:
        return None, None, None

U, V, t = interfunction(rays, pov, u, v, anchor)

print(U, "\n\n", V, "\n\n", t)