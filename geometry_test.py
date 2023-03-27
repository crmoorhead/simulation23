from new_geometry_tools import *
import timeit
from new_scenes import *
from numpy import isclose, pi, linspace, stack, meshgrid, set_printoptions, inf, array, amin, amax, round
set_printoptions(threshold=inf)


"""print(min(timeit.repeat("isclose(array([0,0,0]), array([0,0,0]))",number=1000,setup='from numpy import isclose, array')))
print(min(timeit.repeat("array([0,0,0]) == array([0,0,0])",number=1000,setup='from numpy import isclose, array')))"""

"""test_box = AABB([-1,2,1],[1,3,2])
test_box_2 = AABB([-0.1,2,1],[0.1,2.1,1.05])
filt = test_box.bb_filter([0,0,4])
filt_2 = test_box_2.bb_filter([0,0,4])

theta = linspace(-pi/2, pi/2, 1000)
phi = linspace(0, -pi/2, 1000)
ray_num = 1000*1000
# print(ray_num)
tangles = stack(meshgrid(theta, phi, indexing="xy"))
print(min(timeit.repeat("filt_2(tangles)",number=1,setup='from numpy import pi',globals=globals())))
print(filt_2(tangles))
# print(filt(tangles).shape)"""

"""def box_generator(n,limits, side):
    boxes = {}
    limits = array(limits)
    mins, maxes = limits[:,0], limits[:,1]
    ranges = maxes - mins
    centres = round(random((n,3))*ranges + mins,3)
    for i in range(n):
        boxes["box_"+str(i+1)] = AABB(centres[i]-side/2, centres[i]+side/2)
    return boxes

theta = linspace(-pi/2, pi/2, 1000)
phi = linspace(0, -pi/2, 1000)
tangles = stack(meshgrid(theta, phi, indexing="xy"))
tboxes = box_generator(150,[[-10,10],[2,20],[-5,5]],0.2)
print(min(timeit.repeat("for b in tboxes: bangles = tboxes[b].bb_filter([0,0,4])(tangles)",number=10,setup='',globals=globals())))"""

tcomp = Composite()
print(tcomp)
print(tcomp.__dict__)
t_wall = Wall([2,3,0])
print(t_wall.__dict__)

