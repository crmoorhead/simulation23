# Tools used for geometrical calculations for ray-tracing
from numpy import all, array, zeros, minimum, maximum, \
    clip, amin, amax, stack, multiply, ones, concatenate, divide, \
    isclose, ndarray, arctan2, pi, where, arcsin, cross, dot, matmul, \
    NaN, log10, expand_dims, squeeze, tensordot, full_like, full, asarray
from numpy.random import random
from numpy.linalg import norm
from itertools import product
from collections import OrderedDict
from scipy.optimize import minimize, Bounds, LinearConstraint
from skimage.measure import marching_cubes
from scipy.spatial.transform import Rotation as R
from os import getcwd
from os.path import join
from math import ceil
from math import ceil

# RULES FOR REWRITING

# 1. Removal of  class
# 2. Removal of unnecessary checks
# 3. Substitution of all array conversions by assuming input is array and not list

# SCATTERER CLASS
from inspect import signature, isfunction
from functools import partial

class Scatterer:

    def __init__(self, scatter_funct, **kwargs):
        if "name" not in kwargs:
            self.name = "unnamed_function"
        else:
            self.name = kwargs["name"]
        if not isfunction(scatter_funct):
            raise TypeError("All instances of Scatterer objects must define a function with two arguments "
                            "- incident and return_angle. No functional input.")
        sig = signature(scatter_funct)
        params = sig.parameters
        if "incident" not in params or "return_angle" not in params:
            raise ValueError("All instances of Scatterer objects must define a function with two arguments "
                            "- incident and return_angle. One or both of these arguments are missing.")
        else:
            self.general_function = scatter_funct
        self.add_params = {p: None for p in params if p not in ["incident", "return_angle"]} # Lists all additional parameters in the scatter_funct variable.
        self.fixed_function = None
        self.monostatic = None

    def __str__(self):
        output = "Scatterer(Name: {}, Params: ".format(self.name)
        for p in self.add_params:
            if self.add_params[p] is not None:
                output += p + ": " + str(self.add_params[p]) + ", "
            else:
                output += p+", "
        output = output[:-2]
        output += ")"
        return output

    def __repr__(self):
        return self.__str__()

    def info(self):
        print(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        return copy(self)

    # Class Methods

    ## Ray return methods

    def fix_function(self, param_dict):
        for p in self.add_params:
            if p not in param_dict:
                raise ValueError("{} not assigned a value. This is a compulsory variable.".format(p))
                return
            else:
                self.add_params[p] = param_dict[p]
        code = "partial(self.general_function, "
        for p in param_dict:
            code += p + "=" +str(param_dict[p])+", "
        code += ")"
        self.fixed_function = eval(code)
        self.gen_monostatic()

    def unfix(self):
        self.fixed_function = None
        self.monostatic = None

    def gen_monostatic(self,**kwargs):
        if self.fixed_function is None:
            if "params" not in kwargs:
                raise ValueError("Function is not fixed. Either fix function or supply 'params' keyword")
            else:
                self.fix_function(kwargs["params"])
        else:
            def monostatic(incident):
                return self.fixed_function(incident, pi-incident)
            self.monostatic = monostatic

    # Aliases for easy use

    def SL_bi(self, incident, return_angle):
        return self.fixed_function(incident, return_angle)

    def SL(self, incident):
        return self.monostatic(incident)

    # Misc methods

    def interpolate(self, other, **kwargs):
        if not isinstance(other, Scatterer):
            raise TypeError("Both objects must be of Scatterer class.")

        joint_params = {**self.add_params, **other.add_params}

        if len(self.add_params) + len(other.add_params) != len(joint_params):
            for p in self.add_params:
                if p in other.add_params:
                    if (self.add_params[p] is not None) and (other.add_params[p] is not None):
                        if self.add_params[p] != other.add_params[p]:
                            raise ValueError("Fixed functions have incompatible parameters. Unfix at least one.")
                    else:
                        if self.add_params[p] is None:
                            pass


        if "ratio" in kwargs:
            t=kwargs["ratio"]
        else:
            t=0.5

        def funct_interp(f, g, t, *args, **kwargs):
            def new_funct(incident, return_angle, *args, **kwargs): # A function that outputs the desired output for given inputs
                return t * f(incident, return_angle, *args, **kwargs) + (1 - t) * g(incident, return_angle, *args, **kwargs)
            return new_funct

        interp_scatter_funct=funct_interp(self.scatter_funct, other.scatter_funct,t) # Creates linear interpolation of scatter_funct for two different Scatterer objects
        interp_scatter_funct.__name__=str(t)+"*"+self.scatter_funct.__name__ +" + "+str(1-t)+"*"+other.scatter_funct.__name__

        interp_scatterer= Scatterer(interp_scatter_funct)

        return interp_scatterer

# STANDARD SCATTERERS

from numpy import sin, cos, log10, multiply, isnan, where

# Note: These assume symmetry ATM

def lambertian_bistatic(incident, return_angle, mu):
    print("generating lambertians")
    SL = where(~isnan(incident), mu + 10 * log10(multiply(sin(incident), sin(return_angle))), NaN)
    return SL

LAMBERTIAN_SCAT = Scatterer(lambertian_bistatic)

def gaussian_noise(incident, return_angle, sd):
    pass

# ROTATIONS

# Rotate object is a container for rotation information. Objects have their own rotate methods that call the scipy apply function.

class Rotation(R):

    def __init__(self, zyx_TB_angles, centre=array([0, 0, 0])): #https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles
        self.angles = array(zyx_TB_angles)
        self.rotation = R.from_euler("zyx", array(zyx_TB_angles), degrees=True)
        self.centre = array(centre)

    def __str__(self):
        return "Rotation(Angles:{}, Centre:{})".format(self.angles, self.centre)

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        if not isinstance(other, Rotation):
            raise TypeError("Both objects must be of Rotation type.")
        if any(self.centre != other.centre):
            raise ValueError("Centres are not the same. Combination incompatible")
        angles = self.angles + other.angles
        return Rotation(angles, self.centre)

    def __add__(self, other):
        return self * other

    def __iadd__(self, other):
        return self * other

    def __eq__(self, other):
        if isinstance(other, Rotation):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply(self, points):
        points -= self.centre
        if len(points.shape) != 3:
            points = self.rotation.apply(points)
        else:
            (d1, d2, d3) = points.shape
            if d3 != 3:
                raise ValueError("Depth should be 3. Cannot perform transformation")
            points = self.apply(points.reshape((d1*d2, 3)))
            points = points.reshape((d1, d2, 3))
        points += self.centre
        return points

    def shift(self, vector):
        self.centre += vector

# BOUNDING OBJECT CLASS

class Envelope():

    def __init__(self, bounds=20):
        self.envelope_type = "abstract"
        self.bounds = bounds

    def union(self, other):
        raise ReferenceError("Method not defined.")

    def overlaps(self, other):
        raise ReferenceError("Method not defined.")

    def is_in(self, point):
        raise ReferenceError("Method not defined.")

    def intersection(self, other):
        raise ReferenceError("Method not defined.")

    def ray_filter(self):
        raise ReferenceError("Method not defined.")

# BOUNDING BOX CLASS

class AABB(Envelope):

    def __init__(self, c_1, c_2):
        super().__init__()
        self.corners = array([c_1, c_2])
        self.x_min, self.y_min, self.z_min = clip(amin(self.corners, axis=0), -self.bounds, self.bounds)
        self.x_max, self.y_max, self.z_max = clip(amax(self.corners, axis=0), -self.bounds, self.bounds)
        self.corners = array([[self.x_min, self.y_min, self.z_min],[self.x_max, self.y_max, self.z_max]])
        self.width = self.x_max - self.x_min
        self.length = self.y_max - self.y_min
        self.height = self.z_max - self.z_min

    def __str__(self):
        return "AABB(C_1: [{},{},{}], C_2: [{},{},{}])".format(self.x_min,self.y_min,self.z_min,self.x_max,self.y_max,self.z_max)

    def __repr__(self):
        return self.__str__()

    def union(self, other):
        if isinstance(other, AABB):
            corners = concatenate((self.corners, other.corners))
            self.x_min, self.y_min, self.z_min = amin(corners, axis=0)
            self.x_max, self.y_max, self.z_max = amax(corners, axis=0)
            self.corners = array([[self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max]])
            self.width = self.x_max - self.x_min
            self.length = self.y_max - self.y_min
            self.height = self.z_max - self.z_min
            return self
        if isinstance(other, list) or isinstance(other, array):
            corners = concatenate((self.corners, array(other).reshape((1,3))))
            self.x_min, self.y_min, self.z_min = amin(corners, axis=0)
            self.x_max, self.y_max, self.z_max = amax(corners, axis=0)
            self.corners = array([[self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max]])
            self.width = self.x_max - self.x_min
            self.length = self.y_max - self.y_min
            self.height = self.z_max - self.z_min
            return self
        if other is None:
            return self
        else:
            raise TypeError("Union must be of two AABBs instances or one AABB and one instance of vector or list.")

    def overlaps(self, other):
        if other is None:
            return False
        elif isinstance(other, AABB):
            x_test = self.x_max > other.x_min and self.x_min < other.x_max
            y_test = self.y_max > other.y_min and self.y_min < other.y_max
            z_test = self.z_max > other.z_min and self.z_min < other.z_max
        else:
            raise TypeError(
                "Argument must be a bounding box object.")
        return x_test and y_test and z_test

    def is_in(self, point):
        point = array(point).reshape((1,3))
        x_test = self.x_max >= point[0,0] and self.x_min <=  point[0,0]
        y_test = self.y_max >=  point[0,1] and self.y_min <=  point[0,1]
        z_test = self.z_max >=  point[0,2] and self.z_min <=  point[0,2]
        return x_test and y_test and z_test

    def pad(self, other):
        if other not in [int, float]:
            raise TypeError("Padding must be integer or float.")
        self.x_min -= other
        self.x_max += other
        self.y_min -= other
        self.y_max += other
        self.z_min -= other
        self.z_max += other
        self.corners = array([[self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max]])
        self.width = self.x_max - self.x_min
        self.length = self.y_max - self.y_min
        self.height = self.z_max - self.z_min

    def volume(self):
        return self.width*self.length*self.height

    def surface_area(self):
        return 2*(self.width + self.length + self.height)

    def widest(self):
        if self.width >= self.length and self.width >= self.height:
            return "x"
        elif self.length >= self.height:
            return "y"
        else:
            return "z"

    def lerp(self,coord):
        coord = array(coord).reshape(3)
        if False in [coord[i] <= 1 and coord[i] >= 0 for i in range(3)]:
            raise ValueError("Coordinate indices must all be between 0 and 1")
        return array([self.x_min+self.width*coord[0],self.y_min+self.length*coord[1],self.z_min+self.height*coord[2]])

    def point_position(self, point, *args):
        point = array(point).reshape(3)
        if "test" in args and not self.is_in(point):
            return NotImplemented
        else:
            coords = divide(point-array([self.x_min, self.y_min, self.z_min]), array([self.width, self.length, self.height]))
            return coords

    def bb_filter(self, pov):
        pov = array(pov).reshape(3)
        if self.is_in(pov):
            return [[0,2*pi]]*3
        else:
            cs = array(list(product([self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max])))
            [dx, dy, dz] = [cs[:, i] - pov[i] for i in range(3)]
            cangles = array([arctan2(dx, dy), arctan2(dz, dy)])
            mins, maxes =  amin(cangles, axis=1), amax(cangles, axis=1)
            def filter_function(angles):
                angles = where((angles[0] <= maxes[0]) & (angles[1] <= maxes[1]) & (angles[0] >= mins[0]) & (angles[1] >= mins[1]), angles,0)
                angles = angles[:, ~all(angles[0] == 0, axis=1)]
                angles = angles[:, :, ~all(angles[0] == 0, axis=0)]
                if angles.size != 0:
                    return angles
            return filter_function

# GENERATOR_OBJECT: A class that exists purely to generate a random collection of returns. For testing collation methods.

class GeneratorObject():

    def __init__(self, n, m):
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
        return self.generate_data(3, **kwargs)


# ABSTRACT OBJECT CLASS

''' All objects must contain the following properties:

* Definition of the object parameters
* Way of determining if a point is on the surface of the object.
* A means of testing intersection with rays and outputting the point or points on the ray that intersect
* A means for generating normal vectors for any point on the surface.
* Other properties of the object e.g. material, colour, ability to intersect.
* Ability to name the object.
* Way of defining an anchor point in the object. Usually this will be the centre, but for infinite or non-symmetric 
  objects it may be useful to define otherwise. This may even be undefined.
* Way to print object details
* Any general or unique transformations on the object
* A default object for type checking and prototyping.
* For enclosed objects, whether a point is inside the object
* For surfaces, which side of the boundary a given point is.
* Equation of tangent at a point

'''

# ABSTRACT OBJECT CLASS
class Object:
    def __init__(self, **kwargs):
        self.object_type = "Object"
        if "anchor" in kwargs:
            self.anchor = kwargs["anchor"]
        else:
            self.anchor = array([0,0,0])
        self.epsilon = 10**(-6)
        if "rotation" in kwargs:
            self.rotation = Rotation(kwargs["rotation"], self.anchor)
            self.apply_rotation(self.rotation)
        else:
            self.rotation = Rotation([0, 0, 0], self.anchor)
        if "scatterer" in kwargs:
            if not isinstance(kwargs["scatterer"], Scatterer):
                raise ValueError("Assigned object scatterer must be of Scatterer class.")
            else:
                self.scatterer = kwargs["scatterer"]
        else:
            self.scatterer = LAMBERTIAN_SCAT
            self.scatterer.fix_function({"mu": -29})
            self.scatterer.name = "default_lambertian"
        self.PROPERTIES = {"scatterer": self.scatterer}
        if "name" in kwargs:
            self.name = kwargs["name"]
        else:
            self.name = None

    def __str__(self):
        if self.rotation == Rotation([0, 0, 0], self.anchor):
            return "Object("+str(list(self.anchor))+")"
        else:
            return "Object(" + str(list(self.anchor)) + ", Rotation: "+ str(self.rotation.angles) +")"

    def __repr__(self):
        return "Name: {}\nClass: {}\nParameters: {}\nProperties: {}\n".format(self.name, self.object_type, self.rotation, self.PROPERTIES)

    def info(self):
        print(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # Transformation methods

    def shift(self, vector):
        self.anchor += array(vector).reshape((3,))
        self.rotation.shift(vector)
        return self

    def apply_rotation(self, rotation):
        if not isinstance(rotation, Rotation):
            raise TypeError("Rotation argument must be an instance of Rotation")
        print("Placeholder function only applied. This object has no defined rotation application method.")

    def reset_rotation(self):
        self.rotation = Rotation([0, 0, 0], self.anchor)

    def ison(self, point):
        return isclose(self.anchor, array(point).reshape((3,)))

    def intersect(self, rays, pov):
        dif = self.anchor - pov
        return isclose(dif.dot(rays)*dif.dot(rays), dif.dot(dif)*rays.dot(rays))

    # Finish below
    def intersect_param(self, rays, pov, **kwargs):
        if "min_param" in kwargs:
            min_param = kwargs["min_param"]
        else:
            min_param = 0

        if self.intersect(rays, pov):
            param = vector.value[0]/self.value[0]
            if param >= min_param:
                return param
            else:
                return []
        else:
            return []

    def intersection_point(self, rays, pov, **kwargs):
        intersections = self.intersect_param(rays, pov, **kwargs)
        if intersections is None:
            return None
        else:
            return self.anchor

    # Normal Methods - note these return None, but set the form for inherited classes

    # An option to provide a check for all points is incorporated that can be used if the point may not be on the surface

    def gen_normal(self, point, *args):
        if "test_point" in args:
            if not self.ison(point):
                print("{} is not on the object surface".format(point))
                return None
        else:
            return None

    # adjust for multiple points case

    def normal(self, points, *args):
        if isinstance(points, ndarray):
            return self.gen_normal(point, *args)
        elif isinstance(point, list):
            normals = {}
            for p in point:
                normals[str(p)]=self.gen_normal(p,*args)
            if "filter" in args:
                normals={k:v for k,v in normals.items() if v!= None}
            return normals
        else:
            return None

    def gen_incident(self, rays):
        print("This is the default incident angle generator. Results will not be valid.")
        if rays.ndim == 2:
            return full(rays.shape[0], NaN)
        elif rays.ndim == 3:
            return full(rays.shape[1:], NaN)
        elif rays.size == 0:
            return array(0,)
        else:
            raise ValueError("Rays dimension is wrong.")
            
    # Misc Methods

    def add_property(self,prop_name, prop_value,*args):
        prop_name = str(prop_name)
        if prop_name in self.PROPERTIES.keys():
            if "verbose" in args:
                print("{} rewritten from '{}' to '{}'".format(prop_name,self.PROPERTIES[prop_name],prop_value))
            self.PROPERTIES[str(prop_name)] = prop_value
        else:
            self.PROPERTIES[str(prop_name)]=prop_value

    def change_scatterer(self, new_scat):
        if not isinstance(new_scat, Scatterer):
            raise TypeError("Scatterer must be an instance of Scatterer class.")
        else:
            self.scatterer = new_scat
            self.add_property("scatterer",new_scat)

    def name_object(self, name, *args):
        if self.name is None:
            self.name = name
        else:
            if "verbose" in args:
                print("Name of object changed from {} to {}.".format(self.name, name))
            self.name = name

# COMPOSITE OBJECT
class Composite(Object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_type = "Composite"
        self.PROPERTIES = {"scatterer": self.scatterer}
        self.COMPONENTS = OrderedDict()
        if "name" in kwargs:
            self.name=kwargs["name"]
        else:
            self.name = None
        self.component_count = len(self.COMPONENTS)
        if "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]
        else:
            self.accelerator = None
        # Placeholders for ray intersection memory
        self.rays = None
        self.comp_idxs = None
        if "rotation" in kwargs:
            self.rotation = Rotation(kwargs["rotation"], self.anchor)
        else:
            self.rotation = Rotation([0, 0, 0], self.anchor)

        for c in self.COMPONENTS:
            self.COMPONENTS.anchor = self.anchor
            self.COMPONENTS.rotation = self.rotation

    def __str__(self):
        return "Composite("+str(self.anchor)+")"

    def __repr__(self):
        desc = "Name: {}\nClass: {}\nParameters: {}\nProperties: {}\nComponents: ".format(self.name, self.object_type, self.rotation, self.PROPERTIES)
        if self.component_count == 0:
            desc += "None"
        else:
            for c in self.COMPONENTS:
                desc += "\n"+str(c)+": "+str(self.COMPONENTS[c])
        return desc

    def info(self):
        print(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # Transformation methods

    def shift(self, vector):
        self.anchor += vector
        for c in self.COMPONENTS:
            self.COMPONENTS[c].shift(vector)
        return self

    def apply_rotation(self, rotation):
        for c in self.COMPONENTS:
            self.COMPONENTS[c].rotation += rotation
            self.COMPONENTS[c].apply_rotation(rotation)

    # Intersection methods

    def assign_accelerator(self, accelerator):
        self.accelerator = accelerator

    def ison(self,point):
        for c in self.COMPONENTS:
            if self.COMPONENTS[c].ison(point):
                return True
        return False

    def gen_normal(self, point, *args):
        for c in self.COMPONENTS:
            if self.COMPONENTS[c].ison(point):
                return self.COMPONENTS[c].gen_normal(point)

    # Intersection methods:

    def intersect(self, rays, pov, *args, **kwargs): #???
        pass

    def intersection_params(self, rays, pov, *args, **kwargs):
        self.ray_hit_count = 0
        self.temp_hit = array([])
        if "accelerator" in kwargs:
            pass
        else:
            ob_idxs = full(rays.shape[1:], -1, dtype=float)  # Rays all assigned as not hitting any object
            dists = full(rays.shape[1:], 128, dtype=float)  # Distances all set to 128 (much higher than
            idx_dict = {}
            self.labels = list(self.COMPONENTS.keys())
            objects = list(self.COMPONENTS.values())
            for i in range(self.component_count):
                dist = objects[i].intersection_params(rays, pov)  # Calc distances until intersection for current object
                ob_idxs, dists = self.collect_min(ob_idxs, dists, dist, i)  # updates closest distances for each ray and retains object

            for f in range(-1, self.component_count):  # -1 is for no objects being hit for that ray
                conditions = asarray(where(ob_idxs == f)).T  # Finds indices where rays hit current object
                if array(conditions).size != 0:  # If there are any hits
                    idx_dict[f] = conditions
                    self.ray_hit_count += len(idx_dict[f])
                else:
                    idx_dict[f] = []

            self.ray_hit_count -= len(idx_dict[-1])  # Subtract misses
            self.comp_idxs = idx_dict
            return dists

    def collect_min(self, indices, mins, new_array, index):
        indices = where(new_array < mins, index, indices)
        mins = where(new_array < mins, new_array, mins)
        return indices, mins

    # composite objects need to reference object too. Save during intersect_params stage
    def gen_incident(self, rays):
        incidents = full(self.ray_hit_count, NaN)
        print("ray_hit_count", self.ray_hit_count)
        current_idx = 0
        for c in range(self.component_count):
            print(self.labels[c])
            comp_rays = [rays[:, self.comp_idxs[c][i][0], self.comp_idxs[c][i][1]] for i in range(len(self.comp_idxs[c]))]
            incidents[current_idx:current_idx + len(self.comp_idxs[c])] = self.COMPONENTS[self.labels[c]].gen_incident(comp_rays) # Need to allocate incidents to array
            current_idx += len(self.comp_idxs[c])
        return incidents

    # Misc methods

    def add_property(self,prop_name, prop_value,*args):
        prop_name=str(prop_name)
        if prop_name in self.PROPERTIES.keys():
            if "verbose" in args:
                print("{} rewritten from '{}' to '{}'".format(prop_name,self.PROPERTIES[prop_name],prop_value))
            self.PROPERTIES[str(prop_name)] = prop_value
        else:
            self.PROPERTIES[str(prop_name)] = prop_value
        for c in self.COMPONENTS:
            self.COMPONENTS[c].add_property(prop_name,prop_value)

    def change_scatterer(self,new_scat):
        if isinstance(new_scat, Scatterer):
            raise TypeError("Scatterer must be an instance of Scatterer class.")
        else:
            self.scatterer = new_scat
            self.add_property("scatterer", new_scat)
            for c in self.COMPONENTS:
                c.add_property("scatterer", new_scat)

# PLANE

 # PLANE is described by the formula z + ax + by + c = 0 and in the degenerate case, by the WALL object.
 # The expected input is [a, b, c]

class Plane(Object):

    def __init__(self, plane, **kwargs):
        super().__init__()
        self.object_type = "Plane"
        self.x_coeff = plane[0]
        self.y_coeff = plane[1]
        self.constant = plane[2]
        self.anchor = array([0,0,-self.constant])
        normal = array([-self.x_coeff, -self.y_coeff, 1]) + self.epsilon
        self.unit_normal = normal / norm(normal)
        self.plane_vec = array([self.x_coeff, self.y_coeff, 1]) + self.epsilon
        self.PARAMS={"x_coeff": self.x_coeff, "y_coeff": self.y_coeff, "constant": self.constant, "unit_normal": self.unit_normal, "anchor": self.anchor}
        if "range" in kwargs:
            base = kwargs["range"]
        else:
            base = 20
        c_1= [-base, -base, base*self.x_coeff+base*self.y_coeff-self.constant]
        c_2= [base, base, -base*self.x_coeff-base*self.y_coeff-self.constant]
        self.bounding_box = AABB(c_1, c_2)
        self.temp_ray_prod = None  # Used to store data in ray intersections

    def __str__(self):
        string="Plane(z "
        if self.x_coeff == 0:
            string += ""
        elif self.x_coeff == 1:
            string += "+ x "
        elif self.x_coeff > 0:
            string += "+ "+str(round(self.x_coeff, 4))+"x "
        else:
            string += "- " + str(round(abs(self.x_coeff), 4)) + "x "

        if self.y_coeff == 0:
            string += ""
        elif self.y_coeff == 1:
            string += "+ y "
        elif self.y_coeff > 0:
            string += "+ "+str(round(self.y_coeff, 4))+"y "
        else:
            string += "- " + str(round(abs(self.y_coeff), 4)) + "y "

        if self.constant == 0:
            string += ""
        elif self.constant > 0:
            string += "+ "+str(self.constant)
        else:
            string += "- " + str(abs(self.constant))

        string += " = 0)"

        return string

    # Intersection methods

    # All inputs need to be arrays of appropriate dimensions

    def ison(self, point):
        return self.plane_vec.dot(point)+self.constant == 0

    # Expects an input of rays from a given PoV

    def perpendicularity(self, rays):  # ray_direction * plane_normal (if 0, then the ray is in the plane)
        return dot(rays, expand_dims(self.unit_normal, 0).T)

    def intersection_params(self, rays, pov):
        self.temp_ray_prod = self.perpendicularity(rays)
        intersection_constant = dot(array([0, 0, -self.constant])-pov, self.unit_normal)
        ts = divide(intersection_constant, self.temp_ray_prod)
        return ts

    def gen_incident(self):  # Only called after intersection test already made
        prod = abs(self.temp_ray_prod)  # angle calculation only calculates absolute value of angle
        return arcsin(prod)

        # CHANGE TO ACCEPT ARRAYS
    def in_plane(self, pov, direction):
        point_1 = pov
        point_2 = pov + direction
        return self.ison(point_1) and self.ison(point_2)

    # Normal methods

    def gen_normal(self, point):
        return self.unit_normal

    # Misc methods

    def shift(self, dist):
        if dist.__class__ in [int, float]:
            self.constant -= dist
            self.anchor = array([0, 0, -self.constant])
            self.PARAMS["constant"], self.PARAMS["anchor"] = self.constant, self.anchor
            return self
        else:
            raise TypeError("Shift distance must be integer or float.")

    def gradient(self):
        return -self.unit_normal

    def apply_rotate(self, rotation):
        if not isinstance(rotation, Rotation):
            raise TypeError("Rotation must be an instance of Rotation object.")
        p_1 = rotation.apply(array([1, 0, self.constant-self.x_coeff]))
        p_2 = rotation.apply(array([0, 1, self.constant-self.y_coeff]))
        self.anchor = rotation.apply(self.anchor)
        u = p_1 - self.anchor
        v = p_2 - self.anchor
        normal = cross(u, v)
        c = -float(normal.dot(p_1))
        if normal[0] != 0:
            plane = [normal[1] / normal[0], normal[2] / normal[0], c / normal[0]]
            self.__init__(plane)
        else:
            Wall.__init__(self, [normal[1], normal[2], c])

# WALL

# Needed for sheer vertical faces which may occur when generating meshes.
# Uses formula ax + by +c = 0 (note z coefficient is zero) but normalises to either x + py + q = 0 or y + k = 0
# where p = b/a q = c/a k = c/b
# wall input is [a,b,c]

class Wall(Plane):

    def __init__(self, wall, **kwargs):
        [a, b, c] = list(wall)
        if a != 0:
            a, b, c = 1, b/a, c/a
        else:
            if b == 0:
                raise AttributeError("This is not a valid plane.")
            else:
                b, c = 1, c/b
        super().__init__(wall)
        self.x_coeff = a
        self.y_coeff = b
        self.constant = c
        if self.x_coeff != 0:
            self.anchor = array([-self.constant, 0, 0])
        else:
            self.anchor = array([0, -self.constant, 0])
        self.plane_vec = array([self.x_coeff, self.y_coeff, 0])
        if self.x_coeff != 0:
            normal = array([1, -self.y_coeff, 0])
        else:
            normal = array([0, -1, 0])
        self.unit_normal = normal/norm(normal)
        self.PARAMS = {"x_coeff": self.x_coeff,
                       "y_coeff": self.y_coeff,
                       "constant": self.constant,
                       "unit_normal": self.unit_normal,
                       "anchor": self.anchor}
        self.object_type = "Wall"
        if "base" not in kwargs:
            base = 20
        else:
            base = kwargs["base"]
        if self.x_coeff != 0:
            if self.y_coeff != 0:
                if abs(self.y_coeff) >= 1:
                    h  = (base - self.constant)/self.y_coeff
                    c_1 = [-base, -h, -base]
                    c_2 = [base, h, base]
                else:
                    c_1 = [-self.y_coeff*base-self.constant, -base, -base]
                    c_2 = [self.y_coeff*base-self.constant, base, base]
            else:
                c_1 = [-self.constant, -base, -base]
                c_2 = [-self.constant, base, base]
        else:
            c_1 = [-base, -self.constant, -base]
            c_2 = [base, -self.constant, base]
        self.bounding_box = AABB(c_1, c_2)

    def __str__(self):
        string="Wall("
        if self.x_coeff != 0:
            string += "x"
        else:
            string += "y"
            if self.constant == 0:
                string += " = 0)"
                return string
            elif self.constant > 0:
                string += " + "+str(self.constant)+" = 0)"
                return string
            else:
                string += " - "+str(abs(self.constant))+" = 0)"
                return string

        if self.y_coeff == 0:
            string += ""
        elif self.y_coeff == 1:
            string += " y"
        elif self.y_coeff > 0:
            string += " + " + str(self.y_coeff) + "y"
        else:
            string += " - " + str(abs(self.y_coeff)) + "y"

        if self.constant == 0:
            string += " = 0)"
            return string
        elif self.constant > 0:
            string += " + " + str(self.constant) + " = 0)"
            return string
        else:
            string += " - " + str(abs(self.constant)) + " = 0)"
            return string

    # Redefined methods

    def shift(self, dist):
        self.constant -= dist
        if self.x_coeff == 0:
            self.anchor = array([0, -self.constant, 0])
        else:
            self.anchor = array([-self.constant, 0, 0])
        self.PARAMS["constant"], self.PARAMS["anchor"] = self.constant, self.anchor
        return self

    def apply_rotate(self, rotation):
        if not isinstance(rotation, Rotation):
            raise TypeError("Rotation must be an instance of Rotation object.")
        self.anchor = rotation.apply(self.anchor)
        if self.x_coeff == 0:
            p_1 = rotation.apply(array([1, -self.constant, 0]))
            p_2 = rotation.apply(array([0, -self.constant, 1]))
        else:
            p_1 = rotation.apply(array([0, -self.constant/self.y_coeff, 0]))
            p_2 = rotation.apply(array([0, -self.constant/self.y_coeff, 1]))
        u = p_1 - self.anchor
        v = p_2 - self.anchor
        normal = cross(u, v)
        c = -float(normal.dot(p_1))
        if normal[0] != 0:
            plane = [normal[1] / normal[0], normal[2] / normal[0], c / normal[0]]
            Plane.__init__(self, plane)
        else:
            Wall.__init__([normal[1], normal[2], c])

def colinear_check(p_1, p_2, p_3):
    u, v = p_2 - p_1, p_3 - p_1
    return isclose(norm(cross(u, v)), 0)

def ismultiple(u, v):
    return isclose(u.dot(v) * u.dot(v), u.dot(u) * v.dot(v))


def plane_from_points(p_1, p_2, p_3, *args):  # Generates a plane object for three given points.
    if "colinear_check" in args:
        test = colinear_check(p_1, p_2, p_3)
        if test is True:
            print("Colinear points detected")
        return None
    u = p_2 - p_1
    v = p_3 - p_1
    normal = cross(u, v)
    c = -float(normal.dot(p_1))
    if normal[0] != 0:
        plane = [normal[1]/normal[0], normal[2]/normal[0], c/normal[0]]
        return Plane(plane)
    else:
        return Wall([normal[1], normal[2], c])

class Triangle(Object):

    def __init__(self, p1, p2, p3):
            self.object_type = "Triangle"
            super().__init__()
            [self.p1, self.p2, self.p3] = [array(i, dtype=float) for i in sorted([list(p) for p in [p1, p2, p3]])]
            self.u = self.p2-self.p1
            self.v = self.p3-self.p1
            if isclose(norm(cross(self.v, self.u)), 0):
                raise ValueError("These points are colinear.")
            self.anchor = (self.p1 + self.p2 + self.p3)/3
            self.unit_normal = self.gen_normal()
            self.PARAMS={"anchor": self.anchor, "vec_1": self.u, "vec_2": self.v, "normal": self.unit_normal}
            mins = [min([self.p1[i], self.p2[i], self.p3[i]]) for i in range(3)]
            maxes = [max([self.p1[i], self.p2[i], self.p3[i]]) for i in range(3)]
            self.bounding_box = AABB(mins, maxes)
            self.temp_hits = None # Used for ray intersection indices in interfunction.

    def __str__(self):
        return "Triangle(p_1:{}, p_2:{}, p_3:{})".format(self.p1, self.p2, self.p3)

    # Transformation methods

    def shift(self, other):
        self.p1 += array(other)
        self.p2 += array(other)
        self.p3 += array(other)
        self.anchor = self.p1
        self.PARAMS["anchor"] = self.anchor
        return self

    def apply_rotation(self, rotation):
        if not isinstance(rotation, Rotation):
            raise TypeError("Rotation must be an instance of Rotation object.")
        self.p1 = rotation.apply(self.p1)
        self.p2 = rotation.apply(self.p2)
        self.p3 = rotation.apply(self.p3)
        self.__init__(self.p1, self.p2, self.p3)

    # Intersection methods
    def ison(self, point):
        q = point-self.anchor
        test_1 = self.u[0]*self.v[1]-self.u[1]*self.v[0]
        if test_1 != 0:
            s = (self.v[1]*q[0]-self.v[0]*q[1])/test_1
            t = (-self.u[1]*q[0]+self.u[0]*q[1])/test_1
            c_test = isclose(self.u[2]*s + self.v[2]*t, q[2])
        else:
            test_2 = self.u[0]*self.v[2]-self.u[2]*self.v[0]
            if test_2 != 0:
                s = (self.v[2]*q[0] - self.v[0]*q[2])/test_2
                t = (-self.u[2]*q[0] + self.u[0]*q[2])/test_2
                c_test = isclose(self.u[1]*s + self.v[1]*t, q[1])
            else:
                test_3 = self.u[1]*self.v[2]-self.u[2]*self.v[1]
                if test_3 != 0:
                    s = (self.v[1]*q[2] - self.v[2]*q[1])/test_3
                    t = (-self.u[1]*q[2] + self.u[2]*q[1])/test_3
                    c_test = isclose(self.u[0]*s + self.v[0]*t, q[0])
                else:
                    return False
        if not c_test:
            return False
        s_test = s >= 0
        t_test = t >= 0
        st_test = s + t <= 1
        return s_test and t_test and st_test

    def plane_from_points(self):  # Generates a plane object for three given points.
        c = -float(self.unit_normal.dot(self.p1))
        if self.unit_normal[0] != 0:
            plane = [self.unit_normal[1] / self.unit_normal[0], self.unit_normal[2] / self.unit_normal[0], c / self.unit_normal[0]]
            return Plane(plane)
        else:
            return Wall([self.unit_normal[1], self.unit_normal[2], c])

    def show_plane(self):
        plane = self.plane_from_points()
        print(plane)

    def interfunction(self, rays, pov):
        rshape = rays.shape[1:]
        rays = rays.reshape((3, rays.shape[1] * rays.shape[2])).T
        epsilon = 10 ** -6
        T = pov - self.anchor
        P = cross(rays, self.v.reshape((1, 3)))
        S = dot(P, self.u) + epsilon
        U = dot(P, T)
        U /= S
        if True in (U >= 0) & (U <= 1):
            Q = cross(T, self.u)
            V = where((U >= 0) & (U <= 1), dot(Q, rays.transpose()), NaN) / S
            t = where((V >= 0) & (V <= 1) & (U + V <= 1), dot(Q, self.v), NaN) / S
            t = where(t <= 0, NaN, t)
            V = V.reshape(rshape)
            U = U.reshape(rshape)
            t = t.reshape(rshape)
            self.temp_hits = array(where(~isnan(t))).T
            return U, V, t
        else:
            return None, None, None

    def intersect(self, rays, pov):
        _, _, t = self.interfunction(rays, pov)
        if t is not None:
            t = where((t <= 0) | isnan(t), False, True)
            return t
        else:
            return full(rays.shape[0], False)

    def intersection_params(self, rays, pov):
        _, _, t = self.interfunction(rays, pov)
        if t is not None:
            return t
        else:
            return full(rays.shape[1:], NaN)

    def intersection_points(self, rays, pov):
        U, V, _ = interfunction(rays, pov)
        if U is not None:
            U = U.reshape((U.shape[0], 1))
            V = V.reshape(U.shape)
            V = where((V >= 0) & (V <= 1) & (U + V <= 1), V, NaN)
            [p_1, p_2, p_3] = [p.reshape((1, 3)) for p in [self.p1, self.p2, self.p3]]
            ps = matmul(1 - U - V, p_1) + matmul(U, p_2) + matmul(V, p_3)
            return ps
        else:
            return full(rays.shape[1:], NaN)

    def area(self):
        return abs(self.u.cross(self.v))/2

    def gen_incident(self, rays):  # Only called after intersection test already made
        print(len(rays))
        if len(rays) == 0:
            return array([])
        prod = abs(dot(rays, expand_dims(self.unit_normal, 0).T))  # angle calculation only calculates absolute value of angle
        incidents = squeeze(arcsin(prod), -1)
        return incidents

    # Normal methods
    def gen_normal(self):
        normal = cross(self.u, self.v)
        normal /= norm(normal)
        if normal[0] > 0:
            return normal
        else:
            normal *= -1
            if normal[0] != 0:
                return normal
            else:
                if normal[1] > 0:
                    return normal
                else:
                    normal *= -1
                    if normal[1] != 0:
                        return normal
                    else:
                        return array([0,0,1])

# TETRAHEDRON
class Tetrahedron(Composite):

    def __init__(self, p1, p2, p3, p4):
        super().__init__()
        self.object_subtype = "Tetrahedron"
        [self.p1, self.p2, self.p3, self.p4] = [array(i, dtype=float) for i in sorted([list(p) for p in [p1,p2,p3,p4]])]
        self.u = self.p2-self.p1
        self.v = self.p3-self.p1
        self.w = self.p4-self.p1
        self.anchor = (self.p1 + self.p2 + self.p3 + self.p4)/4
        self.PARAMS={"anchor": self.anchor, "vec_1": self.u, "vec_2": self.v, "vec_3": self.w}
        self.COMPONENTS={"T1":Triangle(p1,p2,p3),"T2":Triangle(p1,p3,p4),"T3":Triangle(p1,p2,p4),"T4":Triangle(p2,p3,p4)}
        mins = [min([self.p1[i], self.p2[i], self.p3[i], self.p4[i]]) for i in range(3)]
        maxes = [max([self.p1[i], self.p2[i], self.p3[i], self.p4[i]]) for i in range(3)]
        self.bounding_box=AABB(mins,maxes)
        self.component_count = len(self.COMPONENTS)

    def __str__(self):
        return "Tetrahedron(p_1:{}, p_2:{}, p_3:{}, p_4:{})".format(self.p1, self.p2, self.p3, self.p4)

    def apply_rotation(self, rotation):
        if not isinstance(rotation, Rotation):
            raise TypeError("Rotation must be an instance of Rotation object.")
        self.p1 = rotation.apply(self.p1)
        self.p2 = rotation.apply(self.p2)
        self.p3 = rotation.apply(self.p3)
        self.p4 = rotation.apply(self.p4)
        self.__init__(self.p1, self.p2, self.p3, self.p4)


######################################################################

# INTRINSIC FUNCTION

# REQUIREMENTS: The function must take in a vector of length three and perform a calculation
#               with output that is a real number. It should be vectorised to allow for processing
#               many points simultaneously. Formula is a function in x, y and z.

# Note that the Implicit class is not a true Object in terms of geometry, but a starting point for the
# Surface and Mesh classes

# APPLY ROTATION NOT IMPLEMENTED.

class Implicit():

    def __init__(self, funct, formula):
        if not (isfunction(funct) or isbuiltin(funct) or funct.__class__ is partial(print).__class__):
            raise TypeError("Implicit function input must be a function object")
        self.function = funct
        self.formula = formula

    def __str__(self):
        return "Implicit({})".format(self.formula)

    def __eq__(self, other):
        if isinstance(other, Implicit):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # COMPUTATION METHODS

    def compute(self, p):
        return self.function(p)

    def side(self, p):
        return self.compute(p) >= 1

    def is_on(self, p):
        return isclose(self.compute(p), 0)

    # TRANSFORMATION METHODS: All return a new Implicit object rather than modify the existing object

    def union(self, other):
        if not isinstance(other, Implicit):
            raise TypeError("Both objects must be instances of Implicit.")
        def unioned(p):
            return max(self.function(p), other.function(p))
        formula = "max("+self.formula+ ", "+other.formula+")"
        return Implicit(unioned, formula)

    def intersection(self, other):
        if not isinstance(other, Implicit):
            raise TypeError("Both objects must be instances of Implicit.")
        formula = "min(" + self.formula + ", " + other.formula + ")"
        def intersected(p):
            return min(self.function(p), other.function(p))
        return Implicit(intersected, formula)

    # Defines which is inside/outside or orientation of surface
    def negate(self):
        def negated(p):
            return -self.function(p)
        formula = "-("+self.formula+")"
        return Implicit(negated, formula)

    def shift(self, v):
        def shifted(p):
            return self.function(p-vector)
        formula = self.formula.replace("x","(x-{})".format(v[0]))
        formula = formula.replace("y", "(y-{})".format(v[1]))
        formula = formula.replace("z", "(z-{})".format(v[1]))
        return Implicit(shifted, formula)

# Needs to inherit bounding box from Scene.

class Surface(Object):
    def __init__(self, implicit, *args, **kwargs):
        if not isinstance(implicit, Implicit):
            raise TypeError("Input function must be an instance of Implicit class.")
        super().__init__()
        self.object_type = "Surface"
        if "x_range" in kwargs:
            self.x_min = kwargs["x_range"][0]
            self.x_max = kwargs["x_range"][1]
        else:
            self.x_min = -20
            self.x_max = 20

        if "y_range" in kwargs:
            self.y_min = kwargs["y_range"][0]
            self.y_max = kwargs["y_range"][1]
        else:
            self.y_min = -20
            self.y_max = 20

        if "z_range" in kwargs:
            self.z_min = kwargs["z_range"][0]
            self.z_max = kwargs["z_range"][1]
        else:
            self.z_min = -20
            self.z_max = 20

        self.bounding_box = AABB([self.x_max, self.y_max, self.z_max], [self.x_min, self.y_min, self.z_min])
        self.function = implicit.function
        self.implicit = implicit_funct
        self.convergence_constant = 0.0001
        self.name = implicit_funct.formula
        self.norm_funct = vectorize(self.normal_funct(*args)) # Does this vectorise help?

    def __str__(self):
        return "Surface({})".format(self.name)

    # REPACKAGED IMPLICIT METHODS

    def side(self, p):
        return self.implicit.side(p)

    def in_range(self, p):
        return self.bounding_box.is_in(p)

    def is_on(self, p):
        return self.implicit.is_on(p)

    # We want a function that takes in a single vector and can be vectorised

    def new_function(self, x, y, z, *args, **kwargs):
        def new(x, y, z):
            v = hstack

    def intersection_params(self, rays, centre, *args, **kwargs):
        def create_loss_funct(rays, centre, *args, **kwargs):
            dir_x, dir_y, dir_z = array(rays), array(ray.direction.y), array(ray.direction.z)
            orig_x, orig_y, orig_z = ray.origin.x, ray.origin.y, ray.origin.z
            def loss_func(t, *args, **kwargs):
                return abs(implicit(dir_x * t + orig_x, dir_y * t + orig_y, dir_z * t + orig_z, *args, **kwargs))
            return loss_func
        loss_funct = create_loss_funct(self.function,ray)
        param = minimize(loss_funct, x0=0, method='Nelder-Mead', tol=self.convergence_constant)
        return param

    # Normal methods

    def normal_funct(self, *args):
        dfdx, dfdy, dfdz = grad(self.function, 0), grad(self.function, 1), grad(self.function, 2)
        return dfdx, dfdy, dfdz

    def gen_normal(self,point,*args):
        return array([d(point) for d in self.normal_funct(*args)]).normalise()

    def gen_normals(self, points, *args):
        if points.__class__ is list:
            gradient = self.normal_funct(*args)
            normals = []
            for p in points:
                if p != None:
                    if "test_point" in args:
                        if not self.ison(p):
                            normals.append(None)
                        else:
                            normals.append(array([d(p) for d in gradient]))
                    else:
                        normals.append(array([d(p) for d in gradient]))
                else:
                    normals.append(None)
            return normals
        elif points.__class__ == array(0).__class__:
            gradient = self.normal_funct(*args)     # Need to apply the gradient to an array of shape (n,m,3)
            n, m = points.shape[:2]
            normals = zeros(points.shape)
            for i in range(n):
                for j in range(m):
                    normals[i, j, :] = array([d(points[i, j, 0], points[i, j, 1], points[i, j, 2]) for d in gradient])
            return normals
        else:
            raise ValueError("Points must be in list or array form.")

# TESSALATED SURFACE FROM FUNCTION

# Extracts values of points in a grid according to an implicit function.
#
# * Value of 0 is on surface.
# * Negative is "inside" or on one side of surface.
# * Positive is "outside" or on another side of surface

# Bounds is a list of three tuples/lists that are the min/max in the x, y, z directions

def array_from_implicit(implicit, sample_width, bounds):
    if not isinstance(implicit, Implicit):
        raise TypeError("Input must be an instance of Implicit.")
    # Create sample
    [n, m, k] = [int((bounds[i][1]-bounds[i][0])//sample_width) for i in range(3)]
    [x_range, y_range, z_range] = [[bounds[i][0], bounds[i][0]+([n, m, k][i]+1)*sample_width] for i in range(3)]
    sample_array = zeros((n+1, m+1, k+1))
    for i in range(n+1):
        for j in range(m+1):
            for l in range(k+1):
                sample_array[i, j, l] = implicit.function(x_range[0]+sample_width*i, y_range[0]+sample_width*j, z_range[0]+sample_width*l)
    return sample_array

# Extract from function

# Inputs: x_range and y_range is an iterable. n is an integer.
# Optional m defines intervals in y_range. Default is square divisions.

def array_from_explicit(funct, x_range, y_range, n, **kwargs):
    grid_width = (x_range[1]-x_range[0])/n
    if "m" in kwargs:
        m = kwargs["m"]
        grid_height = (y_range[1]-y_range[0])/kwargs["m"]
    else:
        m = ceil((y_range[1]-y_range[0])/grid_width)
        y_range = (y_range[0], y_range[0]+grid_width*m)
        grid_height = grid_width
    points = zeros((n+1, m+1, 3))+[x_range[0], y_range[0], 0]
    for i in range(n+1):
        for j in range(m+1):
            points[i, j] += [i*grid_width, j*grid_height, funct(x_range[0]+i*grid_width, y_range[0]+j*grid_height)]
    return points

def scale_sample(funct, x_range, y_range, levels, low=1):
    if not isinstance(levels, int) or low > levels:
        raise ValueError("Subdivision levels must be higher than {}.".format(low))
    sampled_arrays = []
    for i in range(low, levels+low):
        sampled_arrays.append(array_from_explicit(funct, x_range, y_range, 2**i))
    return sampled_arrays

# Input is xyz array, which is a grid sample of the function containing the height map of the function. It must be an array.

class Tesselation(Composite):

    def __init__(self, xyz_array, *args):
        super().__init__()
        self.funct_array = xyz_array
        self.grid_dims = (xyz_array.shape[0]-1, xyz_array.shape[1]-1)
        self.object_subtype = "Tesselation"
        count = 1
        for i in range(self.grid_dims[0]):
            for j in range(self.grid_dims[1]):
                p1 = self.funct_array[i, j]
                p2 = self.funct_array[i+1, j]
                p3 = self.funct_array[i, j+1]
                p4 = self.funct_array[i+1, j+1]
                if "down" in args:
                    self.COMPONENTS["Tess_{}".format(count)] = Triangle(p1, p2, p4)
                    count += 1
                    self.COMPONENTS["Tess_{}".format(count)] = Triangle(p1, p3, p4)
                    count += 1
                elif "random" in args:
                    if random() > 0.5:
                        self.COMPONENTS["Tess_{}".format(count)] = Triangle(p1, p2, p4)
                        count += 1
                        self.COMPONENTS["Tess_{}".format(count)] = Triangle(p1, p3, p4)
                        count += 1
                    else:
                        self.COMPONENTS["Tess_{}".format(count)] = Triangle(p1, p2, p3)
                        count += 1
                        self.COMPONENTS["Tess_{}".format(count)] = Triangle(p2, p3, p4)
                        count += 1
                else:
                    self.COMPONENTS["Tess_{}".format(count)] = Triangle(p1, p2, p3)
                    count += 1
                    self.COMPONENTS["Tess_{}".format(count)] = Triangle(p2, p3, p4)
                    count += 1

        self.t_number = count-1
        mins = amin(self.funct_array, (0, 1))
        maxes = amax(self.funct_array, (0, 1))
        self.bounding_box = AABB(mins, maxes)
        self.anchor = (maxes - mins)/2 + mins
        self.component_count = len(self.COMPONENTS)

    def __str__(self):
        return "Tesselation(Count: {})".format(self.t_number)

    def apply_rotation(self, rotation):
        if not isinstance(rotation, Rotation):
            raise TypeError("Rotation must be an instance of Rotation object.")
        if not all(rotation.centre == self.anchor):
            print("Rotation around point other than bounding box centre.")
        self.funct_array = rotation.apply(self.funct_array)
        mins = amin(self.funct_array, (0, 1))
        maxes = amax(self.funct_array, (0, 1))
        self.bounding_box = AABB(mins, maxes)
        return self

def nested_tesselations(funct, x_range, y_range, levels, low=1, **kwargs):
    samples = scale_sample(funct, x_range, y_range, levels, low)
    if "name" in kwargs:
        stem = kwargs["name"]
    else:
        stem = "unnamed_function"
    tesselations = {}
    for s in samples:
        t = Tesselation(s)
        tesselations[stem+"_{}".format(t.component_count)] = t
    return tesselations

# This lookup function: What is it's purpose?

def function_from_samples(xy_coords, z_grid):

    if len(xy_coords) != len(z_grid):
        ValueError("Number of inputs is not the same as the number of outputs.")
    elif len(set(xy_coords)) != len(z_grid):
        ValueError("Multiple outputs for at least one input")

    def lookup_funct(x_coord, y_coord):
        xy_coord=(x_coord, y_coord)
        return z_grid[xy_coords.index(xy_coord)]

    return lookup_funct

# OBJECT FILES

# ObJFile is not an Object subclass, but rather a reader class to interface with files and then create a Mesh instance

class ObjFile:

    def __init__(self, file):
        if not isinstance(file, str):
            raise TypeError("Input must be a string.")
        elif file[-4:] != ".obj":
            raise ValueError("Not an OBJ file.")
        self.file_path = file
        f = open(self.file_path, "r")
        self.vertices = []
        self.faces = []
        for line in f:
            if line[0] == "v":
                v = [float(p) for p in line[2:].split()]
                self.vertices.append(v)
            elif line[0] == "f":
                h = [int(float(e.split("/")[0])) for e in line[2:].split()]
                self.faces.append(h)
            else:
                pass
        f.close()
        self.vertex_count = len(self.vertices)
        self.face_count = len(self.faces)

    def __str__(self):
        return "Obj(Vertices:{}, Faces:{})".format(self.vertex_count,self.face_count)

    def __repr__(self):
        string = "VERTICES\n"
        for v in self.vertices:
            string += str(v)
            string += "\n"
        string += "\nFACES\n"
        for f in self.faces:
            string += str(f)
            string += "\n"
        return string

    def info(self):
        print(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        self.merge(other)
        return self

    def __radd__(self, other):
        self.merge(other)
        return self

    def __iadd__(self, other):
        self.merge(other)
        return self

    def copy(self, **kwargs):
        kwargs["file_path"] = copy(self.file_path[:-4])+"_copy.obj"
        self.write_obj(**kwargs)
        return ObjFile(kwargs["file_path"])

    # Returns index of face with labeled indices. Canonical order?
    def face_index(self, face_desc):
        if face_desc in self.faces:
            return self.faces.index(face_desc)
        else:
            print("Face not in object. Check for typos.")
            return None

    def remove_face(self, index):
        self.faces.pop(index)

    # LOOKS UP VERTEX DESCRIPTOR (3D COORDINATE)
    def vertex_index(self, vertex_desc):
        if vertex_desc in self.faces:
            return self.vertices.index(vertex_desc)
        else:
            print("Vertex not in object. Check for typos.")
            return None

    def remove_vertex(self, index):
        self.vertices.pop(index)
        self.faces = [f for f in self.faces if index + 1 not in f]
        self.faces = [[v-1 if v > index else v for v in f] for f in self.faces]

    # Write OBJ file and add optional comments.
    def write_obj(self, **kwargs):
        if "file_path" in kwargs:
            self.file_path = kwargs["file_path"]
        else:
            raise ValueError("file_path keyword argument must be supplied.")
        f = open(self.file_path, "w")
        if "comments" in kwargs:
            f.write("# "+str(kwargs["comments"])+"\n\n")
        f.write("# VERTICES\n")
        for v in self.vertices:
            f.write("v")
            for p in v:
                f.write(" "+str(p))
            f.write("\n")
        f.write("\n")
        f.write("# FACES\n")
        for face in self.faces:
            f.write("f")
            for v in face:
                f.write(" "+str(v))
            f.write("\n")
        f.close()

    # ASSUME POSITION IS A LIST OF THREE FLOATS
    def move_vertex(self, index, new_pos):
        self.vertices[index] = new_pos

    def merge(self, other, **kwargs):
        if isinstance(other, ObjFile):
            raise TypeError("Only two ObjFile instances can be merged.")
        else:
            new_verts = list(other.vertices)
            new_faces = [[c+self.vertex_count for c in f] for f in other.faces]
            self.vertices += new_verts
            self.faces += new_faces
            self.vertex_count += other.vertex_count
            self.face_count += other.face_count
        if "file_path" in kwargs:
            self.write_obj(**kwargs)
        return self

    def scale(self, factor):
        self.vertices = [[c*factor for c in v] for v in self.vertices]
        return self

    def shift(self, vector):
        vert_array = array(self.vertices)
        self.vertices = list(vert_array + vector)
        return self

    def apply_rotation(self, rotation):
        if not isinstance(rotation, Rotation):
            raise TypeError("Rotation must be an instance or Rotation class.")
        self.vertices = list(rotation.apply(array(self.vertices)))

# OBJ CONVERSION TOOLS

def tess_to_OBJ(tess, file_path, **kwargs):
    if not isinstance(tess, Tesselation):
        raise TypeError("Input needs to be a Tesselation instance.")
    else:
        file = open(file_path, "w")
        count = 0
        vertices = {}
        faces = []
        # Create lists of vertices and faces
        for c in tess.COMPONENTS:
            ps = [tess.COMPONENTS[c].p1, tess.COMPONENTS[c].p2, tess.COMPONENTS[c].p3]
            for p in ps:
                if str(p) not in vertices:
                    vertices[str(p)] = [count+1, p]
                    count += 1
        for c in tess.COMPONENTS:
            ps = [str(tess.COMPONENTS[c].p1), str(tess.COMPONENTS[c].p2), str(tess.COMPONENTS[c].p3)]
            faces.append([vertices[p][0] for p in ps])
        # Write to .obj file
        if "comments" in kwargs:
            file.write("#" + kwargs["comments"]+"\n\n")
        file.write("#VERTICES\n\n")
        for v in vertices:
            file.write("v "+str(vertices[v][1][0])+" "+str(vertices[v][1][1])+" "+str(vertices[v][1][2])+"\n")
        file.write("\n#FACES\n\n")
        for f in faces:
            file.write("f "+str(f[0])+" "+str(f[1])+" "+str(f[2])+"\n")
        file.close()
    return ObjFile(file_path)

def nested_sample_to_OBJ(funct, x_range, y_range, levels, low=1, **kwargs):
    nt = nested_tesselations(funct, x_range, y_range, levels, low, **kwargs)
    obs = {}
    if "save_dir" in kwargs:
        root = kwargs["save_dir"]
    else:
        root = getcwd()
    for t in nt:
        obs[t] = tess_to_OBJ(nt[t], join(root, t+".obj"))

def implicit_to_OBJ(implicit, sample_width, bounds, file_path, *args, **kwargs):
    if isinstance(implicit, Implicit):
        raise TypeError("Input must be an instance of Implicit.")
    else:
        # CREATE ARRAY FROM FUNCTION
        sample_array = array_from_implicit(implicit, sample_width, bounds)
        verts, faces, _, _ = marching_cubes(sample_array, 0)
        verts *= sample_width
        verts += array([bounds[i][0] for i in range(3)])

        # REMOVE DUPLICATES
        unique_verts = []
        renewed_faces = faces.copy()
        counter = 0
        for v in verts:
            if list(v) not in unique_verts:
                unique_verts.append(list(v))
                counter += 1
            else:
                previous = unique_verts.index(list(v))
                for f in range(len(faces)):
                    for c in range(3):
                        if faces[f][c] > counter:
                            renewed_faces[f][c] -= 1
                        if faces[f][c] == counter:
                            renewed_faces[f][c] = previous
                counter += 1
        faces = [[v+1 for v in f] for f in renewed_faces if len(set(f))==3]
        max_vertex = len(unique_verts)
        faces = [f for f in faces if True not in [f[v] > max_vertex for v in range(3)]]
        # WRITE TO OBJ FILE
        f = open(file_path, "w")
        if "comments" in kwargs:
            f.write("#" + str(kwargs["comments"]) + " " + implicit.formula + "\n\n")
        f.write("#VERTICES\n")
        for v in unique_verts:
            f.write("v")
            for p in v:
                f.write(" " + str(p))
            f.write("\n")
        f.write("\n")
        f.write("#FACES\n")
        for face in faces:
            f.write("f")
            for v in face:
                f.write(" " + str(v))
            f.write("\n")
        f.close()
    implicit_obj = ObjFile(file_path)

    # SHOW PLOT
    if "show" in args:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        xyz_centre = [(bounds[i][1]+bounds[i][0])/2 for i in range(3)]
        xyz_spans = [1.1*(bounds[i][1]-bounds[i][0])/2 for i in range(3)]
        cube_width = max(xyz_spans)

        ax.set_xlim(xyz_centre[0]-cube_width, xyz_centre[0]+cube_width)
        ax.set_ylim(xyz_centre[0]-cube_width, xyz_centre[0]-cube_width)
        ax.set_zlim(xyz_centre[0]-cube_width, xyz_centre[0]-cube_width)

        plt.tight_layout()
        plt.show()

    return implicit_obj

def funct_to_OBJ(funct, ftype, path, sample_width, bounds, *args, **kwargs):
    if ftype not in ["intrinsic", "explicit"]:
        raise ValueError("Second variable 'ftype' must have value 'explicit' or 'intrinsic'. Current value is {}.".format(ftype))
    else:
        if ftype == "explicit":
            n = int((bounds[0][1] - bounds[0][0]) // sample_width)
            x_range = (bounds[0][0], bounds[0][0] + n * sample_width)
            y_range = (bounds[1][0], bounds[1][1])
            if "formula" in kwargs:
                if "comments" in kwargs:
                    kwargs["comments"] += " Formula: " + kwargs["formula"]
                else:
                    kwargs["comments"] = " Formula: " + kwargs["formula"]
                return tess_to_OBJ(Tesselation(array_from_explicit(funct, x_range, y_range, n),*args),
                                   path,
                                   **kwargs)
            else:
                return tess_to_OBJ(Tesselation(array_from_explicit(funct, x_range, y_range, n), *args),
                                   path,
                                   **kwargs)
        else:
            if "formula" in kwargs:
                if "comments" in kwargs:
                    kwargs["comments"] += " Formula: " + kwargs["formula"]
                else:
                    kwargs["comments"] = " Formula: " + kwargs["formula"]
                return implicit_to_OBJ(Implicit(funct, kwargs["formula"]),
                                       sample_width,
                                       bounds,
                                       path,
                                       *args,
                                       **kwargs)
            else:
                return implicit_to_OBJ(Implicit(funct, "custom_function", kwargs["file_path"], *args, **kwargs),
                                       sample_width,
                                       bounds,
                                       path
                                       *args,
                                       **kwargs)

# Takes an array of xyz format and converts to OBJ

def array_to_OBJ(sample_array, *args, **kwargs):
    if "file_path" not in kwargs:
        raise ValueError('Keyword "file_path" must be defined.')
    else:
        return tess_to_OBJ(Tesselation(sample_array, *args), **kwargs)

# MESH

class Mesh(Composite):
    def __init__(self, obj, *args, **kwargs):
        if isinstance(obj, str):
            self.OBJ = ObjFile(obj)
        else:
            if "name" in kwargs:
                fname = kwargs["name"]
            else:
                fname = "mesh_object.obj"
            if "save_dir" in kwargs:
                path = join(kwargs["save_dir"], fname)
            else:
                path = join(getcwd(), fname)
        if isinstance(obj, Tesselation):
            self.OBJ = tess_to_OBJ(obj, path,  **kwargs)
        elif isinstance(obj, Implicit):
            if "sample_width" not in kwargs:
                raise ValueError("Keyword argument 'grid_size' must be supplied.")
            if "bounds" not in kwargs:
                raise ValueError("Keyword argument 'bounds' must be supplied.")
            self.OBJ = implicit_to_OBJ(obj, kwargs["sample_width"], kwargs["bounds"], path, *args, **kwargs)
        elif isinstance(obj, ObjFile):
            self.OBJ = obj
        elif (isfunction(obj) or isbuiltin(obj) or obj.__class__ is partial(print).__class__):
            if "sample_width" not in kwargs:
                raise ValueError("Keyword argument 'grid_size' must be supplied.")
            if "bounds" not in kwargs:
                raise ValueError("Keyword argument 'bounds' must be supplied.")
            if "ftype" not in kwargs:
                raise ValueError("Keyword argument 'ftype' must be supplied.")
            self.OBJ = funct_to_OBJ(obj, kwargs["ftype"], path, kwargs["sample_width"], kwargs["bounds"], *args, **kwargs)
        else:
            raise TypeError("Input is not of an accepted type. Should be one of string, Implicit, ObjFile, Tesselation.")
        super().__init__(**kwargs)
        self.object_subtype = "Mesh"
        self.mesh = None
        self.vertex_count = self.OBJ.vertex_count
        self.component_count = 0
        self.COMPONENTS = {}
        for f in self.OBJ.faces:
            self.component_count += 1
            [p1, p2, p3] = [array(self.OBJ.vertices[f[i]-1]) for i in range(3)]  # Looks up index for each vertex in face edge. Note faces index vertices from 1 in standard OBJ format.
            self.COMPONENTS["Face_" + str(self.component_count)] = Triangle(p1, p2, p3)
        self.anchor = array(self.OBJ.vertices[0])
        self.PARAMS["location"] = self.anchor
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]] = [self.min_max([v[i] for v in self.OBJ.vertices]) for i in range(3)]
        c_1 = [min_x, min_y, min_z]
        c_2 = [max_x, max_y, max_z]
        self.bounding_box = AABB(c_1, c_2)

    def min_max(lst):
        return [min(lst), max(lst)]

    def gen_mesh(self):
        self.mesh = Poly3DCollection(array(self.OBJ.vertices)[array(self.OBJ.faces) - 1])

    def apply_rotation(self, rotation, *args):
        super().apply_rotation(rotation)
        if "update" in args:
            self.OBJ.vertices = rotation.apply(self.OBJ.vertices)
            self.mesh = Poly3DCollection(array(self.OBJ.vertices)[array(self.OBJ.faces) - 1])
            [[min_x, max_x], [min_y, max_y], [min_z, max_z]] = [self.min_max([v[i] for v in self.OBJ.vertices]) for i in range(3)]
            c_1 = [min_x, min_y, min_z]
            c_2 = [max_x, max_y, max_z]
            self.bounding_box = AABB(c_1, c_2)
        return self

    def shift(self, vec):
        super().shift(vec)
        self.OBJ.vertices += vec
        self.mesh = Poly3DCollection(array(self.OBJ.vertices)[array(self.OBJ.faces) - 1])
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]] = [self.min_max([v[i] for v in self.OBJ.vertices]) for i in range(3)]
        c_1 = [min_x, min_y, min_z]
        c_2 = [max_x, max_y, max_z]
        self.bounding_box = AABB(c_1, c_2)
        return self

    def show(self, **kwargs):
        if self.mesh is None:
            print("Viewable mesh not created. Call gen_mesh")
        else:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            self.mesh.set_edgecolor('k')
            ax.add_collection3d(self.mesh)

            x_mean = (self.bounding_box.x_min + self.bounding_box.x_max)/2
            y_mean = (self.bounding_box.y_min + self.bounding_box.y_max)/2
            z_mean = (self.bounding_box.z_min + self.bounding_box.z_max)/2

            x_span = 1.1*(-self.bounding_box.x_min + self.bounding_box.x_max)/2
            y_span = 1.1*(-self.bounding_box.y_min + self.bounding_box.y_max)/2
            z_span = 1.1*(-self.bounding_box.z_min + self.bounding_box.z_max)/2

            max_span = max([x_span, y_span, z_span])

            ax.set_xlim(x_mean-max_span, x_mean+max_span)
            ax.set_ylim(y_mean-max_span, y_mean+max_span)
            ax.set_zlim(z_mean-max_span, z_mean+max_span)
            plt.tight_layout()
            if "im_path" in kwargs:
                plt.savefig(kwargs["im_path"])
            plt.show()

    def save_im(self, im_path):
        if self.mesh is None:
            self.mesh = Poly3DCollection(array(self.OBJ.vertices)[array(self.OBJ.faces)-1])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        self.mesh.set_edgecolor('k')
        ax.add_collection3d(self.mesh)

        x_mean = (self.bounding_box.x_min + self.bounding_box.x_max)/2
        y_mean = (self.bounding_box.y_min + self.bounding_box.y_max)/2
        z_mean = (self.bounding_box.z_min + self.bounding_box.z_max)/2

        x_span = 1.1*(-self.bounding_box.x_min + self.bounding_box.x_max)/2
        y_span = 1.1*(-self.bounding_box.y_min + self.bounding_box.y_max)/2
        z_span = 1.1*(-self.bounding_box.z_min + self.bounding_box.z_max)/2

        max_span=max([x_span, y_span, z_span])

        ax.set_xlim(x_mean-max_span, x_mean+max_span)
        ax.set_ylim(y_mean-max_span, y_mean+max_span)
        ax.set_zlim(z_mean-max_span, z_mean+max_span)
        plt.tight_layout()
        plt.savefig(im_path)
        plt.close(fig)

    def write(self, **kwargs):
        self.OBJ.write_obj(**kwargs)

# CLASSES AND TRANSFORMATIONS FOR MESHES

class ObjTransform:

    def __init__(self, trans,**kwargs):
        if not (isfunction(trans) or isbuiltin(trans) or trans.__class__ is partial(print).__class__): # What is this?!?
            raise TypeError("Transformation must be an instance of function or function-like class")
        sig = signature(trans)
        self.params = sig.parameters
        self.trans = trans
        if "trans_limits" not in kwargs:
            self.domain = [None, None, None]
        else:
            if not isinstance(kwargs["trans_limits"], list):
                raise TypeError("Domain of the transformation must be a list of tuples of length 2 or None. This is not a list.")
            elif len(kwargs["trans_limits"]) != 3:
                raise TypeError("Domain of the transformation must be a list of tuples of length 2 or None. Length of list must be 3.")
            elif False in [b.__class__ in [list, tuple, None.__class__] for b in kwargs["trans_limits"]]:
                raise TypeError("Domain of the transformation must be a list of tuples of length 2 or None. One of the entries is of incorrect type.")
            else:
                self.domain = kwargs["trans_limits"]
        if "name" in kwargs:
            self.name = kwargs["name"]
        else:
            self.name = "Unnamed Transformation"
        self.trans.__name__ = self.name

    def apply_to_point(self, point, *args, **kwargs):
        new_point = self.apply_to_array(array([point]), *args, **kwargs)[0]
        return new_point

    def apply_to_array(self, input, *args, **kwargs):
        print(len(input.shape),input.shape)
        if len(input.shape)==1:
            return self.trans(input[0],input[1],input[2],*args,**kwargs)
        else:
            print(len(input.shape), input.shape,input[:,0])
            print("Many points")
            return self.trans(list(input[:,0]),list(input[:,1]),list(input[:,2]),*args,**kwargs)

    def apply_to_obj(self, obj, **kwargs):
        if not isinstance(obj, ObjFile):
            raise TypeError("Input must be of class ObjFile.")
        obj.vertices = list(self.apply_to_array(array(obj.vertices)))
        if "save_trans" in kwargs:
            obj.write_obj(kwargs["save_trans"])
        return obj

    def apply_to_mesh(self, mesh, **kwargs):
        if not isinstance(obj, Mesh):
            raise TypeError("Input must be of class Mesh.")
        mesh.OBJ.vertices = list(self.apply_to_array(array(mesh.OBJ.vertices)))
        mesh.mesh = Poly3DCollection(array(mesh.OBJ.vertices)[array(mesh.OBJ.faces)-1])
        mesh.component_count = 0
        for f in mesh.OBJ.faces:
            mesh.component_count += 1
            [p1, p2, p3] = [array(mesh.OBJ.vertices[f[i]-1]) for i in range(3)]  # Looks up index for each vertex in face edge. Note faces index vertices from 1 in standard OBJ format.
            mesh.COMPONENTS["Face_" + str(mesh.component_count)] = Triangle(p1, p2, p3)
        mesh.anchor = array(mesh.OBJ.vertices[0])
        mesh.PARAMS["location"] = mesh.anchor
        def min_max(lst):
            return [min(lst), max(lst)]
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]] = [min_max([v[i] for v in mesh.OBJ.vertices]) for i in range(3)]
        c_1 = [min_x, min_y, min_z]
        c_2 = [max_x, max_y, max_z]
        mesh.bounding_box = AABB(c_1, c_2)
        if "save_trans" in kwargs:
            mesh.OBJ.write_obj(kwargs["save_trans"])
        return mesh


    def merge(self,other):
        if other.__class__ is not ObjTransform:
            raise TypeError("Only two MeshTransform objects can be combined.")

    def remove_duplicates(self, mesh):
        pass

def translate(vector):  # vector must be an array
    def fixed_trans(pts):
        return pts + vector
    return fixed_trans

# THE FOLLOWING FRAMEWORKS GENERATE MESH OBJECTS, BUT ARE NOT

# TYRES/SQUARED TORI

# r = radius
# a = horizontal thickness
# b = vertical thickness
# s = squareness factor
# phi = angle of tilt
# theta = rotation about z axis


# ROTATION OBJECT NEEDS FINISHED AND CHANGED IN FOLLOWING CODE

# This allows for iteration of orientations of a given tyre shape.

def tyre_by_param(r, a, b, s, c, phi, theta):

    for i in [phi, theta]:
        if not isinstance(i, list):
            i = [i]
    s += 2
    if phi == 0:
        rot_1s = [0]
        rot_2s = [0]
    else:
        rot_1s = [0]
        for p in phi:
            rot_1s.append(Rotation(Angle(0),Angle(p), Angle(0))) #
    if theta == 0:
        rot_2s = [None]
    else:
        if phi != 0:
            rot_2s = []
            for t in theta:
                rot_2s.append(Rotation(Angle(t), Angle(0), Angle(0)))
    [x_c, y_c, z_c] = c
    def tyre(x,y,z):
        return (abs(z - z_c)/a)**s + (abs(((x - x_c)**2 + (y - y_c)**2)**0.5 - r)/b) ** s - 1
    formula = "Tyre(r={} a={} b={} s={} c={}".format(r, a, b, s-2, c)
    radial, vertical = (r+b)*1.8, a*1.8
    spacing = 2*min([a,b])/3
    bounds = [[x_c - radial, x_c + radial], 
              [y_c - radial, y_c + radial],
              [z_c - vertical, z_c + vertical]]
    implicit = Implicit(tyre, formula=formula)
    # Create default object mesh
    comments = "Squared torus of radius {}, radial width {}, depth {}, squareness factor of {}, " \
               "centred at {}. Phi is 0. Theta is 0.".format(r, 2*a, 2*b, s - 2, c)
    tyre_obj = implicit_to_OBJ(implicit, spacing, bounds,
                               "r={} a={} b={} s={} c={} rot=(0,0,0).obj".format(r, a, b, s - 2, c),
                               comments=comments)
    tyre_mesh = Mesh(tyre_obj)
    tyre_mesh.save_im(formula + ".jpg") # Base object
    if phi == 0 or phi is None:
        return tyre_mesh
    tyre_mesh.rotate_object(rot_1s[0])
    tyre_mesh.rotate_object(rot_2s[0])
    comments = "Squared torus of radius {}, ring width {}, depth {}, squareness factor of {}, " \
               "centred at {}. Phi is {}. Theta is {}.".format(r, 2*a, 2*b, s-2, c, phi[0], theta[0])
    file_path = "r={} a={} b={} s={} c={} rot=({},{},0).obj".format(r, a, b, s-2, c, theta[0], phi[0])
    tyre_mesh.write(file_path=file_path, comments=comments)
    tyre_mesh.save_im(file_path[:-4] + ".jpg") # First phi and theta combination
    pre_theta = rot_2s[0]
    pre_phi = rot_1s[0]
    # All thetas for first phi
    for j in rot_2s[1:]:
        rel_rot_2 = j + (-pre_theta)
        tyre_mesh.rotate_object(rel_rot_2)
        comments = "Squared torus of radius {}, ring width {}, depth {}, squareness factor of {}, " \
                   "centred at {}. Phi is {}. Theta is {}.".format(r, 2*a, 2*b, s-2, c, phi[0], j)
        file_path = "r={} a={} b={} s={} c={} rot=({},{},0).obj".format(r, a, b, s-2, c, j.theta, phi[0])
        tyre_mesh.write(file_path=file_path, comments=comments)
        tyre_mesh.save_im(file_path[:-4]+".jpg")  #  All remaining thetas for first phi
        pre_theta = j
    # Repeat for remaining phis. We need to reset the tyre to the flat position.
    tyre_mesh.reset_rotation()
    for i in rot_1s[1:]:
        tyre_mesh.rotate_object(i)
        tyre_mesh.rotate_object(rot_2s[0])  # Do for the first value of theta of for new phi
        comments = "Squared torus of radius {}, ring width {}, depth {}, squareness factor of {}, " \
                   "centred at {}. Phi is {}. Theta is {}.".format(r, 2 * a, 2 * b, s - 2, c,
                                                                           i.phi, theta[0])
        file_path = "r={} a={} b={} s={} c={} rot=({},{},0).obj".format(r, a, b, s - 2, c, theta[0], i.phi)
        tyre_mesh.write(file_path=file_path, comments=comments)
        tyre_mesh.save_im(file_path[:-4]+".jpg") # Next phi, first theta
        pre_phi = i
        pre_theta = rot_2s[0]
        for j in rot_2s[1:]:
            rel_rot_2=j+(-pre_theta)
            tyre_mesh.rotate_object(rel_rot_2)
            comments = "Squared torus of radius {}, ring width {}, depth {}, squareness factor of {}, " \
                       "centred at {}. Phi is {}. Theta is {}.".format(r, 2 * a, 2 * b, s - 2, c,
                                                                               i.phi, j.theta)
            file_path = "r={} a={} b={} s={} c={} rot=({},{},0).obj".format(r, a, b, s - 2, c, j.theta, i.phi)
            tyre_mesh.write(file_path=file_path, comments=comments)
            tyre_mesh.save_im(file_path[:-4] + ".jpg") # Remaining values of theta for phi
            pre_theta=j
        # reset to flat position each time
        tyre_mesh.rotate_object(-pre_theta)
        tyre_mesh.rotate_object(-pre_phi)
    return tyre_mesh

# FRACTAL SURFACE TOOLS

def fractal_noise(scale, iter, fract_dim):
    sd = scale*exp(-iter*fract_dim)
    return norm(0, sd)

def fractal_array(detail_level, scale, fractal_dim, bounds, *args, **kwargs):
    if not isinstance(detail_level, int):
        raise TypeError("Detail level must be a positive integer.")
    else:
        if detail_level < 1:
            raise TypeError("Detail level must be a positive integer.")
    size = 2**(detail_level+1)+1
    xs = [bounds[0][0]+i*(bounds[0][1]-bounds[0][0])/(size-1) for i in range(size)]
    x_grid = multiply(xs, ones((size, size)))
    ys=[bounds[1][0]+i*(bounds[1][1]-bounds[1][0])/(size-1) for i in range(size)]
    y_grid = multiply(ys, ones((size, size))).transpose()
    z_grid=zeros((size, size))
    if "seeds" in kwargs:
        seeds = kwargs["seeds"]
        if not isinstance(seeds, list):
            raise TypeError("Seeds keyword value be a list of values.")
        else:
            if len(seeds) !=4 :
                raise ValueError("Seeds keyword value must be a list of length 4.")
    else:
        s = 1
        seeds = [fractal_noise(scale, s, fractal_dim, *args, **kwargs) for i in range(4)]
    z_grid[0, 0], z_grid[0, -1], z_grid[-1, 0], z_grid[-1, -1] = seeds[0], seeds[1], seeds[2], seeds[3]
    current_width=size-1
    while current_width>1:
        half=current_width//2
        # SQUARE
        for x in prange(0, size - 1, current_width):
            for y in prange(0, size - 1, current_width):
                corner_sum = z_grid[x][y] + \
                            z_grid[x + current_width][y] + \
                            z_grid[x][y + current_width] + \
                            z_grid[x + current_width][y + current_width]
                avg = corner_sum / 4
                avg += fractal_noise(scale,s,fractal_dim)
                z_grid[x + half][y + half] = avg
        # DIAMOND
        for x in range(0, size , half):
            for y in range((x + half) % current_width, size, current_width):
                avg = z_grid[(x - half + size - 1) % (size - 1)][y] + \
                      z_grid[(x + half) % (size - 1)][y] + \
                      z_grid[x][(y + half) % (size - 1)] + \
                      z_grid[x][(y - half + size - 1) % (size - 1)]

                avg /= 4.0
                avg += fractal_noise(scale,s,fractal_dim)

                z_grid[x][y] = avg
        s+=1
        current_width=max(current_width//2,1)
    if "level" in kwargs:
        z_grid += kwargs["level"]
    frarray = stack([x_grid,y_grid,z_grid],axis=-1)
    if "preview" in args:
        plot_3d_function(xs,ys, z_grid,*args,**kwargs)
    return frarray



# CONVEX HULLS

def unif_sphere(radius,centre,n):
    if centre.__class__ is not SVec:
        raise TypeError("Centre must be an instance of SVec.")
    thetas = uniform(0,2*pi,(n,1))
    phis = arccos(1-uniform(0,2,(n,1)))
    xs = sin(phis)*cos(thetas)
    ys = sin(phis)*sin(thetas)
    zs = cos(phis)
    points = radius*(concatenate([xs,ys,zs],axis=-1)+centre.value)
    return points

def convex(radius,centre,n,*args,**kwargs):
    points = unif_sphere(radius,centre,n)
    chull = ConvexHull(points)
    points, vertices, faces = list(points), list(chull.vertices), list(chull.simplices+1)
    ID = identifier(6)
    path="ConvexHull ({}).obj".format(ID)
    obj_f=open(path,"w")
    if "comments" in kwargs:
        obj_f.write("# "+str(kwargs("comments"))+"\n\n")
    obj_f.write("# VERTICES\n\n")
    for v in vertices:
        obj_f.write("v")
        for p in list(points[v]):
            obj_f.write(" "+str(p))
        obj_f.write("\n")
    obj_f.write("\n# FACES\n\n")
    for f in faces:
        obj_f.write("f")
        for p in list(f):
            obj_f.write(" "+str(p))
        obj_f.write("\n")
    obj_f.close()
    ch_mesh=Mesh(path)
    return ch_mesh

# ELLIPSOID

def ellipsoid(dims,centre,*args,**kwargs):
    if dims.__class__ not in [int,float,list]:
        raise TypeError("Outer dimensions of shape must be defined be int, float or list of ints or floats.")
    if centre.__class__ is not SVec:
        raise TypeError("centre must be an SVec coordinate.")
    if dims.__class__ in [int, float]:
        a = b = c = dims
    else:
        if len(dims)==2:
            a=b=dims[0]
            c=dims[1]
        elif len(dims)==3:
            [a,b,c]=dims
        elif len(dims)==1:
            a=b=c=dims[0]
        else:
            raise ValueError("dims argument is incorrect length")
    x_c, y_c, z_c = centre.x, centre.y, centre.z

    def ellipsoid(x,y,z):
        return ((z-z_c)/a)**2+((y-y_c)/b)**2+((x-x_c)/c)**2-1
    sample_width=min([a,b,c])/4
    [x_r, y_r , z_r] = [1.8*i for i in [c,b,a]]
    bounds=[[x_c-x_r,x_c+x_r],[y_c-y_r,y_c+y_r],[z_c-z_r,z_c+z_r]]
    path="Ellipsoid (dims=({},{},{}),centre=({},{}.{}).obj".format(a,b,c,x_c,y_c,z_c)
    implicit=Implicit(ellipsoid,"((z-{})/{})^2+(y-{})/{})^2+(x-{})/{})^2=1".format(z_c,a,y_c,b,x_c,c),*args,**kwargs)
    ellipse_OBJ=implicit_to_OBJ(implicit,sample_width,bounds,*args,**{**kwargs,**{"file_path":path}})
    mesh=Mesh(ellipse_OBJ,*args,**kwargs)
    if "rotation" in kwargs:
        mesh=mesh.rotate_object(kwargs["rotation"],*args,**kwargs)
    return mesh
