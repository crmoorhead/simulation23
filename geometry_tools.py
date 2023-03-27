# Tools used for geometrical calculations for ray-tracing
from material_props import Scatterer, hit_only
from fundamentals import SVec, Ray, Angle, project_axis, CVec, GVec
from copy import deepcopy
from numpy import all, array, arcsin,zeros, exp, minimum, maximum, \
    clip, vectorize, amin, amax, stack, multiply, ones
from numpy.random import random, normal as norm
from math import tan, ceil, inf
from scipy.spatial.transform import Rotation
from numba import prange
from toolbox import plot_3d_function
from inspect import signature, isfunction, isbuiltin
from functools import partial
from autograd import grad
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from copy import copy
from itertools import product
from scipy.optimize import minimize, Bounds, LinearConstraint

# BOUNDING OBJECT CLASS

class Envelope():

    def __init__(self):
        self.envelope_type="abstract"
        self.bounds=20

    def union(self,other):
        raise ReferenceError("Method not defined.")

    def overlaps(self,other):
        raise ReferenceError("Method not defined.")

    def is_in(self,point):
        raise ReferenceError("Method not defined.")

    def intersection(self,other):
        raise ReferenceError("Method not defined.")

    def ray_filter(self):
        raise ReferenceError("Method not defined.")

# BOUNDING BOX CLASS

class AABB(Envelope):
    def __init__(self,c_1,c_2):
        super().__init__()
        self.envelope_type = "AABB"
        if False in [c.__class__ is SVec for c in [c_1,c_2]]:
            if False in [c.__class__ in [float,int] for c in c_1] + [c.__class__ in [float,int] for c in c_2]:
                raise TypeError("Both corners must be instances of SVec objects.")
            else:
                c_1,c_2=SVec(array(c_1)),SVec(array(c_2))
        self.x_min, self.x_max= clip(min(c_1.x,c_2.x),-self.bounds,self.bounds), clip(max(c_1.x,c_2.x),-self.bounds,self.bounds)
        self.y_min, self.y_max = clip(min(c_1.y, c_2.y),-self.bounds,self.bounds), clip(max(c_1.y, c_2.y),-self.bounds,self.bounds)
        self.z_min, self.z_max = clip(min(c_1.z, c_2.z),-self.bounds,self.bounds), clip(max(c_1.z, c_2.z),-self.bounds,self.bounds)

    def __str__(self):
        return "AABB(C_1: [{},{},{}], C_2: [{},{},{}]).".format(self.x_min,self.y_min,self.z_min,self.x_max,self.y_max,self.z_max)

    def __repr__(self):
        return self.__str__()

    def union(self,other):
        if other.__class__ is SVec:
            self.x_min, self.x_max = clip(min(other.x,self.x_min),-self.bounds,self.bounds), clip(max(other.x,self.x_max),-self.bounds,self.bounds)
            self.y_min, self.y_max = clip(min(other.y, self.y_min),-self.bounds,self.bounds), clip(max(other.y, self.y_max),-self.bounds,self.bounds)
            self.z_min, self.z_max = clip(min(other.z, self.z_min),-self.bounds,self.bounds), clip(max(other.z, self.z_max),-self.bounds,self.bounds)
            return self
        if other.__class__ is AABB:
            self.x_min, self.x_max = clip(min(other.x_min,self.x_min),-self.bounds,self.bounds), clip(max(other.x_max,self.x_max),-self.bounds,self.bounds)
            self.y_min, self.y_max = clip(min(other.y_min, self.y_min),-self.bounds,self.bounds), clip(max(other.y_max, self.y_max),-self.bounds,self.bounds)
            self.z_min, self.z_max = clip(min(other.z_min, self.z_min),-self.bounds,self.bounds), clip(max(other.z_max, self.z_max),-self.bounds,self.bounds)
            return self
        if other is None:
            return self
        else:
            raise TypeError("Union must be of two AABBs instances or one AABB and one SVec instance")

    def upper_corner(self):
        return SVec(array([self.x_max,self.y_max,self.z_max]))

    def lower_corner(self):
        return SVec(array([self.x_min, self.y_min, self.z_min]))

    def overlaps(self,other):
        if other is None:
            return False
        elif other.__class__ is AABB:
            x_test= self.x_max > other.x_min and self.x_min < other.x_max
            y_test = self.y_max > other.y_min and self.y_min < other.y_max
            z_test = self.z_max > other.z_min and self.z_min < other.z_max
        elif other.__class__ is list:
            if len(other) != 2:
                raise ValueError("List input must be a list of 2 lists of length 3 giving the max and min "
                                 "extent of bounding box in the principal directions.")
            else:
                if False in [len(l)==3 for l in other]:
                    raise ValueError("List input must be a list of 2 lists of length 3 giving the max and min "
                                     "extent of bounding box in the principal directions.")
                else:
                    x_test = self.x_max > other[0][0] and self.x_min < other[1][0]
                    y_test = self.y_max > other[0][1] and self.y_min < other[1][1]
                    z_test = self.z_max > other[0][2] and self.z_min < other[1][2]
        else:
            raise TypeError("Argument must be a bounding box object or a list of min and max extents of the object.")
        return x_test and y_test and z_test


    def intersection(self,other):
        if other is None:
            return None
        elif other.__class__ is not AABB:
            raise TypeError("Second object must be another AABB instance.")
        else:
            if self.overlaps(other):
                x_min,x_max = max(self.x_min,other.x_min),min(self.x_max,other.x_max)
                y_min,y_max = max(self.y_min,other.y_min),min(self.y_max,other.y_max)
                z_min,z_max = max(self.z_min,other.z_min),min(self.z_max,other.z_max)
                return AABB(SVec(array([x_min,y_min,z_min])),SVec(array([x_max,y_max,z_max])))
            else:
                return None

    def is_in(self,point):
        if point.__class__ is not SVec:
            raise TypeError("Test point must be an instance of SVec.")
        x_test = self.x_max >= point.x and self.x_min <= point.x
        y_test = self.y_max >= point.y and self.y_min <= point.y
        z_test = self.z_max >= point.z and self.z_min <= point.z
        return x_test and y_test and z_test

    def pad(self,other):
        if other not in [int, float]:
            raise TypeError("Padding must be integer or float.")
        self.x_min -= other
        self.x_max += other
        self.y_min -= other
        self.y_max += other
        self.z_min -= other
        self.z_max += other

    def volume(self):
        return (self.x_max-self.x_min)*(self.y_max-self.y_min)*(self.z_max-self.z_min)

    def surface_area(self):
        dx, dy, dz = (self.x_max-self.x_min),(self.y_max-self.y_min),(self.z_max-self.z_min)
        return 2*(dx*dy + dx*dz + dy*dz)

    def widest(self):
        dx, dy, dz = (self.x_max - self.x_min), (self.y_max - self.y_min), (self.z_max - self.z_min)
        if dx >= dy and dx >= dz:
            return "x"
        elif dy >= dz:
            return "y"
        else:
            return "z"

    def lerp(self,coord):
        if coord.__class__ is not tuple:
            raise TypeError("Coordinates value must be a tuple.")
        if len(coord) != 3:
            raise TypeError("Coordinates value must be a tuple of length 3.")
        if False in [coord[i]>1 or coord[i]<0 for i in range(3)]:
            raise ValueError("Coordinate indices must all be between 0 and 1")
        dx, dy, dz = (self.x_max - self.x_min), (self.y_max - self.y_min), (self.z_max - self.z_min)
        return SVec(array([self.x_min+dx*coord[0],self.y_min+dy*coord[0],self.z_min+dz*coord[0]]))

    def point_position(self,point,*args):
        if point.__class__ is not SVec:
            raise TypeError("Point must be an instance of SVec.")
        if "test" in args and not self.is_in(point):
            return NotImplemented
        else:
            dx, dy, dz = (self.x_max - self.x_min), (self.y_max - self.y_min), (self.z_max - self.z_min)
            coords=((point.x-self.x_min)/dx,(point.y-self.y_min)/dy,(point.z-self.z_min)/dz)
            return coords

def joint_AABB(objects):
    if objects.__class__ is not list:
        raise TypeError("Input must be a list.")
    elif False in [o.__class__ in recognised_objects for o in objects]:
        print([o.__class__ for o in objects])
        raise ValueError("An item in the list is not a recognised object")
    else:
        x_min, x_max = min([o.bounding_box.x_min for o in objects]), max([o.bounding_box.x_max for o in objects])
        y_min, y_max = min([o.bounding_box.y_min for o in objects]), max([o.bounding_box.y_max for o in objects])
        z_min, z_max = min([o.bounding_box.z_min for o in objects]), max([o.bounding_box.z_max for o in objects])
    return AABB(SVec(array([x_min,y_min,z_min])),SVec(array([x_max,y_max,z_max])))

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
    def __init__(self,*args,**kwargs):
        self.object_type = "Object"
        self.anchor=SVec(array([0,0,0]))
        self.rotation=None
        self.PROPERTIES={"color":None, "scatterer":Scatterer(hit_only),"rotation":None}
        self.PARAMS={"location":self.anchor}
        if "name" in kwargs:
            self.name=kwargs["name"]
        else:
            self.name=None

    def __str__(self):
        return "Object("+str(self.anchor.value)+")"

    def __repr__(self):
        return "Name: {}\nClass: {}\nParameters: {}\nProperties: {}\n".format(self.name,self.object_type,self.PARAMS,self.PROPERTIES)

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

    def shift(self,vector):
        if vector.__class__!=SVec:
            raise TypeError("Shift vector must be an SVec object.")
        else:
            self.anchor += vector
            self.PARAMS["location"]=self.anchor
        return self

    # Intersection methods

    def ison(self, point):
        def test(self, point):
            return all(self.anchor.value == point.value)
        if point.__class__!=SVec:
            raise TypeError("Point must be an SVec object.")
        else:
            return test(self, point)

    def intersect(self, ray):
        def test(self, ray):
            return ray.isonray(self.anchor)
        if ray.__class__!=Ray:
            raise TypeError("Input variable must be a Ray object.")
        else:
            return test(self, ray)

    def intersect_param(self,ray,*args,**kwargs):
        if "min_param" in kwargs:
            min_param=kwargs["min_param"]
            if self.intersect(ray) == True:
                param=ray.direction.ismultiple((self.anchor - ray.origin), "give_multiple")
                if param >= min_param:
                    return  param
                else:
                    return []
            else:
                return []

        elif self.intersect(ray)==True:   # Need to implement minimum here too
            return ray.direction.ismultiple((self.anchor-ray.origin), "give_multiple")
        else:
            return []

    def intersection_point(self,ray,*args,**kwargs):
        if ray.__class__!=Ray:
            raise TypeError("Input object must be an instance of Ray class.")
        else:
            intersections=self.intersect_param(ray,*args,**kwargs)
            if intersections==None:
                return None
            else:
                return [ray.origin + ray.direction*t for t in intersections]


    # Normal methods - The base method for providing normals at a given point Object class returns None, but establishes the
    #                  form for the general method in subclasses

    def gen_normal(self,point,*args):
        if "test_point" in args:
            if self.ison(point) == False:
                print("{} is not on the object surface".format(point))
                return None
        else:
            return None

    def normal(self,point,*args):
        if point.__class__==SVec:
            return self.gen_normal(point,*args)
        elif point.__class__==list:
            normals={}
            for p in point:
                normals[str(p)]=self.gen_normal(p,*args)
            if "filter" in args:
                normals={k:v for k,v in normals.items() if v!=None}
            return normals
        else:
            return None

    # Misc methods

    def copy(self):
        return deepcopy(self)

    def add_property(self,prop_name, prop_value,*args):
        prop_name=str(prop_name)
        if prop_name in self.PROPERTIES.keys():
            if "verbose" in args:
                print("{} rewritten from '{}' to '{}'".format(prop_name,self.PROPERTIES[prop_name],prop_value))
            self.PROPERTIES[str(prop_name)] = prop_value
        else:
            self.PROPERTIES[str(prop_name)]=prop_value

    def change_scatterer(self,new_scat):
        if new_scat.__class__!=Scatterer:
            raise TypeError("Scatterer must be an instance of Scatterer class.")
        else:
            self.add_property("scatterer",new_scat)

    def change_colour(self,color):
        if color.__class__ not in [CVec, GVec]:
            raise TypeError("Colour must be an instance of CVec or GVec.")
        else:
            self.PROPERTIES["color"]=color
            if color.__class__ is CVec:
                def hit_only_col(incident, return_angle, *args, **kwargs):
                    return color
                hit_only_col.__dict__["color"] = color
                self.change_scatterer(Scatterer(hit_only_col))
            else:
                def hit_only_gs(incident, return_angle, *args, **kwargs):
                    return color
                hit_only_gs.__dict__["color"] = color
                self.change_scatterer(Scatterer(hit_only_gs))
        return self

    def name_object(self,name,*args):
        if self.name==None:
            self.name=name
        else:
            if "verbose" in args:
                print("Name of object changed from {} to {}.".format(self.name,name))
            self.name=name

    def rotate_object(self,rotation):
        if rotation.__class__ is not Rotation:
            return TypeError("Rotation should be a Rotation instance.")
        else:
            self.rotation=rotation
            self.PROPERTIES["rotation"]=self.rotation


# COMPOSITE OBJECT
class Composite(Object):
    def __init__(self,*args,**kwargs):
        self.object_type = "Composite"
        self.anchor=SVec(array([0,0,0]))
        self.rotation=None
        self.PROPERTIES={"color":None, "scatterer":Scatterer(hit_only),"rotation":None}
        self.COMPONENTS={}
        self.PARAMS={"location":self.anchor}
        if "name" in kwargs:
            self.name=kwargs["name"]
        else:
            self.name=None
        self.component_count=len(self.COMPONENTS)

    def __str__(self):
        return "Composite("+str(self.anchor.value)+")"

    def __repr__(self):
        desc="Name: {}\nClass: {}\nParameters: {}\nProperties: {}\n Components: ".format(self.name,self.object_type,self.PARAMS,self.PROPERTIES)
        if self.COMPONENTS == {}:
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

    def shift(self,vector):
        if vector.__class__!=SVec:
            raise TypeError("Shift vector must be an SVec object.")
        else:
            self.anchor += vector
            self.PARAMS["location"]=self.anchor
            for c in self.COMPONENTS:
                self.COMPONENTS[c].shift(vector)
        return self

    def rotate_object(self,rotation):
        self.rotation=rotation
        self.PARAMS["rotation"] = self.rotation
        for c in self.COMPONENTS:
            self.COMPONENTS[c].rotate_object(rotation)

    # Intersection methods # Only returns closest

    def ison(self,point):
        for c in self.COMPONENTS:
            if self.COMPONENTS[c].ison(point):
                return True
        return False

    def gen_normal(self,point,*args):
        for c in self.COMPONENTS:
            if self.COMPONENTS[c].ison(point):
                return self.COMPONENTS[c].gen_normal(point)

    # Misc methods

    def copy(self):
        return deepcopy(self)

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
        if new_scat.__class__!=Scatterer:
            raise TypeError("Scatterer must be an instance of Scatterer class.")
        else:
            self.add_property("scatterer",new_scat)

# SPHERE
class Sphere(Object):

    def __init__(self, centre, radius, ):
        if centre.__class__!=SVec:
            raise TypeError("Centre must be of class SVec.")
        super().__init__()
        self.anchor = centre
        self.radius=float(radius)
        self.PARAMS["radius"]=self.radius
        self.object_type="Sphere"
        self.bounding_box=AABB(self.anchor-SVec(array([self.radius,self.radius,self.radius])),self.anchor+SVec(array([self.radius,self.radius,self.radius])))

    def __str__(self):
        return "Sphere:(Centre: {}, Radius: {})".format(self.anchor.value, self.radius)

    # Transformation methods

    def __mul__(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Multiplication factor must be an integer or float.")
        self.radius=self.radius*abs(other)
        self.PARAMS["radius"]=self.radius
        return self

    def __rmul__(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Multiplication factor must be an integer or float.")
        self.radius=self.radius*abs(other)
        self.PARAMS["radius"]=self.radius
        return self

    def __imul__(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Multiplication factor must be an integer or float.")
        self.radius = self.radius * abs(other)
        self.PARAMS["radius"] = self.radius
        return self

    # Intersection methods

    def ison(self,point):
        super().ison(point)
        def test(self,point):
            dif=point-self.anchor
            return dif.dot(dif) - self.radius ** 2 == 0
        return test(self,point)

    def isin(self,point):
        super().ison(point)
        def test(self, point):
            dif = point - self.anchor
            return dif.dot(dif) - self.radius ** 2 <= 0
        return test(self, point)

    def intersect(self,ray,*args):
        super().intersect(ray)
        def test(self, ray):
            oc=ray.origin-self.anchor
            a=ray.direction.dot(ray.direction)
            b=2*oc.dot(ray.direction)
            c=oc.dot(oc)-self.radius**2
            discriminant=b**2-4*a*c
            return discriminant >= 0
        return test(self,ray)

    def intersect_param(self,ray,*args,**kwargs):
        if ray.__class__!=Ray:
            raise TypeError("Input must be an instance of Ray class.")
        else:
            oc=ray.origin-self.anchor
            a=ray.direction.dot(ray.direction)
            b=2*oc.dot(ray.direction)
            c=oc.dot(oc)-self.radius**2
            discriminant=b**2-4*a*c
            if discriminant <0:
                return []
            else:
                if discriminant!=0:
                    root=discriminant**0.5
                    candidates = [(-b - root) / (2 * a), (-b + root) / (2 * a)]
                    if "min_param" in kwargs:
                        viable = [c for c in candidates if c >= kwargs["min_param"]]
                    else:
                        viable = [c for c in candidates if c > 0]

                    if "all" in args:
                        return viable
                    else:
                        if viable == []:
                            return []         # Needs to be a list for consistency
                        return [min(viable)]  # Needs to be a list for consistency
                else:
                    if b<0:
                        return [-b/(2*a)]
                    else:
                        return []

    # Normal Methods

    def gen_normal(self,point,*args):
        if "test_point" in args:
            if self.ison(point) == False:
                print("{} is not on the object surface".format(point))
                return None
            else:
                return point-self.anchor
        else:
            return point-self.anchor

# PLANE
class Plane(Object): # PLANE is described by the formula z + ax + by + c = 0 and in the degenerate case, by the WALL object.

    def __init__(self,a,b,c):
        for i in [a,b,c]:
            if i.__class__ ==array(0).__class__:
                i=float(i)
            if i.__class__ not in [int,float]:
                raise TypeError("Plane parameters must be integer or float.")
        super().__init__()
        self.object_type = "Plane"
        self.x_coeff=a
        self.y_coeff=b
        self.constant=c
        self.anchor=SVec(array([0,0,-self.constant]))
        normal=SVec(array([-self.x_coeff,-self.y_coeff,1]))
        self.unit_normal=normal.normalise()
        self.plane_vec=SVec(array([self.x_coeff,self.y_coeff,1]))
        self.PARAMS={"x_coeff":self.x_coeff,"y_coeff":self.y_coeff,"constant":self.constant,"unit_normal":self.unit_normal,"anchor":self.anchor}
        base=20
        c_1=SVec(array([-base,-base,base*self.x_coeff+base*self.y_coeff-self.constant]))
        c_2=SVec(array([base,base,-base*self.x_coeff-base*self.y_coeff-self.constant]))
        self.bounding_box=AABB(c_1,c_2)

    def __str__(self):
        string="Plane(z "
        if self.x_coeff==0:
            string +=""
        elif self.x_coeff==1:
            string+="+ x "
        elif self.x_coeff>0:
            string+="+ "+str(round(self.x_coeff,4))+"x "
        else:
            string += "- " + str(round(abs(self.x_coeff),4)) + "x "

        if self.y_coeff==0:
            string +=""
        elif self.y_coeff==1:
            string+="+ y "
        elif self.y_coeff>0:
            string+="+ "+str(round(self.y_coeff,4))+"y "
        else:
            string += "- " + str(round(abs(self.y_coeff),4)) + "y "

        if self.constant==0:
            string +=""
        elif self.constant>0:
            string+="+ "+str(self.constant)
        else:
            string += "- " + str(abs(self.constant))

        string += " = 0)"

        return string

    # Intersection methods

    def ison(self,point):
        def test(self, point):
            return self.plane_vec.dot(point)+self.constant==0
        if point.__class__ != SVec:
            raise TypeError("Point must be an SVec object.")
        else:
            return test(self, point)

    def intersect(self,ray,*args,**kwargs):
        if ray.__class__!=Ray:
            raise TypeError("Ray input must be of Ray class.")
        if ray.direction.dot(self.plane_vec)==0:
            return False
        else:
            return True

    def intersect_param(self,ray,*args,**kwargs):
        if ray.__class__!=Ray:
            raise TypeError("Input must be an instance of Ray class.")
        else:
            denom=self.plane_vec.dot(ray.direction)
            if denom==0:
                return []
            else:
                num = -self.plane_vec.dot(ray.origin) - self.constant
                param= num/denom
                if "min_param" in kwargs:
                    if param < kwargs["min_param"]:
                        return []
                return [param]

    def in_plane(self,ray):
        point_1=ray.origin
        point_2=ray.origin+ray.direction
        return self.ison(point_1) and self.ison(point_2)

    def plane_angle(self,ray):
        return angle_with_plane(ray,self)

    # Normal methods

    def gen_normal(self,point):
        return self.unit_normal

    # Misc methods

    def shift(self,dist):
        if dist.__class__ in [int,float]:
            self.constant-=dist
            self.anchor = SVec(array([0, 0, -self.constant]))
            self.PARAMS["constant"],self.PARAMS["anchor"]=self.constant,self.anchor
            return self
        else:
            raise TypeError("Shift distance must be integer or float.")

    def near_far(self,ref,point):
        if ref.__class__ is not SVec or point.__class__ is not SVec:
            raise TypeError("Both reference and target points must be SVec instances.")
        path=Ray(ref,point-ref)
        t=self.intersect_param(path)[0]
        if t<=1:
            return True
        else:
            return False

    def gradient(self):
        return -self.unit_normal

# WALL
class Wall(Plane):

    def __init__(self,a,b,c):
        for i in [a, b, c]:
            if i.__class__ ==array(0).__class__:
                i=float(i)
            if i.__class__ not in [int, float]:
                raise TypeError("Plane parameters must be integer or float.")
        if a !=0:
            a,b,c=1,b/a,c/a
        else:
            if b==0:
                raise AttributeError("This is not a valid plane.")
            else:
                b,c=1,c/b
        super().__init__(a,b,c)
        self.x_coeff=a
        self.y_coeff=b
        self.constant=c
        if self.x_coeff!=0:
            self.anchor=SVec(array([-self.constant,0,0]))
        else:
            self.anchor=SVec(array([0,-self.constant,0]))
        self.plane_vec=SVec(array([self.x_coeff,self.y_coeff,0]))
        if self.x_coeff != 0:
            normal=SVec(array([1,-self.y_coeff,0]))
        else:
            normal=SVec(array([0,-1,0]))
        self.unit_normal=normal.normalise()
        self.PARAMS = {"x_coeff": self.x_coeff, "y_coeff": self.y_coeff, "constant": self.constant,"unit_normal": self.unit_normal, "anchor": self.anchor}
        self.object_type = "Wall"
        base=20
        if self.x_coeff != 0:
            if self.y_coeff != 0:
                c_1 = SVec(array([-base, -base, -base]))
                c_2 = SVec(array([base, base, base]))
            else:
                c_1 = SVec(array([-self.constant, -base, -base]))
                c_2 = SVec(array([-self.constant, base, base]))
        else:
            c_1 = SVec(array([-base, -self.constant, -base]))
            c_2 = SVec(array([base, -self.constant, base]))
        self.bounding_box = AABB(c_1, c_2)

    def __str__(self):
        string="Wall("
        if self.x_coeff != 0:
            string+="x"
        else:
            string+="y"
            if self.constant == 0:
                string += " = 0)"
                return string
            elif self.constant>0:
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
            string += " = 0"
            return string
        elif self.constant > 0:
            string += " + " + str(self.constant) + " = 0)"
            return string
        else:
            string += " - " + str(abs(self.constant)) + " = 0)"
            return string

    # Redefined methods

    def shift(self,dist):
        self.constant -= dist
        if self.x_coeff!=1:
            self.anchor=SVec(array([-self.constant,0,0]))
        else:
            self.anchor = SVec(array([0,-self.constant, 0]))
        self.PARAMS["constant"], self.PARAMS["anchor"] = self.constant, self.anchor
        return self

# POINT AND PLANE OPERATIONS
def project_plane(point,plane):
    if point.__class__ is not SVec:
        raise TypeError("Point must be an instance of SVec.")
    if plane.__class__ not in [Plane, Wall]:
        raise TypeError("Plane must be an instance of Plane or Wall.")
    prod_1=plane.plane_vec.dot(point)
    prod_2=plane.plane_vec.dot(plane.normal())
    return point - plane.normal*prod_1/prod_2

def colinear_check(p_1,p_2,p_3):
    for p in [p_1,p_2,p_3]:
        if p.__class__!=SVec:
            raise TypeError("All points must be instances of SVec.")
    p_12=p_2-p_1
    p_13=p_3-p_1
    check=p_12.ismultiple(p_13)
    if check==None:
        return False
    return True

def plane_from_points(p_1,p_2,p_3,*args):  # Generates a plane object for three given points.
    for p in [p_1,p_2,p_3]:
        if p.__class__!=SVec:
            raise TypeError("All points must be instances of SVec.")
    if "colinear_check" in args:
        test=colinear_check(p_1,p_2,p_3)
        if test is True:
            raise ValueError("These points are colinear.")
    p_12 = p_2 - p_1
    p_13 = p_3 - p_1
    normal=p_12.cross(p_13)
    c=-float(normal.dot(p_1))
    coeffs=[float(normal.z),float(normal.x),float(normal.y),c]
    if coeffs[0]!=0:
        coeffs=[co/coeffs[0] for co in coeffs]
        return Plane(coeffs[1],coeffs[2],coeffs[3])
    elif coeffs[1]!=0:
        coeffs=[co/coeffs[1] for co in coeffs]
        return Wall(coeffs[1],coeffs[2],coeffs[3])
    else:
        if coeffs[2]==0:
            print(p_1,p_2,p_3)
        coeffs=[co/coeffs[2] for co in coeffs]
        return Wall(coeffs[1],coeffs[2],coeffs[3])

# CYLINDER
class Cylinder(Object):

    def __init__(self,centre,radius):

        if centre.__class__!=SVec:
            raise TypeError("Centre must be of class SVec.")
        super().__init__()
        self.anchor=project_axis(centre,"z")
        self.radius=float(radius)
        self.PARAMS={"anchor":self.anchor,"radius":self.radius}
        self.object_type = "Cylinder"
        self.bounding_box=AABB(SVec(array([self.anchor.x+self.radius,self.anchor.y+self.radius,100])),
                               SVec(array([self.anchor.x-self.radius,self.anchor.y-self.radius,-100])))

    def __str__(self):
        return "Cylinder:(Centre: {}, Radius: {})".format(self.anchor.value, self.radius)

    # Transformation methods

    def shift(self,other):
        if other.__class__ is not SVec:
            raise TypeError("Input must be an instance of SVec.")
        other=project_axis(other,"z")
        self.anchor+=other
        self.PARAMS["anchor"]=self.anchor
        return self

    def __mul__(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Multiplication factor must be an integer or float.")
        self.radius=self.radius*abs(other)
        self.PARAMS["radius"]=self.radius
        return self

    def __rmul__(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Multiplication factor must be an integer or float.")
        self.radius=self.radius*abs(other)
        self.PARAMS["radius"]=self.radius
        return self

    def __imul__(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Multiplication factor must be an integer or float.")
        self.radius = self.radius * abs(other)
        self.PARAMS["radius"] = self.radius
        return self

    # Intersection methods

    def ison(self, point):
        super().ison(point)
        cyl_vec=project_axis(point,"z")
        diff=cyl_vec-self.anchor
        return diff.dot(diff)-self.radius**2 == 0

    def intersect(self, ray, *args):
        super().intersect(ray)
        def test(self, ray,*args):
            proj_dir=project_axis(ray.direction,"z")
            diff=ray.direction-self.anchor
            proj_diff=project_axis(diff,"z")
            a=proj_dir.dot(proj_dir)
            b=2*proj_dir.dot(proj_diff)
            c=proj_diff.dot(proj_diff)-self.radius**2
            return b**2 - 4*a*c >= 0
        return test(self,ray,*args)

    def intersect_param(self,ray,*args,**kwargs):
        if ray.__class__!=Ray:
            raise TypeError("Input must be an instance of Ray class.")
        else:
            proj_dir=project_axis(ray.direction,"z")
            diff=ray.direction-self.anchor
            proj_diff=project_axis(diff,"z")
            a=proj_dir.dot(proj_dir)
            b=2*proj_dir.dot(proj_diff)
            c=proj_diff.dot(proj_diff)-self.radius**2
            discriminant= b**2 - 4*a*c
            if discriminant <0:
                return []
            else:
                if discriminant!=0:
                    root=discriminant**0.5
                    candidates = [(-b - root) / (2 * a), (-b + root) / (2 * a)]
                    if "min_param" in kwargs:
                        viable = [c for c in candidates if c >= kwargs["min_param"]]
                    else:
                        viable = [c for c in candidates if c > 0]

                    if "all" in args:
                        return viable
                    else:
                        if viable == []:
                            return []         # Needs to be a list for consistency
                        return [min(viable)]  # Needs to be a list for consistency
                else:
                    if b<0:
                        return [-b/(2*a)]
                    else:
                        return []

    # Normal methods

    def gen_normal(self,point,*args):
        if "test_point" in args:
            if self.ison(point) == False:
                print("{} is not on the object surface".format(point))
                return None
            else:
                return point - self.anchor
        else:
            return point - self.anchor

# FINITE CYLINDER
class Rod(Cylinder):

    def __init__(self,centre, radius, length):

        if centre.__class__!=SVec:
            raise TypeError("Centre must be of class SVec.")
        if radius.__class__ not in [float, int]:
            raise TypeError("Radius must be integer or float.")
        if length.__class__ not in [float, int]:
            raise TypeError("Rod length must be integer or float.")
        super().__init__(centre,radius)
        self.anchor=centre
        self.radius=float(radius)
        self.length=length
        self.top=self.anchor.z + self.length/2
        self.bottom = self.anchor.z - self.length / 2
        self.PARAMS={"anchor":self.anchor,"radius":self.radius,"length":self.length,"bottom":self.bottom,"top": self.top}
        self.object_type = "Rod"
        self.bounding_box=AABB(SVec(array([self.anchor.x+self.radius,self.anchor.y+self.radius,self.top])),
                               SVec(array([self.anchor.x-self.radius,self.anchor.y-self.radius,self.bottom])))

    def __str__(self):
        return "Rod:(Centre: {}, Radius: {}, Length: {})".format(self.anchor.value, self.radius, self.length)

    # Transformation methods

    def __pow__(self, power):   # Can be used to elongate or contract rod
        if power.__class__ not in [int, float]:
            raise TypeError("Factor must be an integer or float.")
        if power<=0:  # If == 0 we can collapse to disc object.
            raise TypeError("Factor must be strictly positive.")
        self.length*=power
        self.top = self.anchor.z + self.length / 2
        self.bottom = self.anchor.z - self.length / 2
        self.PARAMS = {"anchor": self.anchor, "radius": self.radius, "length": self.length, "bottom": self.bottom,
                       "top": self.top}
        return self

    def extend(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Factor must be an integer or float.")
        if other <= 1:  # If == 0 we can collapse to disc object.
            raise TypeError("Factor must be greater than 1.")
        return self**other

    def contract(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Factor must be an integer or float.")
        if other <= 1:  # If == 0 we can collapse to disc object.
            raise TypeError("Factor must be greater than 1.")
        other=float(other)
        return self**(1/other)

    def shift(self,vector):
        if vector.__class__!=SVec:
            raise TypeError("Shift vector must be an SVec object.")
        else:
            self.anchor += vector
            self.PARAMS["location"]=self.anchor
        return self

    # Intersection methods

    def ison(self, point):
        super().ison(point)
        if point.z > self.top or point.z < self.bottom:
            return False
        else:
            cyl_vec = project_axis(point, "z")
            diff = cyl_vec - self.anchor
            return diff.dot(diff) - self.radius ** 2 == 0

    def intersect(self, ray, *args):
        super().intersect(ray)

        def test(self, ray, *args):
            proj_dir = project_axis(ray.direction, "z")
            diff = ray.direction - self.anchor
            proj_diff = project_axis(diff, "z")
            a = proj_dir.dot(proj_dir)
            b = 2 * proj_dir.dot(proj_diff)
            c = proj_diff.dot(proj_diff) - self.radius ** 2
            return b ** 2 - 4 * a * c >= 0

        return test(self, ray, *args)

    def intersect_param(self, ray, *args, **kwargs):
        if ray.__class__ != Ray:
            raise TypeError("Input must be an instance of Ray class.")
        else:
            proj_dir = project_axis(ray.direction, "z")
            diff = ray.direction - self.anchor
            proj_diff = project_axis(diff, "z")
            a = proj_dir.dot(proj_dir)
            b = 2 * proj_dir.dot(proj_diff)
            c = proj_diff.dot(proj_diff) - self.radius ** 2
            discriminant = b ** 2 - 4 * a * c
            if discriminant < 0:
                return []
            else:
                if discriminant != 0:
                    root = discriminant ** 0.5
                    candidates = [(-b - root) / (2 * a), (-b + root) / (2 * a)]
                    def z_test(t,ray,min,max):
                        min_test = ray.origin.z + ray.direction.z*t > min
                        max_test = ray.origin.z + ray.direction.z*t < max
                        return min_test and max_test
                    if "min_param" in kwargs:
                        viable = [c for c in candidates if c >= kwargs["min_param"]]
                        viable = [v for v in viable if z_test(v,ray,self.bottom,self.top)]
                    else:
                        viable = [c for c in candidates if c > 0]
                        viable = [v for v in viable if z_test(v, ray, self.bottom, self.top)]

                    if "all" in args:
                        return viable
                    else:
                        if viable == []:
                            return []  # Needs to be a list for consistency
                        return [min(viable)]  # Needs to be a list for consistency
                else:
                    if b < 0:
                        return [-b / (2 * a)]
                    else:
                        return []

# ROTATED FINITE CYLINDER

# DISC
class Disc(Object):

    def __init__(self,centre,radius,plane_desc):
        if centre.__class__ is not SVec:
            raise TypeError("Centre must be an instance of SVec")
        if radius.__class__ not in [int,float]:
            raise TypeError("Radius must be an integer or float")
        super().__init__()
        self.object_type="Disc"
        self.anchor=centre
        self.radius=radius
        if plane_desc.__class__ in [Plane, Wall]:
            self.plane_vec=plane_desc.plane_vec
            self.constant=-centre.dot(plane_desc.plane_vec)
            self.circle_plane=plane_desc.shift(-self.constant+plane_desc.constant)
        elif plane_desc.__class__ is list:
            if len(plane_desc) ==3:
                if plane_desc[2]!=0:
                    self.plane_vec=SVec(array([i/plane_desc[2] for i in plane_desc]))
                    self.constant=-centre.dot(plane_desc)
                    self.circle_plane=Plane(self.plane_vec.x,self.plane_vec.y,self.constant)
                else:
                    self.plane_vec = SVec(array(plane_desc))
                    self.constant = -centre.dot(plane_desc)  # NO
                    self.circle_plane = Wall(self.plane_vec.x, self.plane_vec.y, self.constant)
            else:
                raise ValueError("Length of list must be 3.")
        elif plane_desc.__class__ is SVec:  # THIS NEEDS DONE
            if plane_desc[2] != 0:
                self.plane_vec = plane_desc / plane_desc[2]
                self.constant = -centre.dot(self.plane_vec)
                self.circle_plane = Plane(self.plane_vec.x, self.plane_vec.y, self.constant)
            else:
                self.plane_vec = plane_desc
                self.constant = -centre.dot(self.plane_vec)
                self.circle_plane = Wall(self.plane_vec.x, self.plane_vec.y, self.constant)
        self.PARAMS = {"anchor": self.anchor, "radius": self.radius,"plane_vec":self.plane_vec, "constant":self.constant,"plane":self.circle_plane}
        if self.circle_plane.__class__ is Plane:
            c_1=self.anchor + self.circle_plane.gradient() * self.radius
            c_2=self.anchor - self.circle_plane.gradient() * self.radius
            self.bounding_box=AABB(c_1,c_2)
        else:
            c_1 = self.anchor - self.circle_plane.gradient()*self.radius + SVec(array([0,0,-1]))
            c_2 = self.anchor + self.circle_plane.gradient() * self.radius + SVec(array([0, 0, 1]))
            self.bounding_box=AABB(c_1,c_2)

    def __str__(self):
        return "Disc(Home Plane: {}, Centre: {}, Radius: {})".format(self.circle_plane,self.anchor,self.radius)

    def shift(self,basis):
        if basis.__class__ is not tuple:
            raise TypeError("Shift must be a tuple of movement in (x,y) distances.")
        elif len(basis) != 2:
            raise ValueError("Tuple must be of length 2.")
        if self.circle_plane.__class__ is Plane:
            self.anchor=self.anchor+SVec(array([basis[0],basis[1],-(basis[0]*self.plane_vec.x+basis[1]*self.plane_vec.y)]))
        else:
            if self.plane_vec.x!=0:
                if self.plane_vec.y == 0:
                    self.anchor=self.anchor+SVec(array([0,basis[0],basis[1]]))
                else:
                    self.anchor=self.anchor+SVec(array([basis[0],-basis[0]/self.plane_vec.y,basis[1]]))
            else:
                self.anchor=self.anchor+SVec(array([basis[0],0,basis[1]]))
        self.PARAMS["anchor"] = self.anchor
        return self

    def shift_plane(self,dist):
        self.circle_plane.shift(dist)
        if self.circle_plane.plane_vec.z!=0:
            self.anchor+=SVec(array([0,0,dist]))
        elif self.circle_plane.plane_vec.y==0:
            self.anchor+=SVec(array([dist,0,0]))
        else:
            self.anchor+=SVec(array([0,dist,0]))
        self.constant = -self.anchor.dot(self.plane_vec)
        self.PARAMS["anchor"] = self.anchor
        self.PARAMS["circle_plane"] = self.circle_plane
        self.PARAMS["constant"]=self.constant
        return self


    def __mul__(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Multiplication factor must be an integer or float.")
        self.radius=self.radius*abs(other)
        self.PARAMS["radius"]=self.radius
        return self

    def __rmul__(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Multiplication factor must be an integer or float.")
        self.radius=self.radius*abs(other)
        self.PARAMS["radius"]=self.radius
        return self

    def __imul__(self, other):
        if other.__class__ not in [int, float]:
            raise TypeError("Multiplication factor must be an integer or float.")
        self.radius = self.radius * abs(other)
        self.PARAMS["radius"] = self.radius
        return self

    # Intersection methods

    def ison(self, point):
        return self.circle_plane.ison(point) and abs(self.anchor-point) <= self.radius

    def intersect(self, ray):
        return self.intersect_param(ray) != []

    def intersect_param(self,ray,*args,**kwargs):
        if ray.__class__ != Ray:
            raise TypeError("Input must be an instance of Ray class.")
        params=self.circle_plane.intersect_param(ray,*args,**kwargs)
        points = [ray.origin+ray.direction*t for t in params]
        return [params[i] for i in range(len(params)) if self.ison(points[i])]

    # Normal methods

    def gen_normal(self, point, *args):
        if "test_point" in args:
            if self.ison(point) == False:
                print("{} is not on the object surface".format(point))
                return None
            else:
                return self.circle_plane.unit_normal
        else:
            return self.circle_plane.unit_normal

# ANNULUS

# TRIANGLE
class Triangle(Object):

    def __init__(self,p1,p2,p3):
        self.object_type = "Triangle"
        for p in [p1,p2,p3]:
            if p.__class__ is not SVec:
                raise TypeError("All corners should be an instance of SVec.")
        super().__init__()
        [self.p1,self.p2,self.p3] = [SVec(array(i)) for i in sorted([list(p.value) for p in [p1,p2,p3]])]
        self.u=self.p2-self.p1
        self.v=self.p3-self.p1
        if self.u.ismultiple(self.v):
            raise ValueError("These points are colinear.")
        self.anchor=self.p1
        self.unit_normal=plane_from_points(p1,p2,p3).normal(p1)
        self.PARAMS={"anchor":self.anchor,"vec_1":self.u,"vec_2":self.v,"normal":self.unit_normal}
        mins = SVec(minimum(minimum(p1.value,p2.value),p3.value))
        maxes = SVec(maximum(maximum(p1.value,p2.value),p3.value))
        self.bounding_box=AABB(mins,maxes)

    def __str__(self):
        return "Triangle(p_1:{}, p_2:{}, p_3:{})".format(self.p1.value,self.p2.value,self.p3.value)

    # Transformation methods

    def shift(self,other):
        if other.__class__ is not SVec:
            raise TypeError("Shift argument must be an SVec instance.")
        self.p1 += other
        self.p2 += other
        self.p3 += other
        self.anchor = self.p1
        self.PARAMS["anchor"] = self.anchor
        return self

    # Intersection methods
    def ison(self,point):
        q=point-self.anchor
        test_1=self.u.x*self.v.y-self.u.y*self.v.x
        if test_1 is not 0:
            s=(self.v.y*q.x-self.v.x*q.y)/test_1
            t=(-self.u.y*q.x+self.u.x*q.y)/test_1
            c_test = round(self.u.z*s + self.v.z*t,6) == round(q.z,6)
        else:
            test_2=self.u.x*self.v.z-self.u.z*self.v.x
            if test_2 is not 0:
                s = (self.v.z*q.x - self.v.x*q.z)/test_2
                t = (-self.u.z*q.x + self.u.x*q.z)/test_2
                c_test = round(self.u.y*s + self.v.y*t,6) == round(q.y,6)
            else:
                test_3=self.u.y*self.v.z-self.u.z*self.v.y
                if test_3 is not 0:
                    s = (self.v.y*q.z - self.v.z*q.y)/test_3
                    t = (-self.u.y*q.z + self.u.z*q.y)/test_3
                    c_test = round(self.u.x*s + self.v.x*t,6) == round(q.x,6)
                else:
                    return False
        if c_test == False:
            return False
        s_test = s >= 0
        t_test = t >= 0
        st_test = s + t <= 1
        return s_test and t_test and st_test

    def intersect(self, ray):
        k=ray.origin-self.anchor
        p=ray.direction.cross(self.v)
        q=k.cross(self.u)
        denom=p.dot(self.u)
        s=p.dot(k)
        t=q.dot(ray.direction)
        return s >= 0 and t >= 0 and s+t<=denom

    # P140 of Ray tracing book for explanation

    def intersect_param(self,ray,*args,**kwargs):
        k=ray.origin-self.anchor
        p=ray.direction.cross(self.v)
        q=k.cross(self.u)
        denom=p.dot(self.u)
        s=p.dot(k)/denom
        t=q.dot(ray.direction)/denom
        if s >= 0 and t >= 0 and s+t<=1:
            return [q.dot(self.v)/denom]
        else:
            return []

    def area(self):
        return abs(self.u.cross(self.v))/2
    # Normal methods

    def gen_normal(self,point,*args):
        return self.unit_normal

# TETRAHEDRON
class Tetrahedron(Composite):

    def __init__(self,p1,p2,p3,p4):
        for p in [p1,p2,p3,p4]:
            if p.__class__ is not SVec:
                raise TypeError("All corners should be an instance of SVec.")
        super().__init__()
        self.object_subtype = "Tetrahedron"
        [self.p1,self.p2,self.p3,self.p4] = [SVec(array(i)) for i in sorted([list(p.value) for p in [p1,p2,p3,p4]])]
        self.u=self.p2-self.p1
        self.v=self.p3-self.p1
        self.w=self.p4-self.p1
        if self.u.ismultiple(self.v) or self.u.ismultiple(self.w) or self.w.ismultiple(self.v):
            raise ValueError("These points are colinear.")
        self.anchor=self.p1
        self.unit_normals=[plane_from_points(p1,p2,p3).normal(p1),plane_from_points(p1,p3,p4).normal(p1),
                           plane_from_points(p1,p2,p4).normal(p1),plane_from_points(p2,p3,p4).normal(p2)]
        self.PARAMS={"anchor":self.anchor,"vec_1":self.u,"vec_2":self.v,"vec_3":self.w,"normals":self.unit_normals}
        self.COMPONENTS={"T1":Triangle(p1,p2,p3),"T2":Triangle(p1,p3,p4),"T3":Triangle(p1,p2,p4),"T4":Triangle(p2,p3,p4)}
        mins = SVec(minimum(minimum(minimum(p1.value, p2.value), p3.value),p4.value))
        maxes = SVec(maximum(maximum(maximum(p1.value, p2.value), p3.value), p4.value))
        self.bounding_box=AABB(mins,maxes)
        self.comp_count = len(self.COMPONENTS)

    def __str__(self):
        return "Tetrahedron(p_1:{}, p_2:{}, p_3:{}, p_4:{})".format(self.p1.value,self.p2.value,self.p3.value,self.p4.value)

# TUBE

# TORUS

# EXTRUSION

# REVOLUTION

# CUBOID

class Cuboid(Composite):
    pass

# INTRINSIC FUNCTION

class Implicit:

    def __init__(self, funct, formula):
        if not (isfunction(funct) or isbuiltin(funct) or funct.__class__ is partial(print).__class__):
            raise TypeError("Implicit function input must be a function object")
        if False in [v in signature(funct).parameters for v in ["x","y","z"]]:
            raise AttributeError("Function input must have x, y and z as variables")
        self.function=funct
        self.formula=formula
        self.anchor=SVec(array([0,0,0]))
        self.epsilon=0.0001

    def __str__(self):
        return "Implicit({})".format(self.formula)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def side(self,point):
        if point.__class__ is not SVec:
            raise TypeError("Point must be an instance of SVec.")
        return self.function(point.x,point.y,point.z)>=1

    def is_on(self,point):
        if point.__class__ is not SVec:
            raise TypeError("Point must be an instance of SVec.")
        return self.function(point.x,point.y,point.z)<=self.epsilon

    def union(self,other):
        if other.__class__ is not Implicit:
            raise TypeError("Both objects must be instances of Implicit.")
        def unioned(x, y, z):
            return max(self.function(x, y, z), other.function(x, y, z))
        return Implicit(unioned)

    def intersection(self,other):
        if other.__class__ is not Implicit:
            raise TypeError("Both objects must be instances of Implicit.")
        def intersected(x, y, z):
            return min(self.function(x, y, z), other.function(x, y, z))
        return Implicit(intersected)

    def negate(self,other):
        if other.__class__ is not Implicit:
            raise TypeError("Both objects must be instances of Implicit.")
        def negated(x,y,z):
            return -self.function(x,y,z)

    def shift(self,other):
        if other.__class__ is not SVec:
            raise TypeError("Vector must be an instance of SVec.")
        def shifted(x,y,z):
            return self.function(x-other.x,y-other.y,z-other.z)

    def compute(self,other):
        if other.__class__ is not SVec:
            raise TypeError("Vector must be an instance of SVec.")
        return self.function(other.x,other.y,other.z)



class Surface(Object):
    def __init__(self,implicit_funct,*args,**kwargs):
        if implicit_funct.__class__ is not Implicit:
            raise TypeError("Implicit function must be an")
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

        self.bounding_box=AABB(SVec(array([self.x_max,self.y_max,self.z_max])),SVec(array([self.x_min,self.y_min,self.z_min])))
        self.function=implicit_funct.function
        self.name=implicit_funct.formula
        self.norm_funct=vectorize(self.normal_funct(*args))
        self.epsilon=0.0001

    def __str__(self):
        return "Surface({})".format(self.name)

    # Intersection methods

    def side(self,point):
        if point.__class__ is not SVec:
            raise TypeError("Point must be an instance of SVec.")
        return self.function(point.x,point.y,point.z)>=1

    def in_range(self,point):
        return self.bounding_box.is_in(point)

    def ison(self,point,**kwargs):
        if point.__class__ is not SVec:
            raise TypeError("Point must be an instance of SVec.")
        if not self.in_range(point):
            return False
        if "epsilon" in kwargs:
            return abs(self.function(point.x, point.y, point.z)) <= self.epsilon
        else:
            return self.function(point.x,point.y,point.z) == 0

    def intersect_param(self,ray,*args,**kwargs):
        def create_loss_funct(implicit, ray, *args, **kwargs):
            dir_x, dir_y, dir_z = array(ray.direction.x), array(ray.direction.y), array(ray.direction.z)
            orig_x, orig_y, orig_z = ray.origin.x, ray.origin.y, ray.origin.z
            def loss_func(t, *args, **kwargs):
                return abs(implicit(dir_x * t + orig_x, dir_y * t + orig_y, dir_z * t + orig_z, *args, **kwargs))
            return loss_func
        loss_funct=create_loss_funct(self.function,ray)
        param = minimize(loss_funct,x0=0,method='Nelder-Mead', tol=self.epsilon)

        return
    # Normal methods

    def normal_funct(self,*args):
        dfdx,dfdy,dfdz=grad(self.function,0),grad(self.function,1),grad(self.function,2)
        return (dfdx,dfdy,dfdz)

    def gen_normal(self,point,*args):
        x,y,z=point.x, point.y, point.z
        if "test_point" in args:
            if not self.ison(point):
                return None
        if "unit_norm" not in args:
            return SVec(array([d(x,y,z) for d in self.normal_funct(*args)]))
        else:
            return SVec(array([d(x, y, z) for d in self.normal_funct(*args)])).normalise()

    def gen_normals(self,points,*args):
        if points.__class__ is list:
            if False in [p.__class__ is SVec for p in points if p is not None]:
                raise TypeError("Points must all be instances of SVec")
            else:
                gradient=self.normal_funct(*args)
                normals=[]
                for p in points:
                    if p != None:
                        if "test_point" in args:
                            if not self.ison(p):
                                normals.append(None)
                            else:
                                normals.append(SVec(array([d(p.x, p.y, p.z) for d in gradient])))
                        else:
                            normals.append(SVec(array([d(p.x, p.y, p.z) for d in gradient])))
                    else:
                        normals.append(None)
                return normals
        elif points.__class__ == array(0).__class__:
            if points.shape[2] != 3:
                raise TypeError("Array of points is not correct shape.")
            else:
                gradient = self.normal_funct(*args)     # Need to apply the gradient to an array of shape (n,m,3)
                n,m=points.shape[:2]
                normals=zeros(points.shape)
                for i in prange(n):
                    for j in prange(m):
                        if "test_point" in args:
                            if not self.ison(points[i,j,:]):
                                normals[i, j, :]=array([0,0,0])
                            else:
                                normals[i,j,:] = array([d(points[i,j,0], points[i,j,1], points[i,j,2]) for d in gradient])
                        else:
                            normals[i, j, :] = array([d(points[i,j,0], points[i,j,1], points[i,j,2]) for d in gradient])
                return normals
        else:
            raise ValueError("Points must be in list or array form.")

# TESSALATED SURFACE FROM FUNCTION

def array_from_implicit(implicit, sample_width, bounds,*args,**kwargs):
    if implicit.__class__ is not Implicit:
        raise TypeError("Input must be an instance of Implicit.")
    if sample_width.__class__ not in [int,float]:
        raise TypeError("Cube size must be an integer of float.")
    if bounds.__class__ is not list:
        raise TypeError("Bounds argument must be a list of three two element lists or tuples. This is not a list.")
    else:
        if len(bounds) != 3:
            raise TypeError("Bounds argument must be a list of three two element lists or tuples. "
                            "One of the axes bounds is missing.")
        elif False in [len(i)==2 for i in bounds]:
            raise TypeError("Bounds argument must be a list of two element lists or tuples. Check list/tuple lengths.")
        elif True in [[i[j].__class__ in [int,float] for j in range(len(i))] for i in bounds]:
            raise TypeError("Bounds argument must be a list of two element lists or tuples. "
                            "Bounds must be integers or floats.")

    # Create sample
    [n,m,k]=[int((bounds[i][1]-bounds[i][0])//sample_width) for i in range(3)]
    [x_range,y_range,z_range]=[[bounds[i][0],bounds[i][0]+([n,m,k][i]+1)*sample_width] for i in range(3)]
    sample_array=zeros((n+1,m+1,k+1))
    for i in range(n+1):
        for j in range(m+1):
            for l in range(k+1):
                sample_array[i,j,l]=implicit.function(x_range[0]+sample_width*i,y_range[0]+sample_width*j,z_range[0]+sample_width*l)
    return sample_array

# Extract from function

def array_from_explicit(funct, x_range, y_range, n, *args, **kwargs):
    if not (isfunction(funct) or isbuiltin(funct) or funct.__class__ is partial(print).__class__):
        raise TypeError("Input function must be a function object")
    if x_range.__class__ is not tuple:
        raise TypeError("Range of x should be expressed in tuple form")
    if y_range.__class__ is not tuple:
        raise TypeError("Range of y should be expressed in tuple form")
    if n.__class__ is not int:
        raise TypeError("Number of divisions along x axis must be an integer.")
    x_range = (min(x_range), max(x_range))
    y_range = (min(y_range), max(y_range))
    grid_width = (x_range[1]-x_range[0])/n
    if "m" in kwargs:
        m = kwargs["m"]
        grid_height=(y_range[1]-y_range[0])/kwargs["m"]
    else:
        m = ceil((y_range[1]-y_range[0])/grid_width)
        y_range=(y_range[0],y_range[0]+grid_width*m)
        grid_height=grid_width
    points=zeros((n+1,m+1,3))+[x_range[0],y_range[0],0]
    for i in range(n+1):
        for j in range(m+1):
            points[i,j]+=[i*grid_width,j*grid_height,funct(x_range[0]+i*grid_width,y_range[0]+j*grid_height)]
    return points

class Tesselation(Composite):

    def __init__(self, xyz_array, *args, **kwargs):
        if xyz_array.__class__ is not array(0).__class__:
            raise TypeError("Input must be in the form of an array")
        if xyz_array.shape[2] !=3:
            raise ValueError("Array must have depth 3. Not correct shape.")
        super().__init__()
        self.funct_array=xyz_array
        self.grid_dims=(xyz_array.shape[0]-1,xyz_array.shape[1]-1)
        self.object_subtype="Tesselation"
        count=1
        for i in prange(self.grid_dims[0]):
            for j in prange(self.grid_dims[1]):
                p1 = SVec(self.funct_array[i,j])
                p2 = SVec(self.funct_array[i+1,j])
                p3 = SVec(self.funct_array[i,j+1])
                p4 = SVec(self.funct_array[i+1,j+1])
                if "down" in args:
                    self.COMPONENTS["Tess_{}".format(count)] = Triangle(p1, p2, p4)
                    count += 1
                    self.COMPONENTS["Tess_{}".format(count)] = Triangle(p1, p3, p4)
                    count += 1
                elif "random" in args:
                    if random()>0.5:
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
                    self.COMPONENTS["Tess_{}".format(count)] = Triangle(p1,p2,p3)
                    count += 1
                    self.COMPONENTS["Tess_{}".format(count)] = Triangle(p2, p3, p4)
                    count += 1
        self.t_number=count-1
        x_min, y_min , z_min = amin(self.funct_array, (0, 1))
        x_max, y_max, z_max = amax(self.funct_array, (0, 1))
        self.bounding_box=AABB(SVec(array([x_min,y_min,z_min])),SVec(array([x_max,y_max,z_max])))
        self.component_count = len(self.COMPONENTS)

    def __str__(self):
        return "Tesselation(Count: {})".format(self.t_number)


def function_from_samples(xy_coords,z_grid):
    if len(xy_coords) != len(z_grid):
        ValueError("Number of inputs is not the same as the number of outputs.")
    elif len(set(xy_coords)) != len(z_grid):
        ValueError("Multiple outputs for at least one input")
    def lookup_funct(x_coord,y_coord):
        xy_coord=(x_coord,y_coord)
        return z_grid[xy_coords.index(xy_coord)]
    return lookup_funct

# FRACTAL SURFACE

def fractal_noise(scale,iter,fract_dim,*args, **kwargs):
    if "shape" in kwargs:
        pass
    else:
        sd = scale*exp(-iter*fract_dim)
        return norm(0,sd)

def fractal_array(detail_level,scale,fractal_dim,bounds,*args,**kwargs):
    if detail_level.__class__ is not int:
        raise TypeError("Detail level must be a positive integer.")
    else:
        if detail_level<1:
            raise TypeError("Detail level must be a positive integer.")
    size=2**(detail_level+1)+1
    xs=[bounds[0][0]+i*(bounds[0][1]-bounds[0][0])/(size-1) for i in range(size)]
    x_grid=multiply(xs,ones((size,size)))
    ys=[bounds[1][0]+i*(bounds[1][1]-bounds[1][0])/(size-1) for i in range(size)]
    y_grid = multiply(ys, ones((size, size))).transpose()
    z_grid=zeros((size,size))
    if "seeds" in kwargs:
        seeds = kwargs["seeds"]
        if seeds.__class__ is not list:
            raise TypeError("Seeds keyword value be a list of values.")
        else:
            if len(seeds)!=4:
                raise ValueError("Seeds keyword value must be a list of length 4.")
    else:
        s = 1
        seeds=[fractal_noise(scale,s,fractal_dim,*args,**kwargs) for i in range(4)]
    z_grid[0,0], z_grid[0,-1], z_grid[-1,0], z_grid[-1,-1]=seeds[0],seeds[1],seeds[2],seeds[3]
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
        z_grid+=kwargs["level"]
    frarray=stack([x_grid,y_grid,z_grid],axis=-1)
    if "preview" in args:
        plot_3d_function(xs,ys, z_grid,*args,**kwargs)
    return frarray


# GRILLE

# BLOB

class MixedComposite(Composite):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.object_subtype = "MixedComposite"
        if "name" in kwargs:
            self.name = kwargs["name"]
        else:
            self.name = None
        if "sets" in kwargs:
            if kwargs["sets"].__class__ is not dict:
                raise TypeError("Sets keyword must correspond to a dictionary")
            else:
                for set in kwargs["sets"]:
                    self.add_set(kwargs["sets"][set],set_name=set)
        self.bounding_box=joint_AABB(list(kwargs["sets"].values))
        self.object_count = 0
        for set in self.COMPONENTS:
            self.component_count += len(set)

    def __str__(self):
        return "MixedComposite(" + str(self.anchor.value) + ")"

    def __repr__(self):
        desc = "Name: {}\nClass: {}\nParameters: {}\nProperties: {}\n Components Sets: ".format(self.name,
                                                                                               self.object_type,
                                                                                               self.PARAMS,
                                                                                               self.PROPERTIES)
        if self.COMPONENTS == {}:
                desc += "None"
        else:
            for set in self.COMPONENTS:
                desc += "\n" + str(set) + ": " + str(self.COMPONENTS[set]+"\n")
                for c in self.COMPONENTS[set]:
                    desc += "\n" + str(c) + ": " + str(self.COMPONENTS[set][c])
        return desc

    # Add objects

    def add_set(self,other,**kwargs):
        if other.__class__ in [Composite,Tetrahedron,Tesselation]:
            if "set_name" in kwargs:
                self.COMPONENTS[kwargs["set_name"]]=other.COMPONENTS
                self.object_count += other.object_count
            else:
                self.COMPONENTS[other.name]=other.COMPONENTS
                self.object_count += other.object_count
        elif other.__class__ in recognised_objects:
            if "set_name" in kwargs:
                self.COMPONENTS[kwargs["set_name"]] = {kwargs["set_name"]: other}
                self.object_count += 1
            else:
                self.COMPONENTS[other.name] = {other.name: other}
                self.object_count += 1
        else:
            raise TypeError("Object not recognised.")
        return self

    # Transformation methods

    def shift(self, vector):
        if vector.__class__ != SVec:
            raise TypeError("Shift vector must be an SVec object.")
        else:
            self.anchor += vector
            self.PARAMS["location"] = self.anchor
            for set in self.COMPONENTS:
                for c in self.COMPONENTS[set]:
                    self.COMPONENTS[set][c].shift(vector)
        return self

    def rotate_object(self, rotation):
        self.rotation = rotation
        self.PARAMS["rotation"] = self.rotation
        for set in self.COMPONENTS:
            for c in self.COMPONENTS[set]:
                self.COMPONENTS[set][c].rotate_object(rotation)

    # Intersection methods # Only returns closest

    def ison(self, point):
        for set in self.COMPONENTS:
            for c in self.COMPONENTS[set]:
                if self.COMPONENTS[set][c].ison(point):
                    return True
        return False

    def gen_normal(self, point, *args):
        for set in self.COMPONENTS:
            for c in self.COMPONENTS[set]:
                if self.COMPONENTS[set][c].ison(point):
                    return self.COMPONENTS[set][c].gen_normal(point)

    def add_property(self, set_name, prop_name, prop_value,*args):
        prop_name = str(prop_name)
        if set_name not in self.COMPONENTS:
            raise ValueError("Component set not found.")
        if prop_name in self.PROPERTIES.keys():
            if "verbose" in args:
                print("{} rewritten from '{}' to '{}'".format(prop_name, self.PROPERTIES[prop_name], prop_value))
            self.PROPERTIES[str(prop_name)] = prop_value
        else:
            self.PROPERTIES[str(prop_name)] = prop_value
        for set in self.COMPONENTS:
            for c in self.COMPONENTS[set]:
                self.COMPONENTS[set][c].add_property(prop_name, prop_value)

    def change_scatterer(self, set_name, new_scat):
        if set_name not in self.COMPONENTS:
            raise ValueError("Component set not found.")
        if new_scat.__class__ != Scatterer:
            raise TypeError("Scatterer must be an instance of Scatterer class.")
        else:
            self.add_property("scatterer", new_scat)

# =============================================================================

def plane_from_normal(normal):
    if normal.__class__ is not SVec:
        raise TypeError("Normal must be SVec")

def plane_from_ray(ray):
    pass


def plane_from_angles(theta,phi,c,*args):
    if theta.__class__ is not Angle:
        theta=Angle(theta)
    if phi.__class__ is not Angle:
        phi=Angle(phi)
    if theta==Angle(90):
        return Wall(1,0,-c)
    if phi==Angle(90):
        return Wall(0,1,-c)
    return Plane(-tan(theta.radians()),-tan(phi.radians()),-c)

def tangent_plane(point, surface,*args, **kwargs):
    normal=surface.gen_normal(point,*args,**kwargs)
    dotpn=float(normal.dot(point))
    if normal.z!=0:
        return Plane(normal.x/normal.z,normal.y/normal.z,-dotpn/normal.z)
    elif normal.x !=0:
        return Wall(1,normal.y/normal.x,-dotpn/normal.x)
    else:
        return Wall(0,1,-dotpn/normal.y)

def angle_with_plane(ray, plane):
    normal=plane.unit_normal
    ray_d=ray.direction
    ray_d.normalise()
    prod=ray_d.dot(normal)
    angle=float(arcsin(abs(prod)))
    return Angle(angle,"from_radians")

# OBJECT FILES

class ObjFile:
    def __init__(self,file,*args,**kwargs):
        if file.__class__ != str:
            raise TypeError("Input must be a string.")
        elif file[-4:] !=".obj":
            raise ValueError("Not an OBJ file.")
        self.file_path=file
        f=open(self.file_path,"r")
        self.vertices=[]
        self.faces=[]
        for line in f:
            if line[0]=="v":
                v=[float(p) for p in line[2:].split()]
                self.vertices.append(v)
            elif line[0]=="f":
                h = [int(float(e.split("/")[0])) for e in line[2:].split()]
                self.faces.append(h)
            else:
                pass
        f.close()
        self.vertex_count=len(self.vertices)
        self.face_count=len(self.faces)

    def __str__(self):
        return "Obj(Vertices:{}, Faces:{})".format(self.vertex_count,self.face_count)

    def __repr__(self):
        string="VERTICES\n"
        for v in self.vertices:
            string+=str(v)
            string+="\n"
        string+="\nFACES\n"
        for f in self.faces:
            string+=str(f)
            string+="\n"
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

    def copy(self,**kwargs):
        kwargs["file_path"]=copy(self.file_path[:-4])+"_copy.obj"
        self.write_obj(**kwargs)
        return ObjFile(kwargs["file_path"])

    def face_index(self,face_desc):
        if face_desc in self.faces:
            return self.faces.index(face_desc)
        else:
            print("Face not in object. Check for tyoos.")
            return None

    def remove_face(self,index):
        self.faces.pop(index)

    def vertex_index(self,vertex_desc):
        if vertex_desc in self.faces:
            return self.vertices.index(vertex_desc)
        else:
            print("Vertex not in object. Check for tyoos.")
            return None

    def remove_vertex(self,index):
        self.vertices.pop(index)
        self.faces=[f for f in self.faces if index+1 not in f]
        self.faces=[[v-1 if v>index else v for v in f] for f in self.faces]

    def write_obj(self,**kwargs):
        if "file_path" in kwargs:
            self.file_path=kwargs["file_path"]
        else:
            raise ValueError("file_path keyword argument must be supplied.")
        f=open(self.file_path,"w")
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

    def change_vertex(self,index,new_pos):
        if new_pos.__class__ != list:
            raise TypeError("New point need to be a list of three integers or floats. This is not a list.")
        if len(new_pos)!=3:
            raise TypeError("New point need to be a list of three integers or floats. Length is not 3.")
        else:
            if False in [c.__class__ in [int,float] for c in new_pos]:
                raise TypeError("New point need to be a list of three integers or floats. Entries are correct.")
        self.vertices[index]=new_pos

    def merge(self,other,*args,**kwargs):
        if other.__class__ != ObjFile:
            raise TypeError("Only two ObjFile instances can be merged.")
        else:
            new_verts=list(other.vertices)
            new_faces=[[c+self.vertex_count for c in f] for f in other.faces]
            self.vertices+=new_verts
            self.faces+=new_faces
            self.vertex_count+=other.vertex_count
            self.face_count+=other.face_count
        if "file_path" in kwargs:
            self.write_obj(**kwargs)
        return self

    def scale(self,factor):
        self.vertices = [[c*factor for c in v] for v in self.vertices]
        return self

    def shift(self,vector):
        if vector.__class__ is not SVec:
            raise TypeError("Shift argument must be an SVec object.")
        vert_array=array(self.vertices)
        print("Before shift",vert_array[:10])
        self.vertices=list(vert_array+vector.value)
        print("After shift",self.vertices[:10])
        return self

# OBJ CONVERSION TOOLS

def tess_to_OBJ(tess,file_path,**kwargs):
    if tess.__class__ != Tesselation:
        raise TypeError("Input needs to be a Tesselation instance.")
    else:
        file=open(file_path,"w")
        count=0
        vertices={}
        faces=[]
        # Create lists of vertices and faces
        for c in tess.COMPONENTS:
            ps=[tess.COMPONENTS[c].p1.value,tess.COMPONENTS[c].p2.value,tess.COMPONENTS[c].p3.value]
            for p in ps:
                if str(p) not in vertices:
                    vertices[str(p)]=[count+1,p]
                    count+=1
        for c in tess.COMPONENTS:
            ps=[str(tess.COMPONENTS[c].p1.value),str(tess.COMPONENTS[c].p2.value),str(tess.COMPONENTS[c].p3.value)]
            faces.append([vertices[p][0] for p in ps])
        # Write to .obj file
        if "comments" in kwargs:
            file.write("#"+kwargs["comments"]+"\n\n")
        file.write("#VERTICES\n\n")
        for v in vertices:
            file.write("v "+str(vertices[v][1][0])+" "+str(vertices[v][1][1])+" "+str(vertices[v][1][2])+"\n")
        file.write("\n#FACES\n\n")
        for f in faces:
            file.write("f "+str(f[0])+" "+str(f[1])+" "+str(f[2])+"\n")
        file.close()
    return ObjFile(file_path)

def implicit_to_OBJ(implicit,sample_width,bounds,file_path,*args,**kwargs):
    if implicit.__class__ != Implicit:
        raise TypeError("Input must be an instance of Implicit.")
    else:
        # CREATE ARRAY FROM FUNCTION
        sample_array=array_from_implicit(implicit,sample_width,bounds,*args,**kwargs)
        verts, faces, _ , _ = marching_cubes(sample_array,0)
        verts *= sample_width
        verts += array([bounds[i][0] for i in range(3)])

        # REMOVE DUPLICATES
        unique_verts=[]
        renewed_faces=faces.copy()
        counter=0
        for v in verts:
            if list(v) not in unique_verts:
                unique_verts.append(list(v))
                counter+=1
            else:
                previous=unique_verts.index(list(v))
                for f in range(len(faces)):
                    for c in range(3):
                        if faces[f][c] > counter:
                            renewed_faces[f][c] -= 1
                        if faces[f][c] == counter:
                            renewed_faces[f][c] = previous
                counter+=1
        faces = [[v+1 for v in f] for f in renewed_faces if len(set(f))==3]
        max_vertex=len(unique_verts)
        faces = [f for f in faces if True not in [f[v] > max_vertex for v in range(3)]]
        # WRITE TO OBJ FILE
        f=open(file_path,"w")
        if "comments" in kwargs:
            f.write("#" + str(kwargs["comments"]) +" " + implicit.formula + "\n\n")
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
    implicit_obj=ObjFile(file_path,*args,**kwargs)

    # SHOW PLOT
    if "show" in args:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        xyz_centre=[(bounds[i][1]+bounds[i][0])/2 for i in range(3)]
        xyz_spans=[1.1*(bounds[i][1]-bounds[i][0])/2 for i in range(3)]
        cube_width=max(xyz_spans)

        ax.set_xlim(xyz_centre[0]-cube_width, xyz_centre[0]+cube_width)
        ax.set_ylim(xyz_centre[0]-cube_width, xyz_centre[0]-cube_width)
        ax.set_zlim(xyz_centre[0]-cube_width, xyz_centre[0]-cube_width)

        plt.tight_layout()
        plt.show()

    return implicit_obj

def funct_to_OBJ(funct,sample_width, bounds,*args,**kwargs):
    if "func_type" not in kwargs:
        raise ValueError('Keyword "func_type" bust be defined')
    elif kwargs["func_type"] not in ["intrinsic","explicit"]:
        raise ValueError("Function must be an explicit or intrinsic function")
    if "file_path" not in kwargs:
        raise ValueError('Keyword "file_path" must be defined.')
    else:
        n = int((bounds[0][1] - bounds[0][0]) // sample_width)
        x_range = (bounds[0][0], bounds[0][0] + n * sample_width)
        y_range = (bounds[1][0], bounds[1][1])
        if kwargs["func_type"] == "explicit":
            if "formula" in kwargs:
                return tess_to_OBJ(Tesselation(array_from_explicit(funct, x_range, y_range, n)), kwargs["file_path"],comments=kwargs["formula"])
            else:
                return tess_to_OBJ(Tesselation(array_from_explicit(funct, x_range, y_range, n)), kwargs["file_path"],*args,**kwargs)
        else:
            if "formula" in kwargs:
                return implicit_to_OBJ(Implicit(funct,kwargs["formula"],kwargs["file_path"],*args,**kwargs),sample_width,bounds,
                                       *args,{**kwargs,"comments":kwargs["formula"]})
            else:
                return implicit_to_OBJ(Implicit(funct, "custom_function", kwargs["file_path"],*args, **kwargs), sample_width, bounds,
                                       *args, **kwargs)

def array_to_OBJ(sample_array,*args,**kwargs):
    if sample_array.__class__ is not array(0).__class__:
        raise TypeError("Input must be an x,y,z array. This is not an array.")
    if "file_path" not in kwargs:
        raise ValueError('Keyword "file_path" must be defined.')
    else:
        return tess_to_OBJ(Tesselation(sample_array), kwargs["file_path"])

# MESH

class Mesh(Composite):
    def __init__(self,obj,*args,**kwargs):
        if obj.__class__ is str:
            self.OBJ=ObjFile(obj,*args,**kwargs)
        elif obj.__class__ is Tesselation:
            self.OBJ=ObjFile(tess_to_OBJ(obj,*args,**kwargs).file_path,*args,**kwargs)
        elif obj.__class__ is Implicit:
            if "sample_width" not in kwargs:
                raise ValueError("Keyword argument 'grid_size' must be supplied.")
            if "bounds" not in kwargs:
                raise ValueError("Keyword argument 'bounds' must be supplied.")
            self.OBJ=ObjFile(implicit_to_OBJ(obj,*args,**kwargs).file_path,*args,**kwargs)
        elif obj.__class__ is ObjFile:
            self.OBJ=obj
        elif (isfunction(obj) or isbuiltin(obj) or obj.__class__ is partial(print).__class__):
            if "sample_width" not in kwargs:
                raise ValueError("Keyword argument 'grid_size' must be supplied.")
            if "bounds" not in kwargs:
                raise ValueError("Keyword argument 'bounds' must be supplied.")
            self.OBJ=ObjFile(funct_to_OBJ(obj,*args,**kwargs))
        else:
            raise TypeError("Input is not of an accepted type. Should be one of string, Implicit, ObjFile, Tesselation.")
        super().__init__(*args,**kwargs)
        self.object_subtype="Mesh"
        self.mesh=Poly3DCollection(array(self.OBJ.vertices)[array(self.OBJ.faces)-1])
        self.vertex_count=self.OBJ.vertex_count
        self.component_count=0
        self.COMPONENTS={}
        for f in self.OBJ.faces:
            self.component_count+=1
            [p1,p2,p3]=[SVec(array(self.OBJ.vertices[f[i]-1])) for i in range(3)]
            self.COMPONENTS["Face_" + str(self.component_count)]=Triangle(p1, p2, p3)
        self.anchor=SVec(array(self.OBJ.vertices[0]))
        self.PARAMS["location"]=self.anchor
        def min_max(lst):
            return [min(lst),max(lst)]
        [[min_x, max_x],[min_y,max_y],[min_z,max_z]] = [min_max([v[i] for v in self.OBJ.vertices]) for i in range(3)]
        c_1= SVec(array([min_x,min_y,min_z]))
        c_2 = SVec(array([max_x, max_y, max_z]))
        self.bounding_box=AABB(c_1,c_2)


    def rotate_object(self,rotation,*args,**kwargs):
        if rotation.__class__ != Rotation:
            TypeError("Argument must be an instance of Rotation.")
        self.OBJ.vertices = rotation.rotate_point(array(self.OBJ.vertices),*args,**kwargs)
        self.mesh = Poly3DCollection(array(self.OBJ.vertices)[array(self.OBJ.faces) - 1])
        if self.rotation == None:
            self.rotation = rotation
        else:
            self.rotation += rotation
        self.PROPERTIES["rotation"] = rotation
        def min_max(lst):
            return [min(lst),max(lst)]
        [[min_x, max_x],[min_y,max_y],[min_z,max_z]] = [min_max([v[i] for v in self.OBJ.vertices]) for i in range(3)]
        c_1= SVec(array([min_x,min_y,min_z]))
        c_2 =SVec(array([max_x, max_y, max_z]))
        self.bounding_box=AABB(c_1,c_2)
        return self

    def show(self,*args,**kwargs):
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

        max_span=max([x_span,y_span,z_span])

        ax.set_xlim(x_mean-max_span,x_mean+max_span)
        ax.set_ylim(y_mean-max_span,y_mean+max_span)
        ax.set_zlim(z_mean-max_span,z_mean+max_span)
        plt.tight_layout()
        if "im_path" in kwargs:
            plt.savefig(kwargs["im_path"])
        plt.show()

    def save_im(self,im_path):
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

        max_span=max([x_span,y_span,z_span])

        ax.set_xlim(x_mean-max_span,x_mean+max_span)
        ax.set_ylim(y_mean-max_span,y_mean+max_span)
        ax.set_zlim(z_mean-max_span,z_mean+max_span)
        plt.tight_layout()
        plt.savefig(im_path)
        plt.close(fig)

    def write(self,**kwargs):
        self.OBJ.write_obj(**kwargs)



# OBJECTS IN TIME - these would be 4D objects for which we must have a definition, intersection test

recognised_objects=[Sphere, Object, Plane, Wall, Cylinder, Rod, Triangle, Disc, Tetrahedron, Tesselation, Surface, Mesh]

