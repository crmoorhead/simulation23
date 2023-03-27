from numpy import clip, array, dot, cross, dtype, zeros, stack, ones, transpose as trans,\
    moveaxis, concatenate as concat, linspace, set_printoptions, all, any, inf
from numpy.linalg import norm
from numpy.random import normal as ndist
from toolbox import show_image
from cv2 import imwrite, merge
from copy import deepcopy
from math import pi
from scipy.spatial.transform import Rotation as R
from numba import prange
from math import floor, ceil, sin
from inspect import isfunction,signature, isbuiltin
from numpy.random import normal, poisson, random, rayleigh, gamma, exponential, randint
from functools import partial

set_printoptions(precision=3)

# ANGLE CLASS

DEG_SYM= u'\N{DEGREE SIGN}'

class Angle:
    def __init__(self,angle,*args):
        if angle.__class__ in [float, int]:
            if "from_radians" in args:
                angle*=180/pi
                self.value=float(angle%360)
            else:
                self.value = angle
        elif angle.__class__==Angle:
            self.value=angle.value
        else:
            raise TypeError("Angle must be an integer or float.")

    def __str__(self):
        return str(round(self.value,4))+DEG_SYM

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __neg__(self):
        return Angle(-self.value)

    def __add__(self, other):
        if other is None:
            return self
        elif other.__class__ is Angle:
            return Angle(self.value+other.value)
        elif other.__class__ in [int,float]:
            return Angle(self.value+other)
        else:
            raise TypeError("")

    # Angle methods

    def compliment(self):
        if self.value<=90:
            return Angle(90 - self.value)
        else:
            print("Angle larger than 90 deg.")
            return self

    def supplement(self):
        if self.value<=180:
            return Angle(180 - self.value)
        else:
            print("Angle larger than 180 deg.")
            return self

    def radians(self):
        pi = 3.141592653589793
        return self.value * pi / 180

    def type(self):
        if self.value<90:
            return "acute"
        elif self.value>180:
            return "reflex"
        else:
            return "obtuse"

# We set up the SVec (spacial vector) and CVec (color vector) classes.
# SVec has:
# Properties: value - the R^3 representation of the vector
#             x, y, z - calls the constituent parts of the vector
#
# Methods (overloaded): Printed form will be Spacial(x,y,z)
#                       +, +=, -, -=, *, *=, /, abs()
# Further methods: dot(SVec,SVec) - dot product between two SVec instances.
#                  cross(SVec,SVec) - cross product between two SVec instances. Output is an SVec instance.
#                  normalise(SVec) - normalises the SVec instance by changing the original object.
# notes: Division is elementwise or by a scalar. Division by zero produces standard error. Not defined for other/SVec in general.
#        All operations allow broadcasting in that a constant will be applied to all elements.
#        abs(SVec) gives the magnitude of the vector.
#
# TO DO: Return angle between two SVec instances.

class SVec:
    def __init__(self,coords):
        if coords.__class__ != array(1).__class__:
            raise TypeError("SVec triple must be a numpy 1D array. This is not an array.")
        if coords.shape[0]!=3 or len(coords.shape)!=1:
            raise TypeError("Vec3 triple must be a 1D numpy array of length 3. The array is the wrong shape.")
        self.value = array(coords,dtype=float)
        self.x = float(self.value[0])
        self.y = float(self.value[1])
        self.z = float(self.value[2])

    def __str__(self):
        return "Spacial("+str(round(self.x,6))[:6]+", "+str(round(self.y,6))[:6]+", "+str(round(self.z,6))[:6]+")"

    def __repr__(self):
        return "Spacial({})".format(self.value,3)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # Arithmetic operations

    def __add__(self, other):
        if other.__class__==self.__class__:
            return SVec(self.value+other.value)
        else:
            return SVec(self.value+other)

    def __radd__(self, other):
        if other.__class__!=self.__class__:
            return SVec(self.value+other)

    def __iadd__(self, other):
        if other.__class__==self.__class__:
            self.value += other.value
            return SVec(self.value)
        else:
            self.value += other
            return SVec(self.value)

    def __sub__(self, other):
        if other.__class__==self.__class__:
            return SVec(self.value-other.value)
        else:
            return SVec(self.value-other)

    def __rsub__(self, other):
        if other.__class__!=self.__class__:
            return SVec(other - self.value)

    def __isub__(self, other):
        if other.__class__==self.__class__:
            self.value -= other.value
            return SVec(self.value)
        else:
            self.value -= other
            return SVec(self.value)

    def __mul__(self, other):
        if other.__class__==self.__class__:
            return SVec(self.value*other.value)
        else:
            return SVec(self.value*other)

    def __rmul__(self, other):
        if other.__class__!=self.__class__:
            return SVec(other*self.value)

    def __imul__(self, other):
        if other.__class__==self.__class__:
            self.value *= other.value
            return SVec(self.value)
        else:
            self.value *= other
            return SVec(self.value)

    def __truediv__(self, other):
        if other.__class__==self.__class__:
            return SVec(self.value/other.value)
        else:
            return SVec(self.value/other)

    def __neg__(self):
        return SVec(array([0,0,0]))-SVec(self.value)


    # Class methods

    def __abs__(self):
        return (self.x**2+self.y**2+self.z**2)**0.5

    def dot(self,other):
        if other.__class__==self.__class__:
            return dot(self.value,other.value)
        else:
            return NotImplemented

    def cross(self,other):
        if other.__class__==self.__class__:
            return SVec(cross(self.value,other.value))
        else:
            return NotImplemented

    def normalise(self):
        mag=abs(self)
        if mag == 0:
            raise ValueError("The zero vector cannot be normalised.")
        self.x= self.x/mag
        self.y = self.y /mag
        self.z = self.z /mag
        self.value=array([self.x,self.y,self.z])
        return self

    def ismultiple(self,vector,*args):
        if vector.__class__!=self.__class__:
            raise TypeError("Comparison must be made with another SVec object")
        else:
            ratio=None                       # Set ratio to None initially
            if any(self.value==array([0,0,0])):   # If one of the entries of one vector is zero
                found=None
                i=0
                while found==None and i<3:
                    if self.value[i]!=0:
                        ratio=vector.value[i]/self.value[i]
                        break
                    i+=1
                if ratio==None:
                    ratio=0
            else:
                ratio=vector.value[0]/self.value[0]
            if "give_multiple" in args:
                if all(vector.value==ratio*self.value):
                    return ratio
                else:
                    return None
                if False in [vector.value[i]==ratio*self.value[i] for i in range(3)]:
                    return False
                else:
                    return True

    def isparallel(self,other):
        if other.__class__==Ray:
            return other.direction.ismultiple(self)
        elif other.__class__==self.__class__:
            return other.ismultiple(self)
        else:
            raise TypeError("Can only make parallel test between two rays or a ray and a SVec object.")

    def euler_rotation(self,theta,phi,chi):
        for a in [theta,phi,chi]:
            if a.__class__ is not Angle:
                raise TypeError("All euler angles must be Angle objects.")

    def point_rotation(self,point,rotation):
        pass

    def copy(self):
        return deepcopy(self)

# Project point by collapsing stated axis

def project_axis(point,axis):
    if point.__class__ is not SVec:
        raise TypeError("Point must be an instance of SVec.")
    else:
        if axis=="x":
            return SVec(array([0,point.y,point.z]))
        if axis=="y":
            return SVec(array([point.x,0,point.z]))
        if axis=="z":
            return SVec(array([point.x,point.y,0]))

def random_vec(interval):  # Using Normal distribution
    if interval.__class__ is tuple:
        if len(interval) is not 2:
            raise ValueError("Tuple must have length 2.")
        else:
            return SVec(random(3)*(interval[1]-interval[0])+interval[0])
    elif interval.__class__ is list:
        if len(interval) != 3:
            raise ValueError("Length of list must be 3.")
        for i in interval:
            if i.__class__ is not tuple:
                raise TypeError("All intervals must be tuples of length 2. This is not a tuple.")
            if len(i) is not 2:
                raise ValueError("Each tuple must have length 2.")
        return SVec(array([random()*(interval[i][1]-interval[i][0])+interval[i][0] for i in range(3)]))
    else:
        print("Argument must be a tuple or list of 3 tuples.")

# CVec has:
# Properties: value - The RGB value of the color (an integer triple beteen
#             R, G , B -  The constituent parts of the color.
# Methods (overloaded): Prints vector in the form Color(R,G,B)
#                       +, +=, -, -=, *, *=, \
# Other methods: rlum - Relative luminance of the color.
#                plum - Percieved luminance of the color.
#                taster - Shows block of constant color in seperate window.
#                hexate - Returns the hec value of the color.
#                negative - Changes the colour to its negative value.
# notes:  All operations must be paired with a scalar and all elements are integers between 0 and 255 (inclusive). Operations automatically truncate the output.

# TO DO: If string then try hex2vec and otherwise NotImplemented
#        Produce negative of the original color
#        Return the hex color value of the color

class CVec:

    def __init__(self,RGB):
        if RGB is "random":
            RGB = randint(0, 255, (3,), dtype="uint8")
        if RGB.__class__!=array(1).__class__:
            raise TypeError("CVec triple must be a numpy 1D array. This is not an array.")
        if RGB.shape[0]!=3 or len(RGB.shape)!=1:
            raise TypeError("CVec triple must be a 1D numpy array of length 3. The array is the wrong shape.")
        if RGB.dtype!=dtype('uint8'):
            RGB=array(RGB,dtype="uint8")
        self.value = RGB
        self.R = RGB[0]
        self.G = RGB[1]
        self.B = RGB[2]

    def __str__(self):
        return "Color(" + str(self.R) + ", " + str(self.G) + ", " + str(self.B) + ")"

    def __repr__(self):
        return "Color({})".format(self.value)

    # Arithmetic operations (capped at maximum intensity)

    def __add__(self, other):
        if other.__class__==self.__class__:
            self.value=array(self.value,dtype=int)
            return CVec(clip(self.value+other.value,0,255))
        else:
            self.value = array(self.value, dtype=int)
            return CVec(clip(self.value+other,0,255))

    def __radd__(self, other):
        if other.__class__!=self.__class__:
            self.value = array(self.value, dtype=int)
            return CVec(clip(self.value+other,0,255))

    def __iadd__(self, other):
        if other.__class__==self.__class__:
            self.value = array(self.value, dtype=int)
            self.value += other.value
            return CVec(clip(self.value,0,255))
        else:
            self.value = array(self.value, dtype=int)
            self.value += other
            return CVec(clip(self.value,0,255))

    def __sub__(self, other):
        if other.__class__==self.__class__:
            self.value = array(self.value, dtype=int)
            return CVec(clip(self.value-other.value,0,255))
        else:
            self.value = array(self.value, dtype=int)
            return CVec(clip(self.value-other,0,255))

    def __rsub__(self, other):
        if other.__class__!=self.__class__:
            self.value = array(self.value, dtype=int)
            return CVec(clip(other - self.value,0,255))

    def __isub__(self, other):
        if other.__class__==self.__class__:
            self.value = array(self.value, dtype=int)
            self.value -= other.value
            return CVec(clip(self.value,0,255))
        else:
            self.value = array(self.value, dtype=int)
            self.value -= other
            return CVec(clip(self.value,0,255))

    def __mul__(self, other):
        if other.__class__==float or other.__class__==int:
            self.value = array(self.value, dtype=int)
            return CVec(clip(self.value*other,0,255))
        else:
            return NotImplemented

    def __rmul__(self, other):
        if other.__class__==float or other.__class__==int:
            self.value = array(self.value, dtype=int)
            return CVec(clip(self.value*other,0,255))
        else:
            return NotImplemented

    def __imul__(self, other):
        if other.__class__==float or other.__class__==int:
            self.value = array(self.value, dtype=int)
            self.value *= other
            return CVec(clip(self.value*other,0,255))
        else:
            return NotImplemented

    def __truediv__(self, other):
        if other.__class__==float or other.__class__==int:
            self.value = array(self.value, dtype=int)
            return CVec(self.value/other)
        else:
            return NotImplemented

    # Class methods: Display colour, negative,

    def rlum(self):
        self.rlum = array(0.2126 * self.R + 0.7152 * self.G + 0.0722 * self.B, dtype="uint8")  # Relative luminance
        return self.rlum

    def plum(self):
        self.plum = array((0.299 * self.R ** 2 + 0.587 * self.G ** 2 + 0.114 * self.B ** 2) ** 0.5, dtype="uint8")  # Perceived luminance
        return self.plum

    def taster(self):
        dims=(150,150)
        show_image(stack((array(ones(dims) * self.B), array(ones(dims) * self.G), array(ones(dims) * self.R)), axis=-1)/256)

    def negative(self):
        return CVec(array([255-self.R,255-self.G,255-self.B]))

    def hexate(self):
        return hex(self.R)[2:].zfill(2)+hex(self.G)[2:].zfill(2)+hex(self.B)[2:].zfill(2)

    def copy(self):
        return deepcopy(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

class GVec():

    def __init__(self,i,*args,**kwargs):
        if i.__class__ ==int:
            self.value=array([i],dtype="uint8")
        elif i.__class__==float:
            self.value=array([i],dtype="uint8")
        elif i.__class__==CVec:
            self.value=array([0.30 * i.R + 0.59 * i.G + 0.11 * i.B], dtype="uint8")
        elif i.__class__ is array(1).__class__:
            if i.shape==(1,) or i.shape==(1):
                self.value=array(i,dtype="uint8")
            else:
                raise TypeError("Array object input is not of correct shape.")
        else:
            raise TypeError("Input value is not of a viable type.")

    def __str__(self):
        return "Greyscale(" + str(self.value) + ")"

    def __repr__(self):
        return "Greyscale({})".format(self.value)

    # Arithmetic operations (capped at maximum intensity)

    def __add__(self, other):
        if other.__class__==self.__class__:
            return GVec(clip(self.value+other.value,0,255))
        else:
            return GVec(clip(self.value+other,0,255))

    def __radd__(self, other):
        if other.__class__!=self.__class__:
            return GVec(clip(self.value+other,0,255))

    def __iadd__(self, other):
        if other.__class__==self.__class__:
            self.value += other.value
            return GVec(clip(self.value,0,255))
        else:
            self.value += other
            return GVec(clip(self.value,0,255))

    def __sub__(self, other):
        if other.__class__==self.__class__:
            return GVec(clip(self.value-other.value,0,255))
        else:
            return GVec(clip(self.value-other,0,255))

    def __rsub__(self, other):
        if other.__class__!=self.__class__:
            return GVec(clip(other - self.value,0,255))

    def __isub__(self, other):
        if other.__class__==self.__class__:
            self.value -= other.value
            return GVec(clip(self.value,0,255))
        else:
            self.value -= other
            return GVec(clip(self.value,0,255))

    def __mul__(self, other):
        if other.__class__==float or other.__class__==int:
            return GVec(clip(self.value*other,0,255))
        else:
            return NotImplemented

    def __rmul__(self, other):
        if other.__class__==float or other.__class__==int:
            return GVec(clip(self.value*other,0,255))
        else:
            return NotImplemented

    def __imul__(self, other):
        if other.__class__==float or other.__class__==int:
            self.value *= other
            return GVec(clip(self.value*other,0,255))
        else:
            return NotImplemented

    def __truediv__(self, other):
        if other.__class__==float or other.__class__==int:
            return GVec(self.value/other)
        else:
            return NotImplemented

    def copy(self):
        return deepcopy(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # Other methods

    def taster(self):
        dims=(150,150)
        show_image(stack((array(ones(dims) * self.value), array(ones(dims) * self.value), array(ones(dims) * self.value)), axis=-1)/256)

    def negative(self):
        return GVec(255-self.value)

    def hexate(self):
        return hex(self.value[0])[2:].zfill(2)

def CVec2GVec(color):
    if color.__class__ is not CVec:
        raise TypeError("Conversion must be from an instance of CVec.")
    return GVec(float(0.30 * color.R + 0.59 * color.G + 0.11 * color.B))

# Class for color canvas

class ColCanvas:
    def __init__(self,dims,**kwargs):
        if dims.__class__!=tuple:
            print("Input must be a length 2 tuple.")
            return NotImplemented
        if  len(dims)!=2:
            print("Input must be a length 2 tuple.")
            return NotImplemented
        if "color" in kwargs:
           if kwargs["color"].__class__==CVec:
               self.default_color=kwargs["color"].value
               self.image=stack((array(ones(dims) * self.default_color[0]), array(ones(dims) * self.default_color[1]), array(ones(dims) * self.default_color[2])), axis=-1)
           else:
               print("Color keyword argument must be a CVec object.")
               return NotImplemented
        else:
            self.default_color = array([0,0,0])
            self.image = zeros(dims + tuple([3]), dtype="uint8")

        self.dims=dims

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def assign_pixel(self,location,color):
        if color.__class__==CVec:
            if location.__class__==list:
                if len(location)==2 and False not in [location[i].__class__==int for i in range(2)]:
                    self.image[location[0],location[1],:]=color.value
                elif len(location)==2 and False not in [location[i].__class__==list for i in range(2)]:
                    self.image[location[0][0]:location[0][1],location[1][0]:location[1][1]]=color.value   # Y-coords, then X-coords from bottom
                else:
                    raise TypeError("Input must be a list of 2 integers or a list of 2 lists of 2 integers")
            else:
                raise TypeError("Input must be a list of 2 integers or a list of 2 lists of 2 integers")
        else:
            return NotImplemented

    def show(self):       # Print and display method for canvas instances
        show_image(self.image[::-1]/255)

    def save_canvas(self,save_path):        # Saves to file
        imwrite(save_path, self.image[::-1])

    def change_color(self,color,):
        if color.__class__!=CVec:
            raise TypeError("Input must be a CVec instance.")
        else:
            self.default_color=color.value
            self.image=stack((array(ones(self.dims) * self.default_color[0]), array(ones(self.dims) * self.default_color[1]),
                              (ones(self.dims) * self.default_color[2])), axis=-1)

    def grad_easel(self, flow, *args):
        self.default_color="Gradient"
        if flow.__class__ != list:
            raise TypeError("Input should be a list of 2 or 4 CVec objects.")
        if False in [flow[i].__class__==CVec for i in range(len(flow))]:
            raise TypeError("Input should be a list of 2 or 4 CVec objects.")
        if len(flow) == 2:  # 2 or 4 colours
            if "lr" in args:
                gradients = [[linspace(flow[0].value[i], flow[1].value[i], self.dims[1])] for i in range(3)]
                self.image = merge([ones(self.dims) * gradients[i] for i in range(3)])
            else:
                gradients = [[linspace(flow[0].value[i], flow[1].value[i], self.dims[0])] for i in range(3)]
                self.image = merge([ones(self.dims) * trans(gradients[i]) for i in range(3)])

        elif len(flow) == 4:
                [tl, tr, bl, br] = [flow[i] for i in range(4)]
                left_b2t = [linspace(bl.value[i], tl.value[i], self.dims[0]) for i in range(3)]
                right_b2t = [linspace(br.value[i], tr.value[i], self.dims[0]) for i in range(3)]
                self.image = array([[concat([linspace(left_b2t[k][j], right_b2t[k][j], self.dims[1])], axis=0)
                                for j in range(self.dims[0])]
                               for k in range(3)])[::-1]
                self.image = moveaxis(self.image, 0, 2)
        else:
            raise TypeError("Input should be a list of 2 or 4 CVec objects.")

class GSCanvas(ColCanvas):

    def __init__(self, dims, **kwargs):
        if dims.__class__ != tuple:
            print("Input must be a length 2 tuple.")
            return NotImplemented
        if len(dims) != 2:
            print("Input must be a length 2 tuple.")
            return NotImplemented
        if "shade" in kwargs:
            if kwargs["shade"].__class__ == GVec:
                self.default_shade = kwargs["shade"].value
                self.image = ones(dims,dtype="uint8") * self.default_shade[0]
            else:
                print("Shade keyword argument must be a GVec object.")
                return NotImplemented
        else:
            self.default_shade = array([0], dtype="uint8")
            self.image = zeros(dims + tuple([1]), dtype="uint8")

        self.dims = dims

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def assign_pixel(self, location, shade):
        if shade.__class__ == GVec:
            if location.__class__ == list:
                if len(location) == 2 and False not in [location[i].__class__ == int for i in range(2)]:
                    self.image[location[0], location[1]] = shade.value
                elif len(location) == 2 and False not in [location[i].__class__ == list for i in range(2)]:
                    self.image[location[0][0]:location[0][1], location[1][0]:location[1][1]] = shade.value # Y-coords, then X-coords from bottom
                else:
                    raise TypeError("Input must be a list of 2 integers or a list of 2 lists of 2 integers")
            else:
                raise TypeError("Input must be a list of 2 integers or a list of 2 lists of 2 integers")
        else:
            raise TypeError("Second variable should be an instance of GVec.")

    def change_shade(self, shade):
        if shade.__class__ != GVec:
            raise TypeError("Input must be a GVec instance.")
        else:
            self.default_shade = shade.value
            self.image = ones(self.dims,dtype="uint8") * self.default_shade[0]

    def grad_easel(self, flow, *args):
        self.default_shade = "Gradient"
        if flow.__class__ != list:
            raise TypeError("Input should be a list of 2 or 4 CVec objects.")
        if False in [flow[i].__class__ == GVec for i in range(len(flow))]:
            raise TypeError("Input should be a list of 2 or 4 CVec objects.")
        if len(flow) == 2:  # 2 or 4 colours
            if "lr" in args:
                gradients = [linspace(flow[0].value[0], flow[1].value[0], self.dims[1])]
                self.image = ones(self.dims) * gradients
            else:
                gradients = [linspace(flow[0].value[0], flow[1].value[0], self.dims[0])]
                self.image = ones(self.dims) * trans(gradients)

        elif len(flow) == 4:
            [tl, tr, bl, br] = [flow[i] for i in range(4)]
            left_b2t = linspace(bl.value[0], tl.value[0], self.dims[0])
            right_b2t = linspace(br.value[0], tr.value[0], self.dims[0])
            self.image = array([concat([linspace(left_b2t[j], right_b2t[j], self.dims[1])], axis=0)
                                 for j in range(self.dims[0])])
        else:
            raise TypeError("Input should be a list of 2 or 4 CVec objects.")

def hex2vec(col):
    if col.__class__==str:
        if len(col)==6:
            vec=array([int(col[2*i:2*i+2],16) for i in range(3)])
            return CVec(vec)
        elif len(col)==2:
            vec = ones((3))*int(col,16)
            return CVec(vec)
        else:
            raise TypeError("Incorrect input: Hex string must have length 2 or 6.")
    else:
        raise TypeError("Incorrect input: Must be a hex color string.")

class Ray:

    def __init__(self,origin,direction):
        if origin.__class__!=SVec or direction.__class__!=SVec:
            raise TypeError("Ray origin and direction must be SVec objects.")
        else:
            self.origin=origin
            self.direction=direction.normalise()

    def __str__(self):
        return "Ray(" + str(self.origin) + "+ " + str(self.direction) + "t)"

    def __repr__(self):
        return "Ray({}+{}t)".format(self.direction,self.origin)

    def __mul__(self, other):
        if other.__class__ in [int,float]:
            self.direction=self.direction * other
            return self
        else:
            return NotImplemented

    def __rmul__(self, other):
        if other.__class__ in [int,float]:
            self.direction=self.direction * other
            return self
        else:
            return NotImplemented

    def __imul__(self, other):
        if other.__class__ not in [int,float]:
            self.direction=self.direction * other
            return self
        else:
            return NotImplemented

    def __add__(self, other):
        if other.__class__== SVec:
            self.origin+=other
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        if other.__class__ == SVec:
            self.origin += other
            return self
        else:
            return NotImplemented

    def __iadd__(self, other):
        if other.__class__ == SVec:
            self.origin += other
            return self
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def normalise(self):
        self.direction.normalise()
        return self

    def isparallel(self,other):
        if other.__class__==self.__class__:
            return self.direction.ismultiple(other.direction)
        elif other.__class__==SVec:
            return self.direction.ismultiple(other)
        else:
            raise TypeError("Can only make parallel test between two rays or a ray and a SVec object.")

    def isonray(self,point,*args):
        if point.__class__!=SVec:
            raise TypeError("Point argument must be an SVec instance.")
        else:
            if all(point.value==self.origin.value):
                return True
            else:
                dif=point-self.origin
                mult=dif.ismultiple(self.direction)
                return mult

    def point_at_param(self,t):
        if t.__class__ not in [float, int]:
            raise TypeError("Parameter input must be integer or float.")
        else:
            if t<0:
                raise ValueError("Parameter value must be non-negative.")
            return self.origin + self.direction*t

    def interpolate(self,other,**kwargs):
        if other.__class__ is not Ray:
            raise TypeError("Input argument must be an instance of Ray.")
        if "ratio" in kwargs:
            r=kwargs["ratio"]
        else:
            r=0.5
        return Ray(other.origin*r+self.origin*(1-r),other.direction*r+self.direction*(1-r))

    def directions(self,*args):
        if "array" not in args:
            ds=[int(i/abs(i)) if i!=0 else i for i in self.direction.value]
            print(ds[0].__class__)
            return ds
        else:
            return array([int(i/abs(i)) if i!=0 else i for i in self.direction.value])

    def copy(self):
        return deepcopy(self)

# Simple Field where we are assumed to be looking straight ahead from the origin to a screen with centre positioned along the y-axis

class SimpleField:

    def __init__(self,anchor,screen_dimension, resolution,*args,**kwargs):

        if anchor.__class__ not in [int,float]:
            raise TypeError("Simple canvas centre anchor must be an integer or float.")
        if screen_dimension.__class__!=tuple:
            raise TypeError("Scope of screen must be a tuple of 2 floats or integers.")
        else:
            if len(screen_dimension)!=2:
                raise TypeError("Scope of screen must be a tuple of 2 floats or integers.")
            else:
                screen_dimension=array(screen_dimension,dtype=float)
        if resolution.__class__!=int:
            raise TypeError("Resolution must be an integer representing number of pixels per unit distance.")
        self.origin=SVec(array([0,0,0]))
        self.anchor=SVec(array([0,anchor,0]))     # anchor for simple screen is the distance along y-axis.
        self.span=tuple(screen_dimension)
        self.image_size=tuple(array(screen_dimension*resolution,dtype=int))
        self.resolution=resolution
        self.limit=1  # Sets the closest distance we can place the screen.
        self.pixel_field=self.gen_pixel_field(*args,**kwargs)
        self.pixel_size = self.pixel_field[0][1][1] - self.pixel_field[0][0][0]
        self.ray_field=self.gen_ray_field(*args,**kwargs)
        self.angle_field=self.gen_angle_field(*args,**kwargs)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # Methods that change properties

    def gen_pixel_field(self,*args,**kwargs):
        px = 1 / self.resolution
        (w, h) = self.span
        tl = [-w / 2, self.anchor.y, h / 2]
        tr = [w / 2, self.anchor.y, h / 2]
        bl = [-w / 2, self.anchor.y, -h / 2]
        br = [w / 2, self.anchor.y, -h / 2]
        # x coords
        left_t2b = linspace(tl[0] + px / 2, bl[0] + px / 2, self.image_size[1])
        right_t2b = linspace(tr[0] - px / 2, br[0] - px / 2, self.image_size[1])
        x_coords = array([linspace(left_t2b[k], right_t2b[k], self.image_size[0]) for k in range(self.image_size[1])])
        # y coords
        y_coords = trans(ones(self.image_size) * self.anchor.y)
        # z coords
        left_t2b = linspace(tl[2] - px / 2, bl[2] + px / 2, self.image_size[1])
        right_t2b = linspace(tr[2] - px / 2, br[2] + px / 2, self.image_size[1])
        z_coords = array(
            [linspace(left_t2b[k], right_t2b[k], self.image_size[0]) for k in range(self.image_size[1])])
        centres = concat([[x_coords], [y_coords], [z_coords]], axis=0)
        return centres

    def rescale(self,factor,*args,**kwargs):
        if factor<=0 or factor.__class__ not in [int,float]:
            raise TypeError("Factor muse be a strictly positive integer or float.")
        self.span=(self.span[0]*factor,self.span[1]*factor)
        self.image_size=tuple(array(self.span*self.resolution,dtype=int))
        self.pixel_field=self.gen_pixel_field(*args,**kwargs)
        self.pixel_size = self.pixel_field[0][1][1] - self.pixel_field[0][0][0]
        self.ray_field=self.gen_ray_field(*args,**kwargs)

    def shift(self,distance,*args,**kwargs):
        if distance.__class__ not in [float, int]:
            raise TypeError("Shift distance must be an integer or float")
        else:
            if self.anchor.y+distance<self.limit:
                print("Attempt to move closer than acceptable limit. Screen moved to ",self.limit,"instead.")
                self.anchor=SVec(array([0,1,0]))*self.limit
                self.pixel_field = self.gen_pixel_field(*args,**kwargs)
                self.ray_field=self.gen_ray_field(*args,**kwargs)
            else:
                self.anchor=self.anchor+SVec(array([0,1,0]))*distance
                self.pixel_field = self.gen_pixel_field(*args,**kwargs)
                self.ray_field=self.gen_ray_field(*args,**kwargs)

    def reset_limit(self,new_limit):
        if new_limit<=0 or new_limit not in [float,int]:
            raise TypeError("View limit muse be a strictly positive integer or float.")
        else:
            self.limit=new_limit

    def newres(self,new_res,*args,**kwargs):
        if new_res.__class__!=int:
            raise TypeError("New esolution must be an integer representing number of pixels per unit distance.")
        else:
            self.resolution = new_res
            self.image_size = tuple(array(self.span, dtype=int) * self.resolution)
            self.pixel_field = self.gen_pixel_field(*args,**kwargs)
            self.pixel_size = self.pixel_field[0][1][1] - self.pixel_field[0][0][0]
            self.ray_field=self.gen_ray_field(*args,**kwargs)

    def pixel_coords(self,location):
        if location.__class__!=tuple:
            raise TypeError("Location variable must be a tuple of two integers i.e (x, y) coordinates.")
        if len(location)!=2:
            raise TypeError("Location variable must be a tuple of two integers i.e (x, y) coordinates.")
        (x,y)=location
        if False in [x.__class__==int,y.__class__==int]:
            raise TypeError("Location variable must be a tuple of two integers i.e (x, y) coordinates.")
        return SVec(array([self.pixel_field[i,y-1,x-1] for i in range(3)])) # recall array index offset

    def pixel_ray(self,location):
        return Ray(self.origin,self.ray_field(location))

    # end ray field should be list of unit direction vectors from self.origin

    def gen_ray_field(self,*args,**kwargs):
        dims=self.pixel_field.shape
        ray_directions=zeros(dims)
        for i in range(dims[1]):
            for j in range(dims[2]):
                ray_directions[:,i,j]=self.pixel_field[:,i,j]-self.origin.value
        ray_directions=ray_directions/norm(ray_directions,axis=0)
        return ray_directions

    def gen_angle_field(self,*args,**kwargs):
        dims=self.pixel_field.shape


    def randomise_rays(self,*args,**kwargs):
        centres = self.pixel_field
        layer_dims = (centres.shape[1],centres.shape[2])
        if "within_pixel" in args:
            y_noise = zeros(layer_dims)
            x_noise = (random(layer_dims) - 0.5) * self.pixel_size
            z_noise = (random(layer_dims) - 0.5) * self.pixel_size
            noise = concat([[x_noise], [y_noise], [z_noise]])
            centres += noise
        elif "radius" in kwargs:
            y_noise = zeros(layer_dims)
            x_noise = (ndist(0,kwargs["radius"]*self.pixel_size,layer_dims))
            z_noise = (ndist(0,kwargs["radius"]*self.pixel_size,layer_dims))
            noise = concat([[x_noise], [y_noise], [z_noise]])
            centres += noise
        else:
            pass
        self.pixel_field=centres
        self.ray_field=self.gen_ray_field(*args,**kwargs)

# CUSTOM ROTATION CLASS

THETA="\u03B8"
PHI="\u03C6"
PSI="\u03C8"

class Rotation:

    def __init__(self,theta,phi,psi):
        if False in [angle.__class__ in [Angle,int,float] for angle in [theta,phi,psi]]:
            raise TypeError("All angles should be instances of Angle.")
        [self.theta, self.phi, self.psi] = [Angle(theta),Angle(phi),Angle(psi)]
        self.rotation= R.from_euler('zyx', [[self.theta.value, self.phi.value, self.psi.value]], degrees=True)

    def __str__(self):
        return "Rotation({}: {}, {}: {}, {}: {})".format(THETA,self.theta,PHI,self.phi,PSI,self.psi)

    def __neg__(self):
        return Rotation(-self.theta,-self.phi,-self.psi)

    def __add__(self, other):
        if other == None:
            return self
        elif other.__class__ is Rotation:
            return Rotation(self.theta+other.theta,self.phi+other.phi,self.psi+other.psi)
        else:
            raise TypeError("Both arguments must be Rotation or None instances.")

    def __radd__(self, other):
        if other == None:
            return self
        elif other.__class__ is Rotation:
            return Rotation(self.theta+other.theta,self.phi+other.phi,self.psi+other.psi)
        else:
            raise TypeError("Both arguments must be Rotation or None instances.")

    def __iadd__(self, other):
        if other == None:
            return self
        elif other.__class__ is Rotation:
            return Rotation(self.theta + other.theta, self.phi + other.phi, self.psi + other.psi)
        else:
            raise TypeError("Both arguments must be Rotation or None instances.")

    def rotate_point(self,point,*args,**kwargs):
        if "centre" in kwargs:
            if kwargs["centre"].__class__ != SVec:
                raise TypeError("Centre keyword value bust be an SVec instance.")
        if point.__class__ is SVec:
            if "reverse" in args:
                if "centre" in kwargs:
                    shifted = point.value - kwargs["centre"].value
                    new_point = self.rotation.apply(shifted, inverse=True)[0]
                    return SVec(new_point+kwargs["centre"])
                else:
                    return SVec(self.rotation.apply(point.value, inverse=True)[0])
            else:
                if "centre" in kwargs:
                    shifted = point.value - kwargs["centre"].value
                    new_point = self.rotation.apply(shifted)[0]
                    return SVec(new_point + kwargs["centre"])
                else:
                    return SVec(self.rotation.apply(point.value)[0])
        elif point.__class__ == list:
            if point[0].__class__ == SVec:
               array_form = array([s.value for s in point])
               rotated = self.rotate_point(array_form,*args,**kwargs)
               return [SVec(s) for s in list(rotated)]
            elif point[0].__class__ == array(0).__class__:
                return list(self.rotate_point(point, *args, **kwargs))
            else:
                raise TypeError("Must be a list of SVecs or arrays.")
        elif point.__class__ == array(0).__class__:
            shape=point.shape
            if shape[-1] == 3:
                if "reverse" in args:
                    if "centre" in kwargs:
                        shifted = point - kwargs["centre"].value
                        new_points = self.rotation.apply(shifted, inverse=True)
                        return new_points + kwargs["centre"].value
                    else:
                        return self.rotation.apply(point, inverse=True)
                else:
                    if "centre" in kwargs:
                        shifted = point - kwargs["centre"].value
                        new_points = self.rotation.apply(shifted)
                        return new_points + kwargs["centre"].value
                    else:
                        return self.rotation.apply(point)
            else:
                raise ValueError("Array is the wrong shape. Last dimension must be length 3.")
        else:
            raise TypeError("Point must be an SVec Object, array or list of either.")

    def rotate_ray(self,ray):
        if ray.__class__ is not Ray:
            raise TypeError("Argument must be an instance of Ray.")
        points = [ray.origin, ray.point_at_param(1)]
        rotated = self.rotate_points(points)
        return Ray(rotated[0],rotated[1])

    # THIS MUST BE INCLUDED ABOVE

    def rotate_field(self,field):
        if field.__class__ == array(0).__class__:
            shape = field.shape
            if len(shape) ==3:
                if shape[0] != 3:
                    raise ValueError("Field shape must have length 3.")
                else:
                    for i in prange(shape[1]):
                        for j in prange(shape[2]):
                            field[:, i, j] = self.rotate_point(field[:, i, j])
                    return field
        elif field.__class__ in [Field,SimpleField]:
            rays=field.ray_field
            shape=rays.shape
            for i in prange(shape[1]):
                for j in prange(shape[2]):
                    rays[:,i,j]=self.rotate_point(rays[:,i,j])
            field.ray_field=rays
            return field
        else:
            raise TypeError("Field must be an class instance of Field or a 3D array.")

    def rotation_about_vec(vec, angle):  # Outputs a rotation object
        pass

    '''def rot_ray(ray, rotation, *args):
        if ray.__class__ is not Ray:
            raise TypeError("First argument must be and instance of Ray class.")
        p_1 = ray.origin
        p_2 = ray.point_at_param(1)
        rot_p_1 = rotate_point(p_1, rotation, *args)
        rot_p_2 = rotate_point(p_2, rotation, *args)
        return Ray(rot_p_1, rot_p_2 - rot_p_1)

    def rot_ray_around_centre(ray, rotation, centre, *args):
        if ray.__class__ is not Ray:
            raise TypeError("First argument must be and instance of Ray class.")
        
        rot_p_1 = rotate_around_centre(p_1, rotation, centre, *args)
        rot_p_2 = rotate_around_centre(p_2, rotation, centre, *args)
        return Ray(rot_p_1, rot_p_2 - rot_p_1)'''

# Generalised version of SimpleField that allows us to look in any direction.

class Field(SimpleField):
    pass

# NOISE CLASS AND INSTANCES

class Noise():

    def __init__(self,noise_funct):
        if not (isfunction(noise_funct) or isbuiltin(noise_funct) or noise_funct.__class__ is partial(print).__class__):
            raise TypeError("Noise function must be a function object")
        self.noise_funct=noise_funct

    def __str__(self):
        desc="Noise("
        return desc

    def apply(self,input,*args,**kwargs):
        if input is None:
            return None
        elif input.__class__ == array(0).__class__:
            return self.noise_funct(input.shape)+input
        elif input.__class__ is SVec:
            return SVec(self.apply(input.value))
        elif input.__class__ is CVec:
            return CVec(clip(self.noise_funct(3)+input.value,0,255))
        elif input.__class__ is GVec:
            return GVec(clip(self.noise_funct(1)+input.value,0,255))
        elif input.__class__ in [int,float]:
            return self.noise_funct(*args, **kwargs) + input
        elif input.__class__== Ray:
            if "origin" in args:
                if len(args) != 1:
                    args = tuple(list(args).remove("origin"))
                else:
                    args = ()
                return Ray(self.apply(input.origin,*args,**kwargs),input.direction)
            elif "direction" in args:
                if len(args) != 1:
                    args = tuple(list(args).remove("direction"))
                else:
                    args = ()
                return Ray(input.origin, self.apply(input.direction,*args,**kwargs))
            else:
                return Ray(self.apply(input.origin,*args,**kwargs), self.apply(input.direction,*args,**kwargs))
        elif input.__class__ is Angle:
            return Angle(self.apply(input.value,*args,**kwargs))
        else:
            raise TypeError("Input type is not valid.")

    def interpolate(self,other,*args,**kwargs):
        if other.__class__!=self.__class__:
            raise TypeError("Both objects must be an instance of Noise.")

        else:
            if "ratio" in kwargs:
                t=kwargs["ratio"]
            else:
                t=0.5

            def funct_interp(f, g, t, *args, **kwargs):
                def new_funct(*args, **kwargs):
                    return t * f(*args, **kwargs) + (1 - t) * g(*args, **kwargs)
                return new_funct

            interp_noise_funct=funct_interp(self.noise_funct, other.noise_funct,t)
            interp_noise_funct.__name__=str(t)+"*"+self.noise_funct.__name__ +" + "+str(1-t)+"*"+other.noise_funct.__name__

            interp_Noise=Noise(interp_noise_funct)

            return interp_Noise

    def additive(self,other,*args,**kwargs):
        if other.__class__ is not Noise:
            raise TypeError("Both objects must be instances of Noise class.")
        else:
            def funct_addit(f, g, *args, **kwargs):
                def new_funct(*args, **kwargs):
                    return f(*args, **kwargs) + g(*args,**kwargs)
                return new_funct

            additive_noise_funct = funct_addit(self.noise_funct, other.noise_funct)
            additive_noise_funct.__name__ = self.noise_funct.__name__ + " + " + other.noise_funct.__name__

            additive_Noise = Noise(additive_noise_funct)

            return additive_Noise

    def multiplicative(self,other,*args,**kwargs):
        if other.__class__ is not Noise:
            raise TypeError("Both objects must be instances of Noise class.")
        else:
            def funct_mult(f, g, *args, **kwargs):
                def new_funct(*args, **kwargs):
                    return f(*args, **kwargs)*g(*args, **kwargs)
                return new_funct

            multiplicative_noise_funct = funct_mult(self.noise_funct, other.noise_funct)
            multiplicative_noise_funct.__name__ = self.noise_funct.__name__ + " * " + other.noise_funct.__name__

            multiplicative_Noise = Noise(multiplicative_noise_funct)

            return multiplicative_Noise

def gaussian_noise(mean,sd,*args,**kwargs):
    return Noise(partial(normal,mean,sd,*args,**kwargs))

def poisson_noise(lam,*args,**kwargs):
    return Noise(partial(poisson,lam,*args, **kwargs))

def rayleigh_noise(sigma,*args,**kwargs):
    return Noise(partial(rayleigh,sigma,*args,**kwargs))

def salt_and_pepper(low,high,*args,**kwargs):
    def sp_noise(low,high):
        rand_roll=random()
        if rand_roll<low:
            return -256
        elif rand_roll > (1-high):
            return 256
        else:
            return 0
    return Noise(partial(sp_noise,low,high,*args,**kwargs))

def gamma_noise(k,theta,*args,**kwargs):
    return Noise(partial(gamma,k,theta,*args,**kwargs))

def white_noise(lower,upper,*args,**kwargs):
    def rand_range(lower,upper):
        return lower+random*(upper-lower)
    return Noise(partial(rand_range,lower,upper,*args,**kwargs))

def brownian_noise(scale, *args, **kwargs):
    return Noise(partial(exponential,scale,*args,**kwargs))

def periodic_noise(freq,mag,*args,**kwargs):
    def periodic(input,freq,mag,**kwargs):
        if "phase" in kwargs:
            return mag*sin(freq*input + kwargs["phase"])
        else:
            return mag*sin(freq*input)
    return Noise(partial(periodic,freq,mag,*args,**kwargs))

def no_noise():
    def identity(input):
        return input
    return Noise(identity)
