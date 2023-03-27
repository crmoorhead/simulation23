from inspect import signature, isfunction
from functools import partial
from copy import deepcopy,copy
from fundamentals import CVec, SVec, GVec
from numpy import array

# ABSTRACT SCATTERER TYPE WHICH WILL SERVE AS A CONTAINER FOR A FUNCTION

class Scatterer:

    def __init__(self,scatter_funct):
        self.incident=None
        self.return_angle=None
        if not isfunction(scatter_funct):
            raise TypeError("All instances of Scatterer objects must define a function with two arguments "
                            "- incident and return_angle. No functional input.")
        sig=signature(scatter_funct)
        self.params = sig.parameters      # Lists all parameters in the scatter_funct variable.
                                          # This must, at the very least, include the variables "incident" and "return_angle"

        if "incident" not in self.params or "return_angle" not in self.params:
            raise TypeError("All instances of Scatterer objects must define a function with two arguments "
                            "- incident and return_angle. One or both of these arguments are missing.")
        else:
            self.scatter_funct=scatter_funct
            self.instantiated=None

    def __str__(self):
        if self.incident is None:
            return "Scatterer(Scatter Function: {})".format(self.scatter_funct.__name__)
        elif self.incident.__class__ is list:
            return "Scatterer(Scatter Function: {}, Incident Angles: {})".format(self.scatter_funct.__name__, self.incident)
        else:
            return "Scatterer(Scatter Function: {}, Incident Angle: {})".format(self.scatter_funct.__name__, self.incident)

    def __repr__(self):
        return self.__str__()

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

    def add_source(self,incident,*args,**kwargs):
        if self.incident is None:    # If originally no sources
            self.incident=incident
            new_funct=partial(self.scatter_funct,self.incident,*args,**kwargs)
            new_funct.__name__=copy(self.scatter_funct.__name__)+"(instantiated)"
            self.instantiated=new_funct
        else:                     # If we are adding a secondary source
            if self.incident.__class__!=list:
                self.incident=[self.incident]
            if incident not in self.incident:
                self.incident.append(incident)
                def superimpose(incidents_funct,scatter_funct, new_source,*args,**kwargs):   # This should leave the scatter_funct untouched but change the instantiated function
                    new_source = partial(scatter_funct, new_source, *args, **kwargs)
                    def new_funct(return_angle,*args,**kwargs):
                        return incidents_funct(return_angle,*args,**kwargs)+new_source(return_angle,*args,**kwargs)
                    return new_funct
                self.instantiated=superimpose(self.instantiated,self.scatter_funct,incident)

    def remove_sources(self):
        if self.incident is None:
            pass
        else:
            self.incident=None
            self.instantiated=None

    def interpolate(self,other,*args,**kwargs):
        if other.__class__!=self.__class__:
            raise TypeError("Both objects must be of Scatterer class.")
        if None in [self.incident, other.incident]:  # If incident ray defined, it must be consistent for both scatterers
            if self.incident==other.incident:
                pass
            elif self.incident is None:
                self.add_source(other.incident)
            else:
                other.add_source(self.incident)
        else:
            if self.incident != other.incident:
                raise AttributeError("Incident ray defined differently for two Scatterers.")

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

        interp_scatterer=Scatterer(interp_scatter_funct)

        if self.incident is not None:
            interp_scatterer.add_source(self.incident)

        return interp_scatterer

# SCATTERER FUNCTIONS AND GENERATORS

def hit_only(incident,return_angle,*args,**kwargs):
    return CVec(array([255,255,255]))

def hit_only_gs(incident,return_angle,*args,**kwargs):
    return GVec(255)

def dist_gen(range,*args,**kwargs):
    if range.__class__ not in [int, float]:
        raise TypeError("Range must a strictly positive float or integer. Type is incorrect.")
    if range<=0:
        raise ValueError("Range must a strictly positive float or integer. Number not positive.")
    def dist(incident, return_angle,*args, **kwargs):
        t = kwargs["param"]
        if t>range:
            return CVec(array([0,0,0]))
        else:
            return CVec(array([255,255,255]))*((range-t)/range)
    dist.__dict__["range"]=range
    return Scatterer(dist)

def dist_gen_gs(range, *args, **kwargs):
    if range.__class__ not in [int, float]:
        raise TypeError("Range must a strictly positive float or integer. Type is incorrect.")
    if range<=0:
        raise ValueError("Range must a strictly positive float or integer. Number not positive.")
    def dist(incident, return_angle, *args, **kwargs):
        t=kwargs["param"]
        if t>range:
            return GVec(0)
        else:
            return GVec(255)*((range-t)/range)
    dist.__dict__["range"]=range
    return Scatterer(dist)

def hit_only_gen(color):
    if color.__class__ is not CVec:
        raise TypeError("The output must be a CVec object.")
    else:
        def hit_only_col(incident, return_angle,*args,**kwargs):
            return color
        hit_only_col.__dict__["color"] = color
        return Scatterer(hit_only_col)

def dist_col_gen(color, range, *args, **kwargs):
    if color.__class__ is not CVec:
        raise TypeError("The output must be a CVec object.")
    else:
        def dist_col(incident, return_angle, *args,**kwargs):
            t = kwargs["param"]
            if t> range:
                return CVec(array([0,0,0]))
            else:
                return color*((range-t)/range)
        dist_col.__dict__["range"] = range
        dist_col.__dict__["color"] = color
        return Scatterer(dist_col)

def visualise_normal(incident, return_angle,*args,**kwargs):
    normal=kwargs["normal"]
    normal_abs=SVec(array([abs(normal.x),abs(normal.y),abs(normal.z)]))*255
    col=CVec(normal_abs.value)
    return col

def visualise_lr_normal(incident, return_angle,*args,**kwargs):
    normal=kwargs["normal"]
    t=(normal.x+1)/2
    return t*CVec(array([255,0,0]))+(1-t)*CVec(array([0,0,255]))

def visualise_ud_normal(incident, return_angle,*args,**kwargs):
    normal=kwargs["normal"]
    t=(normal.z+1)/2
    return t*CVec(array([255,0,0]))+(1-t)*CVec(array([0,0,255]))

def visualise_f_normal(incident, return_angle,*args,**kwargs):
    normal=kwargs["normal"]
    t=(normal.y+1)/2
    return t*CVec(array([255,0,0]))

def visualise_normal_dist(dist,*args,**kwargs):
    def dist_col(incident, return_angle, *args, **kwargs):
        normal=kwargs["normal"]
        t = kwargs["param"]
        if t > dist:
            return CVec(array([0, 0, 0]))
        else:
            scale = (dist - t) / dist
            normal_abs = SVec(array([abs(normal.x), abs(normal.y), abs(normal.z)])) * 256 * scale
            col = CVec(normal_abs.value)
            return col
    dist_col.__dict__["range"] = dist
    dist_col.__name__="visualise_normal_dist"
    return Scatterer(dist_col)

def visualise_f_normal_gs(incident, return_angle,*args,**kwargs):
    normal=kwargs["normal"]
    t=(normal.y+1)/2
    return t*GVec(255)

def visualise_incident(*args,**kwargs):
    pass

def diffuse(incident,return_angle,normal,**kwargs):
    pass

