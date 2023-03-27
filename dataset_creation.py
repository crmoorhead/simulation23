from new_geometry_tools import *
from sonar_formulae import *
from new_scenes import *
from random import uniform, choice, normalvariate

# ALSO ASSIGN PROBS FOR DISCRETE VALUES

class ObjectParameter():

    def __init__(self, type, values, name="", **kwargs):
        self.values = values

        if type.upper() not in ["D", "C", "F"]:
            raise ValueError("Type must be (D)iscrete or (C)ontinuous.")
        elif type.upper() == "D":
            if not isinstance(values, list):
                raise TypeError("Discrete parameters must have list values.")
            self.type = "Discrete"
            def sampler():
                return choice(self.values)
        elif type.upper() == "C":
            if not isinstance(values, tuple) or not len(values) != 2:
                raise TypeError("Continuous parameters must be defined as an interval value")
            self.type = "Continuous"
            if "distribution" not in kwargs:
                def sampler():
                    return uniform(self.values[0], self.values[1])
            else:
                sampler = kwargs["distribution"]
        else:
            if not (isinstance(values, str) or isinstance(values, int) or isinstance(values, float)):
                raise TypeError("Fixed parameters must be a number or string.")
            self.type = "Fixed"
            def sampler():
                return self.values

        if name != "" and name != None:
            self.name = name
        self.sampler = sampler

    def __str__(self):
        if self.type.lower() in ["continuous", "fixed"]:
            return "ObjProp({}: {})".format(self.type, self.values)
        else:
            string = "ObjProp({})\n\n".format(self.type)
            string += str(self.values)[:min(len(str(self.values)), 50)]
            if len(str(self.values)) > 50:
                string += "...]"
            return string

    def sample(self):
        if self.type.lower() == "fixed":
            return self.values
        elif self.type.lower() == "discrete":
            if distribution == "uniform":
                return choice(self.values)
            else:
                return distribution()
        else:
            if distribution == "uniform":
                return normal(self.values[0], self.values[1])
            else:
                return distribution()

# Other distributions not requiring arguments to use above
# Rule of thumb - total range is 3 SD and truncate

def normal_dist(mean, sd):
    def normal():
        return normalvariate(mean, sd)
    return normal

class ObjectGenerator():

    def __init__(self, base, param, **kwargs):
        if not isinstance(params, dict):
            pass
        if "name" not in kwargs:
            self.name = "Unspecified Generator Object"

    def sample_params(self, **kwargs):
        sampled_vals = []
        for p in params:
            if "distribution" not in kwargs:
                sampled_vals.append(p.sample())
            else:
                sampled_vals.append(p.sample(distribution=kwargs["distribution"]))
        return params

    def create_instance(self, **kwargs):
        params = self.sample_params(**kwargs)

class SelectionSet():
    def __init__(self):
        pass

class BackgroundSet(SelectionSet):

    def __init__(self):
        pass

class SceneSampler():

    def __init__(self, background_dict, objects_dict):
        pass

    def generate_scenes(self, n):

        for i in range(n):
            surface = generate_background()

