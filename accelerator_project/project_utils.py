# IMPORTS (Sonar Formulae)
from numpy import cos, sin, tan, pi, log10, imag, real, arcsin, exp, array, arctan, where, errstate, nan_to_num, NaN, \
    power, divide, multiply, amin, amax
from toolbox import show_image
from timeit import default_timer
from os.path import join
from cv2 import imwrite

# IMPORTS (A_Scan and Scan)
from numpy import zeros, meshgrid, linspace, concatenate, stack, dot, expand_dims, tensordot, squeeze, divide, abs, \
    nanmin, nanpercentile, floor_divide, nanmax, isnan, argwhere, bincount, transpose, matmul, ones, isneginf, inf, \
    clip, uint8, full
from numpy.random import random
from numpy.linalg import norm
import matplotlib.pyplot as plt
from cv2 import imshow, waitKey, destroyAllWindows
from scipy.spatial.transform import Rotation as R

# IMPORTS (Scene)
from new_geometry_tools import *
from numpy import concatenate, full_like, isnan, asarray
from timeit import default_timer

# SPEED OF SOUND

def mackenzie_sos(temperature,salinity,depth):
    return 1448.96 + 4.591*temperature + 0.05304*temperature**2 + 0.0002374*temperature**3 + 1.34*(salinity-35) + 0.0163*depth + 1.675*10**(-7)*depth**2

SOS_SALT = mackenzie_sos(10, 0.035, 10)  # Speed of sound at 10 degrees C, 35000 parts per million (seawater), depth 10m
LAMB_600 = SOS_SALT/(600*1000)  # Wavelength at 600 Hz
LAMB_1200 = SOS_SALT/(1200*1000) # Wavelength at 1200 Hz

# DEVICE

class SonarDevice():

    def __init__(self, m, n, L, W, x_gap, y_gap, **kwargs):
        self.freq = None  # Device switched off
        self.horizontal_gap = x_gap
        self.vertical_gap = y_gap
        self.crystal_width = W
        self.crystal_length = L
        self.horizontal_count = n
        self.vertical_count = m
        if "sos" in kwargs:
            self.sos = kwargs["sos"]
        else:
            self.sos = 1500 #m/s
        self.wavelength = None
        if "mode" in kwargs:
            if kwargs["mode"] not in ["stop", "scan", "rotate"]:
                raise ValueError("Mode needs to be 'stop', 'scan' or 'rotate'.")
            else:
                self.mode = kwargs["mode"]
        else:
            self.mode = None

    def set_freq(self, freq):
        self.freq = freq
        self.wavelength = self.sos/freq/1000

    def beam_widths(self, threshold=0.5, *args):
        direction = array([0,0.001],dtype=float)
        while self.directivity(direction) > threshold:
            direction += array([0, 0.001])
        vert = direction[1]
        direction = array([0.001,0], dtype=float)
        while self.directivity(direction) > threshold:
            direction += array([0.001, 0])
        horiz = direction[0]
        if "degs" in args:
            horiz, vert = horiz * 180 / pi, vert * 180 / pi
            return 2 * round(horiz, 2), 2 * round(vert, 2)
        return 2 * round(horiz, 4), 2 * round(vert, 4)

    def beam_linear_points(self, angle, number, gap):
        with errstate(divide='ignore', invalid='ignore'):
            B = where(angle == 0, 1, divide(sin(number * pi * gap * sin(angle) / self.wavelength), (number * sin(pi * gap * sin(angle)) / self.wavelength)))
        return power(B,2)

    def beam_linear_unif(self, angle, length):
        with errstate(divide='ignore', invalid='ignore'):
            B = where(angle == 0, 1, divide(sin(pi * length * sin(angle) / self.wavelength), (pi * length * sin(angle) / self.wavelength)))
        return power(B,2)

    def beam_linear(self, angle, number, length, gap):
        B_unif = self.beam_linear_unif(angle, length)
        B_array = self.beam_linear_points(angle, number, gap)
        return multiply(B_unif, B_array)

    def log_beam_linear(self, theta, n, length, spacing):
        return multiply(10, log10(self.beam_linear(theta, n, length, spacing)))

    def directivity(self, angles, *args, **kwargs):
        B_h = self.beam_linear(angles[1], self.vertical_count, self.crystal_width, self.vertical_gap)
        B_v = self.beam_linear(angles[0], self.horizontal_count, self.crystal_length, self.horizontal_gap)
        all = multiply(B_h, B_v)
        if "threshold" not in kwargs:
            return all
        else:
            where(all < kwargs["threshold"], 0, all)

    def log_directivity(self, angles, *args, **kwargs):
        B_h = self.log_beam_linear(angles[1], self.vertical_count, self.crystal_width, self.vertical_gap)
        B_v = self.log_beam_linear(angles[0], self.horizontal_count, self.crystal_length, self.horizontal_gap)
        all = B_h + B_v
        if "show" in args:
            dmin, dmax = amin(all), amax(all)
            show = (all-dmin)/(dmax-dmin)
            show_image(show)
            if "save_dir" in kwargs:
                imwrite(join(kwargs["save_dir"],"directivity_test.png"), show*255)
        return all

    def __str__(self):
        return "Device(Freq:{})".format(self.freq)

# Default devices to be imported.

SV1010 = SonarDevice(2,10,0.004,0.00222,0.0042,0.001,sos=SOS_SALT)
SV1010_600Hz = SonarDevice(2,10,0.004,0.00222,0.0042,0.001,sos=SOS_SALT)
SV1010_600Hz.set_freq(600)
SV1010_1200Hz = SonarDevice(2,10,0.004,0.00222,0.0042,0.001,sos=SOS_SALT)
SV1010_1200Hz.set_freq(1200)

# MEDIUM

class Medium():

    def __init__(self, name, freq=800):
        if name == "salt":
            self.name = name
            self.depth = 10
            self.salinity = 35
            self.pH = 8.1
            self.temp = 5
            self.c = mackenzie_sos(self.temp, self.salinity, self.depth)
            self.rho = 1027  # Density of sea water at surface
        elif name == "fresh":  # VALUES FOR THESE
            self.name = name
            self.depth = 10
            self.salinity = 0.9  # check ppt or decimal
            self.pH = 7
            self.temp = 10
            self.c = mackenzie_sos(self.temp, self.salinity, self.depth)
            self.rho = 1000
        elif isinstance(name, dict):
            needed = ["name", "depth", "salinity", "pH", "temp"]
            for n in needed:
                if n not in name:
                    raise ValueError("Intantiation failed. Invalid input.")
            self.name = name["name"]
            self.depth = name["depth"]
            self.salinity = name["salinity"]
            self.pH = name["pH"]
            self.temp = name["temp"]
            self.c = mackenzie_sos(self.temp, self.salinity, self.depth)
            self.rho = 1000 + 27 * (self.salinity - 0.9) / 34.1
        else:
            raise ValueError("Intantiation failed. Invalid input.")

        self.freq = freq
        self.alpha = self.calculate_alpha()

    def __str__(self):
        properties = ""
        for prop in self.__dict__:
            properties += prop + ": " + str(self.__dict__[prop]) + "\n"
        return properties

    # COEFFICIENTS FOR ALPHA CALCULATIONS
    def A_coeffs(self):
        if self.temp > 20:
            print("This is out of bounds for correct calculation of absorption.")
        A_1 = 8.86 * 10 ** (0.78 * self.pH - self.salinity) / self.c
        A_2 = 21.44 * self.salinity / self.c * (1 + 0.025 * self.temp)
        A_3 = 0.0004947 - 0.0000259 * self.temp + 0.0000007 * self.temp ** 2 - 0.000000015 * self.temp ** 3  # For T < 20 degrees
        return A_1, A_2, A_3

    def freq_factors(self):
        theta = self.temp + 273.15
        f_1 = 2.8 * (self.salinity / 35) ** 0.5 * 10 ** (4 - 1245 / theta)
        f_2 = (8 * 10 ** (8 - 1990 / theta)) / (1 + 0.0018 * (self.salinity - 35))
        return f_1, f_2

    def P_coeffs(self):
        P_1 = 1
        P_2 = 1 - 0.000127 * self.depth + 0.0000000062 * self.depth ** 2
        P_3 = 1 - 0.0000383 * self.depth + 0.00000000049 * self.depth ** 2
        return P_1, P_2, P_3

    # FUNCTIONS TO CALCULATE ALPHA
    def calculate_alpha(self):
        if self.salinity > 2:
            # print(self.name, "Salinity:", self.salinity)
            return self.alpha_salt() / 1000
        else:
            # print(self.name, "Salinity:", self.salinity)
            return self.alpha_fresh()

    def alpha_salt(self):
        A_1, A_2, A_3 = self.A_coeffs()
        P_1, P_2, P_3 = self.P_coeffs()
        f_1, f_2 = self.freq_factors()
        boric = (A_1 * P_1 * f_1 * self.freq ** 2) / (f_1 ** 2 + self.freq ** 2)
        magsulph = (A_2 * P_2 * f_2 * self.freq ** 2) / (f_2 ** 2 + self.freq ** 2)
        water = A_3 * P_3 * self.freq ** 2
        return boric + magsulph + water

    def alpha_fresh(self):
        self.bulk = 5.941 * 10 ** (-3) - 2.371 * 10 ** (-4) * self.temp + 4.948 * 10 ** (
            -6) * self.temp ** 2 - 3.975 * 10 ** (-8) * self.temp ** 3
        ratio = 3.11 - 0.0155 * self.temp
        self.shear = self.bulk / ratio  # Table from text. Assumes approx. 1-5 ATM pressure.
        return 8 * pi ** 2 * (self.freq * 1000) ** 2 * (self.shear + 3 * self.bulk / 4) / (3 * self.rho * self.c ** 3)

    def absorption_dB(self, r):
        return multiply(8.68589 * self.alpha, r)

    def spherical_loss_dB(self, r):
        return multiply(20, log10(r))

    # TRANSMISSION LOSS
    def TL(self, r):
        return self.spherical_loss_dB(r) + self.absorption_dB(r)

    # CHANGE FREQ FROM DEFAULT
    def set_freq(self, freq):
        self.freq = freq
        self.alpha = self.calculate_alpha()

    # EDIT MEDIUM QUANTITIES
    def change_property(self, property_name, new_value):
        if property_name == "temp":
            self.temp = new_value
            self.c = mackenzie_sos(self.temp, self.salinity, self.depth)
            self.alpha = self.calculate_alpha()
        elif property_name == "salinity":
            self.salinity = new_value
            self.c = mackenzie_sos(self.temp, self.salinity, self.depth)
            self.alpha = self.calculate_alpha()
        elif property_name == "depth":
            self.depth = new_value
            self.c = mackenzie_sos(self.temp, self.salinity, self.depth)
            self.alpha = self.calculate_alpha()
        elif property_name == "freq":
            self.freq = new_value
            self.alpha = self.calculate_alpha()
        elif property_name == "pH":
            self.pH == new_value
            self.alpha = self.calculate_alpha()
        elif property_name == "density":
            self.rho = new_value
            self.alpha = self.calculate_alpha()
        elif property_name == "shear":
            self.shear == new_value
            self.alpha == self.calculate_alpha()
        elif property_name == "bulk":
            self.bulk == new_value
            self.alpha == self.calculate_alpha()
        else:
            pass

    def reset_values(self, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = self.name
        self.__init__(name)

# Default Medium instances to be imported

SALT_DICT = {"name": "SALT_WATER", "depth": 10, "salinity": 35, "pH": 8.1, "temp": 10}
SALT_WATER_600 = Medium(SALT_DICT, freq=600)
SALT_WATER_1200 = Medium(SALT_DICT, freq=1200)

# SCENE

class Scene():

    def __init__(self, **kwargs):

        if "background" in kwargs:
            self.background = kwargs["background"]
        else:
            self.background = None
        if "objects" in kwargs:
            if isinstance(kwargs["objects"], dict):
                self.labels = list(kwargs["objects"].keys())
                self.objects = list(kwargs["objects"].values())
            elif isinstance(kwargs["objects"], list):
                self.labels = ["object_"+str(i+1) for i in range(len(kwargs["objects"]))]
                self.objects = kwargs["objects"]
            else:
                self.labels = ["object_1"]
                self.objects = [kwargs["objects"]]
        else:
            self.labels = []
            self.objects = []

        if "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]
        else:
            self.accelerator = None

        if "name" in kwargs:
            self.name = kwargs["name"]
        else:
            self.name = "untitled_scene"

        self.object_count = len(self.objects)
        self.pov = None
        self.rays = None

        if self.background is None and self.object_count == 0:
            raise AttributeError("This is an empty scene. All images will be empty")

    def __str__(self):
        return "Scene(" + self.name +  ", " + str(self.object_count) + " objects in scene, background:{})".format(self.background.__str__())

    def __repr__(self):
        desc = ""
        desc += "SCENE:" + self.name + "\n"
        if self.object_count != 0:
            desc += "OBJECTS\n"
            for i in range(self.object_count):
                desc += str(self.labels[i]) + ": " + str(self.objects[i]) + "\n"
        if self.background is not None:
            desc += "BACKGROUND\n"
            desc += self.background.__str__()
        elif self.background is None and self.object_count == 0:
            desc = "SCENE IS EMPTY\n"

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

    def intersection_params(self, rays, pov, **kwargs): # Possibly add accelerator here
        self.rays = rays
        self.pov = pov
        if self.accelerator is not None: # Placeholder for any actions relevant to the Accelerator eg. reordering
            pass
        ob_idxs = full(self.rays.shape[1:], -1)  # Rays all assigned as not hitting any object
        dists = full(self.rays.shape[1:], 128, dtype=float)  # Distances all set to 128 (much higher than range)
        for i in range(self.object_count):
            dist = self.objects[i].intersection_params(rays, pov)  # Calc distances until intersection for current object
            ob_idxs, dists = self.collect_min(ob_idxs, dists, dist, i)  # updates closest distances for each ray and retains object
        idx_dict = {}
        for f in range(-1, self.object_count): # -1 is for no objects being hit for that ray
            conditions = where(ob_idxs == f)  # Finds indices where rays hit current object
            idx_dict[f] = conditions

        self.idx_dict = idx_dict
        self.dists = dists
        self.process_background()  # Modifies above 3 scene properties and replaces misses with background hits

    def collect_min(self, indices, mins, new_array, index):
        indices = where(new_array < mins, index, indices) # Double calculation here. Optimise?
        mins = where(new_array < mins, new_array, mins)
        return indices, mins

    # Changes already generated ray and idx dicts. Note that assumption that background is not in front of any object.
    def process_background(self):
        if self.background is not None:
            if not isinstance(self.background, Composite):
                rays = array([(self.rays[:, self.idx_dict[-1][0][c], self.idx_dict[-1][1][c]]) for c in range(len(self.idx_dict[-1][0]))])
                br_dists = squeeze(self.background.intersection_params(rays, self.pov))   # gets distances of rays that hit the
                hits = squeeze(asarray(where(~isnan(br_dists))))   # sort between those that actually hit the B/G
                self.idx_dict[-1] = tuple(array(self.idx_dict[-1]).T[hits].T) # Produces correct format of tuple for indexing
                self.dists[self.idx_dict[-1]] = br_dists[hits]   # Allocate distance of hits to 2D array of dists
            else:
                # similar to above but indices of components replace object indices
                pass

    # Map SL results onto shape
    # Maybe input is indices
    def scatter_loss(self):
        SL = full(self.rays.shape[1:], NaN)  # generate expected shape
        print("start checking objects")
        print("input_rays", self.rays.shape)
        start = default_timer()
        for o in range(self.object_count):
            if self.idx_dict[o] is not None:
                print("Calculating incidents for", self.labels[o])
                if issubclass(self.objects[o], Composite): # If it's a composite, the incidents must be calculated for subparts this will use cached component indices
                    incidents = self.objects[o].gen_incident(self.rays) # No index feed needed as this will be stored internally
                else:
                    incidents = self.objects[o].gen_incident(self.rays)
                SL[self.idx_dict[o]] = self.objects[o].scatterer.SL(incidents)  # Input should be incident angles. Composites should have single scatterer object
            else:
                print("no")
        print("finished checking objects:", default_timer()-start)
        if self.background is not None:
            print(array(self.idx_dict[-1]).size/2)
            print(self.rays.shape)
            bg_incidents = squeeze(self.background.gen_incident(self.rays[self.idx_dict[-1]]))
            print(bg_incidents)
            SL[self.idx_dict[-1]] = self.background.scatterer.SL(bg_incidents)  # Input should be incident angles
        return SL

    # Methods for altering Scene properties
    def name_scene(self, name):
        self.scene_name =str(name)

    def add_object(self, object, **kwargs):
        if "object_name" in kwargs:
            object.name_object(str(kwargs["object_name"]))
            self.objects += object
            self.labels += str(kwargs["object_name"])
            self.object_count += 1
        elif object.name != None:
            self.labels += str(object.name)
            self.objects += object
            self.object_count += 1
        else:
            object_name="Scene_object_"+str(self.object_count+1)
            object.name_object(object_name)
            self.labels += object_name
            self.objects += object
            self.object_count += 1

    def add_objects(self, objects, **kwargs):
        if isinstance(objects, dict):
            for object in objects:
                self.add_object(objects[object], object_name=object)
        elif isinstance(objects, list):
            if "object_names" in kwargs:
                if len(objects) != len(kwargs["object_names"]):
                    raise ValueError("Number of objects and labels must match.")
                else:
                    self.objects += objects
                    self.labels += kwargs["object_names"]
            else:
                for object in objects:
                    self.add_object(object)

    def del_object(self, object_name):
        obj_idx = self.labels[object_name]
        self.objects.pop(obj_idx)
        self.labels.pop(obj_idx)
        self.object_count -= 1

    def change_object_name(self, old_name, new_name):
        obj_idx = self.labels.index(old_name)
        self.labels[obj_idx] = new_name


# SAMPLE_SCENES

## Scenes



class A_scan:

    def __init__(self, device, centre, direction, declination=0, res=200, threshold=0.3, *args, **kwargs):
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 0:
                start = default_timer()
        if isinstance(device, SonarDevice):
            self.device = device
            if self.device.freq is None:
                raise AttributeError("Device needs to have frequency set. Use set_freq method before assigning device.")
        else:
            raise TypeError("Device must be an instance of SonarDevice")

        if threshold >= 1 or threshold <= 0.1:
            self.strength_threshold = 0.1
        else:
            self.strength_threshold = threshold
        self.horiz_span, self.vert_span = self.device.beam_widths(threshold=self.strength_threshold)
        self.beam_widths = self.device.beam_widths()

        if "degs" in args:
            self.direction_angle = direction * pi / 180
            self.declination = declination * pi / 180
        else:
            self.direction_angle = direction
            self.declination = declination

        self.direction = self.angles_to_vec((self.direction_angle, -self.declination))
        self.min_vert, self.max_vert = - self.vert_span / 2 - self.declination, \
                                       self.vert_span / 2 - self.declination
        self.min_horiz, self.max_horiz = self.direction_angle - self.horiz_span / 2, \
                                         self.direction_angle + self.horiz_span / 2

        self.res = res
        self.centre = array(centre)
        self.epsilon = 0.001

        # Scene setup

        if "test_plane" in kwargs:
            plane_vec = kwargs["test_plane"]
            self.plane = array(plane_vec) + self.epsilon
            self.plane_normal = array([-self.plane[0], -self.plane[1], 1])
            self.plane_normal /= norm(self.plane_normal)
            self.constant = dot(array([0, 0, self.plane[2]]) - self.centre, self.plane_normal)
        else:
            if "scene" not in kwargs:
                self.scene = None
            else:
                self.scene = kwargs["scene"]

        if "medium" in kwargs:
            if not isinstance(kwargs["medium"], Medium):
                raise TypeError("medium needs to be assigned to a Medium instance.")
            else:
                self.medium = kwargs["medium"]
        else:
            self.medium = Medium("fresh")
            self.medium.set_freq(self.device.freq)

        # Pulse response params

        if "test_plane" in kwargs:
            if "scatterer" not in kwargs:
                self.test_scatterer = "lambertian"
                self.mu = 10 ** (-2.2)
            else:
                self.test_scatterer = kwargs["scatterer"]

        if "sos" in kwargs:
            self.sos = kwargs["sos"]
        else:
            self.sos = 1450  # speed of sound in m/s

        if "range" in kwargs:
            self.range = kwargs["range"]
            self.receiver_time = self.range / self.sos

        if "rx_step" in kwargs:
            self.receiver_time = kwargs["rx_step"]
            self.range = self.receiver_time * self.sos
        else:
            self.range = 20
            self.receiver_time = self.range / self.sos

        # Generate ray pulse
        self.theta_divs, self.phi_divs = int(self.horiz_span * 180 / pi * res), int(self.vert_span * 180 / pi * res)
        self.directivity = self.directivity_filter(*args, **kwargs)  # Generate position independent pulse strengths

        theta = linspace(self.min_horiz, self.max_horiz, self.theta_divs)
        phi = linspace(self.max_vert, self.min_vert, self.phi_divs)
        self.angles = stack(meshgrid(theta, phi, indexing="xy"))
        if "noise" in args:
            noise_mag = pi / 360 / self.res
            self.angles += random(self.angles.shape) * noise_mag - noise_mag / 2
        self.rays = self.angles_to_vec(self.angles)
        self.ray_number = self.rays.shape[1] * self.rays.shape[2]

        # Image construction params
        if "test_plane" in kwargs:
            self.ray_plane_prod = self.perpendicularity()

        if "radial_res" in kwargs:
            self.radial_resolution = kwargs["radial_res"]
        else:
            self.radial_resolution = 400

        if "scan_speed" in kwargs:
            if kwargs["scan_speed"] not in ["slow", "normal", "fast", "superfast", "custom", "stop"]:
                raise AttributeError["Scan speed must be one of: slow, normal, fast, superfast"]
            else:
                self.scan_speed = kwargs["scan_speed"]
                if self.scan_speed == "custom":
                    if "angle_resolution" not in kwargs:
                        raise ValueError(
                            "Custom speed must have angle resolution (steps per full resolution) specified.")
                    else:
                        self.angle_resolution = kwargs["angle_resolution"]
        else:
            self.scan_speed = "normal"

        if self.scan_speed == "normal":
            self.angle_resolution = 512
        elif self.scan_speed == "slow":
            self.angle_resolution = 1024
        elif self.scan_speed == "fast":
            self.angle_resolution = 256
            self.radial_resolution = 200
        elif self.scan_speed == "superfast":
            self.angle_resolution = 128
            self.radial_resolution = 200
        else:
            pass

        if self.scan_speed == "stop":
            self.angle_step = 0
            self.angle_resolution = 512
        else:
            self.angle_step = 2 * pi / self.angle_resolution

        self.step_rotation = R.from_euler('zyx', [[self.angle_step, 0, 0]])

        self.min_intersection = 0.2  # taken from specs
        self.max_intersection = self.receiver_time * self.sos

        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 0:
                print("Setup time:", default_timer() - start)

    # Inbuilt functions

    def lambertian(self, grazing, intensity=1):  # Simple model for surface interaction
        LL = multiply(self.mu, power(sin(grazing), 2))
        return LL

    def log_lambertian(self, grazing):
        return log10(self.lambertian(grazing, intensity=1))

    def scatter_model(self, incident):  # More complicated modelling for surfaces
        pass

    def angles_to_vec(self, angles):  # Returns normalised vectors from an array of angles
        return array([multiply(sin(angles[0]), cos(angles[1])),
                      multiply(cos(angles[0]), cos(angles[1])),
                      sin(angles[1])])

    ### GEOMETRY FUNCTIONS

    def perpendicularity(self):  # ray_direction * plane_normal (if 0, then the ray is in the plane)
        normal = expand_dims(self.plane_normal, 0)
        prod = squeeze(tensordot(normal, self.rays, axes=1), axis=0)
        return prod

    def angles_with_plane(self):
        prod = abs(self.ray_plane_prod)  # angle calculation only calculates absolute value of angle
        return arcsin(prod)

    def intersection_params(self):
        if "plane" in self.__dict__:
            ts = divide(self.constant, self.ray_plane_prod)
            ts = where((ts < self.min_intersection) | (ts > self.max_intersection), NaN, ts)
        else:
            self.scene.intersection_params(self.rays, self.centre)
            ts = self.scene.dists
            ts = where((ts < self.min_intersection) | (ts > self.max_intersection), NaN, ts)
        return ts

    ### RAY RETURN FUNCTIONS

    # Plane: ax + by +z +c = 0
    # Plane vec: [a, b, c] except when flat when it is [0, 0, c]

    def directivity_filter(self, *args, **kwargs):
        theta_range, phi_range = self.beam_widths
        thetas = linspace(-theta_range / 2, theta_range / 2, self.theta_divs)
        phis = linspace(-phi_range / 2, phi_range / 2, self.phi_divs)
        rel_angles = stack(meshgrid(thetas, phis, indexing="xy"))
        DL = self.device.log_directivity(rel_angles, threshold=self.strength_threshold, *args, **kwargs)
        return DL

    def apply_attenuation(self, *args, **kwargs):
        dists = self.intersection_params()
        TL = -self.medium.TL(dists)
        if "show" in args:
            visualise_2d_array(TL, *args, **kwargs)
        return TL

    def total_strength_field(self, *args, **kwargs):
        dists = self.intersection_params()
        TL = -self.medium.TL(dists)
        DL = self.directivity
        if "plane" in self.__dict__:
            if self.test_scatterer == "lambertian":
                SL = self.log_lambertian(self.angles_with_plane())
            else:
                SL = self.test_scatterer.SL(
                    self.angles_with_plane())  # What is this line for? # Is input rays or angles?
        else:
            SL = self.scene.scatter_loss()
        return dists, TL + DL + SL

    def gather(self, dist_array, strength_array, *args, **kwargs):
        if "scaled" in args:
            max_dist, min_dist = nanmax(dist_array), nanmin(dist_array)
            if max_dist < self.min_intersection:
                raise ValueError("Intersections faulty or too close.")
            intervals = (max_dist - min_dist) / self.radial_resolution
        else:
            intervals = self.max_intersection / self.radial_resolution
            min_dist = 0
        dist_array = dist_array.flatten()
        strength_array = strength_array.flatten()
        valid = argwhere(~isnan(dist_array))
        dist_array = dist_array[valid]
        strength_array = squeeze(strength_array[valid])
        if min_dist == 0:
            bin_idxs = squeeze(array(floor_divide(dist_array, intervals), dtype=int))
        else:
            bin_idxs = squeeze(array(floor_divide(dist_array - min_dist, intervals), dtype=int))
        bins = bincount(bin_idxs, weights=strength_array, minlength=self.radial_resolution)
        return bins

    def scan_line(self, *args, **kwargs):
        start = default_timer()
        dist_array, strength_array = self.total_strength_field(*args, **kwargs)
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("strength_field time", default_timer() - start)
        start = default_timer()
        strength_array = power(10, strength_array / 10)  # We need to sum using a non-log scale
        gathered = self.gather(dist_array, strength_array, *args, **kwargs)
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("gathering time", default_timer() - start)
        if "no_gain" not in args:
            gathered = self.auto_gain(gathered, **kwargs)
        if "show_echoes" in args:
            plt.hist(bin_idxs, bins=self.radial_resolution)
            plt.title(
                "Number of ray echoes by return time ({}/{} initial rays)".format(len(bin_idxs), self.ray_number))
            plt.ylabel("Number of returning rays")
            plt.xlabel("Time in ms")
            plt.show()
        if "visualise" in args:
            if "no_gain" in args:
                print("No autogain applied. Saturation may occur. Rerun without 'no_gain' argument")
            canvas = ones((self.radial_resolution, self.radial_resolution))
            canvas = canvas * gathered
            show_image(array(canvas, dtype=float) / 255)
            if "save_dir" in kwargs:
                if "image_name" in kwargs:
                    im_name = kwargs["image_name"]
                else:
                    im_name = "untitled.png"
                imwrite(join(kwargs["save_dir"], im_name), canvas)
        return gathered

    def auto_gain(self, arr, **kwargs):
        start = default_timer()
        max_int = amax(arr)
        if not isnan(max_int):
            arr = divide(arr, max_int)  # Normalise using max response
        else:
            arr = full(arr.shape, -inf)
        arr = where(arr == 0, -inf, arr)
        arr = 10 * log10(arr, out=arr, where=~isneginf(arr))
        dec_min = -nanmin(arr[~isneginf(arr)])
        arr += dec_min
        arr = where(isneginf(arr), 0, arr)
        if "gain" in kwargs:
            arr *= kwargs["gain"]
        if "min_detect" in kwargs:
            min_clip = max(dec_min - kwargs["min_detect"], 0)
        else:
            min_clip = 0
        arr = clip(arr, None, dec_min)
        arr = where(arr < min_clip, 0, arr)
        arr = array(arr / dec_min * 127, dtype=uint8) * 2
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("Gain calc time:", default_timer() - start)
        return arr

    def advance_timestep(self, *args, **kwargs):
        start = default_timer()
        self.angles += [[[self.angle_step]], [[0]]]
        self.direction_angle += self.angle_step
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("Angles time", default_timer() - start)
        start = default_timer()
        self.direction = self.angles_to_vec((self.direction_angle, -self.declination))
        self.rotate_ray()
        if "test_plane" in kwargs:
            self.ray_plane_prod = self.perpendicularity()  # This is only for test plane situation
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("rotate_time", default_timer() - start)

    def rotate_ray(self):
        self.rays = self.angles_to_vec(self.angles)

# SCAN

class Scan():

    def __init__(self, a_scan, mode, *args, **kwargs):
        if not isinstance(a_scan, A_scan):
            raise TypeError("Scan must be instantiated with an A_scan object.")
        self.device = a_scan.device
        self.a_scan = a_scan
        self.scan_speed = a_scan.scan_speed
        self.angle_resolution = a_scan.angle_resolution
        self.angle_step = a_scan.angle_step
        self.set_mode(mode, *args, **kwargs)
        self.scan_idx = 0
        self.radial_resolution = a_scan.radial_resolution
        self.image = zeros((self.radial_resolution, self.steps), dtype=float)
        self.autogain = self.a_scan.auto_gain
        self.total_rays = self.a_scan.ray_number

    def set_mode(self, mode, *args, **kwargs):
        if mode not in ["scan", "rotate", "stop"]:
            raise ValueError("Mode can only be scan, rotate or stop.")
        else:
            self.device.mode = mode
            self.mode = mode
            if mode == "scan":
                if "span" not in kwargs:
                    raise KeyError("Angle span must be given using 'span' keyword argument")
                if "degs" in args:
                    span = kwargs["span"] * pi / 180
                else:
                    span = kwargs["span"]
                self.steps = int(span // self.angle_step) + 1
            elif mode == "stop":
                if "duration" not in kwargs:
                    raise KeyError("Duration must be given using start keyword duration")
                else:
                    self.steps = kwargs["duration"]
            else:
                self.steps = self.angle_resolution

            self.start = self.a_scan.direction_angle * 180 / pi
            self.current = self.start

    def get_line(self, *args, **kwargs):
        sonar_return = self.a_scan.scan_line(*args, **kwargs)
        return sonar_return

    def full_scan(self, *args, **kwargs):
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 0:
                start = default_timer()
                print("STARTING SCAN")
                print("{} rays per pulse".format(self.total_rays))
                print("Calculating {} timesteps.\n".format(self.steps))
        for i in range(self.steps):
            self.current = self.a_scan.direction_angle * 180 / pi
            if "verbosity" in kwargs:
                if kwargs["verbosity"] > 1:
                    print("Starting timestep {}".format(self.scan_idx))
                    print("Sweep angle: {}".format(self.current))
                    step_start = default_timer()
            current_step = self.get_line(*args, "no_gain", **kwargs)
            self.image[:, i] = current_step
            if "show" in args:
                imshow('image', self.image / 255)
                waitKey(1)
            self.a_scan.advance_timestep(*args, **kwargs)
            self.scan_idx += 1
            if "verbosity" in kwargs:
                if kwargs["verbosity"] > 1:
                    print("TIMESTEP DURATION: {}\n".format(default_timer() - step_start))
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 0:
                print("TOTAL TIME: {}".format(default_timer() - start))
        self.image = self.autogain(self.image)
        if "show" in args:
            imshow('image', self.image / 255)
            waitKey(0)
            destroyAllWindows()
        if "save_dir" in kwargs:
            if "image_name" in kwargs:
                img_name = kwargs["image_name"]
            else:
                img_name = "scan_image.png"
            imwrite(join(kwargs["save_dir"], img_name), self.image)


# Visualises with centre 1m above xy plane. Set plane constant 0 to maintain this distance.
# Resolution 100 as default. Threshold at 0.1 as default.
# Degrees as standard.

# SAMPLING FUNCTIONS



def scale_sample(funct, x_range, y_range, levels, low=1):
    if not isinstance(levels, int) or low > levels:
        raise ValueError("Subdivision levels must be higher than {}.".format(low))
    sampled_arrays = []
    for i in range(low, levels+low):
        sampled_arrays.append(array_from_explicit(funct, x_range, y_range, 2**i))
    return sampled_arrays

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

def nested_sample_to_OBJ(funct, x_range, y_range, levels, low=1, **kwargs):
    nt = nested_tesselations(funct, x_range, y_range, levels, low, **kwargs)
    obs = {}
    if "save_dir" in kwargs:
        root = kwargs["save_dir"]
    else:
        root = getcwd()
    for t in nt:
        obs[t] = tess_to_OBJ(nt[t], join(root, t+".obj"))