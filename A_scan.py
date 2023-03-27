from numpy import zeros, meshgrid, cos, sin, linspace, concatenate, stack, array, \
    multiply, pi, where, dot, power, arcsin, expand_dims, tensordot, squeeze, divide, \
    abs, nan, log10, amin, amax, nanmin, nanpercentile, floor_divide, nanmax, isnan, \
    argwhere, bincount, transpose, matmul, ones, isneginf, inf, clip, uint8, full
from numpy.random import random
from numpy.linalg import norm
import matplotlib.pyplot as plt
from cv2 import imshow, waitKey, destroyAllWindows, imwrite
from sonar_formulae import *
from toolbox import plot_function, plot_3d_function, visualise_2d_array, set_computer, show_image
from os import getcwd
from timeit import default_timer
from scipy.spatial.transform import Rotation as R

# Set computer

working_dir = set_computer(subdir=r"Uni_2022\Simulation\notebook\simulation")
A_scan_dir = set_computer(subdir=r"Uni_2022\Simulation\Charts\A_scans")
full_scan_dir = set_computer(subdir=r"Uni_2022\Simulation\Charts\full_scans")

class A_scan:

    def __init__(self, device, centre, direction,  declination=0, res=200, threshold=0.3,*args,**kwargs):
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 0:
                start = default_timer()
        if isinstance(device, SonarDevice):
            self.device = device
            if self.device.freq is None:
                raise AttributeError("Device needs to have frequency set. Use set_freq method before assigning device.")
        else:
            raise TypeError("Device must be an instance of SonarDevice")

        if threshold >= 1 or threshold<=0.1:
            self.strength_threshold = 0.1
        else:
            self.strength_threshold = threshold
        self.horiz_span, self.vert_span = self.device.beam_widths(threshold=self.strength_threshold)
        self.beam_widths = self.device.beam_widths()

        if "degs" in args:
            self.direction_angle = direction*pi/180
            self.declination = declination*pi/180
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
            self.plane = array(plane_vec)+self.epsilon
            self.plane_normal = array([-self.plane[0], -self.plane[1], 1])
            self.plane_normal /= norm(self.plane_normal)
            self.constant = dot(array([0,0,self.plane[2]])-self.centre, self.plane_normal)
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
                self.mu = 10**(-2.2)
            else:
                self.test_scatterer = kwargs["scatterer"]

        if "sos" in kwargs:
            self.sos = kwargs["sos"]
        else:
            self.sos = 1450 # speed of sound in m/s

        if "range" in kwargs:
            self.range = kwargs["range"]
            self.receiver_time = self.range/self.sos

        if "rx_step" in kwargs:
            self.receiver_time = kwargs["rx_step"]
            self.range = self.receiver_time*self.sos
        else:
            self.range = 20
            self.receiver_time = self.range/self.sos

        # Generate ray pulse
        self.theta_divs, self.phi_divs = int(self.horiz_span*180/pi*res), int(self.vert_span*180/pi*res)
        self.directivity = self.directivity_filter(*args, **kwargs) # Generate position independent pulse strengths

        theta = linspace(self.min_horiz, self.max_horiz, self.theta_divs)
        phi = linspace(self.max_vert, self.min_vert, self.phi_divs)
        self.angles = stack(meshgrid(theta, phi, indexing="xy"))
        if "noise" in args:
            noise_mag = pi/360/self.res
            self.angles += random(self.angles.shape)*noise_mag - noise_mag/2
        self.rays = self.angles_to_vec(self.angles)
        self.ray_number = self.rays.shape[1]*self.rays.shape[2]

        # Image construction params
        if "test_plane" in kwargs:
            self.ray_plane_prod = self.perpendicularity()

        if "radial_res" in kwargs:
            self.radial_resolution = kwargs["radial_res"]
        else:
            self.radial_resolution = 400

        if "scan_speed" in kwargs:
            if kwargs["scan_speed"] not in ["slow", "normal", "fast", "superfast","custom","stop"]:
                raise AttributeError["Scan speed must be one of: slow, normal, fast, superfast"]
            else:
                self.scan_speed = kwargs["scan_speed"]
                if self.scan_speed == "custom":
                    if "angle_resolution" not in kwargs:
                        raise ValueError("Custom speed must have angle resolution (steps per full resolution) specified.")
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
            self.angle_step = 2*pi/self.angle_resolution

        self.step_rotation = R.from_euler('zyx', [[self.angle_step, 0, 0]])

        self.min_intersection = 0.2 # taken from specs
        self.max_intersection = self.receiver_time*self.sos
        
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 0:
                print("Setup time:", default_timer()-start)

    # Inbuilt functions

    def lambertian(self, grazing, intensity=1):   # Simple model for surface interaction
        LL = multiply(self.mu, power(sin(grazing),2))
        return LL

    def log_lambertian(self, grazing):
        return log10(self.lambertian(grazing,intensity=1))

    def scatter_model(self, incident):  # More complicated modelling for surfaces
        pass

    def angles_to_vec(self, angles):  # Returns normalised vectors from an array of angles
        return array([multiply(sin(angles[0]), cos(angles[1])),
                      multiply(cos(angles[0]), cos(angles[1])),
                      sin(angles[1])])

    ### GEOMETRY FUNCTIONS

    def perpendicularity(self): # ray_direction * plane_normal (if 0, then the ray is in the plane)
        normal = expand_dims(self.plane_normal, 0)
        prod = squeeze(tensordot(normal, self.rays, axes=1), axis=0)
        return prod

    def angles_with_plane(self):
        prod = abs(self.ray_plane_prod) # angle calculation only calculates absolute value of angle
        return arcsin(prod)

    def intersection_params(self):
        if "plane" in self.__dict__:
            ts = divide(self.constant, self.ray_plane_prod)
            ts = where((ts < self.min_intersection) | (ts > self.max_intersection), nan, ts)
        else:
            self.scene.intersection_params(self.rays, self.centre)
            ts = self.scene.dists
            ts = where((ts < self.min_intersection) | (ts > self.max_intersection), nan, ts)
        return ts

    ### RAY RETURN FUNCTIONS

    # Plane: ax + by +z +c = 0
    # Plane vec: [a, b, c] except when flat when it is [0, 0, c]

    def directivity_filter(self,*args,**kwargs):
        theta_range, phi_range = self.beam_widths
        thetas = linspace(-theta_range/2, theta_range/2, self.theta_divs)
        phis = linspace(-phi_range/2, phi_range/2, self.phi_divs)
        rel_angles = stack(meshgrid(thetas, phis, indexing="xy"))
        DL = self.device.log_directivity(rel_angles, threshold=self.strength_threshold, *args, **kwargs)
        return DL

    def apply_attenuation(self, *args,**kwargs):
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
                SL = self.test_scatterer.SL(self.angles_with_plane()) # What is this line for? # Is input rays or angles?
        else:
            SL = self.scene.scatter_loss(self.rays)
        return dists, TL + DL + SL

    def gather(self, dist_array, strength_array, *args, **kwargs):
        if "scaled" in args:
            max_dist, min_dist = nanmax(dist_array), nanmin(dist_array)
            if max_dist < self.min_intersection:
                raise ValueError("Intersections faulty or too close.")
            intervals = (max_dist-min_dist)/self.radial_resolution
        else:
            intervals = self.max_intersection/self.radial_resolution
            min_dist = 0
        dist_array = dist_array.flatten()
        strength_array = strength_array.flatten()
        valid = argwhere(~isnan(dist_array))
        dist_array = dist_array[valid]
        strength_array = squeeze(strength_array[valid])
        if min_dist == 0:
            bin_idxs = squeeze(array(floor_divide(dist_array, intervals), dtype=int))
        else:
            bin_idxs = squeeze(array(floor_divide(dist_array-min_dist, intervals), dtype=int))
        bins = bincount(bin_idxs,weights=strength_array,minlength=self.radial_resolution)
        return bins

    def scan_line(self,*args,**kwargs):
        start = default_timer()
        dist_array, strength_array = self.total_strength_field(*args, **kwargs)
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("strength_field time", default_timer()-start)
        start = default_timer()
        strength_array = power(10, strength_array/10) # We need to sum using a non-log scale
        gathered = self.gather(dist_array, strength_array,*args, **kwargs)
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("gathering time", default_timer()-start)
        if "no_gain" not in args:
            gathered = self.auto_gain(gathered,**kwargs)
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
            canvas = ones((self.radial_resolution,self.radial_resolution))
            canvas = canvas*gathered
            show_image(array(canvas,dtype=float)/255)
            if "save_dir" in kwargs:
                if "image_name" in kwargs:
                    im_name = kwargs["image_name"]
                else:
                    im_name = "untitled.png"
                imwrite(join(kwargs["save_dir"],im_name),canvas)
        return gathered

    def auto_gain(self, arr,**kwargs):
        start = default_timer()
        max_int = amax(arr)
        if not isnan(max_int):
            arr = divide(arr, max_int) # Normalise using max response
        else:
            arr = full(arr.shape, -inf)
        arr = where(arr == 0, -inf, arr)
        arr = 10*log10(arr, out=arr, where=~isneginf(arr))
        dec_min = -nanmin(arr[~isneginf(arr)])
        arr += dec_min
        arr = where(isneginf(arr), 0, arr)
        if "gain" in kwargs:
            arr *= kwargs["gain"]
        if "min_detect" in kwargs:
            min_clip = max(dec_min - kwargs["min_detect"],0)
        else:
            min_clip = 0
        arr = clip(arr, None, dec_min)
        arr = where(arr<min_clip, 0, arr)
        arr = array(arr/dec_min*127,dtype=uint8)*2
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("Gain calc time:", default_timer() - start)
        return arr

    def advance_timestep(self, **kwargs):
        start = default_timer()
        self.angles += [[[self.angle_step]], [[0]]] # Check this
        self.direction_angle += self.angle_step
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("Angles time", default_timer()-start)
        start = default_timer()
        self.direction = self.angles_to_vec((self.direction_angle, -self.declination))
        self.rotate_ray()
        if "test_plane" in kwargs:
            self.ray_plane_prod = self.perpendicularity()      # This is only for test plane situation
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 2:
                print("rotate_time", default_timer()-start)

    def rotate_ray(self):
        self.rays = self.angles_to_vec(self.angles) ## This seems wrong!

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
        self.image = zeros((self.radial_resolution,self.steps), dtype=float)
        self.autogain = self.a_scan.auto_gain
        self.total_rays = self.a_scan.ray_number

    # NEEDS REWRITTEN
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
                    span = kwargs["span"]*pi/180
                else:
                    span = kwargs["span"]
                self.steps = int(span//self.angle_step)+1
            elif mode == "stop":
                if "duration" not in kwargs:
                    raise KeyError("Duration must be given using start keyword duration")
                else:
                    self.steps = kwargs["duration"]
            else:
                self.steps = self.angle_resolution

            self.start = self.a_scan.direction_angle*180/pi
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
            self.current = self.a_scan.direction_angle*180/pi
            if "verbosity" in kwargs:
                if kwargs["verbosity"] > 1:
                    print("Starting timestep {}".format(self.scan_idx))
                    print("Sweep angle: {}".format(self.current))
                    step_start = default_timer()
            current_step = self.get_line(*args,"no_gain", **kwargs)
            self.image[:,i] = current_step
            if "show" in args:
                imshow('image', self.image/255)
                waitKey(1)
            self.a_scan.advance_timestep(*args, **kwargs)
            self.scan_idx += 1
            if "verbosity" in kwargs:
                if kwargs["verbosity"] > 1:
                    print("TIMESTEP DURATION: {}\n".format(default_timer()-step_start))
        if "verbosity" in kwargs:
            if kwargs["verbosity"] > 0:
                print("TOTAL TIME: {}".format(default_timer()-start))
        self.image = self.autogain(self.image)
        if "show" in args:
            imshow('image', self.image/255)
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

def visualise_scan(device, plane, direction, declination, *args, **kwargs):
    if "res" in kwargs:
        res = kwargs["res"]
    else:
        res = 30
    if "threshold" in kwargs:
        threshold = kwargs["threshold"]
    else:
        threshold = 0.1
    if "save_dir" in kwargs:
        save_dir = kwargs["save_dir"]
    else:
        save_dir = getcwd()
    a_scan = A_scan(device, [0,0,1], direction, declination, res, threshold, "degs", test_plane=plane)
    dists = a_scan.intersection_params()
    DL = a_scan.directivity
    TL = -a_scan.medium.TL(dists)
    SL = a_scan.log_lambertian(a_scan.angles_with_plane())
    combined = DL + TL + SL
    true_plane = str([int(i - a_scan.epsilon) for i in a_scan.plane])
    image_name = "dir={}_plane={}".format(a_scan.direction, true_plane)
    if "normalise" in args:
        image_name += "_normed.png"
        common_min, common_max = nanmin(combined), nanmax(combined)
        visualise_2d_array(DL,"no show","color",image_name="directivity_"+image_name,save_dir=save_dir, norm_range=(common_min, common_max))
        visualise_2d_array(TL, "no show", "color", image_name="transmission_" + image_name, save_dir=save_dir, norm_range=(common_min, common_max))
        visualise_2d_array(SL, "no show", "color", image_name="scattering_" + image_name, save_dir=save_dir, norm_range=(common_min, common_max))
        visualise_2d_array(combined, "no show", "color", image_name="combined_" + image_name, save_dir=save_dir, norm_range=(common_min, common_max))
    else:
        image_name += ".png"
        visualise_2d_array(DL,"no show","color",image_name="directivity_"+image_name,save_dir=save_dir)
        visualise_2d_array(TL, "no show", "color", image_name="transmission_" + image_name, save_dir=save_dir)
        visualise_2d_array(SL, "no show", "color", image_name="scattering_" + image_name, save_dir=save_dir)
        visualise_2d_array(combined, "no show", "color", image_name="combined_" + image_name, save_dir=save_dir)



