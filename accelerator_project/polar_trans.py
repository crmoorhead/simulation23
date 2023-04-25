from numpy import zeros, ones, empty, power, add, pi, array, stack, meshgrid, where, round, sqrt, arctan2, \
    digitize, clip, matmul, concatenate, empty_like
from cv2 import imshow, waitKey, resize, imwrite
from itertools import product
from timeit import default_timer
from os.path import join
from os import getcwd

# TO DO
# Scale this as needed.
# Wedge scanning

def polar_trans(scan, speed="normal", *args, **kwargs):
    start = default_timer()
    res_lookup = {"normal": 512, "slow": 1024, "fast": 256, "superfast": 128}
    angle_res = res_lookup[speed]
    (radial_res, time_steps) = scan.shape
    if "magnify" in kwargs:
        scale = kwargs["magnify"]
    else:
        scale = 1
    if "start_angle" in kwargs:
        if "degs" in args:
            offset = int(angle_res * kwargs["start_angle"]/360) # index in terms of angle_res
        else:
            offset = int(angle_res * kwargs["start_angle"]/(2*pi))

    if time_steps < angle_res:
        scan = concatenate((scan, zeros((radial_res, angle_res - time_steps + 1))),axis=1)

    r_bins = array([i for i in range(radial_res*2)])
    mapping = stack(meshgrid(r_bins, r_bins, indexing="xy"))
    r_sq = radial_res**2
    rs = add(power(mapping[0] - radial_res, 2), power(mapping[1] - radial_res, 2))
    rs = where(rs < r_sq, clip(array(round(sqrt(rs)), dtype=int),0 , 399), None)
    ths = arctan2(mapping[0]-radial_res, mapping[1]-radial_res) # in radians
    angle_step = 2*pi/angle_res # in radians
    theta_bins = [-pi+i*angle_step for i in range(angle_res)] # in radians
    ths = digitize(ths, theta_bins)-1
    if "start_angle" in kwargs:
        ths = (ths + offset) % angle_res
    print("Create map:", default_timer()-start)
    canvas = empty((radial_res*2, radial_res*2))
    start = default_timer()
    if "animate" not in args:
        for i in range(radial_res * 2):
            for j in range(radial_res * 2):
                if rs[i, j] is not None:
                    canvas[i, j] = scan[rs[i, j], ths[i, j]]
                else:
                    canvas[i, j] = 0
        print("Create Image:", default_timer() - start)
        if scale != 1:
            canvas = resize(canvas, (radial_res*scale, radial_res*scale))
        if "show" in args:
            imshow("Sonar scan", canvas)
            waitKey(0)
    else:
        if "time_interval" not in kwargs:
            pause = 10
        else:
            pause = kwargs["time_interval"]
        partial_scan = empty_like(scan)
        start = default_timer()
        for t in range(time_steps):
            canvas = empty((radial_res * 2, radial_res * 2)) # Reset canvas
            partial_scan[:, t] = scan[:, t] # Update partial scan
            for i in range(radial_res*2): # Update canvas
                for j in range(radial_res*2):
                    if rs[i, j] is not None:
                        canvas[i, j] = partial_scan[rs[i, j], ths[i, j]]
                    else:
                        canvas[i, j] = 0 # Set background to black
            if scale != 1:
                # Scale canvas
                canvas = resize(canvas, (radial_res * scale, radial_res * scale))
            imshow("Sonar scan", canvas)
            if t == time_steps - 1:
                waitKey(0) # change to 0 to keep image on screen
            else:
                waitKey(pause)
        print("Animation Length:", default_timer() - start)

    if "save_dir" in kwargs:
        if "image_name" in kwargs:
            name = kwargs["image_name"]
        else:
            name = "untitled_scan.jpg"
        imwrite(join(kwargs["save_dir"], name), canvas * 255)


test = ones((400,128))
fading_radial = array([[1-i/400 for i in range(1,401)]]).transpose()
fading_angular = array([[1-i/128 for i in range(1,129)]])
test = fading_radial*test
# test = test*fading_angular
polar_trans(test, "normal", "show", "degs", start_angle=120, scale=2, save_dir=r"C:\Users\the_n\OneDrive\2023\Uni_2023\Simulation\Images")

class PolarDisplay:

    def __init__(self, speed="normal", start_angle=0, scale=1, *args, **kwargs):
        # Initialise lookup tables
        self.res_lookup = {"normal": 512, "slow": 1024, "fast": 256, "superfast": 128}
        self.radial_lookup = {"normal": 400, "slow": 400, "fast": 200, "superfast": 200}
        # Initialise variables
        self.angle_res = self.res_lookup[speed]
        self.radial_res = self.radial_lookup[speed]
        # Create pixel mapping
        self.r_bins = array([i for i in range(self.radial_res*2)])
        self.mapping = stack(meshgrid(self.r_bins, self.r_bins, indexing="xy"))
        # Create radial mask
        self.r_sq = self.radial_res**2
        self.rs = add(power(self.mapping[0] - self.radial_res, 2), power(self.mapping[1] - self.radial_res, 2))
        self.rs = where(self.rs < self.r_sq, clip(array(round(sqrt(self.rs)), dtype=int),0 , self.radial_res-1), None)
        # Calculate angles of each pixel using arctan2 and centre as (radial_res, radial_res)
        self.ths = arctan2(self.mapping[0]-self.radial_res, self.mapping[1]-self.radial_res)
        # Create what section of the circle each pixel is in
        self.angle_step = 2*pi/self.angle_res
        self.theta_bins = [-pi+i*self.angle_step for i in range(self.angle_res)]
        self.ths = digitize(self.ths, self.theta_bins)-1
        # Modify pixel mapping to account for start angle
        self.start_angle = start_angle # in degrees
        if start_angle != 0:
            self.ths = (self.ths + int(self.angle_res * self.start_angle/360)) % self.angle_res
        # Create canvas
        self.canvas = empty((self.radial_res*2, self.radial_res*2))
        self.scale = scale
        # Create partial scan. Empty unless animation is used
        self.partial_scan = None
        # Create time variable
        self.t = 0

    def perform_mapping(self, scan, *args, **kwargs):
        # Check if animation is being used
        # INSERT ANIMATION CODE HERE
        # Create offset for start angle
        if self.start_angle != 0:
            self.ths = (self.ths + int(self.angle_res * self.start_angle/360)) % self.angle_res
        # Create partial scan
        for i in range(self.radial_res*2):
            for j in range(self.radial_res*2):
                if self.rs[i, j] is not None:
                    self.canvas[i, j] = self.scan[self.rs[i, j], self.ths[i, j]]
                else:
                    self.canvas[i, j] = 0
        return self.canvas


    def show(self, *args, **kwargs):
        if "time_interval" not in kwargs:
            pause = 10
        else:
            pause = kwargs["time_interval"]
        # Scale canvas if required
        if self.scale != 1:
            curr_canvas = resize(self.canvas,
                                 (self.radial_res * self.scale, self.radial_res * self.scale))
        imshow("Sonar scan", curr_canvas)
        if self.t == self.time_steps - 1:
            waitKey(0)

    # Save image to file
    def save(self, save_dir=getcwd(), image_name="untitled_scan.jpg", scale=1):
        # Scale canvas if required
        if scale != 1:
            self.canvas = resize(self.canvas, (self.radial_res * scale, self.radial_res * scale))
        imwrite(join(save_dir, image_name), self.canvas * 255)

    def animate(self, scan, *args, **kwargs):
        start = default_timer()
        self.scan = scan
        self.time_steps = scan.shape[1]
        for self.t in range(self.time_steps):
            self.update(scan, *args, **kwargs)
            self.show(*args, **kwargs)
            self.save(*args, **kwargs)
        print("Animation Length:", default_timer() - start)