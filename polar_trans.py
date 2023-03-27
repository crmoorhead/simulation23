from numpy import zeros, ones, empty, power, add, pi, array, stack, meshgrid, where, round, sqrt, arctan2, \
    digitize, clip, matmul, concatenate
from cv2 import imshow, waitKey, resize, imwrite
from itertools import product
from timeit import default_timer
from os.path import join

# TO DO
# Scale this as needed.
# Wedge scanning

def polar_trans(scan, speed="normal", *args, **kwargs):
    start = default_timer()
    if speed == "normal":
        angle_res = 512
    elif speed == "slow":
        angle_res = 1024
    elif speed == "fast":
        angle_res = 256
    elif speed == "superfast":
        angle_res = 128
    else:
        raise ValueError("Not a valid speed.")

    (radial_res, time_steps) = scan.shape

    if "magnify" in kwargs:
        scale = kwargs["magnify"]
    else:
        scale = 1

    if "start_angle" in kwargs:
        if "degs" in args:
            offset = int(angle_res * kwargs["start_angle"]/360)
        else:
            offset = int(angle_res * kwargs["start_angle"]/(2*pi))

    if time_steps < angle_res:
        scan = concatenate((scan, zeros((radial_res,angle_res - time_steps + 1))),axis=1)

    r_bins = array([i for i in range(radial_res*2)])
    mapping = stack(meshgrid(r_bins, r_bins, indexing="xy"))
    r_sq = radial_res**2
    rs = add(power(mapping[0] - radial_res, 2), power(mapping[1] - radial_res, 2))
    rs = where(rs < r_sq, clip(array(round(sqrt(rs)),dtype=int),0,399), None)
    ths = arctan2(mapping[0]-radial_res, mapping[1]-radial_res)
    angle_step = 2*pi/angle_res
    theta_bins = [-pi+i*angle_step for i in range(angle_res)]
    ths = digitize(ths, theta_bins)-1
    if "start_angle" in kwargs:
        ths = (ths + offset) % angle_res
    # print("Create map:", default_timer()-start)
    canvas = empty((radial_res*2, radial_res*2))
    start=default_timer()
    for i in range(radial_res*2):
        for j in range(radial_res*2):
            if rs[i, j] is not None:
                canvas[i,j] = scan[rs[i,j],ths[i,j]]
            else:
                canvas[i,j] = 0
    # print("Create Image:",default_timer()-start)
    if "animate" not in args:
        if scale != 1:
            canvas = resize(canvas,(radial_res*scale, radial_res*scale))
        if "show" in args:
            imshow("Sonar scan",canvas)
            waitKey(0)
        if "save_dir" in kwargs:
            if "image_name" in kwargs:
                name = kwargs["image_name"]
            else:
                name = "untitled_scan.jpg"
            imwrite(join(kwargs["save_dir"], name), canvas*255)
    else:
        if "time_interval" not in kwargs:
            pause = 200
        else:
            pause = kwargs["time_interval"]


test = ones((400,128))
fading_radial = array([[1-i/400 for i in range(1,401)]]).transpose()
fading_angular = array([[1-i/128 for i in range(1,129)]])
# test = fading_radial*test
test = test*fading_angular
polar_trans(test,"normal","show","degs", start_angle=60, scale=2, save_dir=r"C:\Users\the_n\OneDrive\2022\Uni_2022\Simulation\Charts\full_scans")