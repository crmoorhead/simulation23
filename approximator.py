# Data Generator

from random import random, choice
from sonar_formulae import Medium, Material
from numpy import array, zeros, loadtxt, savetxt

def generate_media():
    temp = round(random()*20,1)
    if random() < 0.5:
        medium = Medium("salt")
        salinity = round(34+2*random(), 1)
        pH = round(8.05+0.1*random(), 1)
        depth = round(10+random()*1990, 0)
    else:
        medium = Medium("fresh")
        salinity = round(0.3+0.6*random(), 1)
        pH = round(6.5 + 1.5*random(), 1)
        depth = round(5 + random()*35, 0)
    
    params = {"temp":temp, "depth":depth, "salinity":salinity, "pH":pH}
    
    for p in params:
        medium.change_property(p, params[p])
    
    return medium

def generate_material():

        # Used in Bell (1997)
        
        sand_bell = {"name": "sand_bell",
                    "density_ratio": 1.94, 
                    "sos_ratio": 1.113, 
                    "loss_param": 0.0115, 
                    "spectral_exp":3.67, 
                    "spectral_str": 0.00422, 
                    "inhomog_str": 0.000127, 
                    "scatter_constant": -22,
                    "fluctuation_ratio": -1,
                    "inhomg_exp": 3}

        silt_bell = {"name": "silt_bell",
                    "density_ratio": 1.15, 
                    "sos_ratio": 0.987, 
                    "loss_param": 0.00386, 
                    "spectral_exp":3.25, 
                    "spectral_str": 0.000518, 
                    "inhomog_str": 0.000306, 
                    "scatter_constant": -27,
                    "fluctuation_ratio": -1,
                    "inhomg_exp": 3}

        # Williams/Jackson

        eckemforde_silty = {"name": "eckemforde_silty",
                    "density_ratio": 1.18, 
                    "sos_ratio": 0.991, 
                    "loss_param": 0.00186, 
                    "spectral_exp":3.42, 
                    "spectral_str": 0.00231, 
                    "inhomog_str": 0.00013, 
                    "scatter_constant": -27,
                    "fluctuation_ratio": -0.69,
                    "inhomg_exp": 4}

        panama_sandy = {"name": "panama_sandy",
                    "density_ratio": 1.97, 
                    "sos_ratio": 1.126, 
                    "loss_param": 0.0166, 
                    "spectral_exp":3.12, 
                    "spectral_str": 0.00849, 
                    "inhomog_str": 0.0000161, 
                    "scatter_constant": -22,
                    "fluctuation_ratio": -2.44,
                    "inhomg_exp": 4}

    # Heniotis/Negreira

        anthropogenic = {"name": "anthropogenic",
                    "density_ratio": 1.35, 
                    "sos_ratio": 1.29, 
                    "loss_param": 0.0001, 
                    "spectral_exp":3.25, 
                    "spectral_str": 0.1656, 
                    "inhomog_str": 0.000127, 
                    "scatter_constant": 25,
                    "fluctuation_ratio": -1,
                    "inhomg_exp": 3}

        cobble = {"name": "cobble",
                    "density_ratio": 2.49, 
                    "sos_ratio": 1.78, 
                    "loss_param": 0.014, 
                    "spectral_exp":3.25, 
                    "spectral_str": 0.061, 
                    "inhomog_str": 0.000127, 
                    "scatter_constant": 27,
                    "fluctuation_ratio": -1,
                    "inhomg_exp": 3}

        coarse_sand = {"name": "coarse_sand",
                    "density_ratio": 2.44, 
                    "sos_ratio": 1.36, 
                    "loss_param": 0.017, 
                    "spectral_exp": 3.25, 
                    "spectral_str": 0.048, 
                    "inhomog_str": 0.000127, 
                    "scatter_constant": -22,
                    "fluctuation_ratio": -1,
                    "inhomg_exp": 3}

        medium_sand = {"name": "medium_sand",
                    "density_ratio": 1.67, 
                    "sos_ratio": 1.16, 
                    "loss_param": 0.01, 
                    "spectral_exp": 3.25, 
                    "spectral_str": 0.059, 
                    "inhomog_str": 0.0002, 
                    "scatter_constant": -22,
                    "fluctuation_ratio": -1,
                    "inhomg_exp": 3}

        fine_sand= {"name": "fine_sand",
                    "density_ratio": 1.44, 
                    "sos_ratio": 1.06, 
                    "loss_param": 0.015, 
                    "spectral_exp": 3.25, 
                    "spectral_str": 0.0015, 
                    "inhomog_str": 0.00025, 
                    "scatter_constant": -22,
                    "fluctuation_ratio": -1,
                    "inhomg_exp": 3}

        medium_silt = {"name": "medium_silt",
                    "density_ratio": 1.15, 
                    "sos_ratio": 1.09, 
                    "loss_param": 0.07, 
                    "spectral_exp": 3.25, 
                    "spectral_str": 0.0005, 
                    "inhomog_str": 0.0003, 
                    "scatter_constant": -27,
                    "fluctuation_ratio": -1,
                    "inhomg_exp": 3}

        muddy_sand = {"name": "muddy_sand",
                    "density_ratio": 1.35, 
                    "sos_ratio": 1.19, 
                    "loss_param": 0.024, 
                    "spectral_exp": 3.25, 
                    "spectral_str": 0.004, 
                    "inhomog_str": 0.00025, 
                    "scatter_constant": -24,
                    "fluctuation_ratio": -1,
                    "inhomg_exp": 3}
        
        materials = [sand_bell, silt_bell, eckemforde_silty, panama_sandy, anthropogenic, cobble, 
                     coarse_sand, medium_sand, fine_sand, medium_silt, muddy_sand]
        
        random_a = choice(materials)
        if random()<0.5:
            random_b = choice(materials)
            while random_b["name"] == random_a["name"]:
                random_b = choice(materials)
            hybrid = {}
            t = random()
            for param in random_a:
                if isinstance(random_a[param], str):
                    hybrid[param] = random_a[param]+": "+str(t)+" "+random_b[param]+": "+str(1-t)
                else:
                    hybrid[param] = t*random_a[param] + (1-t)*random_b[param]

            material = Material("sand")
            material.name = hybrid["name"]
            material.density_ratio = hybrid["density_ratio"]
            material.sos_ratio = hybrid["sos_ratio"]
            material.loss_param = hybrid["loss_param"]
            material.spectral_exp = hybrid["spectral_exp"]
            material.spectral_str = hybrid["spectral_str"]
            material.inhomog_str = hybrid["inhomog_str"]
            material.scatter_constant = hybrid["scatter_constant"]
            material.inhomog_exp = hybrid["inhomg_exp"]

        else:
            material = Material("sand")
            material.name = random_a["name"]
            material.density_ratio = random_a["density_ratio"]
            material.sos_ratio = random_a["sos_ratio"]
            material.loss_param = random_a["loss_param"]
            material.spectral_exp = random_a["spectral_exp"]
            material.spectral_str = random_a["spectral_str"]
            material.inhomog_str = random_a["inhomog_str"]
            material.scatter_constant = random_a["scatter_constant"]
            material.inhomog_exp = random_a["inhomg_exp"]
        material.alpha = material.spectral_exp/2 - 1
        material.C_h_squared = material.C_sq_param()
        return material

# FUNCTION TO GIVE INPUTS TO SYSTEM

def random_interaction():
    med = generate_media()
    mat = generate_material()


# GENERATE FILES STORING INPUT VARIABLES

def gen_in_vars(n, funct):
    m = len(funct())
    a = zeros((n,m))
    for i in range(n):
        inputs = funct()
        a[i,:] = array(inputs)
    return a

# TEST OUTPUT FUNCTION

def test_funct(a,b,c,d):
    return -a**2*sin(b/(c**2+1))+d*a*b*c/10000 + a/d

# GENERATE OUTPUTS FROM FUNCTION

def gen_out_cars(inputs, funct):
    m = len(funct())
    n = inputs.shape[0]
    outputs = zeros(n,m)
    for i in range(n):
        outputs[i] = funct(inputs[i,0], inputs[i,1], inputs[i,2], inputs[i,3])
    return outputs

def big_data(batch_size,files,dataset_name="data",funct=test_funct):
    if dataset_name not in listdir(getcwd()):
        mkdir(dataset_name)
    for f in range(1,files+1):
        data = gen_in_vars(batch_size)
        outs = gen_out_cars(data,funct)
        savetxt(join(dataset_name,"input_data_{}.txt".format(f)),data)
        savetxt(join(dataset_name, "output_data_{}.txt".format(f)), outs)

# We want the input to our system to be values for C_h^2 and q. With

# Read and write values

def gen_inputs(n,):
    
    for i in range(n):
        med = generate_media()
        
def gen_q(n):
    pass
        






       
