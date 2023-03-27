from functools import partial
from numpy import cos,sin,tan, pi, log10, imag, real, arcsin, exp, array, arctan, where, errstate, nan_to_num, NaN, \
    power, divide, multiply, amin, amax
from toolbox import plot_function, create_grid, plot_3d_function, show_image, animate, plot_multiple, plot_from_df
import pandas as pd
from scipy.special import gamma, j0, itj0y0
from scipy.integrate import quad
from timeit import default_timer
from fundamentals import SimpleField
from scipy.optimize import minimize, Bounds
from os import getcwd
from os.path import join
from cv2 import imwrite

# Speed of sound calculations

# Temperature in Celsius
# Salinity in parts per thousand
# Depth in metres

def medwin_sos(temperature,salinity,depth):
    return 1449.2+4.6*temperature-0.055*temperature**2+(1.34-0.01*temperature)*(salinity-35)+0.016*depth

def mackenzie_sos(temperature,salinity,depth):
    return 1448.96 + 4.591*temperature + 0.05304*temperature**2 + 0.0002374*temperature**3 + 1.34*(salinity-35) + 0.0163*depth + 1.675*10**(-7)*depth**2

# DIRECTIVITY FUNCTIONS

# Beam Response (Linear Array)

# theta = angle
# d =  distance between point elements
# n = number
# lambd = lambda, wavelength
# L = length of array

# f = c/lambda

# EXTRACT BEAM WIDTHS IN HORIZONTAL AND VERTICAL DIRECTIONS

def decibel(signal,ref_signal):
    return 10*log10(signal/ref_signal)

def beam_widths(directivity_funct, *args, threshold=0.5):
    vert=0.001
    while directivity_funct(0,vert) > threshold:
        vert += 0.001
    horiz=0.001
    while directivity_funct(horiz,0) > threshold:
        horiz += 0.001
    if "degs" in args:
        horiz, vert = horiz*180/pi, vert*180/pi
    return 2*round(horiz,2), 2*round(vert,2)

def screen_dims(dir_funct,dist=50):
    theta, phi = beam_widths(dir_funct)
    w, h = dist*tan(theta), dist*tan(phi)
    return (w,h)

def angle_field(pixel_coord):
    return array([arctan(pixel_coord[0]/pixel_coord[1]),arctan(pixel_coord[2]/pixel_coord[1])])

def strength_field(test_funct, angle_info,threshold=0.5):
    strength = array(test_funct(angle_info[0], angle_info[1]))
    return where(strength < threshold, 0, strength)

def show_strength_field(s_field,*args,**kwargs):
    image = array(s_field*255,dtype="uint8")
    show_image(image,*args,**kwargs)

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

# SONAR EQUATIONS

#  SL - 2TL + TS = NL - DI + RD_n    (Noise background)

# SL - 2TL + TS = RL + RD_r          (Reverberation background)

# Power of SonaVison SV1010: 205 dB re µPa (nominal)

class Medium():

    def __init__(self,name, freq=800):
        if name == "salt":
            self.name = name
            self.depth = 10
            self.salinity = 35
            self.pH = 8.1
            self.temp = 5
            self.c = mackenzie_sos(self.temp, self.salinity, self.depth)
            self.rho = 1027 # Density of sea water at surface
        elif name == "fresh":  # VALUES FOR THESE
            self.name = name
            self.depth = 10
            self.salinity = 0.9   # check ppt or decimal
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
            self.rho = 1000+27*(self.salinity-0.9)/34.1
        else:
            raise ValueError("Intantiation failed. Invalid input.")
        
        self.freq = freq
        self.alpha = self.calculate_alpha()
        
    def __str__(self):
        properties=""
        for prop in self.__dict__:
            properties += prop +": "+str(self.__dict__[prop])+"\n"
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
        if self.salinity >2:
            # print(self.name, "Salinity:", self.salinity)
            return self.alpha_salt()/1000
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
        self.bulk = 5.941*10**(-3) - 2.371*10**(-4)*self.temp + 4.948*10**(-6)*self.temp**2 - 3.975*10**(-8)*self.temp**3
        ratio = 3.11 - 0.0155*self.temp
        self.shear = self.bulk/ratio  # Table from text. Assumes approx. 1-5 ATM pressure.
        return 8 * pi ** 2 * (self.freq*1000) ** 2 * (self.shear + 3 * self.bulk / 4) / (3 * self.rho * self.c ** 3)

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

    def reset_values(self,**kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = self.name
        self.__init__(name)

# SCATTERING CROSS-SECTION

# Ouput is scattering cross-section per unit solid angle per unit area, sigma

# A = Area over which scattering is measured
# r = reference range
# I_i is the incident intensity
# I_s is the scattered intensity

def scatxsection(r,A,I_s,I_i):
    return r**2*I_s/A/I_i

def lambertian_bistatic(th_i,th_s,**kwargs):
    if "mu" in kwargs:
        scatter_constant = kwargs["mu"]
    else:
        scatter_constant = -27
    return scatter_constant + 10 * log10(sin(th_i)*sin(th_s))

def lambertian_monostatic(th_i,**kwargs):
    if "mu" in kwargs:
        scatter_constant = kwargs["mu"]
    else:
        scatter_constant = -27
    if th_i == 0:
        return NaN
    return scatter_constant + 10 * log10(sin(th_i)**2)

class Material():

    def __init__(self,base,**kwargs):
        if base == "sand":
            self.density_ratio = 1.94  # This should not be fixed
            self.sos_ratio = 1.113   # This should not be fixed
            self.loss_param = 0.0115
            self.spectral_exp = 3.67
            self.spectral_str = 0.00422
            self.inhomog_str = 0.000127
            self.scatter_constant = -22

        elif base == "silt":
            self.density_ratio = 1.15
            self.sos_ratio = 0.987
            self.loss_param = 0.0386
            self.spectral_exp = 3.25
            self.spectral_str = 0.00518
            self.inhomog_str = 0.000306
            self.scatter_constant = -27

        elif base == "intermediate":
            if "hybrid" in kwargs:
                t = kwargs["hybrid"]
                self.density_ratio = 1.94*(1-t) + 1.15*t
                self.sos_ratio = 1.113*(1-t) + 0.987*t
                self.loss_param = 0.0115*(1-t) + 0.0386*t
                self.spectral_exp = 3.67*(1-t) + 3.25*t
                self.spectral_str = 0.00422*(1-t) + 0.0058*t
                self.inhomog_exp = 3
                self.inhomog_str = 0.000127*(1-t) + 0.000306*t
                self.fluctuation_ratio = -1
                self.scatter_constant = -22*(1-t) + -29*t
        else:
            ValueError("Not a recognised type of base.")

        self.fluctuation_ratio = -1 # Constant in test examples, not in genral
        self.inhomog_exp = 3  # Constant in test examples, not in genral
        self.rayleigh = self.gen_rayleigh() # For bistatic only
        self.alpha = self.spectral_exp/2 - 1
        self.C_h_squared = self.C_sq_param()
        self.rayleigh_mono = self.gen_rayleigh_mono()  # For monostatic only

        # DEVICE DEPENDENT PROPERTIES
        self.q = None
        self.freq = None
        self.power = None

        # MEDIUM DEPENDENT PROPERTIES
        self.medium = None
        self.c_med = None
        self.k_med = None
        self.kappa = None
        self.monostatic = None
        self.bistatic = None

    # ASSIGN SOURCE SETTING
    def set_freq(self, freq):
        self.freq = freq

    def set_power(self, power):
        self.power = power

    # ASSIGN MAIN MEDIUM QUALITIES
    def assign_fluid(self, medium):
        if medium.__class__ is not Medium:
            raise TypeError("Fluid medium must be an instance of Medium")
        self.medium = medium
        if self.medium.freq is not None:
            self.freq = self.medium.freq
        else:
            self.freq = 1000*int(input("Frequency not set. Please enter a frequency for device in kHz:"))
        self.c_med = self.medium.c
        self.k_med = self.gen_wave_number(self.c_med)
        self.c_mat = self.c_med * self.sos_ratio
        self.kappa = self.gen_kappa()
        return self

    # GENERATE INTRINSIC FUNCTIONS
    def gen_rayleigh(self):
        n = 1/self.sos_ratio
        m = self.density_ratio
        def raleigh(theta):
            v_1 = m * sin(theta)
            v_2 = (n ** 2 - cos(theta) ** 2)
            if v_2 < 0:
                return NaN
            else:
                v_2 = v_2 ** 0.5
            ratio = ((v_1 - v_2) / (v_1 + v_2)) 
        return raleigh
    
    def gen_rayleigh_mono(self):
        n = 1/self.sos_ratio
        m = self.density_ratio
        return (m-n)/(m+n)
    
    # DIFFUSE LAMBERTIAN SCATTERING
    def lambertian_bistatic(self, th_i, th_s):
        if th_i == 0 or th_s == 0:
            return NaN   # WHY?
        return self.scatter_constant + 10 * log10(sin(th_i) * sin(th_s))

    def lambertian_monostatic(self, th_i):
        if th_i == 0:
            return NaN  # WHY?
        return self.scatter_constant + 10 * log10(sin(th_i) ** 2)

    # BISTATIC SCATTERING
    def bistatic_scattering(self,th_i, bis, th_s):
        return 10*log10(self.roughness_scattering(th_i, bis, th_s) + self.volumetric_scattering(th_i, bis, th_s))

    def roughness_scattering(self,th_i, bis, th_s):
        print("angle",th_i)
        small_roughness = self.perturbution_scattering(th_i, bis, th_s)
        kirchoff_approx = self.kirchoff_scattering(th_i, bis, th_s)
        print("small:",small_roughness)
        print("kirch:", kirchoff_approx,type(kirchoff_approx))
        print("roughness:", power((power(small_roughness,-2)+power(kirchoff_approx,-2)),-0.5))
        return power((power(small_roughness,-2)+power(kirchoff_approx,-2)),-0.5)     # WHERE IS THIS FROM?

    # DELTA_TRANS
    def delta_t_param(self, th_i, bis, th_s):
        if th_i == th_s and bis == pi:
            return cos(th_i)
        else:
            return (cos(th_i) ** 2 - cos(th_i) * cos(th_s) * cos(bis) + cos(th_s) ** 2) ** 0.5 / 2

    # DELTA_V
    def delta_z_param(self, th_i, th_s):
        if th_i == th_s:
            return sin(th_i)
        else:
            return (sin(th_i) + sin(th_s)) / 2

    # DELTA
    def delta_param(self, delta_t, delta_z):
        return (delta_t ** 2 + delta_z ** 2) ** 0.5

    # CONSTANTS FOR BISTATIC KIRCHOFF CROSS-SECTION
    def C_sq_param(self):
        num = 2 * pi * self.spectral_exp * gamma(2 - self.alpha) * 2 ** (-2 * self.alpha)
        denom = self.alpha * (1 - self.alpha) * gamma(1 + self.alpha)
        return num / denom

    def q_param(self,th_i,bis,th_s):
        if th_i == th_s and bis == pi:
            print("Monostatic case (q)")
            return 2 * self.C_h_squared * (self.k_med * sin(th_i) * (2 * self.k_med * cos(th_i)) ** (-self.alpha)) ** 2
        else:
            delta_z = self.delta_z_param(th_i,th_s)
            delta_t = self.delta_t_param(th_i,bis,th_s)
            return 2 * self.C_h_squared * (self.k_med * delta_z * (2 * self.k_med * delta_t) ** (-self.alpha)) ** 2

    def theta_is(self, th_i, bis, th_s):
        if th_i == th_s and bis == pi:
            print("monostatic case (th_is)")
            return pi/2
        else:
            delta_z = self.delta_z_param(th_i, th_s)
            delta_t = self.delta_t_param(th_i, bis, th_s)
            delta = self.delta_param(delta_t,delta_z)
            return arcsin(delta)

    # BISTATIC KIRCHOFF CROSS-SECTION

    def kirchoff_integrand(self, th_i, bis, th_s):
        self.q = self.q_param(th_i,bis,th_s)
        def exponential_part(u):
            return exp(-self.q * u ** (2 * self.alpha))
        def complete_funct(u):
            return exponential_part(u) * j0(u) * u
        return complete_funct

    def kirchoff_integral(self, th_i, bis, th_s):
        integrand = self.kirchoff_integrand(th_i, bis, th_s)
        def integral_funct(x):
            return quad(integrand, 0, x)[0]
        return integral_funct

    def kirchoff_evaluation(self, th_i, bis, th_s):
        integral = self.kirchoff_integral(th_i, bis, th_s)
        limit = 100
        return integral(limit)

    def kirchoff_scattering(self, th_i, bis, th_s):
        D_t = self.delta_t_param(th_i, bis, th_s)
        D_z = self.delta_z_param(th_i, th_s)
        D = self.delta_param(D_t, D_z)
        th_is = self.theta_is(th_i, bis, th_s)
        R_this = self.rayleigh(th_is)
        I = self.kirchoff_evaluation(th_i, bis, th_s)
        return abs(R_this) ** 2 / (8 * pi) * (D / D_t / D_z) ** 2 * I
    
    # MONOSTATIC KIRCHOFF CROSS-SECTION
    
    def q_mono(self):
        k = self.k_med
        c_h_sq = self.C_h_squared
        alpha = self.alpha
        def q_part(th_i):
            return 2*c_h_sq* (k * sin(th_i) * (2 * k * cos(th_i)) ** (-alpha)) ** 2
        return q_part
    
    def kirchoff_integrand_mono(self, th_i):
        alpha = self.alpha
        q = self.q_mono()
        q_eval = q(th_i)
        def int_mono(u):
            return exp(-q_eval*u**(2*alpha))*j0(u)*u
        return int_mono
    
    def kirchoff_integral_mono(self, th_i):
        k_integrand = self.kirchoff_integrand_mono(th_i)
        def integral_funct(u):
            return quad(k_integrand, 0, u)[0]
        return integral_funct
    
    def kirchoff_evaluation_mono(self, th_i):
        k_integrand = self.kirchoff_integrand_mono(th_i)
        return quad(k_integrand, 0, 20)[0]
    
    def kirchoff_scattering_mono(self, th_i):
        r_thi_mono = self.rayleigh_mono
        coeff = (1/(2*pi))*(r_thi_mono/sin(th_i*pi/90))**2
        integral =  self.kirchoff_evaluation_mono(th_i)
        return coeff*integral
    
    def kirchoff_log(self,th_i):
        return 10*log10(self.kirchoff_scattering_mono(th_i))
    
    # CONSTANTS AND FUCNTIONS REQUIRED FOR PERTURBATION COMPONENT
    def gen_wave_number(self, c):
        if self.freq is None:
            self.freq = 1000*int(input("Frequency of device not set. Please enter frequency in kHz:"))
        return self.freq / c

    def gen_complex_wave_number(self):
        return self.k * self.loss_param

    def gen_kappa(self):
        return (1+self.loss_param*1j)/self.sos_ratio

    def complex_coeff(self,theta):  # P function in thesis
        return (self.kappa**2-cos(theta))**0.5

    def W_param(self,k):
        return self.spectral_str / k ** self.spectral_exp

    def G_param(self, th_i, bis, th_s):
        return (1/self.density_ratio)*(cos(th_i)*cos(bis)*cos(th_s)-self.complex_coeff(th_i)*self.complex_coeff(th_s)/self.density_ratio)\
               +1-self.kappa**2/self.density_ratio

    # PERTURBATION COMPONENT CALCULATION
    def perturbution_scattering(self,th_i, bis, th_s):
         return power(self.k_med**2 * abs(1+self.rayleigh(th_i)) * abs(1+self.rayleigh(th_s)) * abs(self.G_param(th_i,bis,th_s)), 2) \
                * self.W_param(2*self.k_med*self.delta_t_param(th_i,bis,th_s))

    # VOLUME SCATTERING COMSTANTS AND FUNCTIONS

    def W_rr(self,k):
        return self.inhomog_str/k**self.inhomog_exp

    # BRAGG WAVE NUMBER  # CHECK THIS FORMULA
    def bragg_k(self, th_i, bis, th_s):
        return self.k_med*(4*self.delta_t_param(th_i,bis,th_s)**2+real(self.complex_coeff(th_i)+self.complex_coeff(th_s))**2)**0.5

    # VOLUME SCATTERING FUNCTION AND CROSS-SECTION
    def volume_scattering(self,th_i,bis,th_s):
        return (pi/2*self.k_med**4)*abs(self.fluctuation_ratio*self.kappa**2+
                                        cos(th_i)*cos(bis)*cos(th_s)-self.complex_coeff(th_i)*self.complex_coeff(th_s))**2 \
               *self.W_rr(self.bragg_k(th_i,bis,th_s))

    def volume_xsection(self,th_i,bis,th_s):
        return (abs(1+self.rayleigh(th_i))*abs(1+self.rayleigh(th_s)))**2*self.volume_scattering(th_i,bis,th_s)/(
                2*self.k_med*self.density_ratio**2*imag(self.complex_coeff(th_i)+self.complex_coeff(th_s)))

    def gen_bistatic_scattering(self):
        def bistatic_scattering(th_i,bis,th_s):
            print("volumetric", self.volume_xsection(th_i, bis, th_s))
            return 10*log10(self.roughness_scattering(th_i, bis, th_s) + self.volume_xsection(th_i, bis, th_s))
        self.bistatic = bistatic_scattering

    def gen_monostatic_return(self):
        if self.bistatic is None:
            self.gen_bistatic_scattering()
        def monostatic_return(th_i):
            return partial(self.bistatic, bis=0, th_s=pi - th_i)
        def simple(th_i):
            return monostatic_return(th_i)(th_i)
        self.monostatic = simple

    def gen_monostatic_roughness(self):
        def monostatic_roughness(th_i):
            return partial(self.roughness_scattering, bis=0, th_s=pi - th_i)
        def simple(th_i):
            return monostatic_roughness(th_i)(th_i)
        return simple

    def gen_monostatic_volume(self):
        def monostatic_volume(th_i):
            return partial(self.volume_scattering, bis=0, th_s=pi - th_i)
        def simple(th_i):
            return monostatic_volume(th_i)(th_i)
        return simple

    # RESET CHARACTERISTIC

    def reset_device(self):
        # DEVICE DEPENDENT PROPERTIES
        self.q = None
        self.freq = None
        self.power = None

        # MEDIUM DEPENDENT PROPERTIES
        self.medium = None
        self.c_med = None
        self.k_med = None
        self.kappa = None
        self.monostatic = None
        self.bistatic = None

    def reset_medium(self):
        # MEDIUM DEPENDENT PROPERTIES
        self.medium = None
        self.c_med = None
        self.k_med = None
        self.kappa = None
        self.monostatic = None
        self.bistatic = None

        
# Function to calculate q when supplied with the incident angle

def q_part(material):
    k = material.k_med
    if k is None:
        raise AttributeError("No medium applied to scene. Call assign_fluid method with a medium first.")
    c_h_sq = material.C_h_squared
    alpha = material.alpha
    def q_mono(th_i):
        return 2*c_h_sq* (k * sin(th_i) * (2 * k * cos(th_i)) ** (-alpha)) ** 2
    return q_mono

# Function to show the integrand when supplied with the incident angle and material

def integrand(material,th_i):
    alpha = material.alpha
    q = q_part(material)
    q_eval = q(th_i)
    def int_mono(u):
        return exp(-q_eval*u**(2*alpha))*j0(u)*u
    return int_mono

def kirchoff_integral(material, th_i):
    k_integrand = integrand(material,th_i)
    def integral_funct(u):
        return quad(k_integrand, 0, u)[0]
    return integral_funct

def trig_part(theta):
    return (2/sin(theta*pi/90))**2

def kirchoff_scatter_deg(material,th_i):
    r_thi_mono = material.rayleigh_mono
    coeff = (1/(2*pi))*(r_thi_mono/sin(theta*pi/90))**2
    return coeff*material.kirchoff_evaluation_mono(th_i)