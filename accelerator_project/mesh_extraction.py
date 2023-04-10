from project_geometry import *
from argparse import ArgumentParser
from math import exp
from os import getcwd
from os.path import join

# ARRAY EXTRACTION

# array_from_implicit and array_from_explicit are used to extract arrays from functions. array from explicit is used to
# extract a height map from a function, f(x,y). array_from_implicit is used to extract sampled values from an implicit
# function, f(x,y,z). The functions are used to create a Tesselation object, which is then used to create an OBJ file.

# HEIGHT MAP EXTRACTIONS

"""Given a function f(x,y) that gives a height map for a surfact, the following code allows for a mesh
extraction of the described continuous surface."""

def f(x, y):
    return 0

def gaussian_hill(x, y, a=4, sigma=3.5, centre=(0, 10)):
    return a*exp(-((x-centre[0])**2+(y-centre[1])**2)/sigma**2)

def gaussian_hole(x, y, a=4, sigma=3.5, centre=(0, 10)):
    return -gaussian_hill(x, y, a, sigma, centre)

def rolling_bank(x, y, a=3, sigma=2.5, dist=10):
    return a*exp(-((y-dist)**2)/sigma**2)

def trough(x, y, depth=-1, x_grad=0.3, y_grad=0.1, surf_level=0):
    return max(abs(x_grad*x-depth),surf_level) + y*y_grad

def valley(x, y):
    return 0

# EDIT AS NEEDED
level = 4
funct = trough
save_dir = getcwd()
subdivisions = "random"

# MAIN CODE
divisions = 2**level
triangle_number = 2*(divisions+1)**2
file_name = funct.__name__ +"_"+ str(triangle_number)+".obj" # Filename format contains function name and no. of triangles
file_path = join(save_dir, file_name)
h_map = array_from_explicit(funct, [-10, 10], [0, 20], divisions) # Creates a sample of a 20m by 20m area with centre (0, 10)
# Second argument can be removed or "down" to change default triangular divisions of quadilateral face
tess = Tesselation(h_map, subdivisions)
# tess_to_OBJ(tess, join(file_path))

# SAMPLE HIERARCHY

""""Does the same as above, but extracts at different levels of sampling in order to increase complexity 
without changing the underlying surface. Be aware, however, of the Nyquist sampling theorem when creating new functions"""

# Necessary functions

def scale_sample(funct, x_range, y_range, levels, low=1):
    if not isinstance(levels, int) or low > levels:
        raise ValueError("Subdivision levels must be higher than {}.".format(low))
    sampled_arrays = []
    for i in range(low, levels+low):
        sampled_arrays.append(array_from_explicit(funct, x_range, y_range, 2**i))
    return sampled_arrays

def nested_tesselations(funct, x_range, y_range, levels, low=1, subdivisions="random", **kwargs):
    samples = scale_sample(funct, x_range, y_range, levels, low)
    if "name" in kwargs:
        stem = kwargs["name"]
    else:
        stem = funct.__name__
    tesselations = {}
    for s in samples:
        t = Tesselation(s, subdivisions)
        tesselations[stem+"_{}".format(t.component_count)] = t
    return tesselations

def nested_sample_to_OBJ(funct, x_range, y_range, levels=4, low=1, **kwargs):
    nt = nested_tesselations(funct, x_range, y_range, levels, low, **kwargs)
    obs = {}
    if "save_dir" in kwargs:
        root = kwargs["save_dir"]
    else:
        root = getcwd()
    for t in nt:
        obs[t] = tess_to_OBJ(nt[t], join(root, t+".obj"))
    return obs

# Change

hierarchy_levels = 4
low = 2

# MAIN CODE
nested_sample_to_OBJ(funct, [-10, 10], [0, 20], levels=hierarchy_levels, low=low, save_dir=save_dir)

# IMPLICIT SURFACE EXTRACTION

def implicit_to_OBJ(implicit, sample_width, bounds, file_path, *args, **kwargs):
    if not isinstance(implicit, Implicit):
        raise TypeError("Input must be an instance of Implicit.")
    else:
        # CREATE ARRAY FROM FUNCTION
        sample_array = array_from_implicit(implicit, sample_width, bounds)
        verts, faces, _, _ = marching_cubes(sample_array, 0)
        verts *= sample_width
        verts += array([bounds[i][0] for i in range(3)])

        # REMOVE DUPLICATES
        unique_verts = []
        renewed_faces = faces.copy()
        counter = 0
        for v in verts:
            if list(v) not in unique_verts:
                unique_verts.append(list(v))
                counter += 1
            else:
                previous = unique_verts.index(list(v))
                for f in range(len(faces)):
                    for c in range(3):
                        if faces[f][c] > counter:
                            renewed_faces[f][c] -= 1
                        if faces[f][c] == counter:
                            renewed_faces[f][c] = previous
                counter += 1
        faces = [[v+1 for v in f] for f in renewed_faces if len(set(f))==3]
        max_vertex = len(unique_verts)
        faces = [f for f in faces if True not in [f[v] > max_vertex for v in range(3)]]
        # WRITE TO OBJ FILE
        f = open(file_path, "w")
        if "comments" in kwargs:
            f.write("#" + str(kwargs["comments"]) + " " + implicit.formula + "\n\n")
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
    implicit_obj = ObjFile(file_path)

    # SHOW PLOT
    if "show" in args:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        xyz_centre = [(bounds[i][1]+bounds[i][0])/2 for i in range(3)]
        xyz_spans = [1.1*(bounds[i][1]-bounds[i][0])/2 for i in range(3)]
        cube_width = max(xyz_spans)

        ax.set_xlim(xyz_centre[0]-cube_width, xyz_centre[0]+cube_width)
        ax.set_ylim(xyz_centre[0]-cube_width, xyz_centre[0]-cube_width)
        ax.set_zlim(xyz_centre[0]-cube_width, xyz_centre[0]-cube_width)

        plt.tight_layout()
        plt.show()

    return implicit_obj


"""Given an intrinsic formula f(x,y,z)=0 describing a 2D manifold in 3D space, this applies the marching cubes 
algorithm that supplies the starting point for mesh simplification

Note that these shapes are not necessarily best represented as a mesh, but they are useful for mesh simplification tests"""


def f(x, y, z):
    return 0

def sphere(x, y, z, r=2, centre=[0,10,3]):
    return (x-centre[0])**2 + (y-centre[1])**2 + (z-centre[2])**2 -r**2

def ellipsoid(x, y, z, a=1, b=2, c=3, centre=[0,0,0]):
    return (x-centre[0])**2/a**2 + (y-centre[1])**2/b**2 + (z-centre[2])**2/c**2 -1

def torus(x, y, z):
    return

def smoothed_cube(x, y, z, r=2, centre=[0,1,0]):
    return (x-centre[0])**8 + (y-centre[1])**8 + (z-centre[2])**8 -r**8

def pipe(x, y, z, r=1, centre=10, slope=0):
    return

# EDIT AS NEEDED
sample_width = 0.2
im_funct = smoothed_cube
scope = 10

# DERIVED AND FIXED VARIABLES
bounds = [[-scope, scope] for i in range(3)]
implicit = Implicit(im_funct, "(x-x_c)^2+(y-y_c)^2+(z-z_c)^2-r^2")
imp_file_name = im_funct.__name__ +"_"+str(sample_width)+".obj"
implicit_to_OBJ(implicit, sample_width, bounds, join(save_dir, imp_file_name))


# SCALED SAMPLING

def scaled_implicit_extract(im_funct, sample_widths=[0.1, 0.2, 0.3, 0.4, 0.5]):
    for s in sample_widths:
        implicit = Implicit(im_funct, "(x-x_c)^2+(y-y_c)^2+(z-z_c)^2-r^2")
        imp_file_name = im_funct.__name__ + "_" + str(s) + ".obj"
        implicit_to_OBJ(implicit, s, bounds, join(save_dir, imp_file_name))

scaled_implicit_extract(ellipsoid)


"""if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("type", type=str, required=True)
    parser.add_argument("f", type=str, required=True)
"""
