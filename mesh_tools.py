from geometry_tools import *
from fundamentals import Rotation, Noise, no_noise, SVec
from inspect import isfunction, isbuiltin, signature
from functools import partial
from numpy.random import uniform
from numpy import arccos, sin, cos, pi, concatenate, vectorize, array
from scipy.spatial import ConvexHull
from toolbox import identifier

class ObjTransform:

    def __init__(self,name,trans,*args,**kwargs):
        if not (isfunction(trans) or isbuiltin(trans) or trans.__class__ is partial(print).__class__):
            raise TypeError("Transformation must be an instance of function or function-like class")
        sig = signature(trans)
        self.params = sig.parameters
        if "x" not in self.params or "y" not in self.params or "z" not in self.params:
            raise ValueError("Transformation function must have x, y and z as arguments.")
        self.trans=vectorize(trans,otypes=[float])
        print(self.trans.__class__)
        if "trans_limits" not in kwargs:
            self.domain=[None,None,None]
        else:
            if kwargs["trans_limits"].__class__ is not list:
                raise TypeError("Domain of the transformation must be a list of tuples of length 2 or None. This is not a list.")
            elif len(kwargs["trans_limits"]) is not 3:
                raise TypeError("Domain of the transformation must be a list of tuples of length 2 or None. Length of list must be 3.")
            elif False in [b.__class__ in [list,tuple,None.__class__] for b in kwargs["trans_limits"]]:
                raise TypeError("Domain of the transformation must be a list of tuples of length 2 or None. One of the entries is of incorrect type.")
            else:
                self.domain = kwargs["trans_limits"]
        self.name=name
        self.trans.__name__=self.name

    def apply_to_point(self,point,*args,**kwargs):
        if point.__class__ not in [SVec,list,array(0).__class__]:
            raise TypeError("Input must be an instance of SVec, list or array.")
        if point.__class__ is SVec:
            point=point.value
            print(point, point.__class__)
            return SVec(self.apply_to_array(point,*args, **kwargs))
        elif point.__class__ is list:
            if len(point) is not 3:
                raise ValueError("Point must be defined using 3 co-ordinates.")
            else:
                point=array(point,dtype=float)
                return list(self.trans(point[0],point[1],point[2],*args,**kwargs))
        else:
            return self.apply_to_array(point,*args,**kwargs)

    def apply_to_array(self,input,*args,**kwargs):
        print(len(input.shape),input.shape)
        if len(input.shape)==1:
            return self.trans(input[0],input[1],input[2],*args,**kwargs)
        else:
            print(len(input.shape), input.shape,input[:,0])
            print("Many points")
            return self.trans(list(input[:,0]),list(input[:,1]),list(input[:,2]),*args,**kwargs)

    def apply_to_obj(self,obj):
        if obj.__class__ not in [str,ObjFile]:
            raise TypeError

    def apply_to_mesh(self,mesh):
        if mesh.__class__ is not Mesh:
            raise TypeError("Input must me mesh")

    def merge(self,other):
        if other.__class__ is not ObjTransform:
            raise TypeError("Only two MeshTransform objects can be combined.")

    def remove_duplicates(self,mesh):
        pass

def translate(vector,*args,**kwargs):
    if vector.__class__ != SVec:
        raise TypeError("Variable must be an instance of SVec.")
    def fixed_trans(x,y,z):
        return array([x+vector.x,y+vector.y,z+vector.z])
    return fixed_trans

# EXAMPLES OF MESH GENERATORS

# TYRES/SQUARED TORI

def tyre_by_param(r,a,b,s,c, phi, theta):
    if c.__class__ is not SVec:
        return TypeError("Centre must be an instance of SVec")
    for p in r,a,b,s:
        if p.__class__ not in [int,float]:
            raise TypeError("Input {} needs to be an integer or float value".format(p))
    for i in [phi,theta]:
        if i.__class__ is not list:
            i=[i]
        for angle in i:
            if angle.__class__ not in [int,float]:
                raise TypeError("Input {} in list {} needs to be an integer or float value".format(angle,i))
    s += 2
    if phi == 0 or phi is None:
        rot_1s=[None]
        rot_2s=[None]
    else:
        rot_1s=[]
        for p in phi:
            rot_1s.append(Rotation(Angle(0),Angle(p),Angle(0)))
    if theta == 0 or theta is None:
        rot_2s = [None]
    else:
        if not (phi == 0 or phi is None):
            rot_2s = []
            for t in theta:
                rot_2s.append(Rotation(Angle(t), Angle(0), Angle(0)))
    x_c,y_c,z_c = c.x, c.y, c.z
    def tyre(x,y,z):
        return (abs(z - z_c)/a)**s + (abs(((x - x_c)**2 + (y - y_c)**2)**0.5 - r)/b) ** s - 1
    formula = "Tyre(r={} a={} b={} s={} c=({},{},{}))".format(r,a,b,s-2,c.x,c.y,c.z)
    radial, vertical = (r+b)*1.8, a*1.8
    spacing = 2*min([a,b])/3
    bounds=[[x_c-radial,x_c+radial], [y_c-radial,y_c+radial], [z_c-vertical,z_c+vertical]]
    implicit=Implicit(tyre,formula=formula)
    # Create default object mesh
    comments = "Squared torus of radius {}, radial width {}, depth {}, squareness factor of {}, " \
               "centred at ({},{},{}). Phi is 0. Theta is 0.".format(r, 2 * a, 2 * b, s - 2, c.x, c.y, c.z)
    tyre_obj = implicit_to_OBJ(implicit, spacing, bounds,
                               "r={} a={} b={} s={} c=({},{},{}) rot=(0,0,0).obj".format(r, a, b, s - 2, c.x, c.y, c.z),
                               comments=comments)
    tyre_mesh = Mesh(tyre_obj)
    tyre_mesh.save_im(formula + ".jpg") # Base object
    if phi == 0 or phi is None:
        return tyre_mesh
    tyre_mesh.rotate_object(rot_1s[0])
    tyre_mesh.rotate_object(rot_2s[0])
    comments = "Squared torus of radius {}, ring width {}, depth {}, squareness factor of {}, " \
               "centred at ({},{},{}). Phi is {}. Theta is {}.".format(r, 2*a, 2*b, s-2, c.x, c.y, c.z, phi[0], theta[0])
    file_path = "r={} a={} b={} s={} c=({},{},{}) rot=({},{},0).obj".format(r, a, b, s-2, c.x, c.y, c.z, theta[0], phi[0])
    tyre_mesh.write(file_path=file_path,comments=comments)
    tyre_mesh.save_im(file_path[:-4] + ".jpg") # First phi and theta combination
    pre_theta= rot_2s[0]
    pre_phi= rot_1s[0]
    # All thetas for first phi
    for j in rot_2s[1:]:
        rel_rot_2 = j + (-pre_theta)
        tyre_mesh.rotate_object(rel_rot_2)
        comments = "Squared torus of radius {}, ring width {}, depth {}, squareness factor of {}, " \
                   "centred at ({},{},{}). Phi is {}. Theta is {}.".format(r, 2*a, 2*b, s-2, c.x, c.y, c.z, phi[0], j.theta.value)
        file_path = "r={} a={} b={} s={} c=({},{},{}) rot=({},{},0).obj".format(r, a, b, s-2, c.x, c.y, c.z, j.theta.value, phi[0])
        tyre_mesh.write(file_path=file_path, comments=comments)
        tyre_mesh.save_im(file_path[:-4]+".jpg")  #  All remaining thetas for first phi
        pre_theta = j
    # Repeat for remaining phis. We need to reset the tyre to the flat position.
    tyre_mesh.rotate_object(-pre_theta)
    tyre_mesh.rotate_object(-pre_phi)
    for i in rot_1s[1:]:
        tyre_mesh.rotate_object(i)
        tyre_mesh.rotate_object(rot_2s[0])  # Do for the first value of theta of for new phi
        comments = "Squared torus of radius {}, ring width {}, depth {}, squareness factor of {}, " \
                   "centred at ({},{},{}). Phi is {}. Theta is {}.".format(r, 2 * a, 2 * b, s - 2, c.x, c.y, c.z,
                                                                           i.phi.value, theta[0])
        file_path = "r={} a={} b={} s={} c=({},{},{}) rot=({},{},0).obj".format(r, a, b, s - 2, c.x, c.y, c.z, theta[0], i.phi.value)
        tyre_mesh.write(file_path=file_path, comments=comments)
        tyre_mesh.save_im(file_path[:-4]+".jpg") # Next phi, first theta
        pre_phi = i
        pre_theta = rot_2s[0]
        for j in rot_2s[1:]:
            rel_rot_2=j+(-pre_theta)
            tyre_mesh.rotate_object(rel_rot_2)
            comments = "Squared torus of radius {}, ring width {}, depth {}, squareness factor of {}, " \
                       "centred at ({},{},{}). Phi is {}. Theta is {}.".format(r, 2 * a, 2 * b, s - 2, c.x, c.y, c.z,
                                                                               i.phi.value, j.theta.value)
            file_path = "r={} a={} b={} s={} c=({},{},{}) rot=({},{},0).obj".format(r, a, b, s - 2, c.x, c.y, c.z,
                                                                                    j.theta.value, i.phi.value)
            tyre_mesh.write(file_path=file_path, comments=comments)
            tyre_mesh.save_im(file_path[:-4] + ".jpg") # Remaining values of theta for phi
            pre_theta=j
        # reset to flat position each time
        tyre_mesh.rotate_object(-pre_theta)
        tyre_mesh.rotate_object(-pre_phi)
    return tyre_mesh

# CONVEX HULLS

def unif_sphere(radius,centre,n):
    if centre.__class__ is not SVec:
        raise TypeError("Centre must be an instance of SVec.")
    thetas=uniform(0,2*pi,(n,1))
    phis=arccos(1-uniform(0,2,(n,1)))
    xs=sin(phis)*cos(thetas)
    ys=sin(phis)*sin(thetas)
    zs=cos(phis)
    points=radius*(concatenate([xs,ys,zs],axis=-1)+centre.value)
    return points

def convex(radius,centre,n,*args,**kwargs):
    points=unif_sphere(radius,centre,n)
    chull=ConvexHull(points)
    points, vertices, faces = list(points), list(chull.vertices), list(chull.simplices+1)
    ID=identifier(6)
    path="ConvexHull ({}).obj".format(ID)
    obj_f=open(path,"w")
    if "comments" in kwargs:
        obj_f.write("# "+str(kwargs("comments"))+"\n\n")
    obj_f.write("# VERTICES\n\n")
    for v in vertices:
        obj_f.write("v")
        for p in list(points[v]):
            obj_f.write(" "+str(p))
        obj_f.write("\n")
    obj_f.write("\n# FACES\n\n")
    for f in faces:
        obj_f.write("f")
        for p in list(f):
            obj_f.write(" "+str(p))
        obj_f.write("\n")
    obj_f.close()
    ch_mesh=Mesh(path)
    return ch_mesh

# ELLIPSOID

def ellipsoid(dims,centre,*args,**kwargs):
    if dims.__class__ not in [int,float,list]:
        raise TypeError("Outer dimensions of shape must be defined be int, float or list of ints or floats.")
    if centre.__class__ is not SVec:
        raise TypeError("centre must be an SVec coordinate.")
    if dims.__class__ in [int, float]:
        a = b = c = dims
    else:
        if len(dims)==2:
            a=b=dims[0]
            c=dims[1]
        elif len(dims)==3:
            [a,b,c]=dims
        elif len(dims)==1:
            a=b=c=dims[0]
        else:
            raise ValueError("dims argument is incorrect length")
    x_c, y_c, z_c = centre.x, centre.y, centre.z
    def ellipsoid(x,y,z):
        return ((z-z_c)/a)**2+((y-y_c)/b)**2+((x-x_c)/c)**2-1
    sample_width=min([a,b,c])/4
    [x_r, y_r , z_r] = [1.8*i for i in [c,b,a]]
    bounds=[[x_c-x_r,x_c+x_r],[y_c-y_r,y_c+y_r],[z_c-z_r,z_c+z_r]]
    path="Ellipsoid (dims=({},{},{}),centre=({},{}.{}).obj".format(a,b,c,x_c,y_c,z_c)
    implicit=Implicit(ellipsoid,"((z-{})/{})^2+(y-{})/{})^2+(x-{})/{})^2=1".format(z_c,a,y_c,b,x_c,c),*args,**kwargs)
    ellipse_OBJ=implicit_to_OBJ(implicit,sample_width,bounds,*args,**{**kwargs,**{"file_path":path}})
    mesh=Mesh(ellipse_OBJ,*args,**kwargs)
    if "rotation" in kwargs:
        mesh=mesh.rotate_object(kwargs["rotation"],*args,**kwargs)
    return mesh

# AMORPHOUS OBJECT

# FRAMES

# EXAMPLE MESH TRANSFORMATIONS

def add_noise(x,y,z,noise,*args,**kwargs):
    if noise.__class__ is Noise:
        x_noise=noise
        y_noise=noise
        z_noise=noise
    elif noise.__class__ is list:
        if len(noise) != 3:
            raise ValueError("Noise must be defined for each axis if supplying a list")
        if True in [c.__class__ not in [Noise,None.__class__] for c in noise]:
            raise ValueError("Noise must be defined as a Noise object or None for each coordinate.")
        else:
            [x_noise, y_noise, z_noise]=[i if i.__class__ is not None.__class__ else no_noise() for i in noise]

    def noisy(x,y,z,*args,**kwargs):
        return x_noise.apply(x), y_noise.apply(x), z_noise.apply(z)

def noisy_triangle(tri,min_area,roughness,frac_dim,*args,**kwargs):
    if tri.__class__ is Triangle:
        p1,p2,p3=tri.p1,tri.p2,tri.p3
    elif tri.__class__ is list:
        if len(tri) == 3:
            ps=[]
            for p in tri:
                if p.__class__ is SVec:
                    ps.append(list(p.value))
                elif p.__class__ is list:
                    ps.append(p)
                elif p.__class__ is array(0).__class__:
                    ps.append(list(p))
                else:
                    raise ValueError("Vertex not defined properly.")
            [p1, p2, p3] = ps
        else:
            raise ValueError("Number of points needs to be 3.")
    scale=roughness*min([abs(tri.u), abs(tri.v),abs(tri.u-tri.v)])/6
    if "iter" in kwargs:
        i = kwargs["iter"]
    else:
        i=1
    if "area" in kwargs:
        area = kwargs["area"] # makes iterative process cleaner
    if tri.__class__ is not Triangle:
        u=SVec(array([p2[i]-p1[i] for i in range(3)]))
        v=SVec(array([p3[i]-p1[i] for i in range(3)]))
        area = abs(u.cross(v))
    else:
        area = tri.area
    if tri.area()<=min_area:
        return tri
    else:
        ns = [tri.p1 + tri.u/2, tri.p1 + tri.v/2, tri.p1 + (tri.u + tri.v)/2]
        cs = [(tri.p1+ns[0]+ns[1])/3, (tri.p2+ns[0]+ns[2])/3, (tri.p3+ns[0]+ns[2])/3, (ns[0]+ns[1]+ns[2])/3]
        cs = [c + fractal_noise(scale,i,frac_dim)*tri.unit_normal for c in cs]
        ps=[[ns[0],ns[1],tri.p1],
            [ns[1],ns[2],tri.p3],
            [ns[0],ns[2],tri.p2],
            [ns[0],ns[1],ns[2]]]
        triangles=[]
        for i in range(4):
            for j in range(3):
                triangles.append([cs[i], ps[i][j%3], ps[i][(j+1)%3]])

    for t in triangles:
        print(t)

    return



'''test_tri=Triangle(ORIGIN(),FSTEP(),LSTEP())
noisy_triangle(test_tri,0.1,2,1.3)'''



def fractalize_mesh(mesh,n,noise,*args,**kwargs):
    if mesh.__class__ is not Mesh:
        raise TypeError("roughen can only be applied to mesh objects.")
    if "filter" in kwargs:
        pass # Do a test then if passes, apply to the face

# TESTING

tyre_by_param(0.5, 0.3, 0.2, 2, 30, 0, 30)






