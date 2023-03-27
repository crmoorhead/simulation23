from scenes import Scene
from fundamentals import SVec, CVec
from test_scenes import *
from mesh_tools import *
from math import ceil
from numpy import floor,inf
from toolbox import uni_div, uni_min, uni_mult

class Accelerator:

    def __init__(self,scene):
        if scene.__class__ is not Scene:
            raise TypeError("Input must be a Scene object.")
        self.scene=scene
        self.object_count=0
        self.component_count=0
        self.simple_objects=[]
        self.simple_comp_count = 0
        self.regions={}
        for ob in self.scene.OBJECTS:
            self.object_count += 1 # COUNT THE OBJECT
            if self.scene.OBJECTS[ob].object_type == "Composite":
                if self.scene.OBJECTS[ob].object_subtype == "Tesselation":     # IF WE HAVE A TESSELATION
                    self.component_count += self.scene.OBJECTS[ob].component_count   # ADD TOTAL NUMBER OF COMPONENTS
                    # CREATE A REGION THAT HAS:
                    # Count of components in that region,
                    # the bounding box that contains it,
                    # ordered list of components,
                    # voxel array linking position to ordered list of components
                    self.regions[ob]= {"component_count": self.scene.OBJECTS[ob].component_count,
                                      "bounding_box":self.scene.OBJECTS[ob].bounding_box,
                                      "components": list(scene.OBJECTS[ob].COMPONENTS.values())}
                    (n, m) = self.scene.OBJECTS[ob].grid_dims
                    self.regions[ob]["voxel_shape"]=(n,m,1,self.regions[ob]["component_count"]+1)
                    self.regions[ob]["voxel_steps"]=[(self.regions[ob]["bounding_box"].x_max-self.regions[ob]["bounding_box"].x_min)/n,
                                                     (self.regions[ob]["bounding_box"].y_max-self.regions[ob]["bounding_box"].y_min)/m,
                                                     (self.regions[ob]["bounding_box"].z_max-self.regions[ob]["bounding_box"].z_min)]
                    self.regions[ob]["voxel_array"]=self.create_voxels(self.regions[ob])
                elif self.scene.OBJECTS[ob].object_subtype == "Mesh":
                    self.component_count += self.scene.OBJECTS[ob].component_count
                    self.regions[ob] = {"component_count": self.scene.OBJECTS[ob].component_count,
                                        "bounding_box": self.scene.OBJECTS[ob].bounding_box,
                                        "components":list(scene.OBJECTS[ob].COMPONENTS.values())}
                    n = ceil(self.regions[ob]["component_count"]**(1/3))
                    self.regions[ob]["voxel_shape"] = (n, n, n, self.regions[ob]["component_count"]+1)
                    self.regions[ob]["voxel_steps"]=[(self.regions[ob]["bounding_box"].x_max-self.regions[ob]["bounding_box"].x_min)/n,
                                                     (self.regions[ob]["bounding_box"].y_max-self.regions[ob]["bounding_box"].y_min)/n,
                                                     (self.regions[ob]["bounding_box"].z_max-self.regions[ob]["bounding_box"].z_min)/n]
                    self.regions[ob]["voxel_array"] = self.create_voxels(self.regions[ob])
                    # FOR MESH, WE ADD THE OUTER BOX TO THE SCENE OBJECTS TOO
                    self.simple_objects.append(self.scene.OBJECTS[ob])
                    self.simple_comp_count += 1
                elif self.scene.OBJECTS[ob].object_subtype in [Tetrahedron]:  # Adds each component of the composite to list of simple objects
                    for c in self.scene.OBJECTS[ob].COMPONENTS:
                        self.simple_objects.append(self.scene.OBJECTS[ob].COMPONENTS[c])
                    self.component_count += self.scene.OBJECTS[ob].component_count
                    self.simple_comp_count += self.scene.OBJECTS[ob].component_count
                else:
                    raise TypeError("Composite subtype not recognised.")
            else:
                self.component_count += 1
                self.simple_comp_count += 1
                self.simple_objects.append(self.scene.OBJECTS[ob])
        if self.simple_comp_count != 0:
            simple_bb = joint_AABB(self.simple_objects)
            self.regions["simple_scene"] = {"component_count": self.simple_comp_count,
                                            "bounding_box": simple_bb,"components": self.simple_objects}
            n = ceil(self.regions["simple_scene"]["component_count"] ** (1 / 3))
            self.regions["simple_scene"]["voxel_shape"] = (n, n, n, self.regions["simple_scene"]["component_count"]+1)
            self.regions["simple_scene"]["voxel_steps"] = [
                (self.regions["simple_scene"]["bounding_box"].x_max - self.regions["simple_scene"]["bounding_box"].x_min) / n,
                (self.regions["simple_scene"]["bounding_box"].y_max - self.regions["simple_scene"]["bounding_box"].y_min) / n,
                (self.regions["simple_scene"]["bounding_box"].z_max - self.regions["simple_scene"]["bounding_box"].z_min) / n]
            self.regions["simple_scene"]["voxel_array"] = self.create_voxels(self.regions["simple_scene"])
        else:
            self.regions["simple_scene"] = {}

    def create_voxels(self, region):
        voxel_shape=region["voxel_shape"]
        bb=region["bounding_box"]
        voxel_steps = region["voxel_steps"]
        voxel_array=zeros(voxel_shape)
        for i in range(voxel_shape[0]):
            for j in range(voxel_shape[1]):
                for k in range(voxel_shape[2]):
                    for o in range(voxel_shape[3]-1):
                        c_min = [bb.x_min + i*voxel_steps[0], bb.y_min + j*voxel_steps[1], bb.z_min + k*voxel_steps[2]]
                        c_max = [c_min[c] + voxel_steps[c] for c in range(3)]
                        ijk_box = [c_min, c_max]
                        if (region["components"][o].bounding_box).overlaps(ijk_box):
                            voxel_array[i][j][k][o] = 1
                    voxel_array[i][j][k][-1] = voxel_array[i][j][k].sum()
        return voxel_array

    def process_ray(self,ray,*args,**kwargs):
        if ray.__class__ is not Ray:
            raise TypeError("Input must be an instance of Ray.")
        # FIRST LEVEL (scene) - this is always last
        if len(self.regions)==1:
            pass
        if len(self.regions)==2:
            pass
        simple_test=self.region_test("simple_scene",ray)
        if simple_test != False:
            [in_point, t, ds] = simple_test
            entry_index = self.point_to_index("simple_scene",in_point,**kwargs)
            hit_object = self.trace_ray("simple_scene",ray,in_point,entry_index,ds,t)
            if hit_object is not False:                           # Test is Mesh, which requires a further region test,
                                                                  # otherwise process the ray to return value
                return # ray_return(hit_object,ray,t,*args,**kwargs)

    def region_test(self,r_name,ray):
        if self.regions[r_name]=={}:
            return False
        x_min, x_max = self.regions[r_name]["bounding_box"].x_min, self.regions[r_name]["bounding_box"].x_max
        y_min, y_max = self.regions[r_name]["bounding_box"].y_min, self.regions[r_name]["bounding_box"].y_max
        z_min, z_max = self.regions[r_name]["bounding_box"].z_min, self.regions[r_name]["bounding_box"].z_max
        [o_x, o_y, o_z],[d_x, d_y, d_z]=list(ray.origin.value),list(ray.direction.value)
        ds=ray.directions()

        if ds[1] == 1: # d_y is never negative by construction if exactly zero, then the ray is in the XZ plane
            # XZ CASE: We test at location of y_min
            t=(y_min-o_y)/d_y
            x_test = o_x + d_x*t
            z_test = o_z + d_z*t
            if x_min <= x_test:
                if x_max >= x_test:
                    if z_min <= z_test:  # X_test is true
                        if z_max >= z_test:
                            return [array([x_test,y_min,z_test]),t,ds]
            # YZ and -YZ CASES: We test at x_min or x_max if direction is negative
            if ds[0] != 0:
                t=([x_max,x_min][ds[0]-1]-o_x)/d_x
                y_test = o_y + d_y*t
                z_test = o_z + d_z*t
                if y_min <= y_test:
                    if y_max >= y_test:
                        if z_min <= z_test:
                            if z_max >= z_test:
                                return [array([[x_max,x_min][dir-1]],y_test,z_test),t,ds]
                # XY and -XY CASES: We test at z_min or z_max
                if ds[2] != 0:
                    t = ([z_max, z_min][ds[2] - 1] - o_z) / d_z
                    y_test=o_y + d_y*t
                    x_test=o_x + d_x*t
                    if y_min <= y_test:
                        if y_max >= y_test:
                            if x_min <= x_test:
                                if x_max >= x_test:
                                    return [array([x_test,y_test,[z_max,z_min][ds[2]]-1]),t,ds]
                else: # We are in the XY plane. Then we have either the XZ or YZ case, which have already been tested
                    pass
            # We are in the YZ plane. We have the YZ or XY case or the XZ case where we are looking straight forward.
            else: # XY case in YZ plane. We test at z_min or z_max
                if x_min <= o_x: # x_component must be in correct plane
                    if x_max >= o_x:
                        if d_z == 0:  # Looking straight forward
                            if y_min <= o_y:
                                if y_max >= o_y:
                                    return [array([o_x,y_min,o_z]),y_min-o_y,ds]

                        else:
                            t = ([z_max, z_min][ds[2]- 1] - o_z) / d_z
                            y_test = o_y + d_y * t
                            if y_min <= y_test:
                                if y_max >= y_test:
                                    return [array([o_x, y_test, [z_max, z_min][ds[2]]]),t,ds]
        else:
            # We are in the XZ plane. We have the XY or YZ CASE
            if o_y <= y_max: # y_component must be correct
                if o_y >= y_min:
                    if d_z != 0:
                        if d_x !=0:
                            # YZ CASE AND -YZ CASE. We test at x_min or x_max
                            t = ([x_max,x_min][ds[0]-1] - o_x)/d_x
                            z_test = o_z + d_z*t
                            if z_min <= z_test:
                                if z_max >= z_test:
                                    return [array([[x_max,x_min][ds[0]-1],o_y,z_test]),t,ds]
                            # -XY CASE AND XY CASE. We test at z_min or z_max.
                            t = ([z_max,z_min][ds[2]-1] - o_z)/d_z
                            x_test = o_x + d_x*t
                            if x_min <= x_test:
                                if x_max >= x_test:
                                    return [array([x_test, o_y, [z_max,z_min][ds[2]-1]]),t,ds]

                        else: # We are looking straight up or down. If d_x = d_y = 0, d_z != 0.
                            if x_min <= o_x:
                                if x_max >= o_x:
                                    t=([z_max, z_min][ds[2] - 1]-o_z)*ds[2]
                                    return [array([o_x, o_y, [z_max, z_min][ds[2] - 1]]),t,ds]
                    else: # We are looking straight left or right. If d_y = d_z = 0, d_y != 0.
                        if z_min <= o_z:
                            if z_max >= o_z:
                                t = ([x_max, x_min][ds[0] - 1] - o_x) * dir
                                return [array([[x_max, x_min][ds[0] - 1], o_y, o_z]),t,ds]

        return False

    def point_to_index(self,region, point,**kwargs):
        voxel_steps=array(self.regions[region]["voxel_steps"])
        c_1=array([self.regions[region]["bounding_box"].x_min,self.regions[region]["bounding_box"].y_min,self.regions[region]["bounding_box"].z_min])
        return floor((point-c_1)/voxel_steps)

    # assume that point is an array of the correct shape and it is indeed in the region

    def trace_ray(self,region,ray,entry_point,entry_index,dirs,t,**kwargs):
        stop=False
        current_index = entry_index.copy()
        current_t=t.copy()
        # get objects in voxel
        # test all objects and return min
        while stop == False:

            ray_return,current_index,current_t,stop = self.next_voxel(region,ray,entry_point,current_index,dirs,current_t,**kwargs)
        return ray_return

    def hit_objects(self,current_index):
        stop = False
        # extract objects based on index
        # test hit for eack object in region. If mesh, we need to localise further
        # return the closest object and t value etc
        # return the ray response for that object only. If object hit, we change stop to True
        ray_return = 0
        return ray_return, stop

    def next_voxel(self,region,ray,entry_point,entry_index,dirs,t,**kwargs):
        stop = False
        if "thresh" in kwargs:
            if t > kwargs["thresh"]:
                return None
        mins=[[self.regions[region]["bounding_box"].x_min,self.regions[region]["bounding_box"].y_min,self.regions[region]["bounding_box"].z_min][i]
              +self.regions[region]["voxel_steps"][i]*entry_index[i] for i in range(3)]
        maxes=[mins[i]+self.regions[region]["voxel_steps"][i] for i in range(3)]
        ts=[uni_div(([maxes,mins][int(dirs[i]-1)//2][i]-entry_point[i]),ray.direction.value[i]) for i in range(3)]
        t += min(ts)
        new_point=ray.origin+t*ray.direction
        entry_index[ts.index(min(ts))] += 1
        if True in [entry_index[i]<0 or entry_index[i]+1>self.regions[region]["voxel_shape"][i] for i in range(3)]:
            print("Ray has left the region.")
            stop = True
        else:
            print("continue to next voxel.")
        return 0,entry_index,t,stop

    def process_ray_array(self,rays):
        pass



# Testing
# Scenes
test_spheres = RAND_SPHERES_SCENE()
test_tess = TESSELATION_SCENE()
test_surf = SURFACE_SCENE()
test_mesh = MESH_SCENE()

# Test Rays

r_1=LOOKF()
r_2=LOOKF().interpolate(LOOKL())
r_3=LOOKF().interpolate(LOOKR())
r_4=LOOKF().interpolate(LOOKL(),ratio=0.25)
r_5=LOOKF().interpolate(LOOKR(),ratio=0.25)

rays=[r_1,r_2,r_3,r_4,r_5]

scenes = [test_spheres, test_tess, test_surf, test_mesh]

# Accelerators

for s in scenes:
    acc=Accelerator(s)
    for r in [r_1,r_2,r_3]:
        print("Ray output:",acc.process_ray(r))
    print()
