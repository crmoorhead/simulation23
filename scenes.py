from numpy import zeros, array, concatenate as concat, clip
from fundamentals import ColCanvas, SimpleField, Ray, SVec, CVec, GVec, GSCanvas, Rotation, Angle, Noise
from toolbox import show_image, stitch_images
from geometry_tools import tangent_plane, angle_with_plane, recognised_objects
from cv2 import transpose as transimage, imwrite
from numba import jit, prange
from tqdm import tqdm

'''
MUST CONTAIN

* List of objects present
* Background class - defining default content of pixels if there are no interactions.
* Limits of view
* Camera position and orientation (from Field object)
* Other configurations under CONFIG dictionary

To create a scene, we first create it empty, then add objects or lists of objects.

When adding objects, we may want to check there are no overlaps in space, or designate objects with 
"intersections allowed" property.

'''
# DIRECTIVITY CLASS

class Directivity:

    def __init__(self,strength_funct):
        pass

# DYNAMIC THRESHOLDING VIA SOURCE AND DIRECTIVITY

# SOURCE CLASS

class Source:

    def __init__(self,origin,strength,*args,**kwargs):
        if origin.__class__ is not SVec:
            raise TypeError("Source origin must be an instance of SVec.")
        if "override_max" not in args:
            if strength <0 or strength>1:
                raise AttributeError("Strength must be an integer between 0 and 1.")
        else:
            if strength.__class__ not in [int,float]:
                raise TypeError("Strength must be an integer or float.")
        self.origin=origin
        self.strength=strength
        if "name" in kwargs:
            self.name=str(kwargs["name"])
        else:
            self.name=None

    def __str__(self):
        if self.name is not None:
            return "Source(Origin:"+str(self.origin.value)+", Strength:"+str(self.strength)+", Name:"+self.name+")"
        return "Source(Origin:"+str(self.origin.value)+", Strength:"+str(self.strength)+")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # Source methods

    def shift(self,other):
        if other._class__ is not SVec:
            raise TypeError("Source origin must be an instance of SVec.")
        else:
            self.origin+=other

    def name_source(self,other):
        self.name=str(other)

# SCENE CLASS

class Scene:

    def __init__(self,distance,screen_dimension,resolution,*args,**kwargs):   # Will add direction when we need to apply general Field class
        dims=(screen_dimension[1],screen_dimension[0])
        self.OBJECTS={}
        self.viewpoint=SVec(array([0,0,0]))                            # Viewpoint needs to be an input for general field generator
        self.field=SimpleField(distance,dims,resolution,*args,**kwargs)   # Generates field object of correct dimensions
        self.pixel_field=self.field.pixel_field                        # Field of pixel locations in 3D space
        self.ray_field=self.field.ray_field                            # Field of rays from viewpoint to each pixel centre. The values are the direction vectors only
        self.focus=self.field.anchor                                # Centre of viewer screen
        self.image_size=self.field.image_size                          # Dimension of image in pixels
        self.resolution=self.field.resolution                          # Resolution of image (density of pixels per unit distance)
        self.images=[]                                                 # Initially, no images are generated from the scene
        self.object_counter=0
        self.pixel_size=self.ray_field[0][1][1]-self.ray_field[0][0][0]
        if "scene_name" in kwargs:
            self.scene_name=str(kwargs["scene_name"])
        else:
            self.scene_name=None
        self.max_dist=20
        self.min_param=0.001
        self.screen_limit=10
        if "greyscale" not in args:
            self.CONFIG={"background":CVec(array([0,0,0])), "image_size": self.image_size,"centre": self.focus.value,
                         "resolution": self.resolution, "max_dist": self.max_dist, "min_inter_param": self.min_param,
                         "nearest_screen": self.screen_limit,"sources":[],"viewpoint":self.viewpoint}
        else:
            self.CONFIG={"background": GVec(0), "image_size": self.image_size, "centre": self.focus.value,
             "resolution": self.resolution, "max_dist": self.max_dist, "min_inter_param": self.min_param,
             "nearest_screen": self.screen_limit, "sources": [], "viewpoint": self.viewpoint}

        # OPTIONAL CONFIG SETTINGS
        if "camera_rotation" in kwargs:
            if kwargs["camera_rotation"].__class__ is not Rotation:
                raise TypeError("Rotation value must be a Rotation instance.")
            self.CONFIG["camera_rotation"]=kwargs["camera_rotation"]
        else:
            self.CONFIG["camera_rotation"] = None
        if "image_noise" in kwargs:
            if kwargs["image_noise"].__class__ is not Noise:
                raise TypeError("Image noise must be supplied with an instance of the Noise class.")
                self.image_noise=None
            else:
                self.image_noise=kwargs["image_noise"]
                self.CONFIG["image_noise"]=self.image_noise
        else:
            self.image_noise = None
        if "ray_noise" in kwargs:
            if kwargs["ray_noise"].__class__ is not Noise:
                raise TypeError("Ray noise must be supplied with an instance of the Noise class.")
                self.ray_noise=None
            else:
                self.ray_noise=kwargs["ray_noise"]
                self.CONFIG["ray_noise"] = self.ray_noise
        else:
            self.ray_noise = None
        if "canvas" in kwargs:
            self.CONFIG["background"]=kwargs["canvas"]
        if self.CONFIG["background"].__class__ is CVec:
            self.background=ColCanvas(self.image_size, color=self.CONFIG["background"])
        elif self.CONFIG["background"].__class__ is GVec:
            self.background = GSCanvas(self.image_size, color=self.CONFIG["background"])
        elif self.CONFIG["background"].__class__==list:
            if "greyscale" not in args:
                self.background=ColCanvas(self.image_size)
                self.background.grad_easel(self.CONFIG["background"],*args)
            else:
                self.background = GSCanvas(self.image_size)
                self.background.grad_easel(self.CONFIG["background"],*args)
        else:
            raise TypeError("Canvas keyword must be a colour, shade or list of 2 or 4 colours or shades.")

        self.image_grid=None

    def __str__(self):
        if self.scene_name!=None:
            return "Scene(" + self.scene_name +", Focus:" + str(self.focus.value) + ", " + str(self.object_counter) + " objects in scene)"
        else:
            return "Scene(Focus:" + str(self.focus.value) + ", " + str(self.object_counter) + " objects in scene)"

    def __repr__(self):
        self.__str__()
        if self.scene_name==None:
            print("SCENE")
        else:
            print("SCENE:",self.scene_name)
        print()
        if len(self.OBJECTS)!=0:
            print("OBJECTS")
            for k,v in self.OBJECTS.items():
                print(k,":",v,"\n")
        else:
            print("SCENE IS EMPTY\n")
        print("CONFIG ENTRIES")
        for k,v in self.CONFIG.items():
            print(k,":",v)
        print()

    def info(self):
        self.__repr__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # Methods for scene properties

    def name_scene(self,name):
        self.scene_name=str(name)

    def change_resolution(self,new_res,*args):
        self.field.newres(new_res)
        self.pixel_field = self.field.pixel_field
        self.image_size=self.field.image_size
        self.ray_field = self.field.ray_field
        self.resolution=new_res
        self.pixel_size = self.ray_field[0, 1, 1] - self.ray_field[0, 0, 0]
        if self.CONFIG["background"].__class__ is CVec:
            self.background=ColCanvas(self.image_size, color=self.CONFIG["background"])
        elif self.CONFIG["background"].__class__ is GVec:
            self.background = GSCanvas(self.image_size, color=self.CONFIG["background"])
        else:
            if "greyscale" not in args:
                self.background=ColCanvas(self.image_size)
                self.background.grad_easel(self.CONFIG["background"])
            else:
                self.background = GSCanvas(self.image_size)
                self.background.grad_easel(self.CONFIG["background"])

    def change_focus(self,new_focus):
        self.field.shift(new_focus-self.field.anchor)
        self.pixel_field = self.field.pixel_field
        self.ray_field = self.field.ray_field
        self.focus = self.field.anchor
        self.pixel_size = self.ray_field[0, 1, 1] - self.ray_field[0, 0, 0]
        # NEEDS TO UPDATE CONFIG TOO

    # This is not functional until general field implemented

    def change_viewpoint(self,new_view):
        if self.viewpoint.__class__ is not SVec:
            raise TypeError("New location must be an instance of SVec.")
        self.viewpoint=new_view
        self.CONFIG["viewpoint"]=self.viewpoint
        # Generate new general field
        self.pixel_field = self.field.pixel_field
        self.ray_field = self.field.ray_field
        self.pixel_size = self.ray_field[0, 1, 1] - self.ray_field[0, 0, 0]

    def change_size(self,factor):
        self.field.rescale(factor)
        self.pixel_field = self.field.pixel_field
        self.ray_field = self.field.ray_field
        self.pixel_size = self.ray_field[0, 1, 1] - self.ray_field[0, 0, 0]

    # The background must be assigned the same size as the screen.

    def assign_background(self,background,*args,**kwargs):
        if background.__class__ is CVec:
            if "greyscale" in args:
                raise TypeError("This scene is greyscale. A colour background cannot be applied")
            self.CONFIG["background"]=background
            self.background.change_color(background)
        elif background.__class__==GVec:
            if "greyscale" not in args:
                raise TypeError("This scene is not greyscale. Background must be a CVec instance.")
            self.CONFIG["background"]=background
            self.background.change_shade(background)
        elif background.__class__==list:
            self.CONFIG["background"]=background
            self.background.grad_easel(background,*args,**kwargs)
        else:
            raise TypeError("Background type must be a CVec or list object.")

    def show_background(self):
        self.background.show()

    def add_source(self,source):
        if source.__class__ is not Source:
            raise TypeError("Source must be an instance of the Source class.")
        else:
            self.CONFIG["sources"].append(source)

    def remove_source(self,source):
        del self.CONFIG["sources"]

    # Methods for modifying sccene objects

    def rotate_object(self,object,rotation):
        if object.__class__ is not str:
            ValueError("Object variable must be a name.")
            return NotImplemented
        elif object not in self.OBJECTS:
            ValueError("Named object not found.")
            return NotImplemented
        else:
            if rotation.__class__ is not Rotation:
                raise TypeError("Rotation variable must be an instance of Rotation class.")
            else:
                self.OBJECTS[object].rotate_object(rotation)

    def rotate_scene(self,rotation):
        if rotation.__class__ is not Rotation:
            raise TypeError("Rotation variable must be an instance of Rotation class.")
        self.ray_field=rotation.rotate_field(self.ray_field)
        self.CONFIG["camera_rotation"] += rotation

    def add_object(self,object,**kwargs):
        if object.__class__ not in recognised_objects:
            raise TypeError("Object is not a recognised type.")
        else:
            if "object_name" in kwargs:
                self.OBJECTS[str(kwargs["object_name"])]=object
                object.name_object(str(kwargs["object_name"]))
                self.object_counter+=1
            elif object.name != None:
                self.OBJECTS[str(object.name)] = object
                self.object_counter += 1
            else:
                object_name="Scene_object_"+str(self.object_counter+1)
                object.name_object(object_name)
                self.OBJECTS[object_name] = object
                self.object_counter += 1

    def add_objects(self,object_list,**kwargs):
        for object in object_list:
            self.add_object(object)

    def del_object(self,object_name):
        del self.OBJECTS[object_name]
        self.object_counter -= 1

    def change_object_name(self, old_name, new_name):
        self.OBJECTS[old_name].name=new_name
        self.OBJECTS[new_name]=self.OBJECTS[old_name]
        del self.OBJECTS[old_name]

    # Methods for generating images from scene

    def accelerator_filter(self):
        pass

    def ray_trace_all(self,ray,*args,**kwargs):
        count=0
        intersections={}
        for ob in self.OBJECTS:
            if self.OBJECTS[ob].object_type is "Composite":
                param=[]
                for c in self.OBJECTS[ob].COMPONENTS:
                    c_param = self.OBJECTS[ob].COMPONENTS[c].intersect_param(ray, *args, **kwargs)
                    if c_param != []:
                        param += c_param
            else:
                param = self.OBJECTS[ob].intersect_param(ray, *args, **kwargs)
            intersections[self.OBJECTS[ob].name]=param
            if param != [] and ("all" in args or "all_objects" in args):
                count+=len(param)
        if "all_objects" not in args:
            intersections={k: sorted(v) for k, v in intersections.items() if v != []}
        if "all" in args or "all_objects" in args:
            intersections["count"]=count
        return intersections

    def ray_trace_closest(self,ray,*args,**kwargs):
        all_obs_all_int=self.ray_trace_all(ray,*args,**kwargs)
        if all_obs_all_int == {}:
            return []
        utmost_min = min({k: all_obs_all_int[k][0] for k, v in all_obs_all_int.items()})
        param_dict={"object_name":utmost_min, "ray_param":all_obs_all_int[utmost_min][0]}
        param_dict["point"]=ray.origin+ray.direction*param_dict["ray_param"]
        return param_dict

    def ray_return(self,ray,*args,**kwargs):   # scatterer needs to have incident and return angle defined.
        closest=self.ray_trace_closest(ray,*args,**kwargs)
        if closest == [] or closest is None:
            return None
        point=closest["point"]
        inter_ob=self.OBJECTS[closest["object_name"]]
        param=closest["ray_param"]   # may be inputs to some scatterers
        scatterer=inter_ob.PROPERTIES["scatterer"]
        normal=inter_ob.normal(point) # may be inputs to some scatterers
        if inter_ob.PROPERTIES["rotation"] is not None:
            normal=inter_ob.PROPERTIES["rotation"].rotate_point(normal)
        if self.CONFIG["camera_rotation"] is not None:
            normal=self.CONFIG["camera_rotation"].rotate_point(normal)
        tang_plane=tangent_plane(point,inter_ob)
        return_angle=angle_with_plane(ray,tang_plane)
        if len(self.CONFIG["sources"])==1:
            source_origin=self.CONFIG["sources"][0].origin
            source_ray=Ray(source_origin,point-source_origin)
            incident=angle_with_plane(source_ray,tang_plane)
            return_val=scatterer.scatter_funct(incident,return_angle,param=param,normal=normal)
        else:
            for source in self.CONFIG["sources"]:
                source_origin = source[0].origin
                source_ray = Ray(source_origin, point - source_origin)
                incident = angle_with_plane(source_ray, tang_plane)
                scatterer.add_source(incident)
            return_val=scatterer.instantiated(return_angle)
        return return_val

    def ray_return_all(self,*args,**kwargs):
        dirs=self.ray_field
        if "rotation" in kwargs:
            dirs=kwargs["rotation"].rotate_field(dir)
        dims=self.ray_field.shape
        image=zeros(self.background.image.shape)
        if image.shape[2]==3:
            for i in tqdm(prange(dims[2])):
                for j in prange(dims[1]):
                    kwargs={**kwargs,"min_param":self.min_param}
                    return_ij=self.ray_return(Ray(self.viewpoint,SVec(dirs[:,j,i])),*args,**kwargs)
                    if return_ij is None:
                        image[i,j,:]=self.background.image[i,j,:]
                    else:
                        image[i, j, :]=[return_ij.value[2]/256,return_ij.value[1]/256,return_ij.value[0]/256]

        else:
            for i in tqdm(prange(dims[2])):
                for j in prange(dims[1]):
                    kwargs={**kwargs,"min_param":self.min_param}
                    return_ij=self.ray_return(Ray(self.viewpoint,SVec(dirs[:,j,i])),*args,**kwargs)
                    if return_ij is None:
                        image[i,j,:]=self.background.image[i,j,:]
                    else:
                        image[i, j,:]=[return_ij.value]
        image=transimage(image)
        if self.image_noise is not None:
            image=clip(self.image_noise.apply(image),0,1)
        self.images.append(image)
        return image

    def show_image(self):
        if len(self.images)==0:
            self.show_background()
        elif len(self.images)==1:
            show_image(self.images[0])
        else:
            self.image_grid=stitch_images(self.images)
            show_image(self.image_grid)

    def save_images(self,filename):
        if len(self.images)==0:
            print("No images for scene generated.")
        elif len(self.images)==1:
            imwrite(filename,self.images[0]*256)
        else:
            if self.image_grid is None:
                self.image_grid=stitch_images(self.images)
            imwrite(filename,self.image_grid*256)

    def randomise_rays(self,*args,**kwargs):
        self.field.randomise_rays(*args,**kwargs)
        self.pixel_field=self.field.pixel_field
        self.ray_field=self.field.ray_field

    def apply_image_noise(self,other,*args,**kwargs):
        if other.__class__ is not Noise:
            raise TypeError("Noise must be an instance of a Noise object.")
        else:
            self.image_noise=other
            self.CONFIG["image_noise"]=self.image_noise
            if self.images == []:
                return self
            else:
                args = ()
                if "apply_index" in kwargs:
                    if kwargs["apply_index"].__class__ is int:
                        indices = [kwargs["apply_index"]]
                    elif kwargs["apply_index"].__class__ is not list:
                        raise TypeError("Apply index value must be an integer or list.")
                    else:
                        indices = kwargs["apply_index"]
                    del kwargs["apply_index"]
                else:
                    indices = range(len(self.images))
                for i in indices:
                    self.images[i] = clip(self.images[i] + other.noise_funct(self.images[i].shape, *args, **kwargs), 0,1)
                return self

    def apply_ray_noise(self,other):
        if other.__class__ is not Noise:
            raise TypeError("Noise must be an instance of a Noise object.")
        self.ray_noise = other
        self.CONFIG["ray_noise"] = self.ray_noise
        self.field.pixel_field = self.field.pixel_field + other.noise_funct(self.field.pixel_field.shape)
        self.field.ray_field = self.field.gen_ray_field()
        self.pixel_field = self.field.pixel_field
        self.ray_field = self.field.ray_field
        return self

# ACCELERATOR CLASS


# READY MADE SCENES


