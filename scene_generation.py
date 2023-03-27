from random import random
from scenes import Scene
from geometry_tools import Sphere, Cylinder, Plane, Wall, Rod, recognised_objects
from fundamentals import SVec, Rotation, CVec
from numpy import array
from material_props import dist_col_gen, hit_only_gen
from constants import *
from cv2 import imwrite, COLOR_BGR2RGB, cvtColor
from imageio import mimsave

def random_spheres(x_range,y_range,z_range,size_range,n,*args, **kwargs):
    for v in x_range, y_range, z_range, size_range:
        if v.__class__ not in [list,tuple]:
            raise TypeError("Check all ranges are integers, floats or lists/tuples of two items.")
        elif len(v)!=2:
            raise TypeError("Check all ranges are integers, floats or lists/tuples of two items.")
        else:
            pass

    if n.__class__ is not int:
        raise TypeError("Number of Objects must be an integer.")

    spheres=[Sphere(SVec(array([v[0]+random()*(v[1]-v[0]) for v in [x_range, y_range, z_range]])),size_range[0]+random()*(size_range[1]-size_range[0])) for s in range(n)]
    if "dist_range" in kwargs:
        for s in spheres:
            s.change_scatterer(dist_col_gen(CVec("random"),kwargs["dist_range"]))
    else:
        for s in spheres:
            s.change_scatterer(hit_only_gen(CVec("random")))
    return spheres

def create_scene(distance,dims,resolution,objects):
    scene=Scene(distance,dims,resolution)
    for o in objects:
        scene.add_object(o)
    scene.add_source(ORIGIN_SOURCE())
    return scene

def create_default_scene(objects,*args):
    return create_scene(80,(40,80),5,objects,*args)

def moving_object(ob,vector,empty_scene,frames,*args,**kwargs):
    if frames.__class__ is not int:
        raise TypeError("Number of frames must be an integer.")
    if vector.__class__ is not SVec:
        raise TypeError("Second argument must be an SVec instance.")
    if ob.__class__ not in recognised_objects:
        raise TypeError("Object is not recognised. Not a valid object class.")
    if empty_scene.__class__ is not Scene:
        raise TypeError("Empty scene must be a Scene object.")
    elif empty_scene.object_counter!=0:
        raise ValueError("Scene must be empty.")
    # Scene must also have source assigned
    step=vector/(frames-1)
    empty_scene.add_object(ob)
    for i in range(frames):
        empty_scene.ray_return_all()
        empty_scene.OBJECTS[ob.name].shift(step)
        print("Frame {} computed.".format(i+1))
    if "save_frames" in args:
        if "filename" in kwargs:
            name_stem=kwargs["filename"]+"_"
        else:
            name_stem="frame_"
        for i in range(len(empty_scene.images)):
            j=empty_scene.images[i]*256
            imwrite(name_stem+str(i+1)+".png",j)
    if "animate" in args:
        if "animation_name" in kwargs:
            anim_name=kwargs["animation_name"]
        else:
            anim_name="animation.gif"
        mimsave(anim_name+".gif", empty_scene.images)
    return empty_scene

def y_rotating_object(ob,empty_scene,frames,*args,**kwargs):
    if frames.__class__ is not int:
        raise TypeError("Number of frames must be an integer.")
    if ob.__class__ not in recognised_objects:
        raise TypeError("Object is not recognised. Not a valid object class.")
    if empty_scene.__class__ is not Scene:
        raise TypeError("Empty scene must be a Scene object.")
    elif empty_scene.object_counter!=0:
        raise ValueError("Scene must be empty.")
    # Scene must also have source assigned
    rotation=Rotation(0,360/(frames-1),0)
    empty_scene.add_object(ob)
    for i in range(frames):
        empty_scene.ray_return_all()
        empty_scene.rotate_scene(rotation)
        print("Frame {} computed.".format(i+1))
    if "save_frames" in args:
        if "filename" in kwargs:
            name_stem=kwargs["filename"]+"_"
        else:
            name_stem="frame_"
        for i in range(len(empty_scene.images)):
            j=empty_scene.images[i]*256
            imwrite(name_stem+str(i+1)+".png",j)
    if "animate" in args:
        if "animation_name" in kwargs:
            anim_name=kwargs["animation_name"]
        else:
            anim_name="animation.gif"
        mimsave(anim_name+".gif", empty_scene.images)
    return empty_scene

def z_rotating_object(ob,empty_scene,frames,*args,**kwargs):
    if frames.__class__ is not int:
        raise TypeError("Number of frames must be an integer.")
    if ob.__class__ not in recognised_objects:
        raise TypeError("Object is not recognised. Not a valid object class.")
    if empty_scene.__class__ is not Scene:
        raise TypeError("Empty scene must be a Scene object.")
    elif empty_scene.object_counter!=0:
        raise ValueError("Scene must be empty.")
    # Scene must also have source assigned
    rotation=Rotation(0,0,360/(frames-1))
    empty_scene.add_object(ob)
    for i in range(frames):
        empty_scene.ray_return_all()
        empty_scene.rotate_scene(rotation)
        print("Frame {} computed.".format(i+1))
    if "save_frames" in args:
        if "filename" in kwargs:
            name_stem=kwargs["filename"]+"_"
        else:
            name_stem="frame_"
        for i in range(len(empty_scene.images)):
            j=empty_scene.images[i]*256
            imwrite(name_stem+str(i+1)+".png",j)
    if "animate" in args:
        if "animation_name" in kwargs:
            anim_name=kwargs["animation_name"]
        else:
            anim_name="animation.gif"
        mimsave(anim_name+".gif", empty_scene.images)
    return empty_scene
