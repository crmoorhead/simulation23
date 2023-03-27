from sonar_formulae import Medium
from new_geometry_tools import *
from numpy import concatenate, full_like, NaN, isnan, asarray, full, multiply, expand_dims
from timeit import default_timer

class Scene():

    def __init__(self, **kwargs):

        if "background" in kwargs:
            self.background = kwargs["background"]
        else:
            self.background = None
        if "objects" in kwargs:
            if isinstance(kwargs["objects"], dict):
                self.labels = list(kwargs["objects"].keys())
                self.objects = list(kwargs["objects"].values())
            elif isinstance(kwargs["objects"], list):
                self.labels = ["object_"+str(i+1) for i in range(len(kwargs["objects"]))]
                self.objects = kwargs["objects"]
            else:
                self.labels = ["object_1"]
                self.objects = [kwargs["objects"]]
        else:
            self.labels = []
            self.objects = []

        if "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]
        else:
            self.accelerator = None

        if "name" in kwargs:
            self.name = kwargs["name"]
        else:
            self.name = "untitled_scene"

        self.object_count = len(self.objects)
        self.pov = None
        self.rays = None

        if self.background is None and self.object_count == 0:
            raise AttributeError("This is an empty scene. All images will be empty")

    def __str__(self):
        return "Scene(" + self.name +  ", " + str(self.object_count) + " objects in scene, background:{})".format(self.background.__str__())

    def __repr__(self):
        desc = ""
        desc += "SCENE:" + self.name + "\n"
        if self.object_count != 0:
            desc += "OBJECTS\n"
            for i in range(self.object_count):
                desc += str(self.labels[i]) + ": " + str(self.objects[i]) + "\n"
        if self.background is not None:
            desc += "BACKGROUND\n"
            desc += self.background.__str__()
        elif self.background is None and self.object_count == 0:
            desc = "SCENE IS EMPTY\n"

        return desc

    def info(self):
        print(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def intersection_params(self, rays, pov, **kwargs): # Possibly add accelerator here
        self.rays = rays
        self.pov = pov
        if self.accelerator is not None: # Placeholder for any actions relevant to the Accelerator eg. reordering
            pass
        ob_idxs = full(self.rays.shape[1:], -1)  # Rays all assigned as not hitting any object
        dists = full(self.rays.shape[1:], 128, dtype=float)  # Distances all set to 128 (much higher than
        for i in range(self.object_count):
            dist = self.objects[i].intersection_params(rays, pov)  # Calc distances until intersection for current object
            ob_idxs, dists = self.collect_min(ob_idxs, dists, dist, i)  # updates closest distances for each ray and retains object
        idx_dict = {}
        for f in range(-1, self.object_count): # -1 is for no objects being hit for that ray
            conditions = where(ob_idxs == f)  # Finds indices where rays hit current object
            idx_dict[f] = conditions

        self.idx_dict = idx_dict
        self.dists = dists

        self.process_background()  # Modifies above 3 scene properties and replaces misses with background hits

    def collect_min(self, indices, mins, new_array, index):
        indices = where(new_array < mins, index, indices) # Double calculation here. Optimise?
        mins = where(new_array < mins, new_array, mins)
        return indices, mins

    # Changes already generated ray and idx dicts. Note that assumption that background is not in front of any object.
    def process_background(self, pov):
        if self.background is not None:
            if not isinstance(self.background, Composite):
                rays = array([(self.rays[:, self.idx_dict[-1][0][c], self.idx_dict[-1][1][c]]) for c in range(len(self.idx_dict[-1][0]))])
                br_dists = squeeze(self.background.intersection_params(rays, pov))   # gets distances of rays that hit the
                hits = squeeze(asarray(where(~isnan(br_dists))))   # sort between those that actually hit the B/G
                self.idx_dict[-1] = tuple(self.idx_dict[-1][hits].T)   # gets 2D indices of hits
                self.dists[self.idx_dict[-1]] = br_dists[hits]   # Allocate distance of hits to 2D array of dists
                # This also needs applied to every object
            else:
                # similar to above but indices of components replace object indices
                pass

    # Map SL results onto shape
    # Maybe input is indices
    def scatter_loss(self):
        SL = full(self.rays.shape[1:], NaN)  # generate expected shape
        print("start checking objects")
        print("input_rays", self.rays.shape)
        start = default_timer()
        for o in range(self.object_count):
            if self.idx_dict[o] is not None:
                print("Calculating incidents for", self.labels[o])
                if issubclass(self.objects[o], Composite): # If it's a composite, the incidents must be calculated for subparts this will use cached component indices
                    incidents = self.objects[o].gen_incident(self.rays) # No index feed needed as this will be stored internally
                else:
                    incidents = self.objects[o].gen_incident(self.rays)
                SL[self.idx_dict[o]] = self.objects[o].scatterer.SL(incidents)  # Input should be incident angles. Composites should have single scatterer object
            else:
                print("no")
        print("finished checking objects:", default_timer()-start)
        if self.background is not None:
            bg_incidents = squeeze(self.background.gen_incident(self.rays, self.idx_dict[-1]))
            print(bg_incidents)
            SL[self.idx_dict[-1]] = self.background.scatterer.SL(bg_incidents)  # Input should be incident angles
        return SL

    # Methods for altering Scene properties
    def name_scene(self, name):
        self.scene_name =str(name)

    def add_object(self, object, **kwargs):
        if "object_name" in kwargs:
            object.name_object(str(kwargs["object_name"]))
            self.objects += object
            self.labels += str(kwargs["object_name"])
            self.object_count += 1
        elif object.name != None:
            self.labels += str(object.name)
            self.objects += object
            self.object_count += 1
        else:
            object_name="Scene_object_"+str(self.object_count+1)
            object.name_object(object_name)
            self.labels += object_name
            self.objects += object
            self.object_count += 1

    def add_objects(self, objects, **kwargs):
        if isinstance(objects, dict):
            for object in objects:
                self.add_object(objects[object], object_name=object)
        elif isinstance(objects, list):
            if "object_names" in kwargs:
                if len(objects) != len(kwargs["object_names"]):
                    raise ValueError("Number of objects and labels must match.")
                else:
                    self.objects += objects
                    self.labels += kwargs["object_names"]
            else:
                for object in objects:
                    self.add_object(object)    

    def del_object(self, object_name):
        obj_idx = self.labels[object_name]
        self.objects.pop(obj_idx)
        self.labels.pop(obj_idx)
        self.object_count -= 1

    def change_object_name(self, old_name, new_name):
        obj_idx = self.labels.index(old_name)
        self.labels[obj_idx] = new_name



