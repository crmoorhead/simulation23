from project_utils import *
from project_geometry import *
from argparse import ArgumentParser

medium = SALT_WATER_1200
device = SV1010_1200Hz
test_tet = Tetrahedron([0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0])
scene_1 = Scene(background=FLAT_PLANE, objects=test_tet)
scan = Scan(A_scan(device, [0, 0, 0], -60, 0, 100, 0.1, "degs", scene=scene_1)
            , "scan", "degs", span=120)
scan.full_scan("show", verbosity=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("s", type=str, required=True)
    parser.add_argument("--freq", type=int, required=False, default=1200)
    parser.add_argument("--medium", type=str, required=False, default="salt")
    args = parser.parse_args()

    print("running")
