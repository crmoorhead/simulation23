from project_utils import *
from project_geometry import *
from toolbox import *
from analysis_toolbox import *

medium = SALT_WATER_1200
device = SV1010_1200Hz

# OBJECTS
# Change object parameters here. Maybe draw them to visualise.
test_tet = Tetrahedron([0, 4, 2], [-2, 4, -1], [2, 4, -1], [0, 3, -1])
test_tri = Triangle([0, 3, -1], [2, 3, -1], [0, 3, 2])
corner_tri = Triangle([0, 1 ,0], [0, 1, 1], [1, 1, 0])

# BACKGROUNDS
# You can use the following backgrounds already imported to this script (see project_geometry.py):
#
# Planes: FLAT_PLANE, FACING_WALL, SLOPING_WALL
# Composites: GAUSSIAN_HILL_X, GAUSSIAN_HOLE_X, ROLLING_BANK_X
# X can take on values 8, 32, 128, 512

# Create a scene with the object and background
scene_1 = Scene(background=FLAT_PLANE, objects=test_tet)

# Run the simulation and save the image. Use your own paths.
log_dir = r"C:\Users\r03cm18\OneDrive\2023\Uni_2023\Journal\04-28-2023\run_logs"
image_dir = r"C:\Users\r03cm18\OneDrive\2023\Uni_2023\Simulation\Images"

# Remove loop for a single run or set to 1.
for i in range(1):
    with log_terminal_output(log_dir) as log_file:
        print("Scene:", scene_1)
        print("Objects:", scene_1.objects)
        print("Device:", device)
        print("Medium:", medium)
        scan = Scan(A_scan(device, [0, 0, 0], -60, 0, 100, 0.1, "degs", scene=scene_1)
                    ,"scan", "degs", span=120)
        scan.full_scan(verbosity=2, save_dir=image_dir)