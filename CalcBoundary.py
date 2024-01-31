#!/opt/homebrew/bin/
import MDAnalysis as mda
import numba as nb
import numpy as np
from scipy.spatial import ConvexHull
import os
import re
import sys
import warnings


DISTANCE_THRESHOLD = 1.15
TIMESTEP = 0.01
LX = 80
LY = 80
LZ = 40
DCDNAME = "traj.dcd"

xml = sys.argv[1]
dcd = sys.argv[2]
HALFLX = LX / 2
HALFLY = LY / 2
HALFLZ = LZ / 2


warnings.filterwarnings("ignore")


@nb.njit
def empty_int64_list():
    l = [nb.int64(10)]
    l.clear()
    return l


@nb.njit
def fold_back(xyz):
    for i in range(len(xyz)):
        xyz[i][0] = (xyz[i][0] + HALFLX) % LX - HALFLX
        xyz[i][1] = (xyz[i][1] + HALFLY) % LY - HALFLY
        xyz[i][2] = (xyz[i][2] + HALFLZ) % LZ - HALFLZ


@nb.njit
def find_qualified_CH_ind(head, tail1, tail2):
    Qualified_CH_ind = empty_int64_list()

    for i in range(len(head)):
        if head[i][2] > tail1[i][2] and head[i][2] > tail2[i][2]:
            Qualified_CH_ind.append(i)

    return Qualified_CH_ind


@nb.njit
def group_points_into_convex_polygons(points):
    grouped_polygons = []

    assigned = np.zeros(len(points))

    for i in range(len(points)):
        if not assigned[i]:
            current_polygon = [points[i]]
            assigned[i] = True

            for j in range(len(points)):
                distancesq = (points[i][0] - points[j][0]) ** 2 + (
                    points[i][1] - points[j][1]
                ) ** 2

                if not assigned[j] and distancesq < DISTANCE_THRESHOLD**2:
                    current_polygon.append(points[j])
                    assigned[j] = True

            grouped_polygons.append(current_polygon)

    return grouped_polygons


@nb.njit
def calc_perimeter(qualified_Cxy, all_vertices):
    perimeter = 0.0

    for i in range(len(all_vertices)):
        j = (i + 1) % len(all_vertices)
        perimeter += np.sqrt(
            (qualified_Cxy[all_vertices[i]][0] - qualified_Cxy[all_vertices[j]][0]) ** 2
            + (qualified_Cxy[all_vertices[i]][1] - qualified_Cxy[all_vertices[j]][1])
            ** 2
        )

    return perimeter


U = mda.Universe(xml, dcd)
C = U.select_atoms("type C")
O = U.select_atoms("type O")
O1 = U.select_atoms("type O1")
t = 0

with open("perimeterByConvexHull.dat", "w") as f:
    f.write("t\tperimeter\n")

for ts in U.trajectory[1:]:
    t += 1
    total_perimeter = 0
    Cxyz = C.positions
    Oxyz = O.positions
    O1xyz = O1.positions
    fold_back(Cxyz)
    fold_back(Oxyz)
    fold_back(O1xyz)

    qualified_C = Cxyz[find_qualified_CH_ind(Cxyz, Oxyz, O1xyz)]
    qualified_Cxy = [i[:2] for i in qualified_C]
    clusters = group_points_into_convex_polygons(qualified_Cxy)

    for cluster in clusters:
        if len(cluster) > 3:
            hull = ConvexHull(cluster)
            vertices = hull.vertices
            total_perimeter += calc_perimeter(qualified_Cxy, vertices)

    with open("perimeterByConvexHull.dat", "a") as f:
        f.write(f"{t}\t{total_perimeter}\n")
