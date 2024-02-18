import MDAnalysis as mda
import numba as nb
import numpy as np
import xml.etree.ElementTree as et
import os
import re
import warnings


NEIGHBOR_DISTANCE = 1.15
MIN_BOUNDARY_NEIGHBOR = 1
TIMESTEP = 0.01
DCDNAME = "traj.dcd"

SQ_NEIGHBOR_DISTANCE = NEIGHBOR_DISTANCE**2


warnings.filterwarnings("ignore")


@nb.njit
def empty_int64_list():
    l = [nb.int64(10)]
    l.clear()
    return l


@nb.njit
def fold_back(xyz, box_size, half_box_size):
    for i in range(len(xyz)):
        for dim in range(3):
            xyz[i][dim] = (xyz[i][dim] + half_box_size[dim]) % box_size[dim]


@nb.njit
def apply_min_img(r, box_size, half_box_size):
    for dim in range(3):
        if r[dim] > half_box_size[dim]:
            r[dim] -= box_size[dim]
        elif r[dim] < -half_box_size[dim]:
            r[dim] += box_size[dim]


@nb.njit
def Find_Qualified_CH_ind(head, tail1, tail2):
    Qualified_CH_ind = empty_int64_list()

    for i in range(len(head)):
        if head[i][2] > tail1[i][2] and head[i][2] > tail2[i][2]:
            Qualified_CH_ind.append(i)

    return Qualified_CH_ind


@nb.njit
def FindBoundary(Q_Cxyz, Q_Hxyz):
    count_C_neighbor_H = np.zeros(Q_Cxyz.shape[0], dtype=nb.int64)
    boundary_C_ind = np.zeros(Q_Cxyz.shape[0], dtype=nb.int64)
    found = 0

    for i in range(len(Q_Cxyz)):
        for j in range(len(Q_Hxyz)):
            r = Q_Cxyz[i] - Q_Hxyz[j]
            apply_min_img(r, box_size, half_box_size)
            distance = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)

            if distance < NEIGHBOR_DISTANCE:
                count_C_neighbor_H[i] += 1

            if count_C_neighbor_H[i] >= MIN_BOUNDARY_NEIGHBOR:
                boundary_C_ind[found] = i
                found += 1
                break

    return_C_ind = boundary_C_ind[: found]
    return return_C_ind


@nb.njit
def Calc_Perimeter(Boundary_C):
    perimeter = 0
    num_Boundary_C = len(Boundary_C)

    for i in range(num_Boundary_C):
        count_C_neighbor_C = 0
        perimeter_i = 0

        for j in range(num_Boundary_C):
            r = Boundary_C[i] - Boundary_C[j]
            apply_min_img(r, box_size, half_box_size)
            sq_distance = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            if sq_distance < SQ_NEIGHBOR_DISTANCE:
                perimeter_i += np.sqrt(sq_distance)
                count_C_neighbor_C += 1

        perimeter += perimeter_i / count_C_neighbor_C

    return perimeter


currentdir = os.getcwd()
wdirs = [
    f for f in os.listdir(currentdir) if (os.path.isdir(f) and re.match(r"[sr]\d+", f))
]
wdirs.sort(key=lambda x: int(re.split(r"(\d+)", x)[1]))
if len(wdirs) == 0:
    raise Exception("Wrong working directory!")

numdir = 0
for wdir in wdirs:
    xmls = [
        xml
        for xml in os.listdir(os.path.join(currentdir, wdir))
        if re.match(r"cpt\.\d+\.xml", xml)
    ]
    xmls.sort(key=lambda x: int(re.split(r"(\d+)", x)[1]))
    xml = os.path.join(currentdir, wdir, xmls[-1])
    dcd = os.path.join(currentdir, wdir, DCDNAME)
    path_to_perimeter_data = os.path.join(currentdir, wdir, "perimeterByDistance.dat")

    tree = et.parse(xml)
    root = tree.getroot()
    box = root.find(".//box")
    lx = float(box.get("lx"))
    ly = float(box.get("ly"))
    lz = float(box.get("lz"))
    box_size = np.array([lx, ly, lz], dtype=float)
    half_box_size = np.array([0.5 * lx, 0.5 * ly, 0.5 * lz], dtype=float)

    U = mda.Universe(xml, dcd)
    C = U.select_atoms("type C")
    O = U.select_atoms("type O")
    O1 = U.select_atoms("type O1")
    H = U.select_atoms("type H")
    T = U.select_atoms("type T")
    T1 = U.select_atoms("type T1")
    t = 1000 * numdir

    with open(path_to_perimeter_data, "w") as f:
        f.write("t\tperimeter\n")

    for ts in U.trajectory[1:]:
        t += 1
        Cxyz = C.positions
        Oxyz = O.positions
        O1xyz = O1.positions
        Hxyz = H.positions
        Txyz = T.positions
        T1xyz = T1.positions
        fold_back(Cxyz, box_size, half_box_size)
        fold_back(Oxyz, box_size, half_box_size)
        fold_back(O1xyz, box_size, half_box_size)
        fold_back(Hxyz, box_size, half_box_size)
        fold_back(Txyz, box_size, half_box_size)
        fold_back(T1xyz, box_size, half_box_size)
        Qualified_C = np.array(Cxyz[Find_Qualified_CH_ind(Cxyz, Oxyz, O1xyz)])
        Qualified_H = np.array(Hxyz[Find_Qualified_CH_ind(Hxyz, Txyz, T1xyz)])
        Boundary_C = np.array(Cxyz[FindBoundary(Qualified_C, Qualified_H)])
        perimeter = Calc_Perimeter(Boundary_C)

        with open(path_to_perimeter_data, "a") as f:
            f.write(f"{t*TIMESTEP:.2f}\t{perimeter:.4f}\n")

    numdir += 1
