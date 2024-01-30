#!/opt/homebrew/bin/
import MDAnalysis as mda
import numba as nb
import numpy as np
import pandas as pd
import os
import re
import warnings
from matplotlib import pyplot as plt


NEIGHBOR_DISTANCE = 1.15
MIN_BOUNDARY_NEIGHBOR = 1
TIMESTEP = 0.01
LX = 80
LY = 80
LZ = 40
DCDNAME = "traj.dcd"

HALFLX = LX / 2
HALFLY = LY / 2
HALFLZ = LZ / 2
SQ_NEIGHBOR_DISTANCE = NEIGHBOR_DISTANCE**2


warnings.filterwarnings("ignore")


@nb.njit
def empty_int64_list():
    l = [nb.int64(10)]
    l.clear()
    return l


@nb.njit
def fold_back(xyz):
    for i in range(0, len(xyz)):
        xyz[i][0] = (xyz[i][0] + HALFLX) % LX
        xyz[i][1] = (xyz[i][1] + HALFLY) % LY
        xyz[i][2] = (xyz[i][2] + HALFLZ) % LZ


@nb.njit
def Find_Qualified_CH_ind(head, tail1, tail2):
    Qualified_CH_ind = empty_int64_list()

    for i in range(len(head)):
        if head[i][2] > tail1[i][2] and head[i][2] > tail2[i][2]:
            Qualified_CH_ind.append(i)

    return Qualified_CH_ind


@nb.njit
def FindBoundary(Q_Cxyz, Q_Hxyz):
    count_C_neighbor_H = np.zeros(len(Q_Cxyz))

    for i in range(len(Q_Cxyz)):
        for j in range(len(Q_Hxyz)):
            if (Q_Cxyz[i][0] - Q_Hxyz[j][0]) ** 2 + (
                Q_Cxyz[i][1] - Q_Hxyz[j][1]
            ) ** 2 + (Q_Cxyz[i][2] - Q_Hxyz[j][2]) ** 2 < SQ_NEIGHBOR_DISTANCE:
                count_C_neighbor_H[i] += 1

    Boundary_C_bool = [
        True if count >= MIN_BOUNDARY_NEIGHBOR else False
        for count in count_C_neighbor_H
    ]
    Boundary_C = [
        element for element, bool_value in zip(Q_Cxyz, Boundary_C_bool) if bool_value
    ]

    return Boundary_C


@nb.njit
def Calc_Perimeter(Boundary_C):
    perimeter = 0
    num_Boundary_C = len(Boundary_C)

    for i in range(num_Boundary_C):
        count_C_neighbor_C = 0
        perimeter_i = 0

        for j in range(num_Boundary_C):
            sq_distance = (
                (Boundary_C[i][0] - Boundary_C[j][0]) ** 2
                + (Boundary_C[i][1] - Boundary_C[j][1]) ** 2
                + (Boundary_C[i][2] - Boundary_C[j][2]) ** 2
            )
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
    numdir += 1
    xmls = [
        xml
        for xml in os.listdir(os.path.join(currentdir, wdir))
        if re.match(r"cpt\.\d+\.xml", xml)
    ]
    xmls.sort(key=lambda x: int(re.split(r"(\d+)", x)[1]))
    xml = os.path.join(wdir, xmls[-1])
    dcd = os.path.join(wdir, DCDNAME)
    path_to_perimeter_data = os.path.join(wdir, "perimeterByDistance.dat")

    U = mda.Universe(xml, dcd)
    C = U.select_atoms("type C")
    O = U.select_atoms("type O")
    O1 = U.select_atoms("type O1")
    H = U.select_atoms("type H")
    T = U.select_atoms("type T")
    T1 = U.select_atoms("type T1")
    t = 0

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
        fold_back(Cxyz)
        fold_back(Oxyz)
        fold_back(O1xyz)
        fold_back(Hxyz)
        fold_back(Txyz)
        fold_back(T1xyz)
        Qualified_C = Cxyz[Find_Qualified_CH_ind(Cxyz, Oxyz, O1xyz)]
        Qualified_H = Hxyz[Find_Qualified_CH_ind(Hxyz, Txyz, T1xyz)]
        Boundary_C = FindBoundary(Qualified_C, Qualified_H)
        perimeter = Calc_Perimeter(Boundary_C)

        with open(path_to_perimeter_data, "a") as f:
            f.write(f"{(numdir-1)*10+t*TIMESTEP}\t{perimeter:.4f}\n")

    # perimeter_data = pd.read_csv("perimeter.dat", delimiter="\t")
    # x = perimeter_data.iloc[1:, 0]
    # y = perimeter_data.iloc[1:, 1]
    # plt.plot(x, y, linewidth=0.6)
    # plt.show()
