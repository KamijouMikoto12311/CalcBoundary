import MDAnalysis as mda
import numba as nb
import numpy as np
import xml.etree.ElementTree as et
from sklearn.cluster import DBSCAN
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

    return np.array(Qualified_CH_ind)


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
        Qualified_C = Cxyz[Find_Qualified_CH_ind(Cxyz, Oxyz, O1xyz)]
        qualified_C_xy = Qualified_C[:, :2]

        dbscan = DBSCAN(eps=1.2, min_samples=10)
        dbscan.fit(qualified_C_xy)
        core_indices = dbscan.core_sample_indices_

        for label in set(dbscan.labels_):
            index_of_points = np.where(dbscan.labels_ == label)[0]
            index_of_core_points = np.intersect1d(index_of_points, core_indices)
            index_of_border_points = np.setdiff1d(index_of_points, index_of_core_points)
            
            
            
            

        with open(path_to_perimeter_data, "a") as f:
            f.write(f"{t*TIMESTEP:.2f}\t{perimeter:.4f}\n")

    numdir += 1
