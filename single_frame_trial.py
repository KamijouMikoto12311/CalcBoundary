import MDAnalysis as mda
import numba as nb
import numpy as np
import xml.etree.ElementTree as et
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import re
import warnings


EPS = 2.0
MINPTS = 8
TIMESTEP = 0.01
DCDNAME = "traj.dcd"


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
def apply_min_img(r):
    for dim in range(3):
        if r[dim] > half_box_size[dim]:
            r[dim] -= box_size[dim]
        elif r[dim] < -half_box_size[dim]:
            r[dim] += box_size[dim]


@nb.njit
def apply_min_img_2D(r):
    for dim in range(2):
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


@nb.njit
def distance_matrix(points):
    num_points = len(points)
    matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i + 1, num_points):
            r = points[i] - points[j]
            apply_min_img_2D(r)
            distance = np.sqrt(np.sum(r**2))
            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix


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

    ts = 1
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
    qualified_C = Cxyz[Find_Qualified_CH_ind(Cxyz, Oxyz, O1xyz)]
    qualified_C_xy = qualified_C[:, :2]

    # for point in qualified_C_xy:
    #     if point[1] > 40:
    #         point[1] -= 80

    dist_matrix = distance_matrix(qualified_C_xy)

    dbscan = DBSCAN(eps=EPS, min_samples=MINPTS, metric="precomputed")
    clusters = dbscan.fit_predict(dist_matrix)

    label_set = set(dbscan.labels_)
    label_set.remove(-1)
    core_indices = dbscan.core_sample_indices_

    for label in label_set:
        index_of_points = np.where(dbscan.labels_ == label)[0]

        core_index = np.intersect1d(index_of_points, core_indices)
        border_index = np.setdiff1d(index_of_points, core_index)

        core_points = qualified_C_xy[core_index]
        border_points = qualified_C_xy[border_index]

        # plt.scatter(points[:, 0], points[:, 1], s=5)

        plt.scatter(core_points[:, 0], core_points[:, 1], c="orange", s=5)
        plt.scatter(border_points[:, 0], border_points[:, 1], c="blue", s=5)

    plt.axvline(x=0, color="black", linewidth=0.8)
    plt.axvline(x=80, color="black", linewidth=0.8)
    plt.axhline(y=0, color="black", linewidth=0.8)
    plt.axhline(y=80, color="black", linewidth=0.8)
    plt.show()
