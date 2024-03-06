import MDAnalysis as mda
import numba as nb
import numpy as np
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import alphashape
import os
import re
import warnings


EPS = 2.15
MINPTS = 9
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
    """Calculates the distacne matrix of a group of points

    Args:
        points (np.array(N, 2)): stores the coordinate of points

    Returns:
        matrix (np.array(N, N)): whose [i,j] postion indicate the dist between point[i] and point[j]
    """

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


@nb.njit
def find_H_in_C_cluster(c_xy, h_xy):
    """Find the index of H which are in cluster of C

    Args:
        c_xy (np.array): stores the xy-coordinate of C
        h_xy (np.array): stores the xy-coordinate of H

    Returns:
        inside_h_index (np.array): stores the index(from h_xy) of H in cluster of C
    """

    found = 0
    inside_h_index = np.zeros(len(h_xy), dtype=nb.int64)

    for i in range(len(h_xy)):
        neighbor_count = 0

        for j in range(len(c_xy)):
            r = h_xy[i] - c_xy[j]
            apply_min_img_2D(r)
            distance = np.sqrt(np.sum(r**2))
            if distance < EPS:
                neighbor_count += 1

        if neighbor_count >= MINPTS:
            inside_h_index[found] = i
            found += 1

    return inside_h_index[:found]


@nb.njit
def make_continuous_cluster(cluster_points):
    for i in range(1, len(cluster_points)):
        r = cluster_points[i] - cluster_points[i - 1]
        apply_min_img(r)
        cluster_points[i] = cluster_points[i - 1] + r


currentdir = os.getcwd()
wdirs = [
    f for f in os.listdir(currentdir) if (os.path.isdir(f) and re.match(r"[sr]\d+", f))
]
wdirs.sort(key=lambda x: int(re.split(r"(\d+)", x)[1]))

if len(wdirs) == 0:
    raise Exception("Wrong working directory!")

imgdir = os.path.join(currentdir, "imgs")
os.makedirs(imgdir)

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
    path_to_perimeter_data = os.path.join(currentdir, wdir, "perimeter.dat")

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
    temp_t = 0

    with open(path_to_perimeter_data, "w") as f:
        f.write("t\tperimeter\n")

    for ts in U.trajectory[1:]:
        t += 1
        temp_t += 1
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
        qualified_H = Hxyz[Find_Qualified_CH_ind(Hxyz, Txyz, T1xyz)]

        qualified_C_xy = qualified_C[:, :2]
        qualified_H_xy = qualified_H[:, :2]

        dist_matrix = distance_matrix(qualified_C_xy)

        ###* Use DBSCAN to find clusters exclude noise *###
        dbscan = DBSCAN(eps=EPS, min_samples=MINPTS, metric="precomputed")
        clusters = dbscan.fit_predict(dist_matrix)
        global_core_indices = dbscan.core_sample_indices_

        label_set = set(dbscan.labels_)
        label_set.remove(-1)

        perimeter = 0
        ###* Process every cluster by label *###
        for label in label_set:
            index_of_points = np.where(dbscan.labels_ == label)[0]
            core_index = np.intersect1d(index_of_points, global_core_indices)
            border_index = np.setdiff1d(index_of_points, core_index)

            points_in_this_cluster = qualified_C_xy[index_of_points]
            core_points = qualified_C_xy[core_index]
            border_points = qualified_C_xy[border_index]

            ###* Insert H into C cluster *###
            inside_H_index = find_H_in_C_cluster(points_in_this_cluster, qualified_H_xy)
            inside_H = qualified_H_xy[inside_H_index]
            new_cluster = np.append(points_in_this_cluster, inside_H, axis=0)
            make_continuous_cluster(new_cluster)

            ###* Use alphashape to find the border points of the cluster and plot them separately *###
            alpha_shape = alphashape.alphashape(new_cluster, alpha=0.5)
            perimeter += alpha_shape.length

            plt.scatter(new_cluster[:, 0], new_cluster[:, 1], s=6)

        plt.axvline(x=0, color="black")
        plt.axvline(x=80, color="black")
        plt.axhline(y=0, color="black")
        plt.axhline(y=80, color="black")
        plt.xlim([-40, 120])
        plt.ylim([-40, 120])
        plt.savefig(f"{imgdir}/{t}.png", format="png")
        plt.close()

        with open(path_to_perimeter_data, "a") as f:
            f.write(f"{t}\t{perimeter}\n")

    numdir += 1
