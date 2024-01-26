#!/opt/homebrew/bin/
import MDAnalysis as mda
import numba as nb
import numpy as np
import sys
import warnings


NHEAD = 10658
LX = 80
LY = 80
LZ = 40
XML = sys.argv[1]
DCD = sys.argv[2]

SIDELENG = int(np.sqrt(NHEAD / 2))
RATE = SIDELENG / LX
HALFLX = LX / 2
HALFLY = LY / 2
HALFLZ = LZ / 2


@nb.njit
def fold_back(xyz):
    for i in range(0, len(xyz)):
        xyz[i][0] = (xyz[i][0] + HALFLX) % LX
        xyz[i][1] = (xyz[i][1] + HALFLY) % LY
        xyz[i][2] = (xyz[i][2] + HALFLZ) % LZ


@nb.njit
def FindC(C, O, O1):
    numC = len(C)
    board = np.zeros((SIDELENG + 2, SIDELENG + 2))

    for i in range(numC):
        if C[i][2] > O[i][2] and C[i][2] > O1[i][2]:
            board[int(RATE * C[i][0]) + 1][int(RATE * C[i][1]) + 1] += 1

    return board


@nb.njit
def FindBoundary(board):
    binboard = np.zeros((SIDELENG, SIDELENG))

    board[0] = board[SIDELENG]
    board[SIDELENG + 1] = board[1]
    for i in range(SIDELENG + 2):
        board[i][0] = board[i][SIDELENG]
        board[i][SIDELENG + 1] = board[i][1]

    for i in range(1, SIDELENG + 1):
        for j in range(1, SIDELENG + 1):
            if board[i][j] >= 1 and (
                board[i - 1][j] == 0
                or board[i + 1][j] == 0
                or board[i][j - 1] == 0
                or board[i][j + 1] == 0
            ):
                binboard[i - 1][j - 1] = 1

    return binboard


U = mda.Universe(XML, DCD)
C = U.select_atoms("type C")
O = U.select_atoms("type O")
O1 = U.select_atoms("type O1")
H = U.select_atoms("type H")
T = U.select_atoms("type T")
T1 = U.select_atoms("type T1")

# for ts in U.trajectory[1:]:
ts = 0
Cxyz = C.positions
Oxyz = O.positions
O1xyz = O1.positions
fold_back(Cxyz)
fold_back(Oxyz)
fold_back(O1xyz)

board = FindC(Cxyz, Oxyz, O1xyz)
binboard = FindBoundary(board)
with open("binboard.dat", "w") as file:
    for i in range(1, SIDELENG + 1):
        for j in range(1, SIDELENG + 1):
            file.write(f"{board[i][j]}  ")
        file.write("\n")

with open("boundary.dat", "w") as file:
    for i in range(0, SIDELENG):
        for j in range(0, SIDELENG):
            file.write(f"{binboard[i][j]}  ")
        file.write("\n")
