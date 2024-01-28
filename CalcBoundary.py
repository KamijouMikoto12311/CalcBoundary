#!/opt/homebrew/bin/
import MDAnalysis as mda
import numba as nb
import numpy as np
import sys
import warnings

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


NHEAD = 10658
TIMESTEP = 0.01
LX = 80
LY = 80
LZ = 40
XML = sys.argv[1]
DCD = sys.argv[2]
t = 0

SIDELENG = 65  # int(np.sqrt(NHEAD / 2))
RATE = SIDELENG / LX
HALFLX = LX / 2
HALFLY = LY / 2
HALFLZ = LZ / 2


warnings.filterwarnings("ignore")


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
    trace = np.zeros((SIDELENG + 2, SIDELENG + 2, 3))

    for i in range(numC):
        if C[i][2] > O[i][2] and C[i][2] > O1[i][2]:
            board[int(RATE * C[i][0]) + 1][int(RATE * C[i][1]) + 1] += 1
            trace[int(RATE * C[i][0]) + 1][int(RATE * C[i][1]) + 1] = C[i]

    # Apply periodic boundary condition (partly) to make it easier to find the boundary
    board[0] = board[SIDELENG]
    board[SIDELENG + 1] = board[1]
    for i in range(SIDELENG + 2):
        board[i][0] = board[i][SIDELENG]
        board[i][SIDELENG + 1] = board[i][1]

    # Due to board fit, for a lattice which has more than one particle, coarsen by add 1 around it
    for i in range(1, SIDELENG + 1):
        for j in range(1, SIDELENG + 1):
            if board[i][j] > 1:
                board[i - 1][j - 1] += int(board[i - 1][j - 1] == 0)
                board[i - 1][j] += int(board[i - 1][j] == 0)
                board[i - 1][j + 1] += int(board[i - 1][j + 1] == 0)
                board[i][j - 1] += int(board[i][j - 1] == 0)
                board[i][j + 1] += int(board[i][j + 1] == 0)
                board[i + 1][j - 1] += int(board[i + 1][j - 1] == 0)
                board[i + 1][j] += int(board[i + 1][j] == 0)
                board[i + 1][j + 1] += int(board[i + 1][j + 1] == 0)
    board[0] = board[SIDELENG]
    board[SIDELENG + 1] = board[1]
    for i in range(SIDELENG + 2):
        board[i][0] = board[i][SIDELENG]
        board[i][SIDELENG + 1] = board[i][1]

    #Fill holes
    for i in range(1, SIDELENG + 1):
        for j in range(1, SIDELENG + 1):
            if (
                board[i][j] == 0
                and board[i - 1][j] != 0
                and board[i + 1][j] != 0
                and board[i][j - 1] != 0
                and board[i][j + 1] != 0
            ):
                board[i][j] = 1
    board[0] = board[SIDELENG]
    board[SIDELENG + 1] = board[1]
    for i in range(SIDELENG + 2):
        board[i][0] = board[i][SIDELENG]
        board[i][SIDELENG + 1] = board[i][1]
        
    #Remove single C
    for i in range(1, SIDELENG + 1):
        for j in range(1, SIDELENG + 1):
            if (
                board[i][j] == 1
                and board[i - 1][j] == 0
                and board[i + 1][j] == 0
                and board[i][j - 1] == 0
                and board[i][j + 1] == 0
            ):
                board[i][j] = 0
    board[0] = board[SIDELENG]
    board[SIDELENG + 1] = board[1]
    for i in range(SIDELENG + 2):
        board[i][0] = board[i][SIDELENG]
        board[i][SIDELENG + 1] = board[i][1]

    return board, trace


@nb.njit
def FindBoundary(board):
    binboard = np.zeros((SIDELENG, SIDELENG))

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

with open("perimeter.dat", "w") as f:
    f.write(f" t  \tperimeter\n")

for ts in U.trajectory[1:]:
    t += 1
    Cxyz = C.positions
    Oxyz = O.positions
    O1xyz = O1.positions
    fold_back(Cxyz)
    fold_back(Oxyz)
    fold_back(O1xyz)

    board, trace = FindC(Cxyz, Oxyz, O1xyz)
    binboard = FindBoundary(board)
    totperi = np.sum(binboard)

    with open("perimeter.dat", "a") as f:
        f.write(f"{t*TIMESTEP:^4.2f}\t{totperi:^9}\n")
    if t == 1:
        with open("binboard.dat", "w") as f:
            f.write(f"{binboard}")
        with open("board.dat", "w") as f:
            f.write(f"{board}")
