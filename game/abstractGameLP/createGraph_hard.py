import numpy as np
import cvxpy as cp

"""
pure grid (2d array) representation
"""

print("using createGraph_hard")




def buildMatrix():

    direcMatrix = np.zeros((10,10,2))

                
    up = [0,20]
    down = [0,-20]
    right = [20,0]
    left = [-20, 0]

    direcMatrix[5][5] = down
    direcMatrix[4][5] = left
    direcMatrix[4][4] = down 
    direcMatrix[3][4] = left
    direcMatrix[3][3] = left
    direcMatrix[3][2] = left
    direcMatrix[3][1] = down
    direcMatrix[2][1] = right
    direcMatrix[2][2] = right
    direcMatrix[2][3] = right
    direcMatrix[2][4] = down
    direcMatrix[1][4] = right
    direcMatrix[1][5] = down
    direcMatrix[0][5] = right
    direcMatrix[0][6] = up
    direcMatrix[1][6] = right
    direcMatrix[1][7] = up
    direcMatrix[2][7] = right
    direcMatrix[2][8] = up
    direcMatrix[3][8] = up
    direcMatrix[4][8] = up
    direcMatrix[5][8] = up
    direcMatrix[6][8] = up
    direcMatrix[7][8] = up
    direcMatrix[8][8] = left
    direcMatrix[8][7] = down
    direcMatrix[7][7] = left
    direcMatrix[7][6] = down
    direcMatrix[6][6] = left
    direcMatrix[6][5] = down

    return direcMatrix, None




M, S = buildMatrix()


def obs2mu(o):

    x = o[0]
    y = o[1]

    # 50 since 50 * 10 = 500
    r = min(int(y // 50), 9)
    c = min(int(x // 50), 9)

    return M[r][c]








"""
plan:
obs2mu: input: o, output: mu

pos2mu: input: o, output: mu

rc2mu: input: r & c, output: mu (using the constructed matrix)

buildMatrix: loop through matrix, first fill non-empty, then fill empty

this matrix will contain the optimal action for the current grid
"""






































