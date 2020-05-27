import numpy as np
import cvxpy as cp

"""
pure grid (2d array) representation
"""

print("using createGraph_v3")

# numNodes = 9

# n is number of nodes in a single row/col
n = 10

numNodes = n*n

def createGrid(n):

	G = np.zeros((n*n, n*n))

	for i in range(0,n):
		for j in range(0,n):
			if i-1 >= 0:
				x = i*n + j
				y = (i-1)*n +j
				G[x][y] = 1
			if i+1 < n:
				x = i*n + j
				y = (i+1)*n +j

				G[x][y] = 1
			if j-1 >= 0:
				x = i*n + j
				y = i*n + j-1
				G[x][y] = 1
			if j+1 < n:
				x = i*n + j
				y = i*n + j+1
				G[x][y] = 1
			
			x = i*n + j
			# G[x][x] = 1

	return G

def assignNodeReward(numNodes):

	r = np.zeros(numNodes)
	
	r[15] = 10
	r[24] = 10
	r[42] = 10
	r[44] = 15
	r[88] = 20

	return r


# G = createGraphMatrix(numNodes, edgeList)

G = createGrid(10)

r = assignNodeReward(numNodes)

# ----------------------------------------------------

# initialize variables

mu = np.empty(numNodes, dtype=object)

v = cp.Variable()

x = np.empty((numNodes, numNodes), dtype=object)

for i in range(0,numNodes):
	mu[i] = cp.Variable()
	for j in range(0, numNodes):
		x[i][j] = cp.Variable()

	

# objective
obj = cp.Minimize(v)


# constraints
constr = []


# x_ij should be positive
for i in range(0, numNodes):
	for j in range(0, numNodes):
		constr.append(x[i][j] >= 0)


# stationary prob is equal to sum of out probabilities
for i in range(0,numNodes):
	curSum = 0
	for j in range(0, numNodes):
		curSum += x[i][j]
	constr.append(curSum == mu[i])

# or stationary prob is equal to sum of in probabilities
for i in range(0,numNodes):
	curSum = 0
	for j in range(0, numNodes):
		curSum += x[j][i]
	constr.append(curSum == mu[i])


# only generate valid paths
for i in range(0, numNodes):
	for j in range(0, numNodes):
		constr.append(x[i][j] <= G[i][j])


# sum of x_ij should be 1
curSum = 0
for i in range(0, numNodes):
	for j in range(0, numNodes):
		curSum += x[i][j]
constr.append(curSum == 1)

# sum of in flow is equal to sum of out flow: this is automatically fulfilled
# for i in range(0, numNodes):
# 	inFlow = 0
# 	outFlow = 0
# 	for j in range(0, numNodes):
# 		inFlow += x[j][i]
# 		outFlow += x[i][j]
# 	constr.append(inFlow == outFlow)

	

# attacker makes optimal choice
for j in range(numNodes):
	constr.append(v >= (1-mu[j])*r[j] + mu[j]*(-r[j]) )


# cvxpy solver
prob = cp.Problem(obj, constr)

res = prob.solve(solver=cp.GLPK)

print(res)
print(prob.status)

def getNodeOutEdges(i):

	print("out probs for node: ", i)

	print("mu:", mu[i].value)

	for j in range(numNodes):
		print("x_", i,j,":", x[i][j].value)

	

# -----------------------------------------------------

X = 500
Y = 500

# ty: important!!! need to change if discretization level changes
# unitx = 167
# unity = 167

# unitx = 50
# unity = 50

actdim = 20



def fillNonEmptyNode(i, j):

	nodeid = i*n + j

	upProb = 0 # row +1
	if (i+1) < n:
		upProb = x[nodeid][(i+1)*n+j].value

	downProb = 0
	if (i-1) >= 0:
		downProb = x[nodeid][(i-1)*n+j].value

	leftProb = 0
	if (j-1) >= 0:
		leftProb = x[nodeid][i*n+(j-1)].value

	rightProb = 0
	if (j+1) < n:
		rightProb = x[nodeid][i*n+(j+1)].value

	stayProb = x[nodeid][nodeid].value

	plist = [upProb, downProb, leftProb, rightProb, stayProb]

	up = [0,20]
	down = [0,-20]
	right = [20,0]
	left = [-20,0]
	stay = [0,0]

	upStr = "DOWN"
	downStr = "UPPP"
	rightStr = "RGHT"
	leftStr = "LEFT"
	stayStr = "STAY"

	actlist = [up, down, left, right, stay]
	strlist = [upStr, downStr, leftStr, rightStr, stayStr]

	optidx = np.argmax(plist)

	return actlist[optidx], strlist[optidx]



from collections import deque
def bfs(i,j):

	visited = set()
	q = deque()

	q.append(((i,j), 0))

	while len(q) != 0:

		((r,c),lv) = q.popleft()

		if (r,c) in visited:
			continue

		visited.add((r,c))

		curSum = 0
		curNodeid = r*n + c
		for k in range(n*n):
			curSum += x[curNodeid][k].value

		if curSum != 0:
			return lv

		if r+1 < n:
			q.append(((r+1,c),lv+1))
		if r-1 >= 0:
			q.append(((r-1,c),lv+1))
		if c+1 < n:
			q.append(((r,c+1),lv+1))
		if c-1 >= 0:
			q.append(((r,c-1),lv+1))

	print("shouldnt be here")
	return None




def fillEmptyNode(i,j):

	nodeid = i*n + j

	upDist = 100000 # row +1
	if (i+1) < n:
		upDist = bfs(i+1, j)

	downDist = 100000
	if (i-1) >= 0:
		downDist = bfs(i-1, j)
		
	leftDist = 100000
	if (j-1) >= 0:
		leftDist = bfs(i, j-1)
		
	rightDist = 100000
	if (j+1) < n:
		rightDist = bfs(i, j+1)

	distList = [upDist, downDist, leftDist, rightDist]

	up = [0,20]
	down = [0,-20]
	left = [-20,0]
	right = [20,0]
	
	
	upStr = "down"
	downStr = "uppp"
	leftStr = "left"
	rightStr = "rght"

	actlist = [up, down, left, right]
	strlist = [upStr, downStr, leftStr, rightStr]

	optidx = np.argmin(distList)

	return actlist[optidx], strlist[optidx]




def buildMatrix(n):

	direcMatrix = np.zeros((n,n,2))

	strMatrix = np.empty((n,n), dtype=object)

	for i in range(n):
		for j in range(n):
			nodeid = i*n + j

			# this should be the sum of the outgoing probabilities
			curSum = 0

			for k in range(n*n):
				curSum += x[nodeid][k].value

			if curSum > 0:
				direcMatrix[i][j], strMatrix[i][j] = fillNonEmptyNode(i,j)
			else:
				direcMatrix[i][j], strMatrix[i][j] = fillEmptyNode(i,j)

	return direcMatrix, strMatrix

M, S = buildMatrix(n)


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






































