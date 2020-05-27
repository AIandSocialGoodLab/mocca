import numpy as np
import cvxpy as cp

"""
explanation for non 1 stationary prob -> multiple solutions, not the one we expect
"""

print("using createGraph_v2")

# numNodes = 9

edgeList = [(0,1), (1,2), (3,4), (4,5), (6,7), (7,8),
			(0,3), (3,6), (1,4), (4,7), (2,5), (5,8),
			(0,0), (1,1), (2,2), (3,3), (4,4), (5,5),
			(6,6),( 7,7), (8,8)]

discretLevels = 6

def createGraphMatrix(numNodes, edgeList):

	G = np.zeros((numNodes, numNodes))

	for (i,j) in edgeList:
		G[i][j] = 1
		G[j][i] = 1

	return G

# def assignNodeReward(numNodes):

# 	r = np.zeros(numNodes)
# 	r[0] = 1
# 	r[1] = 1
# 	r[2] = 1
# 	r[3] = 1
# 	r[4] = 100
# 	r[5] = 1
# 	r[6] = 0
# 	r[7] = 1
# 	r[8] = 1

# 	return r

# def assignNodeReward(numNodes):

# 	r = np.zeros(numNodes)
# 	r[0] = 0
# 	r[1] = 0
# 	r[2] = 0
# 	r[3] = 0
# 	r[4] = 0
# 	r[5] = 0
# 	r[6] = 0
# 	r[7] = 0
# 	r[8] = 10

# 	return r


# def assignNodeReward(numNodes):

# 	r = np.zeros(numNodes)
# 	r[0] = 0
# 	r[1] = 15
# 	r[2] = 0
# 	r[3] = 10
# 	r[4] = 0
# 	r[5] = 0
# 	r[6] = 0
# 	r[7] = 0
# 	r[8] = 20

# 	return r



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
			G[x][x] = 1

	return G

# def assignNodeReward(numNodes):

# 	r = np.zeros(numNodes)
# 	r[0] = 0
# 	r[1] = 15
# 	r[2] = 0
# 	r[3] = 10
# 	r[4] = 0
# 	r[5] = 0
# 	r[6] = 0
# 	r[7] = 0
# 	r[8] = 20

# 	return r

def assignNodeReward(numNodes):

	r = np.zeros(numNodes)
	r[16] = 10
	r[24] = 15
	r[42] = 10
	r[44] = 10
	r[88] = 20

	return r

numNodes = 100

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
# for i in range(0,numNodes):
# 	curSum = 0
# 	for j in range(0, numNodes):
# 		curSum += x[i][j]
# 	constr.append(curSum == mu[i])

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

# sum of in flow is equal to sum of out flow: should not: since there are self loops
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

unitx = 167
unity = 167

actdim = 20


# node unitx & unity should be round up version of units
def pos2node(x, y):

	x = np.minimum(x,X)
	y = np.minimum(y,Y)

	r = y // unity
	c = x // unitx

	rs = Y // unity + 1
	cs = X // unitx + 1

	nodeid = r * rs + c

	return nodeid



def whetherEmpty(numNodes):

	res = np.ones(numNodes)

	for i in range(numNodes):
		curSum = 0
		for j in range(numNodes):
			curSum += x[i][j].value
		if curSum == 0:
			res[i] = 0

	return res

from collections import deque
def bfs(nodeid, whetherEmptyRes, cs):

	visited = set()
	q = deque()
	q.append((nodeid,1))

	while len(q) != 0:

		n, lv = q.popleft()

		if n in visited:
			continue

		visited.add(n)

		if whetherEmptyRes[n] != 0:
			return lv

		if (nodeid - cs) in range(0, numNodes):
			q.append((nodeid-cs, lv+1))

		if (nodeid + cs) in range(0, numNodes):
			q.append((nodeid+cs, lv+1))

		if (nodeid-1) in range(0, numNodes) and (nodeid % cs != 0):
			q.append((nodeid-1, lv+1))

		if (nodeid+1) in range(0, numNodes) and ((nodeid+1)%cs != 0):
			q.append((nodeid+1, lv+1))

	return 100



def emptyDirection(nodeid, G, cs, actdim, whetherEmptyRes):

	leftDist = 100
	if (nodeid - cs) in range(0, numNodes):
		leftDist = bfs(nodeid-cs, whetherEmptyRes, cs)

	rightDist = 100
	if (nodeid + cs) in range(0, numNodes):
		rightDist = bfs(nodeid+cs, whetherEmptyRes, cs)

	upDist = 100
	if (nodeid-1) in range(0, numNodes) and (nodeid % cs != 0):
		upDist = bfs(nodeid-1, whetherEmptyRes, cs)

	downDist = 100
	if (nodeid+1) in range(0, numNodes) and ((nodeid+1)%cs != 0):
		downDist = bfs(nodeid+1, whetherEmptyRes, cs)

	distList = [upDist, downDist, leftDist, rightDist]

	optidx = np.argmin(distList)

	if optidx == 0:
		return np.array([0,-actdim])
	elif optidx == 1:
		return np.array([0, actdim])
	elif optidx == 2:
		return np.array([-actdim, 0])
	else:
		return np.array([actdim, 0])
	


# # could import env to get some parameters

def node2mu(nodeid, numNodes, cs, actdim):

	whetherEmptyRes = whetherEmpty(numNodes)
	
	curSum = 0
	for j in range(numNodes):
		curSum += x[nodeid][j].value

	# if mu_i == 0 output "random" action
	if curSum == 0:
		# return np.array([0,0])
		# go to nearest node with a value
		return emptyDirection(nodeid, G, cs, actdim, whetherEmptyRes)

	upProb = 0
	if (nodeid - cs) in range(0, numNodes):
		upProb = x[nodeid][nodeid-cs].value / curSum


	downProb = 0
	if (nodeid + cs) in range(0, numNodes):
		downProb = x[nodeid][nodeid+cs].value / curSum

	
	leftProb = 0
	if (nodeid - 1) in range(0, numNodes) and (nodeid % cs != 0):
		leftProb = x[nodeid][nodeid-1].value / curSum

	rightProb = 0
	if (nodeid + 1) in range(0, numNodes) and ((nodeid+1)%cs != 0):
		rightProb = x[nodeid][nodeid+1].value / curSum

	stayProb = 0
	if (nodeid) in range(0, numNodes):
		stayProb = x[nodeid][nodeid].value / curSum
	

	plist = [upProb, downProb, leftProb, rightProb, stayProb]

	optidx = np.argmax(plist)

	# todo: need to change this if action dim changes
	if optidx == 0:
		return np.array([0,-actdim])
	elif optidx == 1:
		return np.array([0, actdim])
	elif optidx == 2:
		return np.array([-actdim, 0])
	elif optidx == 3:
		return np.array([actdim, 0])
	else:
		return np.array([0.0])


def getNode2muDict(numNodes, cs, actdim):

	node2muDict = dict()
	for i in range(0, numNodes):
		node2muDict[i] = node2mu(i, numNodes, cs, actdim)

	return node2muDict

cs = X // unitx + 1

node2muDict = getNode2muDict(numNodes, cs, actdim)

def pos2mu(x,y):

	x = np.minimum(x,X)
	y = np.minimum(y,Y)

	nodeid = pos2node(x, y)

	return node2muDict[nodeid]

def obs2mu(o):

	x = o[0]
	y = o[1]

	return pos2mu(x,y)






"""
todo:

apply pos2mu to all observations
"""









































