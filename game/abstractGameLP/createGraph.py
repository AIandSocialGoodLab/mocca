import numpy as np
import cvxpy as cp

numNodes = 9

edgeList = [(0,1), (1,2), (3,4), (4,5), (6,7), (7,8),
			(0,3), (3,6), (1,4), (4,7), (2,5), (5,8)]

discretLevels = 6

def createGraphMatrix(numNodes, edgeList):

	G = np.zeros((numNodes, numNodes))

	for (i,j) in edgeList:
		G[i][j] = 1
		G[j][i] = 1

	return G

def assignNodeReward(numNodes):

	r = np.zeros(numNodes)
	r[0] = 1
	r[1] = 1
	r[2] = 1
	r[3] = 1
	r[4] = 100
	r[5] = 1
	r[6] = 0
	r[7] = 1
	r[8] = 1

	return r

p = np.arange(discretLevels) * (1.0/(discretLevels-1))


# initialize variables
Z = 100000000

d = np.empty((numNodes, numNodes, discretLevels), dtype=object)

mu = np.empty(numNodes, dtype=object)

w = np.empty((numNodes, numNodes, discretLevels), dtype=object)

v = cp.Variable()

x = cp.Variable()

for i in range(0,numNodes):
	mu[i] = cp.Variable()
	for j in range(0, numNodes):
		for l in range(0, discretLevels):
			d[i][j][l] = cp.Variable(boolean=True)
			w[i][j][l] = cp.Variable()

# objective
obj = cp.Minimize(v)


# constraints
constr = []

G = createGraphMatrix(numNodes, edgeList)

r = assignNodeReward(numNodes)

# total probability out of a node must be equal to 1
for i in range(0,numNodes):
	curSum = 0
	for j in range(0, numNodes):
		# note: not checking whether ij edge exists since encoded in the lp
		for l in range(0, discretLevels):
			curSum += p[l] * d[i][j][l]

	constr.append(curSum == 1)

# only make one choice for choice variables
for i in range(0, numNodes):
	for j in range(0, numNodes):
		curSum = 0
		for l in range(0, discretLevels):
			curSum += d[i][j][l]

		constr.append(curSum == 1)

# only generate valid paths
for i in range(numNodes):
	for j in range(numNodes):
		curSum = 0
		for l in range(discretLevels):
			curSum += p[l]*d[i][j][l]

		constr.append(curSum <= G[i][j])

# stationary equations
for j in range(numNodes):
	curSum = 0
	for i in range(numNodes):
		for l in range(discretLevels):
			curSum += p[l] * w[i][j][l]
	constr.append(mu[j] == curSum)

# mu should be valid prob distribution
curSum = 0
for j in range(numNodes):
	curSum += mu[j]
	constr.append(mu[j] >= 0)
	constr.append(mu[j] <= 1)

constr.append(curSum == 1)



# w_ijl constraints
for i in range(numNodes):
	for j in range(numNodes):
		for l in range(discretLevels):

			constr.append(w[i][j][l] >= mu[i] - Z*(1 - d[i][j][l]))
			constr.append(w[i][j][l] <= mu[i] + Z*(1 - d[i][j][l]))

			constr.append(w[i][j][l] >= - Z * d[i][j][l])
			constr.append(w[i][j][l] <= Z * d[i][j][l])

# attacker makes optimal choice
for j in range(numNodes):

	constr.append(v >= (1-mu[j])*r[j] + mu[j]*(-r[j]) )




# cvxpy solver
prob = cp.Problem(obj, constr)

res = prob.solve(solver=cp.GLPK_MI)

print(res)
print(prob.status)

def getNodeOutEdges(i):

	print("out probs for node: ", i)

	print("mu:", mu[i].value)

	for j in range(numNodes):
		curSum = 0
		for l in range(discretLevels):
			curSum += p[l] * d[i][j][l].value

		print("pi_", i,j,":", curSum)

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


# could import env to get some parameters

def node2mu(nodeid, numNodes, cs, actdim):

	upProb = 0
	if (nodeid - cs)  in range(0, numNodes):
		for l in range(discretLevels):
			upProb += p[l] * d[nodeid][nodeid-cs][l].value

	downProb = 0
	if (nodeid + cs) in range(0, numNodes):
		for l in range(discretLevels):
			downProb += p[l] * d[nodeid][nodeid+cs][l].value

	leftProb = 0
	if (nodeid - 1) in range(0, numNodes):
		for l in range(discretLevels):
			downProb += p[l] * d[nodeid][nodeid-1][l].value

	rightProb = 0
	if (nodeid + 1) in range(0, numNodes):
		for l in range(discretLevels):
			downProb += p[l] * d[nodeid][nodeid+1][l].value

	plist = [upProb, downProb, leftProb, rightProb]

	optidx = np.argmax(plist)

	# todo: need to change this if action dim changes
	if optidx == 0:
		return np.array([0,-actdim])
	elif optidx == 1:
		return np.array([0, actdim])
	elif optidx == 2:
		return np.array([-actdim, 0])
	else:
		return np.array([0, actdim])


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

	return np.append(pos2mu(x,y),[0,0])






"""
todo:

apply pos2mu to all observations
"""









































