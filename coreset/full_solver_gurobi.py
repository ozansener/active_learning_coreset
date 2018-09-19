import numpy
from gurobipy import *
import pickle
import numpy.matlib
import time
import pickle
import bisect

def solve_fac_loc(xx,yy,subset,n,budget):
    model = Model("k-center")
    x={}
    y={}
    z={}
    for i in range(n):
        # z_i: is a loss
        z[i] = model.addVar(obj=1, ub=0.0, vtype="B", name="z_{}".format(i))
 
    m = len(xx)
    for i in range(m):
        _x = xx[i]
        _y = yy[i]
        # y_i = 1 means i is facility, 0 means it is not
        if _y not in y:
            if _y in subset: 
                y[_y] = model.addVar(obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y))
            else:
                y[_y] = model.addVar(obj=0,vtype="B", name="y_{}".format(_y))
        #if not _x == _y:
        x[_x,_y] = model.addVar(obj=0, vtype="B", name="x_{},{}".format(_x,_y))
    model.update()

    coef = [1 for j in range(n)]
    var = [y[j] for j in range(n)]
    model.addConstr(LinExpr(coef,var), "=", rhs=budget+len(subset), name="k_center")

    for i in range(m):
        _x = xx[i]
        _y = yy[i]
        #if not _x == _y:
        model.addConstr(x[_x,_y], "<", y[_y], name="Strong_{},{}".format(_x,_y))

    yyy = {}
    for v in range(m):
        _x = xx[v]
        _y = yy[v]
        if _x not in yyy:
            yyy[_x]=[]
        if _y not in yyy[_x]:
            yyy[_x].append(_y)

    for _x in yyy:
        coef = []
        var = []
        for _y in yyy[_x]:
            #if not _x==_y:
            coef.append(1)
            var.append(x[_x,_y])
        coef.append(1)
        var.append(z[_x])
        model.addConstr(LinExpr(coef,var), "=", 1, name="Assign{}".format(_x))
    model.__data = x,y,z
    return model


data = pickle.load(open('feature_vectors_pickled'))
budget = 10000

start = time.clock()
num_images = data.shape[0]
dist_mat = numpy.matmul(data,data.transpose())

sq = numpy.array(dist_mat.diagonal()).reshape(num_images,1)
dist_mat *= -2
dist_mat+=sq
dist_mat+=sq.transpose()

elapsed = time.clock() - start
print "Time spent in (distance computation) is: ", elapsed

num_images = 50000

# We need to get k centers start with greedy solution
budget = 10000
subset = [i for i in range(1)]
 
ub= UB
lb = ub/2.0
max_dist=ub

_x,_y = numpy.where(dist_mat<=max_dist)
_d = dist_mat[_x,_y]
subset = [i for i in range(1)]
model = solve_fac_loc(_x,_y,subset,num_images,budget)
#model.setParam( 'OutputFlag', False )
x,y,z = model.__data
delta=1e-7
while ub-lb>delta:
    print "State",ub,lb
    cur_r = (ub+lb)/2.0
    viol = numpy.where(_d>cur_r)
    new_max_d = numpy.min(_d[_d>=cur_r])
    new_min_d = numpy.max(_d[_d<=cur_r])
    print "If it succeeds, new max is:",new_max_d,new_min_d
    for v in viol[0]:
        x[_x[v],_y[v]].UB = 0

    model.update()
    r = model.optimize()
    if model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
        failed=True
        print "Infeasible"
    elif sum([z[i].X for i in range(len(z))]) > 0:
        failed=True
        print "Failed"
    else:
        failed=False
    if failed:
        lb = max(cur_r,new_max_d)
        #failed so put edges back
        for v in viol[0]:
            x[_x[v],_y[v]].UB = 1
    else:
        print "sol founded",cur_r,lb,ub
        ub = min(cur_r,new_min_d)
        model.write("s_{}_solution_{}.sol".format(budget,cur_r))

