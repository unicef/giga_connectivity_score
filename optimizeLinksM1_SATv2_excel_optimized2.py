import sys, os
import argparse
import numpy as np
from ortools.sat.python import cp_model
import pandas as pd
import networkx as nx

#Model 1
class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self,active,P,fnodesD,cols,distances):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__active = active
        self.__P = P
        self.__fnodesD = fnodesD
        self.__cols = cols
        self.__n = len(cols)+1
        self.__distances = distances

    def on_solution_callback(self):
        print('Solution %i' % self.__solution_count)
        self.__solution_count += 1

        print('  objective value = %i' % self.ObjectiveValue())
        for i in range(len(self.__fnodesD)):
            print(f'Active_{self.__fnodesD[i]} = {self.Value(self.__active[i])}')

        for i in range(self.__n - 1):
            for j in range(self.__n):
                if (j,i) in self.__P:
                    if self.Value(self.__P[(j,i)]) == 1:
                        print(f'C_{self.__cols[i]}_{self.__cols[j]} = {self.__distances[i][j]}')

def lenSp(L,sps):
    le = 0
    for l in L:
        if l in sps:
            le+=1

    return le

def node2remove(T,node):
    asc = list(nx.ancestors(T,node))
    for a in asc:
        if (a,node) in T.edges():
            return a

def fixMST(T,splittersD,fnodesD,schoolsD,maxspinfn,n):
    T2 = T.copy()
    valid = False
    exit = False
    while not exit:
        redo = False
        for i in fnodesD:
            exiti = True
            des = []
            if i in T2.nodes():
                des = list(nx.descendants(T2, i))
            if lenSp(des, splittersD) > maxspinfn:
                exiti = False
                for d in des:
                    if d in splittersD:
                        exitj = False
                        des2 = list(nx.descendants(T2, d))
                        if ((lenSp(des, splittersD) - lenSp(des2, splittersD) - 1) <= maxspinfn) and ((lenSp(des2,splittersD)+1) <= maxspinfn):
                            for j in fnodesD:
                                if i != j:
                                    desj = None
                                    if j not in T2.nodes():
                                        desj = []
                                    else:
                                        desj = list(nx.descendants(T2, j))
                                    if lenSp(desj, splittersD) + lenSp(des2, splittersD)+1 <= maxspinfn:
                                        n2r = node2remove(T2, d)
                                        T2.remove_edge(n2r, d)
                                        if j not in T2.nodes():
                                            T2.add_node(j)
                                            T2.add_edge(n-1,j)

                                        T2.add_edge(j, d)  # ,d=distances[j][d])
                                        exitj = True
                                        break
                            if exitj:
                                exiti = True
                                break
                if exiti:
                    redo = True
                    break
        if not redo:
            exit = True
            valid = exiti

    if valid:
        ###Collapse edges
        removed = True
        while removed:
            removed = False
            nodes = list(T2.nodes())
            for node in nodes:
                if node in splittersD:
                    if nx.degree(T2, node) == 1:
                        T2.remove_node(node)
                    elif nx.degree(T2, node) == 2:
                        succ = list(T2.neighbors(node))
                        pred = list(T2.predecessors(node))
                        if succ[0] not in schoolsD:
                            T2.remove_node(node)
                            T2.add_edge(pred[0], succ[0])#, d=distancesF[neis[0]][neis[1]])
                            removed = True
            ###

    return valid, T2

def doMST(distancesF,distances,fnodesD,schoolsD,splittersD,P,A,active,n,maxdegreesp,maxdegreefn,maxspinfn,maxdist):
    searchVars = []
    used_keys = []

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    for i in schoolsD:
        for j in splittersD:
            G.add_edge(i,j,d=distancesF[i][j])

    for i in fnodesD:
        G.add_edge(i, n - 1, d=0)
        for j in splittersD:
            G.add_edge(i,j,d=distancesF[i][j])

    for i in splittersD:
        for j in splittersD:
            if i!=j:
                G.add_edge(i,j,d=distancesF[i][j])

    T = nx.minimum_spanning_tree(G, weight='d', algorithm='boruvka')
    ###Collapse edges
    removed = True
    while removed:
        removed = False
        nodes = list(T.nodes())
        for node in nodes:
            if node in splittersD:
                if nx.degree(T, node) == 1:
                    T.remove_node(node)
                elif nx.degree(T, node) == 2:
                    neis = list(T.neighbors(node))
                    if neis[0] not in schoolsD and neis[1] not in schoolsD:
                        T.remove_node(node)
                        T.add_edge(neis[0], neis[1], d=distancesF[neis[0]][neis[1]])
                        removed = True
        ###
    valid = True
    for node in T.nodes():
        if node in fnodesD:
            if T.degree(node)>maxdegreefn+1:
                valid = False
        elif node in splittersD:
            if T.degree(node)>maxdegreesp+1:
                valid = False

    T2 = nx.DiGraph()
    T2.add_node(n-1)
    splittersPerFnode = {}
    for i in schoolsD:
        try:
            path_i = nx.shortest_path(T, i, n - 1, weight='d')
        except:
            path_i = []
            valid = False
            print('----school',i,' has invalid path')

        try:
            pdist = nx.dijkstra_path_length(T,i,n-1,weight='d')
        except:
            pdist = maxdist+1
        if pdist>maxdist:
            valid = False
        fnode = None
        if len(path_i)>0:
            fnode = path_i[len(path_i)-2]
            if fnode not in splittersPerFnode:
                splittersPerFnode[fnode] = []

        for k in range(len(path_i) - 1):
            if path_i[k + 1] not in schoolsD and (path_i[k] not in fnodesD or k == len(path_i) - 2):
                if path_i[k+1] not in splittersPerFnode[fnode] and k<len(path_i) - 3:
                    splittersPerFnode[fnode].append(path_i[k+1])
                if path_i[k] not in T2.nodes():
                    T2.add_node(path_i[k])
                if (path_i[k], path_i[k + 1]) not in used_keys:
                    searchVars.append(P[(path_i[k], path_i[k + 1])])
                    used_keys.append((path_i[k], path_i[k + 1]))
                    T2.add_edge(path_i[k+1], path_i[k])#, d=distances[path_i[k]][path_i[k+1]])
            else:
                valid = False
                print('----school', i, ' has invalid path')
    upbound = -1
    if valid:
        for key in splittersPerFnode:
            if len(splittersPerFnode[key]) > maxspinfn:
                valid = False
                break
        if not valid:
            valid,T3 = fixMST(T2, splittersD, fnodesD, schoolsD, maxspinfn,n)
            if valid:
                used_keys = []
                searchVars = []
                for edge in T3.edges():
                    if edge not in used_keys:
                        a,b = edge
                        searchVars.append(P[(b,a)])
                        used_keys.append((b,a))

                for i in fnodesD:
                    if (i,n-1) in used_keys:
                        searchVars.append(active[i])
                upbound = 0
                for edge in T3.edges():
                    a, b = edge
                    if a != n - 1 and b != n - 1:
                        upbound += distances[a][b]
        else:
            upbound = 0
            for edge in T.edges():
                a,b=edge
                if a!=n-1 and b!=n-1:
                    upbound += distances[a][b]

            for i in fnodesD:
                if (i, n - 1) in used_keys:
                    searchVars.append(active[i])

    searchVars2 = []
    for i in fnodesD:
        if (i, n - 1) not in used_keys:
            searchVars2.append(active[i])


    for i in range(n):
        for j in range(n):
            if (i,j) in P and not (i,j) in used_keys:
                searchVars2.append(P[(i,j)])

    for i in range(n):
        for j in range(n):
            if (i,j) in A:
                searchVars2.append(A[(i, j)])

    return searchVars,searchVars2,upbound

def readExcelDistances(fn):
    df_sheet_v = pd.read_excel(fn, sheet_name='vertices')
    splitter_ids = list(df_sheet_v['Splitter nodes'].dropna())
    fnode_ids = list(df_sheet_v['Fiber nodes'].dropna())
    school_ids = list(df_sheet_v['School nodes'].dropna())
    n = len(splitter_ids)+len(fnode_ids)+len(school_ids)
    df_sheet_d = pd.read_excel(fn, sheet_name='distance_matrix')

    for i in range(len(splitter_ids)):
        if type(splitter_ids[i])==str:
            splitter_ids[i] = splitter_ids[i].lstrip()

    print(splitter_ids)
    cols = list(df_sheet_d.columns)[1:]
    fnodesD = []
    schoolsD = []
    splittersD = []
    for i in range(len(cols)):
        if cols[i] in splitter_ids:
            splittersD.append(i)
        elif cols[i] in school_ids:
            schoolsD.append(i)
        elif cols[i] in fnode_ids:
            fnodesD.append(i)
        else:
            print(f"Unknown id {cols[i]}")

    dists = [[0 for i in range(n)] for j in range(n)]
    distsF = [[0 for i in range(n)] for j in range(n)]

    for i in range(len(cols)):
        ds = list(df_sheet_d[cols[i]].dropna())
        for j in range(n):
            dists[i][j] = round(float(ds[j]))
            distsF[i][j] = float(ds[j])

    return distsF,dists,cols,fnodesD,schoolsD,splittersD




def main():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument('--distance-data', '-dd', default='../../data/itu/fibernodes/bw_sample2.xlsx',help='csv file with pairwise distances')#, required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--num-nodes", '-nfn', type=int, default=2, help='number of fiber nodes')
    optional.add_argument("--num-schools", '-nsc', type=int, default=4, help='number of schools')
    optional.add_argument("--num-splitters", '-nsp', type=int, default=7, help='number of splitters')
    optional.add_argument("--splitter-channels", '-sc', type=int, default=3, help='splitter channels')
    optional.add_argument("--max-distance", '-md', type=int, default=2000000, help='max distance for fiber link from school to fiber node')
    optional.add_argument("--node-channels", '-nc', type=int, default=3, help='max channels in nodes')
    optional.add_argument("--max-splitters", '-ms', type=int, default=8, help='max number of splitters per node')
    optional.add_argument("--node-cost", '-fnc', type=int, default=0, help='cost of running a node')
    optional.add_argument("--meter-cost", '-mc', type=int, default=1, help='cost per meter of fiber')
    optional.add_argument("--num-workers", '-nw', type=int, default=8, help='parallelism for ortools solver')
    optional.add_argument("--search-log", '-sl', type=int, default=1, help='output search log from ortools solver (1-on, 0-off)')
    optional.add_argument("--add-strategy", '-as', type=int, default=0,
                          help='adds a search strategy (1-on, 0-off)')
    optional.add_argument("--do-hints", '-dh', type=int, default=1,
                          help='adds hints coming from MST (1-on, 0-off)')
    optional.add_argument("--use-upbound", '-ub', type=int, default=0,
                          help='uses the cost from MST solution if feasible (1-on, 0-off)')
    optional.add_argument("--school2splitter", '-s2p', type=int, default=0,
                          help='force each school to connect to one different splitter (3-on and 2, 2-connect to the closest splitter, 1-on, 0-off)')
    optional.add_argument("--solver-search", '-ss', type=int, default=0,
                          help='set solver search (2-portfolio, 1-fixed, 0-auto)')
    optional.add_argument("--time-limit", '-tl', type=int, default=120,
                          help='time limit in seconds')
    optional.add_argument("--output-file", '-ofn', default='./result_satv2.gpickle')
    args = parser.parse_args()

    distancesF,distances,cols,fnodesD,schoolsD,splittersD = readExcelDistances(args.distance_data)

    print("Data read")
    nfn = len(fnodesD)
    nsc = len(schoolsD)
    nsp = len(splittersD)
    n = nfn+nsc+nsp + 1
    print(nfn,nsc,nsp)
    ofn = args.output_file
    metanode = [n-1]

    schoolsAndsplittersD = schoolsD + splittersD
    fnodesAndsplittersD = fnodesD + splittersD

    model = cp_model.CpModel()

    ####Search Variable#####
    P = {}
    A = {}

    #Metanode
    for fn in fnodesD:
        P[(fn, n - 1)] = model.NewBoolVar(f"P_{fn}_{n - 1}")
        A[(fn, n - 1)] = model.NewBoolVar(f"A_{fn}_{n - 1}")

    for sp in splittersD:
        A[(sp,n-1)] = model.NewBoolVar(f"A_{sp}_{n-1}")

    for sc in schoolsD:
        A[(sc, n - 1)] = model.NewBoolVar(f"A_{sc}_{n - 1}")

    #nodes
    for fn in fnodesD:
        for sp in splittersD:
            P[(sp,fn)] = model.NewBoolVar(f"P_{sp}_{fn}")
            A[(sp,fn)] = model.NewBoolVar(f"A_{sp}_{fn}")

        for sc in schoolsD:
            A[(sc, fn)] = model.NewBoolVar(f"A_{sc}_{fn}")

    #splitters
    for sp in splittersD:
        for sp2 in splittersD:
            if sp!=sp2:
                P[(sp2, sp)] = model.NewBoolVar(f"P_{sp2}_{sp}")
                A[(sp2, sp)] = model.NewBoolVar(f"A_{sp2}_{sp}")

        for sc in schoolsD:
            P[(sc, sp)] = model.NewBoolVar(f"P_{sc}_{sp}")
            A[(sc, sp)] = model.NewBoolVar(f"A_{sc}_{sp}")

    #####Helper variables#####
    active = [model.NewBoolVar(f"Act_{i}") for i in range(len(fnodesD))] #whether a fiber node is active or not

    ####Constraints####

    # ## Limit in the links coming out of the node
    for i in fnodesD:
        model.Add(sum(P[(j,i)] for j in splittersD)<=args.node_channels)

    ## Limit the number of links coming out of a splitter
    for i in splittersD:
        model.Add(sum(P[(j,i)] for j in schoolsAndsplittersD if i!=j)<=args.splitter_channels)

    #Limit the length of the links
    for i in schoolsD:
        for j in fnodesD:
            model.Add(A[(i,j)]*distances[j][i]<=args.max_distance)

    ##Limit the capacity of the node
    for i in fnodesD:
        model.Add(sum(A[(j,i)] for j in splittersD)<=args.max_splitters)
    #do the same for splitters
    for i in splittersD:
        model.Add(sum(A[(j,i)] for j in splittersD if i!=j)<=args.max_splitters)

    ##Schools must connect to a splitter
    for i in schoolsD:
        model.Add(sum(P[(i,j)] for j in splittersD) == 1)

    #####STeiner Constraints ######

    #No node has 2 parents
    for i in splittersD:
        model.Add(sum(P[(i,j)] for j in fnodesAndsplittersD if i!=j)<=1)
    for i in schoolsD:
        model.Add(sum(P[(i, j)] for j in splittersD) <= 1)

    #if node has a child, then it has a parent
    for i in fnodesD:
        aux = model.NewBoolVar('aux')
        model.Add(sum(P[(j,i)] for j in splittersD) > 0).OnlyEnforceIf(aux)
        model.Add(sum(P[(j,i)] for j in splittersD) == 0).OnlyEnforceIf(aux.Not())
        model.Add(aux==P[(i,n-1)])

    for i in splittersD:
        aux = model.NewBoolVar('aux')
        model.Add(sum(P[(j,i)] for j in schoolsAndsplittersD if i!=j)>0).OnlyEnforceIf(aux)
        model.Add(sum(P[(j,i)] for j in schoolsAndsplittersD if i!=j) == 0).OnlyEnforceIf(aux.Not())
        aux2 = model.NewBoolVar('aux2')
        model.Add(sum(P[(i,j)] for j in fnodesAndsplittersD if i!=j)==1).OnlyEnforceIf(aux2)
        model.Add(sum(P[(i,j)] for j in fnodesAndsplittersD if i!=j) == 0).OnlyEnforceIf(aux2.Not())
        model.Add(aux==aux2)

    #if node j is the parent of i then it is also an ancestor of i
    for i in schoolsD:
        for j in splittersD:
            model.AddImplication(P[(i,j)], A[(i,j)])

    for i in splittersD:
        for j in fnodesAndsplittersD:
            if i!=j:
                model.AddImplication(P[(i,j)],A[(i,j)])

    for i in fnodesD:
        model.AddImplication(P[(i,n-1)],A[(i,n-1)])

    #Transitivity constraint for ancestor
    for i in range(n):
        for j in range(n):
            if (i,j) in A:
                for k in range(n):
                    if (i,k) in A and (j,k) in A:
                        model.AddBoolOr([A[(i,j)].Not(), A[(j,k)].Not(), A[(i,k)]])

    #if node j is an ancestor of i, then i cannot be an ancestor of j
    for i in range(n):
        for j in range(n):
            if (i,j) in A and (j,i) in A:
                model.AddImplication(A[(i,j)],A[(j,i)].Not())

    #metanode has to be ancestor to all schools
    for i in schoolsD:
        model.Add(A[(i,n-1)]==True)

    ###################

    #if a splitter is connected to a fiber node, it can either connect to a school
    #or connect to more than 1 other node
    for i in splittersD:
        aux = model.NewBoolVar('aux')
        model.Add(sum(P[(i,j)] for j in fnodesD)==1).OnlyEnforceIf(aux)
        model.Add(sum(P[(i,j)] for j in fnodesD) < 1).OnlyEnforceIf(aux.Not())
        aux2 = model.NewBoolVar('aux2')
        model.Add(sum(P[(j,i)] for j in schoolsD) > 0).OnlyEnforceIf(aux2)
        model.Add(sum(P[(j,i)] for j in schoolsD) == 0).OnlyEnforceIf(aux2.Not())
        aux3 = model.NewBoolVar('aux3')
        model.Add(sum(P[(j,i)] for j in schoolsAndsplittersD if i!=j) > 1).OnlyEnforceIf(aux3)
        model.Add(sum(P[(j,i)] for j in schoolsAndsplittersD if i!=j) <= 1).OnlyEnforceIf(aux3.Not())
        model.AddBoolOr([aux.Not(),aux2,aux3])

    # if a splitter is connected to a splitter, it can either connect to a school
    # or connect to more than 1 other node
    for i in splittersD:
        aux = model.NewBoolVar('aux')
        model.Add(sum(P[(i,j)] for j in splittersD if i!=j) == 1).OnlyEnforceIf(aux)
        model.Add(sum(P[(i,j)] for j in splittersD if i!=j) < 1).OnlyEnforceIf(aux.Not())
        aux2 = model.NewBoolVar('aux2')
        model.Add(sum(P[(j,i)] for j in schoolsD) > 0).OnlyEnforceIf(aux2)
        model.Add(sum(P[(j,i)] for j in schoolsD) == 0).OnlyEnforceIf(aux2.Not())
        aux3 = model.NewBoolVar('aux3')
        model.Add(sum(P[(j,i)] for j in schoolsAndsplittersD if i!=j) > 1).OnlyEnforceIf(aux3)
        model.Add(sum(P[(j,i)] for j in schoolsAndsplittersD if i!=j) <= 1).OnlyEnforceIf(aux3.Not())
        model.AddBoolOr([aux.Not(), aux2, aux3])

    # A fiber node cannot be a leave
    for i in fnodesD:
        aux2 = model.NewBoolVar('aux2')
        model.Add(sum(P[(j,i)] for j in splittersD) == 0).OnlyEnforceIf(aux2)
        model.Add(sum(P[(j,i)] for j in splittersD) > 0).OnlyEnforceIf(aux2.Not())
        model.AddImplication(P[(i,n-1)], aux2.Not())

    # A splitter cannot be a leave
    for i in splittersD:
        aux = model.NewBoolVar('aux')
        model.Add(sum(P[(i,j)] for j in fnodesAndsplittersD if i!=j) == 1).OnlyEnforceIf(aux)
        model.Add(sum(P[(i,j)] for j in fnodesAndsplittersD if i!=j) < 1).OnlyEnforceIf(aux.Not())
        aux2 = model.NewBoolVar('aux2')
        model.Add(sum(P[(j,i)] for j in schoolsAndsplittersD if i!=j) == 0).OnlyEnforceIf(aux2)
        model.Add(sum(P[(j,i)] for j in schoolsAndsplittersD if i!=j) > 0).OnlyEnforceIf(aux2.Not())
        model.AddImplication(aux, aux2.Not())

    #######################

    #Make splitters connect to atmost one school
    if args.school2splitter==1:
        for i in splittersD:
            model.Add(sum(P[(j,i)] for j in schoolsD)<=1)

        #Make it the closest
    elif args.school2splitter == 2:
        for i in schoolsD:
            aux = [distances[i][j] for j in splittersD]
            minidx = np.argmin(aux)
            model.Add(P[(i,splittersD[minidx])]==True)
    elif args.school2splitter == 3:
        for i in splittersD:
            model.Add(sum(P[(j,i)] for j in schoolsD)<=1)
        for i in schoolsD:
            aux = [distances[i][j] for j in splittersD]
            minidx = np.argmin(aux)
            model.Add(P[(i,splittersD[minidx])]==True)

    #############€xtras#############

    #If no child no ancestor
    for i in fnodesD:
        aux = model.NewBoolVar('aux')
        model.Add(sum(P[(j,i)] for j in splittersD)==0).OnlyEnforceIf(aux)
        model.Add(sum(P[(j, i)] for j in splittersD) > 0).OnlyEnforceIf(aux.Not())
        aux2 = model.NewBoolVar('aux2')
        model.Add(sum(A[(j, i)] for j in schoolsAndsplittersD) == 0).OnlyEnforceIf(aux2)
        model.Add(sum(A[(j, i)] for j in schoolsAndsplittersD) > 0).OnlyEnforceIf(aux2.Not())
        model.Add(aux==aux2)

    for i in splittersD:
        aux = model.NewBoolVar('aux')
        model.Add(sum(P[(j, i)] for j in schoolsAndsplittersD if i!=j) == 0).OnlyEnforceIf(aux)
        model.Add(sum(P[(j, i)] for j in schoolsAndsplittersD if i!=j) > 0).OnlyEnforceIf(aux.Not())
        aux2 = model.NewBoolVar('aux2')
        model.Add(sum(A[(j, i)] for j in schoolsAndsplittersD if i!=j) == 0).OnlyEnforceIf(aux2)
        model.Add(sum(A[(j, i)] for j in schoolsAndsplittersD if i!=j) > 0).OnlyEnforceIf(aux2.Not())
        model.Add(aux == aux2)

    # schools can only have one ancestor fiber node
    for i in schoolsD:
        model.Add(sum(A[(i, j)] for j in fnodesD) <= 1)

    ####Channeling Constraints####

    for pos in range(len(fnodesD)):
        i = fnodesD[pos]
        model.Add(active[pos]==P[(i,n-1)])


    ######Cost function
    cost = model.NewIntVar(0,100000000,"Cost")
    aux = []
    for i in range(n-1):
        for j in range(n-1):
            if (j,i) in P:
                auxInt = model.NewIntVar(0,100000000,'auxInt')
                model.Add(auxInt==args.meter_cost*distances[i][j]).OnlyEnforceIf(P[(j,i)])
                model.Add(auxInt == 0).OnlyEnforceIf(P[(j,i)].Not())
                aux.append(auxInt)
    auxInt2 = model.NewIntVar(0,100000000,'auxInt2')
    model.Add(auxInt2==sum(active[i] for i in range(len(fnodesD)))*args.node_cost)
    aux.append(auxInt2)
    model.Add(cost==sum(aux[i] for i in range(len(aux))))
    #cost = sum(dists*connections)*args.meter_cost + sum(active)*args.node_cost

    searchVars,searchVars2,upbound = doMST(distancesF, distances, fnodesD, schoolsD, splittersD, P, A, active,n, args.splitter_channels,args.node_channels,args.max_splitters,args.max_distance)

    if args.do_hints==1:
        if upbound >= 0:
            print('UBound ',upbound)
            for var in searchVars:
                print(var)
                model.AddHint(var,True)

            for var in searchVars2:
                model.AddHint(var,False)

    if args.use_upbound==1:
        if upbound>=0:
            print('Use upbound',upbound)
            model.Add(cost<=upbound)

    model.Minimize(cost)

    ###Decision strategy
    if args.add_strategy==1:
        model.AddDecisionStrategy(searchVars, cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)
        model.AddDecisionStrategy(searchVars2, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
        if args.solver_search==1:
            solver.parameters.search_branching = cp_model.FIXED_SEARCH
        elif args.solver_search==2:
            solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

    solver = cp_model.CpSolver()
    print('Solve')

    if args.search_log==1:
        solver.parameters.log_search_progress = True  # SCALED BACK OBJECTIVE PROGRESS
    else:
        solver.parameters.log_search_progress = False

    solver.parameters.max_time_in_seconds = args.time_limit
    solver.parameters.num_search_workers = args.num_workers
    solution_printer = SolutionPrinter(active,P,fnodesD,cols,distances)
    status = solver.Solve(model, solution_printer)
    print(status)
    print('objective = ', solver.ObjectiveValue())
    #########Convert to graph##########
    G = nx.Graph()
    for i in range(n-1):
        for j in range(n-1):
            if (i,j) in P:
                if solver.Value(P[(i,j)]) == 1:
                    G.add_edge(cols[i], cols[j])

    nx.write_gpickle(G, ofn)



if __name__ == "__main__":
    main()