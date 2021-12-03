# -*- coding: utf-8 -*-
"""
Script to generate AND/OR graph from moving wedge (MW) and liaisons matrix.
"""

import pandas as pd
import hypernetx as hnx
import time
from copy import deepcopy

def run_experiment(product_name, restrict_nonstat_size, max_nonstat_parts):
    liaisons_file = "../data/generated/doe_2/" + product_name + "_Liaisons.csv"
    mw_path_x = "../data/generated/doe_2/" + product_name + "_Moving wedge_x.csv"
    mw_path_y = "../data/generated/doe_2/" + product_name + "_Moving wedge_y.csv"
    mw_path_z = "../data/generated/doe_2/" + product_name + "_Moving wedge_z.csv"

    # ID of base part
    global base_part_id
    base_part_id = 1

    # Read files
    liaison_df = pd.read_csv(liaisons_file, index_col=0)
    mw_dfs = read_mw_data_csv(mw_path_x, mw_path_y, mw_path_z)

    global prohibited_sa
    prohibited_sa = []

    # Initialise product as a list of all part indices
    prod = list(range(1, len(liaison_df) + 1))

    # Start recursive AND/OR graph generation
    start = time.perf_counter()
    and_or(prod, liaison_df, mw_dfs, restrict_nonstat_size, max_nonstat_parts)
    end = time.perf_counter()
    runtime = end - start

    H = hnx.Hypergraph(hyper_str)
    node_count = len(H.nodes)
    edge_count = len(H.edges)

    #with open('../out/' + product_name + '_AND_OR_test.pickle', 'wb') as f:
    #    pickle.dump(hyper, f)
    return {"exp:": product_name, "runtime": runtime, "node_count": node_count, "edge_count": edge_count}

def and_or(prod, liaison_df, mw_dfs, restrict_nonstat_size, max_nonstat_parts):
    global dirs
    global num_edges
    global hyper, hyper_str
    #global restrict_nonstat_size, max_nonstat_parts
    # List of created subassemblies
    sa_list = [[p] for p in prod]
    # The smallest index of SA2 in sa_list (to avoid redundant operations)
    ind2_start = 0
    # Level is equal to the number of parts in SAs on this level
    for level in range(2, len(prod)+1):
        #print('======== Level %d ========' % level)
        num_iter = 0
        num_no_intersect = 0
        num_connect = 0
        num_free = 0
        num_new_sa = 0
        num_new_edges = 0
        temp_sa_list = deepcopy(sa_list)
        for ind1 in range(len(sa_list)):
            for ind2 in range(ind2_start, len(sa_list)):
                if ind1 < ind2:
                    num_iter += 1
                    sa1 = sa_list[ind1]
                    sa2 = sa_list[ind2]
                    l1 = len(sa1)
                    l2 = len(sa2)
                    if restrict_nonstat_size:
                        # check whether the size of subassembly without base part is admissible
                        if is_stationary(sa1):
                            if l2 > max_nonstat_parts:
                                continue
                        elif is_stationary(sa2):
                            if l1 > max_nonstat_parts:
                                continue
                        else:
                            if l1 + l2 > max_nonstat_parts:
                                continue
                    if l1 + l2 <= len(prod):
                        if not intersection(sa1, sa2):
                            num_no_intersect += 1
                            new_sa = sorted(sa1 + sa2)
                            # if new_sa is already in sa_list, then it was created
                            # on a lower level, don't add new operations
                            #if new_sa in sa_list:
                            #    continue
                            if connected_subassemblies(sa1, sa2, liaison_df):
                                num_connect += 1
                                if collision_free_assembly(sa1, sa2, mw_dfs):
                                    num_free += 1
                                    # Check whether this new SA creates any problems for
                                    # higher-level operations
                                    if not sa_prevents_future_assembly(new_sa, prod, mw_dfs):
                                        num_edges += 1
                                        num_new_edges += 1
                                        # Save hyperedge
                                        hyper[num_edges] = (new_sa, sa1, sa2)
                                        hyper_str[num_edges] = (str(new_sa), str(sa1), str(sa2))
                                        if new_sa not in temp_sa_list:
                                            num_new_sa += 1
                                            temp_sa_list.append(new_sa)

        ind2_start = len(sa_list)
        sa_list = deepcopy(temp_sa_list)

    # Beautify: sort key-value pairs by decreasing size of product subassembly
    # print('Reformatting AND/OR graph')
    sorted_edges = sorted(hyper.values(), key=lambda x: len(x[0]), reverse=True)
    hyper = {i: sorted_edges[i-1] for i in range(1, len(sorted_edges)+1)}

def read_mw_data_csv(mw_path_x, mw_path_y, mw_path_z):
    mw_x_df = pd.read_csv(mw_path_x)
    mw_y_df = pd.read_csv(mw_path_y)
    mw_z_df = pd.read_csv(mw_path_z)
    return {'MW_x': mw_x_df, 'MW_y': mw_y_df, 'MW_z': mw_z_df}


# Checks whether there is at least 1 liaison between two subassemblies.
# The subassemblies themselves are assumed to be connected.
def connected_subassemblies(sa1, sa2, liaison_df):
    '''
    prt = sa1 + sa2
    if is_stable(prt, liaison_df) == True:
        return True
    '''
    for p1 in sa1:
        for p2 in sa2:
            if liaison_df.iloc[p1-1,p2-1] == 1:
                return True
    return False

def is_stable(prt, liaison_df):
    # List of visited nodes
    vis = [False for x in range(len(prt))]
    # Submatrix of LG
    prt_idx = [p - 1 for p in prt]
    lg_df = liaison_df.iloc[prt_idx, prt_idx]
    lg = lg_df.to_numpy()
    # DFS to explore the graph from the first node
    dfs(lg, vis, 0)
    # All subassembly parts must be visited via liaisons
    for i in range(len(vis)):
        if vis[i] == False:
            return False
    return True

def dfs(lg, vis, v):
    if vis[v] == True:
        return
    vis[v] = True
    # for all neighbors u of v
    for u in [i for i in range(len(vis)) if lg[v][i] == 1]:
        if vis[u] == False:
            dfs(lg, vis, u)

# check whether disassembly is possible by checking for
# collision-free assembly paths of one of two subsets along all axes
def collision_free_assembly(sa1, sa2, mw_dfs):
    #global dirs
    dirs = {0: 'MW_x',
            1: 'MW_y',
            2: 'MW_z'}

    l1 = len(sa1)
    l2 = len(sa2)
    for i in range(6):
        checksum = 0
        if i < 3:
            mat = mw_dfs[dirs[i]].to_numpy()
            for j in sa1:
                for k in sa2:
                    checksum = checksum + mat[j - 1][k - 1]
        else:
            mat = mw_dfs[dirs[i - 3]].to_numpy()
            for j in sa1:
                for k in sa2:
                    checksum = checksum + mat[k - 1][j - 1]
        if checksum == l1 * l2:
            return True
    return False

def sa_prevents_future_assembly(sa, prod, mw_dfs):
    global prohibited_sa
    if sa in prohibited_sa:
        return True
    if sa == prod:
        return False
    for part in prod:
        if part not in sa:
            appended_part = [part]
            if not collision_free_assembly(sa, appended_part, mw_dfs):
                #print(str(appended_part))
                prohibited_sa.append(sa)
                return True
    return False

def product_subassembly_unused(sa, and_or_graph):
    for k in and_or_graph.keys():
        if and_or_graph[k][1] == sa or and_or_graph[k][2] == sa:
            return False
    return True                 
 
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def is_stationary(prt):
    global base_part_id
    if base_part_id in prt:
        return True
    else:
        return False              
  

if __name__ == '__main__':
    # AND/OR hypergraph as a dictionary
    num_edges = 0
    hyper = {}
    hyper_str = {}  # for visualization

    for i in range(1, 16):
        result_df = pd.DataFrame()
        result = run_experiment(product_name='exp_' + str(i), restrict_nonstat_size=False, max_nonstat_parts=3)
        result_df = result_df.append(result, ignore_index=True)

        with open('../out/performance_test_results/doe_2_bottom_up_results.csv', 'a') as f:
            result_df.to_csv(f, header=False, index=False)
        del result_df
        print(i, ' done.')

