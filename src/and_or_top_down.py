# -*- coding: utf-8 -*-
"""
Script to generate AND/OR graph from moving wedge (MW) and liaisons matrix.
"""

import pandas as pd
import hypernetx as hnx
import pickle
import time


def run_experiment(directory, product_name, restrict_nonstat_size, max_nonstat_parts):
    path = directory + product_name
    liaisons_file = path + "_Liaisons.csv"
    mw_path_x = path + "_Moving wedge_x.csv"
    mw_path_y = path + "_Moving wedge_y.csv"
    mw_path_z = path + "_Moving wedge_z.csv"

    # Initial settings
    global base_part_id
    base_part_id = 1

    # Read liaisons matrix
    liaison_df = pd.read_csv(liaisons_file, index_col=0)
    '''
    liaison_xl_file = pd.ExcelFile(liaisons_file)
    liaison_df = liaison_xl_file.parse('Liaison Matrix', header=None, index_col=None)
    liaison_df.fillna(value=0, inplace=True)
    liaison_df = liaison_df.astype('int32')
    '''
    # Read MW
    mw_dfs = read_mw_data_csv(mw_path_x, mw_path_y, mw_path_z)


    # Initialise product as a list of all part indices
    product_list = list(range(1, len(liaison_df) + 1))

    # List of graph nodes (subassemblies)
    nodes = []

    # Start recursive AND/OR graph generation
    start = time.perf_counter()
    and_or(product_list, liaison_df, mw_dfs, nodes, restrict_nonstat_size, max_nonstat_parts)
    end = time.perf_counter()
    runtime = end - start

    H = hnx.Hypergraph(hyper_str)
    node_count = len(H.nodes)
    edge_count = len(H.edges)

    with open('../out/' + product_name + '_AND_OR.pickle', 'wb') as f:
        pickle.dump(hyper, f)

    return {"exp:": product_name, "runtime": runtime, "node_count": node_count, "edge_count": edge_count}

def read_mw_data_excel(mw_path):
    mw_xl_file = pd.ExcelFile(mw_path)
    mw_dfs = {sheet_name: mw_xl_file.parse(sheet_name, header=None, index_col=None).astype('int32')
              for sheet_name in mw_xl_file.sheet_names}
    return mw_dfs

def read_mw_data_csv(mw_path_x, mw_path_y, mw_path_z):
    mw_x_df = pd.read_csv(mw_path_x)
    mw_y_df = pd.read_csv(mw_path_y)
    mw_z_df = pd.read_csv(mw_path_z)
    return {'MW_x': mw_x_df, 'MW_y': mw_y_df, 'MW_z': mw_z_df}

# yields all subdivisions of a set in two groups
def bin_partitions(lst):
    n = len(lst)
    for i in range(1, 2 ** (n - 1)):
        bitmask = bin(i).replace('0b', '').rjust(n, '0')
        sub0 = [lst[j] for j in range(n) if bitmask[j] == '0']
        sub1 = [lst[j] for j in range(n) if bitmask[j] == '1']
        prt = [sub0, sub1]
        yield prt

def and_or(prod, liaison_df, mw_dfs, nodes, restrict_nonstat_size, max_nonstat_parts):
    global num_edges
    global hyper, hyper_str
    dirs = {0: 'MW_x',
            1: 'MW_y',
            2: 'MW_z'}
    prts = bin_partitions(prod)

    # Remember the parts/subassemblies not to be put apart
    for prt in prts:
        print(prt)
        l1 = len(prt[0])
        l2 = len(prt[1])
        if restrict_nonstat_size:
            # check whether the size of subassembly without base part is admissible
            if is_stationary(prt[0]):
                if l2 > max_nonstat_parts:
                    continue
            elif is_stationary(prt[1]):
                if l1 > max_nonstat_parts:
                    continue
            else:
                if l1 + l2 > max_nonstat_parts:
                    continue
        # check both subassemblies for stability
        # if one of them is unstable, skip current partition
        if is_stable(prt[0], liaison_df) == False:
            continue
        if is_stable(prt[1], liaison_df) == False:
            continue
        # check whether disassembly is possible by checking for
        # collision-free assembly paths of one of two subsets along all axes
        assy_dirs = []
        for i in range(6):
            checksum = 0
            if i < 3:
                mat = mw_dfs[dirs[i]].to_numpy()
                for j in prt[0]:
                    for k in prt[1]:
                        checksum = checksum + mat[j - 1][k - 1]
            else:
                mat = mw_dfs[dirs[i - 3]].to_numpy()
                for j in prt[0]:
                    for k in prt[1]:
                        checksum = checksum + mat[k - 1][j - 1]
            if checksum == l1 * l2:
                assy_dirs.append(i)
        if len(assy_dirs) > 0:
            # print(str(prt[0]) + ' can be assembled to ' + str(prt[1]) + ' along directions ' + str(assy_dirs))
            num_edges = num_edges + 1
            # Save hyperedge
            hyper[num_edges] = (prod, prt[0], prt[1])
            hyper_str[num_edges] = (str(prod), str(prt[0]), str(prt[1]))
            # Create and add 2 nodes (assembly states) and an edge for DGFAS
            'source = str(prt[0]) + str(prt[1]) + fixed_tmp'
            'target = str(prod) + fixed_tmp'
            # sort subassemblies by first element'
            'source = sort_state(source)'
            'target = sort_state(target)'
            'dgfas.add_edge(source, target)'
            # Continue AND/OR procedure for unvisited subassemblies
            if prt[0] not in nodes:
                nodes.append(prt[0])
                and_or(prt[0], liaison_df, mw_dfs, nodes, restrict_nonstat_size, max_nonstat_parts)
            if prt[1] not in nodes:
                nodes.append(prt[1])
                and_or(prt[1], liaison_df, mw_dfs, nodes, restrict_nonstat_size, max_nonstat_parts)

def generate_dgfas(prod, liaison_df, mw_dfs, fixed):
    global dgfas
    dirs = {0: 'MW_x',
            1: 'MW_y',
            2: 'MW_z'}
    prts = bin_partitions(prod)
    # Remember the parts/subassemblies not to be put apart
    fixed_tmp = fixed
    for prt in prts:
        l1 = len(prt[0])
        l2 = len(prt[1])
        # check both subassemblies for stability
        # if one of them is unstable, skip current partition
        if is_stable(prt[0], liaison_df) == False:
            continue
        if is_stable(prt[1], liaison_df) == False:
            continue
        # check whether disassembly is possible by checking for
        # collision-free assembly paths of one of two subsets along all axes
        assy_dirs = []
        for i in range(6):
            checksum = 0
            if i < 3:
                mat = mw_dfs[dirs[i]].to_numpy()
                for j in prt[0]:
                    for k in prt[1]:
                        checksum = checksum + mat[j - 1][k - 1]
            else:
                mat = mw_dfs[dirs[i - 3]].to_numpy()
                for j in prt[0]:
                    for k in prt[1]:
                        checksum = checksum + mat[k - 1][j - 1]
            if checksum == l1 * l2:
                assy_dirs.append(i)
        if len(assy_dirs) > 0:
            # Create and add 2 nodes (assembly states) and an edge for DGFAS
            source = str(prt[0]) + str(prt[1]) + fixed_tmp
            target = str(prod) + fixed_tmp
            # sort subassemblies by first element'
            source = sort_state(source)
            target = sort_state(target)
            dgfas.add_edge(source, target)
            # Continue DGFAS procedure for unvisited subassemblies
            fixed = str(prt[1]) + fixed_tmp
            generate_dgfas(prt[0], liaison_df, mw_dfs, fixed)

            fixed = str(prt[0]) + fixed_tmp
            generate_dgfas(prt[1], liaison_df, mw_dfs, fixed)

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

def is_stationary(prt):
    global base_part_id
    if base_part_id in prt:
        return True
    else:
        return False

def sort_state(state):
    tokens = state.split('][')
    tokens[0] = tokens[0].strip('[')
    tokens[-1] = tokens[-1].strip(']')
    lsts = []
    for t in tokens:
        parts = t.split(', ')
        lst = []
        for s in parts:
            lst.append(int(s))
        lsts.append(lst)
    lsts.sort()
    ans = ''
    for l in lsts:
        ans = ans + str(l)
    return ans

def count_subassemblies(state):
    tokens = state.split('][')
    return len(tokens)

def distribute_dgfas(dgfas, lev, pos_labels):
    levels = []
    pos2 = {}
    counter = 0
    for l in range(lev):
        levels.append([])
    for node in list(dgfas.nodes):
        sa = count_subassemblies(str(node))
        levels[lev - sa].append(str(node))
    dx = 1.0
    dy = 1.0
    dy_label = 0.15
    y = -len(levels) / 2 * dy
    for level in levels:
        x = -len(level) / 2 * dx
        for state in level:
            pos2[state] = (x, y)
            pos_labels[state] = (x, y + dy_label * (-1) ** (counter % 2))
            x = x + dx
            counter = counter + 1
        y = y + dy
    return pos2

def calc_liaisons_count(liaison_df):
    return liaison_df.to_numpy().sum() / 2

def calc_MWF(mw_dfs):
    mw_entries_x = mw_dfs['MW_x'].to_numpy().sum()
    mw_entries_y = mw_dfs['MW_x'].to_numpy().sum()
    mw_entries_z = mw_dfs['MW_x'].to_numpy().sum()
    matrix_index_count = len(mw_dfs['MW_x'].index)
    matrix_size = matrix_index_count * matrix_index_count
    MWF = (mw_entries_x + mw_entries_y + mw_entries_z) / (3 * matrix_size)
    return MWF

if __name__ == '__main__':
    # AND/OR hypergraph as a dictionary
    num_edges = 0
    hyper = {}
    hyper_str = {}  # for visualization

    # MAIN SCRIPT

    directory = "../data/generated/doe_2/"

    for i in range(1, 16):
        result_df = pd.DataFrame()
        result = run_experiment(directory=directory, product_name='exp_' + str(i), restrict_nonstat_size=False, max_nonstat_parts=3)
        result_df = result_df.append(result, ignore_index=True)

        with open('../out/performance_test_results/doe_2_top_down_results.csv', 'a') as f:
            result_df.to_csv(f, header=False, index=False)
        del result_df
        print(i, ' done.')






