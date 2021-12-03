# Tool to generate artificial data for performance analysis

import numpy as np
import pandas as pd
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx import convert_matrix

def calc_liaisons_count(liaison_df):
    return liaison_df.to_numpy().sum()/2

def calc_MWF(mw_df):
    return mw_df.to_numpy().sum() / (len(mw_df.index)*len(mw_df.index))

def generate_liaison_xlsx_file(product_name, n_parts, n_liaisons):
    path = '../data/generated/' + product_name + '_Liaisons.xlsx'
    liaison_df = generate_liaison_df(n_parts, n_liaisons)
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    liaison_df.to_excel(writer, sheet_name='Liaison Matrix', index=False, header=False)
    writer.save()

def generate_liaison_csv_file(product_name, n_parts, p_liaisons):
    path = '../data/generated/doe_2/' + product_name + '_Liaisons.csv'
    liaison_df = generate_liaison_df(n_parts, p_liaisons)
    liaison_df.to_csv(path)

def generate_liaison_df(n_parts, p_liaisons):
    #p = n_liaisons * 2 / (n_parts * n_parts - n_parts)
    p = p_liaisons
    g_liaisons = erdos_renyi_graph(n_parts, p)
    liaison_numpy = convert_matrix.to_numpy_matrix(g_liaisons)
    liaison_numpy = liaison_numpy.astype(int)
    liaison_df = pd.DataFrame(liaison_numpy)
    return liaison_df

def generate_mw_xlsx_file(product_name, n_parts, MWF):
    path = '../data/generated/' + product_name + '_Moving wedge.xlsx'
    mw_df_x = generate_mw_matrix(n_parts, MWF)
    mw_df_y = generate_mw_matrix(n_parts, MWF)
    mw_df_z = generate_mw_matrix(n_parts, MWF)
    save_as_excel(path, mw_df_x, mw_df_y, mw_df_z)

def generate_mw_csv_files(product_name, n_parts, MWF):
    path_x = '../data/generated/doe_2/' + product_name + '_Moving wedge_x.csv'
    path_y = '../data/generated/doe_2/' + product_name + '_Moving wedge_y.csv'
    path_z = '../data/generated/doe_2/' + product_name + '_Moving wedge_z.csv'
    mw_df_x = generate_mw_matrix(n_parts, MWF)
    mw_df_y = generate_mw_matrix(n_parts, MWF)
    mw_df_z = generate_mw_matrix(n_parts, MWF)
    mw_df_x.to_csv(path_x)
    mw_df_y.to_csv(path_y)
    mw_df_z.to_csv(path_z)

def generate_mw_matrix(n_parts, MWF):
    #mw_matrix = np.zeros((n_parts, n_parts))
    mw_matrix = np.random.choice([0,1], size=(n_parts, n_parts), p=[1-MWF, MWF])
    mw_df = pd.DataFrame(mw_matrix)
    return mw_df

def save_as_excel(xlsx_path, mw_df_x, mw_df_y, mw_df_z):
    writer = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')
    mw_df_x.to_excel(writer, sheet_name='MW_x', index=False, header=False)
    mw_df_y.to_excel(writer, sheet_name='MW_y', index=False, header=False)
    mw_df_z.to_excel(writer, sheet_name='MW_z', index=False, header=False)
    writer.save()

def generate_files(product_name, n_parts, p_liaisons, mwf):
    generate_liaison_csv_file(product_name=product_name, n_parts=n_parts, p_liaisons=p_liaisons)
    generate_mw_csv_files(product_name=product_name, n_parts=n_parts, MWF=mwf)

if __name__ == '__main__':
    #DOE2 Box-Behnken
    generate_files('exp_1', 6, 0.2, 0.65)
    generate_files('exp_2', 12, 0.2, 0.65)
    generate_files('exp_3', 6, 0.6, 0.65)
    generate_files('exp_4', 12, 0.6, 0.65)
    generate_files('exp_5', 6, 0.4, 0.5)
    generate_files('exp_6', 12, 0.4, 0.5)
    generate_files('exp_7', 6, 0.4, 0.8)
    generate_files('exp_8', 12, 0.4, 0.8)
    generate_files('exp_9', 9, 0.2, 0.5)
    generate_files('exp_10', 9, 0.6, 0.7)
    generate_files('exp_11', 9, 0.2, 0.8)
    generate_files('exp_12', 9, 0.6, 0.8)
    generate_files('exp_13', 9, 0.4, 0.65)
    generate_files('exp_14', 9, 0.4, 0.65)
    generate_files('exp_15', 9, 0.4, 0.65)

