import FlowCal
import pandas as pd
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight

p = dict()
for i in range(1, 7):
    p[i-1] = dict()
    for ch in ['G', 'K', 'L', 'M', 'N', 'O', 'P']:
        p[i-1][ch] = FlowCal.io.FCSData(f'/home/users/gnn/multiclass/folder1/Case{i}_' + ch + '.fcs')

for i in range(7, 29):
    p[i-1] = dict()
    for ch in ['G', 'K', 'L', 'M', 'N', 'O', 'P']:
        p[i-1][ch] = FlowCal.io.FCSData(f'/home/users/multiclass/folder2/Case{i}_' + ch + '.fcs')

p[29] = dict()
p[29]['G'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case30_G.csv', delimiter=";")[1:]
p[29]['K'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case30_K.csv', delimiter=";")[1:]
p[29]['L'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case30_L.csv', delimiter=";")[1:]
p[29]['M'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case30_M.csv', delimiter=";")[1:]
p[29]['N'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case30_N.csv', delimiter=";")[1:]
p[29]['O'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case30_O.csv', delimiter=";")[1:]
p[29]['P'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case30_P.csv', delimiter=";")[1:]

p[30] = dict()
p[30]['G'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case31_G.csv', delimiter=";")[1:]
p[30]['K'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case31_K.csv', delimiter=";")[1:]
p[30]['L'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case31_L.csv', delimiter=";")[1:]
p[30]['M'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case31_M.csv', delimiter=";")[1:]
p[30]['N'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case31_N.csv', delimiter=";")[1:]
p[30]['O'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case31_O.csv', delimiter=";")[1:]
p[30]['P'] = np.genfromtxt('/home/users/gnn/multiclass/folder2/Case31_P.csv', delimiter=";")[1:]

# Flat Dataset generation with columns O,N,G,P,K
column=('FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5', 'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB', 'FL10 INT_CD45-KO')
for i in range(30):
    df_G=pd.DataFrame(p[i]['G'],columns=column)
    df_K=pd.DataFrame(p[i]['K'],columns=column)
    df_N=pd.DataFrame(p[i]['N'],columns=column)
    df_O=pd.DataFrame(p[i]['O'],columns=column)
    df_P=pd.DataFrame(p[i]['P'],columns=column)

    df_O['label']=0
    df_N['label']=1
    df_G['label']=2
    df_P['label']=3
    df_K['label']=4

    directory = "Data_flat"
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.concat([df_O,df_N,df_G,df_P,df_K])

    df.to_csv(f"{directory}/Case_{i+1}.csv",index=False)

    # Compute class weights
    labels = df['label']
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print(f"Class weights for Case_{i+1}: {class_weights}")
