import pandas as pd 

with open ('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/cif_not_found.txt', 'r') as f:
    id_list = [line.strip() for line in f]

df = pd.read_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/material_ids_with_prop_new.csv')
print(len(df))
filtered_df = df[~df['material_id'].isin(id_list)]
print(len(filtered_df))
filtered_df.to_csv('clean_material_ids_with_prop.csv')