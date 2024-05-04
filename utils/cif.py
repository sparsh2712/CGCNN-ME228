import pandas as pd 
import numpy as np 
from mp_api.client import MPRester

m = MPRester(api_key="PULAwCzMQFzgXRhElkO1T4BoM6mQAjnO")

cif_list = pd.read_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/mp-ids-3402.csv', header=None, names=['material-id'])
cif_list['material-id'] = cif_list['material-id'].astype(str)


for material_id in cif_list['material-id']:
    try:
        structure = m.get_structure_by_material_id(material_id)
        cif_data = structure.to(fmt="cif")
        with open(f'/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/utils/cif_files/{material_id}.cif', 'w') as f:
            f.write(cif_data)
    except Exception as e:
        with open('cif_not_found.txt', 'a') as t:
            t.write(f'{material_id}\n')
        print(f"Error occurred for material ID {material_id}: {e}")
