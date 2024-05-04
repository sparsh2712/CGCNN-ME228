import pandas as pd
from mp_api.client import MPRester

df = pd.read_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/mp-ids-3402.csv', header=None)
id_list = df[0].to_list()

with MPRester("PULAwCzMQFzgXRhElkO1T4BoM6mQAjnO") as mpr:
    docs = mpr.materials.summary.search(material_ids=id_list)

df = pd.DataFrame(columns=['material_id', 'formula_pretty', 'energy_per_atom', 'formation_energy_per_atom', 'band_gap', 'efermi', 'is_metal', 'bulk_modulus', 'shear_modulus', 'homogeneous_poisson'])

i = 0
for doc in docs:
    df.loc[i] =  {'material_id': doc.material_id, 
                  'formula_pretty': doc.formula_pretty,
                  'energy_per_atom': doc.energy_per_atom,
                  'formation_energy_per_atom':doc.formation_energy_per_atom, 
                  'band_gap':doc.band_gap, 
                  'efermi':doc.efermi, 
                  'is_metal':doc.is_metal,
                  'bulk_modulus': doc.bulk_modulus,
                  'shear_modulus':doc.shear_modulus,
                  'homogeneous_poisson':doc.homogeneous_poisson
                  }
    i+=1

df.to_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/material_ids_with_prop_new.csv')

