import pandas as pd
from mp_api.client import MPRester

# Read the CSV file containing material IDs into a DataFrame
df = pd.read_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/mp-ids-3402.csv', header=None)

# Extract the material IDs from the DataFrame and convert them to a list
id_list = df[0].to_list()

# Initialize an instance of MPRester with the provided API key
with MPRester("PULAwCzMQFzgXRhElkO1T4BoM6mQAjnO") as mpr:
    # Search for materials summaries based on the list of material IDs
    docs = mpr.materials.summary.search(material_ids=id_list)

# Initialize an empty DataFrame with specified column names
df = pd.DataFrame(columns=['material_id', 'formula_pretty', 'energy_per_atom', 'formation_energy_per_atom', 'band_gap', 'efermi', 'is_metal', 'bulk_modulus', 'shear_modulus', 'homogeneous_poisson'])

# Iterate over each material summary document retrieved from Materials Project
i = 0
for doc in docs:
    # Populate the DataFrame with properties extracted from each document
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
    i += 1

# Write the DataFrame containing material properties to a new CSV file
df.to_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/material_ids_with_prop_new.csv')
