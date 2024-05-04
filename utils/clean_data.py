import pandas as pd

# Read the file containing material IDs that were not found during CIF fetching
with open('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/cif_not_found.txt', 'r') as f:
    # Create a list containing the material IDs (stripped of newline characters)
    id_list = [line.strip() for line in f]

# Read the CSV file containing material IDs and properties into a DataFrame
df = pd.read_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/material_ids_with_prop_new.csv')

# Print the total number of rows in the DataFrame
print(len(df))

# Filter the DataFrame to exclude rows with material IDs present in the id_list
filtered_df = df[~df['material_id'].isin(id_list)]

# Print the number of rows in the filtered DataFrame
print(len(filtered_df))

# Write the filtered DataFrame to a new CSV file
filtered_df.to_csv('clean_material_ids_with_prop.csv')
