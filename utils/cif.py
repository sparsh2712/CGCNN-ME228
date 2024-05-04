import pandas as pd
import numpy as np
from mp_api.client import MPRester

def fetch_cif_files(api_key, cif_list_path, output_directory):
    """
    Fetches CIF files for materials listed in the provided CSV file.

    Args:
    - api_key (str): The API key for accessing the Materials Project API.
    - cif_list_path (str): Path to the CSV file containing material IDs.
    - output_directory (str): Directory where CIF files will be saved.

    Returns:
    - None
    """

    # Initialize MPRester with the provided API key
    m = MPRester(api_key=api_key)

    # Read the CSV file containing material IDs into a DataFrame
    cif_list = pd.read_csv(cif_list_path, header=None, names=['material-id'])

    # Convert material IDs to string format
    cif_list['material-id'] = cif_list['material-id'].astype(str)

    # Iterate over each material ID in the DataFrame
    for material_id in cif_list['material-id']:
        try:
            # Get the structure object for the material ID from Materials Project
            structure = m.get_structure_by_material_id(material_id)
            
            # Convert the structure object to CIF format
            cif_data = structure.to(fmt="cif")

            # Write the CIF data to a file named after the material ID in the output directory
            with open(f'{output_directory}/{material_id}.cif', 'w') as f:
                f.write(cif_data)
        except Exception as e:
            # If an error occurs, log the material ID to a file and print the error message
            with open('cif_not_found.txt', 'a') as t:
                t.write(f'{material_id}\n')
            print(f"Error occurred for material ID {material_id}: {e}")

if __name__ == '__main__':
    api_key = "PULAwCzMQFzgXRhElkO1T4BoM6mQAjnO"
    cif_list_path = '/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/mp-ids-3402.csv'
    output_directory = '/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/utils/cif_files'

    fetch_cif_files(api_key, cif_list_path, output_directory)
