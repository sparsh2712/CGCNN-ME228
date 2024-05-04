import pandas as pd
import ast

# Define a function to extract a specific value from a stringified dictionary
def get_one_value(string, name='vrh'):
    try:
       d = ast.literal_eval(string)  # Convert string to dictionary
       return d[name]  # Return the value corresponding to the specified key
    except Exception as e:
        pass  # If an error occurs during conversion, do nothing

# Define a function to extract a specific feature from a DataFrame
def get_feature_csv(df, feature_name):
    exceptions = ['bulk_modulus', 'shear_modulus']
    if feature_name in exceptions:
        # If the feature is an exception (bulk_modulus or shear_modulus),
        # extract the value using the get_one_value function
        temp_arr = df[feature_name].apply(lambda x: get_one_value(x))
        # Create a dictionary with material ID, formula, and property value
        dict = {
            'material_id': df['material_id'],
            'formula_pretty': df['formula_pretty'],
            'property_value': temp_arr
        }
        # Create a new DataFrame from the dictionary
        df_new = pd.DataFrame(dict)
    else:
        # If the feature is not an exception, simply extract it along with material ID and formula
        columns = ['material_id', 'formula_pretty', feature_name]
        df_new = df[columns]
        df_new.rename(columns={feature_name: 'property_value'})  # Rename the feature column
    
    return df_new

# Main block of code
if __name__ == "__main__":
    # Read the CSV file into a DataFrame
    df = pd.read_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/material_ids_with_prop.csv')
    df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)  # Rename the 'Unnamed: 0' column to 'Index'
    # Extract the desired feature and create a new DataFrame
    df_new = get_feature_csv(df, 'bulk_modulus')
    # Write the new DataFrame to a new CSV file
    df_new.to_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/id_prop.csv')
