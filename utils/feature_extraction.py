import pandas as pd
import ast

def get_one_value(string, name='vrh'):
    try:
       d = ast.literal_eval(string)
       return d[name]
    except Exception as e:
        pass
    

def get_feature_csv(df, feature_name):
    exceptions = ['bulk_modulus', 'shear_modulus']
    if feature_name in exceptions:
        temp_arr = df[feature_name].apply(lambda x: get_one_value(x))
        dict = {
            'material_id': df['material_id'],
            'formula_pretty': df['formula_pretty'],
            'property_value': temp_arr
        }
        df_new = pd.DataFrame(dict)
    else:
        columns = ['material_id', 'formula_pretty', feature_name]
        df_new = df[columns]
        df_new.rename(columns={feature_name: 'property_value'})
    
    return df_new
        

        

if __name__ == "__main__":
    df = pd.read_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/material_ids_with_prop.csv')
    df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
    df_new = get_feature_csv(df, 'bulk_modulus')
    df_new.to_csv('/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/id_prop.csv')
