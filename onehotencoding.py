import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {'Employee id': [10, 20, 15, 25, 30],
        'Gender': ['M', 'F', 'F', 'M', 'F'],
        'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice'],
        }
df = pd.DataFrame(data)
print(f"Employee data : \n{df}")

categorical_columns = df.select_dtypes(include=['object','int']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
print(f"Categorical columns : {categorical_columns}")
onehotencoding=encoder.fit_transform(df[categorical_columns])
print(f"Categorical one-hot-encoding \n: {onehotencoding}")
one_hot_df = pd.DataFrame(onehotencoding, columns=encoder.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([df, one_hot_df], axis=1)

print(f"final data : \n{df_encoded.drop(categorical_columns, axis=1)}")

print("\n Sorted data \n",df_encoded.sort_values(by='Employee id'))
df_encoded.to_json('onehot.json',index=False)