from house_prices_regression_model.predict import make_prediction
import pandas as pd

columns = ['OverallQual','GrLivArea','TotalBsmtSF','CentralAir','FireplaceQu','BsmtFinSF1','LotArea','GarageCars','YearBuilt','KitchenQual']
to_predict_list = [['7','1710','856','Y','NaN',706,8450,2,2003,'Gd'],
                   [6,1262,1262,'Y','TA',978,9600,2,1976,'TA']]

df = pd.DataFrame(to_predict_list, columns=columns)
print(df)
print(df.dtypes)

result = make_prediction(input_data = df)
prediction = result.get('model_output')
print(prediction)
