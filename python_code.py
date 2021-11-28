
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as ms
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error,r2_score,mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV,RidgeCV
from yellowbrick.regressor import AlphaSelection
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r'D:\vehicles.csv')


df.columns


df = df.reindex(columns=['id', 'url', 'region', 'region_url', 'year', 'manufacturer',
       'model', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status',
       'transmission', 'VIN', 'drive', 'size', 'type', 'paint_color',
       'image_url', 'description', 'county', 'state', 'lat', 'long',
       'posting_date', 'price'])


len(df.columns)

 
# # DATA CLEANING

 
# #### removing useless features -> id, url, region_url, VIN, image_url,description, county, state, posting_date


df = df.drop(['id','url', 'region_url','VIN','image_url','description','county','state','posting_date'], axis = 1)

 
# ### visualising null values


nullVal = df.isnull().sum().to_frame()
nullVal


fig = ms.heatmap(df)
fig_copy = fig.get_figure()
fig_copy.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\heatmap.png', bbox_inches = 'tight')


fig = ms.bar(df)
fig_copy = fig.get_figure()
fig_copy.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\bar.png', bbox_inches = 'tight')


fig = ms.matrix(df , color=(1, 0.01, 0.01))
fig_copy = fig.get_figure()
fig_copy.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\matrix.png', bbox_inches = 'tight')

 
# # PREPROCESSING

 
# ### A) encoding of categorial features


#seperating numeric variable and categorical variable
num_col=['year','odometer','lat','long']
cat_cols=['region','manufacturer','model','condition','cylinders','fuel','title_status','transmission','drive','size','type','paint_color']


df2 = df.copy()


#object of LabelEncoder
encoder=LabelEncoder()
def encodeCatData(df):
    for cat in (cat_cols):  
        data = df[cat]
        non = np.array(data.dropna())
        im = encoder.fit_transform(non)
        data.loc[data.notnull()] = np.squeeze(im)
    return df


edf = encodeCatData(df2)


edf

 
# ### B) handling of NULL values

 
# #### 1.) deleting NULL value rows


delDF = edf.copy()


delDF = delDF.dropna(how='any',axis=0) 


delDF


nullV = delDF.isnull().sum().to_frame()
nullV


delDF.to_csv(r'C:\Users\Asus\Desktop\ML-PROJECT\finaldel.csv',index=False)

 
# #### 2.) iterative imputation (Bayesian Ridge() , KNN(), DecisionTree() , Random Forest() )


iidf = edf.copy()


ierrordf = pd.DataFrame()


def iteImputer(df , k , models):    
    for model in tqdm(models):
        print("Evaluating: ",model.__class__.__name__)
        impute = IterativeImputer(model)
        df1 = df.copy()
        for col in df.columns:
            idata = impute.fit_transform(df1[col].values.reshape(-1,1))
            idata = idata.astype('int64')
            idata = pd.DataFrame(np.ravel(idata))
            df1[col]=idata
            X = df1.iloc[: , :-1].to_numpy()
            Y = df1.iloc[: , -1:].to_numpy().reshape(-1)
            cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=2, random_state=1)
            ierrordf['*'+model.__class__.__name__] = cross_val_score(
        model, X, Y, scoring='neg_mean_squared_error',
        cv=cv)


models = [BayesianRidge(),KNeighborsRegressor(n_neighbors=20)
         ,DecisionTreeRegressor(max_features='sqrt', random_state=0),
         RandomForestRegressor(max_depth = 5 , random_state = 0)]
iteImputer(iidf , 4 , models)

 
# #### 3.) simple imputation (mean, median, mode)


sidf = edf.copy()


def simpleImputer(df , k , base):
    m_arr = ['median' , 'mean' , 'most_frequent']
    for i in tqdm(range(3)):
        stat = m_arr[i]
        print("Imputating: " ,stat)
        X = df.iloc[: , :-1].to_numpy()
        Y = df.iloc[: , -1:].to_numpy().reshape(-1)
        pipeline = Pipeline(steps = [('si' , SimpleImputer(missing_values=np.nan, strategy=stat)) , ('m' , base)])
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=2, random_state=1)
        ierrordf[stat] = cross_val_score(
        pipeline, X, Y, scoring='neg_mean_squared_error',
        cv=cv)
        


simpleImputer(sidf , 4 , BayesianRidge())


ierrordf


fig, ax = plt.subplots(figsize=(15, 8))
mean_arr = -ierrordf.mean()
names = ierrordf.columns
ax.barh(names, mean_arr , color = 'red')
ax.set_title('MSE with Different learning models')
ax.set_xlabel('mean MSE')
for i in ax.patches:
    plt.text(i.get_width()/3, i.get_y()+0.3,
             str(i.get_width()),
             fontsize = 15, fontweight ='bold',
             color ='white')
plt.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\different-imputations-method',dpi=None)
plt.show()

 
# ### C) filling of missing values


finaldfwo = edf.copy()


for col in finaldfwo.columns:
    if(col == 'price'):
        continue
    finaldfwo[col].fillna(finaldfwo[col].mode()[0], inplace=True)


nullV = finaldfwo.isnull().sum().to_frame()
nullV

 
# ### D) removing Outliers using IQR method


Q1 = finaldfwo.quantile(0.25)
Q3 = finaldfwo.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


finaldf = finaldfwo.copy()
low = Q1 - 1.5 * IQR
high = Q3 + 1.5 * IQR
outlier_col = ['odometer' , 'year' , 'price']
for col in outlier_col:
    finaldf = finaldf[(finaldf[col] >= low[col]) & (finaldf[col] <= high[col])]
finaldf = finaldf[finaldf['price']!=0]


print('INITIAL DATASET: ', finaldfwo.shape)
print('FINAL DATASET: ', finaldf.shape)
print('ROWS DROPPED :' ,finaldfwo.shape[0] - finaldf.shape[0] ,'COLUMNS DROPPED:', 26 - finaldf.shape[1])


finaldf.to_csv(r'C:\Users\Asus\Desktop\ML-PROJECT\final.csv',index=False)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


scaler = StandardScaler()


data = finaldf.to_numpy()


X = data[: , :-1]
Y = data[ :  ,-1]


x = finaldf.iloc[: , :-1]
y = finaldf.iloc[ :  ,-1:]


type(x)


x.shape
# y.shape


Xstd = scaler.fit_transform(X) 

 
# ## splitting data into train and test


x_train, x_test, y_train, y_test = train_test_split(Xstd, Y, test_size = 0.2 , random_state = 0)


#remove the negative predicted values
def negativeHandler(y_test,y_pred):
    ind=[idx for idx in range(len(y_pred)) if(y_pred[idx] > 0)]
    y_test = y_test[ind]
    y_pred = y_pred[ind]
    y_pred[y_pred<0]
    return (y_test,y_pred)


#function to give R-2 score of the regression model
def score(y_test,y_pred):
    report=[]
    report.append(mean_squared_log_error(y_test, y_pred))
    report.append(np.sqrt(report[0]))
    report.append(r2_score(y_test,y_pred))
    report.append(round(100 * r2_score(y_test,y_pred) , 8))
    return (report)


#dataframe that store the score of each model
model_reports=pd.DataFrame(index=['MSLE', 'Root MSLE', 'R2 Score','accuracy(%)'])

 
# # 1) Linear Regression


linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred=linear_model.predict(x_test)
y_test_1,y_pred_1=negativeHandler(y_test,y_pred)
linear_r = score(y_test_1,y_pred_1)
print('Coefficients: \n', linear_model.coef_)
print("MSLE : {}".format(linear_r[0]))
print("Root MSLE : {}".format(linear_r[1]))
print("R2 Score : {} or {}%".format(linear_r[2],linear_r[3]))
model_reports['Linear Regression']=linear_r


#Ploting feature importance graph
coef = pd.Series(linear_model.coef_, index = finaldf.columns[:-1])
imp_coef = coef.sort_values()
plt.rcParams['figure.figsize'] = (6.0, 6.0)
imp_coef.plot(kind = "barh" , color = '#010440')
plt.title("Feature importance using Linear Regression Model")
plt.xlabel('Figure 4' , fontsize = 22)
plt.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\Linear-Regression-Feature-Importance.jpg')
plt.show()

 
# # 2) Ridge Regression


#model object and fitting model
Ridge_model = Ridge(alpha=20 , solver='auto')
Ridge_model.fit(x_train,y_train)
y_pred = Ridge_model.predict(x_test)


y_test_2  ,y_pred_2 = negativeHandler(y_test,y_pred)
ridge_score = score(y_test_2,y_pred_2)
print("MSLE : {}".format(ridge_score[0]))
print("Root MSLE : {}".format(ridge_score[1]))
print("R2 Score : {} or {}%".format(ridge_score[2],ridge_score[3]))
model_reports['Ridge Regression'] = ridge_score


#finding optimal regularisation paramater
alphas = 10**np.linspace(10,-2,300)
model = RidgeCV(alphas=alphas)
gp = AlphaSelection(model)
gp.fit(x_train,y_train)



# plotting feature importance of ridge regression
coef = pd.Series(RR.coef_, index = x.columns)
imp_coef = coef.sort_values()
plt.rcParams['figure.figsize'] = (6.0, 6.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Ridge Model")
plt.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\Ridge-Regression-Feature-Importance.jpg')
plt.show()

 
# # 3) Lasso Regression


#fiiting lasso model
lasso_model = Lasso(alpha=0.0003)
lasso_model.fit(x_train,y_train)
y_pred=lasso_model.predict(x_test)


y_test_3,y_pred_3  =negativeHandler(y_test,y_pred)
lasso_score=score(y_test_3,y_pred_3)
print("MSLE : {}".format(lasso_score[0]))
print("Root MSLE : {}".format(lasso_score[1]))
print("R2 Score : {} or {}%".format(lasso_score[2],lasso_score[3]))
model_reports['Lasso Regression'] = lasso_score

 
# # 4) KNN


#estimating MSLE for k=1-9
R_MSLE=[]
for i in range(1,10):
    KNN_model=KNeighborsRegressor(n_neighbors=i)
    KNN_model.fit(x_train,y_train)
    y_pred=KNN_model.predict(x_test)
    error = np.sqrt(mean_squared_log_error(y_test, y_pred))
    R_MSLE.append(error)
    print("K =",i," , Root MSLE =",error)


#plotting error
plt.figure(figsize=(6 ,6))
plt.xticks(list(range(1,10)), list(range(1,10)), rotation='horizontal')
plt.plot(list(range(1,10)),R_MSLE)
plt.xlabel('K')
plt.ylabel('MSLE')
plt.title('Error Plot for Each K')
plt.xlabel('Figure 7' , fontsize = 22)
plt.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\KNN-Error-Plot.jpg')
plt.show()


#for best k =2
best_KNN=KNeighborsRegressor(n_neighbors=2) 
best_KNN.fit(x_train,y_train)
y_pred=best_KNN.predict(x_test)


#model evaluation
knn_score=score(y_test,y_pred)
print("MSLE : {}".format(knn_score[0]))
print("Root MSLE : {}".format(knn_score[1]))
print("R2 Score : {} or {}%".format(knn_score[2],knn_score[3]))
model_reports['KNN']=knn_score

 
# # 5) Random Forest


forest_model = RandomForestRegressor(n_estimators=180,random_state=0, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=True)
forest_model.fit(x_train,y_train)
y_pred = forest_model.predict(x_test)


forest_score = score(y_test,y_pred)
print("MSLE : {}".format(forest_score[0]))
print("Root MSLE : {}".format(forest_score[1]))
print("R2 Score : {} or {}%".format(forest_score[2],forest_score[3]))
model_reports['RandomForest Regressor']=forest_score


feature_pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
feature_pred_df = feature_pred_df.head(25)
feature_pred_df.plot(kind='bar',figsize=(8,8) , color = ['#010440', '#59CCD9'])
plt.grid(which='major', linestyle='dashed', linewidth='0.9', color='#49736B')
plt.title('Performance of Random Forest')
plt.ylabel('Mean Squared Log Error')
plt.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\Random-Forest-Performance.jpg')
plt.show()


imp = forest_model.feature_importances_
features=finaldf.columns[:-1]
with plt.style.context('bmh'):
    x_values = list(range(len(imp)))
    plt.figure(figsize=(6,6))
    plt.bar(x_values, imp, orientation = ('vertical'))
    plt.xticks(x_values, features, rotation=(90))
    plt.ylabel('Importance'); 
    plt.xlabel('Variable/Features'); 
    plt.title('Random Forest Variables Importance')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\Random-Forest-Variables-Imp.png',dpi=600)
    plt.show()

 
# # 6) Adaboost


adb_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=25),n_estimators=190,learning_rate=0.6001)
adb_model.fit(x_train, y_train)
y_pred = adb_model.predict(x_test)


adb_score=score(y_test,y_pred)
print("MSLE : {}".format(adb_score[0]))
print("Root MSLE : {}".format(adb_score[1]))
print("R2 Score : {} or {}%".format(adb_score[2],adb_score[3]))
model_reports['AdaBoost Regressor']=r7_ab


imp = adb_model.feature_importances_
features=finaldf.columns[:-1]
plt.style.use('bmh')
idx = np.argsort(imp)
idx = [x for x in reversed(idx)]
plt.figure(1)
plt.title('Features Importance')
plt.barh(range(len(idx)), imp[idx], color='g', align='center')
plt.yticks(range(len(idx)), features[idx])
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\Adaboost-Features-Importance2.png',dpi=600)
plt.show();

 
# # 7) XGBoost


xg_model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.4005,max_depth = 25, alpha = 5, n_estimators = 190)
xg_model.fit(x_train,y_train)
y_pred = xg_model.predict(x_test)


y_test_1,y_pred_1 = negativeHandler(y_test,y_pred)
xg_score = score(y_test_1,y_pred_1)
print("MSLE : {}".format(xg_score[0]))
print("Root MSLE : {}".format(xg_score[1]))
print("R2 Score : {} or {}%".format(xg_score[2],xg_score[3]))
model_reports['XgBoost Regressor'] = xg_score


xg_model.plot_importance(xg_reg , color = '#010440')
plt.rcParams['figure.figsize'] = [8, 8]
plt.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\XGBoost-Features-Importance.jpg')
plt.show()

 
# # Conclusion


model_reports


model_accuracy = model_reports.loc['accuracy(%)']


x = list(range(len(model_accuracy)))
y = np.arange(0,101)
d = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.figure(figsize=(20,6))
plt.plot(model_accuracy , color = '#010440')
plt.yticks(y)
plt.xticks(fontsize=15)
plt.xticks(rotation = (5))
plt.xlabel("Models",fontsize=20)
plt.ylabel("accuracy(%)-R2-score",fontsize=20)
plt.title("Performance of Models")
for a,b in zip(x,y):
    b=model_accuracy[a]
    val="("+str(round(model_accuracy[a],2.5))+" %)"
    plt.text(a, b+4, val,horizontalalignment='center',verticalalignment='center',color='#010440',bbox=d)
    plt.text(a, b+3, '.',horizontalalignment='center',verticalalignment='center',color='red',fontsize=50)
plt.savefig(r'C:\Users\Asus\Desktop\ML-PROJECT\plots\Overall-Performance.jpg',dpi=800)
plt.show();


