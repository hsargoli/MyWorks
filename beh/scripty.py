


def capcurve_plot(y_values, y_preds_proba):
        input_m = {}
        num_pos_obs = np.sum(y_values)
        num_count = len(y_values)
        rate_pos_obs = float(num_pos_obs) / float(num_count)
        ideal = pd.DataFrame({"x": [0, rate_pos_obs, 1], "y": [0, 1, 1]})
        xx = np.arange(num_count) / float(num_count - 1)
    
        y_cap = np.c_[y_values, y_preds_proba]
        y_cap_df_s = pd.DataFrame(data=y_cap)
        y_cap_df_s.index.name = "index"
        y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(
            "index", drop=True
        )
    
        yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
        yy = np.append(
            [0], yy[0 : num_count - 1]
        )  # add the first curve point (0,0) : for xx=0 we have yy=0
    
        percent = 0.5
        row_index = int(np.trunc(num_count * percent))
        
        val_y1 = yy[row_index]
        val_y2 = yy[row_index + 1]
        if val_y1 == val_y2:
            val = val_y1 * 1.0
        else:
            val_x1 = xx[row_index]
            val_x2 = xx[row_index + 1]
            val = val_y1 + ((val_x2 - percent) / (val_x2 - val_x1)) * (val_y2 - val_y1)
    
        sigma_ideal = (
            1 * xx[num_pos_obs - 1] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
        )
        sigma_model = integrate.simps(yy, xx)
        sigma_random = integrate.simps(xx, xx)
    
        ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
        # ar_label = 'ar value = %s' % ar_value
        xx = xx.tolist()
        yy = yy.tolist()
        del xx[1::2]
        del xx[1::2]
        del xx[1::2]
        del xx[1::2]
        del xx[1::2]
        del xx[1::2]
    
        del yy[1::2]
        del yy[1::2]
        del yy[1::2]
        del yy[1::2]
        del yy[1::2]
        del yy[1::2]
    
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(ideal["x"], ideal["y"], color="grey", label="Perfect Model")
        ax.plot(xx, yy, color="red", label="User Model")
    
        ax.plot(xx, xx, color="blue", label="Random Model")
        ax.plot([percent, percent], [0.0, val], color="green", linestyle="--", linewidth=1)
        ax.plot(
            [0, percent],
            [val, val],
            color="green",
            linestyle="--",
            linewidth=1,
            label=str(val * 100) + "% of positive obs at " + str(percent * 100) + "%",
        )
    
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.25)
        plt.title("CAP Curve - a_r value =" + str(ar_value))
        plt.xlabel("% of the data")
        plt.ylabel("% of positive obs")
        plt.legend()
        plt.show()
    
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        sh = xx.shape[0]
        xx = xx.reshape((sh, 1))
        yy = yy.reshape((sh, 1))
    
        user_d = np.concatenate((xx, yy), axis=1)
        input_m["Ideal"] = ideal.values
        input_m["User"] = user_d
        plt.savefig('capcurve')
        
        return input_m

capcurve_plot(y_test, pred)


total = len(y_test)  

one_count = np.sum(y_test)

zero_count = total - one_count

plt.figure(figsize = (10, 6))


plt.plot([0, total], [0, one_count], c = 'b',
		linestyle = '--', label = 'Random Model')

y_test = np.array(y_test)

lm = [y for y,_ in sorted(zip(pred, y_test), reverse = True)]
x = np.arange(0, total + 1)
y = np.append([0], np.cumsum(lm))
fifty_percent = int(len(x) / 2)
# print('---- ',fifty_percent, y[fifty_percent], y[-1])
# print((y[-1] - y[fifty_percent])/100)


plt.plot(x, y, c = 'r', label = 'Random classifier', linewidth = 3)


plt.plot([0, one_count, total], [0, one_count, one_count],
         c = 'grey', linewidth = 2,linestyle = '--', label = 'Perfect Model')


print([one_count, total, zero_count])



plt.xlabel('total observation')
plt.ylabel('total positive outcomes')
plt.legend()



a = auc([0, float(total)], [0, float(one_count)])

aP = auc([0, float(one_count), float(total)], [0, float(one_count), float(one_count)]) - a

# Area between Trained and Random Model
aR = auc(x, y) - a

auc_capcurve = aR / aP

half_x = total / 2
y_half_x = y[int(half_x)]
method2_res = y_half_x / one_count


# >>>>>>>>>>>>>>>>>> >  >   >

# %% IMPORT LIBRARIES OLD



# -------------------------
# IMPORT LIBRARIES  
# -------------------------

import dask.dataframe as dd

import glob
from sklearn import metrics
def _timeit(func):
    @wraps(func)
    def timeit_wraps(*args, **kwargs):
        st = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        total_time = end - st
        print(f'-- Funcion {func.__name__} -> \tTook {total_time :.2f} seconds \
              \n {args} {kwargs} ---  ')
        return result 
    return timeit_wraps
from sklearn import model_selection
from mlxtend.classifier import StackingCVClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier 
from functools import wraps
import time
from decimal import Decimal
import seaborn as sn
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import datetime
import pickle
import pandas as pd
import warnings,os
warnings.filterwarnings("ignore")
import numpy as np
import sys
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, auc, roc_curve
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
np.set_printoptions(threshold = np.inf)
from sklearn.model_selection import train_test_split,KFold
sys.path.insert(0, r'C:\Users\sargoli\project')


# -------------------------
# Functions for split data by each year
# -------------------------
def data_splitter(df, column, date0, date1):
    a = df.loc[(df[column] >= date0) & (df[column] < date1)]
    return a

def data_splitter_year(df, column, year):
    a = df.loc[(df[column] >= year*10000) & (df[column] < (year+1)*10000 )]
    return a

# <split data>        
def data_splitter_years(df, column_name, dic:dict, base_year = None):
    if base_year is None:
        container = []
        percent_and_year = list(dic.items())
        for i in range(len(percent_and_year)):
            year = percent_and_year[i][0]
            actual_count = int(data_splitter_year(df, column_name, year).count()[0] * percent_and_year[i][1])
            df_of_year = data_splitter_year(df, column_name, year).sample(actual_count)
            print(f' - {actual_count} data are selected from year - {year} -')
            container.append(df_of_year)
        df_of_selected_years = pd.concat(container)
        return df_of_selected_years
    
    if base_year is not None:
        percent_of_base_year = list(dic.values())[list(dic.keys()).index(base_year)]
        dataset_of_base_year = data_splitter_year(df, column_name, base_year)
        count_for_base_year = int(dataset_of_base_year.shape[0] * percent_of_base_year)
        print('count of base year:', count_for_base_year, f'\t percent Base Year: {percent_of_base_year}')
        base_year_apply_percent = dataset_of_base_year.sample(count_for_base_year) 
        del dic[base_year]
        percent_and_year = list(dic.items())
        container = []
        for i in range(len(percent_and_year)):
            desire_count = int(percent_and_year[i][1] * count_for_base_year)
            year = percent_and_year[i][0]
            actual_count = data_splitter_year(df, column_name, year).count()[0]
            countof = min(desire_count, actual_count)
            df_of_year = data_splitter_year(df, column_name, year).sample(countof)
            print(f' - {countof} data are selected from year - {year} -')
            container.append(df_of_year)
        container.append(base_year_apply_percent)
        df_of_selected_years = pd.concat(container)
        return df_of_selected_years

def normalization(x, xtest, tipe = 'standard'): 
    if tipe == 'minmax':
        # MIN_MAX
        scaler = MinMaxScaler()
        x_train_mm = scaler.fit_transform(x)
        x_test_mm = scaler.transform(xtest)
        return [x_train_mm, x_test_mm]

    if tipe == 'robust':
        # ROBUST
        scaler = RobustScaler()
        x_train_r = scaler.fit_transform(x)
        x_test_r = scaler.transform(xtest)
        return [x_train_r, x_test_r]

        
    if tipe == 'standard':
        # STANDARD
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x)
        x_test_s = scaler.fit_transform(xtest)
        return [x_train_s, x_test_s]


def evaluate(model, x, y):
    
    
    
    acc = model_selection.cross_val_score(model, x, y, cv = kfold, scoring = 'accuracy')
    
    roc = model_selection.cross_val_score(model, x, y, cv = kfold, scoring = 'roc_auc')
    
    pred = model.predict(x)
    matrix = confusion_matrix(y, pred)


    pred = model.predict(x)
    report = classification_report(y, pred)
    
    res = (f'\naccuracy: {acc.mean()}\nroc_auc: {roc.mean()}\n \
    \nconfusion matrix:\n {matrix}\nReport:\n {report}\n')
    
    # print(res)
    
    return res,matrix



def cal_f1(xdf, ydf):
    x, y = np.array(xdf), np.array(ydf)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.3,
                                                    shuffle = True,
                                                    random_state = 255)
    _, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.1,
                                                  random_state = 255)
    model = model_used(n_estimators=100, max_depth=4,
                                 random_state=14, class_weight='balanced_subsample', n_jobs=18)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    report = classification_report(y_test, pred, output_dict= True)
    f10 = report['0']['f1-score']
    f11 = report['1']['f1-score']
    f = (f10 + f11) / 2 
    return  f


def f1_eval(y_pred, Y_test):
    y_true = Y_test.get_label()
    err = 1 - f1_score(y_true, np.round(y_pred))
    #print('err',err)
    return 'err', err

# %% prepare data
import risk


time1 = time.time()


pathgh98 = r'E:\dataframes\98problem\gheyr_98.csv'
d_structure = dd.read_csv(pathgh98, blocksize = 64000000)
ddf = d_structure.compute()
dir(risk)


print('elapsed time: ', time.time() - time1, ddf.shape) 






# %% filter_method
# %%% read data


# %%% variance threshold

from sklearn.feature_selection import VarianceThreshold

constant_filter = VarianceThreshold(threshold=1)
data_constant = constant_filter.fit_transform(x_train)

cons_column = [i for i in x_df.columns if i not in x_df.columns[constant_filter.get_support()]]
cons_column


# %%  count of 1 in 97 98 99


summ =[]
df_df = pd.DataFrame()
for i in glob.glob(r'f:\*1_Without_FCL.csv'):
    data = pd.read_csv(i,',')
    print(i, data.shape)
    row_count = len(data.index)
    summ.append(row_count)
    del data




with open('no_of_1.txt', 'w') as f:
    f.write(str(summ))
    
 





# %% memmory usage of all vars in session


local_var = list(locals().items())
a = []
for i,j in local_var:
    aa = [i, sys.getsizeof(j)]
    a.append([aa[0],int(aa[1])])
    print(sys.getsizeof(j))
 
    

print(*a, sep = '\n')























# %% old
# %%% preparing Data 
# %%%% prep data 98 


path98 = r'E:\dataframes\98problem\98_unb.csv'
pathgh98 = r'E:\dataframes\98problem\gheyr_98.csv'

df98_csv = pd.read_csv(path98)
dfgh98_csv = pd.read_csv(pathgh98)

del df98_csv['Unnamed: 0']
del dfgh98_csv['Unnamed: 0']


# -------------------------
# select x - y for apply on models
# -------------------------
x_df = dfgh98_csv.loc[:, dfgh98_csv.columns != 'target']
y_df = dfgh98_csv.loc[:, dfgh98_csv.columns == 'target']
x_train, y_train = np.array(x_df), np.array(y_df)


x_df = df98_csv.loc[:, df98_csv.columns != 'target']
y_df = df98_csv.loc[:, df98_csv.columns == 'target']
x_test, y_test = np.array(x_df), np.array(y_df)

_, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.1,
                                                  random_state = 255,
                                                  shuffle = True)

print(f'xtrain {x_train.shape}\nytrain {y_train.shape}\n'
      f'xtest {x_test.shape}\nytest{y_test.shape}\nxval'
      f'{x_val.shape}\nyval {y_val.shape}\n')

# %%% prep data OLD Data

'''
==============================================================================
-- read data from datasets :M2_Risk_Final_Features_Shakhsi_V2,
                           M2_cnt_Estemhal_Yearly_V5
-- split by dictionary of date 
-- make label with UnderSampling technique (number of 1 and number of 0 is equal)
-- 
==============================================================================
'''

path_shakhsi = r'F:\98\M2_Risk_Final_Features_Shakhsi_V2.csv'
path_estemhal = r'F:\98\M2_cnt_Estemhal_Yearly_V5.csv'
skip_r = 4
chunk_size = 150_000
dict_of_year1 = {1392:1, 1393:1, 1394:1,
                1395:1, 1396:1, 1397:1, 1398:0, 1399:0}
dict_of_year2 = {1398:1}




# -------------------------
# READ DATA 
# -------------------------
#--- key  ---#
featureShakhsi_db = pd.read_csv(path_shakhsi,',', chunksize=chunk_size,
                                skiprows = range(1,skip_r))
est_df = pd.read_csv(path_estemhal, ',')
shakhsi_df = next(featureShakhsi_db)

df_lable = shakhsi_df.columns

shakhsi_df.drop('CNTESTEMHAL', inplace =True, axis=1)

shakhsi_df = pd.merge(shakhsi_df, est_df,how="left", left_on=["INTCUSTID", "DAYDATEOBS"], 
                  right_on=["INTCUSTID", "DAYDATEOBS"])

sample_df = data_splitter_years(shakhsi_df,
                    'DAYDATEOBS',
                    dict_of_year1)   ####  key  ####



del shakhsi_df
# sample_df.to_pickle('98_without_balance.pkl')



# -------------------------
# balance data | UnderBalance
# -------------------------
# split data to [MAXFCL == 1 2] + [MAXFCL == 3 & 4]
data_34 = pd.concat([sample_df[sample_df['MAXFCL'] == 3],
           sample_df[sample_df['MAXFCL'] == 4]])
data_34 = data_34[:]
print('label 1 Count: ',data_34.shape[0])
data_12 = pd.concat([sample_df[sample_df['MAXFCL'] == 1],
           sample_df[sample_df['MAXFCL'] == 2]])
data_12 = data_12[:data_34.shape[0]]   #### UNDERBALANCE


print('label 0 Count: ', data_12.shape[0])
# concatenate them to create TOYDATASET
sample_df = pd.concat([data_34,data_12])
print('all data: ', sample_df.shape[0])


# -------------------------
# make LABEL for maxfcl:3,4 -> 1 and maxfcl: 1,2 -> 0
#       estem>0 ->1
# -------------------------
maxfcl = list(sample_df['MAXFCL'])
estem = list(sample_df['CNTESTEMHAL'])
y = []
for i in range(len(maxfcl)):
    yy = 1 if (maxfcl[i] == 3 or maxfcl[i] == 4) or estem[i]>0 else 0
    y.append(yy)
y = np.array(y)
# insert y as Target column to DataFrame
sample_df['target'] = y
print(sample_df['target'].value_counts())
print('unique of maxFcl in dataset', sample_df['MAXFCL'].unique())







# -------------------------
#  trim dataset. delete columns - fill nan values 
# -------------------------
print('--- number of features columns', sample_df.shape[1])
column_to_delete = ['CNTESTEMHAL', 'INTCUSTID', 'TEL', 'MAXFCL', 'NAME', 'BIRTHLOCATIONCITY',
                   'PASSNO', 'ECONOMICALCODE', 'CREATEDATE', 'FIRSTACCCREATEDATE', 'ISSUEDATE',
                   'BIRTHDATE', 'BIRTHCITYCODE_ID']

sample_df = sample_df.drop(column_to_delete, axis=1)
print('nan values', sample_df.isna().sum().sum())
sample_df = sample_df.fillna(0)
print('nan values', sample_df.isna().sum().sum())
# del sample_df['DAYDATEOBS']    ############### to delete daydate ################
print('--- number of features after delete columns', sample_df.shape[1])





# -------------------------
# save dataframe
# -------------------------
# sample_df.to_pickle('older_than_98.pkl') # save dataframe Pickle 
#sample_df.to_csv('gheyr_98.csv')





# -------------------------
# select x - y for apply on models
# -------------------------
x_df = sample_df.loc[:, sample_df.columns != 'target']
y_df = sample_df.loc[:, sample_df.columns == 'target']
x, y = np.array(x_df), np.array(y_df)

# split data to train, test and validation data
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.3,
                                                    random_state = 255,
                                                    shuffle = True)
_, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.2,
                                                  random_state = 255,
                                                  shuffle = True)

print(f'xtrain {x_train.shape}\nytrain {y_train.shape}\n'
      f'xtest {x_test.shape}\nytest{y_test.shape}\nxval'
      f'{x_val.shape}\nyval {y_val.shape}\n')







# %%% Machin Learning CLASS 

'''
==============================================================================
-- machin learning algorithms


==============================================================================
'''

class MachinLearning:
    def __init__(self, df, x = None, y = None, xtest = None, ytest = None, xval = None, yval = None):
        self.df = df
        self.clf_xg = None
        self.clf_lr = None
        self.clf_rf = None
        self.clf_st = None
        self.predxg = None
        self.predlr = None
        self.predrf = None
        self.FNlr = None
        self.FNrf = None
        self.FNxg = None
        self.x_train = x
        self.y_train = y
        self.x_test = xtest
        self.y_test = ytest
        self.x_val = xval
        self.y_val = yval
        # after: change df to path of dataset    
        # print(x.shape, y.shape, xtest.shape, ytest.shape, xval.shape, yval.shape, sep = '\n--\n')
        
    def _timeit(func):
        @wraps(func)
        def timeit_wraps(*args, **kwargs):
            st = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            total_time = end - st
            print(f'-- Funcion {func.__name__} -> \tTook {total_time :.2f} seconds \
                  \n {args} {kwargs} ---  ')
            return result 
        return timeit_wraps
                
    def capcurve_plot(self, y_values, y_preds_proba):
        input_m = {}
        num_pos_obs = np.sum(y_values)
        num_count = len(y_values)
        rate_pos_obs = float(num_pos_obs) / float(num_count)
        ideal = pd.DataFrame({"x": [0, rate_pos_obs, 1], "y": [0, 1, 1]})
        xx = np.arange(num_count) / float(num_count - 1)
    
        y_cap = np.c_[y_values, y_preds_proba]
        y_cap_df_s = pd.DataFrame(data=y_cap)
        y_cap_df_s.index.name = "index"
        y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(
            "index", drop=True
        )
    
        yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
        yy = np.append(
            [0], yy[0 : num_count - 1]
        )  # add the first curve point (0,0) : for xx=0 we have yy=0

        percent = 0.5
        row_index = int(np.trunc(num_count * percent))
        
        val_y1 = yy[row_index]
        val_y2 = yy[row_index + 1]
        if val_y1 == val_y2:
            val = val_y1 * 1.0
        else:
            val_x1 = xx[row_index]
            val_x2 = xx[row_index + 1]
            val = val_y1 + ((val_x2 - percent) / (val_x2 - val_x1)) * (val_y2 - val_y1)
    
        sigma_ideal = (
            1 * xx[num_pos_obs - 1] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
        )
        sigma_model = integrate.simps(yy, xx)
        sigma_random = integrate.simps(xx, xx)
    
        ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
        # ar_label = 'ar value = %s' % ar_value
        xx = xx.tolist()
        yy = yy.tolist()
        del xx[1::2]
        del xx[1::2]
        del xx[1::2]
        del xx[1::2]
        del xx[1::2]
        del xx[1::2]
    
        del yy[1::2]
        del yy[1::2]
        del yy[1::2]
        del yy[1::2]
        del yy[1::2]
        del yy[1::2]
    
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(ideal["x"], ideal["y"], color="grey", label="Perfect Model")
        ax.plot(xx, yy, color="red", label="User Model")
    
        ax.plot(xx, xx, color="blue", label="Random Model")
        ax.plot([percent, percent], [0.0, val], color="green", linestyle="--", linewidth=1)
        ax.plot(
            [0, percent],
            [val, val],
            color="green",
            linestyle="--",
            linewidth=1,
            label=str(val * 100) + "% of positive obs at " + str(percent * 100) + "%",
        )
    
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.25)
        plt.title("CAP Curve - a_r value =" + str(ar_value))
        plt.xlabel("% of the data")
        plt.ylabel("% of positive obs")
        plt.legend()
        plt.show()
    
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        sh = xx.shape[0]
        xx = xx.reshape((sh, 1))
        yy = yy.reshape((sh, 1))
    
        user_d = np.concatenate((xx, yy), axis=1)
        input_m["Ideal"] = ideal.values
        input_m["User"] = user_d
        plt.savefig('capcurve')
        
        return input_m

    def certain(self, t = 'int64'):
        """
        mention which type do you want to create subset dataset
        as default it returns /int64/ column(s) 
        """
        a = self.df.select_dtypes(t)
        return a
    

# ----------------------------------------------------
# ----------------------------------------------------    train
# ----------------------------------------------------
# <train>
    @_timeit
    def train_xgboost(self):        
        self.clf_xg = XGBClassifier(learning_rate=0.01, n_estimators=50, seed=8,
                                    colsample_bytree=1, scale_pos_weight=5
                                   ,importance_type='gain', reg_alpha=0.8, reg_lambda=4.7, n_jobs=-1)
        self.clf_xg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_val, self.y_val)],
                eval_metric=f1_eval, early_stopping_rounds=30, verbose=0)
        self.importance = self.clf_xg.feature_importances_  # types: numpy.float32 

        self.evaluate_xg, self.cm_xg = self.evaluate(self.clf_xg, self.x_test, self.y_test)
        self.f_xg, self.FNxg = self.feature_importance(self.clf_xg)
        
        
        self.predxg = self.clf_xg.predict(self.x_test)
        
        
        
# <train>    ------------------------------------------------
    @_timeit  
    def train_randomforest(self):
        self.clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,
                             class_weight='balanced_subsample', n_jobs=-1,
                             eval_metric = 'acc')
        # random_state=14
        self.clf_rf.fit(self.x_train, self.y_train)
        
        
        self.importance = self.clf_rf.feature_importances_
        self.evaluate_rf, self.cm_rf = self.evaluate(self.clf_rf, self.x_test, self.y_test)
        self.f_rf, self.FNrf = self.feature_importance(self.clf_rf)
        
        self.predrf = self.clf_rf.predict(self.x_test)
        self.capcurve_plot(self.y_test ,self.predrf)
        
        
# <train>    ------------------------------------------------
    @_timeit
    def train_logit(self):
        x, xtest = self.normalization(self.x_train, self.x_test)
        self.clf_lr = LogisticRegression(C=100, solver='newton-cg',
                                         class_weight='balanced',
                                         n_jobs=-1,
                                         penalty='l2',max_iter = 100000, 
                                         )
        
        # {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'}
        self.clf_lr.fit(x, self.y_train)
        
        self.importance = self.clf_lr.coef_[0]
        # print(type(self.importance[10]),self.importance.shape, self.importance)

        self.evaluate_lr, self.cm_lr = self.evaluate(self.clf_lr, xtest, self.y_test)
        self.f_lr, self.FNlr = self.feature_importance(self.clf_lr)
        
        
        
        self.predlr = self.clf_lr.predict(xtest)
        
        

        
# ----------------------------------------------------
# ----------------------------------------------------    evaluate    
# ----------------------------------------------------
# <evaluate>        
    def evaluate(self, model, x, y):
        kfold = model_selection.KFold(n_splits = 10, random_state = 255, shuffle = True)
        
        self.acc = model_selection.cross_val_score(model, x, y, cv = kfold, scoring = 'accuracy')
        
        self.roc = model_selection.cross_val_score(model, x, y, cv = kfold, scoring = 'roc_auc')
        
        pred = model.predict(x)
        self.matrix = confusion_matrix(y, pred)
    
        self.report = classification_report(y, pred)
        # print(list(zip(y_test,pred)))         
        
        res = (f'\naccuracy: {self.acc.mean()}\nroc_auc: {self.roc.mean()}\n \
        \nconfusion matrix:\n {self.matrix}\nReport:\n {self.report}\n')
        
        # print(res)
        
        return res,self.matrix
        
        
        
        
# <evaluate>       
    def plots(self, model, ypred, name):
    
        x = np.arange(0, 1.1, 0.1)
        fpr, tpr, _ = roc_curve(self.y_test, ypred)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.plot(x, x,'-.')  
        plt.title('ROC curve ' + str(name))
        
        prec, recall, thresholds = precision_recall_curve(self.y_test, ypred)
        PrecisionRecallDisplay(precision=prec, recall=recall).plot()
        plt.title('PrecisionRecall plot ' + str(name))

        
        
# <evaluate>      
    def plots2(self, save_path):
        x = np.arange(0, 1.1, 0.1)
        fig, axx = plt.subplots(nrows=3, ncols = 2, sharey=False, figsize = (28,22))
        legend_list = []
        if self.predxg is not None:
            fprxg, tprxg, _ = roc_curve(self.y_test, self.predxg)            
            RocCurveDisplay(fpr=fprxg, tpr=tprxg).plot(ax = axx[0][0])
            legend_list.append('xgboost')
            df_cm = pd.DataFrame(self.cm_xg)
            df_cm.index.name, df_cm.columns.name = 'Actual', 'Predicted'
            sn.set(font_scale=1.4)
            sn.heatmap(df_cm, ax =axx[0][1], cmap='Blues', annot=True, annot_kws={'size':16}, fmt='g', cbar =False)
            axx[0][1].set_title('Xgboost Confusion matrix')

        if self.predlr is not None:
            fprlr, tprlr, _ = roc_curve(self.y_test, self.predlr) 
            RocCurveDisplay(fpr=fprlr, tpr=tprlr).plot(ax = axx[0][0])
            legend_list.append('logistic regression')
            df_cm = pd.DataFrame(self.cm_lr)
            df_cm.index.name, df_cm.columns.name = 'Actual', 'Predicted'
            sn.set(font_scale=1.4)
            sn.heatmap(df_cm, ax =axx[2][1], cmap='Blues', annot=True, annot_kws={'size':16}, fmt='g', cbar =False)
            axx[2][1].set_title('Logit Confusion matrix')
        if self.predrf is not None:
            fprrf, tprrf, _ = roc_curve(self.y_test, self.predrf)
            RocCurveDisplay(fpr=fprrf, tpr=tprrf).plot(ax = axx[0][0])
            legend_list.append('RandomForest')
            

            df_cm = pd.DataFrame(self.cm_rf)
            df_cm.index.name, df_cm.columns.name = 'Actual', 'Predicted'
            sn.set(font_scale=1.4)
            sn.heatmap(df_cm, ax =axx[1][1], cmap='Blues', annot=True, annot_kws={'size':16}, fmt='g', cbar =False)
            axx[1][1].set_title('RandomForest Confusion matrix')
        axx[0][0].plot(x, x,'-.') 
        axx[0][0].set_title('ROC curve ')
        axx[0][0].legend(legend_list)

        legend_list_pr =[]
        if self.predxg is not None:
            precxg, recallxg, thresholdsxg = precision_recall_curve(self.y_test, self.predxg)
            PrecisionRecallDisplay(precision=precxg, recall=recallxg).plot(ax = axx[1][0])
            legend_list_pr.append('xgboost')

        if self.predlr is not None:
            preclr, recalllr, thresholdslr = precision_recall_curve(self.y_test, self.predlr)
            PrecisionRecallDisplay(precision=preclr, recall=recalllr).plot(ax = axx[1][0])
            legend_list_pr.append('logistic regression')
                    
        if self.predrf is not None:
            precrf, recallrf, thresholdsrf = precision_recall_curve(self.y_test, self.predrf)
            PrecisionRecallDisplay(precision=precrf, recall=recallrf).plot(ax = axx[1][0])
            legend_list_pr.append('Random Forest')
        
        axx[1][0].set_title('PrecisionRecall plot')
        axx[1][0].legend(legend_list_pr)
        
        plt.savefig(str(save_path) + '.png', transparent=False, dpi=100, bbox_inches="tight")
        
        
        
        

# ----------------------------------------------------
# ----------------------------------------------------   feature selection 
# ----------------------------------------------------
# <feature selection>        
    def feature_importance(self, model):
        col = self.df.columns
        combine = np.array(list(zip(col,self.importance)))
        sort_comb = np.sort(combine[:,1])[::-1]
        self.sort_comb2 = [float(i) for i in sort_comb]
        nonzeroes = np.count_nonzero(self.sort_comb2)
        
        
        # print(f'number of {np.count_nonzero(self.sort_comb2)} features affect on data')
        # print(f'max effective col: {combine[np.where(combine[:,1] == str(self.sort_comb2[0]))]} ')
        
        
        
        self.top_value = [np.where(combine[:,1] == str(self.sort_comb2[i])) for i in range(nonzeroes)]

        
        most_eff = [combine[np.array(self.top_value)[i][0][0]] for i in range(nonzeroes)]
        # print('most effective columns: [column Name, Impact]' ,*most_eff, sep = '\n', end = '\n')
        
        feature_name = np.array(most_eff)[:,0]
        return [most_eff,nonzeroes],feature_name

         
# ----------------------------------------------------   
# ----------------------------------------------------   normalization
# ----------------------------------------------------
# <normalization>        
    def normalization(self, x, xtest, tipe = 'standard'): 
        if tipe == 'minmax':
            # MIN_MAX
            scaler = MinMaxScaler()
            x_train_mm = scaler.fit_transform(x)
            x_test_mm = scaler.transform(xtest)
            return [x_train_mm, x_test_mm]

        if tipe == 'robust':
            # ROBUST
            scaler = RobustScaler()
            x_train_r = scaler.fit_transform(x)
            x_test_r = scaler.transform(xtest)
            return [x_train_r, x_test_r]

            
        if tipe == 'standard':
            # STANDARD
            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x)
            x_test_s = scaler.transform(xtest)
            return [x_train_s, x_test_s]

# ----------------------------------------------------
# ----------------------------------------------------    model save & load
# ----------------------------------------------------
# <model save & load>
    def save_model(self, save_path, model_type):
        with open(save_path, 'wb') as file:
            pickle.dump(model_type, file)

# <model save & load>            
    def load_model(self, load_path):
        with open(load_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    
# ----------------------------------------------------
# ----------------------------------------------------     REPORT.TXT 
# ----------------------------------------------------
# <REPORT.TXT>

    
    def make_report2(self, DIR, name):
        
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        current = os.getcwd()
        os.chdir(DIR)
        
        if (os.path.exists(name)): # for overwrite files
            os.remove(name)
        per ='\n' + '....' * 20 + '\n'
        
        date = str(datetime.datetime.now())
        date0, date1 = date[:10], date[11:]
        with open(name + '_' +  date0 + '.txt', 'w') as file: 
            file.write('DATA FRAME description:\n')
            print('time that was created',date0, date1, file = file)
            
            
            file.write(f'shape of DataFrame: {self.df.shape[0]} rows\t{self.df.shape[1]} columns \n')
            
            # file.write(f'{per}, DataFrame describe (count, mean, std and quartile for each column :\n, {self.df.describe()}')

            a = {i: j for i in self.df.columns for j in self.df.dtypes}
            file.write(f'\nunique dtypes: {set(a.values())}\n')

            print(per, file = file)
            if self.clf_xg is not None:
                print('##########\nXGboost MODEL\n##########\n', file = file)
                file.write(self.evaluate_xg)
                print(f' number of features that affect on data: {self.f_xg[1]} ',file = file)
                print(' important Feature [column Name, Impact]:' ,*self.f_xg[0], sep = '\n', end = '\n', file = file)
                np.savetxt('_feature_importance_XG_' + name + '.txt', self.FNxg, encoding='utf-8',fmt="%s")
                self.save_model(name + '_xg_PickleModel', self.clf_xg)
                
            print(per, file = file)
    
            if self.clf_rf is not None:
                print('##########\nRandom forest MODEL\n##########\n', file = file)
                file.write(self.evaluate_rf)
                print(f' number of features that affect on data: {self.f_rf[1]} ',file = file)
                print(' important Feature [column Name, Impact]:',*self.f_rf[0], sep = '\n', end = '\n', file = file)
                np.savetxt('_feature_importance_RF_' + name + '.txt', self.FNrf,encoding='utf-8',fmt="%s")
                self.save_model(name + '_rf_PickleModel', self.clf_rf)

            print(per, file = file)
                
            if self.clf_lr is not None:
                print('##########\nLogistic regression MODEL\n##########\n', file = file)
                file.write(self.evaluate_lr)
                print(f' number of features that affect on data: {self.f_lr[1]}\n ',file = file)
                print(' important Feature [column Name, Impact]:' ,*self.f_lr[0], sep = '\n', end = '\n', file = file)
                np.savetxt('_feature_importance_LR_' + name + '.txt', self.FNlr, encoding='utf-8', fmt="%s")
                self.save_model(name + '_lr_PickleModel', self.clf_lr)
            if self.clf_st is not None:
                print('##########\nStacking MODEL\n##########\n', file = file)
                file.write(self.evaluate_st)
                #print(f' number of features that affect on data: {self.f_lr[1]}\n ',file = file)
                #print(' important Feature [column Name, Impact]:' ,*self.f_lr[0], sep = '\n', end = '\n', file = file)
                self.save_model(name + '_st_PickleModel', self.clf_lr)
                    
        
        
        
        
        self.plots2(name)
        os.chdir(current)
        return [self.FNxg, self.FNrf, self.FNlr],[self.clf_xg, self.clf_rf, self.clf_lr]
       
            
# <REPORT.TXT>        
    def desc(self): 

        a = {i: j for i in self.df.columns for j in self.df.dtypes}
        print('\nunique dtypes:',set(a.values()))
        #print(per, 'COLUMNS and their dtype:\n')
        #for i,j in a.items():
        #    print(i, ' --type-->  ' ,j)
        
    
        
    
    
    def stacking(self, classifiers_list: list, features_list: list):
        

        features = []
        piplines = []
        for i in range(len(classifiers_list)):
            pipline = make_pipeline(ColumnSelector(cols=features_list[i]), classifiers_list[i])
            piplines.append(pipline)
            print('No of features:\t',len(features_list[i]))
        
        #print(piplines,features)

        self.clf_st = StackingCVClassifier(classifiers=piplines,
                                    use_probas=True,
                                    meta_classifier=XGBClassifier(learning_rate=0.01,
                                                                  n_estimators=50,
                                                                  seed=8, colsample_bytree=1,
                                                                  scale_pos_weight=2,
                                                                  importance_type='gain',
                                                                  reg_alpha=0.9,
                                                                  reg_lambda=4.7, 
                                                                  n_jobs=18),
                                    shuffle=True,
                                    use_features_in_secondary=False,
                                    random_state=255)

        ytrn = self.y_train.squeeze()
        ytst = self.y_test.squeeze()
        self.clf_st.fit(self.x_train, ytrn)
        
        
        self.pred_test_data = self.clf_st.predict(self.x_test)
        self.pred_train_data= self.clf_st.predict_proba(self.x_train)
        
        self.evaluate_st, self.cm_st = self.evaluate(self.clf_st, self.x_test, ytst)
        
        self.predst = self.clf_st.predict(self.x_test)
    
    def get_file_line_count(self, path):
            with open(path) as f:
                for count, _ in enumerate(f): 
                    pass
            return count








# %%% TRAIN 
''' 
==============================================================================
 train dataset ~ see the report.txt
==============================================================================
''' 

ins = MachinLearning(df98_csv,
                     x_train, y_train,
                     x_test, y_test, x_val, y_val)




ins.train_logit()
ins.train_randomforest()
ins.train_xgboost()

models_data = ins.make_report2(r'e:\res\oo\\','q2')


# %%% SAVE & LOAD
# %%%%  Load Random Dataset Mr.AZIZI FUNCTIONs

def load_random_sample(file_path, n_total, n_sample, replace=False):
    skip_idxs = sorted(np.random.choice(range(1, n_total), n_total - n_sample, replace=replace))
    #header = 0 if config['has_header'] else None
    df = pd.read_csv(file_path, header=0, skiprows=skip_idxs, nrows=n_total)
    df.reset_index(inplace=True, drop=True)
    return df

random_df = load_random_sample(path_shakhsi, 24813646, 1000000)



def get_file_line_count(path):
        with open(path) as f:
            for count, _ in enumerate(f): 
                pass
        return count

get_file_line_count(path_shakhsi)

#%%% save load

import pandas as pd
import csv

test_data_1 =  pd.read_csv(r'f:/new_data/Custgroup_1_Year_1400_Lable_0_Without_FCL.csv', chunksize = 10000)
test_data_1 = next(test_data_1)

# %%%% save dataFrame by interval
'''
==============================================================================
    exchange dataset for each year 
==============================================================================
'''

sys.path.insert(0, r'C:\Users\sargoli\project')
jpath = r'C:\Users\sargoli\project\json_.json.json'

import newEra

def modoel_creator(columns_to_delete, chunk, skip_rows, save_name):
    step1 = newEra.NOKOL(r'F:\\New folder\\M2_Risk_Final_Features_Shakhsi_V2.csv',
                  r'F:\\New folder\\M2_cnt_Estemhal_Yearly_V5.csv',
                  columns_to_delete,
                  chunk,
                  skip_rows)
    dict_of_year = {1396:1,1397:1, 1398:1, 1392:1, 1395:1, 1394:1, 1393:1, 1399:1}
    column_date = 'DAYDATEOBS'
    # base_year = null
    
    df, xtr, ytr, xtes, ytes, xval, yval = step1.splitt(column_date,
                                                        dict_of_year)
    
    s_path = f'e:\dataframes\{save_name}.pkl'
    #df.to_csv(s_path, sep = '\t')
    df.to_pickle(s_path)

ctod = ['CNTESTEMHAL', 'INTCUSTID', 'TEL', 'MAXFCL', 'NAME', 'BIRTHLOCATIONCITY',
                   'PASSNO', 'ECONOMICALCODE', 'CREATEDATE', 'FIRSTACCCREATEDATE', 'ISSUEDATE',
                   'BIRTHDATE', 'BIRTHCITYCODE_ID'] 


#   



for i in range(7,8):
    a = i * 1000000
    print(a)
    now1 = time.time()
    save_p = f'{i}Ms'
    modoel_creator(ctod, 1000000, a, save_p)
    print(time.time() - now1)



# %%%% load  dataframe from a pickle file 
# =============================

# if you have a pkl dataframe and dont want load from big data files

random_sample_df1 = pd.read_pickle(r'e:\dataframes\random_sample1.pkl')
random_sample_df2 = pd.read_pickle(r'e:\dataframes\random_sample2.pkl')


sample_df_without_1399 = pd.read_pickle(r'e:\dataframes\98problem\withot_1399_daydate.pkl')
sample_df_with_1399 = pd.read_pickle(r'e:\dataframes\98problem\just_1399_with_daydate.pkl')

sample_df = sample_df_without_1399

# ---------------------
# -- SAMPLE DATAFRAME -
# ---------------------



# split data to [MAXFCL == 1 2] + [MAXFCL == 3 & 4]
data_34 = pd.concat([sample_df[sample_df['MAXFCL'] == 3],
           sample_df[sample_df['MAXFCL'] == 4]])
data_34 = data_34[:]
print('label 1 Count: ',data_34.shape[0])

data_12 = pd.concat([sample_df[sample_df['MAXFCL'] == 1],
           sample_df[sample_df['MAXFCL'] == 2]])
data_12 = data_12[:int(data_34.shape[0]*1.1)]
print('label 0 Count: ', data_12.shape[0])

# concatenate them to create TOYDATASET
sample_df = pd.concat([data_34,data_12])
print('all data: ', sample_df.shape[0])




# make LABEL for maxfcl:3,4 == 1 and maxfcl: 1,2 ==0 
maxfcl = list(sample_df['MAXFCL'])
estem = list(sample_df['CNTESTEMHAL'])
y = []
for i in range(len(maxfcl)):
    yy = 1 if (maxfcl[i] == 3 or maxfcl[i] == 4) or estem[i]>0 else 0
    y.append(yy)
y = np.array(y)

# insert y as Target column to DataFrame
sample_df['target'] = y
print('unique of maxFcl in dataset', sample_df['MAXFCL'].unique())
## print('difference in 1 and 0 for estemhal',sample_df['target'].value_counts())




print('--- number of features columns', sample_df.shape[1])


column_to_delete = ['CNTESTEMHAL', 'INTCUSTID', 'TEL', 'MAXFCL', 'NAME', 'BIRTHLOCATIONCITY',
                   'PASSNO', 'ECONOMICALCODE', 'CREATEDATE', 'FIRSTACCCREATEDATE', 'ISSUEDATE',
                   'BIRTHDATE', 'BIRTHCITYCODE_ID']

sample_df = sample_df.drop(column_to_delete, axis=1)




sample_df = data_splitter_years(sample_df,
                    'DAYDATEOBS',
                    dict_of_year)   #### ------- key ----- ####

print('nan values', sample_df.isna().sum().sum())
sample_df = sample_df.fillna(0)
print('nan values', sample_df.isna().sum().sum())


# del sample_df['DAYDATEOBS']    ###############################



print('--- number of features after delete columns', sample_df.shape[1])

# select x - y for apply on models

x_df = sample_df.loc[:, sample_df.columns != 'target']
y_df = sample_df.loc[:, sample_df.columns == 'target']
x, y = np.array(x_df), np.array(y_df)

# split data to train, test and validation data
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.3,
                                                    random_state = 255,
                                                    shuffle = True)
_, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.2,
                                                  random_state = 255,
                                                  shuffle = True)

print(f'xtrain {x_train.shape}\nytrain {y_train.shape}\n'
      f'xtest {x_test.shape}\nytest{y_test.shape}\nxval'
      f'{x_val.shape}\nyval {y_val.shape}\n')


ins = MachinLearning(sample_df,
                     x_train, y_train,
                     x_test, y_test, x_val, y_val)




ins.train_logit()
ins.train_randomforest()
ins.train_xgboost()

models_data = ins.make_report2(r'e:\res\random\\','random_data2')


# %%%% save Current DataFrame
# =============================


sample_df.to_pickle(r'e:/dataframes/specimen.pkl')




# %%%   LAZY predictor 
''' 
==============================================================================
     lazy predictor 
     train all sklearn algorithms

==============================================================================
''' 


from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.utils import shuffle

X, Y = shuffle(x_df, y_df, random_state=255)
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.8)
x_train, y_train = X[:offset], Y[:offset]
x_test, y_test = X[offset:int(offset * 1.2)], Y[int(offset * 1.2)]


reg = LazyClassifier(verbose=0,
                    ignore_warnings=False,
                    custom_metric='auc',
                    predictions=False,
                    random_state=255)

models, predictions = reg.fit(x_train, x_test, y_train, y_test)

model_dict = reg.provide_models(x_train, x_test, y_train, y_test)

# %%%  model creation for each year
'''
==============================================================================
    creating model for each year to use them in stacking


==============================================================================
'''


# until 2 |
save_name = '3n'
year = 2
sample_df = pd.read_pickle(f'e:\\dataframes\\{year}Ms.pkl')


x = sample_df.loc[:, sample_df.columns != 'target']
y = sample_df.loc[:, sample_df.columns == 'target']

# split data to train, test and validation data
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.3,
                                                    random_state = 255,
                                                    shuffle = True)


_, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.2,
                                                  random_state = 255,
                                                  shuffle = True)


print(sample_df.shape, sep='\n--\n--')


ins = MachinLearning(sample_df,
                     x_train, y_train,
                     x_test, y_test, x_val, y_val)






ins.train_logit()
ins.train_randomforest()
ins.train_xgboost()

models_data = ins.make_report2(r'e:\res\eachYear_model\\',save_name)

# %%% STACKING 
# %%%% raw stacking with raw model 

clf_xg = XGBClassifier(learning_rate=0.01, n_estimators=50, seed=8,
                            colsample_bytree=1, scale_pos_weight=5
                           ,importance_type='gain', reg_alpha=0.8, reg_lambda=4.7, n_jobs=-1)
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,
                     class_weight='balanced_subsample', n_jobs=-1)
clf_lr = LogisticRegression(class_weight='balanced',
                                 n_jobs=-1,
                                 penalty='l2',max_iter = 100000, 
                                 )

clf_xg2 = XGBClassifier(learning_rate=0.1, n_estimators=100, seed=8,
                            colsample_bytree=1, scale_pos_weight=5
                           ,importance_type='gain', reg_alpha=0.8, reg_lambda=4.7, n_jobs=-1)
clf_rf2 = RandomForestClassifier(n_estimators=100, max_depth=4,
                     class_weight='balanced_subsample', n_jobs=-1)
clf_lr2 = LogisticRegression(class_weight='balanced',
                                 n_jobs=-1,
                                 penalty='l2',max_iter = 100000, 
                                 )


pipline = make_pipeline(XGBClassifier(),XGBClassifier(),XGBClassifier(),XGBClassifier(),
                        RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),
                        LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),)



#pipline = make_pipeline(clf_xg, clf_rf, clf_lr,
#                       clf_xg2, clf_rf2, clf_lr2)

#print(piplines,features)

clf_st = StackingCVClassifier(classifiers=pipline,
                            use_probas=True,
                            meta_classifier=XGBClassifier(learning_rate=0.01,
                                                          n_estimators=50,
                                                          seed=8, colsample_bytree=1,
                                                          scale_pos_weight=2,
                                                          importance_type='gain',
                                                          reg_alpha=0.9,
                                                          reg_lambda=4.7, 
                                                          n_jobs=18),
                            shuffle=True,
                            use_features_in_secondary=False,
                            random_state=255)





# LOAD dataset
sample_df_1399 = pd.read_pickle(r'e:\dataframes\98problem\just_1399_with_daydate.pkl')
sample_df_w1399 = pd.read_pickle(r'e:\dataframes\98problem\withot_1399_daydate.pkl')

x_dfw98 = sample_df_w1399.loc[:, sample_df_w1399.columns != 'target']
y_dfw98 = sample_df_w1399.loc[:, sample_df_w1399.columns == 'target']

x_df98 = sample_df_1399.loc[:, sample_df_1399.columns != 'target']
y_df98 = sample_df_1399.loc[:, sample_df_1399.columns == 'target']
_, x_val, _, y_val = train_test_split(x_df98, y_df98,
                                                  test_size = 0.1,
                                                  random_state = 255,
                                                  shuffle = True)
print('DAYDATEOBS' in sample_df_w1399.columns)
# =============




ytrn = y_dfw98.squeeze()
ytst = y_df98.squeeze()
clf_st.fit(x_dfw98, ytrn)


@_timeit
def evaluate(model, x, y):
    kfold = model_selection.KFold(n_splits = 3, random_state = 255, shuffle = True)
    
    acc = model_selection.cross_val_score(model, x, y, cv = kfold, scoring = 'accuracy')
    
    roc = model_selection.cross_val_score(model, x, y, cv = kfold, scoring = 'roc_auc')
    
    pred = model.predict(x)
    matrix = confusion_matrix(y, pred)
    
    report = classification_report(y, pred)
    # print(list(zip(y_test,pred)))         
    res = (f'\naccuracy: {acc.mean()}\nroc_auc: {roc.mean()}\n \
    \nconfusion matrix:\n {matrix}\nReport:\n {report}\n')

    # print(res)
    
    return res,matrix


pred_test_data = clf_st.predict(x_df98)
pred_train_data= clf_st.predict_proba(x_dfw98)

evaluate_st, cm_st = evaluate(clf_st, x_df98, ytst)

predst = clf_st.predict(x_df98)

pred = clf_st.predict(x_df98)
matrix = confusion_matrix(ytst, pred)

report = classification_report(ytst, pred)

with open('very_raw_stacking.txt', 'w') as f:
    print(str(matrix), str(report), file=f,sep = '\n')
    
    
# save_model
with open(r'e:\res\raw_stacking', 'wb') as file:
    pickle.dump(clf_st, file)




# %%%% stacking class test
''' -------------------------------------------------------------------------
==============================================================================

    using stacking algorithm {sequence inputed algorithm} and 
    train it on 98 


==============================================================================
''' 

with open(r'E:\res\eachYear_model\2m_xg_PickleModel', 'rb') as f:
    xg0 = pickle.load(f)
with open(r'E:\res\eachYear_model\f0_xg_PickleModel', 'rb') as f:
    xg1 = pickle.load(f)
with open(r'E:\res\eachYear_model\f1_xg_PickleModel', 'rb') as f:
    xg2 = pickle.load(f)
with open(r'E:\res\98_98\with_Vzn2______xg_PickleModel', 'rb') as f:
    xg3 = pickle.load(f)
    
    
with open(r'E:\res\eachYear_model\2m_rf_PickleModel', 'rb') as f:
    rf0 = pickle.load(f)
with open(r'E:\res\eachYear_model\f1_rf_PickleModel', 'rb') as f:
    rf1 = pickle.load(f)
with open(r'E:\res\eachYear_model\f1_rf_PickleModel', 'rb') as f:
    rf2 = pickle.load(f)
with open(r'E:\res\98_98\with_Vzn2______rf_PickleModel', 'rb') as f:
    rf3 = pickle.load(f)
    
    
with open(r'E:\res\eachYear_model\2m_lr_PickleModel', 'rb') as f:
    lr0 = pickle.load(f)
with open(r'E:\res\eachYear_model\f1_lr_PickleModel', 'rb') as f:
    lr1 = pickle.load(f)
with open(r'E:\res\eachYear_model\f1_lr_PickleModel', 'rb') as f:
    lr2 = pickle.load(f)
with open(r'E:\res\98_98\with_Vzn2______lr_PickleModel', 'rb') as f:
    lr3 = pickle.load(f)









# LOAD dataset
sample_df_1399 = pd.read_pickle(r'e:\dataframes\98problem\just_1399_with_daydate.pkl')
sample_df_w1399 = pd.read_pickle(r'e:\dataframes\98problem\withot_1399_daydate.pkl')
# =============


# ************************* feature select *******************
# ***********************************************************
# ==== load feature importance LR/XG/RF ========
fi = np.loadtxt(r'E:\res\to_having_feature_importance_98\REPORTLR_feature_importance.txt', dtype='str')
fi = np.loadtxt(r'E:\res\to_having_feature_importance_98\REPORTXG_feature_importance.txt', dtype='str')
fi = np.loadtxt(r'E:\res\to_having_feature_importance_98\REPORTRF_feature_importance.txt', dtype='str')
# =============

fi = np.insert(fi,0,'target')
fi = np.unique(fi)

# _if we want to delete DAYDATEOBS to FI
daydate_index = int(np.where(fi == 'DAYDATEOBS')[0])
fi = np.delete(fi,daydate_index)
# FIlr = np.insert(models_data[0][2],0,'target')
# FIxg = np.insert(models_data[0][0],0,'target')
# FIrf = np.insert(models_data[0][1],0,'target')
# fi = np.insert(fi,0,'DAYDATEOBS')

# _if we want to add DAYDATEOBS to FI
fi = np.insert(fi,0,'DAYDATEOBS')

sample_df_1399 = sample_df_1399.loc[:,fi]
sample_df_w1399 = sample_df_w1399.loc[:,fi]

# ***********************************************************
# ***********************************************************







filr = np.loadtxt(r'E:\res\to_having_feature_importance_98\REPORTLR_feature_importance.txt', dtype='str')
fixg = np.loadtxt(r'E:\res\to_having_feature_importance_98\REPORTXG_feature_importance.txt', dtype='str')
firf = np.loadtxt(r'E:\res\to_having_feature_importance_98\REPORTRF_feature_importance.txt', dtype='str')
# =============


fi = np.insert(fi,0,'target')
fi = np.unique(fi)

# _if we want to delete DAYDATEOBS to FI
daydate_index = int(np.where(fi == 'DAYDATEOBS')[0])
fi = np.delete(fi,daydate_index)
# FIlr = np.insert(models_data[0][2],0,'target')
# FIxg = np.insert(models_data[0][0],0,'target')
# FIrf = np.insert(models_data[0][1],0,'target')
# fi = np.insert(fi,0,'DAYDATEOBS')

# _if we want to add DAYDATEOBS to FI
fi = np.insert(fi,0,'DAYDATEOBS')

sample_df_1399 = sample_df_1399.loc[:,fi]
sample_df_w1399 = sample_df_w1399.loc[:,fi]






x_dfw98 = sample_df_w1399.loc[:, sample_df_w1399.columns != 'target']
y_dfw98 = sample_df_w1399.loc[:, sample_df_w1399.columns == 'target']

x_df98 = sample_df_1399.loc[:, sample_df_1399.columns != 'target']
y_df98 = sample_df_1399.loc[:, sample_df_1399.columns == 'target']
_, x_val, _, y_val = train_test_split(x_df98, y_df98,
                                                  test_size = 0.1,
                                                  random_state = 255,
                                                  shuffle = True)
print('DAYDATEOBS' in sample_df_w1399.columns)









ins = MachinLearning(sample_df_w1399,
                     x_dfw98, y_dfw98,
                     x_df98, y_df98, x_val, y_val)

models = [xg3, rf3, lr3,
          rf0, rf1, rf2,
          xg0, xg1, xg2,
          lr0, lr1, lr2]

# models = [lr0, xg0, rf2]
fs = [fixg, firf, filr, firf, firf, firf, fixg, fixg, fixg, filr, filr, filr]



now1 = time.time()
ins.stacking(models, fs)
print(time.time() - now1)


ins.train_xgboost()
ins.train_logit()
ins.train_randomforest()

models_data = ins.make_report2(r'e:\res\stacking\\','with12_FS')





# %%% RUN From Jason
# ============================
import sys
sys.path.insert(0, r'C:\Users\sargoli\project')
jpath = r'C:\Users\sargoli\project\json_.json'
import newEra, json

a = newEra.RUN(jpath)




# %%% Vaeziaan 

'''
==============
=====
        VAEZIAN
=====
==============
'''

import pandas as pd
import numpy as np
# from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN,SMOTENC
# from imblearn.under_sampling import NearMiss,ClusterCentroids
# from imblearn.metrics import geometric_mean_score,sensitivity_score,classification_report_imbalanced
# import warnings,os
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import MinMaxScaler,normalize,RobustScaler
#from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
# import xgboost
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, f1_score, auc, roc_curve
# warnings.filterwarnings("ignore")


import pickle
# @vzn: load Models:
# clf_X = pickle.load(open("XGBoost_Shakhsi.p", "rb"))
# booster = xgboost.Booster()
# booster.load_model("XGBoost_Shakhsi.p")


# @vzn: load selected features:
sel_fet_X = pickle.load(open("selc_feat_Xgb_Shakhsi.p", "rb"))
sel_fet_R = pickle.load(open("selc_feat_RF_Shakhsi.p", "rb"))
sel_fet_L = pickle.load(open("selc_feat_Logit_Shakhsi.p", "rb"))


# taghir firstDateCn az tarikh be fasele rozhaye beyn firstDateCn ~ DAYDATEOBS
def change_date(X, ddobs):
    fdate=str(X)
    tday=0
    yearb=int(str(ddobs)[:4])
    #print('yearb', yearb)
    if (float(fdate) > 13000000):
        year=int(fdate[:4])
        month=int(fdate[4:6])
        day=int(fdate[6:8])
        year=(yearb-year) * 365
        month=(1-month) * 30
        day=(1-day)
        tday=year+month+day
    return max(0,tday)

train = pd.read_csv(r'F:\New folder\M2_Risk_Final_Features_Shakhsi_V2.csv',
                                delimiter=',', chunksize=50000, skiprows = range(1,3))
df_train = next(train)

test = pd.read_csv(r'F:\New folder\M2_Risk_Final_Features_Shakhsi_V2.csv',
                                delimiter=',', chunksize=10000, skiprows = range(1,500000))

df_test = next(test)


df_test["firstDateCN"] = df_test[["firstDateCN", "DAYDATEOBS"]].apply(
    lambda row: change_date(row["firstDateCN"], row["DAYDATEOBS"]), axis=1)
df_train["firstDateCN"] = df_train[["firstDateCN", "DAYDATEOBS"]].apply(
    lambda row: change_date(row["firstDateCN"], row["DAYDATEOBS"]), axis=1)



# --------------------------------------------------------------------
# split data to [MAXFCL == 1 2] + [MAXFCL == 3 & 4]
data_34 = pd.concat([df_train[df_train['MAXFCL'] == 3],
           df_train[df_train['MAXFCL'] == 4]])
data_34 = data_34[:]
print('label 1 Count: ',data_34.shape[0])

data_12 = pd.concat([df_train[df_train['MAXFCL'] == 1],
           df_train[df_train['MAXFCL'] == 2]])
data_12 = data_12[:data_34.shape[0]]
print('label 0 Count: ', data_12.shape[0])

# concatenate them to create TOYDATASET
sample_df = pd.concat([data_34,data_12])
print('all data: ', sample_df.shape[0])




# make LABEL for maxfcl:3,4 == 1 and maxfcl: 1,2 ==0 
maxfcl = list(df_train['MAXFCL'])
estem = list(df_train['CNTESTEMHAL'])
y = []
for i in range(len(maxfcl)):
    yy = 1 if (maxfcl[i] == 3 or maxfcl[i] == 4) or estem[i]>0 else 0
    y.append(yy)
y = np.array(y)

# insert y as Target column to DataFrame
df_train['target'] = y
print('unique of maxFcl in dataset', sample_df['MAXFCL'].unique())
## print('difference in 1 and 0 for estemhal',sample_df['target'].value_counts())
# --------------------------------------------------------------------

data_34 = pd.concat([df_test[df_test['MAXFCL'] == 3],
           df_test[df_test['MAXFCL'] == 4]])
data_34 = data_34[:]
print('label 1 Count: ',data_34.shape[0])

data_12 = pd.concat([df_test[df_test['MAXFCL'] == 1],
           df_test[df_test['MAXFCL'] == 2]])
data_12 = data_12[:data_34.shape[0]]
print('label 0 Count: ', data_12.shape[0])

# concatenate them to create TOYDATASET
sample_df = pd.concat([data_34,data_12])
print('all data: ', sample_df.shape[0])




# make LABEL for maxfcl:3,4 == 1 and maxfcl: 1,2 ==0 
maxfcl = list(df_test['MAXFCL'])
estem = list(df_test['CNTESTEMHAL'])
y = []
for i in range(len(maxfcl)):
    yy = 1 if (maxfcl[i] == 3 or maxfcl[i] == 4) or estem[i]>0 else 0
    y.append(yy)
y = np.array(y)

# insert y as Target column to DataFrame
df_test['target'] = y
print('unique of maxFcl in dataset', sample_df['MAXFCL'].unique())
## print('difference in 1 and 0 for estemhal',sample_df['target'].value_counts())







# a_a = df_test['firstDateCN'].head(n = 1000)
# b_b = df_test['DAYDATEOBS'].head(n = 1000)
# c_c = pd.concat([a_a,b_b], axis=1)



PD_Fet=set(sel_fet_X)
PD_Fet=PD_Fet | set(sel_fet_R)
PD_Fet=PD_Fet | set(sel_fet_L)
PD_Fet=list(PD_Fet)

df_train_stacking=df_train[PD_Fet].fillna(0)
df_test_stacking=df_test[PD_Fet].fillna(0)

col_list=list(df_train_stacking.columns)
sel_fet_R_i=[]
sel_fet_X_i=[]
sel_fet_L_i=[]

for k in sel_fet_X:
    sel_fet_X_i.append(col_list.index(k))
for k in sel_fet_R:
    sel_fet_R_i.append(col_list.index(k))
for k in sel_fet_L:
    sel_fet_L_i.append(col_list.index(k))



df_train_stacking.columns.values





from scipy import integrate
import matplotlib.pyplot as plt
model_pa=[]
def capcurve_plot(y_values, y_preds_proba):
    input_m={}
    num_pos_obs = np.sum(y_values)
    num_count = len(y_values)
    rate_pos_obs = float(num_pos_obs) / float(num_count)
    ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})
    xx = np.arange(num_count) / float(num_count - 1)

    y_cap = np.c_[y_values,y_preds_proba]
    y_cap_df_s = pd.DataFrame(data=y_cap)
    y_cap_df_s.index.name="index"
    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index('index', drop=True)

    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
    yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0

    percent = 0.5
    row_index = int(np.trunc(num_count * percent))

    val_y1 = yy[row_index]
    val_y2 = yy[row_index+1]
    if val_y1 == val_y2:
        val = val_y1*1.0
    else:
        val_x1 = xx[row_index]
        val_x2 = xx[row_index+1]
        val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)

    sigma_ideal = 1 * xx[num_pos_obs -  1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
    sigma_model = integrate.simps(yy,xx)
    sigma_random = integrate.simps(xx,xx)

    ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
    #ar_label = 'ar value = %s' % ar_value
    xx=xx.tolist()
    yy=yy.tolist()
    del xx[1::2]
    del xx[1::2]
    del xx[1::2]
    del xx[1::2]
    del xx[1::2]
    del xx[1::2]

    del yy[1::2]
    del yy[1::2]
    del yy[1::2]
    del yy[1::2]
    del yy[1::2]
    del yy[1::2]

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')
    ax.plot(xx,yy, color='red', label='User Model')

    ax.plot(xx,xx, color='blue', label='Random Model')
    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')

    plt.xlim(0, 1.02)
    plt.ylim(0, 1.25)
    plt.title("CAP Curve - a_r value ="+str(ar_value))
    plt.xlabel('% of the data')
    plt.ylabel('% of positive obs')
    plt.legend()
    plt.show()

    xx=np.asarray(xx)
    yy=np.asarray(yy)
    sh=xx.shape[0]
    xx=xx.reshape((sh, 1))
    yy=yy.reshape((sh, 1))
    
    user_d=np.concatenate((xx, yy), axis=1)
    input_m["Ideal"]=ideal.values
    input_m["User"]=user_d
    return input_m



# @vzn: 
# اجرای مدل استاکینگ که بر اساس سه مدل ساخته شده
# به ازای هر مدل، علاوه بر مدل اصلی 3 مدل با تنظیمات مشابه که در ان فقط وزن کلاسی متفاوت است ایجاد شده است
# در واقع استاکینگ بر اساس 12 مدل ساخته می شود
# علت این امر این است که به بالاترین کارایی مدل بخصوص در داده هاکلاس نکول برسیم   
################################## @vzn: 3 instances of Xgboost ############################## 

clf_X = pickle.load(open(r"E:\res\old\1M_xgPickleModel", "rb"))
clf_R = pickle.load(open("Random_Shakhsi.p", "rb"))
clf_L = pickle.load(open("Logit_Shakhsi.p", "rb"))


clf_X1 = XGBClassifier(learning_rate=0.01, n_estimators=500, seed=8,
                            colsample_bytree=1, scale_pos_weight=2
                           , importance_type='gain', reg_alpha=0.9, reg_lambda=4.7, n_jobs=18)

clf_X2 = XGBClassifier(learning_rate=0.01, n_estimators=500, seed=10,
                            colsample_bytree=1, scale_pos_weight=4
                           , importance_type='gain', reg_alpha=0.9, reg_lambda=4.7, n_jobs=18)

clf_X3 = XGBClassifier(learning_rate=0.01, n_estimators=500, seed=8,
                            colsample_bytree=1, scale_pos_weight=1
                           , importance_type='gain', reg_alpha=0.9, reg_lambda=4.7, n_jobs=18)

############################# @vzn: 3 instances of Random Forest #############################
clf_L1=LogisticRegression(random_state=14, class_weight={0:1, 1:4}, n_jobs=-1, max_iter=100)
clf_L2=LogisticRegression(random_state=14, class_weight={0:1, 1:2}, n_jobs=-1, max_iter=100)
clf_L3=LogisticRegression(random_state=14, class_weight={0:1, 1:4}, n_jobs=-1, max_iter=100)
########################### @vzn: 3 instances of Logistic Regression ######################### 
clf_R1=RandomForestClassifier(n_estimators=100, max_depth=3, random_state=14, 
                              class_weight={0:1, 1:1}, n_jobs=18)
clf_R2=RandomForestClassifier(n_estimators=100, max_depth=3, random_state=14, 
                              class_weight={0:1, 1:2}, n_jobs=18)
clf_R3=RandomForestClassifier(n_estimators=100, max_depth=3, random_state=14,
                              class_weight='balanced', n_jobs=18)







X_train_st=df_train_stacking.fillna(0)
X_test_st=df_test_stacking.fillna(0)

pipe_L = make_pipeline(ColumnSelector(cols=sel_fet_L_i),clf_L)
pipe_L1 = make_pipeline(ColumnSelector(cols=sel_fet_L_i),clf_L1)
pipe_L2 = make_pipeline(ColumnSelector(cols=sel_fet_L_i),clf_L2)
pipe_L3 = make_pipeline(ColumnSelector(cols=sel_fet_L_i),clf_L3)

pipe_X = make_pipeline(ColumnSelector(cols=sel_fet_X_i),clf_X)
pipe_X1 = make_pipeline(ColumnSelector(cols=sel_fet_X_i),clf_X1)
pipe_X2 = make_pipeline(ColumnSelector(cols=sel_fet_X_i),clf_X2)
pipe_X3 = make_pipeline(ColumnSelector(cols=sel_fet_X_i),clf_X3)

pipe_R = make_pipeline(ColumnSelector(cols=sel_fet_R_i),clf_R)
pipe_R1 = make_pipeline(ColumnSelector(cols=sel_fet_R_i),clf_R1)
pipe_R2 = make_pipeline(ColumnSelector(cols=sel_fet_R_i),clf_R2)
pipe_R3 = make_pipeline(ColumnSelector(cols=sel_fet_R_i),clf_R3)


sclf = StackingCVClassifier(classifiers=[pipe_L, pipe_L1, pipe_L2, pipe_L3,
                                         pipe_X1, pipe_X2, pipe_X3,
                                         pipe_R, pipe_R1, pipe_R2, pipe_R3],
                            use_probas=True,
                            meta_classifier=XGBClassifier(learning_rate=0.01,
                                                          n_estimators=50,
                                                          seed=8, colsample_bytree=1,
                                                          scale_pos_weight=2,
                                                          importance_type='gain',
                                                          reg_alpha=0.9,
                                                          reg_lambda=4.7, 
                                                          n_jobs=18),
                            shuffle=True,
                            use_features_in_secondary=False,
                            random_state=42)


sclf.fit(X_train_st, df_train['target'])
# @vzn: Evaluation by testset:
# @vzn: predictions:
pred = sclf.predict(X_test_st)
pred1=sclf.predict_proba(X_test_st)
predt=pred1
#print(classification_report_imbalanced(df_test['target'], pred))
# @vzn: AUC
fpr, tpr, thresholds = roc_curve(df_test['target'], pred1[:, 1], pos_label=1)
AUC = auc(fpr, tpr)
print ("AUC:", AUC)
# @vzn: CONFUSION MATRIX
print(confusion_matrix(df_test['target'],pred))   
input_m, ar_value = capcurve_plot(y_values=df_test['target'], y_preds_proba=pred1[:, 1])
model_pa.append(input_m)
# @vzn: Validation by trainset:
# @vzn: predictions:
pred = sclf.predict(X_train_st)
pred1=sclf.predict_proba(X_train_st)
#print(classification_report_imbalanced(df_train['target'], pred))
# @vzn: AUC
fpr, tpr, thresholds = roc_curve(df_train['target'], pred,pos_label=1)
AUC = auc(fpr, tpr)
print ("AUC:", AUC)
# @vzn: CONFUSION MATRIX
print(confusion_matrix(df_train['target'], pred)) 















# %%% 1398
# %%%% stacking on 98 


pipline = make_pipeline(XGBClassifier(),XGBClassifier(),XGBClassifier(),XGBClassifier(),
                        RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),RandomForestClassifier(),
                        LogisticRegression(),LogisticRegression(),LogisticRegression(),LogisticRegression(),)


clf_st = StackingCVClassifier(classifiers=pipline,
                            use_probas=True,
                            meta_classifier=XGBClassifier(learning_rate=0.01,
                                                          n_estimators=50,
                                                          seed=8, colsample_bytree=1,
                                                          scale_pos_weight=2,
                                                          importance_type='gain',
                                                          reg_alpha=0.9,
                                                          reg_lambda=4.7, 
                                                          n_jobs=18),
                            shuffle=True,
                            use_features_in_secondary=False,
                            random_state=255)




ytrn = y_dfw98.squeeze() # input y to stackingcsv must PANDAS SERIES type 
ytst = y_df98.squeeze()



clf_st.fit(x_dfw98, ytrn)



# %%%% DAYDATEOBS
'''
==============================================================================
            wanting to know impact of DAYDATEOBS in train 
==============================================================================
''' 

# LOAD dataset
sample_df_1399 = pd.read_pickle(r'e:\dataframes\1399_daydate.pkl')
sample_df_w1399 = pd.read_pickle(r'e:\dataframes\withot_1399_daydate.pkl')
# =============


# ==== load feature importance LR/XG/RF ========
fi = np.loadtxt(r'E:\res\to_having_feature_importance_98\REPORTLR_feature_importance.txt', dtype='str')
fi = np.loadtxt(r'E:\res\to_having_feature_importance_98\REPORTXG_feature_importance.txt', dtype='str')
fi = np.loadtxt(r'E:\res\to_having_feature_importance_98\REPORTRF_feature_importance.txt', dtype='str')
# =============



fi = np.insert(fi,0,'target')
fi = np.unique(fi)


# _if we want to delete DAYDATEOBS to FI
daydate_index = int(np.where(fi == 'DAYDATEOBS')[0])
fi = np.delete(fi,daydate_index)
# FIlr = np.insert(models_data[0][2],0,'target')
# FIxg = np.insert(models_data[0][0],0,'target')
# FIrf = np.insert(models_data[0][1],0,'target')
# fi = np.insert(fi,0,'DAYDATEOBS')

# _if we want to add DAYDATEOBS to FI
fi = np.insert(fi,0,'DAYDATEOBS')

sample_df_1399 = sample_df_1399.loc[:,fi]
sample_df_w1399 = sample_df_w1399.loc[:,fi]


x_dfw98 = sample_df_w1399.loc[:, sample_df_w1399.columns != 'target']
y_dfw98 = sample_df_w1399.loc[:, sample_df_w1399.columns == 'target']

x_df98 = sample_df_1399.loc[:, sample_df_1399.columns != 'target']
y_df98 = sample_df_1399.loc[:, sample_df_1399.columns == 'target']
_, x_val, _, y_val = train_test_split(x_df98, y_df98,
                                                  test_size = 0.1,
                                                  random_state = 255,
                                                  shuffle = True)
print('DAYDATEOBS' in sample_df_w1399.columns)


ins = MachinLearning(sample_df_w1399,
                     x_dfw98, y_dfw98,
                     x_df98, y_df98, x_val, y_val)

ins.train_logit()

ins.train_randomforest()

ins.train_xgboost()

models_data = ins.make_report2(r'e:\res\98_98\\','rf_daydate_')


# %%%% 98 with & without firstdate VZN
''' -------------------------------------------------------------------------
==============================================================================

         want to know impact of Date and converted 


==============================================================================
''' 
def change_date(X, ddobs):
    fdate=str(X)
    tday=0
    yearb=int(str(ddobs)[:4])
    if (float(fdate) > 13000000):
        year=int(fdate[:4])
        month=int(fdate[4:6])
        day=int(fdate[6:8])
        year=(yearb-year) * 365
        month=(1-month) * 30
        day=(1-day)
        tday=year+month+day
    return max(0,tday)


# LOAD dataset
sample_df_1399 = pd.read_pickle(r'e:\dataframes\98problem\just_1399_with_daydate.pkl')
sample_df_w1399 = pd.read_pickle(r'e:\dataframes\98problem\withot_1399_daydate.pkl')
# =============


sample_df_1399["firstDateCN"] = sample_df_1399[["firstDateCN", "DAYDATEOBS"]].apply(
    lambda row: change_date(row["firstDateCN"], row["DAYDATEOBS"]), axis=1)
sample_df_w1399["firstDateCN"] = sample_df_w1399[["firstDateCN", "DAYDATEOBS"]].apply(
    lambda row: change_date(row["firstDateCN"], row["DAYDATEOBS"]), axis=1)


x_dfw98 = sample_df_w1399.loc[:, sample_df_w1399.columns != 'target']
y_dfw98 = sample_df_w1399.loc[:, sample_df_w1399.columns == 'target']

x_df98 = sample_df_1399.loc[:, sample_df_1399.columns != 'target']
y_df98 = sample_df_1399.loc[:, sample_df_1399.columns == 'target']
_, x_val, _, y_val = train_test_split(x_df98, y_df98,
                                                  test_size = 0.1,
                                                  random_state = 255,
                                                  shuffle = True)



ins = MachinLearning(sample_df_w1399,
                     x_dfw98, y_dfw98,
                     x_df98, y_df98, x_val, y_val)

ins.train_logit()

ins.train_randomforest()

ins.train_xgboost()

models_data = ins.make_report2(r'e:\res\98_98\\','out_Vzn2_____')



# %%%% test 98 unbalanced with hyperparameter tuning logit 

load_path = r'E:\res\stacking\probe_true_FSlr_with_each_models_st_PickleModel'

with open(load_path, 'rb') as f:
    lr = pickle.load(f)

class_percent = dict(df98_csv.target.value_counts()/df98_csv.shape[0])


# -------------------------
# normalization 
# -------------------------
x, xtest = normalization(x_train, x_test)


# -------------------------
# train 
# -------------------------

clf98_lr = LogisticRegression(C=100, solver='newton-cg',
                                     class_weight={0:8,1:1},
                                     n_jobs=-1,
                                     penalty='l2',max_iter = 100000, 
                                     )


clf98_xg = XGBClassifier(learning_rate=0.01, n_estimators=100, seed=8,
                            class_weight = class_percent,
                            colsample_bytree=1, scale_pos_weight=5
                           ,importance_type='gain', reg_alpha=0.8, reg_lambda=4.7, n_jobs=-1)


clf98_rf =RandomForestClassifier(n_estimators=100, max_depth=4,
                     class_weight='balanced_subsample', n_jobs=-1)


clf98_lr.fit(x, y_train)
clf98_xg.fit(x_train, y_train)
clf98_rf.fit(x_train, y_train)


a, b = evaluate(clf98_lr, xtest, y_test)




from imblearn.metrics import classification_report_imbalanced


from sklearn.metrics import fbeta_score


pred = lr.predict(xtest)
matrix = confusion_matrix(y_test, pred)

report = classification_report(y_test, pred)

print('classification_report_imbalanced:\n',
      classification_report_imbalanced(y_test, pred))


fbeta_score(y_test, pred, beta = 8)
# print(res)


from sklearn import metrics
metrics.f1_score(y_test, pred)


with open(r'e:\report\q.txt', 'w') as f:
    print(matrix, report,sep='\n\n', file=f)




with open(r'e:\report\98unbrf0.txt', 'w') as f:
    print(a,b,sep='\n\n', file=f)

# %%% Grid search 


# load data
sample_df_without_1399 = pd.read_pickle(r'e:\dataframes\98problem\withot_1399_daydate.pkl')
sample_df_with_1399 = pd.read_pickle(r'e:\dataframes\98problem\just_1399_with_daydate.pkl')



sample_df = sample_df_without_1399
x_df = sample_df.loc[:, sample_df.columns != 'target']
y_df = sample_df.loc[:, sample_df.columns == 'target']
x, y = np.array(x_df), np.array(y_df)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.01,
                                                    random_state = 255,
                                                    shuffle = True)
_, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.2,
                                                  random_state = 255,
                                                  shuffle = True)


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


sample_df_w98 = pd.read_csv(r'e:\dataframes\98problem\gheyr_98.csv')
sample_df_98 = pd.read_csv(r'e:\dataframes\98problem\98_unb.csv')


x_df = sample_df_w98.loc[:, sample_df_w98.columns != 'target']
y_df = sample_df_w98.loc[:, sample_df_w98.columns == 'target']
x_train, y_train = np.array(x_df), np.array(y_df)


x_df = sample_df_98.loc[:, sample_df_98.columns != 'target']
y_df = sample_df_98.loc[:, sample_df_98.columns == 'target']
x_test, y_test = np.array(x_df), np.array(y_df)


# %%%% LOGISTIC
''' 
==============================================================================
LR --> using Grid search to HyperParameter Tunning in LOGIT Algorithm
best params {'C': 100, 'penalty': 'l2', 'solver': 'newton-cg'}
==============================================================================
''' 

# STANDARD
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

model = LogisticRegression()

paramslr = {}
paramslr['solver'] = ['newton-cg', 'lbfgs', 'linlinear', 'sag', 'saga']
paramslr['penalty'] = ['l1', 'l2']
paramslr['C'] = [0.01, 0.1, 1, 10, 100]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)

searchlr = GridSearchCV(model, paramslr, cv=cv)


now1 = time.time()
result = searchlr.fit(x_train_s[152000:153000], y_train[152000:153000])
print(time.time() - now1)



# unique value of a np array
u,c = np.unique(y_train[152200:152500],return_counts=True)
c,u

# best_params: 
# {'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'}

with open('report_lr_gird.txt', 'w') as f:
        khat = '\n' + ('***' * 40)

        print('best score: ',result.best_score_, file=f)
        print('best params', result.best_params_, file=f)
        print('best estimator_', result.best_estimator_,khat, file=f)
        

        print(*list(zip(result.cv_results_['params'],result.cv_results_['mean_test_score'])),sep='\n\n', file=f)
        print(khat, 'cv_result', result.cv_results_.items(), khat,  file=f)

        


# %%%% XG
''' 
==============================================================================
XG --> using Grid search to HyperParameter Tunning in XG Algorithm
==============================================================================
''' 

import pandas as  pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split




params = {}
params['learning_rate'] = [0.01, 0.1]
params['reg_alpha'] = [0.001, 0.01, 0.1, 1, 10]
params['max_depth'] = [5]
params['gamma'] = [0.0, 0.1, 0.2, 0.3]
params['colsample_bytree'] = [0.3, 0.4, 0.5, 0.7]
params['min_child_weight'] = [1, 3, 5, 7]


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)


model = XGBClassifier()
search = GridSearchCV(model, params)


result = search.fit(x_train[152200:152500], y_train[152200:152500])

with open('report_xg_gird.txt', 'w') as f:
        khat = '\n' + ('***' * 40)

        print('best score: ',result.best_score_, file=f)
        print('best params', result.best_params_, file=f)
        print('best estimator_', result.best_estimator_,khat, file=f)
        

        print(*list(zip(result.cv_results_['params'],result.cv_results_['mean_test_score'])),sep='\n\n', file=f)
        print(khat, 'cv_result', result.cv_results_.items(), file=f)

        
        


# %%%% Random Forest
''' 
==============================================================================
RF --> using Grid search to HyperParameter Tunning in Random Forest Algorithm
==============================================================================
''' 

import pandas as  pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split

# load data
sample_df_without_1399 = pd.read_pickle(r'e:\dataframes\98problem\withot_1399_daydate.pkl')
sample_df_with_1399 = pd.read_pickle(r'e:\dataframes\98problem\just_1399_with_daydate.pkl')
sample_df = sample_df_without_1399
x_df = sample_df.loc[:, sample_df.columns != 'target']
y_df = sample_df.loc[:, sample_df.columns == 'target']
x, y = np.array(x_df), np.array(y_df)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.1,
                                                    random_state = 255,
                                                    shuffle = True)
_, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.2,
                                                  random_state = 255,
                                                  shuffle = True)



params = {}
params['max_depth'] = [3,5,7,14,22,29]
params['min_samples_split'] = [0.1,0.4,0.8]
params['max_features'] = [1,10,100,200]
params['min_samples_leaf'] = [0.1, 0.3,0.5]
params['max_leaf_nodes '] = [3, 5, 8, 12]
params['n_estimators'] = [1, 3, 5, 7]
params['max_samples'] = [10,100,150,250,330]
#params['bootstrap'] = [1, 3, 5, 7]
#params['criterion'] = [1, 3, 5, 7]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)


model = RandomForestClassifier()

search = GridSearchCV(model, params, cv=cv)




result = search.fit(x_test[152200:152500], y_test[152200:152500])


with open('report_rf_gird.txt', 'w') as f:
        khat = '\n' + ('***' * 40)

        print('best score: ',result.best_score_, file=f)
        print('best params', result.best_params_, file=f)
        print('best estimator_', result.best_estimator_,khat, file=f)
        
        

        print(*list(zip(result.cv_results_['params'],result.cv_results_['mean_test_score'])),sep='\n\n', file=f)
        print(khat, 'cv_result', result.cv_results_.items(), file=f)



# %%% FEATURE SELECTIONS
# -------------------------------
#   FINAL Forward & backward    !
#--------------------------------

model_used = XGBClassifier
# XGBClassifier
# LogisticRegression
def cal_acc(xdf, ydf):
    x, y = np.array(xdf), np.array(ydf)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.3,
                                                    shuffle = True,
                                                    random_state = 255)
    _, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.1,
                                                  random_state = 255)
    clf_rf = model_used(n_estimators=100, max_depth=4,
                                 random_state=14, class_weight='balanced_subsample', n_jobs=18)
    clf_rf.fit(x_train, y_train)
    kfold = model_selection.KFold(n_splits = 3, random_state = 255, shuffle = True)
    acc = model_selection.cross_val_score(clf_rf, x_test, y_test, cv = kfold, scoring = 'accuracy')
    
    return acc.mean()



def cal_f1(xdf, ydf):
    x, y = np.array(xdf), np.array(ydf)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.3,
                                                    shuffle = True,
                                                    random_state = 255)
    _, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.1,
                                                  random_state = 255)
    model = model_used(n_estimators=100, max_depth=4,
                                 random_state=14, class_weight='balanced_subsample', n_jobs=18)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    report = classification_report(y_test, pred, output_dict= True)
    f10 = report['0']['f1-score']
    f11 = report['1']['f1-score']
    f = (f10 + f11) / 2 
    return  f


def backward_selection(xdf, ydf, step, dependent = cal_f1):
    with open('feature_selection_backward.txt', 'w') as f:
        scores = []
        for i in range(0, xdf.shape[1], step):
            print( '\n', '__' * 40, '\n', file=f)
            
            if i > 0: 
                print(f'step {int(i/step)}\t', 'i: ',i, file=f)
                print(f'number of features BEFORE this epoch {xdf.shape[1]}', file=f)
    
                cols = list(xdf.columns)
    
                b = xdf[cols[0:step]]
                print('considered features: ', b.columns, file=f)
    
                columns_to_delete = [i for i in cols[0:step]]
    
                xdf.drop(columns_to_delete, axis = 1, inplace = True)
    
                a = dependent(xdf, ydf)
                print(f'-max- accuracy {max(scores)},\nepoch accuracy {a}', file=f)
    
    
                if a <= max(scores):
                    print('\n * * * score now < score past * * *', file=f)
                    xdf[b.columns] = b
                    scores.append(a)
    
                if a > max(scores):
                    scores.append(a)
                    print('\n- - - score now > score past - - -', file=f)
                print(f'number of features AFTER this epoch {xdf.shape[1]}', file=f)
            if i == 0 :
                print(f'step {i}', file=f)
                a = dependent(xdf, ydf)
                scores.append(a)
                print(f'accuracy: {a}', file=f)
    with open('features_backward.txt','w') as features:
        for i in xdf.columns:
            print(i, file=features)
    return xdf
            
            
            
            
def forward_selection(xdf, ydf, step, dependent = cal_f1):
    # FORWARD 
    # -  -
    # = = =
    empty_df = pd.DataFrame()
    scores = []
    with open('feature_selection_forward.txt', 'w') as f:
        for i in range(0, int(xdf.shape[1]/step) + 1):
            st = i * step
            end = (i * step) + step
            print( '\n', '__' * 40, '\n', file=f)  
    
            if i == 0:
                print(f'step {i}', file=f)
                data = xdf.iloc[:, st:end]
                a = dependent(data, ydf)
    
                print(data.shape, file=f)
    
                scores.append(a)
                print(f'accuracy: {a}', file=f)
                empty_df = empty_df.append(data)
    
    
            if i > 0: 
                subsample = xdf.iloc[:, :end]
    
                print(f'step {i}', file=f)
    
                cols = list(xdf.columns)
    
                b = xdf.iloc[:, st:end]
                print('considered features: ', b.columns, file=f)
    
                a = dependent(subsample, ydf)
                print(f'max accuracy {max(scores)},\nepc accuracy {a}', file=f)
    
    
                if a < max(scores):
                    scores.append(a)
                    print('\n * * * score now < score past * * *', file=f)
                    print(subsample.shape, b.shape, empty_df.shape)
    
                if a >= max(scores):
                    scores.append(a)
                    print('\n+ + + score now > score past + + +', file=f)
                    empty_df = empty_df.append(b)
    
                    print(subsample.shape, b.shape, empty_df.shape, file=f)
                print(f'number of features AFTER this epoch {empty_df.shape[1]}', file=f)
    with open('features_forward.txt','w') as features:
        for i in xdf.columns:
            print(i, file=features)
    return empty_df




# LOAD dataset
sample_df_1399 = pd.read_csv(r'e:\dataframes\98problem\98_unb.csv')
sample_df_w1399 = pd.read_csv(r'e:\dataframes\98problem\gheyr_98.csv')
# =============


x_train = sample_df_w1399.loc[:, sample_df_w1399.columns != 'target']
y_train= sample_df_w1399.loc[:, sample_df_w1399.columns == 'target']

x_test = sample_df_1399.loc[:, sample_df_1399.columns != 'target']
y_test = sample_df_1399.loc[:, sample_df_1399.columns == 'target']
_, x_val, _, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.1,
                                                  random_state = 255,
                                                  shuffle = True)





a = backward_selection(x_train[150000:160000], y_train[150000:160000], 100)

b = forward_selection(x_train[150000:160000], y_train[150000:160000], 100)













# %%% tricks  




# %%%% Sea born


cm = ins.matrix

df_cm = pd.DataFrame(cm)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted' 
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, cmap='gist_earth', annot=True, annot_kws={'size':16}, fmt='g', cbar =False)




# unique value of a np array
u,c = np.unique(sample_df_1399.columns,return_counts=True)
u[c>1]



u,c = np.unique(fi,return_counts=True)
u[c>1]



np.where(fi == 'DAYDATEOBS')
fi = np.delete(fi,90)







# %%%% stuf that removes 
def make_report(self, save_path):
        

            
        if (os.path.exists(save_path)): #for overwrite files
            os.remove(save_path)
        per ='\n' + '....' * 20 + '\n'
        
        date = str(datetime.datetime.now())
        date0, date1 = date[:10], date[11:]
        with open('REPORT' + date0 + '.txt', 'w') as file: 
            file.write('DATA FRAME description:\n')
            print('time that was created',date0, date1, file = file)
            
            
            file.write(f'shape of DataFrame: {self.df.shape[0]} rows\t{self.df.shape[1]} columns \n')
            
            # file.write(f'{per}, DataFrame describe (count, mean, std and quartile for each column :\n, {self.df.describe()}')

            a = {i: j for i in self.df.columns for j in self.df.dtypes}
            file.write(f'\nunique dtypes: {set(a.values())}\n')

            print(per, file = file)
            
            if self.clf_xg is not None:
                print('##########\nXGboost MODEL\n##########\n', file = file)
                file.write(self.evaluate_xg)
                print(f' number of features that affect on data: {self.f_xg[1]} ',file = file)
                print(' important Feature [column Name, Impact]:' ,*self.f_xg[0], sep = '\n', end = '\n', file = file)
                self.save_model(save_path + '_xg_PickleModel', self.clf_xg)
                
            print(per, file = file)
    
            if self.clf_rf is not None:
                print('##########\nRandom forest MODEL\n##########\n', file = file)
                file.write(self.evaluate_rf)
                print(f' number of features that affect on data: {self.f_rf[1]} ',file = file)
                print(' important Feature [column Name, Impact]:',*self.f_rf[0], sep = '\n', end = '\n', file = file)
                self.save_model(save_path + '_rf_PickleModel', self.clf_rf)

            print(per, file = file)
                
            if self.clf_lr is not None:
                print('##########\nLogistic regression MODEL\n##########\n', file = file)
                file.write(self.evaluate_lr)
                print(f' number of features that affect on data: {self.f_lr[1]}\n ',file = file)
                print(' important Feature [column Name, Impact]:' ,*self.f_lr[0], sep = '\n', end = '\n', file = file)
                self.save_model(save_path + '_lr_PickleModel', self.clf_lr)
            print(per, file = file)

            if self.clf_st is not None:
                print('##########\nStacking MODEL\n##########\n', file = file)
                file.write(self.evaluate_st)
                #print(f' number of features that affect on data: {self.f_lr[1]}\n ',file = file)
                #print(' important Feature [column Name, Impact]:' ,*self.f_lr[0], sep = '\n', end = '\n', file = file)
                self.save_model(save_path + '_st_PickleModel', self.clf_lr)
                    
        
        
        self.plots2(save_path)
        
        
        return [self.FNxg, self.FNrf, self.FNlr],[self.clf_xg, self.clf_rf, self.clf_lr]
       






'''
classifiers_list = [xgloaded_model, xgloaded_model2, xgloaded_model3]
sclf = StackingCVClassifier(classifiers=classifiers_list,
                                            use_probas=False,
                                            meta_classifier=XGBClassifier(learning_rate=0.1,
                                                                          n_estimators=1000,
                                                                          seed=8, colsample_bytree=1,
                                                                          scale_pos_weight=2,
                                                                          importance_type='gain',
                                                                          reg_alpha=0.9,
                                                                          reg_lambda=4.7, 
                                                                          n_jobs=-1))
'''
sclf.fit(x_train, y_train)
stacking_pred = sclf.predict(x_test)







pred1 = sclf.predict_proba(x_train)



kfold = model_selection.KFold(n_splits = 3, random_state = 255, shuffle = True)

acc = model_selection.cross_val_score(sclf, x_test, y_test, cv = kfold, scoring = 'accuracy')

roc = model_selection.cross_val_score(sclf, x_test, y_test, cv = kfold, scoring = 'roc_auc')

pred = sclf.predict(x_test)
matrix = confusion_matrix(y_test, pred)

pred = sclf.predict(x_test)
report = classification_report(y_test, pred)
# print(list(zip(y_test,pred)))         


res = (f'\naccuracy: {acc.mean()}\nroc_auc: {roc.mean()}\n \ \nconfusion matrix:\n {matrix}\nReport:\n {report}\n')

# print(res)
with open('ok' + '_' + '.txt', 'w') as file: 
    file.write('DATA FRAME description:\n')
    print('time that was created', res, sep = '\n', file = file)


# @vzn: AUC
fpr, tpr, thresholds = roc_curve(y_test, stacking_pred[:, 1], pos_label=1)
AUC = auc(fpr, tpr)
print ("AUC:", AUC)
# @vzn: CONFUSION MATRIX
print(confusion_matrix(y_test, stacking_pred))  


input_m, ar_value = capcurve_plot(y_values = y_test, y_preds_proba=pred1[:, 1])
model_pa.append(input_m)
# @vzn: Validation by trainset:
# @vzn: predictions:
    
pred = sclf.predict(x_train)
pred1 = sclf.predict_proba(x_train)
print(classification_report_imbalanced(y_train, pred1))
# @vzn: AUC
fpr, tpr, thresholds = roc_curve(y_train, pred1, pos_label=1)
AUC = auc(fpr, tpr)
print ("AUC:", AUC)
# @vzn: CONFUSION MATRIX
print(confusion_matrix(y_train, pred1)) 














