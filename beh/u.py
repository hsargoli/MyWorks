# %% LOAD LIBRARIES

# =============================================================================
# 

import os, sys
from functools import wraps
import time, joblib, glob
from pprint import pprint
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import uniform
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFECV
from category_encoders.woe import WOEEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.metrics import (
    roc_curve, precision_recall_fscore_support,
    auc, confusion_matrix,
    brier_score_loss, classification_report, matthews_corrcoef)
#from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
#from sklearn_genetic.space import Continuous, Categorical, Integer
from imblearn.over_sampling import SMOTE
from scipy.stats import randint as sp_randint
import seaborn as sb
import sklearn
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import List
# =============================================================================


# Decorator
def _timeit(func):
    @wraps(func)
    def timeit_wraps(*args, **kwargs):
        st = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        total_time = end - st
        if total_time>5:
            print(f'|>>>>> Funcion: -{func.__name__}- Took {total_time :.2f} seconds to run <<<<<<|\n  ')
        else: 
            pass
        return result 
    return timeit_wraps






#   FUNCTIONS  ----------------------------------------------------------------
#################################### CONTENT ##################################
#1  get_file_line_count        #2  get_random_sample
#4  get_Xy
#5  roc                        #6  evaluate [4 funcs]
#7  train_model                #8  capture_diagram                  
#9  cap_curve                  #10 get_metrics

#10-2   get_metrics_all
#11 fit_woe_encoder 
#12 cross_val_score_                               
#13 hyper_params_selection_by_randomizedsearchcv                  
#14 hyper_params_selection_by_randomizedsearchcv_for_all_models                     
#15 feature_selection_by_rfecv               
#16 feature_selection_by_rfecv_for_all_models            
#17 get_feature_importances
#20 get_model_feat_imps
#21 get_preds_for_evaluation
#22 evaluate_model
#25 predict
#26 predict_pd_proba
#27 get_6to1_metrics
#28 capcurve_get_points
#30 check_line
#31 data_splitter_years
#32 PD statistic: describe 

###############################################################################






#%% 1 get_file_line_count
@_timeit
def get_file_line_count(df_path):
    '''
    Parameters
    ----------
    df_path: path of dataFrame
        
    Description
    ----------
    this function counts lines of the dataset and return it
    
    '''
    with open(df_path) as f:
        for count, _ in enumerate(f): 
            pass
    return count



#%% 2 get_random_sample
@_timeit
def get_random_sample(df_path, n_sample, cols_):
    '''
    Parameters
    ----------
    df_path: path of csv dataset
    n_sample: number of samples that function taken from dataframe
    cols: which column(s) must be cosidered. we can use it for facility\non-facility  Features
    
    Description
    ----------
    The function creates random subsamples of a given CSV datframe and return it
    '''
    # cols = get_cols(is_non_facility)
    # cols.extend(['LABEL'])
    
    n_total = get_file_line_count(df_path)
    if n_total < n_sample:
        df = pd.read_csv(df_path, header=0, encoding='utf-8', usecols = cols_)[cols_]
    else:   
        skip_idxs = sorted(np.random.choice(range(1, n_total), n_total - n_sample, replace=False))        
        df = pd.read_csv(df_path, header=0, skiprows=skip_idxs, nrows=n_total, encoding='utf-8', usecols = cols_)[cols_]
        df.reset_index(inplace=True, drop=True)
    return df






#%% 4 get_Xy
@_timeit
def get_Xy(
    df_path=None,
    n_sample=None,
    target_label='LABEL',
    cols=None
):
    '''
    Parameters
    ----------
    df_path: CSV dataFrame for get X and y from it
    n_sample: if it is a number, using the get random sample Function to create a subset of dataframe with length of it {n_sample}   
    target_label: to split data into X&Y we need the target columns.
                it set by defualt (by considering Shakhsi Custgroup): LABEL
    cols: which column(s) must be cosidered. we can use it for facility Features
    
    
    Description
    ----------
    split data to x and y for train and test 
    '''

    total_cols = cols + [target_label]
    if n_sample:
        df = get_random_sample(df_path=df_path, n_sample=n_sample, cols_=total_cols)
    else:
        df = pd.read_csv(df_path, header=0, usecols=total_cols)[total_cols]
    df.fillna(0, inplace=True)
    
    X = df.loc[:, df.columns != target_label]
    y = df.loc[:, df.columns == target_label]
    
    return X, y


    
    
#%% 5   roc
def roc(y_true,
        y_pred,
        title= '_ROCcurve_',
        save_path=os.getcwd()):
    '''
    Parameters
    ----------
    y_true: True y of an instance
    y_pred: predicted y that comes from model
    title: title of chart 
    save_path: where to save ROC-chart.png
    
    Description
    ----------
    plot ROC chart and label it with ROC_AUC
    '''
    
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    AUC = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=1,
        label='ROC curve (area = %0.2f)' % AUC)    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {title} - without facility')
    plt.legend(loc='lower right')
    plt.savefig(f'{save_path}/{title}_roc_curve.png')
    # plt.show()
    return fpr, tpr, threshold

#%% 6-1  evaluate negative data in chunk  mode

@_timeit
def evaluate_chunk(sclf=[], cols=None,
             chunk_size= 300_000,
            pos_path=None,
            neg_path=None):
    '''
    Parameters
    ----------
    sclf: the model to evaluate. this model must be in mode of stackingClassifier with estimator = [RF, XGB, LOGIT]
    chunk_size: how much split data to the chunks. _chunksize_
    pos_path, neg_path : path of csv positive & negative dataFrame  to evaluate it
 
    Description
    ----------
    evaluat chunk chunk 
    set pos and negative dataFrame and it evaluate it
    especially designed for Personal Custgroup

    
    return
    ----------
    y_test, y_test_pred, y_test_proba, y_test_pred_RF, y_test_pred_XGB, y_test_pred_LOGIT
    '''



    pos_test_df = pd.read_csv(pos_path, header=0, usecols=cols+['LABEL'])[cols+['LABEL']]

    pos_test_df.fillna(0, inplace=True)


    y_test = pos_test_df['LABEL']
    pos_test_df = pos_test_df.loc[:, pos_test_df.columns.map(lambda x : x not in ['LABEL', 'INTCUSTID'])]

    y_test_pred = sclf.predict(pos_test_df)
    y_test_pred_RF = sclf.clfs_[0].predict(pos_test_df)
    y_test_pred_XGB = sclf.clfs_[1].predict(pos_test_df)
    y_test_pred_LOGIT = sclf.clfs_[2].predict(pos_test_df)
    y_test_proba = sclf.predict_proba(pos_test_df)[:, 1]

    del pos_test_df

    neg_test_df = pd.read_csv(neg_path, usecols=cols+['LABEL'], header=0, chunksize=chunk_size)


    #chunk_count = sum(1 for _ in neg_test_df) 
    #print('\n***********\n',f'we have  {chunk_count} chunk','\n***********\n')


    for i,neg_test_part_df in enumerate(neg_test_df):
        print('\n***********\n',f'loop {i} started','\n***********\n')
        neg_test_part_label_df = neg_test_part_df['LABEL']
        
        
        
        
        
        
        #neg_test_part_df = neg_test_part_df.loc[:, neg_test_part_df.columns.map(lambda x: x not in ['LABEL', 'INTCUSTID'])]
        #selected_columns = [col for col in neg_test_part_df.columns if col not in  ['LABEL', 'INTCUSTID']]        
        neg_test_part_df = neg_test_part_df[cols]
        
        neg_test_part_df.fillna(0, inplace=True)

        y_test_pred = np.append(y_test_pred, sclf.predict(neg_test_part_df))

        y_test_pred_RF = np.append(y_test_pred_RF, sclf.clfs_[0].predict(neg_test_part_df))
        y_test_pred_XGB = np.append(y_test_pred_XGB, sclf.clfs_[1].predict(neg_test_part_df))
        y_test_pred_LOGIT = np.append(y_test_pred_LOGIT, sclf.clfs_[2].predict(neg_test_part_df))
        y_test_proba = np.append(y_test_proba, sclf.predict_proba(neg_test_part_df)[:, 1])

        del neg_test_part_df

        y_test = np.append(y_test, neg_test_part_label_df)
        print(f'y test shape: {y_test.shape}, y predicted shape {y_test_pred.shape}')

    return y_test, y_test_pred, y_test_proba, y_test_pred_RF, y_test_pred_XGB, y_test_pred_LOGIT






# %%% 6-2 evaluate both 0&1 in chunk mode (ASLI)


def evaluate_(sclf=[], cols=None, pos_path=None, neg_path=None):   
    '''
    Parameters
    ----------
    sclf: the model to evaluate. this model must be in mode of stackingClassifier with estimator = [RF, XGB, LOGIT]
    chunk_size: how much split data to the chunks. _chunksize_
    pos_path, neg_path : path of csv positive & negative dataFrame  to evaluate it
 
    Description
    ----------
    evaluat chunk chunk 
    set pos and negative dataFrame and it evaluate it
    especially designed for Personal Custgroup
    '''
    
    pos_test_df = pd.read_csv(pos_path, usecols=cols + ['LABEL'], header=0, chunksize=100_000)
        
    
    y_test = np.array([])
    y_test_pred = np.array([])
    y_test_pred_RF = np.array([])
    y_test_pred_XGB= np.array([])
    y_test_pred_LOGIT= np.array([])
    y_test_proba= np.array([])
    
    
    for pos_test_part_df in pos_test_df:
        pos_test_part_df = pos_test_part_df[cols]
        pos_test_part_df['LABEL']=1
        
        pos_test_part_label_df = pos_test_part_df['LABEL']
        pos_test_part_df = pos_test_part_df.loc[:, pos_test_part_df.columns.map(lambda x : x not in ['LABEL', 'INTCUSTID'])]
                
        pos_test_part_df.fillna(0, inplace=True)
           
        y_test_pred = np.append(y_test_pred, sclf.predict(pos_test_part_df))
        y_test_pred_RF = np.append(y_test_pred_RF, sclf.clfs_[0].predict(pos_test_part_df))
        y_test_pred_XGB = np.append(y_test_pred_XGB, sclf.clfs_[1].predict(pos_test_part_df))
        y_test_pred_LOGIT = np.append(y_test_pred_LOGIT, sclf.clfs_[2].predict(pos_test_part_df))
        y_test_proba = np.append(y_test_proba, sclf.predict_proba(pos_test_part_df)[:, 1])
        
        del pos_test_part_df
        
        y_test = np.append(y_test, pos_test_part_label_df)
        print(y_test.shape, y_test_pred.shape)   

    
    
    
    neg_test_df = pd.read_csv(neg_path, usecols=cols + ['LABEL'], header=0, chunksize=100_000)

    for neg_test_part_df in neg_test_df:
        neg_test_part_df = neg_test_part_df[cols]
        neg_test_part_df['LABEL'] = 0
        
        
        neg_test_part_label_df = neg_test_part_df['LABEL']
        neg_test_part_df = neg_test_part_df.loc[:, neg_test_part_df.columns.map(lambda x : x not in ['LABEL', 'INTCUSTID'])]
                
        neg_test_part_df.fillna(0, inplace=True)
           
        y_test_pred = np.append(y_test_pred, sclf.predict(neg_test_part_df))
        y_test_pred_RF = np.append(y_test_pred_RF, sclf.clfs_[0].predict(neg_test_part_df))
        y_test_pred_XGB = np.append(y_test_pred_XGB, sclf.clfs_[1].predict(neg_test_part_df))
        y_test_pred_LOGIT = np.append(y_test_pred_LOGIT, sclf.clfs_[2].predict(neg_test_part_df))
        y_test_proba = np.append(y_test_proba, sclf.predict_proba(neg_test_part_df)[:, 1])
        
        del neg_test_part_df
        
        y_test = np.append(y_test, neg_test_part_label_df)
        print(y_test.shape, y_test_pred.shape)   

    return y_test, y_test_pred, y_test_pred_RF, y_test_pred_XGB, y_test_pred_LOGIT, y_test_proba








# %%% 6-3 evaluate solo (NOT CHUNK) for small dataset 
@_timeit
def evaluate_solo2(sclf=[], cols=None,
             smote=False,
             test_path=None,
             out_path=os.getcwd()+'\\',
             prob=False):
    '''
    Parameters
    ----------
    sclf: the model to evaluate.
    cols: which column(s) must be cosidered. we can use it for facility Features
    smote: use SMOTE technique to overSample dataFrame
    test_path: path of csv dataFrame to evaluate it
    out_path: in the end of use this function it save the csv of target in each
                model. this parameter tell the save directory
    prob
    
    Description
    ----------
    evaluate stackingClassifier. return y_test and y_pred of each estimator {randomForest, XGboost, Logistic Regression}
    
    return
    ----------
    dataFrame of : y_test, y_test_pred, y_test_proba, y_test_pred_RF, y_test_pred_XGB, y_test_pred_LOGIT
    '''

    cols_ = cols + ['LABEL']
    
    test_df = pd.read_csv(test_path, header=0, usecols=cols_)[cols_]
    test_df.fillna(0, inplace=True)
    X_test = test_df.loc[:, ~ test_df.columns.map(str).isin([str('LABEL')])]
    y_test = test_df[str('LABEL')]
    if smote == True:
        oversample = SMOTE()
        X_test, y_test = oversample.fit_resample(X_test, y_test)
        print('\n','-' * 20, '\n','SMOTE applied, evaluateFunc','\n','-' * 20, '\n')
        
    y_test_pred = sclf.predict(X_test)
    y_test_proba = sclf.predict_proba(X_test)[:, 1]

    sample_df = pd.DataFrame()
    
    
    if prob:
        y_test_proba = sclf.predict_proba(X_test)[:, 1]
        sample_df['y_proba'] = y_test_proba
    
    sample_df['y_test_pred_RF'] = sclf.clfs_[0].predict(X_test)
    sample_df['y_test_pred_XGB '] = sclf.clfs_[1].predict(X_test)
    sample_df['y_test_pred_LOGIT '] = sclf.clfs_[2].predict(X_test)
    sample_df['y_test_org'] = y_test
    sample_df['y_test_pred'] = y_test_pred
    sample_df['y_proba'] = y_test_proba
    
    sample_df.to_csv(out_path + 'out_ys.csv', index = False)
    
    return sample_df
    


#%% 7  train_model
@_timeit
def train_model(X=None,
                y=None,
                smote = False,
                model_name=None,
                cat_cols=None,
                params = [None,None,None],
                out_path=os.getcwd()):
    '''
    Parameters
    ----------
    X: dataset without Target_class
    y: Target_class
    smote: use SMOTE technique to OverSample Unbalnced DataFrame (if have low number data)
    model_name: name for Save model in -out_path-
    cat_cols: category columns that WOE and Scaler Needs them
    out_path: directory that model be saved at
    params: determinate HyperParameter. this parameter take a list with 3 member. each \
                member must be a dict with fullified with hyperParams
                example:
                    Params =[lg_params, rf_params, xgb_params]
                    rf_params = {
                        "max_depth": None,         # Maximum depth of trees (None for unlimited depth)
                        "n_estimators": 250,       # Number of decision trees in the forest
                        "max_features": 11,        # Maximum number of features for each tree split
                        "min_samples_split": 8,    # Minimum samples required to split an internal node
                        "min_samples_leaf": 2,     # Minimum samples required in a leaf node
                        "bootstrap": True,         # Whether to use bootstrap samples
                        "criterion": 'log_loss'    # Quality measure for split ('log_loss' in this case)
                    }

                    lg_params = {"penalty": "l2",  # You can change this to "l1" if you want L1 regularization
                                "C": 1.0,         # Inverse of regularization strength
                                "solver": "liblinear",  # Algorithm to use in the optimization problem
                                "max_iter": 1000  # Maximum number of iterations for optimization}
                    xgb_params = {
                                "n_estimators": 100,  # Number of boosting rounds (trees)
                                "learning_rate": 0.1,  # Step size shrinkage used in update to prevent overfitting
                                "max_depth": 3,  # Maximum depth of a tree
                                "min_child_weight": 1,  # Minimum sum of instance weight (hessian) needed in a child
                                "subsample": 1.0,  # Fraction of samples used for fitting the trees
                                "colsample_bytree": 1.0,  # Fraction of features used for building each tree
                                "gamma": 0,  # Minimum loss reduction required to make a further partition on a leaf node
                                "objective": "binary:logistic",  # Binary classification objective
                                "scale_pos_weight": 1,  # Controls the balance of positive and negative weights
                                "random_state": 42  # Random seed for reproducibility
                            }
                    You can adjust these parameters as needed for your specific machine learning tasks.

                                    
                    each member can be None:
                        if None setted it used preHyperParam selected for shakhsi custgroup
                    each member can be 'default': this use default parameter for each model 
                    in Params =[lg,rf,xg] order of each model is important and we 
                        cant set lg params in second position
    
    Description
    ----------
    Train and save StackingCVClassifier model (logisticRegression, xgBoost, randomForest)
    Stacking is an ensemble learning technique to combine multiple classification models
    
    '''
    if smote:
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        print('-' * 20, '\n','SMOTE applied trainFunc','\n','-' * 20)
    
    woe_encoder,scaler = fit_woe_encoder(X=X,
                                         y=y,
                                         model_name=model_name,
                                         cat_cols=cat_cols,
                                         out_path=out_path)
    
    if isinstance(params[0],dict):
        print(f'>>>>>LG setted HYPERparams: {params[0]}')
        lgt = make_pipeline(woe_encoder, scaler, LogisticRegression(**params[0]))
    elif params[0] == 'default':
        print('>>>>>lg  default HYPERparameter setted')
        lgt = make_pipeline(woe_encoder, scaler, LogisticRegression(random_state=255))
    else:
        print('>>>>>lg before(shakhsi1401) HYPERparameter setted')
        lgt = make_pipeline(woe_encoder, scaler, LogisticRegression(C=80,
                                                            penalty='l2',
                                                            solver='lbfgs',
                                                            max_iter=100000,
                                                            tol=1e-4,
                                                            multi_class='auto',                                                         
                                                            verbose=0))  
        
        
        
    if isinstance(params[1],dict):
        print(f'>>>>> RF setted HYPERparams: {params[1]}')
        rf = RandomForestClassifier(**params[1])
    elif params[1] == 'default':
        print('>>>>>RF default HYPERparameter setted')
        rf = RandomForestClassifier(random_state=255)
    else:
        print('>>>>>RF before(shakhsi1401) HYPERparameter setted ')
        rf = RandomForestClassifier(n_estimators=50,
                            class_weight='balanced',
                            criterion='gini',
                            max_features='sqrt',
                            max_depth=30,
                            max_leaf_nodes=20,
                            max_samples=None,
                            ccp_alpha=0,
                            min_samples_split=.1,
                            min_samples_leaf=3,
                            min_weight_fraction_leaf=0,
                            min_impurity_decrease=0,
                            bootstrap=True,
                            oob_score=False,
                            warm_start=False,                                
                            verbose=2)
        
    if isinstance(params[2],dict):
        print(f'>>>>>XG make HYPERparams: {params[2]}')
        xgb = XGBClassifier(**params[2])
        
    elif params[2] == 'default':
        print('>>>>>xgb default HYPERparameter setted')
        xgb = XGBClassifier(random_state=255)
    else:
        print('>>>>>xgb before(shakhsi1401) HYPERparameter setted')
        xgb = XGBClassifier(n_estimators=80,
                    objective='binary:logistic',
                    booster='gbtree',
                    tree_method='hist',
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.5,
                    learning_rate=0.3,
                    max_depth=15,
                    min_child_weight=10,
                    reg_alpha=0,
                    reg_lambda=1,
                    enable_categorical=True)
        

    lr = LogisticRegression(random_state=255)
    #svc = sklearn.svm.SVC(random_state=255)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=255)
    sclf = StackingCVClassifier(classifiers=[rf, xgb, lgt], # 
                                meta_classifier=lr,                             
                                use_probas=True,
                                cv=skf,
                                shuffle=True,
                                use_features_in_secondary=False,
                                store_train_meta_features=True,
                                n_jobs=-1,
                                random_state=255,
                                verbose=2,)


    sclf.fit(X, np.ravel(y))
    
    joblib.dump(sclf, f'{out_path}/_{model_name}_trainedModel_.pkl')

    return sclf



#%% 8 capture_diagram
def capture_diagram(x, y): 
    '''
    Parameters
    ----------
    x, y: the inputs of a plot 

    
    Description
    ----------
    returns the minimum point needed for capturing a diagram
    '''   
    i = 0
    new_points = [(0, 0)]
    for i in range(1, len(y) - 1):
        if y[i]-y[i-1] != y[i+1]-y[i]:
            new_points.append((x[i], y[i]))
    new_points.append((x[len(y)-1], y[len(y)-1])) 
    return new_points





#%% 9 cap_curve
def cap_curve(y, y_pred, title = 'capCurve',save_dir=os.getcwd(), plot=True):
    '''
    Parameters
    ----------
    y: orginal y (target) of data
    y_pred: y predicted by the model
    title: title of chart that be plotted 
    save_dir: where to save the Plot
    plot: if we just want aR_aP, capture_diagram(x, y) 
    
    Description
    ----------
    plot CapCurve chart based on y True and y predicted
    retun aR_aP, capture_diagram(x, y) for metrics

    '''
    total = len(y)
    one_count = np.sum(y)
    # zero_count = total - one_count
    
    lm = [y for _, y in sorted(zip(y_pred, y), reverse=True)]
    x = np.arange(0, total+1)
    y = np.append([0], np.cumsum(lm))
    
    plt.figure()
    
    plt.plot([0,total], [0,one_count], c='b', linestyle='--', label='Random Model')
    plt.plot(x, y, c='r', label='model')
    plt.plot([0, one_count, total], 
             [0, one_count, one_count],
             c='grey', 
             linewidth=2,
             label='Perfect Model')
    
    # # Point where vertical line will cut trained model
    # index = int((50*total / 100))
    
    # ## 50% Vertical line from x-axis
    # plt.plot([index, index], [0, y[index]], c ='g', linestyle = '--')
    
    # ## Horizontal line to y-axis from prediction model
    # plt.plot([0, index], [y[index], y[index]], c = 'g', linestyle = '--')
    
    # class_1_observed = y[index] * 100 / max(y)
    plt.xlabel('total observation')
    plt.ylabel('total positive outcomes')
    
    plt.legend() 
    
    a = auc([0, total], [0, one_count])
    aP = auc([0, one_count, total], [0, one_count, one_count]) - a
    aR = auc(x, y) - a
    aR_aP = aR / aP    
    plt.title(f'{title} Cap Curve ' + 'score:%0.4f' % aR_aP)    
    if plot:
        plt.savefig(f'{save_dir}\{title}_Cap_Curve' + '-score%0.3f' % aR_aP + '.png')
    # plt.show()  
    return aR_aP, capture_diagram(x, y)





#%% 10 get_metrics
def get_metrics(y = [], y_pred = [], save_dir=None, title='_REPORT_', plot=True):     
    '''
    Parameters
    ----------
    y: orginal y of data
    y_pred: y predicted by the model
    save_dir: in the case of None, the metrics result is displyed on the screen
               if it is set The metrics result is printed in path.txt
    title: Title of logs and charts
    plot: if we set it to the False then it not plot.png for capcureve and Roc 
            

    Description
    ----------
    this function makes metrics file from True y and predicted y
    
    '''
    if len(y) == 0:
        if len(y_pred) == 0:
            print('using -evaluate- function to provide y & y_pred from train dataset of config')
    
    
    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label= 1)
    j = tpr - fpr
    best_threshold = thresholds[j.argmax()]
    prec, rec, f1, sup = precision_recall_fscore_support(y, y_pred)
    AUC = auc(fpr, tpr)
    aR_aP, capcurve_points = cap_curve(y, y_pred,title, save_dir, plot)
    log_data = f'''
===========================================
              {title}
===========================================
auc         : {AUC}
-------------------------------------------
classification_report_imbalanced:
{classification_report_imbalanced(y, y_pred)}
-------------------------------------------
confusion_matrix:
{confusion_matrix(y, y_pred)}
-------------------------------------------
precision 0 : {prec[0]}
precision 1 : {prec[1]}
-------------------------------------------
recall    0 : {rec[0]}
recall    1 : {rec[1]}
-------------------------------------------
f1 score  0 : {f1[0]}
f1 score  1 : {f1[1]}
-------------------------------------------
sup       0 : {sup[0]}
sup       1 : {sup[1]}
-------------------------------------------
fpr            : {fpr}
tpr            : {tpr}
thresholds     : {thresholds}
best_threshold : {best_threshold}
-------------------------------------------
accuracy_score : {accuracy_score(y, y_pred)}
-------------------------------------------
balanced_accuracy_score: {balanced_accuracy_score(y, y_pred)}
-------------------------------------------
f1_score       : {f1_score(y, y_pred)}
f1_score_macro : {f1_score(y, y_pred, average='macro')}
f1_score_micro : {f1_score(y, y_pred, average='micro')}
-------------------------------------------
f2_score       : {fbeta_score(y, y_pred, beta=2)}
f2_score_macro : {fbeta_score(y, y_pred, average='macro', beta=2)}
f2_score_micro : {fbeta_score(y, y_pred, average='micro', beta=2)}
-------------------------------------------
f0.5_score       : {fbeta_score(y, y_pred, beta=0.5)}
f0.5_score_macro : {fbeta_score(y, y_pred, average='macro', beta=0.5)}
f0.5_score_micro : {fbeta_score(y, y_pred, average='micro', beta=0.5)}
-------------------------------------------
matthews_corrcoef    : {matthews_corrcoef(y, y_pred)}
-------------------------------------------
geometric_mean_score : {geometric_mean_score(y, y_pred)}
-------------------------------------------
brier_score_loss     : {brier_score_loss(y, y_pred, pos_label= 1)}
-------------------------------------------
classification_report :
{classification_report(y, y_pred)}
-------------------------------------------
roc diagram data:
{list(zip(fpr, tpr))}
-------------------------------------------
cap_curve diagram data:
{aR_aP}
-----------------
{capcurve_points}
===========================================\n\n\n\n'''
    
    
    f = open(save_dir +f'\{title}_MetricsReport.txt', 'a+') if save_dir else None    
    print('='*100, '\n', log_data, '\n', file=f)
    if f:
        f.close()
    if plot == True:  
        roc(y, y_pred,title, save_path=save_dir)
    




    import pandas as pd
    metr_df = pd.DataFrame() 

    metr_df['precision'] = '' ;metr_df['f1']= '';metr_df['recall']=''
    crd = classification_report_imbalanced(y, y_pred, output_dict=True)
    metr_df['precision'] = metr_df['precision'].append(pd.Series([prec[0], prec[1], crd['avg_pre']]))
    
    new_values = [f1[0], f1[1], crd['avg_pre']]
   
    metr_df.at[0, 'f1'] = f1[0];metr_df.at[1, 'f1'] = f1[1];metr_df.at[2, 'f1'] = crd['avg_f1']
    metr_df.at[0, 'recall'] = rec[0];metr_df.at[1, 'recall'] = rec[1];metr_df.at[2, 'recall'] = crd['avg_rec']
    metr_df['AUC'] = AUC
    metr_df['capcurve_points'] = aR_aP
    metr_df['accuracy'] = accuracy_score(y, y_pred)
    metr_df['f2_score'] = fbeta_score(y, y_pred, beta=2)


    return metr_df






#%%% 10-2 ----  get_metrics_all ----

def get_metrics_all(y = [],
                    save_dir=None,
                    plot=True,
                    train_no=1,
                    day_date=14000101,
                    custgroup=1,
                    with_fcl=0,
                    with_mali=0,
                    ):
    '''
    

    Parameters
    ----------
    y: data set of  outys.csv {it returns from evaluate func) -True y and predicte y from test df -
    save_dir: where to save reports 
    plot: if True this function draw ROC and CAPCURVE in the save_dir ,, if false it doesnt draw anything
    train_no: we need this in end file Report for input in table
    day_date: we need this in end file Report for input in table
    custgroup: we need this in end file Report for input in table
    with_fcl: we need this in end file Report for input in table
    with_mali: we need this in end file Report for input in table

    Returns & save
    -------
    metric_df.csv (for set to table) and metrics for each model 

    '''
     
    # -------------------------------------------------------------------------
    stacking = get_metrics(y=y.y_test_org,y_pred=y['y_test_pred'],save_dir=save_dir,
                plot=plot,title='Stacking model')
    # -------------------------------------------------------------------------
    random_forest = get_metrics(y=y.y_test_org,y_pred=y['y_test_pred_RF'],save_dir=save_dir,
                plot=plot,title=' RF ')
    # -------------------------------------------------------------------------
    xg_boost = get_metrics(y=y.y_test_org,y_pred=y['y_test_pred_XGB '],save_dir=save_dir,
                plot=plot,title=' XGB ')
    # -------------------------------------------------------------------------
    logistic = get_metrics(y=y.y_test_org,y_pred=y['y_test_pred_LOGIT '],save_dir=save_dir,
                plot=plot,title=' LOGIT ')
    # -------------------------------------------------------------------------
    
    #——————————————————————————————————————————————————————————————————————————
    # combine them # ——————————————————————————————————————————————————————————
    #——————————————————————————————————————————————————————————————————————————
    
    metric_df = pd.DataFrame(columns=['TRAIN_NO', 'DAYDATE','CUSTGROUP','MODEL_CODE',
                                      'CLASS_CODE','METRIC','METRIC_CODE','METRIC_VALUE',
                                      'WITH_FCL','WITH_MALI'])
    
    #print(xg_boost)
    
    model__code = [10,1,2,3]
    class__code = [1,2,3]
    auc__values = [stacking['AUC'][0]  ,xg_boost['AUC'][0], random_forest['AUC'][0],logistic['AUC'][0]]
    cap__value = [stacking['capcurve_points'][0]  ,xg_boost['capcurve_points'][0], random_forest['capcurve_points'][0],logistic['capcurve_points'][0]]
    accuracy__value = [stacking['accuracy'][0]  ,xg_boost['accuracy'][0], random_forest['accuracy'][0],logistic['accuracy'][0]]
    f2_score_value = [stacking['f2_score'][0]  ,xg_boost['f2_score'][0], random_forest['f2_score'][0],logistic['f2_score'][0]]
    
    # f1 -----------------------------------------------------
    for i in range(12):
        metric_df.at[i, 'METRIC'] = 'f1'
        metric_df.at[i, 'METRIC_CODE'] = 1
        
    for i in range(3):
        metric_df.at[i, 'MODEL_CODE'] = 10
        metric_df.at[i, 'METRIC_VALUE'] = stacking['f1'][i]
        metric_df.at[i, 'CLASS_CODE'] = i
        
        metric_df.at[i+3, 'MODEL_CODE'] = 1
        metric_df.at[i+3, 'METRIC_VALUE'] = xg_boost['f1'][i]            
        metric_df.at[i+3, 'CLASS_CODE'] = i
        
        metric_df.at[i+6, 'MODEL_CODE'] = 2
        metric_df.at[i+6, 'METRIC_VALUE'] = random_forest['f1'][i]            
        metric_df.at[i+6, 'CLASS_CODE'] = i

        metric_df.at[i+9, 'MODEL_CODE'] = 3
        metric_df.at[i+9, 'METRIC_VALUE'] = logistic['f1'][i]            
        metric_df.at[i+9, 'CLASS_CODE'] = i

    # precision -----------------------------------------------------
    for i in range(12,24):
        metric_df.at[i, 'METRIC'] = 'precision'
        metric_df.at[i, 'METRIC_CODE'] = 2
        

    for i in range(4,7):
        metric_df.at[i+8, 'MODEL_CODE'] = 10
        metric_df.at[i+8, 'METRIC_VALUE'] = stacking['precision'][i-4]
        metric_df.at[i+8, 'CLASS_CODE'] = i-4
        
        metric_df.at[i+11, 'MODEL_CODE'] = 1
        metric_df.at[i+11, 'METRIC_VALUE'] = xg_boost['precision'][i-4]           
        metric_df.at[i+11, 'CLASS_CODE'] = i-4
        
        metric_df.at[i+14, 'MODEL_CODE'] = 2
        metric_df.at[i+14, 'METRIC_VALUE'] = random_forest['precision'][i-4]           
        metric_df.at[i+14, 'CLASS_CODE'] = i-4

        metric_df.at[i+17, 'MODEL_CODE'] = 3
        metric_df.at[i+17, 'METRIC_VALUE'] = logistic['precision'][i-4]          
        metric_df.at[i+17, 'CLASS_CODE'] = i-4
        
    # recall -----------------------------------------------------
    for i in range(24,36):
        metric_df.at[i, 'METRIC'] = 'recall'
        metric_df.at[i, 'METRIC_CODE'] = 3


    for i in range(7,10):
        
        metric_df.at[i+17, 'MODEL_CODE'] = 10
        metric_df.at[i+17, 'METRIC_VALUE'] = stacking['recall'][i-7]
        metric_df.at[i+17, 'CLASS_CODE'] = i-7
        
        metric_df.at[i+20, 'MODEL_CODE'] = 1
        metric_df.at[i+20, 'METRIC_VALUE'] = xg_boost['recall'][i-7]           
        metric_df.at[i+20, 'CLASS_CODE'] = i-7
        
        metric_df.at[i+23, 'MODEL_CODE'] = 2
        metric_df.at[i+23, 'METRIC_VALUE'] = random_forest['recall'][i-7]           
        metric_df.at[i+23, 'CLASS_CODE'] = i-7

        metric_df.at[i+26, 'MODEL_CODE'] = 3
        metric_df.at[i+26, 'METRIC_VALUE'] = logistic['recall'][i-7]           
        metric_df.at[i+26, 'CLASS_CODE'] = i-7

    # AUC -----------------------------------------------------
    for i in range(36,40):
        metric_df.at[i, 'METRIC'] = 'AUC'
        metric_df.at[i, 'METRIC_CODE'] = 4    
        
        metric_df.at[i, 'MODEL_CODE'] = model__code[i-36]
        metric_df.at[i, 'METRIC_VALUE'] = auc__values[i-36] 
        
    # CAP -----------------------------------------------------
    for i in range(40,44):
        metric_df.at[i, 'METRIC'] = 'CAP'
        metric_df.at[i, 'METRIC_CODE'] = 5  
        
        metric_df.at[i, 'MODEL_CODE'] = model__code[i-40]
        metric_df.at[i, 'METRIC_VALUE'] = cap__value[i-40]       
        
    # accuracy -----------------------------------------------------
    for i in range(44,48):
        metric_df.at[i, 'METRIC'] = 'accuracy'
        metric_df.at[i, 'METRIC_CODE'] = 6 
        
        metric_df.at[i, 'MODEL_CODE'] = model__code[i-44]
        metric_df.at[i, 'METRIC_VALUE'] = accuracy__value[i-44]       
        
    # f2_score -----------------------------------------------------
    for i in range(48,52):
        metric_df.at[i, 'METRIC'] = 'f2_score'
        metric_df.at[i, 'METRIC_CODE'] = 12
        
        metric_df.at[i, 'MODEL_CODE'] = model__code[i-48]
        metric_df.at[i, 'METRIC_VALUE'] = f2_score_value[i-48]       






    metric_df['TRAIN_NO'] = train_no
    metric_df['DAYDATE'] = day_date
    metric_df['CUSTGROUP'] = custgroup
    metric_df['WITH_FCL'] = with_fcl
    metric_df['WITH_MALI'] = with_mali

    metric_df.to_csv(save_dir + 'metrics_combined.csv', index=True)
    return metric_df
    
    
    
#%% 11 fit_woe_encoder
@_timeit
def fit_woe_encoder(X=None, y=None, model_name='_', cat_cols=None, out_path=os.getcwd()):
    '''
    Parameters
    ----------
    X: data without target Feature 
    y: target feature
    model_name: name for save woe and scaler, if not set, models isnt saved
    cat_cols: category columns that to be considered
    out_path: where to save scaler and woe pickle models 
    
    Description
    ----------
    make model of woe and standard_scaler and dump it (create model file) with JobLib libraries
    replaces categories features by the weight of evidence (WoE) and using standard scaler
    
    '''
    

    woe_encoder = WOEEncoder(cols=cat_cols, return_df=True,
                             drop_invariant=False, verbose=2, handle_unknown='value',
                             handle_missing='value', randomized=True,
                             sigma=.05, regularization=1, random_state=255)
    woe_encoder.fit(X, y)
    
    del y
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    joblib.dump(woe_encoder, f'{out_path}/woe_encoder_{model_name}_.pkl')
    joblib.dump(scaler,f'{out_path}/scaler_{model_name}_.pkl')
    print(f'\n\t> WOEencoder & Scaler saved in {out_path} \n')
    del X
    return woe_encoder, scaler



#%% 12 cross_val_score_
@_timeit
def cross_val_score_(X=None, y=None, model = None, scoring='roc_auc', cv=5, save_dir=None):
    '''
    Parameters
    ----------
    model: using model to evaluate with cross validation
    X: data without target Feature 
    y: target feature
    scoring: which scorig method considered. use sklearn.metrics.get_scorer_names() to see all scoring methods
    cv: How many subsamples have been taken
    save_dir: where to save the result 
    
    Description
    ----------
    Cross-validation is a resampling method that uses different portions of the data to test and train a model on different iterations
    '''
    scores = cross_val_score(estimator=model, X=X, y=y, cv=cv, scoring=scoring, error_score='raise')
    print(f'\n***\nscores: {scores}, mean score:{scores.mean()}\n***\n')
    f = open(save_dir +'_crossVal_REPORT.txt', 'a+') if save_dir else None    
    print('='*100,'\n',
          f'we iterate {cv} times and get result for each of them:\n',
          scores,'\n',
          f'mnean of scorse: {scores.mean()}\n',
          f'standard deviation of scores: {scores.std()}\n',
          '='*100, '\n', file=f)
    if f:
        f.close()
    return scores
    


#%% 13 hyper_params_selection_by_randomizedsearchcv
@_timeit
def hyper_params_selection_by_randomizedsearchcv(model=None,
                                                 param_dists=None,
                                                 X=None, y=None, n=10):
    '''
    Parameters
    ----------
    model: the model that used for hyperParameter selection.The -model- from the config will be used if it is not set
    param_dists: list of parameters that is considered in random search
                example: 
                            params = {'max_depth':[3,5,7],
                                    'min_samples_split':[0.1,0.4,0.8],
                                    'min_samples_leaf':[0.1, 0.3,0.5],
                                    'max_leaf_nodes':[3, 5, 8, 12]}  
                            in randomForest     
    X: data without target Feature 
    y: target feature
    n: how many candidates taken from the given parameteres

    
    Description
    ----------
    fit every hyperParameter in the models and is testing on k Folded data. shows the result for each set of parameter
    '''
    assert model!=None, 'model is not defined'
    assert param_dists!=None, 'param_dists is not defined'

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=255)
    rscv = RandomizedSearchCV(model, 
                              param_distributions=param_dists,
                              n_iter=n, 
                              scoring='roc_auc',
                              cv=skf, 
                              return_train_score=True,
                              verbose=3,
                              n_jobs=-1,
                              random_state=255)
    
    rscv.fit(X, y)
    return rscv   
    


 #%%  14 hyper_params_randomizedsearchcv_for_all_models
@_timeit
def hyper_params_randomizedsearchcv_for_all_models(X=None,
                                                                y=None,
                                                                n=10,
                                                               save_path=os.getcwd()):
    '''
    Parameters
    ----------
    X: data without target Feature 
    y: target feature
    save_path: save the best-selected parameters and their score to the .txt file in this dir
    n: how many candidates taken from the given parameteres
    
    
    Description
    ----------
    do hyperParameter selection for all models (logesticRegression, xgBoost, randomForest) 
    hyperParams are all considered 
    just set x and y
    '''

    lgt_params = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'dual': [True, False],
        'tol': uniform(loc=0, scale=1e-2),
        'C': uniform(loc=0, scale=4),
        'fit_intercept': [True, False],
        'intercept_scaling': uniform(loc=0, scale=10),
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        'max_iter': [2,20,40,100,200],
        'multi_class': ['auto', 'ovr'],
        'warm_start': [True, False],
        'class_weight': [None, 'balanced']}    
    
    

    rf_params = {"max_depth": [3, None],
                 "n_estimators": [10,50,100,250],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy","log_loss"]}
    
    
    xgb_params = {'n_estimators': [10, 20, 80, 120, 200],
    			  'objective': ['binary:logistic'],
    			  'booster' : ['gbtree', 'dart'],
    			  'tree_method': ['gbtree', 'hist'], 
    			  'subsample': [.6, .8, 1],     
    			  'colsample_bytree': [.6, .8, 1],
    			  'gamma': [.01, .25, .3, .5, 1.5, 5],    
    			  'learning_rate':[.01, .1, .5, 1],
    			  'max_depth': [3, 10, 20,40],	
    			  'min_child_weight': [1, 2, 5, 10, 40],
    			  'reg_alpha': [0, 3],
    			  'reg_lambda': [1, 3, 5],
    			  'enable_categorical': [True]}

    
    rf = RandomForestClassifier(random_state=255, verbose=0, n_jobs=-1)
    xgb = XGBClassifier(random_state=255, verbose=0, n_jobs=-1)
    
    rf_rscv = hyper_params_selection_by_randomizedsearchcv(rf, rf_params, X, y, n)
    print('****\n' *15)
    xgb_rscv = hyper_params_selection_by_randomizedsearchcv(xgb, xgb_params, X, y, n)
    print('****\n' *15)
    
    lgt = LogisticRegression(random_state=255, verbose=0, n_jobs=-1)    
    woe_encoder,scaler = fit_woe_encoder(X,y)
    X_woe = woe_encoder.transform(X) 
    X_scaled = scaler.transform(X_woe) 
    lgt_rscv = hyper_params_selection_by_randomizedsearchcv(lgt, lgt_params, X_scaled, y, n)
    
    with open(save_path + '\\_hyperparameter_result.txt', 'a+') as f:
        print('\n*******\n*random Forest*\n:', file = f)
        print(*list(rf_rscv.best_params_.items()), sep='\n', file = f)
        print('score: ', rf_rscv.best_score_, file = f)
        print('\n*******\n*xg boost*\n:', file = f)
        print(*list(xgb_rscv.best_params_.items()), sep='\n', file = f)
        print('score: ', xgb_rscv.best_score_, file = f)

        
        print('\n*******\n*logistic regression*\n:', file = f)
        print(*list(lgt_rscv.best_params_.items()), sep='\n', file = f)
        print('score: ', lgt_rscv.best_score_, file = f)
        
        #a= pd.DataFrame(xgb_rscv.cv_results_)

        
    return [rf_rscv, xgb_rscv, lgt_rscv]




#%% 15 feature_selection_by_rfecv
@_timeit
def feature_selection_by_rfecv(model=None, X=None, y=None): 
    '''
    Parameters
    ----------
    model: the model that is to be considered
    X: data without target Feature 
    y: target feature
    
    Description
    ----------
    Feature selection is the process of reducing the number of input variables when developing a predictive model.
    Recursive Feature Elimination, Cross-Validated (RFECV) feature selection Selects the best subset of features recursive
    '''
    X.fillna(0, inplace=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=255)
    rfecv = RFECV(
        estimator=model,
        min_features_to_select=100, 
        scoring='roc_auc',
        cv=skf,
        step=100,
        n_jobs=-1,
        verbose=2,)
    rfecv.fit(X, np.ravel(y))    
    sorted_idxs = rfecv.ranking_.argsort()
    sorted_feats = X.columns[sorted_idxs]
    sorted_ranks =  rfecv.ranking_[sorted_idxs]   
    return rfecv, zip(sorted_feats, sorted_ranks)


#%% 16 feature_selection_rfecv_all_models
@_timeit
def feature_selection_rfecv_all_models(X=None, y=None, save_path=os.getcwd()):
    '''
    Parameters
    ----------
    X: data without target Feature 
    y: target feature
    save_path: the path to save the Report
    
    Description
    ----------
    do rfecv feature selection for all models (logesticRegression, xgBoost, randomForest) 
    just set x and y
    '''
    lgt = LogisticRegression(C=150,
                             penalty='l2',
                             solver='newton-cg',
                             max_iter=100,
                             tol=1e-4,
                             multi_class='auto',
                             class_weight=None,                                                                
                             verbose=1)    
    
    rf = RandomForestClassifier(n_estimators=50,
                                class_weight='balanced',
                                criterion='gini',
                                max_features='sqrt',
                                max_depth=30,
                                max_leaf_nodes=20,
                                max_samples=None,
                                ccp_alpha=0,
                                min_samples_split=.1,
                                min_samples_leaf=3,
                                min_weight_fraction_leaf=0,
                                min_impurity_decrease=0,
                                bootstrap=True,
                                oob_score=False,
                                warm_start=False,                                
                                verbose=1)
    xgb = XGBClassifier(
        n_estimators=80,
        objective='binary:logistic',
        booster='gbtree',
        tree_method='hist',
        subsample=.8,
        colsample_bytree=.8,
        gamma=.5,
        learning_rate=.3,
        max_depth=15,
        min_child_weight=10,
        reg_alpha=0,
        reg_lambda=1,
        enable_categorical=True)
    
    
    X.fillna(0, inplace=True)
    woe_encoder, scaler =  fit_woe_encoder(X,y)
    X_woe = woe_encoder.transform(X) 
    
    X_scaled = pd.DataFrame(scaler.transform(X_woe))  

    lgt_rfecv, _ = feature_selection_by_rfecv(lgt, X_scaled, y)
    
    
    rf_rfecv, _ = feature_selection_by_rfecv(rf, X,y)
    xgb_rfecv, _ = feature_selection_by_rfecv(xgb, X,y)
    

    
    joblib.dump(rf_rfecv, save_path + 'rf_rfecv.pkl')
    joblib.dump(xgb_rfecv, save_path + 'xgb_rfecv.pkl')
    joblib.dump(lgt_rfecv, save_path + 'lgt_rfecv.pkl')
    
    with open(save_path +'__FeatureSelection_REPORT_selectedColumn.txt', 'a+') as file:
        print('RANDOM FOREST', file=file)
        aaa = list(zip(rf_rfecv.support_, X.columns))
        rf_t = [i for i in aaa if i[0] == True]
        rf_f = [f'*** {i} \tnot selected' for i in aaa if i[0] == False]
        print(*rf_t,sep='\n', end='\n\n',file = file)
        print('\nFALSE COLUMNS:\n',file=file)

        print(*rf_f,sep='\n', end='\n\n*************\n\n',file = file)
        

        del aaa,rf_t,rf_f
        
        print('XG BOOST', file=file)
        aaa = list(zip(xgb_rfecv.support_, X.columns))
        xg_t = [i for i in aaa if i[0] == True]  

        xg_f = [f'*** {i} not selected' for i in aaa if i[0] == False]
        print(*xg_t, sep='\n', end='\n\n',file = file)
        print('\nFALSE COLUMNS:\n',file=file)
        print(*xg_f, sep='\n', end='\n\n*************\n\n',file = file)
        
        
        
        del aaa,xg_t,xg_f
        
        
        print('Logistic_regression', file=file)
        aaa = list(zip(lgt_rfecv.support_, X.columns))
        lgt_t = [i for i in aaa if i[0] == True]        
        lr_f = [f'*** {i} not selected' for i in aaa if i[0] == False]
        print(*lgt_t,sep='\n', end='\n\n',file = file)
        print('\nFALSE COLUMNS:\n',file=file)
        print(*lr_f,sep='\n', end='\n\n*************\n\n',file = file)
        

        del aaa,lgt_t    
    return xgb_rfecv,rf_rfecv,lgt_rfecv



# %% 17 get_feature_importances

def get_feature_importances(imps,cols,save_path,save_name):
    '''
    Parameters
    ----------
    imps: feature importances that the model returns
    cols: which column(s) must be cosidered. we can use it for facility Features
    save_path: save path to save the Report 
    save_name: it is created for parent function. for each model set explict name 
    
    Description
    ----------
    The function returns FeatureImportances, sorts them by their values, then adds the feature names to them
    returns: (feature_name: importance of that)
    '''   
    if os.path.exists(save_path + '\\Feature_importance\\') == False:
        os.mkdir(save_path + '\\Feature_importance\\')
    
    save_path = save_path + '\\Feature_importance\\'
        
    imps /= np.sum(abs(imps))
    sorted_idxs = abs(imps).argsort()
    sorted_imps = imps[sorted_idxs]
    sorted_imps = 100 * sorted_imps
    sorted_cols =  np.array(cols)[sorted_idxs]
    
    DF = pd.DataFrame()
    DF['features'] =  sorted_cols
    DF['importance'] = sorted_imps
    s = ''
    for feat, imp in zip(sorted_cols, sorted_imps):
        s += f'{feat}: {imp} {chr(10)}'

    with open(save_path + f'{save_name}_featureImportance_REPORT.txt' , 'a+') as f:
        print(s, sep='\n', file=f)
        
        
    DF.to_csv(save_path + f'{save_name}_FeatureImportance_.csv')
    
    features_piechart(DF,9,save_path,save_name)

    biz_stat_of_fi(DF,save_path,save_name + 'biz_stats')
    
    
    return DF



#%% 20 get_model_feat_imps
def get_model_feat_imps(sclf=None, save_path=os.getcwd(), cols=None,
                        # next parameter has been used to combine FI in 1 file with proper structure
                        CUSTGROUP_num=1,
                        FEATURE_TYPE_CODE=1,
                        FEATURE_TYPE_DESC='USUAL',
                        from_date = 13970101,
                        to_date = 15000101,
                        with_fcl = 1,
                        with_mali = 1,):
    '''
    Parameters
    ----------
    sclf: StackingCVClassifier model that has estimators = (randomForest, xgBoost, Logistic)
    save_path: save path to save the Report 
    cols: which column(s) must be cosidered. we can use it for facility Features

    Description
    ----------
    Get the log file of feature importance for stacking classifiers
    
    '''


    
    rf = get_feature_importances(imps=sclf.clfs_[0].feature_importances_, save_path=save_path, save_name='_randomForest_',cols=cols)
    

    xg = get_feature_importances(imps=sclf.clfs_[1].feature_importances_, save_path=save_path, save_name='_XGboost_',cols=cols)

    
    
    #  sclf.clfs_[2][2].coef_.ravel(),
    
    #lg = get_feature_importances(imps=sclf.clfs_[2][2].feature_importances_, save_path=save_path, save_name='_Logistic_',cols=cols)


    #except:
        #print('sa')
    lg = get_feature_importances(imps=sclf.clfs_[2][2].coef_.ravel(), save_path=save_path, save_name='_Logistic_',cols=cols)

        
    print('Getting feature importances is done!')
    print('the txt file of all models saved in', save_path)
    
    
    advanced_feature_importance(xgfi=xg,rffi=rf,lgfi=lg,
                                      CUSTGROUP_num=CUSTGROUP_num,
                                      with_fcl=with_fcl,
                                      with_mali=with_mali,
                                      from_date=from_date,to_date=to_date,
                                      FEATURE_TYPE_CODE=FEATURE_TYPE_CODE,
                                      FEATURE_TYPE_DESC=FEATURE_TYPE_DESC,
                                      save_path=save_path  + 'feature_combined')
    

    return [rf,xg,lg]


#%% 21 get_preds_for_evaluation  positive|negative ratio | shakhsi 1401
@_timeit
def get_preds_for_evaluation(y_true=None, y_pred=None, pos_rate=2):
    '''
    Parameters
    ----------
    y_true: y true(target) of data 
    y_pred: y predicted by the model 
    pos_rate: Ratio of 0 class to class 1 
    
    
    Description
    ----------
    Making y true and y pred in an appropriate ratio of classes and return it
    positive|negative ratio
    '''
    if len(y_pred) == 0:
        print('''y_predict is not defined
        you must training a model then apply it on desire dataset to have y_pred
        if you have the trained model, use -evaluate- function and your model to provide y_pred
        
        this function automatically use stacking model and dataset that in config
        to provide y_pred.
        ''')
        qq = train_model(False)
        y_true, y_pred, *_ = evaluate(sclf = qq)       

    if pos_rate == 1:
        print('\n\n*** positive|negative ratio set to default 1. ***\n\n')
    
    
    
    
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    zero_idxs  = np.where(y_true==0)
    one_idxs = np.where(y_true==1)

    sample_zero_idxs = np.random.randint(0, len(zero_idxs[0]), size=(1, len(one_idxs[0])*pos_rate))
    print(sample_zero_idxs[0], len(sample_zero_idxs[0]))
    y_one_true = y_true[one_idxs]
    print(len(y_one_true), '\tone index')
    y_zero_true = y_true[sample_zero_idxs][0]
    print(len(y_zero_true), '\tzero index')
    
    y_one_preds = y_pred[one_idxs]
    
    y_zero_preds = y_pred[sample_zero_idxs][0]
    
    y_total_true = np.concatenate([y_one_true, y_zero_true])
    print('y_total_true', y_total_true.shape)
    
    y_total_pred = np.concatenate([y_one_preds, y_zero_preds])
    print('y_total_pred', y_total_pred.shape)
    
    #print('class 0:',np.sum(y_total_true==0),'\tclass 1: np.sum(y_total_true==1))
    
    return y_total_true, y_total_pred
  
    
    
#%% 25 predict


@_timeit
def predict2(file,
             chunk_size=100_000,
             without_path=None,
             with_path=None,
             save_path=os.getcwd(),
             config=None):
    
    '''
    Parameters
    ----------
    file: which file is considered, directory of file
    chunk_size: number of chunk size. in which number data splitted
    without_path: directory of model for predict without Fcl data 
                    if our data have the facility Features we dont need to set it
    with_path: directory of model for predict with Facility data 
                    if our data doesnt have the facility Features we dont need to set it
    save_path: where to save the result 
    config: config for provide columns(features)
    
    
    
    Description
    ----------
    predict the csv file with considering it is belongs to "_Without_FCL", ...
    ... "_UNDER_STRESS_" and "With_FCL"
    '''

    import os
    
    file_path = save_path + '.csv'  
    
    if os.path.exists(file_path):
        print("\n\n\n\nThe file exists.")
        return 0
    else:
        print(">:>The file does not exist.<:<")
    
        
    print(f'predicting {file} file is started')
    current_index = 0.0
    
    if file.find('_Without_FCL') > -1:
        print('||||||||||||  WithOut FCL  ||||||||||||')
        withFCL = 0
        cols = config['without_facility_stacking_cols']
        sclf = joblib.load(without_path)
        
    else:
        if file.find('_UNDER_STRESS_') > -1:
            withFCL = 2 # for stress test files
            print('||||||||||||  Under Stress  ||||||||||||')
        else:
            withFCL = 1            
        cols = config['with_facility_stacking_cols']
        
        sclf = joblib.load(with_path)
        print('**** ' ,sclf)
        print('||||||||||||  With FCL  ||||||||||||')
    if withFCL == 2:
        data = pd.read_csv(file, usecols=cols+['INTCUSTID', 'ECONOMIC_INDEX_CODE', 'SHOCK_TYPE_CODE'], header=0, chunksize=chunk_size)
    else:
        data = pd.read_csv(file, usecols=cols+['INTCUSTID'], header=0, chunksize=chunk_size)
    #print('----* * * * * -- ', data.columns)
    for itr, df in enumerate(data):        
        current_index += chunk_size
        df.fillna(0, inplace=True)

        if withFCL == 0:
            X = df.loc[:, config['without_facility_stacking_cols']]
            
        else:
            X = df.loc[:, config['with_facility_stacking_cols']]
        
        print(X.shape, f'index {current_index} read from {file}')
        
        if withFCL == 2:
            y_pred_proba_ret = df.loc[:, ['INTCUSTID', 'CUSTGROUP', 'OBSERVATION_DATE', 'ECONOMIC_INDEX_CODE', 'SHOCK_TYPE_CODE']]            
        else:
            y_pred_proba_ret = df.loc[:, ['INTCUSTID', 'CUSTGROUP', 'OBSERVATION_DATE']]
        # y_pred = sclf.predict(X)
        
        if withFCL != 2:
            y_pred_proba_ret['WITHFCL'] = withFCL
        
        
        y_pred_proba_ret['DEFAULT'] = sclf.predict(X)
        print(f'index {current_index} predicted stacking default value from {file}')
       
        y_pred_proba_ret['PD'] = sclf.predict_proba(X)[:, 1]
        print(f'index {current_index} predicted stacking proba value from {file}')
        
        if withFCL != 2:    
            y_pred_proba_ret['PD_RF'] = sclf.clfs_[0].predict_proba(X)[:, 1]
            print(f'index {current_index} predicted RF proba value from {file}')
                          
            y_pred_proba_ret['PD_XGB'] = sclf.clfs_[1].predict_proba(X)[:, 1] 
            print(f'index {current_index} predicted XGB proba value from {file}')
            
            y_pred_proba_ret['PD_LOGIT'] = sclf.clfs_[2].predict_proba(X)[:, 1]
            print(f'index {current_index} predicted LOGIT proba value from {file}')

        del X        
               
        
        y_pred_proba_ret.to_csv(save_path + '.csv', mode='a+', index=False, header=False if itr>0 else True)
        
        

        print(y_pred_proba_ret.shape, f' written to {file}')
        del y_pred_proba_ret
        print(f'file {file} {current_index} is predicted')
    
    print(f'predicting {file} file is finished')


#%% 26 predict_pd_proba
def predict_pd_proba(folder_path=''):
    '''
    Description
    ----------
    this function considering all files in folder_path and append them to
        threads to predict them with "predict" Function
        use threds to work better
    '''

    os.makedirs(config['out_path'] + '\\folder_pred\\', exist_ok=True)
    
    # files4prediction = list(os.walk(config['datasets4prediction_path']))[0][2]
    files4prediction = [i for i in glob.glob(f'{folder_path}/*.csv', recursive = True)]
    pprint(files4prediction)
    threads = []

    for file in files4prediction:
        threads.append(threading.Thread(target=predict, args=(file, )))
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print('prediction for all files is done!')
    
    

#%% 28 capcurve_get_points
def capcurve_get_points(sclf = None, ys = [], config=None):
    '''
    Parameters
    ----------
    sclf: stacking model
    ys: if you predict a model and save the result, load it and set it to the functions,
        this attribute  obtained from evaluate's function
    config: config of RISK_MANGEMENT
        
    Description
    ----------
    printing points that needed to draw CapCurve for models
    please set one of the parameter, Either sclf or ys
    
    '''
    if sclf:
        print('=== evaluate model ===')
        y_test, y_test_pred, y_test_proba, y_test_pred_RF, y_test_pred_XGB, y_test_pred_LOGIT= evaluate_(sclf,stacking_model = True)
    if len(ys) != 0:
        y_test, y_test_pred, y_test_proba, y_test_pred_RF, y_test_pred_XGB, y_test_pred_LOGIT = ys
    
    all_ = cap_curve(y_test, y_test_pred, 'all')
    print(all_)
    
    xg_ = cap_curve(y_test, y_test_pred_XGB, 'xg')
    print(xg_)
        
    rf_ = cap_curve(y_test, y_test_pred_RF, 'rf')
    print(rf_)            
    
    lg_ = cap_curve(y_test, y_test_pred_LOGIT, 'lg')
    print(lg_)

    with open(config['out_path'] + 'capCurve_points.txt', 'w') as f:
            print('points for whole stacking:', all_,
                  'points for xgBoost: ',xg_,
                  'points for random forest:', rf_,
                  'points for logistic regression', lg_,
                  sep = '\n-----------------------------\n', file=f)

        
        
    return [all_, xg_, rf_, lg_]






#%% 30 check_line
def check_line(path1, path2):
    '''
    Parameters
    ----------
    path1 & path2: path for dataset to check line count of them 

    
    Description
    ----------
    check line count of two dataset
    can pass dir with *.csv to check multiple dataFrame
    '''
    lp = []
    for i in glob.glob(path1, recursive = True):
        a = get_file_line_count(i)
        lp.append(a)
        
    lnp = []
    for i in glob.glob(path2, recursive = True):
        a = get_file_line_count(i)
        lnp.append(a)
        
    for i in range(len(lp)):
        print(f'{i}-equality:{lp[i]==lnp[i]}', lp[i], lnp[i])
    
    return list(zip(lp,lnp))



    
    


# %% 32 ♦♦♦ PD statistic: describe   ♦♦♦ 

def pred_stats(dirr, save_name=None):
    '''
    Parameters
    ----------
    dirr: path direction of dataset(s)
    save_name: 

    
    Description
    ----------
    
    '''
    if os.path.exists(dirr + '/describes/') == False:
        os.mkdir(dirr + '/describes/')
        
    for i in glob.glob(dirr + '/*.csv'):
        j = i.split('\\')[-1].replace('__pred','_describe_')
        readed_csv = pd.read_csv(i)
        desc = readed_csv.describe()
        desc.to_csv(dirr + f'/describes/{j}', index = True)
    
    
    
    DF = pd.DataFrame()
    
    date_obs = []
    mean_stk, mean_rf, mean_xg, mean_lg, mean_ = [],[],[],[],[]
    std_stk, std_rf, std_xg, std_lg =[],[],[],[]
    median_stk, median_rf, median_xg, median_lg = [],[],[],[]
    min_stk, min_rf, min_xg, min_lg = [],[],[],[]
    max_stk, max_rf, max_xg, max_lg = [],[],[],[]
    mean_of_mean, count = [], []

    
    for i in glob.glob(dirr + '/describes/*.csv'):
        r_c = pd.read_csv(i)
        if int(r_c.OBSERVATION_DATE[1]) > 10000000:
            mean_stk.append(r_c.PD[1]);mean_rf.append(r_c.PD_RF[1])
            mean_xg.append(r_c.PD_XGB[1]);mean_lg.append(r_c.PD_LOGIT[1])
            mean_.append([r_c.PD[1],r_c.PD_RF[1],r_c.PD_XGB[1],r_c.PD_LOGIT[1]])
            
            date_obs.append(int(r_c.OBSERVATION_DATE[1]))
            
            std_stk.append(r_c.PD[2]);std_rf.append(r_c.PD_RF[2])
            std_xg.append(r_c.PD_XGB[2]);std_lg.append(r_c.PD_LOGIT[2])
            
            median_stk.append(r_c.PD[5]);median_rf.append(r_c.PD_RF[5])
            median_xg.append(r_c.PD_XGB[5]);median_lg.append(r_c.PD_LOGIT[5])
            
            min_stk.append(r_c.PD[3]);min_rf.append(r_c.PD_RF[3])
            min_xg.append(r_c.PD_XGB[3]);min_lg.append(r_c.PD_LOGIT[3])
            
            max_stk.append(r_c.PD[7]);max_rf.append(r_c.PD_RF[7])
            max_xg.append(r_c.PD_XGB[7]);max_lg.append(r_c.PD_LOGIT[7])
            
            count.append(r_c.PD[0])
            
        else:
            print('\nthis file is currepted:\n',i)
            
    mean_of_mean = [sum(i)/4 for i in mean_]
    DF['OBSERVATION_DATE'] = date_obs;DF['count'] = count
    DF['mean_stc'] = mean_stk
    DF['mean_rf'] = mean_rf;DF['mean_xgb'] = mean_xg
    DF['mean_lg'] = mean_lg;DF['mean_allModels'] = mean_of_mean
    
    
    DF['std_stacking'] = std_stk;DF['std_rf'] = std_rf
    DF['std_xg'] = std_xg;DF['std_logit'] = std_lg
    
    
    DF['median_stacking'] = median_stk;DF['median_rf'] = median_rf
    DF['median_xg'] = median_xg;DF['median_logit'] = median_lg

    DF['min_stacking'] = min_stk;DF['min_rf'] = min_rf
    DF['min_xg'] = min_xg;DF['min_logit'] = min_lg
    
    DF['max_stacking'] = max_stk;DF['max_rf'] = max_rf
    DF['max_xg'] = max_xg;DF['max_logit'] = max_lg
    
    
    

    sb.set_theme(style="darkgrid")
    
    ob_date_string = [str(i) for i in list(DF.OBSERVATION_DATE.values)]
    
    figure,ax = plt.subplots(nrows = 2, ncols = 2, sharey=True,figsize =(20,15),)
    # figure.canvas.set_window_title('mean of models')
    
    
    ax[0][0].barh(ob_date_string,list(DF.mean_stc.values *100),edgecolor='m',color = 'tan',height=0.7)
    ax[0][0].set(title = 'stacking_PD_means', xlabel='PD %', ylabel='OBSERVATION_DATE')
    ax[0][0].bar_label(ax[0][0].containers[0],label_type='center')

    
    ax[1][0].barh(ob_date_string,list(DF.mean_rf.values *100),edgecolor='darkseagreen',color = 'tan',height=0.7)
    ax[1][0].set(title = 'Rf_PD_Mean', xlabel='PD %', ylabel='OBSERVATION_DATE')
    ax[1][0].bar_label(ax[1][0].containers[0],label_type='center')
    
    ax[0][1].barh(ob_date_string,list(DF.mean_xgb.values *100),edgecolor='m',color = 'c',height=0.7)
    ax[0][1].set(title = 'Xgb_PD_Mean', xlabel='PD %', ylabel='OBSERVATION_DATE')
    ax[0][1].bar_label(ax[0][1].containers[0],label_type='center')
    
    ax[1][1].barh(ob_date_string,list(DF.mean_lg.values *100),edgecolor='darkseagreen',color = 'c',height=0.7)
    ax[1][1].set(title = 'Logit_PD_Mean', xlabel='PD %', ylabel='OBSERVATION_DATE')
    ax[1][1].bar_label(ax[1][1].containers[0],label_type='center')
    
    figure.suptitle(f'{save_name} means of trained model on predicted data', fontsize=20) 
    
    plt.savefig(dirr + f"/describes/__means_barplot_of_{save_name}.png")

    plt.show()



    DF.to_csv(dirr + f'/describes/__total_stats_of_{save_name}.csv', index=False)

    return DF


# %% 33 compare label and predictedPds

def comprise(path_train, observation_date, path_predicted, path_DF_=None, save_path=os.getcwd(),df_col1='-'):
    
    test_data = pd.read_csv(path_train)
    value_ = test_data.loc[(test_data.OBSERVATION_DATE == observation_date)].LABEL.value_counts()
    predicted_data = pd.read_csv(path_predicted)
    
    print('label values:', value_,
          '**** defaualt to total ratio **** :',list(value_)[1]/sum(value_),
          'stacking mean:', predicted_data.PD.mean(),
          'PD_RF mean:', predicted_data.PD_RF.mean(),
          'PD_XGB mean:', predicted_data.PD_XGB.mean(),
          'PD_LOGIT mean:', predicted_data.PD_LOGIT.mean(),
          sep='\n--\n')
    
    DF = pd.DataFrame()
    DF['TYPE'] = [df_col1]
    DF['observation_date'] = [observation_date]
    DF['defaualt'] = [list(value_)[1]/sum(value_)]
    DF['stacking'] = predicted_data.PD.mean()
    DF['PD_RF'] = [predicted_data.PD_RF.mean()]
    DF['PD_XGB'] = [predicted_data.PD_XGB.mean()]
    DF['PD_LOGIT'] = [predicted_data.PD_LOGIT.mean()]
    
    if path_DF_:
        DF_ = pd.read_csv(path_DF_)
        DF = pd.concat([DF,DF_], ignore_index=False)
    
    DF.to_csv(save_path + 'defualt&meansPD.csv', index=False)
    return DF













# =============================================================================
# 
# 
# # ♦♦♦  PD statistic: mean,std & more   ♦♦♦ 
# 
# 
# def pd_stats(path, out_path=os.getcwd()):
#          '''
#          Parameters
#          ----------
# 
#          model_name: name for Save model in -out_path-
#          cat_cols: category columns that WOE and Scaler Needs them
#          
#          Description
#          ----------
#      
#          
#          '''
#          
#          readed_df = pd.read_csv(path)
#          
#          data_container = pd.DataFrame(columns=readed_df.mean().index)
#          
#          
#          for i,j in enumerate(data_container.columns):    
#              print(data_container[j])
#              data_container[j] = [list(readed_df.mean())[i]]
#              data_container[j] = [list(readed_df.std())[i]]
#              data_container[j] = [list(readed_df.median())[i]]
#              
# # =============================================================================
# #          data_container.index.name = ['mean', 'std', 'median']
# # =============================================================================
#          
#          print(data_container)
#          
#          data_container.to_csv(out_path + 'stats_.csv', index = False)
#          
#          print(data_container)
#          
# # =============================================================================
# #          with open(out_path + 'stats_.txt','a+') as f:
# #              print(f'mean: {meann}',
# #                    f'standard deviation: {std}',
# #                    file=f, sep = '\n')
# # =============================================================================
#              
# 
# =============================================================================

# %% 34 pick top n features From FeatureImportances

def pick_top_n(featureImportance_path,n: int):
    '''
    Parameters
    ----------
    featureImportance_path : string 
        directory of csv of feature importance
    n : int
        how many features has been considered.
    
    Returns
    -------
    returns a list with top n features of them 

    '''
    r_df = pd.read_csv(featureImportance_path)
    
    LC_ = list(r_df[r_df.features.str.contains('^LC_')][-n:].features)
    CNTRCT_ = list(r_df[r_df.features.str.contains('^CNTRCT_')][-n:].features)
    FCL_ = list(r_df[r_df.features.str.contains('^FCL_')][-n:].features)
    TRN_ = list(r_df[r_df.features.str.contains('^TRN_')][-n:].features)
    IND_ = list(r_df[r_df.features.str.contains('^IND_')][-n:].features)
    CBI_ = list(r_df[r_df.features.str.contains('^CBI_')][-n:].features)
    ACC_ = list(r_df[r_df.features.str.contains('^ACC_')][-n:].features)
    CHQ_ = list(r_df[r_df.features.str.contains('^CHQ_')][-n:].features)

    all_feature = []
    all_feature.extend([*LC_]); all_feature.extend([*CNTRCT_])
    all_feature.extend([*FCL_]); all_feature.extend([*TRN_])
    all_feature.extend([*IND_]); all_feature.extend([*CBI_])
    all_feature.extend([*ACC_]); all_feature.extend([*CHQ_])
    
    all_feature = all_feature + ['INTCUSTID', 'CUSTGROUP', 'OBSERVATION_DATE']
    return all_feature


# %% 35 features_piechart

def features_piechart(f_df, n, save_path, save_name):
    '''
    Parameters
    ----------
    f_df : Pandas DataFrame  
        dataFrame of feature importance
    n : int
        how many features has been considered.
    save_path: where to save 
        
    save_name: name of title and file 
        
    saves
    -------
    the pie plot of features 

    '''
    f_df.importance = f_df.importance.abs()
    a = list(f_df.importance[-n:])
    b = sum(f_df.importance[:-n])
    a = a[::-1]
    a.append(b)
    
    c = list(zip(f_df.features[-n:],f_df.importance[-n:]))
    c = c[::-1]
    c.extend([('others',sum(f_df.importance[:-n]))])
    
    plt.figure(figsize=(15,12))
    
    patches, texts,*_ = plt.pie(a, startangle=90,
                                radius=1.2, autopct='%1.2f%%',
                                shadow=True)
    
    plt.legend(patches, c, loc='right',fontsize=12, bbox_to_anchor=(-0.1, 1.),
               title='*list of top important feature*')
    plt.title(f'title: pie chart of {save_name}', loc='center')
    
    plt.savefig(fname=save_path + '_Feature_pieChart_' + save_name, bbox_inches='tight')
    


# %% 36 biz_stat_of_fi


def biz_stat_of_fi(df_of_feature, save_path, save_name):
    '''
    Parameters
    ----------
    df_of_feature:
    save_path : path od save dir 
    save_name : name of the file for save 
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df_of_feature.importance = df_of_feature.importance.abs()
    
# =============================================================================
# find feature of each biziness 
    LC_ = list(df_of_feature[df_of_feature.features.str.contains('^LC_')][::-1].features)
    CNTRCT_ = list(df_of_feature[df_of_feature.features.str.contains('^CNTRCT_')][::-1].features)
    FCL_ = list(df_of_feature[df_of_feature.features.str.contains('^FCL_')][::-1].features)
    TRN_ = list(df_of_feature[df_of_feature.features.str.contains('^TRN_')][::-1].features)
    IND_ = list(df_of_feature[df_of_feature.features.str.contains('^IND_')][::-1].features)
    CBI_ = list(df_of_feature[df_of_feature.features.str.contains('^CBI_')][::-1].features)
    ACC_ = list(df_of_feature[df_of_feature.features.str.contains('^ACC_')][::-1].features)
    CHQ_ = list(df_of_feature[df_of_feature.features.str.contains('^CHQ_')][::-1].features)
# =============================================================================
# get importances of each feature in each biz
    LC_measure = [(float(df_of_feature[df_of_feature.eq(LC_[i]).any(1)].importance)) for i in range(len(LC_))]
    CNTRCT_measure = [(float(df_of_feature[df_of_feature.eq(CNTRCT_[i]).any(1)].importance)) for i in range(len(CNTRCT_))]
    FCL__measure = [(float(df_of_feature[df_of_feature.eq(FCL_[i]).any(1)].importance)) for i in range(len(FCL_))]
    TRN__measure = [(float(df_of_feature[df_of_feature.eq(TRN_[i]).any(1)].importance)) for i in range(len(TRN_))]
    IND__measure = [(float(df_of_feature[df_of_feature.eq(IND_[i]).any(1)].importance)) for i in range(len(IND_))]
    CBI__measure = [(float(df_of_feature[df_of_feature.eq(CBI_[i]).any(1)].importance)) for i in range(len(CBI_))]
    ACC_measure = [(float(df_of_feature[df_of_feature.eq(ACC_[i]).any(1)].importance)) for i in range(len(ACC_))]
    CHQ_measure = [(float(df_of_feature[df_of_feature.eq(CHQ_[i]).any(1)].importance)) for i in range(len(CHQ_))]
# ============================================================================= 
# in this section we find min and max of feature of biz 
# LC CNTRCT_ FCL_ IND_ are biziness features and if modell trained without facility ....
# ... these features does not exist so we make an if statement in the beginnig of them 
# ============================================================================= 
# for fine MAX:
    
    if LC_:
        #if all(v == 0 for v in LC_):
        LC_max = list(zip(list(df_of_feature.loc[df_of_feature.importance == max(LC_measure), :].importance),list(df_of_feature.loc[df_of_feature.importance == max(LC_measure), :].features)))

        if len(LC_max) > 1:
            LC_max = [[0]]
    else:
        LC_max = [[0]]
    
    # >>> next biz <<<
    if CNTRCT_:
        CNTRCT_max = list(zip(list(df_of_feature.loc[df_of_feature.importance == max(CNTRCT_measure), :].importance), list(df_of_feature.loc[df_of_feature.importance == max(CNTRCT_measure), :].features)))
        if len(CNTRCT_max) > 1:
            CNTRCT_max = [[0]]  
    else:
        CNTRCT_max = [[0]]
    
    # >>> next biz <<<
    if FCL_:
        FCL_max = list(zip( list(df_of_feature.loc[df_of_feature.importance == max(FCL__measure), :].importance), list(df_of_feature.loc[df_of_feature.importance == max(FCL__measure), :].features)))
        if len(FCL_max) > 1:
            FCL_max = [[0]]  
    else:
        FCL_max = [[0]]
    
    # >>> next biz <<<
    TRN_max = list(zip(list(df_of_feature.loc[df_of_feature.importance == max(TRN__measure), :].importance), list(df_of_feature.loc[df_of_feature.importance == max(TRN__measure), :].features)))
    if len(TRN_max) > 1:
        TRN_max = [[0]]  
    
    # >>> IND_ biz <<<
    if IND_:
        IND_max = list(zip(list(df_of_feature.loc[df_of_feature.importance == max(IND__measure), :].importance),list(df_of_feature.loc[df_of_feature.importance == max(IND__measure), :].features)))
        if len(IND_max) > 1:
            IND_max = [[0]] 
    else:
        IND_max = [[0]]
    # >>> next biz <<<
    CBI_max = list(zip(list(df_of_feature.loc[df_of_feature.importance == max(CBI__measure), :].importance), list(df_of_feature.loc[df_of_feature.importance == max(CBI__measure), :].features)))
    if len(CBI_max) > 1:
        CBI_max = [[0]] 
    
    # >>> next biz <<<
    ACC_max = list(zip(list(df_of_feature.loc[df_of_feature.importance == max(ACC_measure), :].importance),list(df_of_feature.loc[df_of_feature.importance == max(ACC_measure), :].features)))
    if len(ACC_max) > 1:
        ACC_max = [[0]] 
    
    # >>> next biz <<<
    CHQ_max = list(zip( list(df_of_feature.loc[df_of_feature.importance == max(CHQ_measure), :].importance),list(df_of_feature.loc[df_of_feature.importance == max(CHQ_measure), :].features)))
    if len(CHQ_max) > 1:
        CHQ_max = [[0]] 
# =============================================================================    
# for find MIN 
    if LC_:
        LC_min = list(zip(list(df_of_feature.loc[df_of_feature.importance == min(LC_measure), :].importance),list(df_of_feature.loc[df_of_feature.importance == max(LC_measure), :].features)))
        if len(LC_min) > 1:
            LC_min = [[0]] 
    else:
        LC_min = [[0]]
    
    # >>> next biz <<<
    if CNTRCT_:
        CNTRCT_min = list(zip(list(df_of_feature.loc[df_of_feature.importance == min(CNTRCT_measure), :].importance), list(df_of_feature.loc[df_of_feature.importance == max(CNTRCT_measure), :].features)))
        if len(CNTRCT_min) > 1:
            CNTRCT_min = [[0]] 
    else:
        CNTRCT_min = [[0]]
    
    # >>> next biz <<<  
    if FCL_:
        FCL_min = list(zip( list(df_of_feature.loc[df_of_feature.importance == min(FCL__measure), :].importance), list(df_of_feature.loc[df_of_feature.importance == max(FCL__measure), :].features)))
        if len(FCL_min) > 1:
            FCL_min = [[0]] 
    else:
        FCL_min = [[0]]
   
    # >>> next biz <<<
    TRN_min = list(zip(list(df_of_feature.loc[df_of_feature.importance == min(TRN__measure), :].importance), list(df_of_feature.loc[df_of_feature.importance == max(TRN__measure), :].features)))
    if len(TRN_min) > 1:
        TRN_min = [[0]]
    
    # >>> next biz <<<
    if IND_:
        IND_min = list(zip(list(df_of_feature.loc[df_of_feature.importance == min(IND__measure), :].importance),list(df_of_feature.loc[df_of_feature.importance == max(IND__measure), :].features)))
        if len(IND_min) > 1:
            IND_min = [[0]]
    else:
        IND_max = [[0]]
    
    # >>> next biz <<<            
    CBI_min = list(zip(list(df_of_feature.loc[df_of_feature.importance == min(CBI__measure), :].importance), list(df_of_feature.loc[df_of_feature.importance == max(CBI__measure), :].features)))
    if len(CBI_min) > 1:
        CBI_min = [[0]]
    
    # >>> next biz <<<
    ACC_min = list(zip(list(df_of_feature.loc[df_of_feature.importance == min(ACC_measure), :].importance),list(df_of_feature.loc[df_of_feature.importance == max(ACC_measure), :].features)))
    if len(ACC_min) > 1:
        ACC_min = [[0]]
   
    # >>> next biz <<<
    CHQ_min = list(zip( list(df_of_feature.loc[df_of_feature.importance == min(CHQ_measure), :].importance),list(df_of_feature.loc[df_of_feature.importance == max(CHQ_measure), :].features)))
    if len(CHQ_min) > 1:
        CHQ_min = [[0]]
# =============================================================================  

    new_df = pd.DataFrame(index=['LC','CNTRCT', 'FCL', 'TRN', 'IND', 'CBI', 'ACC', 'CHQ', '_SUM_'])
    new_df['count'] = [len(LC_measure),
                        len(CNTRCT_measure),
                        len(FCL__measure),
                        len(TRN__measure),
                        len(IND__measure),
                        len(CBI__measure),
                        len(ACC_measure),
                        len(CHQ_measure),
                        len(df_of_feature.features)]
    
    new_df['sum_of_importance'] = [sum(LC_measure),
                      sum(CNTRCT_measure),
                      sum(FCL__measure),
                      sum(TRN__measure),
                      sum(IND__measure),
                      sum(CBI__measure),
                      sum(ACC_measure),
                      sum(CHQ_measure),
                      1]
    new_df['means'] = new_df['sum_of_importance']/new_df['count']
    
    #print(IND_max,CBI_max, ACC_max, CHQ_max)
    
    new_df['max_of_biz'] = [*LC_max,
                            *CNTRCT_max,
                            *FCL_max,
                            *TRN_max,
                            *IND_max,
                            *CBI_max,
                            *ACC_max,
                            *CHQ_max,
                            LC_max[0][0]+
                            CNTRCT_max[0][0]+
                            FCL_max[0][0]+
                            TRN_max[0][0]+
                            IND_max[0][0]+
                            CBI_max[0][0]+
                            ACC_max[0][0]+
                            CHQ_max[0][0]]
    
    new_df['min_of_biz'] = [*LC_min,
                            *CNTRCT_min,
                            *FCL_min,
                            *TRN_min,
                            *IND_min,
                            *CBI_min,
                            *ACC_min,
                            *CHQ_min,
                            LC_min[0][0]+
                            CNTRCT_min[0][0]+
                            FCL_min[0][0]+
                            TRN_min[0][0]+
                            IND_min[0][0]+
                            CBI_min[0][0]+
                            ACC_min[0][0]+
                            CHQ_min[0][0]]
    
    new_df.to_csv(save_path + '/' + '_bizStatsOfFeatures_' + save_name + '.csv', index=True)
    return new_df



# %% Mrs.khalife wants feature importance in a proper structure









def advanced_feature_importance(xgfi=None,
                                rffi=None,
                                lgfi=None,
                                CUSTGROUP_num=1,
                                FEATURE_TYPE_CODE=1,
                                FEATURE_TYPE_DESC='USUAL',
                                from_date = 13970101,
                                to_date = 15000101,
                                with_fcl = 1,
                                with_mali = 1,
                                save_path=os.getcwd()
                                ):
    '''
    Parameters
    ----------
    xgfi : csv of feature importance of xgb
    rffi : csv of feature importance of rf
    lgfi : csv of feature importance of logit
    CUSTGROUP_num : need this in end file Report for input in table
    FEATURE_TYPE_CODE : need this in end file Report for input in table
    FEATURE_TYPE_DESC : need this in end file Report for input in table
    from_date : need this in end file Report for input in table
    to_date : need this in end file Report for input in table
    with_fcl : need this in end file Report for input in table
    with_mali : need this in end file Report for input in table
    save_path : save dir 

    Returns
    -------
    save a csv file of proper structure for .

    '''
    xgfi = xgfi.iloc[::-1]
    rffi = rffi.iloc[::-1]
    lgfi = lgfi.iloc[::-1]
    
# ---------------------------------------------------------------------------            
# XGBOOST
    xgfi['MODEL_CODE'] = 1
    xgfi['MODEL_NAME'] = 'XGBOOST'

    
    b = []
    c = []
    rank = 1
    for i in range(xgfi.shape[0]):
        if float(xgfi.iloc[i][1]) in c:
            
            b.append(rank-1)
            mask = xgfi['features'] == xgfi.iloc[i][0]
            xgfi.loc[mask, 'rank'] = rank -1
            continue
        else:
            b.append(rank)
            
            mask = xgfi['features'] == xgfi.iloc[i][0]
            xgfi.loc[mask, 'rank'] = rank
            c.append(float(xgfi.iloc[i][1]))
            rank += 1
            
# ---------------------------------------------------------------------------            
# random forest
    rffi['MODEL_NAME'] = 'RANDOM_FOREST'
    rffi['MODEL_CODE'] = 2



    b = []
    c = []
    rank = 1
    for i in range(rffi.shape[0]):
        if float(rffi.iloc[i][1]) in c:
            
            b.append(rank-1)
            mask = rffi['features'] == rffi.iloc[i][0]
            rffi.loc[mask, 'rank'] = rank -1
            continue
        else:
            b.append(rank)
            
            mask = rffi['features'] == rffi.iloc[i][0]
            rffi.loc[mask, 'rank'] = rank
            c.append(float(rffi.iloc[i][1]))
            rank += 1
            
# ---------------------------------------------------------------------------            
# logistic            
    lgfi['MODEL_NAME'] = 'LOGISTIC_REGRESSION'
    lgfi['MODEL_CODE'] = 3


    
    
    
    b = []
    c = []
    rank = 1
    for i in range(lgfi.shape[0]):
        
        
        if float(lgfi.iloc[i][1]) in c:
            
            b.append(rank-1)
            mask = lgfi['features'] == lgfi.iloc[i][0]
            lgfi.loc[mask, 'rank'] = rank - 1
            continue
        else:
            
            
            b.append(rank)
            
            mask = lgfi['features'] == lgfi.iloc[i][0]
            lgfi.loc[mask, 'rank'] = rank
            c.append(float(lgfi.iloc[i][1]))
            rank += 1



    total_df = pd.concat([xgfi, lgfi, rffi])
    
    total_df = total_df.rename(columns={'features':'FEATURE_NAME'})
    
    total_df['FEATURE_TYPE_CODE'] = FEATURE_TYPE_CODE
    total_df['FEATURE_TYPE_DESC'] = FEATURE_TYPE_DESC
    
    
    
    
    total_df = total_df.rename(columns={'rank':'RANK_WAVG'})
    
    total_df.insert(4, 'RANK_AVG', total_df['RANK_WAVG'])
    
    total_df.insert(0, 'FROM_DATE', from_date)
    total_df.insert(1, 'TO_DATE', to_date)
    total_df.insert(2, 'CUSTGROUP', CUSTGROUP_num)
    total_df = total_df.rename(columns={'importance':'IMP_WAVG'})
    
    total_df.insert(5, 'IMP_AVG', total_df['IMP_WAVG'])

    total_df['IS_EFFECTIVE'] = None
    total_df = total_df.reset_index(drop=True)    
    
    for z,i in enumerate(total_df['IMP_WAVG']):
        
        if i == 0:
            total_df.loc[z, 'IS_EFFECTIVE'] = 0
            
        elif i != 0:
            total_df.loc[z, 'IS_EFFECTIVE'] = 1
            
    total_df['WITH_FCL'] = with_fcl
    total_df['WITH_MALI'] = with_mali
    total_df['ID'] = range(len(total_df))

    
    
    total_df = total_df.reindex(columns=['ID','FROM_DATE', 'TO_DATE',
                                         'CUSTGROUP',
                                         'MODEL_CODE',
                                         'MODEL_NAME',
                                         'FEATURE_NAME',
                                         'FEATURE_TYPE_CODE',
                                         'FEATURE_TYPE_DESC',
                                         'IMP_WAVG',
                                         'IMP_AVG',
                                         'RANK_WAVG',
                                         'RANK_AVG',
                                         'IS_EFFECTIVE',
                                         'WITH_FCL',
                                         'WITH_MALI'
                                         ])






    total_df.to_csv(save_path+'.csv', index=False)
    return total_df




# %% turn metrics to CSV

import re

# =============================================================================
# 
# oo = 'E:\HYPER\sherkati\sherkati__withHyper\XGB _MetricsReport.txt'
# 
# 
# def metric_to_csv(path):
#     pattern = r'precision 0'
#     with open(path) as f:
#         contents = f.read()
#         for i in f:
#             if re.search(pattern, i):
#                 print(line)
#     # return contents
# metric_to_csv(oo)
# 
# =============================================================================


