import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform

from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, mutual_info_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, classification_report, make_scorer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import MinMaxScaler

import joblib

# reading train data
data_descriptions = pd.read_csv("data/data_descriptions.csv")
train = pd.read_csv("data/train.csv")

# removing customerID as it is not useful for predicting churn
train.drop(columns=['CustomerID'], inplace=True)

# splitting train data into train, val sets with 80%, 20% data
df_train, df_val = train_test_split(train, test_size=0.2, random_state=1)

# separating target variable
y_train = df_train.Churn.values
y_val = df_val.Churn.values

# deleting target variable from feature set
del df_train['Churn']
del df_val['Churn']

# one hot encoding the categorical variables
dv = DictVectorizer(sparse=False)
train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

prop_of_churn_cases = y_train.sum()/len(y_train)
IR = prop_of_churn_cases/(1-prop_of_churn_cases)

def get_model(n_splits=5):

    cv = StratifiedKFold(**{
        'n_splits': n_splits,
        'shuffle': True,
        'random_state': 1
    })

    # recall_scorer = make_scorer(
    #     recall_score,
    #     greater_is_better=True,
    #     needs_proba=True
    #     )

    scoring = 'f1_macro'

    scaler = MinMaxScaler()

    sampler = RandomUnderSampler(**{
        'random_state': 1
    })

    classifier = XGBClassifier(**{
        'random_state': 1,
        'n_jobs': -1,
        'verbosity': 2,
        'validate_parameters': True,
        'disable_default_eval_metric': True,
        'eval_metric' : 'aucpr',
        'scale_pos_weight' : 1/IR # Since scale_pos_weight quantifies the cost of a false negative, we set its value to the inverse of the imbalance ratio.
    })

    pipeline = Pipeline([
        ('scaler', scaler),
        ('sampler', sampler),
        ('classifier', classifier)
    ])

    param_grid = {
      'sampler__sampling_strategy': ['all', 'auto'], #all = resamples both classes, auto = resamples 'not minority' class so no of samples is equalized in both classes.
      'classifier__n_estimators': [25, 50, 100, 125, 150, 200], # no of trees
      'classifier__max_depth': [3, 6, 9, 12, 15, 18, 20, 25, 30, 35],
      'classifier__eta' : [0.001, 0.01, 0.1, 1],
      'classifier__min_child_weight' : [1, 3, 5, 7],
      'classifier__max_delta_step' : [0, 1, 3, 5], #needed when imbalance exists
      'classifier__subsample' : [0.5, 0.75, 1], # 0.75 for preventing overfitting
      'classifier__alpha' : [0, 1, 3, 5, 10, 15], # L1 regularization. Can act as in-built feature selection as we have seen many unimportant features
      'classifier__colsample_bytree' : [0.3, 0.4, 0.5 , 0.7, 0.9],
      'classifier__gamma' : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ]
    }

    model = RandomizedSearchCV(**{
        'estimator': pipeline,
        'cv': cv,
        'param_distributions': param_grid,
        'verbose': 4,
        'scoring': scoring,
        'error_score': np.NaN,
        'n_jobs': -1,
        'random_state' : 2
    })
    return model

model = get_model(n_splits=3)
model.fit(X_train, y_train)

# predictions on validation set
y_pred = model.predict_proba(X_val)[:,1]

precision, recall, thresholds = precision_recall_curve(y_val, y_pred, drop_intermediate=True)

score_df = pd.DataFrame([precision, recall, thresholds]).T
score_df.columns = ['precision','recall','thresholds']
score_df['f1_score'] = (2*score_df['precision']*score_df['recall'])/(score_df['precision']+score_df['recall'])
score_df.sort_values(by='f1_score', ascending=False)

# threshold for which f1_score is highest
decision_threshold = round(score_df.sort_values(by='f1_score', ascending=False)['thresholds'].reset_index(drop=True)[0],2)

# save the churn classification model as a pickle file
model_file_name = "churn_xgb_model.sav"
dv_file_name = "dv.sav"

joblib.dump(model, model_file_name)
joblib.dump(dv, dv_file_name)