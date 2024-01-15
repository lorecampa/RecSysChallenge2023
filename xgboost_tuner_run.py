import scipy.sparse as sps
from Builder import Builder
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample, split_train_in_two_percentage_global_sample_k_fold
from Recommenders.XGboostRecommender import XGboostRecommender
import pickle
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tqdm import tqdm
from tuners.hybrid.xgboost_tuner import TunerXGboostRecommender



load = True
base_path = 'saved_models'
if (load):
    URM = sps.load_npz(f'{base_path}/URM.npz')
    URM_train = sps.load_npz(f'{base_path}/URM_train.npz')
    URM_val = sps.load_npz(f'{base_path}/URM_val.npz')
    URM_test = sps.load_npz(f'{base_path}/URM_test.npz')
    URM_train_val = URM_train + URM_val
    print(f'Loading URM at path: {base_path}')
else:
    print("Generating URM...")
    builder = Builder()
    builder.preprocess()
    URM = builder.URM_sps.tocsr()
    URM_train_val, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage = 0.8)
    URM_train, URM_val = split_train_in_two_percentage_global_sample(URM_train_val, train_percentage = 0.8)

tuner_instance = TunerXGboostRecommender
study_name = 'TunerXGboostRecommenderCrossVal'
load_model_path = f'{base_path}/cross_val/train'
tuner = tuner_instance(URM_train + URM_val, URM_test, load_model_path=load_model_path, categorical_method=None, study_name=study_name, local=True, delete_if_exists=False)
tuner.optimize(n_trials=500, cross_validate=True, n_jobs=1)

