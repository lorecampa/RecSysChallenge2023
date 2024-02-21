from sklearn.model_selection import KFold
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample_k_fold
import pickle
import scipy.sparse as sps
import scipy.sparse as sps
import numpy as np
from Recommenders.XGboostRecommender import XGboostRecommender
import pickle
import os
from Builder import Builder
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample, split_train_in_two_percentage_global_sample_k_fold
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.XGboostRecommender import XGboostRecommender
from tuners.hybrid.hybrid_tuner import TunerItemKNNSimilarityHybridRecommender
from tuners.hybrid.xgboost_tuner import TunerXGboostRecommender


load = True
base_urm_path = '/kaggle/input/rec-sys'
if (load):
    URM = sps.load_npz(f'{base_urm_path}/URM.npz')
    URM_train = sps.load_npz(f'{base_urm_path}/URM_train.npz')
    URM_val = sps.load_npz(f'{base_urm_path}/URM_val.npz')
    URM_test = sps.load_npz(f'{base_urm_path}/URM_test.npz')
else:
    builder = Builder()
    builder.preprocess()
    URM = builder.URM_sps.tocsr()
    URM_train_val, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage = 0.8)
    URM_train, URM_val = split_train_in_two_percentage_global_sample(URM_train_val, train_percentage = 0.8)
    sps.save_npz('URM.npz', URM)
    sps.save_npz('URM_train_val.npz', URM_train_val)
    sps.save_npz('URM_test.npz', URM_test)
    sps.save_npz('URM_train.npz', URM_train)
    sps.save_npz('URM_val.npz', URM_val)


with open('best_models_info.pickle', 'rb') as f:
            best_models_info = pickle.load(f)
            
models_to_select = [
    'RP3betaRecommenderCrossValRecall40',
    'SLIMElasticNetRecommenderCrossValRecall40', 
    'TopPop',
    'ItemKNNCFRecommenderCrossValNDCG',
    'UserKNNCFRecommenderCrossValNDCG',
    'P3alphaRecommenderCrossValNDCG',
    'RP3betaRecommenderCrossValNDCG',
    'UserKNNCFRecommenderCrossVal',
    # 'NMFRecommender',
    'IALSRecommender',
    'SLIMElasticNetRecommenderCrossValNDCG',
    # 'RP3betaRecommenderCrossVal', 
    # 'ItemKNNCFRecommenderCrossVal', 
    # 'P3alphaRecommenderCrossVal', 
    # 'UserKNNCFRecommenderCrossVal',
    'MultVAERecommender',
    ]
info = {key: best_models_info[key] for key in models_to_select if key in best_models_info}  


save_model_base_path = 'saved_models'

# train
path = f'{save_model_base_path}/train'
for rec_name, rec_info in info.items():
    rec_instance = rec_info['instance'](URM_train)
    rec_instance.fit(**rec_info['hs'])
    
    if not os.path.exists(path):
        os.makedirs(path)
    rec_instance.save_model(path, rec_name)  

# train_val
path = f'{save_model_base_path}/train_val'
for rec_name, rec_info in info.items():
    rec_instance = rec_info['instance'](URM_train + URM_val)
    rec_instance.fit(**rec_info['hs'])
    
    if not os.path.exists(path):
        os.makedirs(path)
    rec_instance.save_model(path, rec_name)

#train_val_test
path = f'{save_model_base_path}/train_val_test'
for rec_name, rec_info in info.items():
    rec_instance = rec_info['instance'](URM)
    rec_instance.fit(**rec_info['hs'])
    
    if not os.path.exists(path):
        os.makedirs(path)
    rec_instance.save_model(path, rec_name)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
URM_split = URM_train
path = f'{save_model_base_path}/cross_val/train'
indices_for_sampling = np.arange(0, URM_split.nnz, dtype=int)
k_to_skip = []
for fold_number, (train_index, test_index) in enumerate(kf.split(indices_for_sampling), 1):
    if (fold_number in k_to_skip):
        continue
    print(f'-------------[Starting fold {fold_number}...]-------------')
    train_idx = indices_for_sampling[train_index]
    val_idx = indices_for_sampling[test_index]
    URM_k_train, URM_k_val= split_train_in_two_percentage_global_sample_k_fold(URM_split, train_idx, val_idx)
    
    fold_path = f'{path}/{fold_number}'
    for rec_name, rec_info in info.items():
        rec_instance = rec_info['instance'](URM_k_train)
        rec_instance.fit(**rec_info['hs'])
        
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        rec_instance.save_model(fold_path, rec_name)
        
        
        
kf = KFold(n_splits=5, shuffle=True, random_state=42)
URM_split = URM_train + URM_val
path = f'{save_model_base_path}/cross_val/train_val'
indices_for_sampling = np.arange(0, URM_split.nnz, dtype=int)
k_to_skip = []
for fold_number, (train_index, test_index) in enumerate(kf.split(indices_for_sampling), 1):
    if (fold_number in k_to_skip):
        continue
    print(f'-------------[Starting fold {fold_number}...]-------------')
    train_idx = indices_for_sampling[train_index]
    val_idx = indices_for_sampling[test_index]
    URM_k_train, URM_k_val= split_train_in_two_percentage_global_sample_k_fold(URM_split, train_idx, val_idx)
    
    fold_path = f'{path}/{fold_number}'
    for rec_name, rec_info in info.items():
        rec_instance = rec_info['instance'](URM_k_train)
        rec_instance.fit(**rec_info['hs'])
        
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        rec_instance.save_model(fold_path, rec_name)