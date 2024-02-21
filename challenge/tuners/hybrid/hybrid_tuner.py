
import logging
import pickle
import sys
import time
import optuna
from sklearn.model_selection import KFold
from Builder import Builder
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample, split_train_in_two_percentage_global_sample_k_fold
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, SLIMElasticNetRecommender
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender
from Recommenders.XGboostRecommender import XGboostRecommender
from tuners.tuner import BaseTuner, Tuner
import scipy.sparse as sps
import numpy as np


class TunerHybridRecommender(Tuner):
    def __init__(self, study_name='ScoresHybridRecommender', **args):
        super().__init__(study_name, **args)
        
    def compute_k_result(self, URM_k_train, URM_k_val, hs, fold_number, metric, cutoff):
        pass
        
    def k_fold_objective_function(self, optuna_trial, k=5, metric='MAP', cutoff=10):
        hs = self.get_hs(optuna_trial)
        start_time = time.time()
        scores = []
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        URM_split = self.URM_train + self.URM_val
        indices_for_sampling = np.arange(0, URM_split.nnz, dtype=int)
        for fold_number, (train_index, test_index) in enumerate(kf.split(indices_for_sampling), 1):
            print(f'-------------[Starting fold {fold_number}...]-------------')
            train_idx = indices_for_sampling[train_index]
            val_idx = indices_for_sampling[test_index]
            URM_k_train, URM_k_val= split_train_in_two_percentage_global_sample_k_fold(URM_split, train_idx, val_idx)
            scores.append(self.compute_k_result(URM_k_train, URM_k_val, hs, fold_number, metric, cutoff))
        
        print(f'{metric} scores: {scores}, cutoff = {cutoff}, execution time: {(time.time() - start_time)/60} minutes')
        return np.mean(scores)             

class TunerScoresHybridRecommender(TunerHybridRecommender):
    def __init__(self, URM_train, URM_val, load_model_path=None, study_name='ScoresHybridRecommender', base_models=[], **args):
        super().__init__(study_name, **args)
        self.rec = ScoresHybridRecommender
        self.URM_train = URM_train
        self.URM_val = URM_val
                        
        
        # Build rec instances for each fold
        self.rec_instances = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        URM_split = URM_train + URM_val
        indices_for_sampling = np.arange(0, URM_split.nnz, dtype=int)
        for fold_number, (train_idx, val_idx) in enumerate(kf.split(indices_for_sampling), 1):
            print(f'-------------[Buildining instance fold {fold_number}...]-------------')
            train_indices = indices_for_sampling[train_idx]
            val_indices = indices_for_sampling[val_idx]
            URM_k_train, URM_k_val= split_train_in_two_percentage_global_sample_k_fold(URM_split, train_indices, val_indices)
            
            model_1_info = self.best_models[base_models[0]]
            model_1 = model_1_info['instance'](URM_k_train)            
            model_2_info = self.best_models[base_models[1]]
            model_2 = model_2_info['instance'](URM_k_train)
            
            if (load_model_path is None):
                model_1.fit(**model_1_info['hs'])
                model_2.fit(**model_2_info['hs'])
            else:
                fold_path = f'{load_model_path}/{fold_number}'
                model_1.load_model(fold_path, base_models[0])
                model_2.load_model(fold_path, base_models[1])

            self.rec_instances.append(self.rec(URM_k_train, model_1, model_2))
        
        
            
    def compute_k_result(self, URM_k_train, URM_k_val, hs, fold_number, metric, cutoff):
        hybrid = self.rec_instances[fold_number-1]            
        hybrid.fit(**hs)
        k_eval = EvaluatorHoldout(URM_k_val, cutoff_list=[cutoff])    
        result_df, _ = k_eval.evaluateRecommender(hybrid)
        return result_df.loc[cutoff][metric]

    def get_hs(self, optuna_trial):        
        hs = {
            'alpha': optuna_trial.suggest_float('alpha', 0.2, 0.8),
        }
        return hs
    

class TunerGeneralizedLinearHybridRecommender(TunerHybridRecommender):
    def __init__(self, URM_train, URM_val, study_name='GeneralizedLinearHybridRecommender', **args):
        super().__init__(study_name, **args)
        self.rec = GeneralizedLinearHybridRecommender
        self.URM_train = URM_train
        self.URM_val = URM_val
        
        models_to_select = ['SLIMElasticNetRecommender', 'RP3betaRecommenderCrossVal', 'SLIM_BPR']

        self.info = {key: self.best_models[key] for key in models_to_select if key in self.best_models}
            
    def compute_k_result(self, URM_k_train, URM_k_val, hs, fold_number):
        recs = []
        for rec_name, rec_info in self.info.items():
            rec_instance = rec_info['instance'](URM_k_train)
            rec_instance.load_model(f'saved_models/xgboost/cross_val/train_val/{fold_number}', rec_name)
            # rec_instance.fit(**rec_info['hs'])
            recs.append(rec_instance)
            
        hybrid = self.rec(URM_k_train, recs)
        hybrid.fit(**hs)
        k_eval = EvaluatorHoldout(URM_k_val, cutoff_list=[10])    
        result_df, _ = k_eval.evaluateRecommender(hybrid)
        return result_df.loc[10]["MAP"]

    def get_hs(self, optuna_trial):
        alphas = []
        for rec_name in self.info.keys():
            if (rec_name == 'SLIMElasticNetRecommender'):
                alphas.append(optuna_trial.suggest_float(f'alpha_{rec_name}', 0.1, 2.5))
            elif (rec_name == 'RP3betaRecommenderCrossVal'): 
                alphas.append(optuna_trial.suggest_float(f'alpha_{rec_name}', 0.1, 1.5)) 
            else:
                alphas.append(optuna_trial.suggest_float(f'alpha_{rec_name}', 0.1, 1)) 

        hs = {
            'alphas': alphas,
        }
        return hs
    

class TunerItemKNNSimilarityHybridRecommender(TunerHybridRecommender):
    def __init__(self, URM_train, URM_val, study_name='TunerItemKNNSimilarityHybridRecommender', **args):
        super().__init__(study_name, **args)
        self.rec = ItemKNNSimilarityHybridRecommender
        self.URM_train = URM_train
        self.URM_val = URM_val
        self.eval = EvaluatorHoldout(URM_val, cutoff_list=[10])    
        
        models_to_select = ['ItemKNNCFRecommenderCrossVal', 'P3alphaRecommenderCrossVal']
        self.info = {key: self.best_models[key] for key in models_to_select if key in self.best_models}

    def compute_k_result(self, URM_k_train, URM_k_val, hs, fold_number):
        w_sparses = []
        for rec_name, rec_info in self.info.items():
            rec_instance = rec_info['instance'](URM_k_train)
            rec_instance.load_model(f'saved_models/xgboost/cross_val/train_val/{fold_number}', rec_name)
            # rec_instance.fit(**rec_info['hs'])
            w_sparses.append(rec_instance.W_sparse)
            
        hybrid = self.rec(URM_k_train, w_sparses[0], w_sparses[1])
        hybrid.fit(**hs)
        k_eval = EvaluatorHoldout(URM_k_val, cutoff_list=[10])    
        result_df, _ = k_eval.evaluateRecommender(hybrid)
        return result_df.loc[10]["MAP"]
        
    
    def get_hs(self, optuna_trial):        
        hs = {
            'topK': optuna_trial.suggest_int('topK', 20, 1000),
            'alpha': optuna_trial.suggest_float('alpha', 0.2, 0.7),
        }
        return hs
    
        
