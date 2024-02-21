
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from Builder import  Builder
import pandas as pd
import optuna
import logging
import pickle
import sys
import time
import numpy as np
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample_k_fold
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseRecommender import BaseRecommender

######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

# KNN
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

# KNN machine learning
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender

# Matrix Factorization
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython,\
    MatrixFactorization_SVDpp_Cython, MatrixFactorization_AsySVD_Cython

from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
import os
from dotenv import load_dotenv


class Tuner:
    def __init__(self, study_name='example_study_name', load_if_exists=True, local=False, random_sampler=False, delete_if_exists=False, cutoff_list=[10, 35]):
        self.cutoff_list = cutoff_list
        load_dotenv()
                
        if (local):
            self.db_connection = os.getenv('DB_CONNECTION_LOCAL')
        else:
            self.db_connection = os.getenv('DB_CONNECTION')
        
        self.study_name = study_name
        all_study_names = optuna.study.get_all_study_names(storage=self.db_connection)
        if (delete_if_exists and study_name in all_study_names):
            print(f'Study already existed, deleting it since delete_if_exists is {delete_if_exists}')
            optuna.delete_study(study_name=self.study_name, storage=self.db_connection)
        
        if (random_sampler):
            self.sampler = optuna.samplers.RandomSampler()
            self.pruner = None
        else:
            #default sampler
            constant_liar = not local
            if (constant_liar):
                print('Constant liar activated')
            self.sampler = optuna.samplers.TPESampler(constant_liar=constant_liar)
            # self.pruner = optuna.pruners.HyperbandPruner()
            self.pruner = None
            
        self.optuna_study = optuna.create_study(
            study_name=self.study_name, 
            storage=self.db_connection,
            direction="maximize",
            load_if_exists=load_if_exists,
            sampler=self.sampler,
            pruner=self.pruner,
        )
        print(f"Sampler is {self.optuna_study.sampler.__class__.__name__}")
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        self.results_df = self.optuna_study.trials_dataframe(attrs=("number", "value", "params", "state"))
        
        # Load dictionary from file
        with open('best_models_info.pickle', 'rb') as f:
            self.best_models = pickle.load(f)
            
    def delete_study(self):
        optuna.delete_study(study_name=self.study_name, storage=self.db_connection)
              
    def optimize(self, n_trials=200, cross_validate=False, n_jobs=1, metric='MAP', cutoff=10): 
        objective_function = self.k_fold_objective_function if cross_validate else self.objective_function              
        self.optuna_study.optimize(
                lambda trial: objective_function(trial, metric=metric, cutoff=cutoff),
                callbacks=[],
                n_trials=n_trials,
                n_jobs=n_jobs
            )
            
    
    def show_statistics(self):
        pruned_trials = [t for t in self.optuna_study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in self.optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.optuna_study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        print("  Value Validation: ", self.optuna_study.best_trial.value)
        print("  Params: ", self.optuna_study.best_trial.params)
    
    

class MLTuner(Tuner):
    def __init__(self, study_name, **args):
        super().__init__(study_name, **args)
        
        self.earlystopping_keywargs = {
            "validation_every_n": 5,
            "stop_on_validation": True,
            "lower_validations_allowed": 5,
            "validation_metric": 'MAP',
        }
        
    def objective_function(self, optuna_trial, metric='MAP', cutoff=10):
        start_time = time.time()
        hs = self.get_hs(optuna_trial)
        eval = EvaluatorHoldout(self.URM_val, cutoff_list=[cutoff])  
        
        rec_instance = self.rec(self.URM_train)
        rec_instance.fit(**hs, **self.earlystopping_keywargs, evaluator_object=eval)
        epochs = rec_instance.get_early_stopping_final_epochs_dict()["epochs"]
        
        exec_time = time.time() - start_time
        optuna_trial.set_user_attr("epochs", epochs)
        optuna_trial.set_user_attr("train_time (min)", f'{(exec_time/60):.1f}')
        
        result_df, _ = eval.evaluateRecommender(rec_instance)
        print(f'Execution time: {(exec_time/60):.1f}')
        return result_df.loc[cutoff][metric]
    
    
    
    def k_fold_objective_function(self, optuna_trial, k=5,  metric='MAP', cutoff=10):
        hs = self.get_hs(optuna_trial)
        
        scores = []
        epochs = []
        train_times = []
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        URM_split = self.URM_train + self.URM_val
        indices_for_sampling = np.arange(0, URM_split.nnz, dtype=int)
        for fold_number, (train_index, test_index) in enumerate(kf.split(indices_for_sampling), 1):
            print(f'-------------[Starting fold {fold_number}...]-------------')
            start_time = time.time()
            train_idx = indices_for_sampling[train_index]
            val_idx = indices_for_sampling[test_index]
            URM_k_train, URM_k_val= split_train_in_two_percentage_global_sample_k_fold(URM_split, train_idx, val_idx)
            k_eval = EvaluatorHoldout(URM_k_val, cutoff_list=[cutoff])  
              
            rec_instance = self.rec(URM_k_train)
            rec_instance.fit(**hs, **self.earlystopping_keywargs, evaluator_object=k_eval)
            result_df, _ = k_eval.evaluateRecommender(rec_instance)
            k_epochs = rec_instance.get_early_stopping_final_epochs_dict()["epochs"]
            train_times.append((time.time() - start_time)/60)
            epochs.append(k_epochs)
            scores.append(result_df.loc[cutoff][metric])
        
        optuna_trial.set_user_attr("epochs", epochs)
        optuna_trial.set_user_attr("best_epoch", int(np.mean(epochs)))
        optuna_trial.set_user_attr("train_time (min)", np.mean(train_times))
        
        print(f'{metric} scores: {scores}, Execution time: {np.sum(train_times):.1f}m')
        return np.mean(scores)
    

class BaseTuner(Tuner):
    def __init__(self, study_name, **args):
        super().__init__(study_name, **args)
        
    def objective_function(self, optuna_trial,  metric='MAP', cutoff=10):
        start_time = time.time()
        hs = self.get_hs(optuna_trial)        
        rec_instance = self.rec(self.URM_train)
        rec_instance.fit(**hs) 
        eval = EvaluatorHoldout(self.URM_val, cutoff_list=[cutoff])       
        result_df, _ = eval.evaluateRecommender(rec_instance)
        print(f'Execution time: {((time.time() - start_time)/60):.1f}')
        return result_df.loc[cutoff][metric]
    
    def k_fold_objective_function(self, optuna_trial, k=5,  metric='MAP', cutoff=10):
        hs = self.get_hs(optuna_trial)
                
        scores = []
        train_times = []
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        URM_split = self.URM_train + self.URM_val
        indices_for_sampling = np.arange(0, URM_split.nnz, dtype=int)
        for fold_number, (train_index, test_index) in enumerate(kf.split(indices_for_sampling), 1):
            print(f'-------------[Starting fold {fold_number}...]-------------')
            start_time = time.time()
            train_idx = indices_for_sampling[train_index]
            val_idx = indices_for_sampling[test_index]
            URM_k_train, URM_k_val= split_train_in_two_percentage_global_sample_k_fold(URM_split, train_idx, val_idx)
            k_eval = EvaluatorHoldout(URM_k_val, cutoff_list=[cutoff])  
              
            rec_instance = self.rec(URM_k_train)
            rec_instance.fit(**hs)
            train_times.append((time.time() - start_time)/60)
            result_df, _ = k_eval.evaluateRecommender(rec_instance)
            scores.append(result_df.loc[cutoff][metric])
        
        print(f'{metric} scores: {scores}, Execution time: {np.sum(train_times):.1f}m')
        return np.mean(scores)
    
    
    
        
    
    
    
    






    
    
        
        
        