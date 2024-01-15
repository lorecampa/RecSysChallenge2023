from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample, split_train_in_two_percentage_global_sample_k_fold
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.XGboostRecommender import XGboostRecommender
import numpy as np
import time

from tuners.tuner import Tuner




class TunerXGboostRecommender(Tuner):
    def __init__(self, URM_train, URM_val, load_model_path = None, study_name='XGboostRecommender', categorical_method=None, **args):
        super().__init__(study_name, **args)
        self.rec = XGboostRecommender
        self.URM_train = URM_train
        self.URM_val = URM_val
        self.cateogrical_method = categorical_method
        
        # Build rec instances for each fold
        self.rec_instances = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        indices_for_sampling = np.arange(0, URM_train.nnz, dtype=int)
        for fold_number, (train_rec_index, train_xgboost_index) in enumerate(kf.split(indices_for_sampling), 1):
            print(f'-------------[Buildining instance fold {fold_number}...]-------------')
            train_rec_idx = indices_for_sampling[train_rec_index]
            xgboost_idx = indices_for_sampling[train_xgboost_index]
            URM_k_train_rec, URM_k_xgboost= split_train_in_two_percentage_global_sample_k_fold(URM_train, train_rec_idx, xgboost_idx)
            fold_path=f'{load_model_path}/{fold_number}' if load_model_path is not None else None
            self.rec_instances.append(self.rec(URM_k_train_rec, URM_k_xgboost, load_model_path=fold_path, categorical_method=categorical_method))

            
    def k_fold_objective_function(self, optuna_trial, metric='MAP', cutoff=10, k=5):
        hs = self.get_hs(optuna_trial)
        eval = EvaluatorHoldout(self.URM_val, cutoff_list=[cutoff])

        scores = []
        exec_times = []
        for fold_number in range(1, k + 1):
            print(f'-------------[Starting fold {fold_number}...]-------------')   
            start_time = time.time()
            rec_instance = self.rec_instances[fold_number - 1]
            rec_instance.fit(**hs)            
            result_df, _ = eval.evaluateRecommender(rec_instance)
            scores.append(result_df.loc[cutoff][metric])
            exec_times.append((time.time() - start_time)/60)
        
        print(f'{metric} scores: {scores}, Execution time: {np.sum(exec_times):.1f}m, mean: {np.mean(exec_times):.1f}m')
        return np.mean(scores)
    
    def get_hs(self, optuna_trial):
        hs = {
            'n_estimators': optuna_trial.suggest_int('n_estimators', 10, 100),
            'learning_rate': optuna_trial.suggest_float('learning_rate', 1e-4, 1e0, log=True),
            'reg_alpha': optuna_trial.suggest_float('reg_alpha', 1e-4, 1e2, log=True),
            'reg_lambda': optuna_trial.suggest_float('reg_lambda', 1e-4, 1e2, log=True),
            'max_depth': optuna_trial.suggest_int('max_depth', 3, 10),
            'max_leaves': optuna_trial.suggest_int('max_leaves', 0, 100),
            'grow_policy': optuna_trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'objective': 'rank:pairwise',
            'booster': 'gbtree',
            'tree_method': 'hist',
            'cutoff': optuna_trial.suggest_categorical('cutoff', [35]),
        }
        return hs