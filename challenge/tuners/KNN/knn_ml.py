
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender

from tuners.tuner import BaseTuner, MLTuner



class TunerSLIM_BPR(MLTuner):
    def __init__(self, URM_train, URM_val, study_name='SLIM_BPR', **args):
        super().__init__(study_name, **args)
        self.rec = SLIM_BPR_Cython
        self.rec_instance = self.rec(URM_train)
        self.URM_train = URM_train
        self.URM_val = URM_val

    def get_hs(self, optuna_trial):        
        hs = {
            'positive_threshold_BPR': None,
            'train_with_sparse_weights': None,
            'allow_train_with_sparse_weights': True,
            'topK': optuna_trial.suggest_int('topK', 5, 1000),
            'symmetric': optuna_trial.suggest_categorical('symmetric', [True, False]),
            'sgd_mode': optuna_trial.suggest_categorical('sgd_mode', ["sgd", "adagrad", "adam"]),
            'lambda_i': optuna_trial.suggest_float('lambda_i', 1e-5, 1e-2, log=True),
            'lambda_j': optuna_trial.suggest_float('lambda_j', 1e-5, 1e-2, log=True),
            'learning_rate': optuna_trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            "epochs": 500,
        }
        return hs
        
        
class TunerSLIMElasticNetRecommender(BaseTuner):
    def __init__(self, URM_train, URM_val, multithread=True, study_name='SLIMElasticNetRecommender', **args):
        super().__init__(study_name, **args)
        if (multithread):
            self.rec = MultiThreadSLIM_SLIMElasticNetRecommender
        else:
            self.rec = SLIMElasticNetRecommender

        self.rec_instance = self.rec(URM_train)
        self.URM_train = URM_train
        self.URM_val = URM_val
         
    def get_hs(self, optuna_trial):
        hs = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "l1_ratio": optuna_trial.suggest_float("l1_ratio", 0.001, 0.1, log=True),
            "alpha": optuna_trial.suggest_float("alpha", 0.001, 0.1),
        }
        return hs
         