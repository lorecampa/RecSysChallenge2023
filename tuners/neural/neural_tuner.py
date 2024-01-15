from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from Recommenders.FeatureWeighting.Cython.CFW_D_Similarity_Cython import CFW_D_Similarity_Cython
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_SVDpp_Cython
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender
from tuners.tuner import MLTuner, Tuner
import time

class TunerMultVAERecommender(MLTuner):        
    def __init__(self, URM_train, URM_val, study_name='MultVAERecommender', **args):
        super().__init__(study_name, **args)
        self.rec = MultVAERecommender
        _, self.n_items = URM_train.shape
        self.URM_train = URM_train
        self.URM_val = URM_val
 
    def get_hs(self, optuna_trial):  
       hs = {
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
            "l2_reg": optuna_trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True),
            "dropout": optuna_trial.suggest_float("dropout", 0., 0.8),
            "total_anneal_steps": optuna_trial.suggest_int("total_anneal_steps", 100000, 600000),
            "anneal_cap": optuna_trial.suggest_float("anneal_cap", 0., 0.6),
            "batch_size": optuna_trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "encoding_size": optuna_trial.suggest_int("encoding_size", 1, min(512, self.n_items-1)),
            "next_layer_size_multiplier": optuna_trial.suggest_int("next_layer_size_multiplier", 2, 10),
            "max_n_hidden_layers": optuna_trial.suggest_int("max_n_hidden_layers", 1, 4),
            "max_parameters": optuna_trial.suggest_categorical("max_parameters", [7*1e9*8/32]),            
            "epochs": 500
        }
       return hs
   
   
   
   
