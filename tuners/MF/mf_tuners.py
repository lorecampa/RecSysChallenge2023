from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_AsySVD_Cython, MatrixFactorization_BPR_Cython, MatrixFactorization_SVDpp_Cython
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.MatrixFactorization.SVDFeatureRecommender import SVDFeature
from tuners.tuner import BaseTuner, MLTuner, Tuner
import time


class TunerSVDpp(MLTuner):        
    def __init__(self, URM_train, URM_val, study_name='SVDpp', **args):
        super().__init__(study_name, **args)
        self.rec = MatrixFactorization_SVDpp_Cython
        self.URM_train = URM_train
        self.URM_val = URM_val
 
    def get_hs(self, optuna_trial):  
        hs = {
            "num_factors": optuna_trial.suggest_int("num_factors", 1, 200),
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
            "batch_size": optuna_trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "item_reg": optuna_trial.suggest_float("item_reg", 1e-5, 1e-2, log=True),
            "user_reg": optuna_trial.suggest_float("user_reg", 1e-5, 1e-2, log=True),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        }
        return hs

class TunerLightFMCFRecommender(MLTuner):        
    def __init__(self, builder, study_name='LightFMCFRecommender', **args):
        super().__init__(builder, study_name, **args)
        self.rec = LightFMCFRecommender
 
    def get_hs(self, optuna_trial):  
        hs = {
            "n_components": optuna_trial.suggest_int("n_components", 1, 200),
            "loss": optuna_trial.suggest_categorical("loss", ['bpr', 'warp', 'warp-kos']),
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ['adagrad', 'adadelta']),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
            "item_alpha": optuna_trial.suggest_float("item_alpha", 1e-5, 1e-2, log=True),
            "user_alpha": optuna_trial.suggest_float("user_alpha", 1e-5, 1e-2, log=True),
        }
        return hs
    
class TunerAsySVD(MLTuner):
    def __init__(self, URM_train, URM_val, study_name='AsySVD', **args):
        super().__init__(study_name, **args)
        self.rec = MatrixFactorization_AsySVD_Cython
        self.URM_train = URM_train
        self.URM_val = URM_val
        
    def get_hs(self, optuna_trial):
        
        hs = {
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
            "epochs": 500,
            "use_bias": optuna_trial.suggest_categorical("use_bias", [True, False]),
            "batch_size": optuna_trial.suggest_categorical("batch_size", [1]),
            "num_factors": optuna_trial.suggest_int("num_factors", 1, 200),
            "item_reg": optuna_trial.suggest_float("item_reg", 1e-5, 1e-2, log=True),
            "user_reg": optuna_trial.suggest_float("user_reg", 1e-5, 1e-2, log=True),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "negative_interactions_quota": optuna_trial.suggest_float("negative_interactions_quota", 0.0, 0.5),
            "epochs": 500,
        }
        return hs


class TunerIALSRecommender(MLTuner):
    def __init__(self, URM_train, URM_val, study_name='IALSRecommender', **args):
        super().__init__(study_name, **args)
        self.rec = IALSRecommender
        self.URM_train = URM_train
        self.URM_val = URM_val
        
    def get_hs(self, optuna_trial):
        hs = {
            "num_factors": optuna_trial.suggest_int("num_factors", 1, 200),
            "confidence_scaling": optuna_trial.suggest_categorical("confidence_scaling", ["linear", "log"]),
            "alpha": optuna_trial.suggest_float("alpha", 1e-3, 50.0, log=True),
            "epsilon": optuna_trial.suggest_float("epsilon", 1e-3, 10.0, log=True),
            "reg": optuna_trial.suggest_float("reg", 1e-5, 1e-2, log=True),
            "epochs": 300,
        }
        return hs



class TunerNMFRecommender(BaseTuner):
    def __init__(self, URM_train, URM_val, study_name='NMFRecommender', **args):
        super().__init__(study_name, **args)
        self.rec = NMFRecommender
        self.URM_train = URM_train
        self.URM_val = URM_val
        self.rec_instance = self.rec(URM_train)
        
    def get_hs(self, optuna_trial):
        hs = {
            "num_factors": optuna_trial.suggest_int("num_factors", 1, 350),
            "init_type": optuna_trial.suggest_categorical("init_type", ["random", "nndsvda"]),
            "beta_loss": optuna_trial.suggest_categorical("beta_loss", ["frobenius", "kullback-leibler"]),
        }
        
        if (hs["beta_loss"] != "frobenius"):
            hs["solver"] = 'multiplicative_update'
        else:
            hs['solver'] = optuna_trial.suggest_categorical("solver", ["coordinate_descent", "multiplicative_update"])
            
        return hs
    
    

class TunerMatrixFactorization_BPR(MLTuner):
    def __init__(self, URM_train, URM_val, study_name='MatrixFactorization_BPR', **args):
        super().__init__(study_name, **args)
        self.rec = MatrixFactorization_BPR_Cython
        self.URM_train = URM_train
        self.URM_val = URM_val
        self.rec_instance = self.rec(URM_train)

        
    def get_hs(self, optuna_trial):
        hs = {
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
            "num_factors": optuna_trial.suggest_int("num_factors", 1, 200),
            "batch_size": optuna_trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "positive_reg": optuna_trial.suggest_float("positive_reg", 1e-5, 1e-2, log=True),
            "negative_reg": optuna_trial.suggest_float("negative_reg", 1e-5, 1e-2, log=True),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "positive_threshold_BPR": None,
            "epochs": 1000,
            }
        allow_dropout = optuna_trial.suggest_categorical("allow_dropout", [True, False])

        
        if (allow_dropout):
            hs["dropout_quota"] = optuna_trial.suggest_float("dropout_quota", 0.01, 0.7)
            
        return {hs}
    
    
class TunerMatrixFactorization_BPR(MLTuner):
    def __init__(self, URM_train, URM_val, study_name='MatrixFactorization_BPR', **args):
        super().__init__(study_name, **args)
        self.rec = MatrixFactorization_BPR_Cython
        self.URM_train = URM_train
        self.URM_val = URM_val
        self.rec_instance = self.rec(URM_train)

        
    def get_hs(self, optuna_trial):

        hs = {
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
            "num_factors": optuna_trial.suggest_int("num_factors", 1, 200),
            "batch_size": optuna_trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            "positive_reg": optuna_trial.suggest_float("positive_reg", 1e-5, 1e-2, log=True),
            "negative_reg": optuna_trial.suggest_float("negative_reg", 1e-5, 1e-2, log=True),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "positive_threshold_BPR": None,
            "epochs": 1000,
            }
        allow_dropout = optuna_trial.suggest_categorical("allow_dropout", [True, False])

        
        if (allow_dropout):
            hs["dropout_quota"] = optuna_trial.suggest_float("dropout_quota", 0.01, 0.7)
            
        return hs
    
    
class TunerSVDFeature(BaseTuner):
    def __init__(self, URM_train, URM_val, study_name='SVDFeature', **args):
        super().__init__(study_name, **args)
        self.rec = SVDFeature
        self.URM_train = URM_train
        self.URM_val = URM_val
        
    def get_hs(self, optuna_trial):

        hs = {
            "num_factors": optuna_trial.suggest_int("num_factors", 1, 200),
            "item_bias_reg": optuna_trial.suggest_float("item_reg", 1e-5, 1e-2, log=True),
            "user_bias_reg": optuna_trial.suggest_float("user_reg", 1e-5, 1e-2, log=True),
            "item_reg": optuna_trial.suggest_float("item_reg", 1e-5, 1e-2, log=True),
            "user_reg": optuna_trial.suggest_float("user_reg", 1e-5, 1e-2, log=True),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "epochs": 30,
        }
        return hs
