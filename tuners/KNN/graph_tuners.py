
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

from tuners.tuner import BaseTuner        

class TunerP3alphaRecommender(BaseTuner):
    def __init__(self, URM_train, URM_val, study_name='P3alphaRecommender', **args):
        super().__init__(study_name, **args)
        self.rec = P3alphaRecommender
        self.URM_train = URM_train
        self.URM_val = URM_val
    
    def get_hs(self, optuna_trial):        
        hs = {
            "topK": optuna_trial.suggest_int("topK", 20, 100),
            'normalize_similarity': optuna_trial.suggest_categorical("normalize_similarity", [True]),
            'alpha': optuna_trial.suggest_float("alpha", 0.05, 0.5),
            }    
        return hs
        

class TunerRP3betaRecommender(BaseTuner):
    def __init__(self, URM_train, URM_val, study_name='RP3betaRecommender', **args):
        super().__init__(study_name, **args)
        self.rec = RP3betaRecommender
        self.URM_train = URM_train
        self.URM_val = URM_val
        
    def get_hs(self, optuna_trial):        
        hs = {
            "topK": optuna_trial.suggest_int("topK", 20, 100),
            'normalize_similarity': optuna_trial.suggest_categorical("normalize_similarity", [True]),
            'alpha': optuna_trial.suggest_float("alpha", 0.05, 0.5),
            'beta': optuna_trial.suggest_float("beta", 0.05, 0.5),
            }    
        return hs