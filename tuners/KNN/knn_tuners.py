
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

from tuners.tuner import BaseTuner, Tuner

class KNNCFTuner(BaseTuner):
    def __init__(self, study_name, **args):
        super().__init__(study_name, **args)
                
    def get_hs(self, optuna_trial):        
        hs = {
            "similarity": optuna_trial.suggest_categorical("similarity", ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean']),
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
            }
        if hs['similarity'] == "asymmetric":
            hs["asymmetric_alpha"] = optuna_trial.suggest_float("asymmetric_alpha", 0, 2, log=False)
            hs["normalize"] = True     

        elif hs['similarity'] == "tversky":
            hs["tversky_alpha"] = optuna_trial.suggest_float("tversky_alpha", 0, 2, log=False)
            hs["tversky_beta"] = optuna_trial.suggest_float("tversky_beta", 0, 2, log=False)
            hs["normalize"] = True 

        elif  hs['similarity'] == "euclidean":
            hs["normalize_avg_row"] = optuna_trial.suggest_categorical("normalize_avg_row", [True, False])
            hs["similarity_from_distance_mode"] = optuna_trial.suggest_categorical("similarity_from_distance_mode", ["lin", "log", "exp"])
            hs["normalize"] = optuna_trial.suggest_categorical("normalize", [True, False])
            
        return hs
    

class TunerItemKNNCFRecommender(KNNCFTuner):
    def __init__(self, URM_train, URM_val, study_name='ItemKNNCFRecommender', **args):
        super().__init__(study_name, **args)
        self.rec = ItemKNNCFRecommender
        self.rec_instance = self.rec(URM_train)
        self.URM_train = URM_train
        self.URM_val = URM_val
        

class TunerUserKNNCFRecommender(KNNCFTuner):
    def __init__(self, URM_train, URM_val, study_name='UserKNNCFRecommender', **args):
        super().__init__(study_name, **args)
        self.rec = UserKNNCFRecommender
        self.rec_instance = self.rec(URM_train)
        self.URM_train = URM_train
        self.URM_val = URM_val
        
    