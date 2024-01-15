import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import scipy.sparse as sps
import numpy as np
from xgboost import XGBRanker, plot_importance
from Builder import Builder
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample, split_train_in_two_percentage_global_sample_k_fold
from Recommenders.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender
from Recommenders.XGboostRecommender import XGboostRecommender
import pickle
from tuners.KNN.graph_tuners import TunerP3alphaRecommender, TunerRP3betaRecommender
from tuners.KNN.knn_ml import TunerSLIM_BPR, TunerSLIMElasticNetRecommender
from tuners.KNN.knn_tuners import TunerItemKNNCFRecommender, TunerUserKNNCFRecommender
from tuners.MF.mf_tuners import TunerAsySVD, TunerIALSRecommender, TunerMatrixFactorization_BPR, TunerNMFRecommender, TunerSVDFeature, TunerSVDpp
from tuners.hybrid.hybrid_tuner import TunerGeneralizedLinearHybridRecommender, TunerItemKNNSimilarityHybridRecommender, TunerScoresHybridRecommender
from tuners.hybrid.xgboost_tuner import TunerXGboostRecommender
from tuners.neural.neural_tuner import TunerMultVAERecommender


load = False

if (load):
    saved_urm_path = 'saved_models'
    URM = sps.load_npz(f'{saved_urm_path}/URM.npz')
    URM_train = sps.load_npz(f'{saved_urm_path}/URM_train.npz')
    URM_val = sps.load_npz(f'{saved_urm_path}/URM_val.npz')
    URM_test = sps.load_npz(f'{saved_urm_path}/URM_test.npz')
else:
    builder = Builder()
    builder.preprocess()
    URM = builder.URM_sps.tocsr()
    URM_train_val, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage = 0.8)
    URM_train, URM_val = split_train_in_two_percentage_global_sample(URM_train_val, train_percentage = 0.8)
   
    
# tuner_instance = TunerRP3betaRecommender
# cutoff=10
# metric='NDCG'
# study_name = f'TunerRP3betaRecommenderCrossVal{metric}{cutoff}'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=False)
# curr_best_trial = {
#             'alpha': 0.37207056615993445,
#             'beta': 0.18674235472914766,
#             'normalize_similarity': True,
#             'topK': 71
#         }
# tuner.optuna_study.enqueue_trial(curr_best_trial, skip_if_exists=True)
# tuner.optimize(n_trials=400, cross_validate=True, n_jobs=1, metric=metric, cutoff=cutoff)


# tuner_instance = TunerP3alphaRecommender
# cutoff=10
# metric='NDCG'
# study_name = f'TunerP3alphaRecommenderCrossVal{metric}{cutoff}Err'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=False, delete_if_exists=False)
# curr_best_trial = {
#             'alpha': 0.3522590971187324,
#             'normalize_similarity': True,
#             'topK': 40
#         },
# tuner.optuna_study.enqueue_trial(curr_best_trial, skip_if_exists=True)
# tuner.optimize(n_trials=500, cross_validate=True, n_jobs=1, metric=metric, cutoff=cutoff)



# tuner_instance = TunerScoresHybridRecommender
# metric='RECALL'
# cutoff=35
# study_name = f'TunerScoresHybridRecommenderCrossVal{metric}{cutoff}'
# base_models = ['SLIMElasticNetRecommenderCrossValRecall35', 'RP3betaRecommenderCrossValRecall35']
# load_model_path=None
# tuner = tuner_instance(URM_train, URM_val, load_model_path=load_model_path, study_name=study_name, base_models=base_models, local=True, delete_if_exists=True)
# tuner.optimize(n_trials=500, cross_validate=True, n_jobs=1, metric=metric, cutoff=cutoff)

# tuner_instance = TunerXGboostRecommender
# study_name = 'TunerXGboostRecommenderCrossValFinalWithRecall'
# tuner = tuner_instance(URM_train + URM_val, URM_test, study_name=study_name, local=False, delete_if_exists=False)
# curr_best_trial = {
#     'n_estimators': 20,
#     'learning_rate': 0.829621061172,
#     'reg_alpha': 34.11908349111817,
#     'reg_lambda': 0.00019598362705460567,
#     'max_depth': 3,
#     'max_leaves': 67,
#     'grow_policy': 'depthwise'
# }
# tuner.optuna_study.enqueue_trial(curr_best_trial, skip_if_exists=True)
# tuner.optimize(n_trials=500, cross_validate=True, n_jobs=1)


# tuner_instance = TunerItemKNNSimilarityHybridRecommender
# metric='NDCG'
# cutoff=10
# study_name = 'TunerItemKNNSimilarityHybridRecommenderAlphaItemCrossVal'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=True, delete_if_exists=True)
# tuner.optimize(n_trials=400, cross_validate=True, n_jobs=1, metric=metric, cutoff=cutoff)


# tuner_instance = TunerSLIM_BPR
# study_name = 'TunerSLIM_BPRCrossVal'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=True)
# tuner.optimize(n_trials=1000, cross_validate=True, n_jobs=1)


# tuner_instance = TunerUserKNNCFRecommender
# metric='NDCG'
# cutoff=10
# study_name = f'TunerUserKNNCFRecommenderCrossVal{metric}{cutoff}'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=False)
# curr_best_trial = {
#             'shrink': 0,
#             'similarity': "asymmetric",
#             'asymmetric_alpha': 0.547922546527745,
#             'topK': 281,
#             'normalize': True,
#             'feature_weighting': 'TF-IDF'
#         }
# tuner.optuna_study.enqueue_trial(curr_best_trial, skip_if_exists=True)
# tuner.optimize(n_trials=400, cross_validate=True, n_jobs=1, metric=metric, cutoff=cutoff)

# tuner_instance = TunerIALSRecommender
# metric='NDCG'
# cutoff=10
# study_name = f'TunerIALSRecommenderCrossVal{metric}{cutoff}'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=False, delete_if_exists=False)
# curr_best_trial = {
#             'num_factors': 80,
#             'confidence_scaling': 'linear',
#             'alpha': 2.1390328625415935,
#             'epsilon': 0.7642171305803429,
#             'reg': 0.0020589743478293086,
#         }
# tuner.optuna_study.enqueue_trial(curr_best_trial, skip_if_exists=False)
# tuner.optimize(n_trials=200, cross_validate=True, n_jobs=1, metric=metric, cutoff=cutoff)


# tuner_instance = TunerGeneralizedLinearHybridRecommender
# study_name = 'TunerGeneralizedLinearHybridRecommenderCrossVal'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=True, delete_if_exists=True)
# tuner.optimize(n_trials=1000, cross_validate=True, n_jobs=1)


# tuner_instance = TunerNMFRecommender
# study_name = 'TunerNMFRecommenderCrossVal'
# tuner = tuner_instance(URM_train + URM_val, URM_test, study_name=study_name, local=False)
# tuner.optimize(n_trials=200, cross_validate=False, n_jobs=1)


# tuner_instance = TunerSVDFeature
# study_name = 'TunerSVDFeature'
# tuner = tuner_instance(URM_train + URM_val, URM_test, study_name=study_name, local=True)
# tuner.optimize(n_trials=200, cross_validate=False, n_jobs=1)

# tuner_instance = TunerItemKNNCFRecommender
# metric='NDCG'
# cutoff=10
# study_name = f'TunerItemKNNCFRecommenderCrossVal{metric}{cutoff}'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=False)
# curr_best_trial = {
#             "topK": 8,
#             "shrink": 18,
#             "similarity": "tversky",
#             "normalize": True,
#             "feature_weighting": "BM25",
#             "tversky_alpha": 0.345724508158624,
#             "tversky_beta": 1.9857406443895325,
#         }
# tuner.optuna_study.enqueue_trial(curr_best_trial, skip_if_exists=True)
# tuner.optimize(n_trials=200, cross_validate=True, n_jobs=1, metric=metric, cutoff=cutoff)


# tuner_instance = TunerMatrixFactorization_BPR
# study_name = 'TunerMatrixFactorization_BPR'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=False)
# tuner.optimize(n_trials=500, cross_validate=False, n_jobs=1)


# tuner_instance = TunerAsySVD
# metric='NDCG'
# cutoff=10
# study_name = f'TunerAsySVD{metric}{cutoff}'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=False)
# tuner.optimize(n_trials=500, cross_validate=False, n_jobs=1, metric=metric, cutoff=cutoff)


# tuner_instance = TunerSVDpp
# study_name = 'TunerSVDpp'
# metric='NDCG'
# cutoff=10
# tuner = tuner_instance(URM_train + URM_val, URM_test, study_name=study_name, local=False)
# tuner.optimize(n_trials=500, cross_validate=False, n_jobs=1, metric=metric, cutoff=cutoff)


# tuner_instance = TunerMultVAERecommender
# metric='NDCG'
# cutoff=10
# study_name = f'TunerMultVAERecommenderCrossVal{metric}{cutoff}Good'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=False, delete_if_exists=False)
# curr_best_trial = {
#     'learning_rate': 6.01205916987149e-05,
#     'l2_reg': 6.432492997938307e-05,
#     'dropout': 0.675210990245954,
#     'total_anneal_steps': 125963,
#     'anneal_cap': 0.5229656445618818,
#     'batch_size': 128,
#     'encoding_size': 500,
#     'next_layer_size_multiplier': 10,
#     'max_n_hidden_layers': 4,
#     'max_parameters': 1750000000.0,
#     'epochs': 470,
# }
# tuner.optuna_study.enqueue_trial(curr_best_trial, skip_if_exists=True)
# tuner.optimize(n_trials=100, cross_validate=True, n_jobs=1, metric=metric, cutoff=cutoff)

# tuner_instance = TunerSLIMElasticNetRecommender
# metric='RECALL'
# cutoff=35
# study_name = f'TunerSLIMElasticNetRecommenderCrossVal{metric}{cutoff}'
# tuner = tuner_instance(URM_train, URM_val, study_name=study_name, local=False, multithread=False, delete_if_exists=False)
# curr_best_trial = {
#             'alpha': 0.011064350293646389,
#             'l1_ratio': 0.0014475981999092396,
#             'positive_only': True,
#             'topK': 294
#         }
# tuner.optuna_study.enqueue_trial(curr_best_trial, skip_if_exists=True)
# curr_best_trial = {
#             'alpha': 0.007975787964890693,
#             'l1_ratio': 0.0010572185474883203,
#             'positive_only': True,
#             'topK': 910
#         }
# tuner.optuna_study.enqueue_trial(curr_best_trial, skip_if_exists=True)
# tuner.optimize(n_trials=200, cross_validate=True, n_jobs=1, metric=metric, cutoff=cutoff)