import numpy as np
import pandas as pd
import scipy.sparse as sps
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import os

from Recommenders.NonPersonalizedRecommender import TopPop

class Builder():
    def __init__(self):
        self.dataset_folder_path = "dataset/book_dataset"
        
        self.raw_target_user_ids = pd.read_csv(self.dataset_folder_path + '/data_target_users_test.csv',
                                           sep=',',
                                           names=['user_id'],
                                           dtype={0:int})

        self.raw_URM = pd.read_csv(self.dataset_folder_path + '/data_train.csv',
                                    sep=",", 
                                    names=["user_id", "item_id", "ratings"],
                                    dtype={0:int, 1:int, 2:float})
        
    
    def preprocess(self):
        self.preprocess_URM()
        self.preprocess_target()
        self.compute_top_popular()
    
    def preprocess_URM(self):
        mapped_id, original_id = pd.factorize(self.raw_URM['user_id'].unique())
        self.user_original_id_to_index = pd.Series(mapped_id, index=original_id)
        self.user_index_to_original_id = pd.Series(original_id, index=mapped_id)
        mapped_id, original_id = pd.factorize(self.raw_URM['item_id'].unique())
        self.item_original_id_to_index = pd.Series(mapped_id, index=original_id)
        self.item_index_to_original_id = pd.Series(original_id, index=mapped_id)
        
        URM = self.raw_URM.copy()
        URM['user_id'] = URM['user_id'].map(self.user_original_id_to_index)
        URM['item_id'] = URM['item_id'].map(self.item_original_id_to_index)
        
        self.processed_URM = URM
        self.URM_sps = sps.coo_matrix((URM["ratings"].values, 
                          (URM["user_id"].values, URM["item_id"].values)))
    
    def preprocess_target(self):        
        mapped_user_ids = []
        dropped_user_ids = []
        for user_id in self.raw_target_user_ids.copy()['user_id']:
            mapped_id = self.user_original_id_to_index.get(user_id, None)
            if mapped_id is None:
                dropped_user_ids.append(user_id)
            else:
                mapped_user_ids.append(mapped_id)
                
        self.target_user_ids = pd.DataFrame(mapped_user_ids, columns=['user_id']).astype(int).to_numpy().ravel()
        self.dropped_user_ids = pd.DataFrame(dropped_user_ids, columns=['user_id']).astype(int).to_numpy().ravel()
                
    def compute_top_popular(self, n=10):
        top_pop_rec = TopPop(self.URM_sps)
        top_pop_rec.fit()
        self.top_pop = np.array(top_pop_rec.recommend(self.target_user_ids[0], cutoff=n, remove_seen_flag=False))
        
            
    def get_sparse_urm_preprocessed(self, sparsity='coo'):
        URM_sps = self.URM_sps            
        if (sparsity=='csr'):
            return URM_sps.tocsr()
        elif (sparsity=='csc'):
            return URM_sps.tocsc()
        
        return URM_sps                    
        
    def split(self, val_percentage = 0.2, test_percentage=0.2, cutoff_list=[10]):        
        URM_train_val, URM_test = split_train_in_two_percentage_global_sample(self.URM_sps, train_percentage = 1 - test_percentage)
        URM_train, URM_val = split_train_in_two_percentage_global_sample(URM_train_val, train_percentage = 1 - val_percentage)

        eval = EvaluatorHoldout(URM_val, cutoff_list=cutoff_list)
        etest = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
        
        return URM_train, URM_val, eval, URM_test, etest
    
    def prepare_sub(self, recommender, cutoff = 10):        
        recommendations=recommender.recommend(self.target_user_ids, cutoff=cutoff)
        return self.prepare_sub_with_recommendations(self.target_user_ids, recommendations)
    
    def prepare_sub_with_recommendations(self, target_user_ids, recommendations, add_top_pop_users=True):
        pairs = []
        for user_id, items_list in zip(target_user_ids, recommendations):
            rec_items_original = [self.item_index_to_original_id[item_id] for item_id in items_list]
            original_user_id = self.user_index_to_original_id[user_id]
            pairs.append((original_user_id, rec_items_original))
        
        # Adding dropped  users with top_pop recommendations
        if (add_top_pop_users):
            mapped_top_pop = [self.item_index_to_original_id[item_id] for item_id in self.top_pop]
            for user_id in self.dropped_user_ids:
                pairs.append((user_id, mapped_top_pop))
                
        df = pd.DataFrame(pairs, columns=['user_id', 'item_list'])
        return df
        
    
    def create_sub(self, submission: pd.DataFrame, folder_path='submissions', file_name='submission'):
        # If directory does not exist, create
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        with open(f"{folder_path}/{file_name}.csv", "w") as f:
            f.write("user_id,item_list\n")
            for user_id, items in zip(submission['user_id'], submission['item_list']):
                f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")
        
        