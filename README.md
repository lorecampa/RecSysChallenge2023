<p align="center">
    <img height="200" src="http://www.esplore.polimi.it/wp-content/uploads/2017/05/polimi-logo-1-1.png", alt="logo">
</p>

# RecSys Challenge 2023

I participated in the [Recommender System 2023 Kaggle competition](https://www.kaggle.com/competitions/recommender-system-2023-challenge-polimi) for the Recommender System course as a single-member team among approximately 70 teams.

## Overview

The application domain is **book recommendations**. The datasets provided contain interactions of users with books, specifically, if the user attributed to the book a rating of at least 4 (implicit dataset). The main goal of the competition is to discover books that a user is more likely to interact with.

## Dataset

The dataset includes around 600k interactions, 13k users, and 22k items (books).

The dataset contains:

- `challenge/dataset/book_dataset/data_train.csv`:
  Contains the training set, describing implicit preferences expressed by the users (implicit URM).
- `challenge/dataset/book_dataset/data_target_users_test.csv`:
  Contains the IDs of the users that should have appeared in the subfile. Note that this file also contains IDs of users not present in the training dataset; therefore, recommendations to newly unseen users should be provided, referred to as the "cold start problem."

## Evaluation

The goal is to recommend a list of 10 potentially relevant items for each user. The metric used for evaluation is:
$$\text{MAP@}K = \frac{1}{N} \sum_{u=1}^{N} \frac{1}{\min(K, m)} \sum_{k=1}^{K} P(k) * rel(k)$$

## Solution

The best solution was achieved by using an XGBoost re-ranker model using a hybrid of scores between SLIMElasticNet and RP3Beta as a candidate generator with a cutoff of 35. See the [presentation slides](https://github.com/lorecampa/rec_sys_challenge_2024/blob/main/slides/main.pdf) for a comparison between algorithms used.
