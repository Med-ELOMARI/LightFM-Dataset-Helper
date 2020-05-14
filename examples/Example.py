import pandas as pd
from lightfm import LightFM

from lightfm_dataset_helper.tools import DatasetHelper


def read_csv(filename):
    return pd.read_csv(filename, sep=";", error_bad_lines=False, encoding="latin-1", low_memory=False)


# Loading the data from Data Folder
books = read_csv("Data/BX-Books.csv")
users = read_csv("Data/BX-Users.csv")
ratings = read_csv("Data/BX-Book-Ratings.csv")

# Columns Definitions
items_column = "ISBN"
user_column = "User-ID"
ratings_column = "Book-Rating"

items_feature_columns = [
    "Book-Title",
    "Book-Author",
    "Year-Of-Publication",
    "Publisher",
]

user_features_columns = ["Location", "Age"]

# just cutting down the amount of data to 500 for less time (making sure no missing data will be passed )
Test_amount = 500
ratings = ratings[:Test_amount]
books = books[books[items_column].isin(ratings[items_column])]
users = users[users[user_column].isin(ratings[user_column])]

dataset_helper_instance = DatasetHelper(
    users_dataframe=users,
    items_dataframe=books,
    interactions_dataframe=ratings,
    item_id_column=items_column,
    items_feature_columns=items_feature_columns,
    user_id_column=user_column,
    user_features_columns=user_features_columns,
    interaction_column=ratings_column,
    clean_unknown_interactions=True,
)

# run the routine
dataset_helper_instance.routine()

model = LightFM(no_components=24, loss="warp", k=15)
model.fit(
    interactions=dataset_helper_instance.interactions,
    sample_weight=dataset_helper_instance.weights,
    item_features=dataset_helper_instance.item_features_list,
    user_features=dataset_helper_instance.user_features_list,
    verbose=True,
    epochs=10,
    num_threads=20,
)
