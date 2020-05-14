# LightFM Dataset Helper
 A class to help preparing the data for LightFM Training 

i see a lot of issues opened in [LightFM](https://github.com/lyst/lightfm/issues) repo , about how to use the dataset
 class or how to add features ... ([1](https://github.com/lyst/lightfm/issues/494#issuecomment-543332968) [2](
 (https://github.com/lyst/lightfm/issues/491)) ...)

i was also struggling to do it an get a functional program , after search and many tries i assume this is a way to do
 it .

## Used Dataset
using books Dataset from [here](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

The Book-Crossing dataset comprises 3 tables.

    BX-Users
    Contains the users. Note that user IDs (`User-ID`) have been anonymized and map to integers. Demographic data is provided (`Location`, `Age`) if available. Otherwise, these fields contain NULL-values.

    BX-Books
    Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (`Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (`Image-URL-S`, `Image-URL-M`, `Image-URL-L`), i.e., small, medium, large. These URLs point to the Amazon web site.

    BX-Book-Ratings
    Contains the book rating information. Ratings (`Book-Rating`) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.

## Example

imports ...
```python
import pandas as pd
from lightfm import LightFM

from DatasetHelper import DatasetHelper
```

loading csv files
```python
def read_csv(filename):
    return pd.read_csv(filename, sep=";", error_bad_lines=False, encoding="latin-1", low_memory=False)

books = read_csv("Data/BX-Books.csv")
users = read_csv("Data/BX-Users.csv")
ratings = read_csv("Data/BX-Book-Ratings.csv")
```
Columns Definitions 
```python
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
```
testing on small amount of data (500)
```python
# just cutting down the amount of data to 500 for less time (making sure no missing data will be passed )
Test_amount = 500
ratings = ratings[:Test_amount]
books = books[books[items_column].isin(ratings[items_column])]
users = users[users[user_column].isin(ratings[user_column])]
```

feeding the dataframes to the helper and running the routine
```python
dataset_helper_instance = dataset_helper(
    users_dataframe=users,
    items_dataframe=books,
    interactions_dataframe=ratings,
    item_id_column=items_column,
    items_feature_columns=items_feature_columns,
    user_id_column=user_column,
    user_features_columns=user_features_columns,
    interaction_column=ratings_column,
    clean_unknown_interactions=True,
    fix_columns_names=False,
)

# run the routine 
dataset_helper_instance.routine()
```

feeding the dataset to the LightFM class
```python 
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
```

###### *hope you did it , and if you see any improvements don't hesistate to create pull request or an issue*
