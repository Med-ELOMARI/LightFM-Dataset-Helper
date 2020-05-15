=====
Usage
=====

To use LightFM Dataset helper in a project

imports the module

.. code:: python

   from lightfm_dataset_helper.lightfm_dataset_helper import DatasetHelper

loading csv files

.. code:: python

   # using pandas to load csv files
   import pandas as pd

   def read_csv(filename):
       return pd.read_csv(filename, sep=";", error_bad_lines=False, encoding="latin-1", low_memory=False)

   books = read_csv("Data/BX-Books.csv")
   users = read_csv("Data/BX-Users.csv")
   ratings = read_csv("Data/BX-Book-Ratings.csv")

Columns Definitions

.. code:: python

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

-  Optional\* for testing on small amount of data (500)

.. code:: python

   # just cutting down the amount of data to 500 for less time (making sure no missing data will be passed )
   Test_amount = 500
   ratings = ratings[:Test_amount]
   books = books[books[items_column].isin(ratings[items_column])]
   users = users[users[user_column].isin(ratings[user_column])]

feeding the dataframes to the helper and running the routine

.. code:: python

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

feeding the dataset to the LightFM class

.. code:: python

   from lightfm import LightFM

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

Used Dataset
------------

using books Dataset from `here`_

The Book-Crossing dataset comprises 3 tables.
::

   BX-Users
   Contains the users. Note that user IDs (`User-ID`) have been anonymized and map to integers. Demographic data is provided (`Location`, `Age`) if available. Otherwise, these fields contain NULL-values.

   BX-Books
   Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (`Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (`Image-URL-S`, `Image-URL-M`, `Image-URL-L`), i.e., small, medium, large. These URLs point to the Amazon web site.

   BX-Book-Ratings
   Contains the book rating information. Ratings (`Book-Rating`) are either

.. _here: http://www2.informatik.uni-freiburg.de/~cziegler/BX/
