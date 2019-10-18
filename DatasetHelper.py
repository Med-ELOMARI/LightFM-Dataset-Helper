#  Yoruser Recommendation System
#
#  Copyright (c) 2019 Nektiu S.L
#  All Rights Reserved
#
#  Developer  Mohamed EL Omari
#

import itertools
import json

import numpy as np
from lightfm.data import Dataset
from sqlalchemy.util import NoneType


class Preprocessor:
    def __init__(
            self,
            users_dataframe=None,
            items_dataframe=None,
            interactions_dataframe=None,
            item_id_column=None,
            items_feature_columns=None,
            user_id_column=None,
            user_features_columns=None,
            interaction_column=None,
            fix_columns_names=True,
    ):
        """
        this class dedicated to preprocess the Dataframes , ready to be fed into the model

        :param users_dataframe: a dataframe contain  users
        :param items_dataframe: a dataframe contain  items
        :param interactions_dataframe: a dataframe contain ratings of items - users
        :param item_id_column:  name of items column
        :param items_feature_columns: items_feature_columns
        :param user_id_column:  name of users column
        :param user_features_columns: user_features_columns
        """

        self.fix_columns_names = fix_columns_names

        self.items_dataframe = None
        self.users_dataframe = None
        self.interactions_dataframe = None

        if not isinstance(users_dataframe, NoneType):
            self.add_users_dataframe(users_dataframe)
            self.user_id_column = user_id_column
            self.user_features_columns = user_features_columns

        if not isinstance(items_dataframe, NoneType):
            self.add_items_dataframe(items_dataframe)
            self.item_id_column = item_id_column
            self.items_feature_columns = items_feature_columns

        if not isinstance(interactions_dataframe, NoneType):
            self.add_interactions_dataframe(interactions_dataframe)
            self.interaction_column = interaction_column

    def get_data_status(self):
        return {
            "items_dataframe": self.get_dataframe_status(self.items_dataframe),
            "users_dataframe": self.get_dataframe_status(self.users_dataframe),
            "interactions_dataframe": self.get_dataframe_status(
                self.interactions_dataframe
            ),
        }

    @staticmethod
    def get_dataframe_status(data):
        try:
            return not data.empty
        except:
            return False

    def add_items_dataframe(self, items_dataframe):
        self.fix_headers(items_dataframe)
        self.items_dataframe = items_dataframe

    def add_users_dataframe(self, users_dataframe):
        self.fix_headers(users_dataframe)
        self.users_dataframe = users_dataframe

    def add_interactions_dataframe(self, interactions_dataframe):
        self.fix_headers(interactions_dataframe)
        self.interactions_dataframe = interactions_dataframe

    def fix_headers(self, data):
        data.columns = (
            [x.replace("-", "_") for x in data.columns]
            if self.fix_columns_names
            else data.columns
        )
        return data

    @staticmethod
    def lowercase(dataframe):
        return dataframe.apply(lambda x: x.astype(str).str.lower())

    def get_unique_users(self):
        return self.get_uniques_from(self.users_dataframe, self.user_id_column)

    def get_unique_items(self):
        return self.get_uniques_from(self.items_dataframe, self.item_id_column)

    def get_unique_items_from_ratings(self):
        return self.serialize_list(
            self.get_uniques_from(self.interactions_dataframe, self.item_id_column)
        )

    def get_unique_users_from_ratings(self):
        return self.serialize_list(
            self.get_uniques_from(self.interactions_dataframe, self.user_id_column)
        )

    @staticmethod
    def get_uniques_from(dataframe, column):
        return dataframe[column].unique()

    def clean_unknown_interactions_func(self):
        """
        this function to remove all the  existing ratings with unknown items and users
        :return:
        """
        self.interactions_dataframe = self.interactions_dataframe[
            self.interactions_dataframe[self.item_id_column].isin(
                self.items_dataframe[self.item_id_column]
            )
        ]

        self.interactions_dataframe = self.interactions_dataframe[
            self.interactions_dataframe[self.user_id_column].isin(
                self.users_dataframe[self.user_id_column]
            )
        ]

    @staticmethod
    def serialize_list(in_list):
        return list(itertools.chain.from_iterable(in_list))

    def get_unique_items_features(self):
        return self.get_uniques_by_columns(
            self.items_dataframe, self.items_feature_columns
        )

    def get_unique_users_features(self):
        return self.get_uniques_by_columns(
            self.users_dataframe, self.user_features_columns
        )

    def get_uniques_by_columns(self, dataframe, columns):
        uniques = list()
        dataframe = dataframe.applymap(str)
        for col in columns:
            uniques.append(dataframe[col].unique())
        return self.serialize_list(uniques)

    def get_interactions_format(self):
        """
            Todo : it was a generator but light FM need the len (if len(datum) == 3) so i changed it to an array
        :return: iterable of (user_id, item_id, weight)
            An iterable of interactions. The user and item ids will be
            translated to internal model indices using the mappings
            constructed during the fit call
        """
        return [
            (
                row[self.user_id_column],
                row[self.item_id_column],
                np.float(row[self.interaction_column]),
            )
            for idx, row in self.interactions_dataframe.iterrows()
        ]

    @staticmethod
    def prepare_features_format(data, id, feature_columns):
        # for row in itertools.islice(data.iterrows(), 10): # for small dataset_helper_instance 10 .
        for row in data.iterrows():
            yield (row[1][id], [str(row[1][feature]) for feature in feature_columns])


class dataset_helper(Preprocessor):
    def __init__(
            self,
            users_dataframe=None,
            items_dataframe=None,
            interactions_dataframe=None,
            item_id_column="items",
            items_feature_columns: list = None,
            user_id_column="users",
            user_features_columns: list = None,
            interaction_column="interactions",
            clean_unknown_interactions=False,
            fix_columns_names=False,
    ):
        """

        :param users_dataframe:
        :param items_dataframe:
        :param interactions_dataframe:
        :param item_id_column:
        :param items_feature_columns:
        :param user_id_column:
        :param user_features_columns:
        :param interaction_column:
        :param clean_unknown_interactions:  remove unknown data ( what's in the interactions must be also in users
        and items dataframes )
        :param fix_columns_names:
        """
        self.data_ok = True

        super().__init__(
            users_dataframe,
            items_dataframe,
            interactions_dataframe,
            item_id_column,
            items_feature_columns,  # self.fix_headers_names(items_feature_columns)
            user_id_column,
            user_features_columns,  # self.fix_headers_names(user_features_columns),
            interaction_column,
            fix_columns_names,
        )
        if False in self.get_data_status().values():
            print(
                "[!] Warning ,There is some missing Dataframe {}".format(
                    self.get_data_status()
                )
            )
            self.data_ok = False
        else:
            if clean_unknown_interactions:
                self.clean_unknown_interactions_func()

            self.dataset = Dataset()
            self.item_features_list = None
            self.user_features_list = None

            self.done = False

    @staticmethod
    def fix_headers_names(data):
        try:
            return [i.replace("-", "_") for i in data]
        except TypeError:
            pass

    def dataset_fit(self):
        # building the dataset with features
        self.dataset.fit(
            users=self.get_unique_users(),
            items=self.get_unique_items(),
            item_features=self.get_unique_items_features(),
            user_features=self.get_unique_users_features(),
        )

    def build_interactions(self):
        (self.interactions, self.weights) = self.dataset.build_interactions(
            self.get_interactions_format()
        )

    def build_item_features(self):
        self.item_features_list = self.dataset.build_item_features(
            self.prepare_features_format(
                self.items_dataframe, self.item_id_column, self.items_feature_columns
            )
        )

    def build_user_features(self):
        self.user_features_list = self.dataset.build_user_features(
            self.prepare_features_format(
                self.users_dataframe, self.user_id_column, self.user_features_columns
            )
        )

    def get_all_mappings(self):
        return (
            self.dataset._user_id_mapping,
            self.dataset._user_feature_mapping,
            self.dataset._item_id_mapping,
            self.dataset._item_feature_mapping,
        )

    @staticmethod
    def get_metadata(_id, dataframe, desired_column):
        data = dataframe.loc[dataframe[desired_column] == _id]
        if len(data):
            return json.loads(data.to_json(orient="records"))

    def get_user_id_mapping(self):
        return self.dataset._user_id_mapping

    def get_item_id_mapping(self):
        return self.dataset._item_id_mapping

    def get_user_feature_mapping(self):
        return self.dataset._user_feature_mapping

    def get_item_feature_mapping(self):
        return self.dataset._item_feature_mapping

    def routine(self):
        if not self.data_ok:
            raise Exception("Missing Dataframe {}".format(self.get_data_status()))
        self.dataset_fit()
        self.build_interactions()
        self.build_user_features()
        self.build_item_features()
        self.done = True
