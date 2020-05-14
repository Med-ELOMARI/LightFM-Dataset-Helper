import numpy as np


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

        self.items_dataframe = None
        self.users_dataframe = None
        self.interactions_dataframe = None

        if users_dataframe is not None:
            self.add_users_dataframe(users_dataframe)
            self.user_id_column = user_id_column
            self.user_features_columns = user_features_columns

        if items_dataframe is not None:
            self.add_items_dataframe(items_dataframe)
            self.item_id_column = item_id_column
            self.items_feature_columns = items_feature_columns

        if interactions_dataframe is not None:
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
        self.items_dataframe = items_dataframe

    def add_users_dataframe(self, users_dataframe):
        self.users_dataframe = users_dataframe

    def add_interactions_dataframe(self, interactions_dataframe):
        self.interactions_dataframe = interactions_dataframe

    def get_unique_users(self):
        return self.get_uniques_from(self.users_dataframe, self.user_id_column)

    def get_unique_items(self):
        return self.get_uniques_from(self.items_dataframe, self.item_id_column)

    def get_unique_items_from_ratings(self):
        return self.get_uniques_from(self.interactions_dataframe, self.item_id_column)

    def get_unique_users_from_ratings(self):
        return self.get_uniques_from(self.interactions_dataframe, self.user_id_column)

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

    def get_unique_items_features(self):
        return self.get_uniques_by_columns(
            self.items_dataframe, self.items_feature_columns
        )

    def get_unique_users_features(self):
        return self.get_uniques_by_columns(
            self.users_dataframe, self.user_features_columns
        )

    def get_uniques_by_columns(self, dataframe, columns):
        dataframe = dataframe.applymap(str)
        uniques = list()
        for col in columns:
            uniques.extend(dataframe[col].unique())
        return uniques

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
        for row in data.iterrows():
            yield (row[1][id], [str(row[1][feature]) for feature in feature_columns])
