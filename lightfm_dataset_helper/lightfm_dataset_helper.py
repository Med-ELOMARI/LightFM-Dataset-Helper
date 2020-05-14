"""Main module."""

from Preprocessor import Preprocessor
from lightfm.data import Dataset


class DatasetHelper(Preprocessor):
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
        """
        self.data_ok = True

        super().__init__(
            users_dataframe,
            items_dataframe,
            interactions_dataframe,
            item_id_column,
            items_feature_columns,
            user_id_column,
            user_features_columns,
            interaction_column,
        )
        if False in self.get_data_status().values():
            self.data_ok = False
            raise Exception(
                "[!] Warning ,There is some missing Dataframe {}".format(
                    self.get_data_status()
                )
            )
        else:
            if clean_unknown_interactions:
                self.clean_unknown_interactions_func()

            self.dataset = Dataset()
            self.item_features_list = None
            self.user_features_list = None

            self.done = False

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
