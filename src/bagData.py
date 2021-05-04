import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from typing import Callable


class BagData:
    """
    """

    def __init__(self, bag_data: dict) -> None:

        self.bag_data = bag_data
        self.num_bags = len(bag_data.keys())
        self.bags = list(bag_data.keys())

        y = np.zeros(self.num_bags)
        for i, bag in enumerate(self.bags):
            y[i] = bag_data[bag]["y"]

        self.y = np.expand_dims(y, 1)

    def _create_subset(
        self, initial_num_items: int, initialisationMethod: Callable = None
    ) -> None:

        bag_data = {}
        for key in self.bags:
            weights = np.zeros((initial_num_items, 1))
            weights[:initial_num_items, 0] = np.array(
                [1 / initial_num_items] * initial_num_items
            )
            # this can be changed to include better initialisation methods
            # such as Uniform selection, quantile selection, Stein SVGP
            if initialisationMethod:
                x = initialisationMethod(self.bag_data[key]["x"], initial_num_items)
            else:
                x = self.bag_data[key]["x"][:initial_num_items]

            bag_data[key] = {
                "N": initial_num_items,
                "weights": weights,
                "x": x,
                "y": self.bag_data[key]["y"],
            }
        self.full_bag_data = self.bag_data.copy()
        self.bag_data = bag_data

    def _update_subset(self) -> None:
        """Updates the dataset using active learning

        """
        raise NotImplementedError()

    def __str__(self):
        msg = "BagData with {} bags".format(self.num_bags)
        return msg
    def __len__(self):
        return self.num_bags
    def __getitem__(self, bag):
        """
            bag: bag index from indexing system
        """
        return tuple(self.bag_data[bag].values())
        
class MultiResolutionBagDataGenerator(BagData):
    def __init__(self, bag_data: dict, num_inducing_u1: int = 1, num_inducing_u2: int = 1) -> None:
        super().__init__(bag_data)
        self.num_inducing_u1 = num_inducing_u1
        self.num_inducing_u2 = num_inducing_u2
        self.kmeans_1 = KMeans(n_clusters=num_inducing_u1)
        self.kmeans_2 = KMeans(n_clusters=num_inducing_u2)
        self.dim_1 = self.__getitem__(self.bags[0])[-3].shape[1]
        self.dim_2 = self.__getitem__(self.bags[0])[-2].shape[1]

    def gen_bags(self):
        for bag in self.bags:
            yield self.__getitem__(bag)

    def gen_inducing(self) -> np.ndarray:
        landmark_points_u1 = np.zeros(
            [self.num_bags * self.num_inducing_u1, self.dim_1]
        )
        landmark_points_u2 = np.zeros(
            [self.num_bags * self.num_inducing_u1, self.dim_2]
        )
        for i, bag in enumerate(self.bags):
            landmark_points_u1[
                i * self.num_inducing_u1 : i * self.num_inducing_u1
                + self.num_inducing_u1,
                :,
            ] = self.kmeans_1.fit(
                self.__getitem__(bag)[-3]
            ).cluster_centers_
            
            landmark_points_u2[
                i * self.num_inducing_u2 : i * self.num_inducing_u2
                + self.num_inducing_u2,
                :,
            ] = self.kmeans_2.fit(
                self.__getitem__(bag)[-2]
            ).cluster_centers_
        return landmark_points_u1, landmark_points_u2

class BagDataGenerator(BagData):
    def __init__(self, bag_data: dict, num_inducing_clusters: int = 1) -> None:
        super().__init__(bag_data)
        self.num_inducing_clusters = num_inducing_clusters
        self.kmeans = KMeans(n_clusters=num_inducing_clusters)

    def gen_bags(self):
        for bag in self.bags:
            yield self.__getitem__(bag)

    def gen_inducing(self, active_dims) -> np.ndarray:
        landmark_points = np.zeros(
            [self.num_bags * self.num_inducing_clusters, len(active_dims)]
        )
        for i, bag in enumerate(self.bags):
            landmark_points[
                i * self.num_inducing_clusters : i * self.num_inducing_clusters
                + self.num_inducing_clusters,
                :,
            ] = self.kmeans.fit(
                self.__getitem__(bag)[2][:, active_dims]
            ).cluster_centers_

        return landmark_points

class BinomialBagDataGenerator(BagData):
    def __init__(self, bag_data: dict, num_inducing_clusters: int = 1) -> None:
        super().__init__(bag_data)
        self.num_inducing_clusters = num_inducing_clusters
        self.kmeans = KMeans(n_clusters=num_inducing_clusters)

    def gen_bags(self):
        for bag in self.bags:
            yield self.__getitem__(bag)

    def gen_inducing(self, active_dims) -> np.ndarray:
        landmark_points = np.zeros(
            [self.num_bags * self.num_inducing_clusters, len(active_dims)]
        )
        for i, bag in enumerate(self.bags):
            landmark_points[
                i * self.num_inducing_clusters : i * self.num_inducing_clusters
                + self.num_inducing_clusters,
                :,
            ] = self.kmeans.fit(
                self.__getitem__(bag)[3][:, active_dims]
            ).cluster_centers_

        return landmark_points