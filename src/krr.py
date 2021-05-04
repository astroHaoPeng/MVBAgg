import numpy as np
import gpflow
from tqdm import tqdm
from src.bagData import BagData
import scipy


class LRe:
    def __init__(
        self, train_bag: BagData, kernel: gpflow.kernels.Kernel, lmbda: np.float = 0.1
    ) -> None:
        self.train_bag = train_bag
        self.kernel = kernel
        self.lmbda = lmbda

    def bag_cov(self, bag_data1: BagData, bag_data2: BagData = None) -> np.ndarray:
        bags1 = bag_data1.bags
        if bag_data2:
            bags2 = bag_data2.bags
        else:
            bag_data2 = bag_data1
            bags2 = bag_data1.bags

        K = np.zeros((len(bags1), len(bags2)))

        for i, bag1 in tqdm(enumerate(bags1)):
            # neighbours, N_i, w_i, x_i, _ = bag_data1[bag1]
            N_i, w_i, x_i, _ = bag_data1[bag1]
            for j, bag2 in enumerate(bags2):
                # for j, bag2 in enumerate(neighbours.intersection(set(bags2))):
                N_j, w_j, x_j, _ = bag_data2[bag2]
                w_i, w_j, x_i, x_j = w_i[:N_i], w_j[:N_j], x_i[:N_i], x_j[:N_j]
                # K[i, j] = w_i.T @ self.kernel(x_i, x_j) @ w_j
                K[i, j] = np.mean(self.kernel(x_i, x_j))
        return K

    def fit(self, K: np.ndarray = None) -> None:
        """
        Fits the representer coefficients alpha = (K + \lambda*I)^{-1}y
        """
        if not K:
            K = self.bag_cov(self.train_bag)
        alpha = np.linalg.solve(K + self.lmbda * np.eye(K.shape[0]), self.train_bag.y)
        self.alpha = alpha

    def predict(self, new_bag: BagData) -> np.ndarray:
        """Makes a prediction with f_star = K_*^T alpha"""
        if self.alpha is None:
            raise ValueError(
                "Model not fitted yet or alpha is invalid. Try fit() first."
            )

        K_star_T = self.bag_cov(new_bag, self.train_bag)
        f_star = K_star_T @ self.alpha

        return f_star


class KRRe(LRe):
    def __init__(
        self,
        train_bag: BagData,
        kernel: gpflow.kernels.Kernel,
        lmbda: np.float = 0.1,
        lengthscale_rho: np.float = 1,
        scale_rho: np.float = 1,
        median_heuristic: bool = True,
    ) -> None:

        super().__init__(train_bag, kernel, lmbda)
        self.lengthscale_rho = lengthscale_rho
        self.scale_rho = scale_rho
        self.median_heuristic = median_heuristic

    def bag_cov(
        self, bag_data1: BagData, bag_data2: BagData = None, median_heuristic=False
    ) -> np.ndarray:
        bags1 = bag_data1.bags
        if bag_data2:
            bags2 = bag_data2.bags
        else:
            bag_data2 = bag_data1
            bags2 = bag_data1.bags

        K = np.zeros((len(bags1), len(bags2)))

        for i, bag1 in tqdm(enumerate(bags1)):
            # neighbours, N_i, w_i, x_i, _ = bag_data1[bag1]
            N_i, w_i, x_i, _ = bag_data1[bag1]
            for j, bag2 in enumerate(bags2):
                # for j, bag2 in enumerate(neighbours.intersection(set(bags2))):
                N_j, w_j, x_j, _ = bag_data2[bag2]
                w_i, w_j, x_i, x_j = w_i[:N_i], w_j[:N_j], x_i[:N_i], x_j[:N_j]
                # K[i, j] = w_i.T @ self.kernel(x_i, x_j) @ w_j
                K[i, j] = (
                    np.mean(self.kernel(x_i, x_i))
                    - 2 * np.mean(self.kernel(x_i, x_j))
                    + np.mean(self.kernel(x_j, x_j))
                )
        if median_heuristic == True:
            mean_embedding_dist = np.expand_dims(K.flatten(), axis=1)
            # restrict number of elements due to computational constraints
            # idx = random.sample(list(range(mean_embedding_dist.shape[0])), 2000)
            # mean_embedding_dist = mean_embedding_dist[idx]
            self.lengthscale_rho = np.quantile(
                scipy.spatial.distance.pdist(mean_embedding_dist, metric="euclidean"),
                0.5,
            )
        return self.scale_rho * np.exp(-K / (2 * self.lengthscale_rho))

    def fit(self, K: np.ndarray = None, median_heuristic=False) -> None:
        """
        Fits the representer coefficients alpha = (K + \lambda*I)^{-1}y
        """
        if not K:
            K = self.bag_cov(self.train_bag, median_heuristic=median_heuristic)
        alpha = np.linalg.solve(K + self.lmbda * np.eye(K.shape[0]), self.train_bag.y)
        self.alpha = alpha
