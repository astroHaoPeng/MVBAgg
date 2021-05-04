import tensorflow as tf
import gpflow
from gpflow import kullback_leiblers
from gpflow import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive, triangular
from gpflow.models import (
    SVGP,
    GPModel,
    ExternalDataTrainingLossMixin,
    InternalDataTrainingLossMixin,
)
from gpflow.conditionals import conditional
from gpflow.models.util import inducingpoint_wrapper
from gpflow.likelihoods import Gaussian
from gpflow.models.util import data_input_to_tensor
from gpflow.logdensities import multivariate_normal
import numpy as np
from typing import Tuple
from src.bagData import BagData
from tqdm import tqdm


class VBagg(SVGP):
    """
    This is the VBagg
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        num_data=None,
    ):
        """
        Modified from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html
        """
        # init the super class, accept args
        super().__init__(
            kernel,
            likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
        )
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        num_inducing = len(self.inducing_variable)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        N, w, X, Y = data
        kl = self.prior_kl()
        f_mean, f_cov = self.predict_f(X, full_cov=True)
        f_cov = tf.squeeze(f_cov, axis=1)
        var_exp = self.likelihood.variational_expectations(
            tf.reduce_sum(tf.multiply(w, f_mean), axis=1),
            tf.squeeze(
                tf.matmul(tf.matmul(tf.transpose(w, perm=[0, 2, 1]), f_cov), w), axis=1
            ),
            Y,
        )
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        # tf.expand_dims(self.mean_function(Xnew[:,:,-1:]), 1) + mu, var for USA???
        return mu, var

    def _build_variational_params(self, w: np.ndarray, x: np.ndarray):
        f_mean, f_cov = self.predict_f(x, full_cov=True)
        f_cov = tf.squeeze(f_cov, axis=1)
        # argmax_ind = tf.argmax(f_mean, axis=1).numpy()[:,0]
        # ind = [(i, argmax_ind[i]) for i in range(w.shape[0])]
        # f_mean_agg = tf.expand_dims(tf.gather_nd(f_mean, ind), axis=1)
        # ind = [(i, argmax_ind[i], i) for i in range(w.shape[0])]
        # f_cov_agg = tf.gather_nd(f_cov, ind)

        return (
            tf.reduce_sum(tf.multiply(w, f_mean), axis=1),
            tf.squeeze(
                tf.matmul(tf.matmul(tf.transpose(w, perm=[0, 2, 1]), f_cov), w), axis=1
            ),
        )

class BinomialVBagg(VBagg):
    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        N, total_count, w, X, Y = data
        kl = self.prior_kl()
        f_mean, f_cov = self.predict_f(X, full_cov=True)
        f_cov = tf.squeeze(f_cov, axis=1)
        var_exp = self.likelihood.variational_expectations(
            tf.reduce_sum(tf.multiply(w, f_mean), axis=1),
            tf.squeeze(
                tf.matmul(tf.matmul(tf.transpose(w, perm=[0, 2, 1]), f_cov), w), axis=1
            ),
            Y,
            total_count,
        )
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl


class MultiResolutionVBagg(GPModel, ExternalDataTrainingLossMixin):
    """
    Modified from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html
    """

    def __init__(
        self,
        kernel,
        likelihood,
        z1,
        z2,
        num_outputs: int = 1,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        whiten: bool = True,
        num_data=None,
    ):
        """"""
        # init the super class, accept args
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_outputs = num_outputs
        self.whiten = whiten
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.z1 = inducingpoint_wrapper(z1)
        self.z2 = inducingpoint_wrapper(z2)

        # unpack kernel
        self.kernel1 = self.kernel[0]
        self.kernel2 = self.kernel[1]

        # unpack mean
        # init variational parameters
        num_inducing_z1 = len(self.z1)
        num_inducing_z2 = len(self.z2)

        q_mu_z1 = np.zeros((num_inducing_z1, self.num_latent_gps))
        self.q_mu_z1 = Parameter(q_mu_z1, dtype=default_float())

        q_sqrt_z1 = [
            np.eye(num_inducing_z1, dtype=default_float())
            for _ in range(self.num_latent_gps)
        ]
        q_sqrt_z1 = np.array(q_sqrt_z1)
        self.q_sqrt_z1 = Parameter(q_sqrt_z1, transform=triangular())

        q_mu_z2 = np.zeros((num_inducing_z2, self.num_latent_gps))
        self.q_mu_z2 = Parameter(q_mu_z2, dtype=default_float())

        q_sqrt_z2 = [
            np.eye(num_inducing_z2, dtype=default_float())
            for _ in range(self.num_latent_gps)
        ]
        q_sqrt_z2 = np.array(q_sqrt_z2)
        self.q_sqrt_z2 = Parameter(q_sqrt_z2, transform=triangular())

    def prior_kl_z1(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.z1, self.kernel1, self.q_mu_z1, self.q_sqrt_z1, whiten=self.whiten
        )

    def prior_kl_z2(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.z2, self.kernel2, self.q_mu_z2, self.q_sqrt_z2, whiten=self.whiten
        )

    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        _, _, w1, w2, x1, x2, y = data
        kl = self.prior_kl_z1() + self.prior_kl_z2()

        mu, var = self.predict_f(w1, x1, w2, x2)
        # var_exp = self.likelihood._variational_expectations(w1, w2, mu, var, y)
        var_exp = self.likelihood.variational_expectations(mu, var, y)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(y)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def maximum_log_likelihood_objective(
        self,
        data: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ) -> tf.Tensor:
        return self.elbo(data)

    def _build_variational_params_z1(self, w: np.ndarray, x: np.ndarray):
        f_mean, f_cov = self.predict_f_z1(x, full_cov=True)
        f_cov = tf.squeeze(f_cov, axis=1)
        # argmax_ind = tf.argmax(f_mean, axis=1).numpy()[:,0]
        # ind = [(i, argmax_ind[i]) for i in range(w.shape[0])]
        # f_mean_agg = tf.expand_dims(tf.gather_nd(f_mean, ind), axis=1)
        # ind = [(i, argmax_ind[i], i) for i in range(w.shape[0])]
        # f_cov_agg = tf.gather_nd(f_cov, ind)

        return (
            tf.reduce_sum(tf.multiply(w, f_mean), axis=1),
            tf.squeeze(
                tf.matmul(tf.matmul(tf.transpose(w, perm=[0, 2, 1]), f_cov), w), axis=1
            ),
        )
        # return f_mean_agg, f_cov_agg

    def _build_variational_params_z2(self, w: np.ndarray, x: np.ndarray):
        f_mean, f_cov = self.predict_f_z2(x, full_cov=True)
        f_cov = tf.squeeze(f_cov, axis=1)

        # argmax_ind = tf.argmax(f_mean, axis=1)[:,0]
        # ind = [(i, argmax_ind[i]) for i in range(w.shape[0])]
        # f_mean_agg = tf.expand_dims(tf.gather_nd(f_mean, ind), axis=1)
        # ind = [(i, argmax_ind[i], i) for i in range(w.shape[0])]
        # f_cov_agg = tf.gather_nd(f_cov, ind)

        return (
            tf.reduce_sum(tf.multiply(w, f_mean), axis=1),
            tf.squeeze(
                tf.matmul(tf.matmul(tf.transpose(w, perm=[0, 2, 1]), f_cov), w), axis=1
            ),
        )
        # return f_mean_agg, f_cov_agg

    def predict_f_z1(self, xnew: np.ndarray, full_cov=False, full_output_cov=False):
        q_mu_z1 = self.q_mu_z1
        q_sqrt_z1 = self.q_sqrt_z1
        mu, var = conditional(
            xnew,
            self.z1,
            self.kernel1,
            q_mu_z1,
            q_sqrt=q_sqrt_z1,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        return mu, var

    def predict_f_z2(self, xnew: np.ndarray, full_cov=False, full_output_cov=False):
        q_mu_z2 = self.q_mu_z2
        q_sqrt_z2 = self.q_sqrt_z2
        mu, var = conditional(
            xnew,
            self.z2,
            self.kernel2,
            q_mu_z2,
            q_sqrt=q_sqrt_z2,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        return mu, var

    def predict_f(
        self,
        w1: np.ndarray,
        x1: np.ndarray,
        w2: np.ndarray,
        x2: np.ndarray,
        full_cov=False,
        full_output_cov=False,
    ):
        muz1, var1 = self._build_variational_params_z1(w1, x1)
        muz2, var2 = self._build_variational_params_z2(w2, x2)

        return muz1 + muz2, var1 + var2


class MultiResolutionSpatialVBagg(GPModel, ExternalDataTrainingLossMixin):
    """
    Modified from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html
    """

    def __init__(
        self,
        kernel,
        likelihood,
        zs,
        z1,
        z2,
        num_outputs: int = 1,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        whiten: bool = True,
        num_data=None,
    ):
        """TODO:
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_outputs = num_outputs
        self.whiten = whiten
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.zs = inducingpoint_wrapper(zs)
        self.z1 = inducingpoint_wrapper(z1)
        self.z2 = inducingpoint_wrapper(z2)

        # unpack kernel
        self.kernels = self.kernel[0]
        self.kernel1 = self.kernel[1]
        self.kernel2 = self.kernel[2]

        # unpack mean
        # init variational parameters
        num_inducing_zs = len(self.zs)
        num_inducing_z1 = len(self.z1)
        num_inducing_z2 = len(self.z2)

        q_mu_zs = np.zeros((num_inducing_zs, self.num_latent_gps))
        self.q_mu_zs = Parameter(q_mu_zs, dtype=default_float())

        q_sqrt_zs = [
            np.eye(num_inducing_zs, dtype=default_float())
            for _ in range(self.num_latent_gps)
        ]
        q_sqrt_zs = np.array(q_sqrt_zs)
        self.q_sqrt_zs = Parameter(q_sqrt_zs, transform=triangular())

        q_mu_z1 = np.zeros((num_inducing_z1, self.num_latent_gps))
        self.q_mu_z1 = Parameter(q_mu_z1, dtype=default_float())

        q_sqrt_z1 = [
            np.eye(num_inducing_z1, dtype=default_float())
            for _ in range(self.num_latent_gps)
        ]
        q_sqrt_z1 = np.array(q_sqrt_z1)
        self.q_sqrt_z1 = Parameter(q_sqrt_z1, transform=triangular())

        q_mu_z2 = np.zeros((num_inducing_z2, self.num_latent_gps))
        self.q_mu_z2 = Parameter(q_mu_z2, dtype=default_float())

        q_sqrt_z2 = [
            np.eye(num_inducing_z2, dtype=default_float())
            for _ in range(self.num_latent_gps)
        ]
        q_sqrt_z2 = np.array(q_sqrt_z2)
        self.q_sqrt_z2 = Parameter(q_sqrt_z2, transform=triangular())

    def prior_kl_zs(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.zs, self.kernels, self.q_mu_zs, self.q_sqrt_zs, whiten=self.whiten
        )

    def prior_kl_z1(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.z1, self.kernel1, self.q_mu_z1, self.q_sqrt_z1, whiten=self.whiten
        )

    def prior_kl_z2(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.z2, self.kernel2, self.q_mu_z2, self.q_sqrt_z2, whiten=self.whiten
        )

    def elbo(self, data) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        _, _, _, ws, w1, w2, xs, x1, x2, y = data
        kl = self.prior_kl_zs() + self.prior_kl_z1() + self.prior_kl_z2()

        mu, var = self.predict_f(ws, xs, w1, x1, w2, x2)
        # var_exp = self.likelihood._variational_expectations(w1, w2, mu, var, y)
        var_exp = self.likelihood.variational_expectations(mu, var, y)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(y)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def maximum_log_likelihood_objective(
        self,
        data: Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ) -> tf.Tensor:
        return self.elbo(data)

    def _build_variational_params_zs(self, w: np.ndarray, x: np.ndarray):
        f_mean, f_cov = self.predict_f_zs(x, full_cov=True)
        f_cov = tf.squeeze(f_cov, axis=1)
        # argmax_ind = tf.argmax(f_mean, axis=1).numpy()[:,0]
        # ind = [(i, argmax_ind[i]) for i in range(w.shape[0])]
        # f_mean_agg = tf.expand_dims(tf.gather_nd(f_mean, ind), axis=1)
        # ind = [(i, argmax_ind[i], i) for i in range(w.shape[0])]
        # f_cov_agg = tf.gather_nd(f_cov, ind)

        return (
            tf.reduce_sum(tf.multiply(w, f_mean), axis=1),
            tf.squeeze(
                tf.matmul(tf.matmul(tf.transpose(w, perm=[0, 2, 1]), f_cov), w), axis=1
            ),
        )

    def _build_variational_params_z1(self, w: np.ndarray, x: np.ndarray):
        f_mean, f_cov = self.predict_f_z1(x, full_cov=True)
        f_cov = tf.squeeze(f_cov, axis=1)
        # argmax_ind = tf.argmax(f_mean, axis=1).numpy()[:,0]
        # ind = [(i, argmax_ind[i]) for i in range(w.shape[0])]
        # f_mean_agg = tf.expand_dims(tf.gather_nd(f_mean, ind), axis=1)
        # ind = [(i, argmax_ind[i], i) for i in range(w.shape[0])]
        # f_cov_agg = tf.gather_nd(f_cov, ind)

        return (
            tf.reduce_sum(tf.multiply(w, f_mean), axis=1),
            tf.squeeze(
                tf.matmul(tf.matmul(tf.transpose(w, perm=[0, 2, 1]), f_cov), w), axis=1
            ),
        )
        # return f_mean_agg, f_cov_agg

    def _build_variational_params_z2(self, w: np.ndarray, x: np.ndarray):
        f_mean, f_cov = self.predict_f_z2(x, full_cov=True)
        f_cov = tf.squeeze(f_cov, axis=1)

        # argmax_ind = tf.argmax(f_mean, axis=1)[:,0]
        # ind = [(i, argmax_ind[i]) for i in range(w.shape[0])]
        # f_mean_agg = tf.expand_dims(tf.gather_nd(f_mean, ind), axis=1)
        # ind = [(i, argmax_ind[i], i) for i in range(w.shape[0])]
        # f_cov_agg = tf.gather_nd(f_cov, ind)

        return (
            tf.reduce_sum(tf.multiply(w, f_mean), axis=1),
            tf.squeeze(
                tf.matmul(tf.matmul(tf.transpose(w, perm=[0, 2, 1]), f_cov), w), axis=1
            ),
        )
        # return f_mean_agg, f_cov_agg

    def predict_f_zs(self, xnew: np.ndarray, full_cov=False, full_output_cov=False):
        q_mu_zs = self.q_mu_zs
        q_sqrt_zs = self.q_sqrt_zs
        mu, var = conditional(
            xnew,
            self.zs,
            self.kernels,
            q_mu_zs,
            q_sqrt=q_sqrt_zs,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        return mu, var

    def predict_f_z1(self, xnew: np.ndarray, full_cov=False, full_output_cov=False):
        q_mu_z1 = self.q_mu_z1
        q_sqrt_z1 = self.q_sqrt_z1
        mu, var = conditional(
            xnew,
            self.z1,
            self.kernel1,
            q_mu_z1,
            q_sqrt=q_sqrt_z1,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        return mu, var

    def predict_f_z2(self, xnew: np.ndarray, full_cov=False, full_output_cov=False):
        q_mu_z2 = self.q_mu_z2
        q_sqrt_z2 = self.q_sqrt_z2
        mu, var = conditional(
            xnew,
            self.z2,
            self.kernel2,
            q_mu_z2,
            q_sqrt=q_sqrt_z2,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        return mu, var

    def predict_f(
        self,
        ws: np.ndarray,
        xs: np.ndarray,
        w1: np.ndarray,
        x1: np.ndarray,
        w2: np.ndarray,
        x2: np.ndarray,
        full_cov=False,
        full_output_cov=False,
    ):
        muzs, vars = self._build_variational_params_zs(ws, xs)
        muz1, var1 = self._build_variational_params_z1(w1, x1)
        muz2, var2 = self._build_variational_params_z2(w2, x2)

        return muzs + muz1 + muz2, vars + var1 + var2
