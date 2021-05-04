import scipy.special as sc
from gpflow.likelihoods import ScalarLikelihood
import tensorflow as tf


class Binomial(ScalarLikelihood):
    """

    Args:
        ScalarLikelihood ([type]): [description]
    """

    def __init__(self, invlink=None, **kwargs):
        super().__init__(**kwargs)

        if invlink is None:
            self.invlink = lambda f: 1 / (1 + tf.exp(-f))

    # def _predict_mean_and_var(self, Fmu, Fvar):
    #     if self.invlink is inv_probit:
    #         p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
    #         return p, p - tf.square(p)
    #     else:
    #         # for other invlink, use quadrature
    #         return super()._predict_mean_and_var(Fmu, Fvar)

    def _scalar_log_prob(self, F, Y, total_count):
        """log_prob inspired by PyTorch's Binomial class

        Args:
            F ([type]): batch x num_data x 1
            Y ([type]): batch x 1 x 1
            w ([type]): batch x num_data x 1
            total_num : batch x 1 x 1

        Returns:
            [type]: [description]
        """
        log_factorial_n = tf.math.lgamma(total_count + 1)
        log_factorial_k = tf.math.lgamma(Y + 1)
        log_factorial_nmk = tf.math.lgamma(total_count - Y + 1)
        # batch x 1
        logits = self.invlink(F)
        return (
            (total_count - Y) * tf.math.log(1 - logits)
            + Y * tf.math.log(logits)
            + log_factorial_n
            - log_factorial_k
            - log_factorial_nmk
        )

    def _quadrature_log_prob(self, F, Y, total_count):
        """
        Returns the appropriate log prob integrand for quadrature.

        Quadrature expects f(X), here logp(F), to return shape [N_quad_points]
        + batch_shape + [d']. Here d' corresponds to the last dimension of both
        F and Y, and _scalar_log_prob simply broadcasts over this.

        Also see _quadrature_reduction.
        """
        return self._scalar_log_prob(F, Y, total_count)

    def _quadrature_reduction(self, quadrature_result):
        """
        Converts the quadrature integral appropriately.

        The return shape of quadrature is batch_shape + [d']. Here, d'
        corresponds to the last dimension of both F and Y, and we want to sum
        over the observations to obtain the overall predict_log_density or
        variational_expectations.

        Also see _quadrature_log_prob.
        """
        return tf.reduce_sum(quadrature_result, axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y, total_count):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: variational expectations, with shape [...]
        """
        return self._quadrature_reduction(
            self.quadrature(
                self._quadrature_log_prob, Fmu, Fvar, Y=Y, total_count=total_count
            )
        )

    def variational_expectations(self, Fmu, Fvar, Y, total_count):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values,

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           ∫ log(p(y=Y|f)) q(f) df.

        This only works if the broadcasting dimension of the statistics of q(f) (mean and variance)
        are broadcastable with that of the data Y.

        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: expected log density of the data given q(F), with shape [...]
        """
        tf.debugging.assert_equal(tf.shape(Fmu), tf.shape(Fvar))
        # returns an error if Y[:-1] and Fmu[:-1] do not broadcast together
        _ = tf.broadcast_dynamic_shape(tf.shape(Fmu)[:-1], tf.shape(Y)[:-1])
        self._check_last_dims_valid(Fmu, Y)
        ret = self._variational_expectations(Fmu, Fvar, Y, total_count)
        self._check_return_shape(ret, Fmu, Y)
        return ret
