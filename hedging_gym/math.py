import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.lax import map
from jaxfin.price_engine.math.norm import cum_normal


@jit
def compute_d1(spots_1, spots_2, sigma_sqrt_t):
    return (jnp.log(spots_2 / spots_1) / sigma_sqrt_t) + 0.5 * sigma_sqrt_t


@jit
def compute_d1_d2(spots_1, spots_2, expires, sigma_1, sigma_2, corr):
    sigma = jnp.sqrt(
        jnp.sum(jnp.asarray([sigma_1, sigma_2]) ** 2)
        - 2 * corr * jnp.prod(jnp.asarray([sigma_1, sigma_2]))
    )

    sigma_sqrt_t = sigma * jnp.sqrt(expires)
    d1 = compute_d1(spots_1, spots_2, sigma_sqrt_t)
    d2 = d1 - sigma_sqrt_t

    return d1, d2


@jit
def margrabe(
    spots_1: jax.Array,
    spots_2: jax.Array,
    expires: jax.Array,
    sigma_1: jax.Array,
    sigma_2: jax.Array,
    corr: jax.Array,
):
    """
    Calculate the price of a spread option max(S_2 - S_1, 0) using the Margrabe formula.
    Both S_2 and S_1 are assumed to be log-normally distributed with correlation corr.

    :param spots_1: The spot price of the first asset
    :param spots_2: The spot price of the second asset
    :param expires: The time to expiration of the option
    :param sigma_1: The volatility of the first asset
    :param sigma_2: The volatility of the second asset
    :param corr: The correlation between the two assets
    :param dtype: The dtype of the input
    :return: The price of the spread option
    """
    d1, d2 = compute_d1_d2(spots_1, spots_2, expires, sigma_1, sigma_2, corr)

    return spots_2 * cum_normal(d1) - spots_1 * cum_normal(d2)


@jit
def margrabe_deltas(spots_1, spots_2, expires, sigma_1, sigma_2, corr) -> jax.Array:
    """
    Calculate the deltas (with respect to S1 and S2 respectively) of a spread option max(S_2 - S_1, 0) using the Margrabe formula.

    :param spots_1: The spot price of the first asset
    :param spots_2: The spot price of the second asset
    :param expires: The time to expiration of the option
    :param vols: The volatility of the two assets
    :param corr: The correlation between the two assets
    :return: The deltas of the spread option
    """
    delta_1 = grad(margrabe, argnums=0)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )
    delta_2 = grad(margrabe, argnums=1)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )

    return jnp.asarray([delta_1, delta_2])


@jit
def margrabe_gammas(spots_1, spots_2, expires, sigma_1, sigma_2, corr):
    """

    :param spots_1:
    :param spots_2:
    :param expires:
    :param vols:
    :param corr:
    :param exchange_first:
    :param dtype:
    :return:
    """
    gamma_1 = grad(grad(margrabe, argnums=0), argnums=0)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )
    gamma_2 = grad(grad(margrabe, argnums=1), argnums=1)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )

    return jnp.asarray([gamma_1, gamma_2])


@jit
def margrabe_cross_gamma(spots_1, spots_2, expires, sigma_1, sigma_2, corr):
    """

    :param spots_1:
    :param spots_2:
    :param expires:
    :param vols:
    :param corr:
    :param exchange_first:
    :param dtype:
    :return:
    """
    cross_gamma_1 = grad(grad(margrabe, argnums=0), argnums=1)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )
    cross_gamma_2 = grad(grad(margrabe, argnums=1), argnums=0)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )

    return jnp.asarray([cross_gamma_1, cross_gamma_2])


def v_margrabe(
    spots_1: jax.Array,
    spots_2: jax.Array,
    expires: jax.Array,
    sigma_1: jax.Array,
    sigma_2: jax.Array,
    corr: jax.Array,
):
    """
    Vectorized version of the Margrabe formula.

    :param spots_1:
    :param spots_2:
    :param expires:
    :param sigma_1:
    :param sigma_2:
    :param corr:
    :param exchange_first:
    :param dtype:
    :return:
    """
    if spots_1.shape == ():
        spots_1 = jnp.expand_dims(spots_1, axis=0)
        spots_2 = jnp.expand_dims(spots_2, axis=0)
        expires = jnp.expand_dims(expires, axis=0)
        sigma_1 = jnp.expand_dims(sigma_1, axis=0)
        sigma_2 = jnp.expand_dims(sigma_2, axis=0)
        corr = jnp.expand_dims(corr, axis=0)

    return map(
        lambda spots: margrabe(spots[0], spots[1], expires, sigma_1, sigma_2, corr),
        (spots_1, spots_2),
    )
