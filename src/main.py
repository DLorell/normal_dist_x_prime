from collections import OrderedDict
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
from jax import ops
from jax.scipy import special
jax.default_device(jax.devices('cpu')[0])

from src.dag_kl import dag_kl
from src.better_graphs import process_graph

def get_cov_test(n: int, alpha: float) -> jax.Array:
    ones_mat = jnp.eye(n+1)
    ones_mat = ones_mat.at[:, 0].set(-1)
    ones_mat = ones_mat.at[0, 0].set(1)

    alpha_mat = jnp.eye(n+1) * alpha
    alpha_mat = alpha_mat.at[0, 0].set(1)

    ones_inv = jnp.linalg.inv(ones_mat)

    cov = (ones_inv @ alpha_mat @ alpha_mat @ ones_inv.T)
    return cov



# TODO
"""
1) Low dimensional analysis of X'. (Should be ~1 here.) (Use SVD of cross_s)
2) Do similar analyses for other systems.
3) Clustering
4) Profit
"""

def basic_dkl_test(s):
    n = len(s) - 1

    true_independence = OrderedDict([
        ('names', {
            'theta': [0],
            'X': list(range(1, n + 1))
        }),
        ('theta', [None]),
        ('X', ['theta', None])
        # The None indicates independence within 'X' conditional on the specified parents (theta)
    ])
    print("Independence for 'true' latent:", dag_kl(jnp.array(s, dtype=jnp.float64), process_graph(true_independence)))

    true_strong_invar = OrderedDict([
        ('names', {
            'theta': [0],
            'X_i': [1],
            'X_i_bar': list(range(2, n + 1))
        }),
        ('theta', [None]),
        ('X_i', ['theta']),
        ('X_i_bar', ['X_i'])
    ])
    print("Strong Invariance for 'true' latent:", dag_kl(s, process_graph(true_strong_invar)))

    true_weak_invar = OrderedDict([
        ('names', {
            'theta': [0],
            'X_i': [1],
            'X_i_bar': list(range(2, n + 1))
        }),
        ('theta', [None]),
        ('X_i_bar', ['theta']),
        ('X_i', ['X_i_bar'])
    ])
    print("Weak Invariance for 'true' latent:", dag_kl(s, process_graph(true_weak_invar)))
    print("")


def x_prime_dkl_test(n, s_xxp):
    len_xp = len(s_xxp) - n
    xp_independence = OrderedDict([
        ('names', {
            'X': list(range(n)),
            'Xp': list(range(n, n + len_xp))
        }),
        ('X', ['Xp', None]),
        ('Xp', [])
    ])
    indep_dkl = dag_kl(s_xxp, process_graph(xp_independence))
    print("Independence for X given X'", indep_dkl)

    xp_single_var_independence = OrderedDict([
        ('names', {
            'X_i': [0],
            'X_i_bar': list(range(1, n)),
            'Xp': list(range(n, n + len_xp))
        }),
        ('Xp', []),
        ('X_i', ['Xp']),
        ('X_i_bar', ['Xp'])
    ])
    print("Single Var Independence for X':", dag_kl(s_xxp, process_graph(xp_single_var_independence)))

    xp_strong_invar = OrderedDict([
        ('names', {
            'X_i': [0],
            'X_i_bar': list(range(1, n)),
            'Xp': list(range(n, n + len_xp))
        }),
        ('Xp', []),
        ('X_i', ['Xp']),
        ('X_i_bar', ['X_i'])
    ])

    strong_invar_dkl = dag_kl(s_xxp, process_graph(xp_strong_invar))
    print("Strong Invariance for X':", strong_invar_dkl)

    xp_weak_invar = OrderedDict([
        ('names', {
            'X_i': [0],
            'X_i_bar': list(range(1, n)),
            'Xp': list(range(n, n + len_xp))
        }),
        ('Xp', []),
        ('X_i_bar', ['Xp']),
        ('X_i', ['X_i_bar'])
    ])
    print("Weak Invariance for X':", dag_kl(s_xxp, process_graph(xp_weak_invar)))
    print("")




if __name__ == '__main__':
    n = 24
    alpha = 0.5

    cov = get_cov_test(n, alpha)
    s_full = jnp.linalg.inv(cov)
    s_x = jnp.linalg.inv(cov[1:, 1:])

    cross_s = s_x - jnp.diag(jnp.diag(s_x))
    s_xxp = jnp.block([[s_x + (cross_s / jnp.diag(s_x)) @ cross_s, cross_s], [cross_s, jnp.diag(jnp.diag(s_x))]])

    s = s_full

    # Testing if Xp is a nat lat
    basic_dkl_test(s)
    x_prime_dkl_test(n, s_xxp)

    # Low dim analysis of Xp
    u, s, v = jnp.linalg.svd(cross_s, full_matrices=False)
    sigma_xxp = jnp.linalg.inv(s_xxp)
    sigma_xp = sigma_xxp[n:, n:]
    sqrt_sigma_xp = jnp.linalg.cholesky(sigma_xp)
    A = jnp.block([[jnp.block([jnp.eye(n), jnp.zeros((n, n))])], [0.0]*n +[v[0] @ sqrt_sigma_xp.T]])
    sigma_xlambda = A @ sigma_xxp @ A.T
    s_xlambda = jnp.linalg.inv(sigma_xlambda)

    print("> Below, X' is actually (supposed to be) low-dim Lambda <")
    x_prime_dkl_test(n, s_xlambda)
