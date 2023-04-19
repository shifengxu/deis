# @title Autoload all modules
# %load_ext autoreload
# %autoreload 2

# from deis.ipynb

# deis.ipynb: In 1
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import matplotlib
import importlib
import os
import functools
import itertools
import jax.random as random

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import inspect

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import run_lib

import sampling
import losses as losses_lib
import utils
import evaluation
from models import up_or_down_sampling as stylegan_layers
import datasets
from models import wideresnet_noise_conditional
import sde_lib
import likelihood
import controllable_generation

from sampling import *
from sde_lib import *
from scipy import integrate

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# deis.ipynb: In 2
# @title Load the score-based model

from configs.vp import cifar10_ddpmpp_continuous as configs

# TODO: change ckpt path
ckpt_filename = os.path.expanduser("~/Coding/deis/ckpt/jax/vp/cifar10_ddpmpp_continuous/checkpoint_26.pth")
assert os.path.exists(ckpt_filename), f"file not exist: {ckpt_filename}"
# config = configs.get_config()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
sampling_eps = 1e-3

batch_size = 64 # @param {"type":"integer"}
local_batch_size = batch_size // jax.local_device_count()
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 # @param {"type": "integer"}
rng = jax.random.PRNGKey(random_seed)
rng, run_rng = jax.random.split(rng)
rng, model_rng = jax.random.split(rng)
score_model, init_model_state, initial_params = mutils.init_model(run_rng, config)
optimizer = losses_lib.get_optimizer(config).create(initial_params)

state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                      model_state=init_model_state,
                      ema_rate=config.model.ema_rate,
                      params_ema=initial_params,
                      rng=rng)  # pytype: disable=wrong-keyword-args
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
state = utils.load_training_state(ckpt_filename, state)


# deis.ipynb: In 3
# @title Visualization code
def image_grid(x):
    size = config.data.image_size
    channels = config.data.num_channels
    img = x.reshape(-1, size, size, channels)
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
    return img

def show_samples(x):
    img = image_grid(x)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# deis.ipynb: In 5
rng = random.PRNGKey(888)
noise = random.normal(rng, (100, 32, 32, 3))

# play deis ###################################################################
# deis.ipynb In 5, 6, 7, 8, 9, 10
import jax_deis as jdeis

t2alpha_fn, alpha2t_fn = jdeis.get_linear_alpha_fns(sde.beta_0, sde.beta_1)
vpsde = jdeis.VPSDE(t2alpha_fn, alpha2t_fn, sampling_eps, sde.T)
score_fn = mutils.get_score_fn(sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)

def eps_fn(x, scalar_t):
  vec_t = scalar_t * jnp.ones(x.shape[0])
  score = score_fn(x, vec_t)
  std = sde.marginal_prob(jnp.zeros_like(score), vec_t)[1]
  eps = - batch_mul(score, std)
  return eps

num_step = 7
t_ab_fn = jdeis.get_sampler(vpsde, eps_fn, "t", 2, num_step, method="t_ab", ab_order=3)
t_ab_7_img = t_ab_fn(noise)
show_samples(inverse_scaler(t_ab_7_img))
