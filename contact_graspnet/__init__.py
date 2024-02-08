from importlib.metadata import version

import tensorflow.compat.v1 as tf

from .estimator import CGNGraspEstimator

__version__ = version("contact_graspnet")

tf.disable_eager_execution()
