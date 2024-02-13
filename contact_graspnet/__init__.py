import os
from importlib.metadata import version

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # INFO messages are not printed

import tensorflow.compat.v1 as tf  # type: ignore

from .estimator import CGNGraspEstimator

__version__ = version("contact_graspnet")

tf.disable_eager_execution()
