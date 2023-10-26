# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math

from absl.testing import absltest

import numpy as np

import jax
from jax import lax
from jax import cppyy_utils
from jax._src import test_util as jtu
from jax import config

config.parse_flags_with_absl()

ignore_jit_of_pmap_warning = partial(
    jtu.ignore_warning,message=".*jit-of-pmap.*")


def compute_recall(result_neighbors, ground_truth_neighbors) -> float:
  """Computes the recall of an approximate nearest neighbor search.

  Args:
    result_neighbors: int32 numpy array of the shape [num_queries,
      neighbors_per_query] where the values are the indices of the dataset.
    ground_truth_neighbors: int32 numpy array of with shape [num_queries,
      ground_truth_neighbors_per_query] where the values are the indices of the
      dataset.

  Returns:
    The recall.
  """
  assert len(
      result_neighbors.shape) == 2, "shape = [num_queries, neighbors_per_query]"
  assert len(ground_truth_neighbors.shape
            ) == 2, "shape = [num_queries, ground_truth_neighbors_per_query]"
  assert result_neighbors.shape[0] == ground_truth_neighbors.shape[0]
  gt_sets = [set(np.asarray(x)) for x in ground_truth_neighbors]
  hits = sum(
      len(list(x
               for x in nn_per_q
               if x.item() in gt_sets[q]))
      for q, nn_per_q in enumerate(result_neighbors))
  return hits / ground_truth_neighbors.size


class AnnTest(jtu.JaxTestCase):

  # TODO(b/258315194) Investigate probability property when input is around
  # few thousands.
  @jtu.sample_product(
    qy_shape=[(200, 512), (128, 512)],
    db_shape=[(512, 500), (512, 3000)],
    dtype=jtu.dtypes.all_floating,
    k=[1, 10],
    recall=[0.95],
  )
  def test_approx_max_k(self, qy_shape, db_shape, dtype, k, recall):
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, np.float64)
    db = rng(db_shape, np.float64)
    scores = jax.cppyy_utils.std_vecmul(qy, db)
    _, gt_args = lax.top_k(scores, k)
    _, ann_args = lax.approx_max_k(scores, k, recall_target=recall)
    self.assertEqual(k, len(ann_args[0]))
    ann_recall = compute_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreaterEqual(ann_recall, recall*0.9)

  @jtu.sample_product(
    qy_shape=[(200, 512), (128, 512)],
    db_shape=[(512, 500), (512, 3000)],
    dtype=jtu.dtypes.all_floating,
    k=[1, 10],
    recall=[0.95],
  )
  def test_approx_min_k(self, qy_shape, db_shape, dtype, k, recall):
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, np.float64)
    db = rng(db_shape, np.float64)
    scores = jax.cppyy_utils.std_vecmul(qy, db)
    _, gt_args = lax.top_k(-scores, k)
    _, ann_args = lax.approx_min_k(scores, k, recall_target=recall)
    ann_recall = compute_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreaterEqual(ann_recall, recall*0.9)
  
if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())

