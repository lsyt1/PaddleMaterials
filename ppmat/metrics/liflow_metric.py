# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LiFlow nodal velocity metric.

Matches the reference ``FlowModule.compute_loss`` at commit
``e6fc475361d046865f12cae1aee11c4f56c48d87``: sum the squared error over the
three Cartesian components first, then average over nodes.  This deliberately
avoids the three-fold error one would introduce by averaging the three
components before taking the node mean.
"""

from __future__ import annotations

import paddle

__all__ = ["LiFlowMSE"]


class LiFlowMSE:
    """Nodal mean squared error, summed over the xyz dimensions."""

    def __call__(self, prediction, label):
        return paddle.mean(paddle.sum((prediction - label) ** 2, axis=-1))
