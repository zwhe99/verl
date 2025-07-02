# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from verl.utils import omega_conf_to_dataclass
from verl.utils.debug import ProfilerConfig
from verl.utils.debug.nvtx_profile import NsightSystemsProfiler


class TestNsightSystemsProfiler(unittest.TestCase):
    """Test suite for NsightSystemsProfiler functionality.

    Test Plan:
    1. Initialization: Verify profiler state after creation
    2. Basic Profiling: Test start/stop functionality
    3. Discrete Mode: Test discrete profiling behavior
    4. Annotation: Test the annotate decorator in both normal and discrete modes
    5. Config Validation: Verify proper config initialization from OmegaConf
    """

    def setUp(self):
        self.config = ProfilerConfig(all_ranks=True)
        self.rank = 0
        self.profiler = NsightSystemsProfiler(self.rank, self.config)

    def test_initialization(self):
        self.assertEqual(self.profiler.this_rank, True)
        self.assertEqual(self.profiler.this_step, False)
        self.assertEqual(self.profiler.discrete, False)

    def test_start_stop_profiling(self):
        with patch("torch.cuda.profiler.start") as mock_start, patch("torch.cuda.profiler.stop") as mock_stop:
            # Test start
            self.profiler.start()
            self.assertTrue(self.profiler.this_step)
            mock_start.assert_called_once()

            # Test stop
            self.profiler.stop()
            self.assertFalse(self.profiler.this_step)
            mock_stop.assert_called_once()

    def test_discrete_profiling(self):
        discrete_config = ProfilerConfig(discrete=True, all_ranks=True)
        profiler = NsightSystemsProfiler(self.rank, discrete_config)

        with patch("torch.cuda.profiler.start") as mock_start, patch("torch.cuda.profiler.stop") as mock_stop:
            profiler.start()
            self.assertTrue(profiler.this_step)
            mock_start.assert_not_called()  # Shouldn't start immediately in discrete mode

            profiler.stop()
            self.assertFalse(profiler.this_step)
            mock_stop.assert_not_called()  # Shouldn't stop immediately in discrete mode

    def test_annotate_decorator(self):
        mock_self = MagicMock()
        mock_self.profiler = self.profiler
        mock_self.profiler.this_step = True

        @NsightSystemsProfiler.annotate(message="test")
        def test_func(self, *args, **kwargs):
            return "result"

        with patch("torch.cuda.profiler.start") as mock_start, patch("torch.cuda.profiler.stop") as mock_stop, patch(
            "verl.utils.debug.nvtx_profile.mark_start_range"
        ) as mock_start_range, patch("verl.utils.debug.nvtx_profile.mark_end_range") as mock_end_range:
            result = test_func(mock_self)
            self.assertEqual(result, "result")
            mock_start_range.assert_called_once()
            mock_end_range.assert_called_once()
            mock_start.assert_not_called()  # Not discrete mode
            mock_stop.assert_not_called()  # Not discrete mode

    def test_annotate_discrete_mode(self):
        discrete_config = ProfilerConfig(discrete=True, all_ranks=True)
        profiler = NsightSystemsProfiler(self.rank, discrete_config)
        mock_self = MagicMock()
        mock_self.profiler = profiler
        mock_self.profiler.this_step = True

        @NsightSystemsProfiler.annotate(message="test")
        def test_func(self, *args, **kwargs):
            return "result"

        with patch("torch.cuda.profiler.start") as mock_start, patch("torch.cuda.profiler.stop") as mock_stop, patch(
            "verl.utils.debug.nvtx_profile.mark_start_range"
        ) as mock_start_range, patch("verl.utils.debug.nvtx_profile.mark_end_range") as mock_end_range:
            result = test_func(mock_self)
            self.assertEqual(result, "result")
            mock_start_range.assert_called_once()
            mock_end_range.assert_called_once()
            mock_start.assert_called_once()  # Should start in discrete mode
            mock_stop.assert_called_once()  # Should stop in discrete mode

    def test_config_init(self):
        cfg = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
        arr = cfg.actor_rollout_ref
        for config in [
            cfg.critic.profiler,
            arr.actor.profiler,
            cfg.reward_model.profiler,
            arr.ref.profiler,
            arr.rollout.profiler,
        ]:
            profiler_config = omega_conf_to_dataclass(config, ProfilerConfig)
            self.assertEqual(profiler_config.discrete, False)
            self.assertEqual(profiler_config.all_ranks, False)
            self.assertEqual(profiler_config.ranks, [])
            assert isinstance(profiler_config, ProfilerConfig)


if __name__ == "__main__":
    unittest.main()
