"""Which pools the fused RoPE + KV store may write on ROCm.

The fused store writes window layers through `full_to_swa_index_mapping`. A
unified pool's window view takes ids only its write door derives, so the fused
store must step aside for it and leave the write to `set_kv_buffer`.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.kv.router import PublicSubPool
from sglang.srt.mem_cache.kv_loc_plan import IdSpaceKind
from sglang.srt.models import utils as model_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_P16 = dict(page_size=16, full_tokens=64 * 16, swa_tokens=64 * 16)


def _unified_pool(**geometry):
    setup_allocator_context()
    return build_unified_bundle(UnifiedGeometry(**geometry, **_P16)).token_to_kv_pool


def _static_window_pool():
    """A static full + window pool: its window view writes at ids the mapping gives."""
    window = PublicSubPool(name=IdSpaceKind.SLIDING_WINDOW, layer_ids=(1,))
    views = {
        IdSpaceKind.SLIDING_WINDOW: SimpleNamespace(requires_physical_write_loc=False)
    }
    router = SimpleNamespace(
        sub_pool=lambda name: window if name == window.name else None,
        view=lambda name: views[name],
    )
    return SimpleNamespace(layer_router=router)


def _hip_gate(pool, dcp_kv_mask=None) -> bool:
    with (
        patch.object(model_utils, "_is_hip", True),
        patch.object(model_utils, "_is_cuda", False),
        patch.object(model_utils, "get_token_to_kv_pool", return_value=pool),
    ):
        return model_utils.enable_fused_set_kv_buffer(
            SimpleNamespace(dcp_kv_mask=dcp_kv_mask)
        )


class TestHipFusedStoreGate(CustomTestCase):
    def test_unified_window_pool_takes_the_write_door(self):
        pool = _unified_pool(chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3))
        self.assertFalse(_hip_gate(pool))

    def test_unified_three_band_pool_takes_the_write_door(self):
        pool = _unified_pool(
            chain="tri",
            full_layer_ids=(0, 1),
            swa_layer_ids=(2, 3),
            state_layer_ids=(0, 2),
        )
        self.assertFalse(_hip_gate(pool))

    def test_unified_pool_without_a_window_stays_fused(self):
        pool = _unified_pool(
            chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3)
        )
        self.assertTrue(_hip_gate(pool))

    def test_static_window_pool_stays_fused(self):
        self.assertTrue(_hip_gate(_static_window_pool()))

    def test_decode_context_parallel_mask_declines(self):
        self.assertFalse(_hip_gate(_static_window_pool(), dcp_kv_mask=object()))


if __name__ == "__main__":
    unittest.main()
