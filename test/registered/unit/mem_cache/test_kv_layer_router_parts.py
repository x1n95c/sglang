"""Unit tests for the layer router's part routes and their load-back waits."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.mem_cache.kv.router import STATE_PART, KVLayerRouter
from sglang.test.test_utils import CustomTestCase

_START_LAYER = 4
_FULL_LAYERS = {7: 0, 11: 1}
_STATE_LAYERS = {4: 0, 5: 1, 6: 2, 8: 3}


class _RecordingCounter:
    def __init__(self):
        self.waits = []

    def wait_until(self, threshold: int) -> None:
        self.waits.append(threshold)


class _Pool:
    """A pipeline stage starting at layer 4: full layers 7 and 11, state
    layers 4, 5, 6 and 8."""

    def __init__(self):
        self.start_layer = _START_LAYER
        self.layer_transfer_counter = None
        self.full_view = object()
        self.state_view = object()
        self.ngram_view = object()
        self.router = KVLayerRouter.from_layer_maps(
            self, [("full", self.full_view, _FULL_LAYERS)]
        )
        self.router.bind_part(
            sub_pool="mamba",
            part=STATE_PART,
            view=self.state_view,
            indices=_STATE_LAYERS,
        )
        self.router.bind_part(
            sub_pool="mamba",
            part="ngram",
            view=self.ngram_view,
            indices={min(_STATE_LAYERS): 0},
        )


class TestKVLayerRouterParts(CustomTestCase):
    def test_state_reads_wait_only_once_the_host_stack_loads_state(self):
        """A counter registered for KV alone must not hold up state reads, and
        one that also restores state must, at the same stage-local layer a KV
        read of that layer would wait for."""
        pool = _Pool()
        counter = _RecordingCounter()
        pool.layer_transfer_counter = counter

        pool.router.read(7)
        state = pool.router.read(6, STATE_PART)
        self.assertEqual(counter.waits, [7 - _START_LAYER])
        self.assertIs(state.view, pool.state_view)
        self.assertEqual(state.view_layer_id, 2)
        self.assertEqual(state.route.sub_pool, "mamba")

        pool.router.load_parts_layer_wise()
        pool.router.read(6, STATE_PART)
        self.assertEqual(counter.waits, [7 - _START_LAYER, 6 - _START_LAYER])

    def test_out_of_loop_readers_wait_for_their_own_layer(self):
        """The N-gram context is read before the layer loop, so it waits for the
        first state layer only; a whole-state copy reads every state layer, so
        it waits for the last. Waiting for the last before the loop would
        serialize the load-back with the forward."""
        pool = _Pool()
        counter = _RecordingCounter()
        pool.layer_transfer_counter = counter
        pool.router.load_parts_layer_wise()

        ngram = pool.router.read_part("ngram")
        self.assertIs(ngram.view, pool.ngram_view)
        pool.router.wait_for_part(STATE_PART)
        self.assertEqual(counter.waits, [4 - _START_LAYER, 8 - _START_LAYER])

    def test_parts_stay_out_of_the_token_sub_pools(self):
        """Callers that pick "the first layer holding KV" or iterate the token
        sub-pools must not see state layers."""
        pool = _Pool()
        self.assertEqual(pool.router.layer_ids(), (7, 11))
        self.assertEqual(
            [sub_pool.name for sub_pool in pool.router.sub_pools()], ["full"]
        )
        self.assertIsNone(pool.router.sub_pool("mamba"))
        with self.assertRaises(KeyError):
            pool.router.layer(6)
        with self.assertRaises(KeyError):
            pool.router.layer(7, STATE_PART)


if __name__ == "__main__":
    unittest.main()
