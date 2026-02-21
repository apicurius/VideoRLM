"""Tests for RLM performance optimizations."""

from __future__ import annotations

import json
import os
import tempfile

import pytest


class TestCompletionCache:
    """Test the hash-based LM completion cache in BaseLM."""

    def test_cache_key_deterministic(self):
        from rlm.clients.base_lm import _prompt_cache_key

        key1 = _prompt_cache_key("hello world")
        key2 = _prompt_cache_key("hello world")
        assert key1 == key2

    def test_cache_key_differs_for_different_prompts(self):
        from rlm.clients.base_lm import _prompt_cache_key

        key1 = _prompt_cache_key("hello")
        key2 = _prompt_cache_key("world")
        assert key1 != key2

    def test_cache_key_handles_message_lists(self):
        from rlm.clients.base_lm import _prompt_cache_key

        msgs = [{"role": "user", "content": "test"}]
        key = _prompt_cache_key(msgs)
        assert key is not None
        assert isinstance(key, str)

    def test_cache_key_returns_none_for_unhashable(self):
        from rlm.clients.base_lm import _prompt_cache_key

        # Circular reference cannot be JSON-serialized
        d = {}
        d["self"] = d
        key = _prompt_cache_key(d)
        assert key is None

    def test_cache_store_and_retrieve(self):
        from rlm.clients.base_lm import BaseLM, _prompt_cache_key

        # Create a concrete subclass for testing
        class FakeLM(BaseLM):
            def completion(self, prompt):
                return "response"

            async def acompletion(self, prompt):
                return "response"

            def get_usage_summary(self):
                return None

            def get_last_usage(self):
                return None

        lm = FakeLM(model_name="test")

        # Cache miss
        cached, key = lm.cached_completion("test prompt")
        assert cached is None
        assert key is not None

        # Store
        lm.cache_store(key, "cached response")

        # Cache hit
        cached, key2 = lm.cached_completion("test prompt")
        assert cached == "cached response"
        assert lm._cache_hits == 1

    def test_cache_disabled(self):
        from rlm.clients.base_lm import BaseLM

        class FakeLM(BaseLM):
            def completion(self, prompt):
                return "response"

            async def acompletion(self, prompt):
                return "response"

            def get_usage_summary(self):
                return None

            def get_last_usage(self):
                return None

        lm = FakeLM(model_name="test", enable_cache=False)
        cached, key = lm.cached_completion("test")
        assert cached is None
        assert key is None


class TestAsyncLogger:
    """Test the async file writer in RLMLogger."""

    def test_logger_writes_to_disk_async(self):
        from rlm.core.types import RLMIteration, RLMMetadata
        from rlm.logger.rlm_logger import RLMLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RLMLogger(log_dir=tmpdir)

            # Log metadata
            metadata = RLMMetadata(
                root_model="test",
                max_depth=1,
                max_iterations=10,
                backend="openai",
                backend_kwargs={},
                environment_type="local",
                environment_kwargs={},
            )
            logger.log_metadata(metadata)

            # Log an iteration
            iteration = RLMIteration(
                prompt="test",
                response="test response",
                code_blocks=[],
            )
            logger.log(iteration)

            # Flush to ensure writes complete
            logger.flush()

            # Verify file was written
            assert logger.log_file_path is not None
            with open(logger.log_file_path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # metadata + 1 iteration

            # Verify content is valid JSON
            meta_entry = json.loads(lines[0])
            assert meta_entry["type"] == "metadata"
            iter_entry = json.loads(lines[1])
            assert iter_entry["type"] == "iteration"

    def test_logger_in_memory_only(self):
        from rlm.core.types import RLMIteration, RLMMetadata
        from rlm.logger.rlm_logger import RLMLogger

        logger = RLMLogger()  # No log_dir
        assert logger._writer is None

        metadata = RLMMetadata(
            root_model="test",
            max_depth=1,
            max_iterations=10,
            backend="openai",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
        )
        logger.log_metadata(metadata)
        logger.log(
            RLMIteration(prompt="p", response="r", code_blocks=[])
        )

        trajectory = logger.get_trajectory()
        assert trajectory is not None
        assert len(trajectory["iterations"]) == 1


class TestContextWindowing:
    """Test the sliding window in RLM._apply_history_window."""

    def _make_rlm_instance(self, max_history_messages):
        """Create an RLM with minimal config for testing windowing."""
        from rlm.core.rlm import RLM

        # Mock just enough to test the windowing method
        rlm = object.__new__(RLM)
        rlm.max_history_messages = max_history_messages
        return rlm

    def test_no_window_when_disabled(self):
        rlm = self._make_rlm_instance(None)
        history = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg {i}"} for i in range(20)
        ]
        result = rlm._apply_history_window(history)
        assert len(result) == 21  # unchanged

    def test_window_trims_old_messages(self):
        rlm = self._make_rlm_instance(10)
        history = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg {i}"} for i in range(20)
        ]
        result = rlm._apply_history_window(history)
        # system + summary + 10 kept = 12
        assert len(result) == 12
        assert result[0]["role"] == "system"
        assert "summarized" in result[1]["content"]
        # Last message should be the most recent
        assert result[-1]["content"] == "msg 19"

    def test_window_preserves_system_prompt(self):
        rlm = self._make_rlm_instance(5)
        history = [{"role": "system", "content": "important system"}] + [
            {"role": "user", "content": f"msg {i}"} for i in range(10)
        ]
        result = rlm._apply_history_window(history)
        assert result[0]["content"] == "important system"

    def test_window_noop_when_under_limit(self):
        rlm = self._make_rlm_instance(50)
        history = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg {i}"} for i in range(5)
        ]
        result = rlm._apply_history_window(history)
        assert len(result) == 6  # unchanged


class TestLocalREPLDirectInjection:
    """Test that add_context uses direct injection instead of file I/O."""

    def test_string_context_injected_directly(self):
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(context_payload="hello world")
        assert repl.locals["context_0"] == "hello world"
        assert repl.locals["context"] == "hello world"
        repl.cleanup()

    def test_dict_context_injected_directly(self):
        from rlm.environments.local_repl import LocalREPL

        data = {"key": "value", "nested": [1, 2, 3]}
        repl = LocalREPL(context_payload=data)
        assert repl.locals["context_0"] == data
        assert repl.locals["context_0"] is not data  # deep copy
        repl.cleanup()

    def test_list_context_injected_directly(self):
        from rlm.environments.local_repl import LocalREPL

        data = [1, 2, 3]
        repl = LocalREPL(context_payload=data)
        assert repl.locals["context_0"] == data
        repl.cleanup()

    def test_add_multiple_contexts(self):
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(context_payload="first")
        repl.add_context("second")
        assert repl.locals["context_0"] == "first"
        assert repl.locals["context_1"] == "second"
        assert repl.get_context_count() == 2
        repl.cleanup()
