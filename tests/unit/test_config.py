"""
tests/unit/test_config.py
--------------------------
Tests for AgentConfig in agent/config_loader.py.

What we test:
  - Valid config loads and fields are correct
  - Invalid configs fail loudly at construction time
  - Config is immutable — no accidental mutation during a session
  - Missing config file raises a clear error

What we do NOT test:
  - Any API calls
  - Any database operations
  - Any file I/O beyond config loading
"""

import json
import pytest


# ---------------------------------------------------------------------------
# These imports will fail until Phase 1 creates the module.
# That is expected. The test file exists now so the structure is in place.
# Uncomment when agent/config_loader.py exists.
# ---------------------------------------------------------------------------
from agent.config_loader import AgentConfig


class TestAgentConfig:
    """Config loads correctly and validates its own invariants."""

    @pytest.fixture
    def valid_config_file(self, tmp_path):
        """A valid voices.json for use across tests."""
        config_file = tmp_path / "voices.json"
        config_file.write_text(json.dumps({
            "socrates_weight": 0.6,
            "feynman_weight": 0.4,
            "temperature": 0.5,
            "prompt_version": "v1.0",
            "active_thinkers": ["socrates", "feynman"],
        }))
        return config_file

    
    def test_valid_config_loads(self, valid_config_file):
        config = AgentConfig.load(valid_config_file)
        assert config.socrates_weight == 0.6
        assert config.feynman_weight == 0.4
        assert config.temperature == 0.5
        assert config.prompt_version == "v1.0"
        assert "socrates" in config.active_thinkers

    
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            AgentConfig(
                socrates_weight=0.6,
                feynman_weight=0.6,
                temperature=0.5,
                prompt_version="v1.0",
                active_thinkers=("socrates",),
            )

    
    def test_temperature_out_of_range(self):
        with pytest.raises(ValueError, match="temperature"):
            AgentConfig(
                socrates_weight=0.6,
                feynman_weight=0.4,
                temperature=1.5,
                prompt_version="v1.0",
                active_thinkers=("socrates",),
            )

    
    def test_weight_below_zero_rejected(self):
        with pytest.raises(ValueError):
            AgentConfig(
                socrates_weight=-0.1,
                feynman_weight=1.1,
                temperature=0.5,
                prompt_version="v1.0",
                active_thinkers=("socrates",),
            )

    
    def test_config_is_immutable(self, valid_config_file):
        config = AgentConfig.load(valid_config_file)
        with pytest.raises((AttributeError, TypeError)):
            config.temperature = 0.9

    
    def test_missing_config_file_raises_clearly(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            AgentConfig.load(missing)

    
    def test_malformed_json_raises_clearly(self, tmp_path):
        bad_file = tmp_path / "voices.json"
        bad_file.write_text("{ this is not valid json }")
        with pytest.raises(ValueError):
            AgentConfig.load(bad_file)
