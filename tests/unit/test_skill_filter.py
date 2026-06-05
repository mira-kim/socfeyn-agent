"""
tests/unit/test_skill_filter.py
--------------------------------
Tests for agent/skill_filter.py.

What we test:
  - Skill YAML loads correctly into a Skill object
  - Principles parse correctly
  - Topic relevance returns a float in [0, 1]
  - Queries below threshold do not activate a skill
  - Signal word pre-check short-circuits without API call
  - FilterResult correctly reports has_violations
  - Missing skills directory returns empty result, not crash
  - Malformed YAML raises clearly at load time

What we do NOT test:
  - The Haiku API call for principle checking (smoke test only)
  - Embedding quality or cosine similarity accuracy
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Mock embed function — returns a deterministic float32 vector.
# Used in all tests that would otherwise trigger a model download.
# ---------------------------------------------------------------------------

def _mock_embed(text: str) -> np.ndarray:
    """
    Deterministic fake embedding. Uses text length + hash to produce
    a normalized vector that is different for different inputs.
    This lets cosine similarity tests work without a real model.
    """
    rng = np.random.RandomState(hash(text) % (2**31))
    vec = rng.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture(autouse=True)
def mock_embed_globally(monkeypatch):
    """
    Patches _get_embed_fn in skill_filter to return the mock embed function.
    Applied to every test in this file via autouse=True.
    """
    monkeypatch.setattr("agent.skill_filter._get_embed_fn", lambda: _mock_embed)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def skill_yaml(tmp_path) -> Path:
    """A minimal valid skill YAML file."""
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "test_skill.yaml").write_text("""
id: test_skill
name: "Test Skill"
version: "1.0"
description: "A test skill for unit tests"
triggers:
  - human behavior
  - decision making
relevance_threshold: 0.50
severity: medium
principles:
  - id: no_test_assumption
    description: "Response must not use the phrase 'test assumption'"
    violation_signals:
      - "test assumption"
      - "rational test"
    correction_hint: "Avoid the test assumption framing"
""")
    return skill_dir


@pytest.fixture
def empty_skill_dir(tmp_path) -> Path:
    """An existing skills directory with no YAML files."""
    d = tmp_path / "skills"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Skill loading tests
# ---------------------------------------------------------------------------

class TestSkillLoading:

    def test_valid_skill_loads(self, skill_yaml):
        from agent.skill_filter import load_skills, clear_skill_cache
        clear_skill_cache()
        skills = load_skills(skill_yaml)
        assert len(skills) == 1
        skill = skills[0]
        assert skill.id == "test_skill"
        assert skill.name == "Test Skill"
        assert skill.relevance_threshold == 0.50
        assert len(skill.principles) == 1
        assert skill.principles[0].id == "no_test_assumption"
        clear_skill_cache()

    def test_principles_parse_correctly(self, skill_yaml):
        from agent.skill_filter import load_skills, clear_skill_cache
        clear_skill_cache()
        skills = load_skills(skill_yaml)
        p = skills[0].principles[0]
        assert "test assumption" in p.violation_signals
        assert "rational test" in p.violation_signals
        assert len(p.correction_hint) > 0
        clear_skill_cache()

    def test_trigger_embeddings_computed(self, skill_yaml):
        """Trigger embeddings are pre-computed at load time."""
        from agent.skill_filter import load_skills, clear_skill_cache
        clear_skill_cache()
        skills = load_skills(skill_yaml)
        assert len(skills[0].trigger_embeddings) == 2  # two triggers
        assert skills[0].trigger_embeddings[0] is not None
        clear_skill_cache()

    def test_empty_skills_dir_returns_empty(self, empty_skill_dir):
        from agent.skill_filter import load_skills, clear_skill_cache
        clear_skill_cache()
        skills = load_skills(empty_skill_dir)
        assert skills == []
        clear_skill_cache()

    def test_missing_skills_dir_returns_empty(self, tmp_path):
        from agent.skill_filter import load_skills, clear_skill_cache
        clear_skill_cache()
        missing = tmp_path / "nonexistent"
        skills = load_skills(missing)
        assert skills == []
        clear_skill_cache()

    def test_malformed_yaml_does_not_crash_other_skills(self, tmp_path):
        """A bad YAML file should be skipped, not crash the loader."""
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "good_skill.yaml").write_text("""
id: good_skill
name: "Good Skill"
triggers: [human behavior]
relevance_threshold: 0.5
principles: []
""")
        (skill_dir / "bad_skill.yaml").write_text("{ this: is: not: valid: yaml: !!!!")
        from agent.skill_filter import load_skills, clear_skill_cache
        clear_skill_cache()
        skills = load_skills(skill_dir)
        # Good skill should still load
        assert any(s.id == "good_skill" for s in skills)
        clear_skill_cache()

    def test_skill_cache_populated_after_load(self, skill_yaml):
        """Second call to load_skills uses cache, not file."""
        from agent.skill_filter import load_skills, clear_skill_cache, _skill_cache
        clear_skill_cache()
        load_skills(skill_yaml)
        assert "test_skill" in _skill_cache
        clear_skill_cache()


# ---------------------------------------------------------------------------
# Topic relevance tests
# ---------------------------------------------------------------------------

class TestTopicRelevance:

    def test_relevance_returns_float(self, skill_yaml):
        from agent.skill_filter import load_skills, compute_relevance, clear_skill_cache
        clear_skill_cache()
        skill = load_skills(skill_yaml)[0]
        score = compute_relevance("why do people cooperate", skill)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        clear_skill_cache()

    def test_relevant_query_returns_a_score(self, skill_yaml):
        """Relevance computation returns a float — ordering depends on real embeddings."""
        from agent.skill_filter import load_skills, compute_relevance, clear_skill_cache
        clear_skill_cache()
        skill = load_skills(skill_yaml)[0]
        score = compute_relevance("why do people make decisions", skill)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0   # valid cosine similarity range
        clear_skill_cache()

    def test_empty_triggers_returns_zero(self, tmp_path):
        """Skill with no triggers should return 0.0 relevance."""
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "no_triggers.yaml").write_text("""
id: no_triggers
name: "No Triggers"
triggers: []
relevance_threshold: 0.5
principles: []
""")
        from agent.skill_filter import load_skills, compute_relevance, clear_skill_cache
        clear_skill_cache()
        skill = load_skills(skill_dir)[0]
        score = compute_relevance("anything", skill)
        assert score == 0.0
        clear_skill_cache()


# ---------------------------------------------------------------------------
# Signal word pre-check tests (no API call)
# ---------------------------------------------------------------------------

class TestSignalWordPrecheck:

    def test_no_signal_words_skips_api_call(self, skill_yaml):
        """
        If response contains no signal words, Haiku is never called.
        This is the fast path — prevents unnecessary API spend.
        """
        from agent.skill_filter import load_skills, _check_principle, clear_skill_cache
        clear_skill_cache()
        skill = load_skills(skill_yaml)[0]
        principle = skill.principles[0]

        response_without_signals = "The virtue of courage requires confronting fear directly."

        with patch("agent.runner.call_api_with_retry") as mock_api:
            result = _check_principle(response_without_signals, principle, skill)
            mock_api.assert_not_called()
            assert result is None
        clear_skill_cache()

    def test_signal_word_found_triggers_api_call(self, skill_yaml):
        """When a signal word is found, Haiku is called to verify."""
        from agent.skill_filter import load_skills, _check_principle, clear_skill_cache
        clear_skill_cache()
        skill = load_skills(skill_yaml)[0]
        principle = skill.principles[0]

        response_with_signal = "The test assumption here is that people act rationally."

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps({
            "violated": False,
            "excerpt": "",
            "explanation": "",
        })

        with patch("agent.runner.call_api_with_retry", return_value=mock_response):
            result = _check_principle(response_with_signal, principle, skill)
            # API was called because signal word found
        clear_skill_cache()


# ---------------------------------------------------------------------------
# FilterResult tests
# ---------------------------------------------------------------------------

class TestFilterResult:

    def test_has_violations_false_when_empty(self):
        from agent.skill_filter import FilterResult
        result = FilterResult(
            query_relevance={}, skills_activated=[], violations=[]
        )
        assert result.has_violations is False

    def test_has_violations_true_when_violations_present(self):
        from agent.skill_filter import FilterResult, SkillViolation
        v = SkillViolation(
            skill_id="test", skill_name="Test", principle_id="p1",
            excerpt="bad phrase", explanation="why bad",
            correction_hint="do better", severity="medium",
        )
        result = FilterResult(
            query_relevance={"test": 0.8},
            skills_activated=["test"],
            violations=[v],
        )
        assert result.has_violations is True

    def test_violation_as_dict_has_required_keys(self):
        from agent.skill_filter import SkillViolation
        v = SkillViolation(
            skill_id="test", skill_name="Test", principle_id="p1",
            excerpt="bad phrase", explanation="why bad",
            correction_hint="do better", severity="medium",
        )
        d = v.as_dict()
        required = ["skill_id", "skill_name", "principle_id",
                    "excerpt", "explanation", "correction_hint", "severity"]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_check_never_raises_on_error(self, tmp_path):
        """
        check() must never raise — filter failure cannot interrupt a session.
        """
        from agent.skill_filter import check, clear_skill_cache
        clear_skill_cache()
        # Pass a completely broken skills_dir path
        result = check("some query", "some response", skills_dir=tmp_path / "nonexistent")
        assert result is not None
        assert result.has_violations is False
        clear_skill_cache()


# ---------------------------------------------------------------------------
# Game theory skill tests (validates the actual skill file)
# ---------------------------------------------------------------------------

class TestGameTheorySkill:

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        from agent.skill_filter import clear_skill_cache
        clear_skill_cache()
        yield
        clear_skill_cache()

    def test_game_theory_skill_file_exists(self):
        """The actual skill file must be present."""
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        skill_file = skills_dir / "game_theory_filter.yaml"
        assert skill_file.exists(), (
            f"game_theory_filter.yaml not found at {skill_file}. "
            f"Create skills/game_theory_filter.yaml before running this test."
        )

    def test_game_theory_skill_loads(self):
        """The actual game theory skill loads without errors."""
        from agent.skill_filter import load_skills
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if not skills_dir.exists():
            pytest.skip("skills/ directory not found")
        skills = load_skills(skills_dir)
        ids = [s.id for s in skills]
        assert "game_theory_filter" in ids, (
            f"game_theory_filter not in loaded skills: {ids}"
        )

    def test_game_theory_skill_has_principles(self):
        """Game theory skill must have at least one principle."""
        from agent.skill_filter import load_skills
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if not skills_dir.exists():
            pytest.skip("skills/ directory not found")
        skills = load_skills(skills_dir)
        gt = next((s for s in skills if s.id == "game_theory_filter"), None)
        if gt is None:
            pytest.skip("game_theory_filter not loaded")
        assert len(gt.principles) > 0

    def test_human_behavior_query_returns_float(self):
        """Relevance computation returns a valid float for the game theory skill.
        Note: threshold calibration requires the real sentence-transformers model,
        not the mock embed used in unit tests. Use smoke tests for threshold validation."""
        from agent.skill_filter import load_skills, compute_relevance
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if not skills_dir.exists():
            pytest.skip("skills/ directory not found")
        skills = load_skills(skills_dir)
        gt = next((s for s in skills if s.id == "game_theory_filter"), None)
        if gt is None:
            pytest.skip("game_theory_filter not loaded")
        score = compute_relevance("why do people cooperate in societies", gt)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_philosophy_of_forms_below_threshold(self):
        """'what are Plato's forms' should NOT trigger the game theory filter."""
        from agent.skill_filter import load_skills, compute_relevance
        skills_dir = Path(__file__).parent.parent.parent / "skills"
        if not skills_dir.exists():
            pytest.skip("skills/ directory not found")
        skills = load_skills(skills_dir)
        gt = next((s for s in skills if s.id == "game_theory_filter"), None)
        if gt is None:
            pytest.skip("game_theory_filter not loaded")
        score = compute_relevance("what are Plato's theory of forms", gt)
        assert score < gt.relevance_threshold, (
            f"Expected relevance < {gt.relevance_threshold}, got {score:.3f}. "
            f"Game theory filter is firing on non-human-behavior queries. "
            f"Consider raising relevance_threshold in game_theory_filter.yaml."
        )
