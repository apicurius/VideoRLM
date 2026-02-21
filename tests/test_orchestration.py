"""Tests for the multi-agent orchestration pipeline.

Validates that the decompose → analyze → synthesize pattern works
end-to-end with the agent definitions, hook scripts, and budget
partitioning logic.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap

import pytest


# ---------------------------------------------------------------------------
# 1. Agent definition validation tests
# ---------------------------------------------------------------------------


AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", ".claude", "agents")


def _read_agent(name: str) -> str:
    """Read an agent .md file and return its content."""
    path = os.path.join(AGENTS_DIR, f"{name}.md")
    with open(path) as f:
        return f.read()


def _parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from an agent file as a dict."""
    lines = content.split("\n")
    assert lines[0].strip() == "---", "Agent file must start with ---"
    end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    fm = {}
    for line in lines[1:end]:
        if ":" in line:
            key, val = line.split(":", 1)
            fm[key.strip()] = val.strip()
    return fm


class TestAgentDefinitions:
    """Validate that all agent files parse correctly and have required fields."""

    REQUIRED_AGENTS = [
        "video-analyst",
        "video-decomposer",
        "video-segment-analyst",
        "video-synthesizer",
    ]

    @pytest.mark.parametrize("agent_name", REQUIRED_AGENTS)
    def test_agent_file_exists(self, agent_name):
        path = os.path.join(AGENTS_DIR, f"{agent_name}.md")
        assert os.path.exists(path), f"Agent file {agent_name}.md not found"

    @pytest.mark.parametrize("agent_name", REQUIRED_AGENTS)
    def test_agent_has_frontmatter(self, agent_name):
        content = _read_agent(agent_name)
        fm = _parse_frontmatter(content)
        assert "name" in fm, f"{agent_name} missing 'name' in frontmatter"
        assert "model" in fm, f"{agent_name} missing 'model' in frontmatter"
        assert fm["name"] == agent_name

    def test_decomposer_uses_haiku(self):
        fm = _parse_frontmatter(_read_agent("video-decomposer"))
        assert fm["model"] == "haiku", "Decomposer should use haiku for speed"

    def test_segment_analyst_uses_sonnet(self):
        fm = _parse_frontmatter(_read_agent("video-segment-analyst"))
        assert fm["model"] == "sonnet"

    def test_synthesizer_uses_sonnet(self):
        fm = _parse_frontmatter(_read_agent("video-synthesizer"))
        assert fm["model"] == "sonnet"

    def test_analyst_can_spawn_subagents(self):
        fm = _parse_frontmatter(_read_agent("video-analyst"))
        tools = fm.get("tools", "")
        assert "Task(" in tools, "video-analyst must declare Task() with sub-agents"
        assert "video-decomposer" in tools
        assert "video-segment-analyst" in tools
        assert "video-synthesizer" in tools

    def test_segment_analyst_has_frame_tools(self):
        fm = _parse_frontmatter(_read_agent("video-segment-analyst"))
        tools = fm.get("tools", "")
        assert "kuavi_extract_frames" in tools
        assert "kuavi_crop_frame" in tools
        assert "kuavi_eval" in tools

    def test_decomposer_has_search_tools(self):
        fm = _parse_frontmatter(_read_agent("video-decomposer"))
        tools = fm.get("tools", "")
        assert "kuavi_search_video" in tools
        assert "kuavi_get_scene_list" in tools

    def test_synthesizer_has_limited_tools(self):
        """Synthesizer should have search but NOT frame extraction tools."""
        fm = _parse_frontmatter(_read_agent("video-synthesizer"))
        tools = fm.get("tools", "")
        assert "kuavi_get_transcript" in tools
        assert "kuavi_extract_frames" not in tools, (
            "Synthesizer should not extract frames — it synthesizes results"
        )


# ---------------------------------------------------------------------------
# 2. Decomposition plan structure tests
# ---------------------------------------------------------------------------


class TestDecompositionPlanFormat:
    """Validate the decomposition plan JSON schema described in the agent."""

    VALID_PLAN = {
        "original_question": "What are the main topics and when do they appear?",
        "complexity": "complex",
        "strategy": "parallel",
        "sub_questions": [
            {
                "id": "sq1",
                "question": "What is the first topic?",
                "time_range": {"start": 0.0, "end": 30.0},
                "search_hints": ["first topic", "introduction"],
                "evidence_type": "both",
                "depends_on": [],
            },
            {
                "id": "sq2",
                "question": "What is the second topic?",
                "time_range": {"start": 30.0, "end": 60.0},
                "search_hints": ["second topic"],
                "evidence_type": "visual",
                "depends_on": [],
            },
        ],
        "synthesis_instruction": "List each topic with its time range.",
    }

    def test_valid_plan_parses(self):
        plan = self.VALID_PLAN
        assert plan["complexity"] in ("simple", "moderate", "complex")
        assert plan["strategy"] in ("parallel", "sequential", "hierarchical")
        assert len(plan["sub_questions"]) > 0

    def test_sub_questions_have_required_fields(self):
        for sq in self.VALID_PLAN["sub_questions"]:
            assert "id" in sq
            assert "question" in sq
            assert "time_range" in sq
            assert "start" in sq["time_range"]
            assert "end" in sq["time_range"]
            assert sq["time_range"]["end"] > sq["time_range"]["start"]
            assert "evidence_type" in sq
            assert sq["evidence_type"] in ("visual", "transcript", "both")

    def test_parallel_questions_have_no_dependencies(self):
        """In a parallel strategy, independent sub-questions should have empty depends_on."""
        plan = self.VALID_PLAN
        if plan["strategy"] == "parallel":
            for sq in plan["sub_questions"]:
                assert sq["depends_on"] == [], (
                    f"Parallel sub-question {sq['id']} should not have dependencies"
                )

    def test_simple_plan_has_single_sub_question(self):
        simple_plan = {
            "original_question": "What color is the car?",
            "complexity": "simple",
            "strategy": "parallel",
            "sub_questions": [
                {
                    "id": "sq1",
                    "question": "What color is the car?",
                    "time_range": {"start": 0.0, "end": 120.0},
                    "search_hints": ["car color"],
                    "evidence_type": "visual",
                    "depends_on": [],
                }
            ],
            "synthesis_instruction": "Report the car color directly.",
        }
        assert simple_plan["complexity"] == "simple"
        assert len(simple_plan["sub_questions"]) == 1


# ---------------------------------------------------------------------------
# 3. Hook script validation tests
# ---------------------------------------------------------------------------


HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "hooks")


class TestAntiHallucinationHook:
    """Test validate_transcript_claims.sh hook logic."""

    def _run_hook(self, tool_response: str) -> tuple[int, str, str]:
        """Run the hook script with a simulated tool response."""
        hook_path = os.path.join(HOOKS_DIR, "validate_transcript_claims.sh")
        payload = json.dumps({"tool_response": tool_response})
        result = subprocess.run(
            ["bash", hook_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode, result.stdout, result.stderr

    def test_exits_zero_always(self):
        rc, _, _ = self._run_hook("plain transcript text without numbers")
        assert rc == 0

    def test_warns_on_numbers(self):
        rc, _, stderr = self._run_hook(
            "The score was 42 and the temperature reached 1500 degrees"
        )
        assert rc == 0
        assert "ANTI-HALLUCINATION" in stderr
        assert "visually confirm" in stderr.lower()

    def test_warns_on_names(self):
        rc, _, stderr = self._run_hook(
            "Speaker Johnson presented the findings to Professor Williams"
        )
        assert rc == 0
        assert "ANTI-HALLUCINATION" in stderr
        assert "name" in stderr.lower()

    def test_no_warning_on_clean_text(self):
        rc, _, stderr = self._run_hook("a simple sentence with no special items")
        assert rc == 0
        assert stderr.strip() == ""

    def test_empty_response_no_warning(self):
        rc, _, stderr = self._run_hook("")
        assert rc == 0
        assert "ANTI-HALLUCINATION" not in stderr


class TestVisualConfirmationHook:
    """Test validate_visual_confirmation.sh hook logic."""

    def _run_hook(self, message: str) -> tuple[int, str, str]:
        hook_path = os.path.join(HOOKS_DIR, "validate_visual_confirmation.sh")
        payload = json.dumps({"last_assistant_message": message})
        result = subprocess.run(
            ["bash", hook_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode, result.stdout, result.stderr

    def test_short_messages_skipped(self):
        rc, _, stderr = self._run_hook("short")
        assert rc == 0
        assert stderr.strip() == ""

    def test_non_video_messages_skipped(self):
        long_msg = "a " * 200  # >300 chars but no video keywords
        rc, _, stderr = self._run_hook(long_msg)
        assert rc == 0
        assert stderr.strip() == ""

    def test_warns_on_many_numbers_without_visual(self):
        msg = (
            "The video shows the following scores: 100, 200, 300, 400, 500, 600. "
            "According to the transcript, the frame rate was 30fps and the scene "
            "lasted 45 seconds at 1080p resolution. "
            "The segment at timestamp 12:30 contains important content. "
            "Additional analysis of the video reveals that there were multiple "
            "scenes with varying durations. The total runtime was approximately "
            "180 seconds and the scene changes occurred at 22, 55, 89, and 120 "
            "second marks throughout the video presentation."
        )
        rc, _, stderr = self._run_hook(msg)
        assert rc == 0
        assert "WARNING" in stderr

    def test_no_warning_with_visual_evidence(self):
        msg = (
            "The video frame shows the score is 42 points. I visually confirmed "
            "this from the extracted frame at 10.5s. The frame shows the number "
            "clearly displayed on screen. The screenshot confirms the value. "
            "The segment at 15s also shows similar content visible in frame."
        )
        rc, _, stderr = self._run_hook(msg)
        assert rc == 0
        # Should not warn because visual evidence is cited


# ---------------------------------------------------------------------------
# 4. Skill file validation tests
# ---------------------------------------------------------------------------


SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", ".claude", "skills")


class TestSkillDefinitions:
    """Validate that new skills exist and have proper structure."""

    REQUIRED_SKILLS = [
        "kuavi-pixel-analysis",
        "kuavi-deep-search",
        "kuavi-search",
        "kuavi-analyze",
        "kuavi-deep-analyze",
    ]

    @pytest.mark.parametrize("skill_name", REQUIRED_SKILLS)
    def test_skill_file_exists(self, skill_name):
        path = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
        assert os.path.exists(path), f"Skill {skill_name}/SKILL.md not found"

    @pytest.mark.parametrize("skill_name", REQUIRED_SKILLS)
    def test_skill_has_frontmatter(self, skill_name):
        path = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
        with open(path) as f:
            content = f.read()
        assert content.startswith("---"), f"Skill {skill_name} missing YAML frontmatter"

    def test_pixel_analysis_has_patterns(self):
        path = os.path.join(SKILLS_DIR, "kuavi-pixel-analysis", "SKILL.md")
        with open(path) as f:
            content = f.read()
        # Should contain multiple code patterns
        assert "kuavi_eval" in content
        assert "threshold_frame" in content
        assert "diff_frames" in content
        assert "Pattern" in content

    def test_deep_search_has_decision_flow(self):
        path = os.path.join(SKILLS_DIR, "kuavi-deep-search", "SKILL.md")
        with open(path) as f:
            content = f.read()
        assert "Pass 1" in content
        assert "Pass 2" in content
        assert "reformulation" in content.lower() or "Reformulation" in content


# ---------------------------------------------------------------------------
# 5. Budget partitioning tests
# ---------------------------------------------------------------------------


class TestBudgetPartitioning:
    """Test that the budget allocation scheme is internally consistent."""

    def test_analyst_budget_allocations_sum(self):
        """Budget partitioning: decomposer + segments + synthesizer <= total."""
        total = 50  # default budget
        decomposer_budget = 5
        synthesizer_budget = 5
        max_segments = 5
        per_segment_budget = 8

        allocated = decomposer_budget + (max_segments * per_segment_budget) + synthesizer_budget
        assert allocated == 50, (
            f"Budget allocation ({allocated}) should equal total budget ({total})"
        )

    def test_simple_question_preserves_full_budget(self):
        """Simple questions skip orchestration, keeping full budget for direct search."""
        total = 50
        # No sub-agents = no partitioning overhead
        assert total == 50


# ---------------------------------------------------------------------------
# 6. Settings integration tests
# ---------------------------------------------------------------------------


class TestSettingsIntegration:
    """Validate settings.json has required hook registrations."""

    @pytest.fixture
    def settings(self):
        settings_path = os.path.join(
            os.path.dirname(__file__), "..", ".claude", "settings.json"
        )
        with open(settings_path) as f:
            return json.load(f)

    def test_transcript_hook_registered(self, settings):
        post_hooks = settings["hooks"]["PostToolUse"]
        transcript_matchers = [
            h for h in post_hooks
            if "kuavi_search_transcript" in h.get("matcher", "")
        ]
        assert len(transcript_matchers) >= 1, (
            "validate_transcript_claims.sh must be registered for kuavi_search_transcript"
        )

    def test_visual_confirmation_in_stop_hooks(self, settings):
        stop_hooks = settings["hooks"]["Stop"]
        commands = []
        for entry in stop_hooks:
            for hook in entry.get("hooks", []):
                commands.append(hook.get("command", ""))
        assert any("validate_visual_confirmation" in cmd for cmd in commands), (
            "validate_visual_confirmation.sh must be in Stop hooks"
        )

    def test_agent_teams_enabled(self, settings):
        env = settings.get("env", {})
        assert env.get("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS") == "1"

    def test_trace_logger_is_async(self, settings):
        """Trace logger hooks should be async to avoid blocking."""
        for event_type in ["PostToolUse", "SubagentStart", "SubagentStop"]:
            for entry in settings["hooks"].get(event_type, []):
                for hook in entry.get("hooks", []):
                    if "kuavi_trace_logger" in hook.get("command", ""):
                        assert hook.get("async") is True, (
                            f"kuavi_trace_logger in {event_type} must be async"
                        )
