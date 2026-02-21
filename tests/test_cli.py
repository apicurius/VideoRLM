"""Tests for the KUAVi CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys

import pytest


class TestCliImport:
    """Test that the CLI module can be imported."""

    def test_import_main(self):
        from kuavi.cli import main

        assert callable(main)

    def test_import_subcommands(self):
        from kuavi.cli import cmd_index, cmd_search, cmd_analyze

        assert callable(cmd_index)
        assert callable(cmd_search)
        assert callable(cmd_analyze)

    def test_import_helpers(self):
        from kuavi.cli import _build_analyze_prompt, _analyze_single_video

        assert callable(_build_analyze_prompt)
        assert callable(_analyze_single_video)


class TestCliHelp:
    """Test CLI help output."""

    def test_no_args_prints_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "kuavi.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        output = result.stdout + result.stderr
        assert "kuavi" in output or "usage" in output.lower()

    def test_index_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "kuavi.cli", "index", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "video" in result.stdout.lower()

    def test_search_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "kuavi.cli", "search", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "query" in result.stdout.lower()

    def test_analyze_help_shows_batch(self):
        result = subprocess.run(
            [sys.executable, "-m", "kuavi.cli", "analyze", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--batch" in result.stdout
        assert "--output-format" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--max-parallel" in result.stdout


class TestAnalyzeArgParsing:
    """Test argument parsing for the analyze subcommand."""

    @pytest.fixture()
    def parser(self):
        """Build the CLI parser."""
        p = argparse.ArgumentParser(prog="kuavi")
        subparsers = p.add_subparsers(dest="command")
        p_analyze = subparsers.add_parser("analyze")
        p_analyze.add_argument("video", nargs="?", default=None)
        p_analyze.add_argument("-q", "--question", required=True)
        p_analyze.add_argument("--batch", metavar="FILE")
        p_analyze.add_argument(
            "--output-format", choices=["text", "json"], default="text"
        )
        p_analyze.add_argument("--output-dir", metavar="DIR")
        p_analyze.add_argument("--max-parallel", type=int, default=1)
        return p

    def test_single_video_args(self, parser):
        args = parser.parse_args(["analyze", "video.mp4", "-q", "What happens?"])
        assert args.video == "video.mp4"
        assert args.question == "What happens?"
        assert args.batch is None
        assert args.output_format == "text"
        assert args.max_parallel == 1

    def test_batch_args(self, parser):
        args = parser.parse_args([
            "analyze", "--batch", "videos.txt", "-q", "What?",
            "--output-format", "json", "--output-dir", "/tmp/out", "--max-parallel", "4",
        ])
        assert args.video is None
        assert args.batch == "videos.txt"
        assert args.output_format == "json"
        assert args.output_dir == "/tmp/out"
        assert args.max_parallel == 4

    def test_batch_with_video_uses_video_arg(self, parser):
        # When both positional and --batch are given, both are set
        args = parser.parse_args([
            "analyze", "video.mp4", "--batch", "videos.txt", "-q", "What?",
        ])
        assert args.video == "video.mp4"
        assert args.batch == "videos.txt"

    def test_default_output_format(self, parser):
        args = parser.parse_args(["analyze", "v.mp4", "-q", "Q"])
        assert args.output_format == "text"


class TestAnalyzeBatch:
    """Test batch mode logic (without calling claude CLI)."""

    def test_no_video_no_batch_exits(self, capsys):
        from kuavi.cli import cmd_analyze

        args = argparse.Namespace(
            video=None, question="Q", batch=None,
            output_format="text", output_dir=None, max_parallel=1,
        )
        with pytest.raises(SystemExit, match="1"):
            cmd_analyze(args)
        captured = capsys.readouterr()
        assert "Provide a video path or --batch" in captured.out

    def test_batch_file_not_found_exits(self, capsys):
        from kuavi.cli import cmd_analyze

        args = argparse.Namespace(
            video=None, question="Q", batch="/nonexistent/batch.txt",
            output_format="text", output_dir=None, max_parallel=1,
        )
        with pytest.raises(SystemExit, match="1"):
            cmd_analyze(args)
        captured = capsys.readouterr()
        assert "Batch file not found" in captured.out

    def test_batch_missing_video_exits(self, tmp_path, capsys):
        from kuavi.cli import cmd_analyze

        batch_file = tmp_path / "batch.txt"
        batch_file.write_text("/nonexistent/video.mp4\n")
        args = argparse.Namespace(
            video=None, question="Q", batch=str(batch_file),
            output_format="text", output_dir=None, max_parallel=1,
        )
        with pytest.raises(SystemExit, match="1"):
            cmd_analyze(args)
        captured = capsys.readouterr()
        assert "Video file not found" in captured.out

    def test_batch_skips_comments_and_blanks(self, tmp_path, capsys, monkeypatch):
        from kuavi.cli import cmd_analyze

        # Create real video files (empty, just for path validation)
        v1 = tmp_path / "a.mp4"
        v1.touch()
        v2 = tmp_path / "b.mp4"
        v2.touch()

        batch_file = tmp_path / "batch.txt"
        batch_file.write_text(f"# comment\n\n{v1}\n{v2}\n\n# another\n")

        results = []

        def mock_analyze(video_path, question, output_format):
            results.append(video_path)
            return {"video": video_path, "returncode": 0, "stdout": "ok", "stderr": ""}

        monkeypatch.setattr("kuavi.cli._analyze_single_video", mock_analyze)

        args = argparse.Namespace(
            video=None, question="Q", batch=str(batch_file),
            output_format="text", output_dir=None, max_parallel=1,
        )
        # All succeed → no SystemExit (returns normally)
        cmd_analyze(args)

        # Should have processed exactly 2 videos (skipping comments and blanks)
        assert len(results) == 2

    def test_batch_writes_output_dir(self, tmp_path, monkeypatch):
        from kuavi.cli import cmd_analyze

        v1 = tmp_path / "clip.mp4"
        v1.touch()
        batch_file = tmp_path / "batch.txt"
        batch_file.write_text(str(v1) + "\n")
        out_dir = tmp_path / "results"

        def mock_analyze(video_path, question, output_format):
            return {"video": video_path, "returncode": 0, "stdout": "ok", "stderr": ""}

        monkeypatch.setattr("kuavi.cli._analyze_single_video", mock_analyze)

        args = argparse.Namespace(
            video=None, question="Q", batch=str(batch_file),
            output_format="json", output_dir=str(out_dir), max_parallel=1,
        )
        cmd_analyze(args)

        assert (out_dir / "clip.json").exists()

    def test_batch_parallel_execution(self, tmp_path, monkeypatch):
        from kuavi.cli import cmd_analyze

        videos = []
        for name in ["x.mp4", "y.mp4", "z.mp4"]:
            v = tmp_path / name
            v.touch()
            videos.append(str(v))

        batch_file = tmp_path / "batch.txt"
        batch_file.write_text("\n".join(videos) + "\n")

        analyzed = []

        def mock_analyze(video_path, question, output_format):
            analyzed.append(video_path)
            return {"video": video_path, "returncode": 0, "stdout": "done", "stderr": ""}

        monkeypatch.setattr("kuavi.cli._analyze_single_video", mock_analyze)

        args = argparse.Namespace(
            video=None, question="Q", batch=str(batch_file),
            output_format="text", output_dir=None, max_parallel=3,
        )
        cmd_analyze(args)
        assert len(analyzed) == 3

    def test_batch_failure_exits_nonzero(self, tmp_path, monkeypatch):
        from kuavi.cli import cmd_analyze

        v1 = tmp_path / "fail.mp4"
        v1.touch()
        batch_file = tmp_path / "batch.txt"
        batch_file.write_text(str(v1) + "\n")

        def mock_analyze(video_path, question, output_format):
            return {"video": video_path, "returncode": 1, "stdout": "", "stderr": "err"}

        monkeypatch.setattr("kuavi.cli._analyze_single_video", mock_analyze)

        args = argparse.Namespace(
            video=None, question="Q", batch=str(batch_file),
            output_format="text", output_dir=None, max_parallel=1,
        )
        with pytest.raises(SystemExit, match="1"):
            cmd_analyze(args)
