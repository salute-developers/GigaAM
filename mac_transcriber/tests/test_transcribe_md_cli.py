from pathlib import Path

from mac_transcriber.scripts import transcribe_md


def test_default_report_output_paths_follow_source_slug(tmp_path):
    source = tmp_path / "Planning Call.webm"
    output_dir = tmp_path / "out"

    assert (
        transcribe_md.default_report_output_path(source, output_dir, suffix=".md")
        == output_dir / "Planning_Call_report.md"
    )
    assert (
        transcribe_md.default_report_output_path(source, output_dir, suffix=".pdf")
        == output_dir / "Planning_Call_report.pdf"
    )
