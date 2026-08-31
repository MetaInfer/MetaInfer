from __future__ import annotations

import pytest

from ..orchestrator.config import load_config
from ..orchestrator.skill_store import (
    generate_merged_skill,
    generate_worker_skill,
    list_skill_library,
    publish_skill,
)


def test_skill_library_lists_and_publishes_without_overwrite(
    tmp_path, monkeypatch
):
    existing_root = tmp_path / "claude-skills"
    existing = existing_root / "already-here"
    existing.mkdir(parents=True)
    (existing / "SKILL.md").write_text(
        "---\nname: already-here\ndescription: Existing test skill.\n---\n"
    )
    workspace = tmp_path / "workspace"
    pending = workspace / "skills" / "pending" / "new-kernel-skill"
    pending.mkdir(parents=True)
    (pending / "SKILL.md").write_text(
        "---\nname: new-kernel-skill\ndescription: New test skill.\n---\n"
    )
    (pending / "manifest.json").write_text(
        '{"kind":"merged","source":"main_agent"}'
    )
    monkeypatch.setenv(
        "METAINFER_CLAUDE_SKILLS_DIR", str(existing_root)
    )

    library = list_skill_library(workspace)
    assert [item["name"] for item in library["existing"]] == ["already-here"]
    assert [item["name"] for item in library["pending"]] == ["new-kernel-skill"]

    result = publish_skill(workspace, "new-kernel-skill")
    assert result["status"] == "existing"
    assert (existing_root / "new-kernel-skill" / "SKILL.md").is_file()
    assert not (existing_root / "new-kernel-skill" / "manifest.json").exists()
    assert not pending.exists()

    with pytest.raises(FileNotFoundError):
        publish_skill(workspace, "new-kernel-skill")


def test_agent_authored_skill_drafts_keep_managed_frontmatter(tmp_path):
    req = {
        "task_id": "skill-test",
        "answers": {
            "execution_mode": "Generate & optimize (auto-create kernel repo)",
            "operator": "Quantized GEMM",
            "dtype": "INT8 W8A8",
            "target_hardware": "gfx928",
            "kernel_language": "HIP C++",
            "shape_config": """
shapes:
  - {id: m2, M: 2, N: 16, K: 32}
assignments:
  worker_0: {gpu: 0, shapes: [m2]}
""",
        },
    }
    config = load_config(req)
    assignment = config.assignments[0]
    worker = generate_worker_skill(
        config=config,
        assignment=assignment,
        workspace_dir=tmp_path,
        agent_draft="# Measured worker evidence\n\nOnly measured facts.",
    )
    merged = generate_merged_skill(
        config=config,
        assignments=[assignment],
        workspace_dir=tmp_path,
        agent_draft="# Main synthesis\n\nRoute m2 to worker_0.",
    )

    worker_text = (
        tmp_path / "skills" / "pending" / worker["name"] / "SKILL.md"
    ).read_text(encoding="utf-8")
    merged_text = (
        tmp_path / "skills" / "pending" / merged["name"] / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert worker_text.startswith("---\nname:")
    assert "# Measured worker evidence" in worker_text
    assert merged_text.startswith("---\nname:")
    assert "# Main synthesis" in merged_text
