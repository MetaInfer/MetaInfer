from __future__ import annotations

import pytest

from ..orchestrator.config import load_config
from ..orchestrator.skill_store import (
    _apply_fuse_decision,
    _parse_fuse_decision,
    dsh_skills_root,
    generate_merged_skill,
    generate_worker_skill,
    list_skill_library,
    publish_skill,
    rollback_skill,
    sync_skill_libraries,
)


def test_skill_library_lists_and_publishes_without_overwrite(
    tmp_path, monkeypatch
):
    dsh_root = tmp_path / "dsh-skills"
    ccb_root = tmp_path / "claude-skills"
    existing = dsh_root / "already-here"
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
    monkeypatch.setenv("DSH_SKILLS_DIR", str(dsh_root))
    monkeypatch.setenv("METAINFER_CLAUDE_SKILLS_DIR", str(ccb_root))

    library = list_skill_library(workspace)
    assert [item["name"] for item in library["existing"]] == ["already-here"]
    assert [item["name"] for item in library["pending"]] == ["new-kernel-skill"]

    result = publish_skill(workspace, "new-kernel-skill")
    assert result["status"] == "existing"
    assert (dsh_root / "new-kernel-skill" / "SKILL.md").is_file()
    assert not (dsh_root / "new-kernel-skill" / "manifest.json").exists()
    assert not pending.exists()
    # publish mirrors the dsh library into the ccb library automatically.
    assert (ccb_root / "new-kernel-skill" / "SKILL.md").is_file()

    with pytest.raises(FileNotFoundError):
        publish_skill(workspace, "new-kernel-skill")


def test_sync_skill_libraries_mirrors_dsh_to_ccb(tmp_path, monkeypatch):
    dsh_root = tmp_path / "dsh"
    ccb_root = tmp_path / "ccb"
    (dsh_root / "alpha").mkdir(parents=True)
    (dsh_root / "alpha" / "SKILL.md").write_text("# alpha v2\n", encoding="utf-8")
    (dsh_root / "beta").mkdir(parents=True)
    (dsh_root / "beta" / "SKILL.md").write_text("# beta\n", encoding="utf-8")
    (ccb_root / "alpha").mkdir(parents=True)
    (ccb_root / "alpha" / "SKILL.md").write_text("# alpha v1\n", encoding="utf-8")
    (ccb_root / "ccb-only").mkdir(parents=True)
    (ccb_root / "ccb-only" / "SKILL.md").write_text("# ccb only\n", encoding="utf-8")
    monkeypatch.setenv("DSH_SKILLS_DIR", str(dsh_root))
    monkeypatch.setenv("METAINFER_CLAUDE_SKILLS_DIR", str(ccb_root))

    summary = sync_skill_libraries(workspace_dir=tmp_path / "ws")

    assert summary["added"] == ["beta"]
    assert summary["updated"] == ["alpha"]
    assert summary["ccb_only"] == ["ccb-only"]
    assert (ccb_root / "beta" / "SKILL.md").read_text() == "# beta\n"
    assert (ccb_root / "alpha" / "SKILL.md").read_text() == "# alpha v2\n"
    # the overwritten ccb skill was backed up
    assert len(list((ccb_root / "alpha").glob("SKILL.md.bak-*"))) == 1
    # idempotent
    summary2 = sync_skill_libraries(workspace_dir=tmp_path / "ws")
    assert summary2["added"] == []
    assert summary2["updated"] == []
    assert summary2["skipped"] == ["alpha", "beta"]


def test_rollback_skill_restores_backup(tmp_path, monkeypatch):
    dsh_root = tmp_path / "dsh"
    ccb_root = tmp_path / "ccb"
    (dsh_root / "alpha").mkdir(parents=True)
    (dsh_root / "alpha" / "SKILL.md").write_text("# v2\n", encoding="utf-8")
    monkeypatch.setenv("DSH_SKILLS_DIR", str(dsh_root))
    monkeypatch.setenv("METAINFER_CLAUDE_SKILLS_DIR", str(ccb_root))
    (dsh_root / "alpha" / "SKILL.md.bak-1").write_text("# v1\n", encoding="utf-8")
    sync_skill_libraries(workspace_dir=tmp_path / "ws")

    result = rollback_skill("alpha", workspace_dir=tmp_path / "ws")
    assert result["restored_from"].endswith("SKILL.md.bak-1")
    assert (dsh_root / "alpha" / "SKILL.md").read_text() == "# v1\n"
    assert (ccb_root / "alpha" / "SKILL.md").read_text() == "# v1\n"

    with pytest.raises(FileNotFoundError):
        rollback_skill("alpha", workspace_dir=tmp_path / "ws")


def test_fuse_decision_parse_and_apply(tmp_path, monkeypatch):
    dsh_root = tmp_path / "dsh"
    ccb_root = tmp_path / "ccb"
    (dsh_root / "existing-skill").mkdir(parents=True)
    (dsh_root / "existing-skill" / "SKILL.md").write_text(
        "---\nname: existing-skill\ndescription: Old scope.\n---\n# old body\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DSH_SKILLS_DIR", str(dsh_root))
    monkeypatch.setenv("METAINFER_CLAUDE_SKILLS_DIR", str(ccb_root))

    # a "new" decision
    decision = _parse_fuse_decision(
        '{"action": "new", "name": "my-new-skill", '
        '"description": "New scope", "content": "# new body"}'
    )
    assert decision["action"] == "new"
    applied = _apply_fuse_decision(decision)
    assert applied["name"] == "my-new-skill"
    text = (dsh_root / "my-new-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: my-new-skill\n")
    assert "# new body" in text

    # a "merge" decision backs up + replaces the body, keeping frontmatter
    merge = _parse_fuse_decision(
        '{"action": "merge", "name": "existing-skill", '
        '"content": "# merged body"}'
    )
    applied = _apply_fuse_decision(merge)
    assert applied["action"] == "merge"
    assert applied["backup"] is not None
    assert "old body" in applied["diff"]
    text = (dsh_root / "existing-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: existing-skill\n")
    assert "# merged body" in text

    # fenced JSON also parses
    fenced = _parse_fuse_decision(
        '```json\n{"action": "new", "name": "fenced-skill", "content": "x"}\n```'
    )
    assert fenced["name"] == "fenced-skill"

    with pytest.raises(ValueError):
        _parse_fuse_decision("the agent wrote prose instead of JSON")


def test_dsh_skills_root_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DSH_SKILLS_DIR", str(tmp_path / "x"))
    assert dsh_skills_root() == tmp_path / "x"


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
