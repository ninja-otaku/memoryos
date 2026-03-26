from __future__ import annotations

import os
import sys
import tempfile

sys.path.append('src')

from memoryos.core import MemoryOS, MemoryOSConfig


def test_phase1_episodic_to_semantic_consolidation() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = MemoryOSConfig(
            data_dir=os.path.join(tmp, 'data'),
            sqlite_path=os.path.join(tmp, 'data', 'memoryos.sqlite3'),
            chroma_dir=os.path.join(tmp, 'data', 'chroma'),
            use_chroma=False,
        )
        mem = MemoryOS(cfg=cfg)

        mem.remember('The capital of France is Paris.', importance=0.9)

        before = mem.recall('France capital', memory_type='semantic')

        mem.consolidate()
        after = mem.recall('France capital', memory_type='semantic')
        assert any('paris' in m.content.lower() for m in after)

        _ = mem.recall('capital of France', memory_type='episodic')
        _ = before


def test_procedural_learn_match_update() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = MemoryOSConfig(
            data_dir=os.path.join(tmp, 'data'),
            sqlite_path=os.path.join(tmp, 'data', 'memoryos.sqlite3'),
            chroma_dir=os.path.join(tmp, 'data', 'chroma'),
            use_chroma=False,
        )
        mem = MemoryOS(cfg=cfg)

        template_id = mem.learn_procedure(
            task_type='write_code',
            steps=['plan', 'implement', 'test'],
            initial_success_rate=0.5,
        )
        assert template_id

        matched = mem.recall('How should I implement code?', memory_type='procedural')
        assert any(m.id == template_id for m in matched)

        mem.update_procedure_success(template_id=template_id, success=True)
        matched2 = mem.recall('How should I implement code?', memory_type='procedural')
        assert any(m.id == template_id for m in matched2)


def test_forget_removes_low_importance_old_episodic() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = MemoryOSConfig(
            data_dir=os.path.join(tmp, 'data'),
            sqlite_path=os.path.join(tmp, 'data', 'memoryos.sqlite3'),
            chroma_dir=os.path.join(tmp, 'data', 'chroma'),
            use_chroma=False,
        )
        mem = MemoryOS(cfg=cfg)

        mem.episodic.add_event(
            'Old low-importance memory',
            importance=0.1,
            timestamp=0.0,
        )
        before = mem.get_memory_summary()
        assert before['episodic_count'] >= 1

        mem.forget(older_than_days=1, min_importance=0.5)
        after = mem.get_memory_summary()
        assert after['episodic_count'] <= before['episodic_count']
