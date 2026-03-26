from __future__ import annotations

import math
import os
import tempfile
import time
from datetime import timedelta

import matplotlib.pyplot as plt

from memoryos.core import MemoryOS, MemoryOSConfig


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = MemoryOSConfig(
            data_dir=os.path.join(tmp, 'data'),
            sqlite_path=os.path.join(tmp, 'data', 'memoryos.sqlite3'),
            chroma_dir=os.path.join(tmp, 'data', 'chroma'),
        )
        mem = MemoryOS(cfg=cfg)

        # Simulate 20 conversation turns (each 12 hours apart).
        now = time.time()
        for i in range(20):
            ts = now - (12.0 * 3600.0) * i
            mem.episodic.add_event(
                f'Turn {i}: user preferences updated (opt {i % 3}).',
                importance=0.6,
                timestamp=ts,
            )

        # Measure how many episodic memories would still be retained.
        time_points_hours = [i for i in range(0, 24 * 10, 12)]
        retained_counts: list[int] = []
        threshold = mem.cfg.episodic_decay_threshold

        for h in time_points_hours:
            cutoff = now + (h * 3600.0)
            kept = 0
            for e in mem.episodic.store.list_all():
                t_seconds = max(0.0, cutoff - float(e['timestamp']))
                stability_seconds = float(e['decay_factor']) * (
                    1.0 + float(e['rehearsal_count']) * mem.cfg.rehearsal_growth
                )
                retention = math.exp(-t_seconds / stability_seconds) if stability_seconds > 0 else 0.0
                if retention >= threshold:
                    kept += 1
            retained_counts.append(kept)

        plt.figure(figsize=(10, 5))
        plt.plot(time_points_hours, retained_counts, marker='o')
        plt.axhline(0, color='black', linewidth=1)
        plt.title('Episodic retention over time (MemoryOS demo)')
        plt.xlabel('Time after now (hours)')
        plt.ylabel('# memories above decay threshold')
        plt.tight_layout()
        out_path = os.path.join(tmp, 'retention_demo.png')
        plt.savefig(out_path)
        print(out_path)


if __name__ == '__main__':
    main()
