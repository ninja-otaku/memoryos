from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

MemoryType = Literal['episodic', 'semantic', 'procedural']
RecallMemoryType = Literal['all', 'episodic', 'semantic', 'procedural']


class Memory(BaseModel):
    id: str
    memory_type: MemoryType
    content: str
    score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryOSConfig(BaseModel):
    data_dir: str = 'data'
    sqlite_path: str = 'data/memoryos.sqlite3'
    chroma_dir: str = 'data/chroma'

    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_dim: int = 384

    episodic_decay_threshold: float = 0.3
    base_stability_seconds: float = 60.0 * 60.0 * 24.0 * 7.0
    rehearsal_growth: float = 0.5

    consolidation_importance_threshold: float = 0.7
    consolidation_retention_threshold: float = 0.3

    semantic_stale_days: int = 30
    use_chroma: bool = True


def _now_ts() -> float:
    return time.time()


class Embedder:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = int(embedding_dim)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    '''Deterministic fallback embedder (token hashing + L2 normalize).'''

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for text in texts:
            v = np.zeros(self.embedding_dim, dtype=np.float32)
            for tok in text.lower().split():
                if not tok:
                    continue
                h = hashlib.sha256(tok.encode('utf-8')).digest()
                idx = int.from_bytes(h[:4], 'little') % self.embedding_dim
                v[idx] += 1.0
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
            vecs.append(v)
        return np.vstack(vecs).astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, *, model_name: str, embedding_dim: int) -> None:
        super().__init__(embedding_dim=embedding_dim)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError('sentence-transformers is required for embeddings') from e

        self._model = SentenceTransformer(model_name)
        dim_fn = getattr(self._model, 'get_sentence_embedding_dimension', None)
        if callable(dim_fn):
            try:
                self.embedding_dim = int(dim_fn())
            except Exception:
                pass

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


def build_embedder(cfg: MemoryOSConfig) -> Embedder:
    try:
        return SentenceTransformerEmbedder(
            model_name=cfg.embedding_model,
            embedding_dim=cfg.embedding_dim,
        )
    except Exception:
        return HashingEmbedder(embedding_dim=cfg.embedding_dim)


@dataclass(frozen=True)
class EpisodicDecay:
    retention: float
    stability_seconds: float

class SQLiteEpisodicStore:
    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    event_description TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    decay_factor REAL NOT NULL,
                    rehearsal_count INTEGER NOT NULL DEFAULT 0
                )
                '''
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_event(
        self,
        *,
        event_description: str,
        embedding: np.ndarray,
        timestamp: float,
        importance_score: float,
        decay_factor: float,
    ) -> str:
        mem_id = str(uuid.uuid4())
        embedding_json = json.dumps(embedding.astype(float).tolist())
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO episodic_memories (
                    id, timestamp, event_description, embedding_json,
                    importance_score, decay_factor, rehearsal_count
                ) VALUES (?, ?, ?, ?, ?, ?, 0)
                ''',
                (
                    mem_id,
                    float(timestamp),
                    event_description,
                    embedding_json,
                    float(importance_score),
                    float(decay_factor),
                ),
            )
            conn.commit()
        return mem_id

    def list_all(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT id, timestamp, event_description, embedding_json,
                       importance_score, decay_factor, rehearsal_count
                FROM episodic_memories
                '''
            ).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    'id': r['id'],
                    'timestamp': float(r['timestamp']),
                    'event_description': r['event_description'],
                    'embedding': np.asarray(json.loads(r['embedding_json']), dtype=np.float32),
                    'importance_score': float(r['importance_score']),
                    'decay_factor': float(r['decay_factor']),
                    'rehearsal_count': int(r['rehearsal_count']),
                }
            )
        return out

    def update_rehearsals(self, ids: list[str], inc: int) -> None:
        if not ids:
            return
        with self._connect() as conn:
            conn.executemany(
                '''
                UPDATE episodic_memories
                SET rehearsal_count = rehearsal_count + ?
                WHERE id = ?
                ''',
                [(int(inc), mem_id) for mem_id in ids],
            )
            conn.commit()

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._connect() as conn:
            conn.executemany(
                '''DELETE FROM episodic_memories WHERE id = ?''',
                [(mem_id,) for mem_id in ids],
            )
            conn.commit()


class EpisodicMemory:
    def __init__(self, cfg: MemoryOSConfig, embedder: Embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.store = SQLiteEpisodicStore(cfg.sqlite_path)

    def _decay(self, *, timestamp: float, decay_factor: float, rehearsal_count: int) -> EpisodicDecay:
        now = time.time()
        t_seconds = max(0.0, now - float(timestamp))
        stability_seconds = float(decay_factor) * (1.0 + float(rehearsal_count) * self.cfg.rehearsal_growth)
        if stability_seconds <= 0:
            return EpisodicDecay(retention=0.0, stability_seconds=0.0)
        retention = math.exp(-t_seconds / stability_seconds)
        return EpisodicDecay(retention=retention, stability_seconds=stability_seconds)

    def add_event(self, event_description: str, importance: float = 0.5, *, timestamp: float | None = None) -> str:
        imp = float(importance)
        if not (0.0 <= imp <= 1.0):
            raise ValueError('importance must be in [0,1]')

        ts = time.time() if timestamp is None else float(timestamp)
        decay_factor = self.cfg.base_stability_seconds * (0.5 + imp)
        emb = self.embedder.embed_texts([event_description])[0]
        return self.store.add_event(
            event_description=event_description,
            embedding=emb,
            timestamp=ts,
            importance_score=imp,
            decay_factor=decay_factor,
        )

    def retrieve_recent(self, *, k: int = 10, decay_threshold: float | None = None) -> list[Memory]:
        threshold = self.cfg.episodic_decay_threshold if decay_threshold is None else float(decay_threshold)
        entries = self.store.list_all()

        scored: list[tuple[float, dict[str, Any]]] = []
        accessed_ids: list[str] = []
        for e in entries:
            decay = self._decay(
                timestamp=e['timestamp'],
                decay_factor=e['decay_factor'],
                rehearsal_count=e['rehearsal_count'],
            )
            if decay.retention < threshold:
                continue
            scored.append((decay.retention, e))
            accessed_ids.append(e['id'])

        self.store.update_rehearsals(accessed_ids, inc=1)
        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[Memory] = []
        for score, e in scored[:k]:
            out.append(
                Memory(
                    id=e['id'],
                    memory_type='episodic',
                    content=e['event_description'],
                    score=float(score),
                    metadata={
                        'importance': e['importance_score'],
                        'decay_factor': e['decay_factor'],
                        'rehearsal_count': e['rehearsal_count'],
                    },
                )
            )
        return out

    def retrieve_relevant(self, *, query: str, k: int = 5, decay_threshold: float | None = None) -> list[Memory]:
        threshold = self.cfg.episodic_decay_threshold if decay_threshold is None else float(decay_threshold)
        q_emb = self.embedder.embed_texts([query])[0]
        entries = self.store.list_all()

        scored: list[tuple[float, dict[str, Any]]] = []
        accessed_ids: list[str] = []
        for e in entries:
            decay = self._decay(
                timestamp=e['timestamp'],
                decay_factor=e['decay_factor'],
                rehearsal_count=e['rehearsal_count'],
            )
            if decay.retention < threshold:
                continue
            similarity = float(np.dot(q_emb, e['embedding']))
            score = similarity * float(decay.retention)
            scored.append((score, e))
            accessed_ids.append(e['id'])

        self.store.update_rehearsals(accessed_ids, inc=1)
        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[Memory] = []
        for score, e in scored[:k]:
            out.append(
                Memory(
                    id=e['id'],
                    memory_type='episodic',
                    content=e['event_description'],
                    score=float(score),
                    metadata={
                        'importance': e['importance_score'],
                        'rehearsal_count': e['rehearsal_count'],
                    },
                )
            )
        return out

    def forget(self, *, older_than_days: int, min_importance: float) -> None:
        cutoff_ts = time.time() - float(older_than_days) * 86400.0
        entries = self.store.list_all()
        to_delete: list[str] = []
        for e in entries:
            if e['timestamp'] < cutoff_ts and e['importance_score'] < float(min_importance):
                to_delete.append(e['id'])
        self.store.delete_ids(to_delete)

    def consolidate_to_semantic(
        self,
        *,
        semantic: 'SemanticMemory',
        importance_threshold: float,
        retention_threshold: float,
    ) -> None:
        entries = self.store.list_all()
        selected: list[tuple[float, dict[str, Any]]] = []

        for e in entries:
            if e['importance_score'] < float(importance_threshold):
                continue
            decay = self._decay(
                timestamp=e['timestamp'],
                decay_factor=e['decay_factor'],
                rehearsal_count=e['rehearsal_count'],
            )
            if decay.retention < float(retention_threshold):
                continue
            selected.append((float(e['importance_score']) * float(decay.retention), e))

        selected.sort(key=lambda x: x[0], reverse=True)
        to_delete: list[str] = []
        for _, e in selected:
            semantic.update(fact=e['event_description'], confidence=e['importance_score'], source='episodic')
            to_delete.append(e['id'])

        self.store.delete_ids(to_delete)

class SemanticMemory:
    def __init__(self, cfg: MemoryOSConfig, embedder: Embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self._use_chroma = False
        self._collection = None
        self._facts: list[dict[str, Any]] = []

        if not getattr(cfg, 'use_chroma', True):
            return

        try:
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=cfg.chroma_dir)
            self._collection = client.get_or_create_collection(
                name='semantic_memory',
                metadata={'hnsw:space': 'cosine'},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False

    def _fact_hash(self, fact: str) -> str:
        return hashlib.sha256(fact.encode('utf-8')).hexdigest()

    def update(self, *, fact: str, confidence: float, source: str) -> None:
        conf = max(0.0, min(1.0, float(confidence)))
        now = time.time()
        fact_hash = self._fact_hash(fact)

        if not self._use_chroma:
            for f in self._facts:
                if f['fact_hash'] == fact_hash:
                    alpha = float(f.get('alpha', conf))
                    beta = float(f.get('beta', 1.0 - conf))
                    alpha += conf
                    beta += 1.0 - conf
                    f['alpha'] = alpha
                    f['beta'] = beta
                    f['confidence'] = alpha / (alpha + beta) if alpha + beta > 0 else conf
                    f['last_accessed'] = now
                    return

            emb = self.embedder.embed_texts([fact])[0]
            self._facts.append(
                {
                    'fact_hash': fact_hash,
                    'fact': fact,
                    'confidence': conf,
                    'alpha': conf,
                    'beta': 1.0 - conf,
                    'source': source,
                    'last_accessed': now,
                    'embedding': emb,
                }
            )
            return

        assert self._collection is not None

        got = self._collection.get(
            where={'fact_hash': fact_hash},
            include=['ids', 'metadatas'],
        )
        ids = got.get('ids') or []

        emb = self.embedder.embed_texts([fact])[0]
        if ids:
            mem_id = str(ids[0])
            meta_list = got.get('metadatas') or []
            meta = meta_list[0] or {}
            alpha = float(meta.get('alpha', conf))
            beta = float(meta.get('beta', 1.0 - conf))
            alpha += conf
            beta += 1.0 - conf
            new_conf = alpha / (alpha + beta) if alpha + beta > 0 else conf

            self._collection.delete(ids=[mem_id])
            self._collection.add(
                ids=[mem_id],
                documents=[fact],
                embeddings=[emb.tolist()],
                metadatas=[
                    {
                        'fact_hash': fact_hash,
                        'confidence': new_conf,
                        'alpha': alpha,
                        'beta': beta,
                        'source': source,
                        'last_accessed': now,
                    }
                ],
            )
            return

        self._collection.add(
            ids=[fact_hash],
            documents=[fact],
            embeddings=[emb.tolist()],
            metadatas=[
                {
                    'fact_hash': fact_hash,
                    'confidence': conf,
                    'alpha': conf,
                    'beta': 1.0 - conf,
                    'source': source,
                    'last_accessed': now,
                }
            ],
        )

    def retrieve(self, *, query: str, k: int = 5) -> list[Memory]:
        q_emb = self.embedder.embed_texts([query])[0]
        now = time.time()

        if not self._use_chroma:
            scored: list[tuple[float, dict[str, Any]]] = []
            for f in self._facts:
                sim = float(np.dot(q_emb, f['embedding']))
                freshness_days = (now - float(f.get('last_accessed', now))) / 86400.0
                stale_penalty = math.exp(-freshness_days / float(self.cfg.semantic_stale_days))
                score = sim * float(f.get('confidence', 0.5)) * stale_penalty
                scored.append((score, f))

            scored.sort(key=lambda x: x[0], reverse=True)
            out: list[Memory] = []
            for score, f in scored[:k]:
                out.append(
                    Memory(
                        id=f['fact_hash'],
                        memory_type='semantic',
                        content=f['fact'],
                        score=float(score),
                        metadata={
                            'confidence': float(f.get('confidence', 0.5)),
                            'source': f.get('source'),
                            'last_accessed': f.get('last_accessed'),
                        },
                    )
                )
            return out

        assert self._collection is not None
        res = self._collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances'],
        )

        docs = res.get('documents') or [[]]
        metas = res.get('metadatas') or [[]]
        dists = res.get('distances') or [[]]

        out: list[Memory] = []
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            dist_f = float(dist)
            sim = 1.0 - dist_f
            conf = float((meta or {}).get('confidence', 0.5))
            freshness_days = (now - float((meta or {}).get('last_accessed', now))) / 86400.0
            stale_penalty = math.exp(-freshness_days / float(self.cfg.semantic_stale_days))
            score = sim * conf * stale_penalty

            out.append(
                Memory(
                    id=(meta or {}).get('fact_hash', ''),
                    memory_type='semantic',
                    content=str(doc),
                    score=float(score),
                    metadata={
                        'confidence': conf,
                        'source': (meta or {}).get('source'),
                        'last_accessed': (meta or {}).get('last_accessed'),
                    },
                )
            )
        return out

    def forget_outdated(self, *, threshold: float = 0.3) -> None:
        conf_thr = float(threshold)
        cutoff_ts = time.time() - float(self.cfg.semantic_stale_days) * 86400.0

        if not self._use_chroma:
            self._facts = [
                f
                for f in self._facts
                if float(f.get('confidence', 0.5)) >= conf_thr
                and float(f.get('last_accessed', cutoff_ts)) >= cutoff_ts
            ]
            return

        assert self._collection is not None
        got = self._collection.get(include=['ids', 'metadatas'])
        ids = got.get('ids') or []
        metas = got.get('metadatas') or []

        to_delete: list[str] = []
        for mem_id, meta in zip(ids, metas):
            meta = meta or {}
            conf = float(meta.get('confidence', 0.5))
            last_accessed = float(meta.get('last_accessed', cutoff_ts))
            if conf < conf_thr or last_accessed < cutoff_ts:
                to_delete.append(str(mem_id))

        if to_delete:
            self._collection.delete(ids=to_delete)


class SQLiteProceduralStore:
    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS procedural_templates (
                    template_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    last_used REAL NOT NULL
                )
                '''
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_template(
        self,
        *,
        template_id: str,
        task_type: str,
        steps: list[str],
        embedding: np.ndarray,
        initial_success_rate: float,
    ) -> None:
        steps_json = json.dumps(steps)
        embedding_json = json.dumps(embedding.astype(float).tolist())
        now = time.time()

        with self._connect() as conn:
            existing = conn.execute(
                '''SELECT template_id FROM procedural_templates WHERE template_id = ?''',
                (template_id,),
            ).fetchone()

            if existing is None:
                conn.execute(
                    '''
                    INSERT INTO procedural_templates (
                        template_id, task_type, steps_json, embedding_json,
                        success_rate, last_used
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        template_id,
                        task_type,
                        steps_json,
                        embedding_json,
                        float(initial_success_rate),
                        float(now),
                    ),
                )
            else:
                conn.execute(
                    '''
                    UPDATE procedural_templates
                    SET task_type = ?, steps_json = ?, embedding_json = ?,
                        last_used = ?
                    WHERE template_id = ?
                    ''',
                    (task_type, steps_json, embedding_json, float(now), template_id),
                )
            conn.commit()

    def list_templates(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT template_id, task_type, steps_json, embedding_json,
                       success_rate, last_used
                FROM procedural_templates
                '''
            ).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    'template_id': r['template_id'],
                    'task_type': r['task_type'],
                    'steps': json.loads(r['steps_json']),
                    'embedding': np.asarray(json.loads(r['embedding_json']), dtype=np.float32),
                    'success_rate': float(r['success_rate']),
                    'last_used': float(r['last_used']),
                }
            )
        return out

    def update_success(self, template_id: str, *, success: bool, alpha: float = 0.2) -> None:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                '''SELECT success_rate FROM procedural_templates WHERE template_id = ?''',
                (template_id,),
            ).fetchone()
            if row is None:
                return

            old = float(row['success_rate'])
            target = 1.0 if success else 0.0
            new = old * (1.0 - float(alpha)) + target * float(alpha)

            conn.execute(
                '''
                UPDATE procedural_templates
                SET success_rate = ?, last_used = ?
                WHERE template_id = ?
                ''',
                (new, float(now), template_id),
            )
            conn.commit()


class ProceduralMemory:
    def __init__(self, cfg: MemoryOSConfig, embedder: Embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.store = SQLiteProceduralStore(cfg.sqlite_path)

    def _template_embedding(self, task_type: str, steps: list[str]) -> np.ndarray:
        text = task_type + ' :: ' + ' | '.join(steps)
        return self.embedder.embed_texts([text])[0]

    def learn(self, *, task_type: str, steps: list[str], initial_success_rate: float = 0.5) -> str:
        template_id = hashlib.sha256((task_type + '|' + json.dumps(steps)).encode('utf-8')).hexdigest()
        emb = self._template_embedding(task_type, steps)
        self.store.upsert_template(
            template_id=template_id,
            task_type=task_type,
            steps=steps,
            embedding=emb,
            initial_success_rate=float(initial_success_rate),
        )
        return template_id

    def match(self, *, task_description: str) -> Optional[Memory]:
        templates = self.store.list_templates()
        if not templates:
            return None

        q_emb = self.embedder.embed_texts([task_description])[0]
        best: tuple[float, dict[str, Any]] | None = None

        for t in templates:
            sim = float(np.dot(q_emb, t['embedding']))
            if best is None or sim > best[0]:
                best = (sim, t)

        assert best is not None
        sim, t = best
        return Memory(
            id=t['template_id'],
            memory_type='procedural',
            content=t['task_type'] + ' :: steps=' + str(t['steps']),
            score=float(sim),
            metadata={
                'task_type': t['task_type'],
                'steps': t['steps'],
                'success_rate': t['success_rate'],
                'last_used': t['last_used'],
            },
        )

    def update_success(self, template_id: str, *, success: bool) -> None:
        self.store.update_success(template_id, success=success)


class MemoryOS:
    def __init__(self, cfg: MemoryOSConfig | None = None) -> None:
        self.cfg = cfg or MemoryOSConfig()
        os.makedirs(self.cfg.data_dir, exist_ok=True)
        self.embedder = build_embedder(self.cfg)

        self.episodic = EpisodicMemory(cfg=self.cfg, embedder=self.embedder)
        self.semantic = SemanticMemory(cfg=self.cfg, embedder=self.embedder)
        self.procedural = ProceduralMemory(cfg=self.cfg, embedder=self.embedder)

    def remember(self, event: str, importance: float = 0.5) -> str:
        return self.episodic.add_event(event_description=event, importance=importance)

    def recall(self, query: str, memory_type: RecallMemoryType = 'all') -> list[Memory]:
        out: list[Memory] = []

        if memory_type in ('all', 'episodic'):
            out.extend(self.episodic.retrieve_recent(k=10))
            out.extend(self.episodic.retrieve_relevant(query=query, k=5))

        if memory_type in ('all', 'semantic'):
            out.extend(self.semantic.retrieve(query=query, k=5))

        if memory_type in ('all', 'procedural'):
            m = self.procedural.match(task_description=query)
            if m is not None:
                out.append(m)

        best_by_id: dict[str, Memory] = {}
        for m in out:
            if m.id not in best_by_id:
                best_by_id[m.id] = m
                continue
            existing = best_by_id[m.id]
            if existing.score is None:
                best_by_id[m.id] = m
                continue
            if m.score is not None and m.score > existing.score:
                best_by_id[m.id] = m

        res = list(best_by_id.values())
        res.sort(key=lambda m: float(m.score) if m.score is not None else -1e9, reverse=True)
        return res

    def forget(self, older_than_days: int, min_importance: float) -> None:
        self.episodic.forget(older_than_days=older_than_days, min_importance=min_importance)
        self.semantic.forget_outdated(threshold=min_importance)

    def consolidate(self) -> None:
        self.episodic.consolidate_to_semantic(
            semantic=self.semantic,
            importance_threshold=self.cfg.consolidation_importance_threshold,
            retention_threshold=self.cfg.consolidation_retention_threshold,
        )

    def get_memory_summary(self) -> dict[str, Any]:
        episodic_count = len(self.episodic.store.list_all())

        if not self.semantic._use_chroma:
            semantic_count = len(self.semantic._facts)
        else:
            assert self.semantic._collection is not None
            got = self.semantic._collection.get(include=['ids'])
            semantic_count = len(got.get('ids') or [])

        procedural_count = len(self.procedural.store.list_templates())

        return {
            'episodic_count': episodic_count,
            'semantic_count': semantic_count,
            'procedural_templates': procedural_count,
        }

    def learn_procedure(self, task_type: str, steps: list[str], initial_success_rate: float = 0.5) -> str:
        return self.procedural.learn(task_type=task_type, steps=steps, initial_success_rate=initial_success_rate)

    def update_procedure_success(self, template_id: str, success: bool) -> None:
        self.procedural.update_success(template_id=template_id, success=success)
