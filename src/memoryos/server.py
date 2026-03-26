from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from memoryos.core import MemoryOS, MemoryOSConfig

app = FastAPI(title='MemoryOS')


class RememberRequest(BaseModel):
    event: str
    importance: float = Field(0.5, ge=0.0, le=1.0)


class RecallRequest(BaseModel):
    query: str
    memory_type: Literal['all', 'episodic', 'semantic', 'procedural'] = 'all'
    k: int = Field(5, ge=1)


class ForgetRequest(BaseModel):
    older_than_days: int = Field(..., ge=0)
    min_importance: float = Field(0.0, ge=0.0, le=1.0)


class SummaryResponse(BaseModel):
    summary: dict[str, Any]


def build_memoryos() -> MemoryOS:
    cfg = MemoryOSConfig(
        data_dir=os.getenv('MEMORYOS_DATA_DIR', 'data'),
        sqlite_path=os.getenv('MEMORYOS_SQLITE_PATH', 'data/memoryos.sqlite3'),
        chroma_dir=os.getenv('MEMORYOS_CHROMA_DIR', 'data/chroma'),
    )
    return MemoryOS(cfg=cfg)


_memoryos = build_memoryos()


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/remember')
def remember(req: RememberRequest) -> dict[str, str]:
    mem_id = _memoryos.remember(req.event, importance=req.importance)
    return {'id': mem_id}


@app.post('/recall')
def recall(req: RecallRequest) -> dict[str, Any]:
    memories = _memoryos.recall(req.query, memory_type=req.memory_type)
    return {'memories': [m.model_dump() for m in memories]}


@app.post('/forget')
def forget(req: ForgetRequest) -> dict[str, str]:
    _memoryos.forget(older_than_days=req.older_than_days, min_importance=req.min_importance)
    return {'status': 'ok'}


@app.post('/consolidate')
def consolidate() -> dict[str, str]:
    _memoryos.consolidate()
    return {'status': 'ok'}


@app.get('/summary', response_model=SummaryResponse)
def summary() -> SummaryResponse:
    return SummaryResponse(summary=_memoryos.get_memory_summary())


@app.post('/procedural/learn')
def procedural_learn(body: dict[str, Any]) -> dict[str, str]:
    task_type = str(body['task_type'])
    steps = list(body['steps'])
    success_rate = float(body.get('initial_success_rate', 0.5))
    template_id = _memoryos.learn_procedure(
        task_type=task_type,
        steps=steps,
        initial_success_rate=success_rate,
    )
    return {'template_id': template_id}


@app.post('/procedural/success')
def procedural_success(body: dict[str, Any]) -> dict[str, str]:
    template_id = str(body['template_id'])
    success = bool(body['success'])
    _memoryos.update_procedure_success(template_id=template_id, success=success)
    return {'status': 'ok'}


def main() -> None:
    import uvicorn

    port = int(os.getenv('PORT', '8000'))
    host = os.getenv('HOST', '127.0.0.1')
    uvicorn.run('memoryos.server:app', host=host, port=port, reload=False)
