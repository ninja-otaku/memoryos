from __future__ import annotations

from typing import Any, Optional

from memoryos.core import MemoryOS


try:  # pragma: no cover
    from langchain.memory import BaseMemory  # type: ignore
except Exception:  # pragma: no cover
    BaseMemory = object  # type: ignore


class MemoryOSChatMemory(BaseMemory):  # pragma: no cover
    '''Minimal LangChain BaseMemory adapter (dependency-free at runtime).'''

    def __init__(self, memory_os: MemoryOS, *, query_key: str = 'input') -> None:
        self.memory_os = memory_os
        self.query_key = query_key

    @property
    def memory_variables(self) -> list[str]:
        return ['memory_context']

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        query = str(inputs.get(self.query_key, ''))
        memories = self.memory_os.recall(query, memory_type='semantic')
        context = '\n'.join([m.content for m in memories])
        return {'memory_context': context}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        # Store the assistant output as episodic memory.
        event = str(outputs.get('output') or outputs.get('text') or '')
        if event.strip():
            self.memory_os.remember(event, importance=0.5)


class MemoryOSNode:  # pragma: no cover
    '''Minimal stateful node for LangGraph-like execution.'''

    def __init__(self, memory_os: MemoryOS) -> None:
        self.memory_os = memory_os

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        query = str(state.get('memory_query', ''))
        mem_type = str(state.get('memory_type', 'all'))
        memories = self.memory_os.recall(query, memory_type=mem_type)  # type: ignore[arg-type]
        state['memories'] = [m.model_dump() for m in memories]
        return state


class MemoryOSTool:  # pragma: no cover
    '''Dependency-free CrewAI-ish tool wrapper.'''

    name: str = 'memoryos'
    description: str = 'Episodic + semantic memory tool for AI agents'

    def __init__(self, memory_os: MemoryOS) -> None:
        self.memory_os = memory_os

    def run(self, query: str, memory_type: str = 'all') -> str:
        memories = self.memory_os.recall(query, memory_type=memory_type)  # type: ignore[arg-type]
        return '\n'.join([m.content for m in memories])
