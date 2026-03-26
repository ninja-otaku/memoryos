# MemoryOS

[![CI](https://github.com/ninja-otaku/memoryos/actions/workflows/ci.yml/badge.svg)](https://github.com/ninja-otaku/memoryos/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/memoryos)](https://pypi.org/project/memoryos/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)

**Episodic + semantic memory backend for long-running AI agents.**
Think "hippocampus as a service."

```python
from memoryos import MemoryOS
mem = MemoryOS()
mem.remember("User prefers concise answers.", importance=0.8)
results = mem.recall("How should I answer?")
```

## Install

```bash
pip install memoryos
# with vector search (ChromaDB + sentence-transformers):
pip install "memoryos[full]"
```

## Quickstart

```python
from memoryos import MemoryOS

mem = MemoryOS()
mem.remember("The user is debugging a FastAPI timeout issue.", importance=0.7)

for r in mem.recall("FastAPI performance"):
    print(r.content, r.score)

mem.consolidate()  # episodic -> semantic
mem.forget(older_than_days=30, min_importance=0.3)
```

## HTTP Server

```bash
memoryos-server   # http://localhost:8000
```

## Architecture

```
MemoryOS
├── EpisodicMemory   SQLite -- timestamped events, Ebbinghaus decay
├── SemanticMemory   ChromaDB -- consolidated facts, confidence scores
└── ProceduralMemory SQLite -- action templates, success-rate tracking
```

## Integrations

| Framework  | Adapter               |
|------------|-----------------------|
| LangChain  | `MemoryOSChatMemory`  |
| LangGraph  | `MemoryOSNode`        |
| CrewAI     | `MemoryOSTool`        |
| HTTP       | FastAPI REST server   |
| MCP        | `tools.json`          |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome!

## License

Apache 2.0 -- see [LICENSE](LICENSE).
