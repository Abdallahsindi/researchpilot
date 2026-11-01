"""Persistence for completed research runs.

One JSON file per run is sufficient for this local release. The optional pgvector backend
stores document chunks separately; completed runs remain in this local store.
"""

import os
import tempfile
from pathlib import Path
from uuid import UUID

from app.schemas.research import ResearchResult


class JsonRunStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: UUID) -> Path:
        return self.directory / f"{run_id}.json"

    def save(self, result: ResearchResult) -> None:
        # Write to a temp file and rename so readers never see a half-written run.
        fd, tmp_path = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(result.model_dump_json(indent=2))
        os.replace(tmp_path, self._path(result.run_id))

    def get(self, run_id: UUID) -> ResearchResult | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        return ResearchResult.model_validate_json(path.read_text(encoding="utf-8"))
