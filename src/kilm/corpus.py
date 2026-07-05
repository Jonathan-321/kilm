"""Corpus manifest loading and safety checks for KILM runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


TRAINABLE_STATUSES = {"approved", "toy"}


@dataclass(frozen=True)
class CorpusRecord:
    id: str
    path: Path
    status: str
    description: str = ""
    source: str = ""
    license: str = ""
    notes: str = ""

    @property
    def is_trainable(self) -> bool:
        return self.status in TRAINABLE_STATUSES

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "path": str(self.path),
            "status": self.status,
            "description": self.description,
            "source": self.source,
            "license": self.license,
            "notes": self.notes,
        }


def load_manifest(manifest_path: Path) -> dict[str, CorpusRecord]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload["corpora"] if isinstance(payload, dict) else payload

    records: dict[str, CorpusRecord] = {}
    for row in rows:
        corpus_id = str(row["id"])
        corpus_path = Path(row["path"])
        if not corpus_path.is_absolute():
            corpus_path = manifest_path.parent / corpus_path
        if corpus_id in records:
            raise ValueError(f"duplicate corpus id: {corpus_id}")
        records[corpus_id] = CorpusRecord(
            id=corpus_id,
            path=corpus_path.resolve(),
            status=str(row["status"]),
            description=str(row.get("description", "")),
            source=str(row.get("source", "")),
            license=str(row.get("license", "")),
            notes=str(row.get("notes", "")),
        )
    return records


def load_corpus_text(
    *,
    manifest_path: Path,
    corpus_id: str,
    corpus_path: Path | None = None,
    allow_unapproved: bool = False,
) -> tuple[CorpusRecord, str]:
    if corpus_path is not None:
        record = CorpusRecord(
            id=corpus_path.stem,
            path=corpus_path.resolve(),
            status="custom",
            description="Direct corpus path supplied on the command line.",
            source="local file",
            license="unknown",
            notes="Mark this source in the manifest before treating it as approved.",
        )
    else:
        records = load_manifest(manifest_path)
        try:
            record = records[corpus_id]
        except KeyError as error:
            known = ", ".join(sorted(records))
            raise ValueError(
                f"unknown corpus id {corpus_id!r}; known ids: {known}"
            ) from error

    if not record.is_trainable and not allow_unapproved:
        raise ValueError(
            f"corpus {record.id!r} has status {record.status!r}. "
            "Use --allow-unapproved-corpus only for local debugging, or mark "
            "the source as approved/toy in the manifest."
        )
    if not record.path.exists():
        raise FileNotFoundError(f"corpus file does not exist: {record.path}")

    text = record.path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"corpus file is empty: {record.path}")
    return record, text
