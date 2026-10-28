"""SQLite metadata store for projects, inputs and versioned knowledge layers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.app.domain.models import (
    KGTriple,
    KnowledgeBaseVersion,
    Project,
    QueryEvent,
    QueryRecord,
    Source,
    TextEntry,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProjectNotFoundError(LookupError):
    pass


class VersionNotFoundError(LookupError):
    pass


class SQLiteStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'empty',
                    current_version_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_projects_updated_at
                    ON projects(updated_at DESC);

                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_format TEXT NOT NULL,
                    mime_type TEXT,
                    checksum_sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    parse_status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
                    UNIQUE(project_id, checksum_sha256)
                );
                CREATE INDEX IF NOT EXISTS idx_sources_project_created
                    ON sources(project_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS knowledge_base_versions (
                    version_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    version_no INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    text_status TEXT NOT NULL,
                    kg_status TEXT NOT NULL,
                    source_snapshot_json TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
                    UNIQUE(project_id, version_no)
                );
                CREATE INDEX IF NOT EXISTS idx_versions_project_created
                    ON knowledge_base_versions(project_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS text_entries (
                    entry_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source_location_json TEXT NOT NULL,
                    parent_id TEXT,
                    kg_triple_ids_json TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY(version_id) REFERENCES knowledge_base_versions(version_id) ON DELETE CASCADE,
                    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE RESTRICT,
                    UNIQUE(version_id, source_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_text_entries_version
                    ON text_entries(version_id, source_id, ordinal);

                CREATE TABLE IF NOT EXISTS vector_indexes (
                    index_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    index_type TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    status TEXT NOT NULL,
                    collection_name TEXT,
                    storage_uri TEXT NOT NULL DEFAULT '',
                    embedding_model TEXT NOT NULL,
                    embedding_input_rule TEXT NOT NULL,
                    dimension INTEGER NOT NULL DEFAULT 0,
                    metric TEXT NOT NULL DEFAULT 'cosine',
                    vector_count INTEGER NOT NULL DEFAULT 0,
                    content_fingerprint TEXT NOT NULL DEFAULT '',
                    config_json TEXT NOT NULL DEFAULT '{}',
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
                    FOREIGN KEY(version_id) REFERENCES knowledge_base_versions(version_id) ON DELETE CASCADE,
                    UNIQUE(version_id, index_type)
                );
                CREATE INDEX IF NOT EXISTS idx_vector_indexes_project
                    ON vector_indexes(project_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_vector_indexes_status
                    ON vector_indexes(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS kg_entities (
                    entity_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    FOREIGN KEY(version_id) REFERENCES knowledge_base_versions(version_id) ON DELETE CASCADE,
                    UNIQUE(version_id, normalized_name)
                );

                CREATE TABLE IF NOT EXISTS kg_triples (
                    triple_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_entry_id TEXT,
                    source_line INTEGER,
                    head TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    tail TEXT NOT NULL,
                    head_norm TEXT NOT NULL,
                    relation_norm TEXT NOT NULL,
                    tail_norm TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    evidence TEXT,
                    triple_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(version_id) REFERENCES knowledge_base_versions(version_id) ON DELETE CASCADE,
                    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE RESTRICT,
                    FOREIGN KEY(source_entry_id) REFERENCES text_entries(entry_id) ON DELETE SET NULL,
                    UNIQUE(version_id, triple_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_kg_triples_version_head_relation
                    ON kg_triples(version_id, head_norm, relation_norm);
                CREATE INDEX IF NOT EXISTS idx_kg_triples_version_tail_relation
                    ON kg_triples(version_id, tail_norm, relation_norm);

                CREATE TABLE IF NOT EXISTS kg_conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    triple_id TEXT NOT NULL,
                    other_triple_id TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    anchor TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(version_id) REFERENCES knowledge_base_versions(version_id) ON DELETE CASCADE,
                    FOREIGN KEY(triple_id) REFERENCES kg_triples(triple_id) ON DELETE CASCADE,
                    FOREIGN KEY(other_triple_id) REFERENCES kg_triples(triple_id) ON DELETE CASCADE,
                    UNIQUE(version_id, triple_id, other_triple_id, conflict_type)
                );
                CREATE INDEX IF NOT EXISTS idx_kg_conflicts_version
                    ON kg_conflicts(version_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS queries (
                    query_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    trace_level TEXT NOT NULL DEFAULT 'summary',
                    status TEXT NOT NULL DEFAULT 'pending',
                    current_step TEXT,
                    answer TEXT,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
                    FOREIGN KEY(version_id) REFERENCES knowledge_base_versions(version_id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_queries_project_created
                    ON queries(project_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS query_events (
                    event_id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    constraint_id TEXT,
                    variable_id TEXT,
                    pass_index INTEGER,
                    branch_id TEXT,
                    input_ref TEXT,
                    output_ref TEXT,
                    counts_json TEXT NOT NULL DEFAULT '{}',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    reason_code TEXT,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    elapsed_ms REAL,
                    FOREIGN KEY(query_id) REFERENCES queries(query_id) ON DELETE CASCADE,
                    UNIQUE(query_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_query_events_query_seq
                    ON query_events(query_id, seq);

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_updated_at
                    ON tasks(updated_at DESC);

                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR REPLACE INTO schema_meta(key, value)
                    VALUES ('schema_version', '7');
                """
            )
            self._ensure_text_entry_columns(connection)
            connection.commit()

    @staticmethod
    def _ensure_text_entry_columns(connection: sqlite3.Connection) -> None:
        """Add text-index columns to databases created before schema v6."""

        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(text_entries)").fetchall()
        }
        if "parent_id" not in columns:
            connection.execute("ALTER TABLE text_entries ADD COLUMN parent_id TEXT")
        if "kg_triple_ids_json" not in columns:
            connection.execute(
                "ALTER TABLE text_entries ADD COLUMN kg_triple_ids_json TEXT NOT NULL DEFAULT '[]'"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_text_entries_parent "
            "ON text_entries(version_id, parent_id)"
        )

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> Project:
        return Project(
            project_id=row["project_id"],
            name=row["name"],
            description=row["description"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            current_version_id=row["current_version_id"],
        )

    def list_projects(self) -> list[Project]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_project(row) for row in rows]

    def get_project(self, project_id: str) -> Project:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
        if row is None:
            raise ProjectNotFoundError(project_id)
        return self._row_to_project(row)

    def create_project(self, name: str, description: str = "") -> Project:
        project_id = str(uuid4())
        now = _now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO projects(
                    project_id, name, description, status,
                    current_version_id, created_at, updated_at
                ) VALUES (?, ?, ?, 'empty', NULL, ?, ?)
                """,
                (project_id, name, description, now, now),
            )
            connection.commit()
        return self.get_project(project_id)

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        return Source(
            source_id=row["source_id"],
            project_id=row["project_id"],
            original_name=row["original_name"],
            stored_path=row["stored_path"],
            source_kind=row["source_kind"],
            source_format=row["source_format"],
            mime_type=row["mime_type"],
            checksum_sha256=row["checksum_sha256"],
            size_bytes=row["size_bytes"],
            parse_status=row["parse_status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def list_sources(self, project_id: str) -> list[Source]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM sources WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [self._row_to_source(row) for row in rows]

    def get_source(self, source_id: str) -> Source:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        if row is None:
            raise LookupError(source_id)
        return self._row_to_source(row)

    def find_source_by_checksum(self, project_id: str, checksum_sha256: str) -> Source | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM sources
                WHERE project_id = ? AND checksum_sha256 = ?
                """,
                (project_id, checksum_sha256),
            ).fetchone()
        return None if row is None else self._row_to_source(row)

    def create_source(
        self,
        project_id: str,
        original_name: str,
        stored_path: str,
        source_kind: str,
        source_format: str,
        mime_type: str | None,
        checksum_sha256: str,
        size_bytes: int,
    ) -> tuple[Source, bool]:
        source_id = str(uuid4())
        now = _now().isoformat()
        with closing(self._connect()) as connection:
            project = connection.execute(
                "SELECT project_id FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if project is None:
                raise ProjectNotFoundError(project_id)
            existing = connection.execute(
                """
                SELECT * FROM sources
                WHERE project_id = ? AND checksum_sha256 = ?
                """,
                (project_id, checksum_sha256),
            ).fetchone()
            if existing is not None:
                return self._row_to_source(existing), True
            connection.execute(
                """
                INSERT INTO sources(
                    source_id, project_id, original_name, stored_path,
                    source_kind, source_format, mime_type, checksum_sha256,
                    size_bytes, parse_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    source_id,
                    project_id,
                    original_name,
                    stored_path,
                    source_kind,
                    source_format,
                    mime_type,
                    checksum_sha256,
                    size_bytes,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE projects
                SET status = CASE WHEN status = 'empty' THEN 'input_ready' ELSE status END,
                    updated_at = ?
                WHERE project_id = ?
                """,
                (now, project_id),
            )
            connection.commit()
        return self.get_source(source_id), False

    def delete_source(self, source_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
            connection.commit()

    def update_source_path(self, source_id: str, stored_path: str) -> None:
        now = _now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE sources SET stored_path = ?, updated_at = ? WHERE source_id = ?",
                (stored_path, now, source_id),
            )
            connection.commit()

    def update_source_parse_status(self, source_id: str, parse_status: str) -> None:
        now = _now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE sources SET parse_status = ?, updated_at = ? WHERE source_id = ?",
                (parse_status, now, source_id),
            )
            connection.commit()

    @staticmethod
    def _normalize_value(value: str) -> str:
        return " ".join(value.strip().casefold().split())

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> KnowledgeBaseVersion:
        return KnowledgeBaseVersion(
            version_id=row["version_id"],
            project_id=row["project_id"],
            version_no=row["version_no"],
            mode=row["mode"],
            status=row["status"],
            text_status=row["text_status"],
            kg_status=row["kg_status"],
            source_snapshot=tuple(json.loads(row["source_snapshot_json"])),
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_text_entry(row: sqlite3.Row) -> TextEntry:
        try:
            kg_triple_ids = json.loads(row["kg_triple_ids_json"] or "[]")
        except (KeyError, TypeError, json.JSONDecodeError):
            kg_triple_ids = []
        if not isinstance(kg_triple_ids, list):
            kg_triple_ids = []
        return TextEntry(
            entry_id=row["entry_id"],
            version_id=row["version_id"],
            source_id=row["source_id"],
            ordinal=row["ordinal"],
            title=row["title"],
            content=row["content"],
            content_hash=row["content_hash"],
            source_location=json.loads(row["source_location_json"]),
            parent_id=row["parent_id"] if "parent_id" in row.keys() else None,
            kg_triple_ids=tuple(str(item) for item in kg_triple_ids if str(item).strip()),
        )

    @staticmethod
    def _row_to_triple(row: sqlite3.Row) -> KGTriple:
        return KGTriple(
            triple_id=row["triple_id"],
            version_id=row["version_id"],
            source_id=row["source_id"],
            source_entry_id=row["source_entry_id"],
            source_line=row["source_line"],
            head=row["head"],
            relation=row["relation"],
            tail=row["tail"],
            origin=row["origin"],
            evidence=row["evidence"],
            triple_hash=row["triple_hash"],
        )

    def create_version(self, project_id: str, mode: str, source_snapshot: list[str]) -> KnowledgeBaseVersion:
        version_id = str(uuid4())
        now = _now().isoformat()
        with closing(self._connect()) as connection:
            project = connection.execute(
                "SELECT project_id FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if project is None:
                raise ProjectNotFoundError(project_id)
            next_number = connection.execute(
                """
                SELECT COALESCE(MAX(version_no), 0) + 1
                FROM knowledge_base_versions WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO knowledge_base_versions(
                    version_id, project_id, version_no, mode, status,
                    text_status, kg_status, source_snapshot_json,
                    error_code, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'building', 'pending', 'pending', ?, NULL, NULL, ?, ?)
                """,
                (
                    version_id,
                    project_id,
                    next_number,
                    mode,
                    json.dumps(source_snapshot, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            connection.commit()
        return self.get_version(version_id)

    def get_version(self, version_id: str) -> KnowledgeBaseVersion:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_base_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
        if row is None:
            raise VersionNotFoundError(version_id)
        return self._row_to_version(row)

    def list_versions(self, project_id: str) -> list[KnowledgeBaseVersion]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM knowledge_base_versions
                WHERE project_id = ? ORDER BY version_no DESC
                """,
                (project_id,),
            ).fetchall()
        return [self._row_to_version(row) for row in rows]

    def update_version_status(
        self,
        version_id: str,
        *,
        status: str,
        text_status: str | None = None,
        kg_status: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> KnowledgeBaseVersion:
        current = self.get_version(version_id)
        now = _now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE knowledge_base_versions
                SET status = ?, text_status = ?, kg_status = ?,
                    error_code = ?, error_message = ?, updated_at = ?
                WHERE version_id = ?
                """,
                (
                    status,
                    text_status if text_status is not None else current.text_status,
                    kg_status if kg_status is not None else current.kg_status,
                    error_code,
                    error_message,
                    now,
                    version_id,
                ),
            )
            connection.commit()
        return self.get_version(version_id)

    def publish_version(
        self,
        version_id: str,
        *,
        status: str,
        text_status: str,
        kg_status: str,
    ) -> KnowledgeBaseVersion:
        version = self.update_version_status(
            version_id,
            status=status,
            text_status=text_status,
            kg_status=kg_status,
            error_code=None,
            error_message=None,
        )
        now = _now().isoformat()
        project_status = "ready" if status == "ready" else status
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE projects
                SET current_version_id = ?, status = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (version_id, project_status, now, version.project_id),
            )
            connection.commit()
        return self.get_version(version_id)

    def create_text_entry(
        self,
        version_id: str,
        source_id: str,
        ordinal: int,
        title: str,
        content: str,
        content_hash: str,
        source_location: dict[str, object],
        parent_id: str | None = None,
    ) -> TextEntry:
        entry_id = str(uuid4())
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO text_entries(
                    entry_id, version_id, source_id, ordinal, title, content,
                    content_hash, source_location_json, parent_id, kg_triple_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '[]')
                """,
                (
                    entry_id,
                    version_id,
                    source_id,
                    ordinal,
                    title,
                    content,
                    content_hash,
                    json.dumps(source_location, ensure_ascii=False),
                    parent_id,
                ),
            )
            connection.commit()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM text_entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
        return self._row_to_text_entry(row)

    def update_text_entry_kg_triple_ids(
        self,
        entry_id: str,
        triple_ids: list[str] | tuple[str, ...],
    ) -> TextEntry:
        """Persist the KG IDs expressed by one text retrieval chunk."""

        unique_ids: list[str] = []
        seen: set[str] = set()
        for triple_id in triple_ids:
            value = str(triple_id).strip()
            if value and value not in seen:
                unique_ids.append(value)
                seen.add(value)
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE text_entries
                SET kg_triple_ids_json = ?
                WHERE entry_id = ?
                """,
                (json.dumps(unique_ids, ensure_ascii=False), entry_id),
            )
            if cursor.rowcount != 1:
                raise LookupError(entry_id)
            connection.commit()
        return self.get_text_entry(entry_id)

    def list_text_entries(self, version_id: str) -> list[TextEntry]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM text_entries
                WHERE version_id = ? ORDER BY source_id, ordinal
                """,
                (version_id,),
            ).fetchall()
        return [self._row_to_text_entry(row) for row in rows]

    def get_text_entry(self, entry_id: str) -> TextEntry:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM text_entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
        if row is None:
            raise LookupError(entry_id)
        return self._row_to_text_entry(row)

    def upsert_vector_index(
        self,
        project_id: str,
        version_id: str,
        index_type: str,
        manifest: dict[str, object],
        *,
        status: str = "ready",
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, object]:
        """Register one rebuildable vector index in the metadata database."""

        normalized_type = index_type.strip()
        if not normalized_type:
            raise ValueError("向量索引类型不能为空")
        now = _now().isoformat()
        values = {
            "backend": str(manifest.get("backend") or "local_file"),
            "status": status,
            "collection_name": (
                str(manifest.get("collection_name"))
                if manifest.get("collection_name")
                else None
            ),
            "storage_uri": str(manifest.get("storage_uri") or ""),
            "embedding_model": str(manifest.get("embedding_model") or ""),
            "embedding_input_rule": str(manifest.get("embedding_input_rule") or ""),
            "dimension": int(manifest.get("dimension") or 0),
            "metric": str(manifest.get("metric") or "cosine"),
            "vector_count": int(manifest.get("vector_count") or 0),
            "content_fingerprint": str(manifest.get("content_fingerprint") or ""),
            "config_json": json.dumps(manifest, ensure_ascii=False),
            "error_code": error_code,
            "error_message": error_message,
        }
        with closing(self._connect()) as connection:
            existing = connection.execute(
                """
                SELECT index_id FROM vector_indexes
                WHERE version_id = ? AND index_type = ?
                """,
                (version_id, normalized_type),
            ).fetchone()
            if existing is None:
                index_id = str(uuid4())
                connection.execute(
                    """
                    INSERT INTO vector_indexes(
                        index_id, project_id, version_id, index_type, backend, status,
                        collection_name, storage_uri, embedding_model,
                        embedding_input_rule, dimension, metric, vector_count,
                        content_fingerprint, config_json, error_code, error_message,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        index_id,
                        project_id,
                        version_id,
                        normalized_type,
                        values["backend"],
                        values["status"],
                        values["collection_name"],
                        values["storage_uri"],
                        values["embedding_model"],
                        values["embedding_input_rule"],
                        values["dimension"],
                        values["metric"],
                        values["vector_count"],
                        values["content_fingerprint"],
                        values["config_json"],
                        values["error_code"],
                        values["error_message"],
                        now,
                        now,
                    ),
                )
            else:
                index_id = str(existing["index_id"])
                connection.execute(
                    """
                    UPDATE vector_indexes
                    SET project_id = ?, backend = ?, status = ?, collection_name = ?,
                        storage_uri = ?, embedding_model = ?, embedding_input_rule = ?,
                        dimension = ?, metric = ?, vector_count = ?,
                        content_fingerprint = ?, config_json = ?, error_code = ?,
                        error_message = ?, updated_at = ?
                    WHERE index_id = ?
                    """,
                    (
                        project_id,
                        values["backend"],
                        values["status"],
                        values["collection_name"],
                        values["storage_uri"],
                        values["embedding_model"],
                        values["embedding_input_rule"],
                        values["dimension"],
                        values["metric"],
                        values["vector_count"],
                        values["content_fingerprint"],
                        values["config_json"],
                        values["error_code"],
                        values["error_message"],
                        now,
                        index_id,
                    ),
                )
            connection.commit()
        return self.get_vector_index(version_id, normalized_type) or {}

    def get_vector_index(
        self,
        version_id: str,
        index_type: str = "text",
    ) -> dict[str, object] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM vector_indexes
                WHERE version_id = ? AND index_type = ?
                """,
                (version_id, index_type),
            ).fetchone()
        return self._row_to_vector_index(row) if row is not None else None

    def list_vector_indexes(self, project_id: str) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM vector_indexes
                WHERE project_id = ? ORDER BY updated_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [self._row_to_vector_index(row) for row in rows]

    @staticmethod
    def _row_to_vector_index(row: sqlite3.Row) -> dict[str, object]:
        config: dict[str, object]
        try:
            parsed = json.loads(row["config_json"] or "{}")
            config = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            config = {}
        return {
            "index_id": row["index_id"],
            "project_id": row["project_id"],
            "version_id": row["version_id"],
            "index_type": row["index_type"],
            "backend": row["backend"],
            "status": row["status"],
            "collection_name": row["collection_name"],
            "storage_uri": row["storage_uri"],
            "embedding_model": row["embedding_model"],
            "embedding_input_rule": row["embedding_input_rule"],
            "dimension": row["dimension"],
            "metric": row["metric"],
            "vector_count": row["vector_count"],
            "content_fingerprint": row["content_fingerprint"],
            "config": config,
            "error_code": row["error_code"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def insert_kg_triple(
        self,
        version_id: str,
        source_id: str,
        source_entry_id: str | None,
        source_line: int | None,
        head: str,
        relation: str,
        tail: str,
        origin: str,
        evidence: str | None,
    ) -> tuple[KGTriple, bool, list[dict[str, str]]]:
        head_norm = self._normalize_value(head)
        relation_norm = self._normalize_value(relation)
        tail_norm = self._normalize_value(tail)
        triple_hash = hashlib.sha256(
            "\x1f".join((head_norm, relation_norm, tail_norm)).encode("utf-8")
        ).hexdigest()
        now = _now().isoformat()
        with closing(self._connect()) as connection:
            duplicate_row = connection.execute(
                """
                SELECT * FROM kg_triples
                WHERE version_id = ? AND triple_hash = ?
                """,
                (version_id, triple_hash),
            ).fetchone()
            if duplicate_row is not None:
                return self._row_to_triple(duplicate_row), True, []

            conflicts = connection.execute(
                """
                SELECT * FROM kg_triples
                WHERE version_id = ? AND (
                    (head_norm = ? AND relation_norm = ? AND tail_norm <> ?)
                    OR
                    (tail_norm = ? AND relation_norm = ? AND head_norm <> ?)
                )
                ORDER BY created_at
                """,
                (
                    version_id,
                    head_norm,
                    relation_norm,
                    tail_norm,
                    tail_norm,
                    relation_norm,
                    head_norm,
                ),
            ).fetchall()
            triple_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO kg_triples(
                    triple_id, version_id, source_id, source_entry_id, source_line,
                    head, relation, tail, head_norm, relation_norm, tail_norm,
                    origin, evidence, triple_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    triple_id,
                    version_id,
                    source_id,
                    source_entry_id,
                    source_line,
                    head,
                    relation,
                    tail,
                    head_norm,
                    relation_norm,
                    tail_norm,
                    origin,
                    evidence,
                    triple_hash,
                    now,
                ),
            )
            for entity_name in (head, tail):
                entity_norm = self._normalize_value(entity_name)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO kg_entities(
                        entity_id, version_id, normalized_name, display_name, origin
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(uuid4()), version_id, entity_norm, entity_name, origin),
                )

            conflict_payload: list[dict[str, str]] = []
            for other in conflicts:
                if other["head_norm"] == head_norm and other["relation_norm"] == relation_norm:
                    conflict_type = "same_head_relation"
                    anchor = head
                else:
                    conflict_type = "same_tail_relation"
                    anchor = tail
                connection.execute(
                    """
                    INSERT OR IGNORE INTO kg_conflicts(
                        conflict_id, version_id, triple_id, other_triple_id,
                        conflict_type, anchor, relation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        version_id,
                        triple_id,
                        other["triple_id"],
                        conflict_type,
                        anchor,
                        relation,
                        now,
                    ),
                )
                conflict_payload.append(
                    {
                        "conflict_type": conflict_type,
                        "triple_id": triple_id,
                        "other_triple_id": other["triple_id"],
                    }
                )
            connection.commit()
        return self.get_triple(triple_id), False, conflict_payload

    def get_triple(self, triple_id: str) -> KGTriple:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM kg_triples WHERE triple_id = ?", (triple_id,)
            ).fetchone()
        if row is None:
            raise LookupError(triple_id)
        return self._row_to_triple(row)

    def list_kg_triples(self, version_id: str) -> list[KGTriple]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM kg_triples WHERE version_id = ? ORDER BY created_at",
                (version_id,),
            ).fetchall()
        return [self._row_to_triple(row) for row in rows]

    def list_kg_conflicts(self, version_id: str) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT c.conflict_id, c.triple_id, c.other_triple_id,
                       c.conflict_type, c.anchor, c.relation, c.created_at,
                       t.head AS triple_head, t.tail AS triple_tail,
                       o.head AS other_head, o.tail AS other_tail
                FROM kg_conflicts c
                JOIN kg_triples t ON t.triple_id = c.triple_id
                JOIN kg_triples o ON o.triple_id = c.other_triple_id
                WHERE c.version_id = ? ORDER BY c.created_at
                """,
                (version_id,),
            ).fetchall()
        return [
            {
                "conflict_id": row["conflict_id"],
                "triple_id": row["triple_id"],
                "other_triple_id": row["other_triple_id"],
                "conflict_type": row["conflict_type"],
                "anchor": row["anchor"],
                "relation": row["relation"],
                "created_at": row["created_at"],
                "triple": {
                    "head": row["triple_head"],
                    "relation": row["relation"],
                    "tail": row["triple_tail"],
                },
                "other_triple": {
                    "head": row["other_head"],
                    "relation": row["relation"],
                    "tail": row["other_tail"],
                },
            }
            for row in rows
        ]

    def resolve_graph_entities(self, version_id: str, names: list[str]) -> dict[str, dict[str, str]]:
        """Read existing entity identities for a graph page; never merge or write."""
        normalized = list(dict.fromkeys(self._normalize_value(name) for name in names))
        if not normalized:
            return {}
        found: dict[str, dict[str, str]] = {}
        with closing(self._connect()) as connection:
            for start in range(0, len(normalized), 400):
                batch = normalized[start:start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT entity_id, normalized_name, display_name FROM kg_entities "
                    f"WHERE version_id = ? AND normalized_name IN ({placeholders})",
                    (version_id, *batch),
                ).fetchall()
                found.update({
                    row["normalized_name"]: {"id": row["entity_id"], "label": row["display_name"]}
                    for row in rows
                })
        return {name: found[self._normalize_value(name)] for name in names if self._normalize_value(name) in found}

    def version_counts(self, version_id: str) -> dict[str, int]:
        with closing(self._connect()) as connection:
            text_count = connection.execute(
                "SELECT COUNT(*) FROM text_entries WHERE version_id = ?", (version_id,)
            ).fetchone()[0]
            triple_count = connection.execute(
                "SELECT COUNT(*) FROM kg_triples WHERE version_id = ?", (version_id,)
            ).fetchone()[0]
            entity_count = connection.execute(
                "SELECT COUNT(*) FROM kg_entities WHERE version_id = ?", (version_id,)
            ).fetchone()[0]
            conflict_count = connection.execute(
                "SELECT COUNT(*) FROM kg_conflicts WHERE version_id = ?", (version_id,)
            ).fetchone()[0]
        return {
            "text_entries": text_count,
            "kg_triples": triple_count,
            "kg_entities": entity_count,
            "kg_conflicts": conflict_count,
        }

    @staticmethod
    def _row_to_query(row: sqlite3.Row) -> QueryRecord:
        return QueryRecord(
            query_id=row["query_id"],
            project_id=row["project_id"],
            version_id=row["version_id"],
            question=row["question"],
            trace_level=row["trace_level"],
            status=row["status"],
            current_step=row["current_step"],
            answer=row["answer"],
            warnings=tuple(json.loads(row["warnings_json"])),
            metadata=json.loads(row["metadata_json"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_query_event(row: sqlite3.Row) -> QueryEvent:
        return QueryEvent(
            query_id=row["query_id"],
            seq=row["seq"],
            step=row["step"],
            status=row["status"],
            message=row["message"],
            constraint_id=row["constraint_id"],
            variable_id=row["variable_id"],
            pass_index=row["pass_index"],
            branch_id=row["branch_id"],
            input_ref=row["input_ref"],
            output_ref=row["output_ref"],
            counts=json.loads(row["counts_json"]),
            metrics=json.loads(row["metrics_json"]),
            reason_code=row["reason_code"],
            warnings=tuple(json.loads(row["warnings_json"])),
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            elapsed_ms=row["elapsed_ms"],
        )

    def create_query(
        self,
        project_id: str,
        version_id: str,
        question: str,
        trace_level: str = "summary",
    ) -> QueryRecord:
        query_id = str(uuid4())
        now = _now().isoformat()
        with closing(self._connect()) as connection:
            project = connection.execute(
                "SELECT project_id FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if project is None:
                raise ProjectNotFoundError(project_id)
            version = connection.execute(
                """
                SELECT version_id FROM knowledge_base_versions
                WHERE version_id = ? AND project_id = ? AND status IN ('ready', 'graph_only')
                """,
                (version_id, project_id),
            ).fetchone()
            if version is None:
                raise VersionNotFoundError(version_id)
            connection.execute(
                """
                INSERT INTO queries(
                    query_id, project_id, version_id, question, trace_level,
                    status, current_step, answer, warnings_json, metadata_json,
                    error_code, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, '[]', '{}', NULL, NULL, ?, ?)
                """,
                (query_id, project_id, version_id, question, trace_level, now, now),
            )
            connection.commit()
        return self.get_query(query_id)

    def get_query(self, query_id: str) -> QueryRecord:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM queries WHERE query_id = ?", (query_id,)
            ).fetchone()
        if row is None:
            raise LookupError(query_id)
        return self._row_to_query(row)

    def list_queries(
        self,
        project_id: str,
        *,
        version_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[QueryRecord]:
        clauses = ["project_id = ?"]
        values: list[object] = [project_id]
        if version_id:
            clauses.append("version_id = ?")
            values.append(version_id)
        if status:
            clauses.append("status = ?")
            values.append(status)
        values.append(max(1, min(limit, 200)))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM queries WHERE {' AND '.join(clauses)} "
                "ORDER BY updated_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._row_to_query(row) for row in rows]

    def update_query(
        self,
        query_id: str,
        *,
        status: str | None = None,
        current_step: str | None = None,
        answer: str | None = None,
        warnings: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> QueryRecord:
        current = self.get_query(query_id)
        now = _now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE queries
                SET status = ?, current_step = ?, answer = ?, warnings_json = ?,
                    metadata_json = ?, error_code = ?, error_message = ?, updated_at = ?
                WHERE query_id = ?
                """,
                (
                    status if status is not None else current.status,
                    current_step if current_step is not None else current.current_step,
                    answer if answer is not None else current.answer,
                    json.dumps(warnings if warnings is not None else list(current.warnings), ensure_ascii=False),
                    json.dumps(metadata if metadata is not None else current.metadata, ensure_ascii=False),
                    error_code,
                    error_message,
                    now,
                    query_id,
                ),
            )
            connection.commit()
        return self.get_query(query_id)

    def append_query_event(
        self,
        query_id: str,
        *,
        step: str,
        status: str,
        message: str = "",
        constraint_id: str | None = None,
        variable_id: str | None = None,
        pass_index: int | None = None,
        branch_id: str | None = None,
        input_ref: str | None = None,
        output_ref: str | None = None,
        counts: dict[str, object] | None = None,
        metrics: dict[str, object] | None = None,
        reason_code: str | None = None,
        warnings: list[str] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        elapsed_ms: float | None = None,
    ) -> QueryEvent:
        start = started_at or _now()
        with closing(self._connect()) as connection:
            query = connection.execute(
                "SELECT query_id FROM queries WHERE query_id = ?", (query_id,)
            ).fetchone()
            if query is None:
                raise LookupError(query_id)
            seq = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM query_events WHERE query_id = ?",
                (query_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO query_events(
                    event_id, query_id, seq, step, status, message,
                    constraint_id, variable_id, pass_index, branch_id,
                    input_ref, output_ref, counts_json, metrics_json,
                    reason_code, warnings_json, started_at, finished_at, elapsed_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    query_id,
                    seq,
                    step,
                    status,
                    message,
                    constraint_id,
                    variable_id,
                    pass_index,
                    branch_id,
                    input_ref,
                    output_ref,
                    json.dumps(counts or {}, ensure_ascii=False),
                    json.dumps(metrics or {}, ensure_ascii=False),
                    reason_code,
                    json.dumps(warnings or [], ensure_ascii=False),
                    start.isoformat(),
                    finished_at.isoformat() if finished_at else None,
                    elapsed_ms,
                ),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM query_events WHERE query_id = ? AND seq = ?",
                (query_id, seq),
            ).fetchone()
        return self._row_to_query_event(row)

    def list_query_events(self, query_id: str, after_seq: int = 0) -> list[QueryEvent]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM query_events
                WHERE query_id = ? AND seq > ? ORDER BY seq
                """,
                (query_id, after_seq),
            ).fetchall()
        return [self._row_to_query_event(row) for row in rows]

    def upsert_task(self, record: dict[str, object]) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, task_type, status, progress, message,
                    payload_json, result_json, error_code, error_message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    task_type = excluded.task_type,
                    status = excluded.status,
                    progress = excluded.progress,
                    message = excluded.message,
                    payload_json = excluded.payload_json,
                    result_json = excluded.result_json,
                    error_code = excluded.error_code,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    record["task_id"],
                    record["task_type"],
                    record["status"],
                    record["progress"],
                    record["message"],
                    json.dumps(record.get("payload") or {}, ensure_ascii=False),
                    json.dumps(record["result"], ensure_ascii=False) if record.get("result") is not None else None,
                    record.get("error_code"),
                    record.get("error_message"),
                    record["created_at"],
                    record["updated_at"],
                ),
            )
            connection.commit()

    @staticmethod
    def _row_to_task_record(row: sqlite3.Row) -> dict[str, object]:
        return {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "status": row["status"],
            "progress": row["progress"],
            "message": row["message"],
            "payload": json.loads(row["payload_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error_code": row["error_code"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_task_record(self, task_id: str) -> dict[str, object] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._row_to_task_record(row) if row is not None else None

    def list_task_records(self, limit: int = 200) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_task_record(row) for row in rows]
