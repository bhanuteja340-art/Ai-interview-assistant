import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DB = CACHE_DIR / "chatbot_cache.db"

SCHEMA_VERSION = 3

CACHE_ENABLED = os.getenv("CHAT_CACHE_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

CACHE_TTL_DAYS = int(os.getenv("CHAT_CACHE_TTL_DAYS", "0"))

BUSY_TIMEOUT_MS = int(os.getenv("CHAT_CACHE_BUSY_TIMEOUT_MS", "15000"))


try:
    from src.generator import (
        INSUFFICIENT_INFO_MESSAGE,
        is_insufficient_info,
    )
except ImportError:
    INSUFFICIENT_INFO_MESSAGE = (
        "I don't have enough information in the knowledge base to answer that."
    )

    def is_insufficient_info(answer: str) -> bool:
        normalized = "".join(
            character
            for character in (answer or "").lower()
            if character.isalnum() or character.isspace()
        )

        target = "".join(
            character
            for character in INSUFFICIENT_INFO_MESSAGE.lower()
            if character.isalnum() or character.isspace()
        )

        return " ".join(normalized.split()) == " ".join(target.split())


def build_fingerprint(
    model: str | None = None,
    system_instruction: str | None = None,
    index_version: str | None = None,
) -> str:
    parts = {
        "model": model if model is not None else os.getenv(
            "GENERATION_MODEL",
            "unknown",
        ),
        "prompt": hashlib.sha256(
            (system_instruction or os.getenv("GENERATION_PROMPT_VERSION", ""))
            .encode("utf-8")
        ).hexdigest()[:16],
        "index": index_version if index_version is not None else os.getenv(
            "KB_INDEX_VERSION",
            "unknown",
        ),
        "schema": SCHEMA_VERSION,
    }

    payload = json.dumps(parts, sort_keys=True)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ChatCache:
    def __init__(
        self,
        db_path: Path = CACHE_DB,
        fingerprint: str | None = None,
        enabled: bool = CACHE_ENABLED,
        ttl_days: int = CACHE_TTL_DAYS,
    ):
        self.db_path = Path(db_path)
        self.enabled = enabled
        self.ttl_days = max(0, int(ttl_days))
        self.fingerprint = fingerprint or build_fingerprint()

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._create_table()

    @staticmethod
    def normalize_question(question: str) -> str:
        collapsed = " ".join((question or "").strip().lower().split())

        return collapsed.strip(" \"'?!.,;:")

    def _cache_key(self, question: str) -> str:
        normalized = self.normalize_question(question)

        payload = f"{self.fingerprint}\n{normalized}"

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    def _question_hash(self, question: str) -> str:
        return self._cache_key(question)

    def _connect(self):
        connection = sqlite3.connect(self.db_path)

        connection.execute(
            f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}"
        )

        return connection

    def _create_table(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]

            table_exists = connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'question_cache'
                """
            ).fetchone()

            if table_exists and version != SCHEMA_VERSION:
                connection.execute("DROP TABLE question_cache")

            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS question_cache (
                        question_hash TEXT PRIMARY KEY,
                        fingerprint TEXT NOT NULL,
                        normalized_question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        grounded_content TEXT,
                        retrieved_chunks TEXT NOT NULL DEFAULT '[]',
                        finish_reason TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_question_cache_fingerprint
                    ON question_cache (fingerprint)
                    """
                )

                connection.execute(
                    f"PRAGMA user_version = {SCHEMA_VERSION}"
                )

    def _is_expired(self, created_at: str) -> bool:
        if not self.ttl_days:
            return False

        try:
            created = datetime.fromisoformat(created_at)
        except (TypeError, ValueError):
            return True

        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - created

        return age > timedelta(days=self.ttl_days)

    def get(self, question: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        question_hash = self._cache_key(question)

        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                """
                SELECT
                    question_hash,
                    fingerprint,
                    normalized_question,
                    answer,
                    grounded_content,
                    retrieved_chunks,
                    finish_reason,
                    created_at
                FROM question_cache
                WHERE question_hash = ?
                """,
                (question_hash,),
            ).fetchone()

        if row is None:
            return None

        entry = dict(row)
        entry["retrieved_chunks"] = json.loads(
            entry["retrieved_chunks"] or "[]"
)

        if self._is_expired(entry["created_at"]):
            self.delete(question)
            return None

        return entry

    def set(
        self,
        question: str,
        answer: str,
        grounded_content: str = "",
        finish_reason: str | None = None,
        truncated: bool = False,
        retrieved_chunks: list[dict[str, Any]] | None = None,
    ) -> bool:
        if not self.enabled:
            return False

        if not question or not question.strip():
            return False

        if not answer or not answer.strip():
            return False

        if truncated or is_insufficient_info(answer):
            return False

        if finish_reason not in (
            None,
            "STOP",
            "FINISH_REASON_UNSPECIFIED",
        ):
            return False
        chunks_json = json.dumps(
        retrieved_chunks if retrieved_chunks is not None else [],
        ensure_ascii=False,
)
        normalized_question = self.normalize_question(question)
        question_hash = self._cache_key(question)
        created_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )

        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO question_cache (
                        question_hash,
                        fingerprint,
                        normalized_question,
                        answer,
                        grounded_content,
                        retrieved_chunks,
                        finish_reason,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(question_hash) DO UPDATE SET
                        answer = excluded.answer,
                        grounded_content = excluded.grounded_content,
                        retrieved_chunks = excluded.retrieved_chunks,
                        finish_reason = excluded.finish_reason,
                        created_at = excluded.created_at
                    """,
                    (
                        question_hash,
                        self.fingerprint,
                        normalized_question,
                        answer,
                        grounded_content,
                        chunks_json,
                        finish_reason,
                        created_at,
                    ),
                )

        return True

    def delete(self, question: str) -> int:
        question_hash = self._cache_key(question)

        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM question_cache WHERE question_hash = ?",
                    (question_hash,),
                )

        return cursor.rowcount

    def clear(self) -> int:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM question_cache"
                )

        return cursor.rowcount

    def clear_stale(self) -> int:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM question_cache WHERE fingerprint != ?",
                    (self.fingerprint,),
                )

        return cursor.rowcount

    def count(self, current_only: bool = False) -> int:
        with closing(self._connect()) as connection:
            if current_only:
                result = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM question_cache
                    WHERE fingerprint = ?
                    """,
                    (self.fingerprint,),
                ).fetchone()
            else:
                result = connection.execute(
                    "SELECT COUNT(*) FROM question_cache"
                ).fetchone()

        return int(result[0])