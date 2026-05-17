"""
server/persistence/database.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Thread-safe SQLite logger for every phase decision made by
                 timing_algo.py. Provides query methods for PFE report graphs.

Design
    - Class-based; instantiated once by main.py at startup.
    - threading.Lock guards every write; SQLite opened with
      check_same_thread=False so the connection is shared safely.
    - Schema is created/migrated on connect(); idempotent — safe to call on an
      existing database.
    - insert_decision() accepts a PhaseDecision + frame_index directly.
    - insert_frame() wraps multiple lane decisions in one atomic transaction.
    - inf raw_duration (Webster oversaturation) is stored as NULL — SQLite has
      no IEEE 754 infinity; NULL is the correct sentinel for "unbounded".
    - Query methods return typed DecisionRecord dataclasses, not raw tuples.

Schema
    decisions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              REAL    NOT NULL,   -- Unix timestamp (time.time())
        frame_index     INTEGER NOT NULL,
        lane_id         TEXT    NOT NULL,
        vehicle_count   INTEGER NOT NULL,
        green_duration  REAL    NOT NULL,
        raw_duration    REAL,              -- NULL when oversaturated (inf)
        algorithm       TEXT    NOT NULL,
        was_clamped     INTEGER NOT NULL,  -- 0 / 1
        clamp_reason    TEXT    NOT NULL
    )
    schema_version (version INTEGER PRIMARY KEY)

Author : Oussama (server side)
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from server.logic.timing_algo import PhaseDecision

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class DatabaseError(RuntimeError):
    """Raised for unrecoverable database errors."""


class DatabaseNotConnectedError(DatabaseError):
    """Raised when a DB operation is attempted before connect()."""


@dataclass
class DecisionRecord:
    """A single row from the decisions table, with Python-native types."""
    id:             int
    ts:             float       # Unix timestamp
    frame_index:    int
    lane_id:        str
    vehicle_count:  int
    green_duration: float
    raw_duration:   Optional[float]   # None when oversaturated
    algorithm:      str
    was_clamped:    bool
    clamp_reason:   str


@dataclass
class LaneSummary:
    """Aggregate statistics for one lane, used for PFE graphs."""
    lane_id:            str
    total_decisions:    int
    avg_vehicle_count:  float
    avg_green_duration: float
    max_green_duration: float
    min_green_duration: float
    oversaturated_count: int   # decisions where clamp_reason == 'oversaturated'


# ──────────────────────────────────────────────────────────────────────────────
# Database class
# ──────────────────────────────────────────────────────────────────────────────

class Database:
    """
    Thread-safe SQLite logger for ATCS phase decisions.

    Usage
    -----
        db = Database("data/atcs.db")
        db.connect()
        db.insert_frame(timing_result.decisions, frame_index=42)
        records = db.query_lane("north", limit=100)
        db.close()
    """

    def __init__(self, db_path: str | Path = "data/atcs.db") -> None:
        """
        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite file. Parent directories are created on connect().
        """
        self._db_path  = Path(db_path)
        self._conn:    Optional[sqlite3.Connection] = None
        self._lock     = threading.Lock()
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Open the database, create parent directories, initialise the schema.

        Safe to call on an existing database — schema creation is idempotent.

        Raises
        ------
        DatabaseError
            If the directory cannot be created or the DB file cannot be opened.
        """
        if self._connected:
            log.warning("Database.connect() called again — already connected, skipping.")
            return

        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatabaseError(
                f"Cannot create database directory '{self._db_path.parent}': {exc}"
            ) from exc

        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,    # we guard with _lock ourselves
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")   # better concurrency
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._init_schema()
            self._connected = True
            log.info("Database connected: '%s'", self._db_path)

        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to open database '{self._db_path}': {exc}") from exc

    def close(self) -> None:
        """
        Flush and close the database connection.
        Safe to call even if not connected or already closed.
        """
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                log.warning("Error closing database: %s", exc)
            finally:
                self._conn = None
                self._connected = False
                log.info("Database closed.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ── Writes ────────────────────────────────────────────────────────────────

    def insert_decision(
        self,
        decision:    PhaseDecision,
        frame_index: int,
        ts:          Optional[float] = None,
    ) -> int:
        """
        Insert a single PhaseDecision row.

        Parameters
        ----------
        decision    : PhaseDecision  – output of timing_algo.compute()
        frame_index : int            – forwarded from SmoothedResult/TimingResult
        ts          : float | None   – Unix timestamp; defaults to time.time()

        Returns
        -------
        int  – rowid of the inserted row

        Raises
        ------
        DatabaseNotConnectedError  if connect() has not been called.
        DatabaseError              on SQLite write failure.
        """
        self._require_connected()
        self._validate_decision(decision, frame_index)

        ts = ts if ts is not None else time.time()
        raw_duration_db = (
            None if (decision.raw_duration is None or
                     not math.isfinite(decision.raw_duration))
            else decision.raw_duration
        )

        sql = """
            INSERT INTO decisions
                (ts, frame_index, lane_id, vehicle_count, green_duration,
                 raw_duration, algorithm, was_clamped, clamp_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            ts,
            frame_index,
            decision.lane_id,
            decision.vehicle_count,
            decision.green_duration,
            raw_duration_db,
            decision.algorithm.value,
            int(decision.was_clamped),
            decision.clamp_reason,
        )

        with self._lock:
            try:
                cursor = self._conn.execute(sql, params)
                self._conn.commit()
                rowid = cursor.lastrowid
                log.debug(
                    "Inserted decision rowid=%d lane=%s green=%.1fs frame=%d",
                    rowid, decision.lane_id, decision.green_duration, frame_index,
                )
                return rowid
            except sqlite3.Error as exc:
                raise DatabaseError(f"Insert failed: {exc}") from exc

    def insert_frame(
        self,
        decisions:   list[PhaseDecision],
        frame_index: int,
        ts:          Optional[float] = None,
    ) -> list[int]:
        """
        Insert multiple PhaseDecisions for the same frame atomically.

        All decisions share the same timestamp. On any failure the entire
        frame is rolled back — no partial writes.

        Returns
        -------
        list[int]  – rowids in the same order as decisions
        """
        self._require_connected()
        if not decisions:
            return []

        ts = ts if ts is not None else time.time()
        sql = """
            INSERT INTO decisions
                (ts, frame_index, lane_id, vehicle_count, green_duration,
                 raw_duration, algorithm, was_clamped, clamp_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        rows = []
        for d in decisions:
            self._validate_decision(d, frame_index)
            raw_db = (
                None if (d.raw_duration is None or not math.isfinite(d.raw_duration))
                else d.raw_duration
            )
            rows.append((
                ts, frame_index, d.lane_id, d.vehicle_count, d.green_duration,
                raw_db, d.algorithm.value, int(d.was_clamped), d.clamp_reason,
            ))

        with self._lock:
            try:
                rowids = []
                with self._conn:       # context manager → auto commit/rollback
                    for row in rows:
                        cur = self._conn.execute(sql, row)
                        rowids.append(cur.lastrowid)
                log.debug(
                    "Inserted frame=%d (%d decisions) rowids=%s",
                    frame_index, len(decisions), rowids,
                )
                return rowids
            except sqlite3.Error as exc:
                raise DatabaseError(f"Frame insert failed (rolled back): {exc}") from exc

    # ── Queries ───────────────────────────────────────────────────────────────

    def query_recent(self, limit: int = 100) -> list[DecisionRecord]:
        """Return the most recent `limit` decisions, newest first."""
        self._require_connected()
        sql = "SELECT * FROM decisions ORDER BY ts DESC LIMIT ?"
        return self._fetch_records(sql, (limit,))

    def query_lane(
        self,
        lane_id: str,
        limit:   int = 500,
    ) -> list[DecisionRecord]:
        """Return decisions for a specific lane, newest first."""
        self._require_connected()
        sql = "SELECT * FROM decisions WHERE lane_id = ? ORDER BY ts DESC LIMIT ?"
        return self._fetch_records(sql, (lane_id, limit))

    def query_time_range(
        self,
        start_ts: float,
        end_ts:   float,
        lane_id:  Optional[str] = None,
    ) -> list[DecisionRecord]:
        """
        Return decisions within [start_ts, end_ts] (Unix timestamps).
        Optionally filter by lane_id.

        Raises
        ------
        DatabaseError  if start_ts > end_ts.
        """
        self._require_connected()
        if start_ts > end_ts:
            raise DatabaseError(
                f"start_ts ({start_ts}) must be <= end_ts ({end_ts})."
            )

        if lane_id is not None:
            sql = ("SELECT * FROM decisions "
                   "WHERE ts BETWEEN ? AND ? AND lane_id = ? ORDER BY ts ASC")
            params = (start_ts, end_ts, lane_id)
        else:
            sql = "SELECT * FROM decisions WHERE ts BETWEEN ? AND ? ORDER BY ts ASC"
            params = (start_ts, end_ts)

        return self._fetch_records(sql, params)

    def lane_summary(self, lane_id: str) -> Optional[LaneSummary]:
        """
        Aggregate stats for one lane — used to generate PFE report graphs.
        Returns None if the lane has no recorded decisions.
        """
        self._require_connected()
        sql = """
            SELECT
                COUNT(*)                            AS total_decisions,
                AVG(vehicle_count)                  AS avg_vehicle_count,
                AVG(green_duration)                 AS avg_green_duration,
                MAX(green_duration)                 AS max_green_duration,
                MIN(green_duration)                 AS min_green_duration,
                SUM(CASE WHEN clamp_reason = 'oversaturated' THEN 1 ELSE 0 END)
                                                    AS oversaturated_count
            FROM decisions
            WHERE lane_id = ?
        """
        with self._lock:
            cur = self._conn.execute(sql, (lane_id,))
            row = cur.fetchone()

        if row is None or row[0] == 0:
            return None

        return LaneSummary(
            lane_id=lane_id,
            total_decisions=int(row[0]),
            avg_vehicle_count=float(row[1]),
            avg_green_duration=float(row[2]),
            max_green_duration=float(row[3]),
            min_green_duration=float(row[4]),
            oversaturated_count=int(row[5]),
        )

    def all_lane_ids(self) -> list[str]:
        """Return all distinct lane IDs that have been logged."""
        self._require_connected()
        with self._lock:
            cur = self._conn.execute(
                "SELECT DISTINCT lane_id FROM decisions ORDER BY lane_id"
            )
            return [row[0] for row in cur.fetchall()]

    def count_decisions(self) -> int:
        """Total number of rows in the decisions table."""
        self._require_connected()
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM decisions")
            return cur.fetchone()[0]

    # ── Private ───────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create tables if they don't exist and stamp the schema version."""
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              REAL    NOT NULL,
                    frame_index     INTEGER NOT NULL,
                    lane_id         TEXT    NOT NULL,
                    vehicle_count   INTEGER NOT NULL,
                    green_duration  REAL    NOT NULL,
                    raw_duration    REAL,
                    algorithm       TEXT    NOT NULL,
                    was_clamped     INTEGER NOT NULL,
                    clamp_reason    TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_lane_ts
                    ON decisions(lane_id, ts);
                CREATE INDEX IF NOT EXISTS idx_ts
                    ON decisions(ts);
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );
                INSERT OR IGNORE INTO schema_version (version) VALUES (1);
            """)
            self._conn.commit()
        log.debug("Schema initialised (version=%d).", SCHEMA_VERSION)

    def _require_connected(self) -> None:
        if not self._connected or self._conn is None:
            raise DatabaseNotConnectedError(
                "Database is not connected. Call db.connect() before using it."
            )

    def _fetch_records(
        self, sql: str, params: tuple
    ) -> list[DecisionRecord]:
        """Execute a SELECT and return typed DecisionRecord objects."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: tuple) -> DecisionRecord:
        """Map a raw SQLite row tuple to a DecisionRecord dataclass."""
        (id_, ts, frame_index, lane_id, vehicle_count, green_duration,
         raw_duration, algorithm, was_clamped, clamp_reason) = row
        return DecisionRecord(
            id=int(id_),
            ts=float(ts),
            frame_index=int(frame_index),
            lane_id=str(lane_id),
            vehicle_count=int(vehicle_count),
            green_duration=float(green_duration),
            raw_duration=float(raw_duration) if raw_duration is not None else None,
            algorithm=str(algorithm),
            was_clamped=bool(was_clamped),
            clamp_reason=str(clamp_reason),
        )

    @staticmethod
    def _validate_decision(decision: object, frame_index: object) -> None:
        if not isinstance(decision, PhaseDecision):
            raise DatabaseError(
                f"Expected PhaseDecision, got {type(decision).__name__}."
            )
        if not isinstance(frame_index, int) or frame_index < 0:
            raise DatabaseError(
                f"frame_index must be a non-negative int, got {frame_index!r}."
            )