"""
server/persistence/database.py
─────────────────────────────────────────────────────────────────────────────
Responsibility : Thread-safe SQLite logger for every phase decision made by
                 timing_algo.py. Provides the raw data for PFE report graphs.

Design
    - Database is a class; instantiated and opened once by main.py.
    - threading.Lock guards every write so the inference loop and any future
      background thread can both call log_decision() safely.
    - SQLite is opened with check_same_thread=False (lock is ours, not SQLite's).
    - Schema is created on open() if the table does not yet exist — idempotent.
    - db_path=':memory:' is fully supported for testing.

Schema  (table: events)
    id            INTEGER  PRIMARY KEY AUTOINCREMENT
    ts            REAL     Unix timestamp (time.time())
    lane_id       TEXT     Lane identifier
    vehicle_count INTEGER  Smoothed count that drove this decision
    green_duration REAL    Final clamped green duration in seconds
    raw_duration  REAL     Pre-clamp computed duration
    algorithm     TEXT     'webster' or 'linear'
    was_clamped   INTEGER  0 or 1 (SQLite has no BOOLEAN)
    clamp_reason  TEXT     'min', 'max', 'oversaturated', or ''
    frame_index   INTEGER  Frame counter from SmoothedResult

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

from server.logic.timing_algo import GreenDecision

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL    NOT NULL,
    lane_id        TEXT    NOT NULL,
    vehicle_count  INTEGER NOT NULL,
    green_duration REAL    NOT NULL,
    raw_duration   REAL    NOT NULL,
    algorithm      TEXT    NOT NULL,
    was_clamped    INTEGER NOT NULL,
    clamp_reason   TEXT    NOT NULL DEFAULT '',
    frame_index    INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts);
"""


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────

class DatabaseError(RuntimeError):
    """Raised for unrecoverable database failures."""


class DatabaseNotOpenError(DatabaseError):
    """Raised when a DB operation is attempted before open()."""


@dataclass
class EventRecord:
    """A single row read back from the events table."""
    id:             int
    ts:             float
    lane_id:        str
    vehicle_count:  int
    green_duration: float
    raw_duration:   float
    algorithm:      str
    was_clamped:    bool
    clamp_reason:   str
    frame_index:    int


# ──────────────────────────────────────────────────────────────────────────────
# Database class
# ──────────────────────────────────────────────────────────────────────────────

class Database:
    """
    Thread-safe SQLite logger for phase decisions.

    Usage
    -----
        db = Database(db_path="atcs.db")
        db.open()
        db.log_decision(decision, frame_index=1)
        records = db.fetch_recent(limit=100)
        db.close()
    """

    def __init__(self, db_path: str | Path = "atcs.db") -> None:
        """
        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite file. Use ':memory:' for in-memory (tests).
            Parent directory must already exist for file-based DBs.
        """
        self._db_path  = str(db_path)
        self._conn:    Optional[sqlite3.Connection] = None
        self._lock:    threading.Lock = threading.Lock()
        self._is_open: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open (or create) the SQLite database and ensure the schema exists.

        Raises
        ------
        DatabaseError
            If the parent directory does not exist, or SQLite fails to open.
        """
        if self._is_open:
            log.warning("Database.open() called again — already open, skipping.")
            return

        # Validate parent directory for file-based DBs
        if self._db_path != ":memory:":
            parent = Path(self._db_path).parent
            if not parent.exists():
                raise DatabaseError(
                    f"Cannot open database: parent directory '{parent}' does not exist."
                )

        try:
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,   # we manage thread-safety with our lock
            )
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.execute(_CREATE_TABLE)
                self._conn.execute(_CREATE_INDEX)
                self._conn.commit()
            self._is_open = True
            log.info("Database opened: '%s'.", self._db_path)

        except sqlite3.DatabaseError as exc:
            raise DatabaseError(
                f"Failed to open database '{self._db_path}': {exc}"
            ) from exc

    def close(self) -> None:
        """Close the database connection. Safe to call even if not open."""
        if not self._is_open or self._conn is None:
            log.debug("Database.close() called but DB is not open — no-op.")
            return
        with self._lock:
            self._conn.close()
            self._conn    = None
            self._is_open = False
        log.info("Database closed.")

    @property
    def is_open(self) -> bool:
        return self._is_open

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_decision(
        self,
        decision:    GreenDecision,
        frame_index: int = 0,
        ts:          Optional[float] = None,
    ) -> int:
        """
        Insert one GreenDecision into the events table.

        Parameters
        ----------
        decision    : GreenDecision  – output of TimingAlgo.compute() (per-lane)
        frame_index : int            – frame counter (forwarded from SmoothedResult)
        ts          : float | None   – Unix timestamp; defaults to time.time()

        Returns
        -------
        int  – row id of the inserted record (useful for testing / tracing)

        Raises
        ------
        DatabaseNotOpenError  if open() has not been called.
        DatabaseError         on any SQLite write failure.
        """
        self._require_open("log_decision")
        self._validate_decision(decision)

        if not isinstance(frame_index, int) or frame_index < 0:
            raise DatabaseError(
                f"frame_index must be a non-negative int, got {frame_index!r}."
            )

        timestamp = ts if ts is not None else time.time()

        # Serialise raw_duration — store inf as a sentinel (-1.0)
        raw = decision.raw_duration
        raw_stored = -1.0 if (math.isinf(raw) or math.isnan(raw)) else float(raw)

        row = (
            timestamp,
            decision.lane_id,
            decision.vehicle_count,
            decision.green_duration,
            raw_stored,
            decision.algorithm.value,
            int(decision.was_clamped),
            decision.clamp_reason,
            frame_index,
        )

        try:
            with self._lock:
                cur = self._conn.execute(
                    """
                    INSERT INTO events
                        (ts, lane_id, vehicle_count, green_duration,
                         raw_duration, algorithm, was_clamped, clamp_reason,
                         frame_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                self._conn.commit()
                row_id = cur.lastrowid
                log.debug(
                    "Logged decision: lane=%s count=%d green=%.1fs (row %d)",
                    decision.lane_id, decision.vehicle_count,
                    decision.green_duration, row_id,
                )
                return row_id

        except sqlite3.DatabaseError as exc:
            raise DatabaseError(f"Failed to write event: {exc}") from exc

    # ── Read ──────────────────────────────────────────────────────────────────

    def fetch_recent(self, limit: int = 100) -> list[EventRecord]:
        """
        Return the most recent `limit` events, newest first.

        Returns an empty list if the table is empty (never raises on empty).

        Raises
        ------
        DatabaseNotOpenError  if open() has not been called.
        DatabaseError         on SQLite read failure.
        """
        self._require_open("fetch_recent")

        if not isinstance(limit, int) or limit < 1:
            raise DatabaseError(f"limit must be a positive int, got {limit!r}.")

        try:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
                )
                rows = cur.fetchall()
            return [self._row_to_record(r) for r in rows]

        except sqlite3.DatabaseError as exc:
            raise DatabaseError(f"Failed to fetch events: {exc}") from exc

    def fetch_by_lane(self, lane_id: str, limit: int = 100) -> list[EventRecord]:
        """
        Return the most recent events for a specific lane, newest first.

        Raises
        ------
        DatabaseNotOpenError  if open() has not been called.
        DatabaseError         on SQLite read failure.
        """
        self._require_open("fetch_by_lane")

        if not isinstance(lane_id, str) or not lane_id:
            raise DatabaseError("lane_id must be a non-empty string.")

        if not isinstance(limit, int) or limit < 1:
            raise DatabaseError(f"limit must be a positive int, got {limit!r}.")

        try:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT * FROM events WHERE lane_id = ? ORDER BY ts DESC LIMIT ?",
                    (lane_id, limit),
                )
                rows = cur.fetchall()
            return [self._row_to_record(r) for r in rows]

        except sqlite3.DatabaseError as exc:
            raise DatabaseError(f"Failed to fetch events for lane '{lane_id}': {exc}") from exc

    def count_events(self) -> int:
        """Return total number of logged events."""
        self._require_open("count_events")
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM events")
            return cur.fetchone()[0]

    def fetch_lane_summary(self) -> dict[str, dict]:
        """
        Return aggregate stats per lane — useful for PFE report graphs.

        Returns
        -------
        dict[lane_id, {count: int, avg_vehicles: float, avg_green: float}]
        """
        self._require_open("fetch_lane_summary")
        try:
            with self._lock:
                cur = self._conn.execute(
                    """
                    SELECT
                        lane_id,
                        COUNT(*)            AS event_count,
                        AVG(vehicle_count)  AS avg_vehicles,
                        AVG(green_duration) AS avg_green
                    FROM events
                    GROUP BY lane_id
                    """
                )
                rows = cur.fetchall()
            return {
                r["lane_id"]: {
                    "event_count":  r["event_count"],
                    "avg_vehicles": r["avg_vehicles"],
                    "avg_green":    r["avg_green"],
                }
                for r in rows
            }
        except sqlite3.DatabaseError as exc:
            raise DatabaseError(f"Failed to fetch lane summary: {exc}") from exc

    # ── Private helpers ────────────────────────────────────────────────────────

    def _require_open(self, method: str) -> None:
        if not self._is_open:
            raise DatabaseNotOpenError(
                f"Database.{method}() called before open(). "
                "Call db.open() at startup."
            )

    @staticmethod
    def _validate_decision(decision: object) -> None:
        if decision is None:
            raise DatabaseError("GreenDecision is None.")
        if not isinstance(decision, GreenDecision):
            raise DatabaseError(
                f"Expected GreenDecision, got {type(decision).__name__}."
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            id=             row["id"],
            ts=             row["ts"],
            lane_id=        row["lane_id"],
            vehicle_count=  row["vehicle_count"],
            green_duration= row["green_duration"],
            raw_duration=   row["raw_duration"],
            algorithm=      row["algorithm"],
            was_clamped=    bool(row["was_clamped"]),
            clamp_reason=   row["clamp_reason"],
            frame_index=    row["frame_index"],
        )