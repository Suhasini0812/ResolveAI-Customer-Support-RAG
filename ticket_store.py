"""Local support-ticket storage, independent of the UI and retrieval workflow.

Call init_db() before using the store. Importing this module creates no files.
All public operations raise TicketStoreError on SQLite failures; callers can
show its safe message without displaying database paths or SQL details.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DB_PATH = Path(__file__).resolve().parent / "resolveai_tickets.db"


class TicketStoreError(RuntimeError):
    """A storage failure with a message suitable for display to a user."""


@contextmanager
def _connection():
    """Commit or roll back each operation, and always close its connection."""
    try:
        with closing(sqlite3.connect(DB_PATH)) as connection:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
    except sqlite3.Error:
        raise TicketStoreError(
            "Ticket storage is unavailable. Please try again later."
        ) from None


def init_db() -> None:
    """Create the local database and tickets table if needed; preserve records."""
    with _connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY NOT NULL,
                customer_name TEXT NOT NULL,
                issue_description TEXT NOT NULL,
                escalation_reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Open'
            )
        """)


def create_ticket(
    customer_name: str, issue_description: str, escalation_reason: str
) -> str:
    """Persist a new Open ticket and return its UUID, independent of restarts.

    created_at is an ISO 8601 UTC timestamp with microseconds and a +00:00 offset.
    The primary key also enforces uniqueness at the database level.
    """
    ticket_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    with _connection() as connection:
        connection.execute(
            """INSERT INTO tickets
               (ticket_id, customer_name, issue_description, escalation_reason,
                created_at, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticket_id, customer_name, issue_description, escalation_reason,
             created_at, "Open"),
        )
    return ticket_id


def get_ticket(ticket_id: str) -> dict | None:
    """Return a ticket as a dictionary, or None when the ID is unknown."""
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def list_tickets() -> list[dict]:
    """Return all tickets newest first, with ID as a stable timestamp tie-breaker."""
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM tickets ORDER BY created_at DESC, ticket_id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def update_ticket_status(ticket_id: str, new_status: str) -> bool:
    """Update an existing ticket; return True if found, otherwise False.

    Only exact status names are accepted. Invalid values raise ValueError
    before opening a connection. Updating to the current status also succeeds.
    SQLite failures use the same TicketStoreError as other storage operations.
    """
    if new_status not in ("Open", "In Progress", "Resolved"):
        raise ValueError("Status must be Open, In Progress, or Resolved.")
    with _connection() as connection:
        cursor = connection.execute(
            "UPDATE tickets SET status = ? WHERE ticket_id = ?",
            (new_status, ticket_id),
        )
        updated = cursor.rowcount > 0
    return updated
