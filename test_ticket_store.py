"""Run with: python -m unittest test_ticket_store.py (from starter-repo).

Every storage operation uses a patched path inside a TemporaryDirectory.
The application database is never opened or created by these tests.
"""
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

import ticket_store


class TicketStoreTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.db_path = Path(temporary_directory.name) / "test_tickets.db"
        path_patch = patch.object(ticket_store, "DB_PATH", self.db_path)
        path_patch.start()
        # Cleanup is LIFO: restore the path before deleting the directory.
        self.addCleanup(path_patch.stop)
        ticket_store.init_db()

    def create_sample(self):
        return ticket_store.create_ticket(
            "Alex O'Neil", "Can't access shared files", "Insufficient context"
        )

    def test_database_initialization(self):
        self.assertTrue(self.db_path.is_file())
        self.assertEqual(ticket_store.list_tickets(), [])
        ticket_id = self.create_sample()
        ticket_store.init_db()  # Reinitializing must not drop existing tickets.
        self.assertIsNotNone(ticket_store.get_ticket(ticket_id))
        with closing(sqlite3.connect(self.db_path)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(tickets)")}
        self.assertEqual(columns, {
            "ticket_id", "customer_name", "issue_description",
            "escalation_reason", "created_at", "status",
        })

    def test_successful_ticket_creation_and_retrieval(self):
        ticket_id = self.create_sample()
        self.assertEqual(str(UUID(ticket_id)), ticket_id)
        ticket = ticket_store.get_ticket(ticket_id)
        self.assertEqual(ticket["ticket_id"], ticket_id)
        self.assertEqual(ticket["customer_name"], "Alex O'Neil")
        self.assertEqual(ticket["issue_description"], "Can't access shared files")
        self.assertEqual(ticket["escalation_reason"], "Insufficient context")
        timestamp = datetime.fromisoformat(ticket["created_at"])
        self.assertEqual(timestamp.utcoffset(), timedelta(0))

    def test_ticket_id_uniqueness_across_initializations(self):
        first = self.create_sample()
        ticket_store.init_db()
        second = self.create_sample()
        self.assertNotEqual(first, second)
        self.assertEqual(len(ticket_store.list_tickets()), 2)

    def test_initial_status_is_open(self):
        ticket_id = self.create_sample()
        self.assertEqual(ticket_store.get_ticket(ticket_id)["status"], "Open")

    def test_unknown_ticket_id(self):
        self.assertIsNone(ticket_store.get_ticket("missing-ticket"))
        self.assertIsNone(ticket_store.get_ticket("' OR 1=1 --"))

    def test_list_tickets_newest_first(self):
        # Control timestamps to avoid depending on clock resolution or sleeps.
        with patch.object(ticket_store, "datetime") as clock:
            clock.now.return_value = datetime.fromisoformat("2026-01-01T12:00:00+00:00")
            first = self.create_sample()
            clock.now.return_value = datetime.fromisoformat("2026-01-02T12:00:00+00:00")
            second = self.create_sample()
        tickets = ticket_store.list_tickets()
        self.assertEqual([ticket["ticket_id"] for ticket in tickets], [second, first])

    def test_persistence_across_separate_connections(self):
        ticket_id = self.create_sample()
        # This independent connection confirms that the insert was committed.
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT customer_name FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        self.assertEqual(row[0], "Alex O'Neil")
        self.assertIsNotNone(ticket_store.get_ticket(ticket_id))

    def test_update_open_to_in_progress(self):
        ticket_id = self.create_sample()
        self.assertTrue(ticket_store.update_ticket_status(ticket_id, "In Progress"))
        self.assertEqual(ticket_store.get_ticket(ticket_id)["status"], "In Progress")

    def test_update_in_progress_to_resolved(self):
        ticket_id = self.create_sample()
        ticket_store.update_ticket_status(ticket_id, "In Progress")
        self.assertTrue(ticket_store.update_ticket_status(ticket_id, "Resolved"))
        self.assertEqual(ticket_store.get_ticket(ticket_id)["status"], "Resolved")

    def test_invalid_status_is_rejected(self):
        ticket_id = self.create_sample()
        for status in ("Closed", "open", "resolved", " In Progress", "", None):
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    ticket_store.update_ticket_status(ticket_id, status)
                self.assertEqual(ticket_store.get_ticket(ticket_id)["status"], "Open")

    def test_update_nonexistent_ticket(self):
        ticket_id = self.create_sample()
        before = ticket_store.list_tickets()
        for unknown_id in ("missing-ticket", "' OR 1=1 --"):
            with self.subTest(ticket_id=unknown_id):
                self.assertFalse(ticket_store.update_ticket_status(unknown_id, "Resolved"))
        self.assertEqual(ticket_store.list_tickets(), before)
        self.assertEqual(ticket_store.get_ticket(ticket_id)["status"], "Open")

    def test_status_update_persists_across_connections(self):
        ticket_id = self.create_sample()
        ticket_store.update_ticket_status(ticket_id, "Resolved")
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT status FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        self.assertEqual(row[0], "Resolved")

    def test_status_update_preserves_other_fields_and_tickets(self):
        ticket_id = self.create_sample()
        other_id = self.create_sample()
        before = ticket_store.get_ticket(ticket_id)
        other_before = ticket_store.get_ticket(other_id)
        ticket_store.update_ticket_status(ticket_id, "In Progress")
        expected = {**before, "status": "In Progress"}
        self.assertEqual(ticket_store.get_ticket(ticket_id), expected)
        self.assertEqual(ticket_store.get_ticket(other_id), other_before)

    def test_reopen_and_same_status_update(self):
        ticket_id = self.create_sample()
        ticket_store.update_ticket_status(ticket_id, "Resolved")
        self.assertTrue(ticket_store.update_ticket_status(ticket_id, "Open"))
        self.assertTrue(ticket_store.update_ticket_status(ticket_id, "Open"))
        self.assertEqual(ticket_store.get_ticket(ticket_id)["status"], "Open")

    def test_status_update_database_error(self):
        with patch.object(ticket_store, "DB_PATH", self.db_path / "unavailable.db"):
            with self.assertRaises(ticket_store.TicketStoreError):
                ticket_store.update_ticket_status("any-ticket", "Resolved")

    def test_database_errors_have_safe_messages(self):
        with patch.object(ticket_store, "DB_PATH", self.db_path / "unavailable.db"):
            with self.assertRaises(ticket_store.TicketStoreError) as caught:
                ticket_store.init_db()
        self.assertEqual(
            str(caught.exception),
            "Ticket storage is unavailable. Please try again later.",
        )


if __name__ == "__main__":
    unittest.main()
