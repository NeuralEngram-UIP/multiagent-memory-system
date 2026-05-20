# working_memory.py

"""
Enterprise-Grade Working Memory

Short-term active conversational memory.

Responsibilities
────────────────
- maintain recent interaction context
- provide bounded memory window
- support prompt assembly
- remain lightweight and thread-safe
- provide snapshot-safe retrieval
- support observability metrics
- enforce memory constraints
- support graceful truncation

Architecture
────────────
Working memory is intentionally:
- lightweight
- bounded
- fast
- non-persistent

Long-term cognition belongs to:
- episodic memory
- semantic memory
"""

import copy
import logging
import threading

from collections import deque
from dataclasses import dataclass

from typing import (
    Dict,
    List,
    Optional
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

VALID_ROLES = {
    "user",
    "assistant",
    "system"
}

DEFAULT_CAPACITY = 20

MAX_CONTENT_LENGTH = 10000

TRUNCATION_SUFFIX = (
    "\n...[truncated]"
)


# ─────────────────────────────────────────────────────────────
# Working Memory Item
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkingMemoryItem:
    """
    Immutable working memory item.
    """

    role: str

    content: str


# ─────────────────────────────────────────────────────────────
# Working Memory
# ─────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Bounded conversational context.

    Stores only recent interactions
    required for active reasoning.

    Design Goals
    ────────────
    - bounded growth
    - thread safety
    - snapshot-safe retrieval
    - prompt-friendly formatting
    - observability support
    """

    def __init__(
        self,
        capacity: int = (
            DEFAULT_CAPACITY
        )
    ):

        if capacity <= 0:

            raise ValueError(
                "capacity must be > 0"
            )

        self.capacity = capacity

        self.buffer = deque(
            maxlen=capacity
        )

        # IMPORTANT:
        # RLock is intentionally used
        # because helper methods may
        # call other lock-acquiring
        # methods internally.
        #
        # Replacing this with Lock
        # may introduce deadlocks.

        self.lock = threading.RLock()

        # observability metrics

        self.total_adds = 0

        self.total_evictions = 0

        self.total_snapshots = 0

        self.total_clears = 0

        self.total_truncations = 0

        logger.info(
            "WorkingMemory initialized "
            "capacity=%s",
            capacity
        )

    # ─────────────────────────────────────────────────────────
    # Add
    # ─────────────────────────────────────────────────────────

    def add(
        self,
        role: str,
        content: str
    ):
        """
        Add working memory item.
        """

        role = role.strip()

        if role not in VALID_ROLES:

            raise ValueError(
                f"role must be one of "
                f"{VALID_ROLES}"
            )

        content = content.strip()

        if not content:

            raise ValueError(
                "content cannot be empty"
            )

        truncated = False

        # graceful truncation

        if len(content) > (
            MAX_CONTENT_LENGTH
        ):

            logger.warning(
                "working memory content "
                "truncated role=%s "
                "original_length=%s",
                role,
                len(content)
            )

            allowed_length = (
                MAX_CONTENT_LENGTH
                - len(
                    TRUNCATION_SUFFIX
                )
            )

            content = (
                content[
                    :allowed_length
                ]
                + TRUNCATION_SUFFIX
            )

            truncated = True

        item = WorkingMemoryItem(
            role=role,
            content=content
        )

        with self.lock:

            if truncated:

                self.total_truncations += 1

            # detect eviction

            if (
                len(self.buffer)
                == self.capacity
            ):

                evicted = self.buffer[0]

                self.total_evictions += 1

                logger.info(
                    "working memory eviction "
                    "role=%s "
                    "content_length=%s "
                    "preview=%s",
                    evicted.role,
                    len(
                        evicted.content
                    ),
                    evicted.content[:80]
                )

            self.buffer.append(item)

            self.total_adds += 1

            current_size = len(
                self.buffer
            )

        logger.debug(
            "working memory add "
            "role=%s "
            "size=%s",
            role,
            current_size
        )

    # ─────────────────────────────────────────────────────────
    # Recent
    # ─────────────────────────────────────────────────────────

    def recent(
        self,
        limit: int = 5,
        role_filter: Optional[
            str
        ] = None
    ) -> List[
        WorkingMemoryItem
    ]:
        """
        Retrieve recent working memory.

        Returned list is snapshot-safe
        and isolated from future writes.
        """

        if limit <= 0:

            raise ValueError(
                "limit must be > 0"
            )

        if (
            role_filter is not None
            and role_filter
            not in VALID_ROLES
        ):

            raise ValueError(
                f"invalid role_filter: "
                f"{role_filter}"
            )

        with self.lock:

            items = list(
                self.buffer
            )

        if role_filter:

            items = [
                item
                for item in items
                if item.role
                == role_filter
            ]

        return items[-limit:]

    # ─────────────────────────────────────────────────────────
    # Snapshot
    # ─────────────────────────────────────────────────────────

    def snapshot(
        self
    ) -> List[
        WorkingMemoryItem
    ]:
        """
        Return deep-copy snapshot of
        working memory buffer.

        Snapshot is fully isolated
        from future mutations.
        """

        with self.lock:

            snapshot = copy.deepcopy(
                list(self.buffer)
            )

            self.total_snapshots += 1

        logger.debug(
            "working memory snapshot "
            "size=%s",
            len(snapshot)
        )

        return snapshot

    # ─────────────────────────────────────────────────────────
    # Chat Messages
    # ─────────────────────────────────────────────────────────

    def as_messages(
        self,
        limit: int = 10,
        role_filter: Optional[
            str
        ] = None,

        # accepted for orchestrator
        # compatibility

        agent_id: Optional[
            str
        ] = None
    ) -> List[
        Dict[str, str]
    ]:
        """
        Format working memory into
        chat-compatible messages.

        agent_id is currently unused
        but accepted for compatibility
        with orchestrator APIs.
        """

        messages = self.recent(
            limit=limit,
            role_filter=role_filter
        )

        return [
            {
                "role": item.role,
                "content": item.content
            }
            for item in messages
        ]

    # ─────────────────────────────────────────────────────────
    # Prompt Formatting
    # ─────────────────────────────────────────────────────────

    def format_for_prompt(
        self,
        limit: int = 10,
        role_filter: Optional[
            str
        ] = None
    ) -> str:
        """
        Format working memory
        into prompt-compatible text.

        Supports optional role filtering.
        """

        messages = self.recent(
            limit=limit,
            role_filter=role_filter
        )

        formatted = []

        for item in messages:

            formatted.append(
                f"{item.role}: "
                f"{item.content}"
            )

        return "\n".join(
            formatted
        )

    # ─────────────────────────────────────────────────────────
    # Clear
    # ─────────────────────────────────────────────────────────

    def clear(
        self
    ):
        """
        Clear working memory buffer.
        """

        with self.lock:

            self.buffer.clear()

            self.total_clears += 1

        logger.info(
            "working memory cleared"
        )

    # ─────────────────────────────────────────────────────────
    # Size
    # ─────────────────────────────────────────────────────────

    def size(
        self
    ) -> int:
        """
        Current working memory size.
        """

        with self.lock:

            return len(
                self.buffer
            )

    # ─────────────────────────────────────────────────────────
    # Empty Check
    # ─────────────────────────────────────────────────────────

    def is_empty(
        self
    ) -> bool:
        """
        Check whether working memory
        is empty.
        """

        return self.size() == 0

    # ─────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────

    def metrics(
        self
    ) -> Dict[str, int]:
        """
        Working memory observability.

        These metrics can later be exported to:
        - Prometheus
        - Grafana
        - OpenTelemetry
        """

        with self.lock:

            return {
                "capacity": (
                    self.capacity
                ),

                "current_size": (
                    len(self.buffer)
                ),

                "total_adds": (
                    self.total_adds
                ),

                "total_evictions": (
                    self.total_evictions
                ),

                "total_snapshots": (
                    self.total_snapshots
                ),

                "total_truncations": (
                    self.total_truncations
                ),

                "total_clears": (
                    self.total_clears
                )
            }

    # ─────────────────────────────────────────────────────────
    # Length
    # ─────────────────────────────────────────────────────────

    def __len__(
        self
    ) -> int:

        return self.size()