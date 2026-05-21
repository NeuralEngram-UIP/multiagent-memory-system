# episodic_memory.py

"""
FINAL HARDENED
ENTERPRISE-GRADE
MULTI-AGENT EPISODIC MEMORY STORE

Features
────────
- SQLite persistence
- Ebbinghaus forgetting
- adaptive reinforcement
- shared/private memories
- cross-agent consensus
- replay prioritisation
- production-safe caching
- thread-safe connection pooling
- semantic retrieval support
- importance scoring
- multi-agent cognition
"""

import hashlib
import json
import logging
import math
import sqlite3
import threading
import uuid

from collections import OrderedDict

from dataclasses import (
    dataclass,
    field
)

from datetime import (
    datetime,
    timezone
)

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional
)

import anthropic

from sentence_transformers import (
    SentenceTransformer
)

from memory.ebbinghaus import (
    DEFAULT_STABILITY_HOURS,
    compute_retention,
    is_memory_forgotten,
    reinforce_memory,
    time_until_forgotten,
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Lazy Dependency Loading
# ─────────────────────────────────────────────────────────────

_client = None

_model = None

_client_lock = threading.Lock()

_model_lock = threading.Lock()


def get_client():

    global _client

    if _client is None:

        with _client_lock:

            if _client is None:

                _client = anthropic.Anthropic()

    return _client


def get_embedding_model():

    global _model

    if _model is None:

        with _model_lock:

            if _model is None:

                _model = SentenceTransformer(
                    "all-MiniLM-L6-v2"
                )

    return _model


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.20

MIN_RESULTS_REQUIRED = 1

DEFAULT_LIMIT = 100

REPLAY_HORIZON_HOURS = 6

DB_PATH = "multi_agent_memory.db"


# shared memory weights

W_ACCESS_FREQ = 0.15

W_TASK_IMPORTANCE = 0.18

W_AGENT_FEEDBACK = 0.32

W_TASK_SUCCESS = 0.27

W_CONSENSUS = 0.08


assert 0.0 < W_CONSENSUS < 1.0

assert abs(
    W_ACCESS_FREQ +
    W_TASK_IMPORTANCE +
    W_AGENT_FEEDBACK +
    W_TASK_SUCCESS +
    W_CONSENSUS -
    1.0
) < 1e-9, (
    "shared weights must sum to 1.0"
)


# private memory weights

_PRIVATE_TOTAL = (
    1.0 - W_CONSENSUS
)

W_ACCESS_FREQ_P = (
    W_ACCESS_FREQ / _PRIVATE_TOTAL
)

W_TASK_IMPORTANCE_P = (
    W_TASK_IMPORTANCE / _PRIVATE_TOTAL
)

W_AGENT_FEEDBACK_P = (
    W_AGENT_FEEDBACK / _PRIVATE_TOTAL
)

W_TASK_SUCCESS_P = (
    W_TASK_SUCCESS / _PRIVATE_TOTAL
)


# ─────────────────────────────────────────────────────────────
# Thread-Local SQLite Connections
# ─────────────────────────────────────────────────────────────

_thread_local = threading.local()


def get_connection(
    db_path: str = DB_PATH
) -> sqlite3.Connection:
    """
    Per-thread SQLite connection.

    Connections are isolated:
    - per thread
    - per db_path
    """

    attr = (
        f"connection_{db_path}"
        .replace("/", "_")
        .replace(".", "_")
    )

    conn = getattr(
        _thread_local,
        attr,
        None
    )

    if conn is None:

        conn = sqlite3.connect(
            db_path,
            check_same_thread=False
        )

        conn.execute(
            "PRAGMA journal_mode=WAL;"
        )

        conn.execute(
            "PRAGMA synchronous=NORMAL;"
        )

        setattr(
            _thread_local,
            attr,
            conn
        )

    return conn


# ─────────────────────────────────────────────────────────────
# Importance Cache
# ─────────────────────────────────────────────────────────────

_importance_cache = OrderedDict()

_importance_cache_lock = (
    threading.Lock()
)

MAX_IMPORTANCE_CACHE = 5000


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class RecallStatus(Enum):

    UPDATED = "updated"

    FORGOTTEN = "forgotten"

    NOT_FOUND = "not_found"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def safe_json_dumps(
    obj: Any
) -> str:

    try:

        return json.dumps(obj)

    except (
        TypeError,
        ValueError
    ):

        return json.dumps(
            str(obj)
        )


def safe_json_loads(
    s: Any
) -> Any:

    try:

        return json.loads(s)

    except (
        TypeError,
        ValueError,
        json.JSONDecodeError
    ):

        return s


def simple_embedding(
    text: str
) -> List[float]:

    model = get_embedding_model()

    return model.encode(
        text
    ).tolist()


def cosine_similarity(
    a: List[float],
    b: List[float]
) -> float:

    if not a or not b:
        return 0.0

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    na = math.sqrt(
        sum(x * x for x in a)
    )

    nb = math.sqrt(
        sum(x * x for x in b)
    )

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


# ─────────────────────────────────────────────────────────────
# Importance Scoring
# ─────────────────────────────────────────────────────────────

def compute_importance(
    text: str,
    agent_id: str
) -> float:

    cache_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]

    cache_key = (
        f"{agent_id}:{cache_hash}"
    )

    with _importance_cache_lock:

        cached = (
            _importance_cache.get(
                cache_key
            )
        )

        if cached is not None:

            # maintain LRU ordering

            _importance_cache.move_to_end(
                cache_key
            )

            return cached

    # ─────────────────────────────────────────
    # Fast Heuristic Paths
    # ─────────────────────────────────────────
    #
    # Heuristic scores are intentionally
    # cached to avoid repeated evaluation
    # for ultra-common short messages.
    #

    if len(text.split()) <= 3:

        score = 0.2

    elif text.strip().endswith("?"):

        score = 0.15

    else:

        # ─────────────────────────────────────
        # LLM Importance Scoring
        # ─────────────────────────────────────

        try:

            client = get_client()

            resp = client.messages.create(
                model=(
                    "claude-haiku-4-5-20251001"
                ),

                max_tokens=5,

                timeout=5.0,

                system=(
                    f"You are agent "
                    f"'{agent_id}'. "
                    "Rate importance "
                    "from 0.0 to 1.0. "
                    "Only return a number."
                ),

                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )

            score = float(
                resp.content[0]
                .text
                .strip()
            )

            score = max(
                0.0,
                min(score, 1.0)
            )

        except Exception as exc:

            logger.warning(
                "compute_importance failed "
                "for agent '%s': %s",
                agent_id,
                exc
            )

            score = _heuristic_importance(
                text
            )

    # ─────────────────────────────────────────
    # Cache Writeback
    # ─────────────────────────────────────────

    with _importance_cache_lock:

        # bounded cache growth
        #
        # eviction happens BEFORE insert
        # to maintain strict upper bounds

        if len(_importance_cache) >= (
            MAX_IMPORTANCE_CACHE
        ):

            logger.warning(
                "importance cache full "
                "evicting oldest entries "
                "cache_size=%d",
                len(_importance_cache)
            )

            eviction_count = int(
                MAX_IMPORTANCE_CACHE * 0.10
            )

            for _ in range(eviction_count):

                _importance_cache.popitem(
                    last=False
                )

        _importance_cache[
            cache_key
        ] = score

        # maintain LRU ordering

        _importance_cache.move_to_end(
            cache_key
        )

    return score


def _heuristic_importance(
    text: str
) -> float:

    text_lower = text.lower()

    high = [
        "i am",
        "i'm",
        "i love",
        "i hate",
        "my goal",
        "i need"
    ]

    low = [
        "okay",
        "thanks",
        "hello",
        "hi"
    ]

    if any(
        s in text_lower
        for s in high
    ):
        return 0.75

    if any(
        s in text_lower
        for s in low
    ):
        return 0.15

    return 0.45


def should_store_memory(
    text: str,
    importance: float
) -> bool:
    """
    Storage thresholds:

    - all memories require:
      importance >= 0.15

    - ultra-short memories
      require:
      importance >= 0.20
    """

    if importance < 0.15:
        return False

    if (
        len(text.split()) <= 2
        and importance < 0.20
    ):
        return False

    return True


# ─────────────────────────────────────────────────────────────
# Episode
# ─────────────────────────────────────────────────────────────

@dataclass
class Episode:

    content: Any

    context: Dict[str, Any]

    agent_id: str

    tags: List[str] = field(
        default_factory=list
    )

    shared: bool = False

    stability_hours: float = (
        DEFAULT_STABILITY_HOURS
    )

    last_reviewed_at: datetime = field(
        default_factory=lambda:
        datetime.now(timezone.utc)
    )

    review_count: int = 0

    cross_agent_recall_count: int = 0

    episode_id: str = field(
        default_factory=lambda:
        str(uuid.uuid4())
    )

    created_at: datetime = field(
        default_factory=lambda:
        datetime.now(timezone.utc)
    )

    importance: float = 0.6

    agent_feedback: float = 0.5

    task_success_rate: float = 0.5

    def retention(
        self,
        now: Optional[datetime] = None
    ) -> float:

        return compute_retention(
            self.last_reviewed_at,
            self.stability_hours,
            now
        )

    def is_forgotten(
        self,
        now: Optional[datetime] = None
    ) -> bool:

        return is_memory_forgotten(
            self.retention(now)
        )

    def access_frequency_score(
        self
    ) -> float:

        return min(
            math.log1p(
                self.review_count
            ) / math.log1p(100),
            1.0
        )

    def consensus_score(
        self
    ) -> float:

        if not self.shared:
            return 0.0

        return min(
            math.log1p(
                self.cross_agent_recall_count
            ) / math.log1p(20),
            1.0
        )

    def priority_score(
        self,
        now: Optional[datetime] = None
    ) -> float:
        """
        Replay priority score.

        Retention acts as a global
        decay multiplier.
        """

        af = self.access_frequency_score()

        ti = max(
            0.0,
            min(self.importance, 1.0)
        )

        fb = max(
            0.0,
            min(self.agent_feedback, 1.0)
        )

        ts = max(
            0.0,
            min(self.task_success_rate, 1.0)
        )

        if self.shared:

            cs = self.consensus_score()

            weighted = (
                W_ACCESS_FREQ * af +
                W_TASK_IMPORTANCE * ti +
                W_AGENT_FEEDBACK * fb +
                W_TASK_SUCCESS * ts +
                W_CONSENSUS * cs
            )

        else:

            weighted = (
                W_ACCESS_FREQ_P * af +
                W_TASK_IMPORTANCE_P * ti +
                W_AGENT_FEEDBACK_P * fb +
                W_TASK_SUCCESS_P * ts
            )

        return weighted * self.retention(now)
    

# ─────────────────────────────────────────────────────────────
# EpisodicMemoryStore — compatibility shim
# ─────────────────────────────────────────────────────────────

# class EpisodicMemoryStore:
#     """
#     Thin store wrapper around Episode.
#     Satisfies orchestrator import.
#     """

#     def __init__(self):
#         self._episodes: List[Episode] = []

#     def add(self, episode: Episode) -> None:
#         self._episodes.append(episode)

#     def get_all(self) -> List[Episode]:
#         return list(self._episodes)

#     def get_by_id(
#         self,
#         episode_id: str
#     ) -> Optional[Episode]:
#         for ep in self._episodes:
#             if ep.episode_id == episode_id:
#                 return ep
#         return None

#     def remove(
#         self,
#         episode_id: str
#     ) -> bool:
#         before = len(self._episodes)
#         self._episodes = [
#             ep for ep in self._episodes
#             if ep.episode_id != episode_id
#         ]
#         return len(self._episodes) < before
    

# #-------------------------------------------------------------------------------------------------------------------------------------------
#     def apply_decay(self) -> None:
#         """
#         Remove forgotten episodes
#         based on Ebbinghaus retention.
#         """
#         self._episodes = [
#             ep for ep in self._episodes
#             if not ep.is_forgotten()
#         ]

#     def prune_forgotten(self) -> List[str]:
#         """
#         Remove forgotten episodes.
#         Returns list of pruned episode IDs.
#         """
#         forgotten = [
#             ep.episode_id
#             for ep in self._episodes
#             if ep.is_forgotten()
#         ]
#         self._episodes = [
#             ep for ep in self._episodes
#             if not ep.is_forgotten()
#         ]
#         return forgotten
    


#     def grounded_retrieve(
#         self,
#         query: str,
#         embedding: List[float],
#         top_k: int = 5,
#         agent_id: Optional[str] = None
#     ) -> List[Episode]:
#         """
#         Retrieve episodes by semantic similarity.
#         Filters by agent_id if provided.
#         Skips forgotten episodes.
#         """
#         candidates = [
#             ep for ep in self._episodes
#             if not ep.is_forgotten()
#             and (
#                 agent_id is None
#                 or ep.agent_id == agent_id
#                 or ep.shared
#             )
#         ]

#         if not candidates or not embedding:
#             return []

#         scored = []
#         for ep in candidates:
#             ep_embedding = simple_embedding(
#                 str(ep.content)
#             )
#             score = cosine_similarity(
#                 embedding,
#                 ep_embedding
#             )
#             if score >= SIMILARITY_THRESHOLD:
#                 scored.append((score, ep))

#         scored.sort(key=lambda x: x[0], reverse=True)

#         return [ep for _, ep in scored[:top_k]]


class EpisodicMemoryStore:
    """
    Thin store wrapper around Episode.
    Satisfies orchestrator import.
    """

    def __init__(self):
        self._episodes: List[Episode] = []

    def add(
        self,
        memory_id: str,
        agent_id: str,
        content: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        ep = Episode(
            episode_id=memory_id,
            agent_id=agent_id,
            content=content,
            context=context or {},
            shared=(context or {}).get("shared", False)
        )
        self._episodes.append(ep)
        return ep.episode_id

    def get_all(self) -> List[Episode]:
        return list(self._episodes)

    def get_by_id(
        self,
        episode_id: str
    ) -> Optional[Episode]:
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def remove(
        self,
        episode_id: str
    ) -> bool:
        before = len(self._episodes)
        self._episodes = [
            ep for ep in self._episodes
            if ep.episode_id != episode_id
        ]
        return len(self._episodes) < before

    def delete(
        self,
        episode_id: str
    ) -> bool:
        return self.remove(episode_id)

    def recall(
        self,
        episode_id: str
    ) -> Optional[Episode]:
        ep = self.get_by_id(episode_id)
        if ep:
            ep.review_count += 1
            ep.last_reviewed_at = datetime.now(timezone.utc)
            ep = reinforce_memory(ep)
        return ep

    def grounded_retrieve(
        self,
        query: str,
        embedding: List[float],
        top_k: int = 5,
        agent_id: Optional[str] = None
    ) -> List[Episode]:
        candidates = [
            ep for ep in self._episodes
            if not ep.is_forgotten()
            and (
                agent_id is None
                or ep.agent_id == agent_id
                or ep.shared
            )
        ]
        if not candidates or not embedding:
            return []
        scored = []
        for ep in candidates:
            ep_embedding = simple_embedding(str(ep.content))
            score = cosine_similarity(embedding, ep_embedding)
            if score >= SIMILARITY_THRESHOLD:
                scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def apply_decay(self) -> None:
        self._episodes = [
            ep for ep in self._episodes
            if not ep.is_forgotten()
        ]

    def prune_forgotten(self) -> List[str]:
        forgotten = [
            ep.episode_id
            for ep in self._episodes
            if ep.is_forgotten()
        ]
        self._episodes = [
            ep for ep in self._episodes
            if not ep.is_forgotten()
        ]
        return forgotten