import hashlib
import json
from typing import Any


def hash_config(config: Any) -> str:
    """
    Create a deterministic hash of a configuration object by first converting it to JSON.
    """
    serialized = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


