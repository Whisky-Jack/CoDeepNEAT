from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.main.generation import Generation

instance: Optional[Generation] = None
