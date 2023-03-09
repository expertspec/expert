from __future__ import annotations

from app.libs.expert.expert.core.confidence.liedet.models.registry import registry
from app.libs.expert.expert.core.confidence.liedet.models.heads.timesformer_head import TimeSformerHead


@registry.register_module()
class ASTHead(TimeSformerHead):
    """Similar to TimeSformerHead."""

    pass
