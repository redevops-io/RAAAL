"""Life-event templates. The part that cannot be generated."""
from .base import (
    InputKind,
    MissionTemplate,
    TemplateAssumption,
    TemplateCitation,
    TemplateInput,
    TemplateLimitation,
)
from .rsu import (
    IMPLEMENTED as RSU_IMPLEMENTED,
    RSU_TEMPLATE,
    SUPPLEMENTAL_RATE,
    SUPPLEMENTAL_RATE_HIGH,
    SUPPLEMENTAL_THRESHOLD,
    disposition_program,
    grants_for,
    net_shares,
    next_open_session,
    withholding_for,
)

#: Every template, by name. A registry rather than an import list so the
#: verifier can walk them all.
TEMPLATES = {RSU_TEMPLATE.name: RSU_TEMPLATE}

__all__ = [
    "InputKind", "MissionTemplate", "RSU_IMPLEMENTED", "RSU_TEMPLATE",
    "SUPPLEMENTAL_RATE", "SUPPLEMENTAL_RATE_HIGH", "SUPPLEMENTAL_THRESHOLD",
    "TEMPLATES", "TemplateAssumption", "TemplateCitation", "TemplateInput",
    "TemplateLimitation", "disposition_program", "grants_for", "net_shares",
    "next_open_session", "withholding_for",
]
