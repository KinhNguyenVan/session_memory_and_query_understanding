"""
Schema Configuration
Controls which schema version to use (core or advanced)
"""

from enum import Enum


class SchemaVersion(str, Enum):
    """Schema version to use"""
    CORE = "core"  # Simple, stable schema aligned with test requirements
    ADVANCED = "advanced"  # Comprehensive schema demonstrating production design


# Default to CORE for stable demo execution
DEFAULT_SCHEMA_VERSION = SchemaVersion.CORE
