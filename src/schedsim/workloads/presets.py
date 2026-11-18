"""
1. What this file does:
   Defines standard named preset workloads and retrieval helpers for CPU scheduling simulation.
2. Overall design:
   Maintains pre-configured canonical process collections representing classic OS scheduling
   scenarios (convoy effect, Section 4.6 fixture, simultaneous arrivals, staggered arrivals,
   equal bursts, long idle gaps, high contention, and mixed priorities). Provides case-insensitive
   lookup with alias support, returning fresh, validated Process instances to ensure immutability
   across independent simulation runs.
3. Elements in the file:
   - _PRESET_SPECS: Raw process parameter tuples for canonical presets.
   - _PRESET_DESCRIPTIONS: Human-readable descriptions for all presets and aliases.
   - _ALIAS_MAP: Mapping from normalized string keys to canonical preset names.
   - get_preset: Factory function returning a fresh list of Process instances for a preset name.
   - list_presets: Introspection function returning available preset keys and their summaries.
4. Final output of the file:
   Exports get_preset and list_presets for CLI commands, benchmark suites, API endpoints,
   and automated unit tests.
"""

from schedsim.domain.process import Process, validate_processes

# Tuple structure: (pid, arrival, burst, priority)
_PresetTuple = tuple[str, int, int, int]

_PRESET_SPECS: dict[str, tuple[_PresetTuple, ...]] = {
    "section_4_6": (
        ("P1", 0, 5, 0),
        ("P2", 1, 3, 0),
        ("P3", 2, 2, 0),
    ),
    "convoy": (
        ("P1", 0, 20, 0),
        ("P2", 1, 2, 0),
        ("P3", 2, 2, 0),
        ("P4", 3, 1, 0),
        ("P5", 4, 2, 0),
    ),
    "equal_bursts": (
        ("P1", 0, 4, 0),
        ("P2", 1, 4, 0),
        ("P3", 2, 4, 0),
        ("P4", 3, 4, 0),
    ),
    "simultaneous": (
        ("P1", 0, 8, 0),
        ("P2", 0, 4, 0),
        ("P3", 0, 2, 0),
        ("P4", 0, 6, 0),
    ),
    "long_idle_gap": (
        ("P1", 0, 3, 0),
        ("P2", 1, 2, 0),
        ("P3", 15, 4, 0),
        ("P4", 16, 2, 0),
    ),
    "staggered": (
        ("P1", 0, 4, 0),
        ("P2", 3, 4, 0),
        ("P3", 6, 4, 0),
        ("P4", 9, 4, 0),
    ),
    "high_contention": (
        ("P1", 0, 8, 0),
        ("P2", 0, 2, 0),
        ("P3", 1, 1, 0),
        ("P4", 1, 6, 0),
        ("P5", 2, 3, 0),
        ("P6", 3, 2, 0),
    ),
    "mixed_priorities": (
        ("P1", 0, 6, 3),
        ("P2", 1, 3, 1),
        ("P3", 2, 8, 4),
        ("P4", 3, 2, 2),
        ("P5", 4, 4, 0),
    ),
}

_PRESET_DESCRIPTIONS: dict[str, str] = {
    "section_4_6": (
        "Hand-verified reference fixture from Section 4.6 "
        "(P1: arr 0, burst 5; P2: arr 1, burst 3; P3: arr 2, burst 2) [alias: '1']."
    ),
    "1": (
        "Alias for Section 4.6 reference fixture "
        "(P1: arr 0, burst 5; P2: arr 1, burst 3; P3: arr 2, burst 2)."
    ),
    "convoy": (
        "Convoy effect scenario with a long CPU-bound job (burst 20) followed by "
        "multiple short jobs, demonstrating FCFS head-of-line blocking."
    ),
    "equal_bursts": (
        "Uniform workload where all processes have identical burst lengths (4 ticks) "
        "arriving sequentially at ticks 0, 1, 2, 3."
    ),
    "simultaneous": (
        "Concurrent arrival scenario where all four processes arrive at tick 0 "
        "with varying burst lengths (8, 4, 2, 6 ticks)."
    ),
    "long_idle_gap": (
        "Bimodal workload with an extended CPU idle gap (10+ ticks) between "
        "early batch (ticks 0-1) and late batch (ticks 15-16)."
    ),
    "staggered": (
        "Regularly staggered arrivals every 3 ticks with identical burst lengths "
        "(4 ticks each) causing partial execution overlaps."
    ),
    "high_contention": (
        "High contention mix of 6 processes arriving in rapid succession (ticks 0-3) "
        "with diverse burst lengths (1 to 8 ticks)."
    ),
    "mixed_priorities": (
        "Diverse workload with non-default process priorities (0 to 4) "
        "designed for priority-based scheduling policies."
    ),
}

_ALIAS_MAP: dict[str, str] = {
    "1": "section_4_6",
    "section_4_6": "section_4_6",
    "section_4.6": "section_4_6",
    "section-4-6": "section_4_6",
    "section 4.6": "section_4_6",
    "convoy": "convoy",
    "equal_bursts": "equal_bursts",
    "equal-bursts": "equal_bursts",
    "equal bursts": "equal_bursts",
    "simultaneous": "simultaneous",
    "long_idle_gap": "long_idle_gap",
    "long-idle-gap": "long_idle_gap",
    "long idle gap": "long_idle_gap",
    "staggered": "staggered",
    "high_contention": "high_contention",
    "high-contention": "high_contention",
    "high contention": "high_contention",
    "mixed_priorities": "mixed_priorities",
    "mixed-priorities": "mixed_priorities",
    "mixed priorities": "mixed_priorities",
}


def _normalize_key(key: str) -> str:
    """
    1. The use of function:
       Normalizes an incoming preset query string into a uniform lookup key.
    2. How does the specific function work:
       Strips leading and trailing whitespace, converts to lowercase, and replaces hyphens,
       spaces, and dots with underscores.
    3. Concepts used in the function:
       String sanitization, canonical key normalization, defensive programming.
    4. Places where this function is imported and used:
       Internal helper used exclusively within presets.py by get_preset.
    5. Final output of the function, before and after:
       Before: Raw user string, potentially with whitespace, mixed casing, or delimiters.
       After: Uniform lowercase string with standardized underscore separators.
    """
    return key.strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_")


def get_preset(name_or_id: str) -> list[Process]:
    """
    1. The use of function:
       Retrieves a fresh, validated list of Process instances corresponding to a named preset.
    2. How does the specific function work:
       Validates that name_or_id is a string. Normalizes the string and resolves aliases via
       _ALIAS_MAP. If unrecognized, raises a descriptive KeyError listing all valid presets.
       Otherwise, constructs brand new Process dataclass instances from the immutable specs.
    3. Concepts used in the function:
       Factory pattern, alias resolution, defensive type checking, object immutability,
       isolation of state between simulation runs.
    4. Places where this function is imported and used:
       Exported in schedsim.workloads, used by CLI simulation runner, benchmarking scripts,
       interactive dashboards, and automated unit tests.
    5. Final output of the function, before and after:
       Before: Name or identifier string corresponding to a preset workload.
       After: Brand new list of independent Process instances ready for scheduling simulation.
    """
    if not isinstance(name_or_id, str):
        raise TypeError(
            f"Preset name_or_id must be a str, got {type(name_or_id).__name__}."
        )

    stripped = name_or_id.strip()
    if not stripped:
        avail = ", ".join(sorted(_PRESET_DESCRIPTIONS.keys()))
        raise KeyError(
            f"Unknown preset workload '{name_or_id}'. Available presets: {avail}."
        )

    # Check direct alias match first (e.g. "1" or "section 4.6")
    lower_stripped = stripped.lower()
    canonical = _ALIAS_MAP.get(lower_stripped)
    if canonical is None:
        normalized = _normalize_key(stripped)
        canonical = _ALIAS_MAP.get(normalized)

    if canonical is None or canonical not in _PRESET_SPECS:
        avail = ", ".join(sorted(_PRESET_DESCRIPTIONS.keys()))
        raise KeyError(
            f"Unknown preset workload '{name_or_id}'. Available presets: {avail}."
        )

    specs = _PRESET_SPECS[canonical]
    return [
        Process(pid=pid, arrival=arrival, burst=burst, priority=priority)
        for pid, arrival, burst, priority in specs
    ]


def list_presets() -> dict[str, str]:
    """
    1. The use of function:
       Returns a mapping of all available preset workload names and aliases to summaries.
    2. How does the specific function work:
       Creates and returns a shallow copy of _PRESET_DESCRIPTIONS to prevent external mutation
       of the internal registry.
    3. Concepts used in the function:
       Introspection, defensive copying, dictionary mapping.
    4. Places where this function is imported and used:
       Exported in schedsim.workloads, used by CLI help messages, API documentation,
       web dashboard workload selectors, and unit tests.
    5. Final output of the function, before and after:
       Before: Internal static description dictionary.
       After: Newly created dict mapping preset names and aliases to descriptive summaries.
    """
    return dict(_PRESET_DESCRIPTIONS)


# Self-validation at import time: ensure all internal preset specs satisfy domain invariants
for _name, _specs in _PRESET_SPECS.items():
    _procs = [
        Process(pid=pid, arrival=arrival, burst=burst, priority=priority)
        for pid, arrival, burst, priority in _specs
    ]
    validate_processes(_procs)
