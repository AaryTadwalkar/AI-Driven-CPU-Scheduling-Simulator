"""
1. What this file does:
   Defines canonical workload scenario constants, enumeration tuples, and validation helpers
   for the synthetic CPU workload generation subsystem.
2. Overall design:
   Establishes standard scenario identifiers representing 9 distinct scheduling regimes:
   short jobs, long jobs, mixed durations, bimodal distributions, simultaneous arrivals,
   staggered arrivals, convoy effects, equal bursts, and high-load stress conditions. Provides
   case-insensitive validation and normalization against canonical scenario identifiers.
3. Elements in the file:
   - SCENARIO_SHORT_JOBS: Identifier string for short jobs scenario.
   - SCENARIO_LONG_JOBS: Identifier string for long jobs scenario.
   - SCENARIO_MIXED: Identifier string for mixed burst scenario.
   - SCENARIO_BIMODAL: Identifier string for bimodal burst distribution scenario.
   - SCENARIO_SIMULTANEOUS: Identifier string for simultaneous arrival scenario.
   - SCENARIO_STAGGERED: Identifier string for staggered arrival scenario.
   - SCENARIO_CONVOY: Identifier string for convoy effect scenario.
   - SCENARIO_EQUAL_BURSTS: Identifier string for equal bursts scenario.
   - SCENARIO_HIGH_LOAD: Identifier string for high load arrival saturation scenario.
   - ALL_SCENARIOS: Immutable tuple of all 9 canonical scenario identifier strings.
   - get_scenario: Normalization and validation helper returning canonical scenario strings.
4. Final output of the file:
   Exports canonical scenario constants and the get_scenario lookup helper for workloads modules,
   CLI generators, ML feature extractors, and test fixtures.
"""

# Canonical scenario identifier constants
SCENARIO_SHORT_JOBS: str = "short_jobs"
SCENARIO_LONG_JOBS: str = "long_jobs"
SCENARIO_MIXED: str = "mixed"
SCENARIO_BIMODAL: str = "bimodal"
SCENARIO_SIMULTANEOUS: str = "simultaneous"
SCENARIO_STAGGERED: str = "staggered"
SCENARIO_CONVOY: str = "convoy"
SCENARIO_EQUAL_BURSTS: str = "equal_bursts"
SCENARIO_HIGH_LOAD: str = "high_load"

# Short aliases for ergonomic usage
SHORT_JOBS: str = SCENARIO_SHORT_JOBS
LONG_JOBS: str = SCENARIO_LONG_JOBS
MIXED: str = SCENARIO_MIXED
BIMODAL: str = SCENARIO_BIMODAL
SIMULTANEOUS: str = SCENARIO_SIMULTANEOUS
STAGGERED: str = SCENARIO_STAGGERED
CONVOY: str = SCENARIO_CONVOY
EQUAL_BURSTS: str = SCENARIO_EQUAL_BURSTS
HIGH_LOAD: str = SCENARIO_HIGH_LOAD

# Canonical collection of all supported scenarios
ALL_SCENARIOS: tuple[str, ...] = (
    SCENARIO_SHORT_JOBS,
    SCENARIO_LONG_JOBS,
    SCENARIO_MIXED,
    SCENARIO_BIMODAL,
    SCENARIO_SIMULTANEOUS,
    SCENARIO_STAGGERED,
    SCENARIO_CONVOY,
    SCENARIO_EQUAL_BURSTS,
    SCENARIO_HIGH_LOAD,
)


def get_scenario(name: str) -> str:
    """
    1. The use of function:
       Validates and normalizes a scenario name string against canonical ALL_SCENARIOS.
    2. How does the specific function work:
       Ensures input name is a string, strips surrounding whitespace, converts to lowercase,
       and validates membership against ALL_SCENARIOS. Returns the canonical lowercase string
       or raises ValueError if unrecognized and TypeError if the type is invalid.
    3. Concepts used in the function:
       String normalization, case-insensitive mapping, defensive input validation.
    4. Places where this function is imported and used:
       Imported by generator.py, io.py, CLI command handlers, and test suites.
    5. Final output of the function, before and after:
       Before: Raw scenario name string with arbitrary casing and potential whitespace.
       After: Canonical scenario name string matching one of ALL_SCENARIOS, or raises
       ValueError/TypeError on invalid input.
    """
    if not isinstance(name, str):
        raise TypeError(f"Scenario name must be a str, got {type(name).__name__}.")
    normalized = name.strip().lower()
    if normalized in ALL_SCENARIOS:
        return normalized
    valid_list = ", ".join(repr(s) for s in ALL_SCENARIOS)
    raise ValueError(f"Unknown scenario '{name}'. Valid scenarios are: {valid_list}")


__all__ = [
    "ALL_SCENARIOS",
    "BIMODAL",
    "CONVOY",
    "EQUAL_BURSTS",
    "HIGH_LOAD",
    "LONG_JOBS",
    "MIXED",
    "SCENARIO_BIMODAL",
    "SCENARIO_CONVOY",
    "SCENARIO_EQUAL_BURSTS",
    "SCENARIO_HIGH_LOAD",
    "SCENARIO_LONG_JOBS",
    "SCENARIO_MIXED",
    "SCENARIO_SHORT_JOBS",
    "SCENARIO_SIMULTANEOUS",
    "SCENARIO_STAGGERED",
    "SHORT_JOBS",
    "SIMULTANEOUS",
    "STAGGERED",
    "get_scenario",
]
