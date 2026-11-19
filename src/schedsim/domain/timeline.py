"""
1. What this file does:
   Defines the TimelineSegment value object and Timeline execution tracker for CPU scheduling
   simulations, recording tick intervals and merging contiguous segments.
2. Overall design:
   Provides discrete-time execution logging for scheduling engines. Merges consecutive atomic
   ticks into contiguous intervals labelled with process identifiers, context switches ('CS'),
   idle periods ('IDLE'), or AI inference stalls ('AI'), and verifies partition integrity
   over [0, makespan).
3. Elements in the file:
   - AI_LABEL: String constant for AI model inference stall intervals ('AI').
   - CS_LABEL: String constant for context switch overhead execution intervals ('CS').
   - IDLE_LABEL: String constant for idle CPU intervals ('IDLE').
   - TimelineSegment: Frozen dataclass representing a discrete half-open execution interval.
   - Timeline: Mutable execution tracker collecting ticks and segments with merging and validation.
4. Final output of the file:
   Exports TimelineSegment and Timeline for domain engine execution, metric calculation,
   Gantt chart rendering, and simulation evaluation.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

AI_LABEL: Final[str] = "AI"
CS_LABEL: Final[str] = "CS"
IDLE_LABEL: Final[str] = "IDLE"


@dataclass(frozen=True)
class TimelineSegment:
    """
    1. The use of class:
       Represents an immutable half-open time interval [start, end) on the CPU execution timeline.
    2. What the class contains:
       - label: Process identifier string, or special indicator 'CS' / 'IDLE'.
       - start: Non-negative integer start tick (inclusive).
       - end: Strictly greater integer end tick (exclusive).
       - duration: Property returning the interval length (end - start).
       - __post_init__: Integrity validator enforcing non-negative, non-empty, and positive bounds.
    3. Technologies used and why:
       - Frozen dataclass for immutability, thread-safety, and automatic value-based equality/hash.
       - Python 3.11 type hints with strict runtime type and boundary assertions.
    4. Key design and overall design of the class:
       Acts as a domain value object adhering to half-open interval conventions [start, end)
       standard in operating systems and Gantt visualizers.
    """

    label: str
    start: int
    end: int

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates segment bounds and label formatting upon instantiation.
        2. How does the specific function work:
           Checks types (rejecting bools for ints), ensures start >= 0, end > start, and validates
           that label is a non-empty, non-whitespace string.
        3. Concepts used in the function:
           Dataclass post-init validation, type boundary enforcement, defensive programming.
        4. Places where this function is imported and used:
           Invoked automatically during TimelineSegment instantiation.
        5. Final output of the function, before and after:
           Before: Unverified raw attribute values.
           After: Verified invariant state or raises TypeError / ValueError.
        """
        if not isinstance(self.label, str):
            raise TypeError(
                f"TimelineSegment label must be a str, got {type(self.label).__name__}."
            )
        if not self.label.strip():
            raise ValueError("TimelineSegment label cannot be empty or whitespace only.")

        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError(
                f"TimelineSegment start must be an int, got {type(self.start).__name__}."
            )
        if self.start < 0:
            raise ValueError(f"TimelineSegment start must be >= 0, got {self.start}.")

        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError(
                f"TimelineSegment end must be an int, got {type(self.end).__name__}."
            )
        if self.end <= self.start:
            raise ValueError(
                f"TimelineSegment end ({self.end}) must be strictly greater than "
                f"start ({self.start})."
            )

    @property
    def duration(self) -> int:
        """
        1. The use of function:
           Computes the duration of the execution segment in ticks.
        2. How does the specific function work:
           Returns the difference between end tick and start tick (end - start).
        3. Concepts used in the function:
           Half-open interval arithmetic, property decorator.
        4. Places where this function is imported and used:
           Used by Timeline aggregation properties, Gantt chart plotting, and metric calculations.
        5. Final output of the function, before and after:
           Before: Accessed on a valid TimelineSegment instance.
           After: Returns positive integer duration.
        """
        return self.end - self.start


class Timeline:
    """
    1. The use of class:
       Maintains the discrete CPU execution history and produces merged segments for visualization.
    2. What the class contains:
       - _raw_segments: Internal list of recorded TimelineSegment instances.
       - record_tick: Records a single unit interval [tick, tick + 1).
       - record_segment: Records an arbitrary valid half-open interval [start, end).
       - to_segments: Merges contiguous intervals with identical labels into ordered segments.
       - makespan: Maximum end tick recorded in the timeline.
       - busy_ticks: Total ticks spent executing process tasks.
       - overhead_ticks: Total ticks consumed by context switches ('CS').
       - ai_ticks: Total ticks consumed by AI inference stall overhead ('AI').
       - idle_ticks: Total ticks spent idling ('IDLE').
       - validate_partition: Verifies gapless and overlap-free coverage of [0, makespan).
    3. Technologies used and why:
       - Encapsulated mutable collection with immutable segment outputs.
       - Python properties for zero-argument computed simulation metrics.
    4. Key design and overall design of the class:
       Decouples fine-grained per-tick simulation logging from consolidated reporting and
       visualization layers while enforcing timeline partition invariants.
    """

    def __init__(self, segments: Sequence[TimelineSegment] | None = None) -> None:
        """
        1. The use of function:
           Initializes a new Timeline instance with optional starting segments.
        2. How does the specific function work:
           Creates an internal list for raw segments; validates and appends initial segments.
        3. Concepts used in the function:
           Constructor initialization, defensive sequence copying.
        4. Places where this function is imported and used:
           Instantiated by simulation engines, testing fixtures, and benchmarks.
        5. Final output of the function, before and after:
           Before: Uninitialized Timeline reference.
           After: Initialized Timeline with empty or seeded raw segment list.
        """
        self._raw_segments: list[TimelineSegment] = []
        if segments is not None:
            for s in segments:
                if not isinstance(s, TimelineSegment):
                    raise TypeError(
                        f"Expected TimelineSegment, got {type(s).__name__}."
                    )
                self._raw_segments.append(s)

    def record_tick(self, tick: int, label: str) -> None:
        """
        1. The use of function:
           Records a single-tick execution interval [tick, tick + 1) for the given label.
        2. How does the specific function work:
           Validates tick integer type and non-negative constraint, then delegates to
           record_segment.
        3. Concepts used in the function:
           Discrete time stepping, delegation, boundary validation.
        4. Places where this function is imported and used:
           Called on every engine step during simulation execution.
        5. Final output of the function, before and after:
           Before: Timeline without the current tick.
           After: Appends a 1-tick segment [tick, tick + 1) to internal history.
        """
        if isinstance(tick, bool) or not isinstance(tick, int):
            raise TypeError(f"Timeline tick must be an int, got {type(tick).__name__}.")
        if tick < 0:
            raise ValueError(f"Timeline tick must be >= 0, got {tick}.")
        self.record_segment(start=tick, end=tick + 1, label=label)

    def record_segment(self, start: int, end: int, label: str) -> None:
        """
        1. The use of function:
           Records an arbitrary execution interval [start, end) for the specified label.
        2. How does the specific function work:
           Constructs a TimelineSegment (which enforces type and range checks) and appends it.
        3. Concepts used in the function:
           Interval logging, entity validation, append-only history.
        4. Places where this function is imported and used:
           Called by engine for idle jumps, batch dispatches, or directly in test scenarios.
        5. Final output of the function, before and after:
           Before: Timeline without the new segment.
           After: Timeline with new TimelineSegment appended to raw history.
        """
        segment = TimelineSegment(label=label, start=start, end=end)
        self._raw_segments.append(segment)

    def to_segments(self) -> list[TimelineSegment]:
        """
        1. The use of function:
           Produces an ordered list of timeline segments with contiguous identical labels merged.
        2. How does the specific function work:
           Sorts recorded segments by start then end tick. Merges adjacent segments where labels
           match and previous end equals current start. Returns merged list without mutating
           raw logs.
        3. Concepts used in the function:
           Interval merging, stable sorting, immutable result projection.
        4. Places where this function is imported and used:
           Invoked by Gantt plotting utilities, metrics evaluators, partition validators, and tests.
        5. Final output of the function, before and after:
           Before: Unordered or fine-grained raw tick history in timeline.
           After: Returns newly allocated list of merged, contiguous TimelineSegments.
        """
        if not self._raw_segments:
            return []

        sorted_segments = sorted(self._raw_segments, key=lambda s: (s.start, s.end))
        merged: list[TimelineSegment] = []

        for seg in sorted_segments:
            if not merged:
                merged.append(seg)
                continue

            last = merged[-1]
            if last.label == seg.label and last.end == seg.start:
                merged[-1] = TimelineSegment(
                    label=last.label,
                    start=last.start,
                    end=seg.end,
                )
            else:
                merged.append(seg)

        return merged

    @property
    def makespan(self) -> int:
        """
        1. The use of function:
           Calculates the final completion tick of the timeline.
        2. How does the specific function work:
           Finds the maximum end tick among all recorded segments, returning 0 if empty.
        3. Concepts used in the function:
           Reduction, default-handling in empty sequences, makespan metric.
        4. Places where this function is imported and used:
           Used by simulation result summaries, CPU utilization metrics, and partition checking.
        5. Final output of the function, before and after:
           Before: Property accessed on timeline.
           After: Returns maximum end tick as non-negative integer.
        """
        if not self._raw_segments:
            return 0
        return max(s.end for s in self._raw_segments)

    @property
    def busy_ticks(self) -> int:
        """
        1. The use of function:
           Calculates total ticks spent actively executing process workloads.
        2. How does the specific function work:
           Sums duration of all merged segments whose label is neither 'CS', 'IDLE', nor 'AI'.
        3. Concepts used in the function:
           Category filtering, interval duration summation, CPU work accounting.
        4. Places where this function is imported and used:
           Used in CPU utilization calculation and burst conservation test checks.
        5. Final output of the function, before and after:
           Before: Property accessed on timeline.
           After: Returns non-negative integer sum of active process execution ticks.
        """
        return sum(
            s.duration
            for s in self.to_segments()
            if s.label not in (CS_LABEL, IDLE_LABEL, AI_LABEL)
        )

    @property
    def overhead_ticks(self) -> int:
        """
        1. The use of function:
           Calculates total ticks spent performing context switches ('CS').
        2. How does the specific function work:
           Sums duration of all merged segments whose label matches CS_LABEL ('CS').
        3. Concepts used in the function:
           Overhead accounting, filtered summation.
        4. Places where this function is imported and used:
           Used in aggregate metrics reporting and context switch overhead analysis.
        5. Final output of the function, before and after:
           Before: Property accessed on timeline.
           After: Returns non-negative integer sum of context switch ticks.
        """
        return sum(
            s.duration
            for s in self.to_segments()
            if s.label == CS_LABEL
        )

    @property
    def ai_ticks(self) -> int:
        """
        1. The use of function:
           Calculates total ticks consumed by AI inference stall overhead ('AI').
        2. How does the specific function work:
           Sums duration of all merged segments whose label matches AI_LABEL ('AI').
        3. Concepts used in the function:
           AI overhead accounting, filtered summation.
        4. Places where this function is imported and used:
           Used in aggregate metrics reporting and AI inference overhead analysis.
        5. Final output of the function, before and after:
           Before: Property accessed on timeline.
           After: Returns non-negative integer sum of AI inference stall ticks.
        """
        return sum(
            s.duration
            for s in self.to_segments()
            if s.label == AI_LABEL
        )

    @property
    def idle_ticks(self) -> int:
        """
        1. The use of function:
           Calculates total ticks where the CPU was idle with no ready tasks ('IDLE').
        2. How does the specific function work:
           Sums duration of all merged segments whose label matches IDLE_LABEL ('IDLE').
        3. Concepts used in the function:
           Idle time tracking, filtered interval summation.
        4. Places where this function is imported and used:
           Used in CPU idle ratio computation and timeline conservation checks.
        5. Final output of the function, before and after:
           Before: Property accessed on timeline.
           After: Returns non-negative integer sum of idle ticks.
        """
        return sum(
            s.duration
            for s in self.to_segments()
            if s.label == IDLE_LABEL
        )

    @property
    def raw_segments(self) -> list[TimelineSegment]:
        """
        1. The use of function:
           Provides a copy of the unmerged raw segment sequence recorded in the timeline.
        2. How does the specific function work:
           Returns a shallow copy of the internal _raw_segments list to prevent external mutation.
        3. Concepts used in the function:
           Encapsulation, defensive copying.
        4. Places where this function is imported and used:
           Used in low-level debugging, unit tests, and audit logging.
        5. Final output of the function, before and after:
           Before: Property accessed on timeline.
           After: Returns list of unmerged TimelineSegment objects.
        """
        return list(self._raw_segments)

    def validate_partition(self) -> bool:
        """
        1. The use of function:
           Verifies that the merged timeline partitions [0, makespan) without gaps or overlaps.
        2. How does the specific function work:
           Retrieves merged segments. If empty, returns True. Otherwise verifies that the first
           segment starts at 0, each subsequent segment starts exactly at the previous segment's
           end, and the final segment ends at makespan.
        3. Concepts used in the function:
           Partition verification, gap detection, overlap detection, interval topology.
        4. Places where this function is imported and used:
           Called by simulation engine assertions, validation test suites, and visualization checks.
        5. Final output of the function, before and after:
           Before: Called on a timeline instance.
           After: Returns True if segments partition [0, makespan) cleanly, False otherwise.
        """
        segments = self.to_segments()
        if not segments:
            return True

        if segments[0].start != 0:
            return False

        for i in range(len(segments) - 1):
            if segments[i + 1].start != segments[i].end:
                return False

        return True

    def __len__(self) -> int:
        """
        1. The use of function:
           Returns the count of raw recorded segments in the timeline.
        2. How does the specific function work:
           Returns len(self._raw_segments).
        3. Concepts used in the function:
           Magic method implementation for collection sizing.
        4. Places where this function is imported and used:
           Used by tests, diagnostics, and container introspection.
        5. Final output of the function, before and after:
           Before: Evaluated via len(timeline).
           After: Returns integer count of raw segments.
        """
        return len(self._raw_segments)

    def __repr__(self) -> str:
        """
        1. The use of function:
           Generates a developer-readable string representation of the timeline.
        2. How does the specific function work:
           Formats the class name and merged segments list.
        3. Concepts used in the function:
           String representation, object inspection.
        4. Places where this function is imported and used:
           Used during logging, interactive debugging, and REPL evaluation.
        5. Final output of the function, before and after:
           Before: Evaluated via repr(timeline).
           After: Returns formatted string.
        """
        return f"Timeline(segments={self.to_segments()!r})"

    def __eq__(self, other: object) -> bool:
        """
        1. The use of function:
           Checks value equality between two Timeline instances based on merged segments.
        2. How does the specific function work:
           Returns False if other is not a Timeline, otherwise compares merged segment lists.
        3. Concepts used in the function:
           Value-based equivalence comparison.
        4. Places where this function is imported and used:
           Used in unit test assertions and simulation equivalence checks.
        5. Final output of the function, before and after:
           Before: Evaluated via timeline == other.
           After: Returns True if merged representations are identical, False otherwise.
        """
        if not isinstance(other, Timeline):
            return False
        return self.to_segments() == other.to_segments()


__all__ = ["AI_LABEL", "CS_LABEL", "IDLE_LABEL", "Timeline", "TimelineSegment"]
