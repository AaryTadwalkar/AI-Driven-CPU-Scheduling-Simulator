"""
1. What this file does:
   Defines unit tests for the TimelineSegment dataclass and Timeline execution history tracker.
2. Overall design:
   Validates domain constraints, dataclass immutability, tick recording, segment merging,
   counter aggregations, and partition invariant checks across varied simulation histories.
3. Elements in the file:
   - test_segment_valid_creation: Tests valid parameter initialization and duration computation.
   - test_segment_immutability: Tests frozen dataclass modification prevention.
   - test_segment_equality_and_hashing: Tests value-based equality and hashable set operations.
   - test_segment_invalid_label: Tests type and non-empty string validation on segment label.
   - test_segment_invalid_start: Tests type and non-negative constraints on segment start tick.
   - test_segment_invalid_end: Tests type and strictly greater boundary checks on segment end tick.
   - test_timeline_empty: Tests properties and behaviors of an unpopulated Timeline.
   - test_timeline_record_tick_and_segment: Tests single-tick and range interval recording.
   - test_timeline_contiguous_merging: Tests combining contiguous ticks with identical labels.
   - test_timeline_non_contiguous_separation: Tests distinct segments across different labels/gaps.
   - test_timeline_out_of_order_merging: Tests correct chronological sorting before merging.
   - test_timeline_tick_counters: Tests busy, overhead, idle, and makespan counter properties.
   - test_timeline_validate_partition_valid: Tests partition check on complete traces.
   - test_timeline_validate_partition_gap: Tests partition rejection when gaps exist.
   - test_timeline_validate_partition_overlap: Tests partition rejection on overlaps.
   - test_timeline_init_and_defensive_copies: Tests constructor and defensive copying.
   - test_timeline_equality_and_repr: Tests timeline equality comparison and string representations.
   - test_timeline_invalid_inputs: Tests parameter type and boundary checks on recording methods.
4. Final output of the file:
   Pytest assertion passes confirming robust timeline aggregation and partition invariant checks.
"""

from dataclasses import FrozenInstanceError

import pytest

from schedsim.domain.timeline import CS_LABEL, IDLE_LABEL, Timeline, TimelineSegment


def test_segment_valid_creation() -> None:
    """
    1. The use of function:
       Tests successful instantiation and duration property of TimelineSegment with valid bounds.
    2. How does the specific function work:
       Instantiates segments with process IDs, CS, and IDLE labels, asserting fields and duration.
    3. Concepts used in the function:
       Dataclass field inspection, derived property calculation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Uninstantiated valid segment parameters.
       After: Asserts fields match inputs and duration equals (end - start).
    """
    seg1 = TimelineSegment(label="P1", start=0, end=5)
    assert seg1.label == "P1"
    assert seg1.start == 0
    assert seg1.end == 5
    assert seg1.duration == 5

    seg2 = TimelineSegment(label=CS_LABEL, start=5, end=6)
    assert seg2.label == "CS"
    assert seg2.duration == 1

    seg3 = TimelineSegment(label=IDLE_LABEL, start=6, end=10)
    assert seg3.label == "IDLE"
    assert seg3.duration == 4


def test_segment_immutability() -> None:
    """
    1. The use of function:
       Verifies that TimelineSegment instances are strictly immutable.
    2. How does the specific function work:
       Attempts to assign a new label to an existing instance and asserts FrozenInstanceError.
    3. Concepts used in the function:
       Frozen dataclass immutability, exception handling.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Instantiated TimelineSegment.
       After: Asserts FrozenInstanceError is raised on mutation attempt.
    """
    seg = TimelineSegment(label="P1", start=0, end=3)
    with pytest.raises(FrozenInstanceError):
        seg.label = "P2"  # type: ignore[misc]


def test_segment_equality_and_hashing() -> None:
    """
    1. The use of function:
       Verifies value-based equality and hashing behavior of TimelineSegment.
    2. How does the specific function work:
       Creates identical and distinct segment instances, testing equality and set membership.
    3. Concepts used in the function:
       Value object semantics, hash collisions, set uniqueness.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Distinct TimelineSegment instances.
       After: Asserts equal instances match and deduplicate in sets.
    """
    s1 = TimelineSegment(label="P1", start=0, end=2)
    s2 = TimelineSegment(label="P1", start=0, end=2)
    s3 = TimelineSegment(label="P1", start=0, end=3)
    s4 = TimelineSegment(label="P2", start=0, end=2)

    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert len({s1, s2, s3, s4}) == 3


def test_segment_invalid_label() -> None:
    """
    1. The use of function:
       Tests rejection of invalid labels during TimelineSegment construction.
    2. How does the specific function work:
       Passes non-string types and empty/whitespace strings, asserting TypeError and ValueError.
    3. Concepts used in the function:
       Type checking, boundary validation, defensive programming.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Malformed label arguments.
       After: Asserts appropriate TypeError or ValueError is raised.
    """
    with pytest.raises(TypeError, match="TimelineSegment label must be a str"):
        TimelineSegment(label=123, start=0, end=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="TimelineSegment label cannot be empty"):
        TimelineSegment(label="", start=0, end=1)

    with pytest.raises(ValueError, match="TimelineSegment label cannot be empty"):
        TimelineSegment(label="   ", start=0, end=1)


def test_segment_invalid_start() -> None:
    """
    1. The use of function:
       Tests rejection of invalid start tick values during TimelineSegment construction.
    2. How does the specific function work:
       Passes non-integer types, booleans, and negative integers, asserting errors.
    3. Concepts used in the function:
       Type checking, boolean discrimination, lower bound enforcement.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Invalid start tick arguments.
       After: Asserts TypeError or ValueError is raised.
    """
    with pytest.raises(TypeError, match="TimelineSegment start must be an int"):
        TimelineSegment(label="P1", start=True, end=5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="TimelineSegment start must be an int"):
        TimelineSegment(label="P1", start="0", end=5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="TimelineSegment start must be >= 0"):
        TimelineSegment(label="P1", start=-1, end=5)


def test_segment_invalid_end() -> None:
    """
    1. The use of function:
       Tests rejection of invalid end tick values during TimelineSegment construction.
    2. How does the specific function work:
       Passes non-integer types, booleans, and values <= start tick, asserting errors.
    3. Concepts used in the function:
       Type checking, strict interval ordering (end > start).
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Invalid end tick arguments.
       After: Asserts TypeError or ValueError is raised.
    """
    with pytest.raises(TypeError, match="TimelineSegment end must be an int"):
        TimelineSegment(label="P1", start=0, end=False)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="TimelineSegment end must be an int"):
        TimelineSegment(label="P1", start=0, end=2.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="strictly greater than start"):
        TimelineSegment(label="P1", start=3, end=3)

    with pytest.raises(ValueError, match="strictly greater than start"):
        TimelineSegment(label="P1", start=3, end=2)


def test_timeline_empty() -> None:
    """
    1. The use of function:
       Tests the state, properties, and validation behavior of an unpopulated Timeline.
    2. How does the specific function work:
       Instantiates Timeline with no data and asserts empty segments, 0 makespan, 0 ticks,
       and valid partition.
    3. Concepts used in the function:
       Empty collection handling, default metric computation, vacuous partition truth.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Empty Timeline instance.
       After: Asserts all zero/empty properties and True for validate_partition().
    """
    timeline = Timeline()
    assert timeline.to_segments() == []
    assert timeline.raw_segments == []
    assert len(timeline) == 0
    assert timeline.makespan == 0
    assert timeline.busy_ticks == 0
    assert timeline.overhead_ticks == 0
    assert timeline.idle_ticks == 0
    assert timeline.validate_partition() is True


def test_timeline_record_tick_and_segment() -> None:
    """
    1. The use of function:
       Tests recording individual execution ticks and multi-tick segments.
    2. How does the specific function work:
       Records a single tick for P1 and an explicit range segment for P2, verifying properties.
    3. Concepts used in the function:
       Tick recording, range recording, half-open interval boundaries.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Empty Timeline instance.
       After: Recorded segments reflect exact boundaries and labels.
    """
    timeline = Timeline()
    timeline.record_tick(tick=0, label="P1")
    assert len(timeline) == 1
    assert timeline.to_segments() == [TimelineSegment("P1", 0, 1)]
    assert timeline.makespan == 1
    assert timeline.busy_ticks == 1

    timeline.record_segment(start=1, end=4, label="P2")
    assert len(timeline) == 2
    assert timeline.to_segments() == [
        TimelineSegment("P1", 0, 1),
        TimelineSegment("P2", 1, 4),
    ]
    assert timeline.makespan == 4
    assert timeline.busy_ticks == 4


def test_timeline_contiguous_merging() -> None:
    """
    1. The use of function:
       Tests automatic merging of contiguous ticks and intervals sharing the same label.
    2. How does the specific function work:
       Records sequential ticks 0, 1, 2 for P1 and asserts consolidation into [0, 3).
    3. Concepts used in the function:
       Segment consolidation, contiguous boundary merging.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Separate single-tick records in raw timeline.
       After: to_segments() returns a single merged TimelineSegment.
    """
    timeline = Timeline()
    timeline.record_tick(0, "P1")
    timeline.record_tick(1, "P1")
    timeline.record_tick(2, "P1")

    assert len(timeline) == 3
    segments = timeline.to_segments()
    assert len(segments) == 1
    assert segments[0] == TimelineSegment("P1", 0, 3)
    assert segments[0].duration == 3
    assert timeline.makespan == 3
    assert timeline.busy_ticks == 3


def test_timeline_non_contiguous_separation() -> None:
    """
    1. The use of function:
       Tests that non-contiguous intervals or differing labels remain separate segments.
    2. How does the specific function work:
       Records P1, CS, P2, and P1 again, asserting each transition creates a distinct segment.
    3. Concepts used in the function:
       Label transition segmentation, non-contiguous recurrence isolation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Interleaved raw ticks.
       After: Produces distinct segments without improperly merging recurring labels across bounds.
    """
    timeline = Timeline()
    timeline.record_tick(0, "P1")
    timeline.record_tick(1, "P1")
    timeline.record_tick(2, CS_LABEL)
    timeline.record_tick(3, "P2")
    timeline.record_tick(4, "P2")
    timeline.record_tick(5, "P1")

    segments = timeline.to_segments()
    expected = [
        TimelineSegment("P1", 0, 2),
        TimelineSegment("CS", 2, 3),
        TimelineSegment("P2", 3, 5),
        TimelineSegment("P1", 5, 6),
    ]
    assert segments == expected


def test_timeline_out_of_order_merging() -> None:
    """
    1. The use of function:
       Tests that intervals recorded out of chronological order are correctly sorted and merged.
    2. How does the specific function work:
       Records ticks 2, 0, and 1 for P1 and verifies that to_segments() merges them into [0, 3).
    3. Concepts used in the function:
       Chronological sort normalization, out-of-order event assimilation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Out-of-order raw segment records.
       After: Correctly ordered and merged segment list.
    """
    timeline = Timeline()
    timeline.record_tick(2, "P1")
    timeline.record_tick(0, "P1")
    timeline.record_tick(1, "P1")

    assert timeline.to_segments() == [TimelineSegment("P1", 0, 3)]
    assert timeline.makespan == 3
    assert timeline.validate_partition() is True


def test_timeline_tick_counters() -> None:
    """
    1. The use of function:
       Tests accurate accounting for busy_ticks, overhead_ticks, idle_ticks, and makespan.
    2. How does the specific function work:
       Constructs a timeline containing IDLE, process execution, and CS intervals; asserts counters.
    3. Concepts used in the function:
       CPU time accounting, conservation law (busy + overhead + idle == makespan).
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Timeline with varied work categories.
       After: Asserts each counter matches expected duration sum.
    """
    timeline = Timeline()
    timeline.record_segment(0, 3, IDLE_LABEL)  # 3 idle
    timeline.record_segment(3, 7, "P1")        # 4 busy
    timeline.record_segment(7, 9, CS_LABEL)    # 2 overhead
    timeline.record_segment(9, 12, "P2")       # 3 busy
    timeline.record_segment(12, 14, CS_LABEL)  # 2 overhead
    timeline.record_segment(14, 15, "P3")      # 1 busy

    assert timeline.makespan == 15
    assert timeline.idle_ticks == 3
    assert timeline.overhead_ticks == 4
    assert timeline.busy_ticks == 8
    assert (
        timeline.busy_ticks + timeline.overhead_ticks + timeline.idle_ticks == timeline.makespan
    )
    assert timeline.validate_partition() is True


def test_timeline_validate_partition_valid() -> None:
    """
    1. The use of function:
       Tests that a contiguous timeline covering [0, makespan) without gaps or overlaps passes.
    2. How does the specific function work:
       Builds a contiguous sequence of P1, CS, P2, asserting validate_partition() returns True.
    3. Concepts used in the function:
       Partition verification, interval continuity.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Complete valid timeline.
       After: validate_partition() returns True.
    """
    timeline = Timeline()
    timeline.record_segment(0, 5, "P1")
    timeline.record_segment(5, 6, "CS")
    timeline.record_segment(6, 10, "P2")
    assert timeline.validate_partition() is True


def test_timeline_validate_partition_gap() -> None:
    """
    1. The use of function:
       Tests that validate_partition() detects and rejects gaps in execution.
    2. How does the specific function work:
       Tests gap at origin (starts > 0) and gap in middle ([0, 2) and [3, 5)), asserting False.
    3. Concepts used in the function:
       Gap detection, origin alignment verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Timeline containing unrecorded time gaps.
       After: validate_partition() returns False.
    """
    # Gap at origin
    t1 = Timeline()
    t1.record_segment(1, 4, "P1")
    assert t1.validate_partition() is False

    # Gap in middle
    t2 = Timeline()
    t2.record_segment(0, 2, "P1")
    t2.record_segment(3, 5, "P2")
    assert t2.validate_partition() is False


def test_timeline_validate_partition_overlap() -> None:
    """
    1. The use of function:
       Tests that validate_partition() detects and rejects overlapping intervals.
    2. How does the specific function work:
       Tests partial overlap, complete overlap, and duplicate tick recording, asserting False.
    3. Concepts used in the function:
       Overlap detection, mutual exclusion verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Timeline containing overlapping intervals.
       After: validate_partition() returns False.
    """
    # Partial overlap between different processes
    t1 = Timeline()
    t1.record_segment(0, 3, "P1")
    t1.record_segment(2, 5, "P2")
    assert t1.validate_partition() is False

    # Duplicate tick
    t2 = Timeline()
    t2.record_tick(0, "P1")
    t2.record_tick(0, "P2")
    assert t2.validate_partition() is False

    # Subsumed interval overlap
    t3 = Timeline()
    t3.record_segment(0, 6, "P1")
    t3.record_segment(2, 4, CS_LABEL)
    assert t3.validate_partition() is False


def test_timeline_init_and_defensive_copies() -> None:
    """
    1. The use of function:
       Tests Timeline constructor with initial segments and verifies defensive copying.
    2. How does the specific function work:
       Passes initial segments, verifies state, asserts modifying returned lists does not mutate
       internal timeline state, and verifies TypeError on invalid constructor items.
    3. Concepts used in the function:
       Defensive copying, constructor validation, encapsulation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Segment collections and timeline instances.
       After: Asserts internal state remains isolated and constructor validates types.
    """
    initial = [TimelineSegment("P1", 0, 2), TimelineSegment("P2", 2, 4)]
    timeline = Timeline(segments=initial)
    assert len(timeline) == 2
    assert timeline.makespan == 4

    # Modifying returned raw_segments does not affect timeline
    raws = timeline.raw_segments
    raws.append(TimelineSegment("P3", 4, 6))
    assert len(timeline) == 2

    # Modifying returned to_segments does not affect timeline
    segs = timeline.to_segments()
    segs.clear()
    assert len(timeline.to_segments()) == 2

    with pytest.raises(TypeError, match="Expected TimelineSegment"):
        Timeline(segments=["invalid"])  # type: ignore[list-item]


def test_timeline_equality_and_repr() -> None:
    """
    1. The use of function:
       Tests string representation and value equality comparison between Timeline instances.
    2. How does the specific function work:
       Compares identical and distinct timelines, checks non-Timeline equality, and verifies repr.
    3. Concepts used in the function:
       Equality comparison, string formatting.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Separate Timeline instances.
       After: Asserts equality matches merged content and repr formats correctly.
    """
    t1 = Timeline()
    t1.record_tick(0, "P1")
    t1.record_tick(1, "P1")

    t2 = Timeline()
    t2.record_segment(0, 2, "P1")

    assert t1 == t2
    assert t1 != "not_a_timeline"

    t3 = Timeline()
    t3.record_segment(0, 2, "P2")
    assert t1 != t3

    assert repr(t1) == "Timeline(segments=[TimelineSegment(label='P1', start=0, end=2)])"


def test_timeline_invalid_inputs() -> None:
    """
    1. The use of function:
       Tests rejection of invalid tick and segment arguments by Timeline recording methods.
    2. How does the specific function work:
       Calls record_tick and record_segment with invalid types, booleans, negative values,
       and inverted ranges, asserting TypeError and ValueError.
    3. Concepts used in the function:
       Input boundary validation, type checking, error propagation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Invalid arguments supplied to record methods.
       After: Asserts appropriate TypeError or ValueError is raised.
    """
    timeline = Timeline()

    with pytest.raises(TypeError, match="Timeline tick must be an int"):
        timeline.record_tick(tick=True, label="P1")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Timeline tick must be an int"):
        timeline.record_tick(tick="0", label="P1")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Timeline tick must be >= 0"):
        timeline.record_tick(tick=-1, label="P1")

    with pytest.raises(ValueError, match="strictly greater than start"):
        timeline.record_segment(start=4, end=2, label="P1")
