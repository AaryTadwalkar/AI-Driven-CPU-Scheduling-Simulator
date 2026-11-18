"""
1. What this file does:
   Initializes the policies package for CPU scheduling algorithms and exports the public API.
2. Overall design:
   Exposes the abstract Policy base class, classical scheduling policies (FCFS, SJF, SRTF, RR),
   and the centralized policy registry and factory functions (get_policy, list_policies,
   register_policy).
3. Elements in the file:
   - Policy: Abstract base class for scheduling policies.
   - FCFSPolicy: Classical First-Come, First-Served scheduling policy.
   - SJFPolicy: Classical non-preemptive Shortest Job First scheduling policy.
   - SRTFPolicy: Classical preemptive Shortest Remaining Time First scheduling policy.
   - RoundRobinPolicy: Classical Round Robin scheduling policy with configurable quantum.
   - get_policy: Factory function to retrieve and instantiate policies by name.
   - list_policies: Function returning sorted unique canonical policy names.
   - register_policy: Function allowing dynamic registration of custom policy implementations.
4. Final output of the file:
   Exports Policy classes and registry functions for package consumers.
"""

from schedsim.policies.base import Policy
from schedsim.policies.fcfs import FCFSPolicy
from schedsim.policies.registry import get_policy, list_policies, register_policy
from schedsim.policies.round_robin import RoundRobinPolicy
from schedsim.policies.sjf import SJFPolicy
from schedsim.policies.srtf import SRTFPolicy

__all__ = [
    "FCFSPolicy",
    "Policy",
    "RoundRobinPolicy",
    "SJFPolicy",
    "SRTFPolicy",
    "get_policy",
    "list_policies",
    "register_policy",
]
