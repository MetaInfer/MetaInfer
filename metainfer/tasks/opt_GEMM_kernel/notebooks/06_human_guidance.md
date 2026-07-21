# Live optimization guidance

The task owner may submit optimization ideas from the WebUI while a run is in
progress. Guidance is durable and is delivered at the next safe action-agent
boundary:

- before a planner starts, pending guidance is included in its prompt;
- guidance submitted while the planner is running is delivered to the next
  implementer;
- guidance submitted during compilation, correctness, benchmark, or review is
  retained for the next iteration's planner.

The underlying agents run non-interactively, so guidance is not injected into
an already-running process. The UI records every item as `pending` or
`applied`, including the iteration and role that received it.

Optimization guidance is a high-priority optimization hypothesis, not a judge
override. It may change candidate source code and allowed build options, but it
cannot change the frozen BuildProfile, evaluator, correctness cases,
benchmark protocol, scoring formula, or champion promotion gates.
