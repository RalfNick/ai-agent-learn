# Phase 7: Agent Engineering

Phase 7 does not introduce another Agent framework. It hardens the existing
Phase 6 enterprise knowledge-base Agent into a system that can be evaluated,
recovered, observed, governed, and operated.

The engineering track is paired with the Chinese blog series
`AI Agent 工程进阶`.

## Companion project

`agent-reliability-lab/` is the only project used throughout the series. Each
article adds one main engineering variable while keeping the task contract,
dataset, baseline, and reports comparable.

The first checkpoint provides:

- a versioned Agent System Card;
- a deterministic non-Agent baseline;
- a small task dataset with executable graders;
- JSON and Markdown reports;
- tests for contract validation and baseline behavior.

Run it with:

```bash
cd phase-7-agent-engineering/agent-reliability-lab
python run_lab.py check-contract
python run_lab.py baseline
python -m unittest discover -s tests -v
```

See [agent-reliability-lab/README.md](agent-reliability-lab/README.md) for the
project walkthrough and planned article checkpoints.
