from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_lab.contracts import ContractError, load_contract, validate_contract


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_project_contract_is_valid(self) -> None:
        contract = load_contract(ROOT / "contracts" / "agent-system-card.json")
        self.assertEqual(contract["version"], "0.1.0")

    def test_missing_boundary_is_rejected(self) -> None:
        contract = json.loads(
            (ROOT / "contracts" / "agent-system-card.json").read_text(encoding="utf-8")
        )
        del contract["boundaries"]["prohibited_actions"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "prohibited_actions"):
                load_contract(path)

    def test_conflicting_action_policy_is_rejected(self) -> None:
        contract = json.loads(
            (ROOT / "contracts" / "agent-system-card.json").read_text(encoding="utf-8")
        )
        contract["boundaries"]["prohibited_actions"].append("read_knowledge")
        with self.assertRaisesRegex(ContractError, "allowed and prohibited"):
            validate_contract(contract)

    def test_success_and_failure_states_cannot_overlap(self) -> None:
        contract = json.loads(
            (ROOT / "contracts" / "agent-system-card.json").read_text(encoding="utf-8")
        )
        contract["failure"]["terminal_states"].append("answered")
        with self.assertRaisesRegex(ContractError, "successful and failed"):
            validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
