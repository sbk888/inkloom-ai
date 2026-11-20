"""Offline regression tests for Inkloom's deterministic review guardrails."""
import json
from pathlib import Path
from verification import enforce_review_guardrails


def run_tests():
    cases_path = Path(__file__).with_name("test_cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    passed = 0
    failures = []

    for case in cases:
        checked = enforce_review_guardrails(
            case["model_result"],
            case["approved_facts"],
            case["restrictions"],
        )
        actual = checked["status"]
        if actual == case["expected_status"]:
            passed += 1
            print(f"PASS {case['id']}: {actual}")
        else:
            failures.append((case["id"], case["expected_status"], actual))
            print(f"FAIL {case['id']}: expected {case['expected_status']}, got {actual}")

    print(f"\nRegression tests: {passed}/{len(cases)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
