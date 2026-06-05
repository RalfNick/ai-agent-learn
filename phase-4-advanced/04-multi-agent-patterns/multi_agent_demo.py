from __future__ import annotations

import argparse

from supervisor import MultiAgentSupervisor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic multi-agent pattern demo.")
    parser.add_argument(
        "question",
        nargs="?",
        default="请评估 Phase4 Memory 的代码、文章和测试证据",
        help="Question to route through the supervisor.",
    )
    args = parser.parse_args()

    supervisor = MultiAgentSupervisor()
    result = supervisor.run(args.question)

    print("Question:")
    print(args.question)
    print("\nTrace:")
    for step in result.trace:
        print(f"- {step}")

    print("\nHandoffs:")
    for packet in result.handoffs:
        print(f"- {packet.target.value}: {packet.task}")

    print("\nAnswer:")
    print(result.answer)

    print("\nEvidence:")
    for item in result.evidence:
        print(f"- {item}")

    print("\nReview:")
    print(f"- status: {result.review.status.value}")
    print(f"- score: {result.review.score:.2f}")
    for comment in result.review.comments:
        print(f"- {comment}")


if __name__ == "__main__":
    main()
