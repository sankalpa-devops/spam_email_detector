
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"
REPORT_PATH = BASE_DIR / "artifacts" / "quality_report.json"

MIN_ACCURACY = 0.90
MIN_SPAM_RECALL = 0.60
MIN_SPAM_F1 = 0.70


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    checks = {
        "accuracy": (metrics["accuracy"], MIN_ACCURACY),
        "spam_recall": (metrics["spam_recall"], MIN_SPAM_RECALL),
        "spam_f1_score": (metrics["spam_f1_score"], MIN_SPAM_F1),
    }

    results = {}
    failed = []

    for name, (actual, threshold) in checks.items():
        passed = actual >= threshold

        results[name] = {
            "actual": float(actual),
            "minimum": float(threshold),
            "passed": passed,
        }

        status = "PASS" if passed else "FAIL"
        print(
            f"{name}: {actual:.4f} "
            f"(minimum: {threshold:.2f}) -> {status}"
        )

        if not passed:
            failed.append(name)

    report = {
        "quality_gate_passed": not failed,
        "checks": results,
        "failed_checks": failed,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nQuality report saved: {REPORT_PATH}")

    if failed:
        raise SystemExit(
            "Quality gate failed: " + ", ".join(failed)
        )

    print("All model quality checks passed.")


if __name__ == "__main__":
    main()
