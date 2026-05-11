"""
basic_eval.py — standalone example: evaluate a single Q&A pair without pytest.

Run:
    python examples/basic_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_eval.client import MockLLMClient, Message
from llm_eval.metrics import run_all_metrics, overall_score

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]LLM Eval — Basic Example[/bold cyan]\n"
        "Evaluating a single Q&A pair with custom metrics (no API key needed)",
        border_style="cyan",
    ))

    # Sample data
    question = "What is the Python GIL and why does it exist?"
    contexts = [
        "Python's GIL (Global Interpreter Lock) is a mutex that protects access to Python "
        "objects, preventing multiple threads from executing Python bytecodes simultaneously. "
        "The GIL simplifies CPython's memory management but limits CPU-bound multithreaded performance."
    ]
    ground_truth = (
        "The GIL is a mutex in CPython that ensures only one thread executes Python bytecode "
        "at a time. It simplifies memory management and C extension compatibility."
    )

    # Generate answer using mock client (or real if API key set)
    client = MockLLMClient()
    prompt = f"Context: {contexts[0]}\n\nQuestion: {question}"
    answer = client.complete(prompt)

    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"[bold]Answer:[/bold] {answer}\n")

    # Run metrics
    metrics = run_all_metrics(answer=answer, contexts=contexts, keywords=["GIL", "mutex"])
    score = overall_score(metrics)

    # Display results
    table = Table(title="Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    table.add_column("Details")

    for m in metrics:
        status = "✅ PASS" if m.passed else "❌ FAIL"
        table.add_row(m.name, f"{m.score:.3f}", status, m.details)

    console.print(table)
    console.print(f"\n[bold]Overall Score:[/bold] {score:.3f}")

    if score >= 0.7:
        console.print("[bold green]✅ Evaluation PASSED[/bold green]")
    else:
        console.print("[bold red]❌ Evaluation FAILED[/bold red]")


if __name__ == "__main__":
    main()
