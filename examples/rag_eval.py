"""
rag_eval.py — evaluate the RAG pipeline on the sample dataset.

Demonstrates:
1. Loading dataset from JSON
2. Running the RAG pipeline
3. Scoring with custom metrics
4. Pretty-printing results with Rich

Run:
    python examples/rag_eval.py

With a real LLM:
    OPENAI_API_KEY=sk-... python examples/rag_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich import box
from rich.console import Console
from rich.progress import track
from rich.table import Table

from llm_eval.metrics import overall_score, run_agentic_workflow, run_all_metrics
from llm_eval.rag_pipeline import RAGPipeline, SimpleRetriever

console = Console()
DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def main():
    console.print("\n[bold cyan]RAG Pipeline Evaluation[/bold cyan]\n")

    # Load dataset
    with (DATASETS_DIR / "sample_qa.json").open() as f:
        qa_data = json.load(f)[:5]  # Use first 5 for quick demo

    # Set up pipeline
    pipeline = RAGPipeline(retriever=SimpleRetriever())
    console.print(f"LLM provider: [yellow]{pipeline.llm.provider}[/yellow]")
    console.print(f"Running {len(qa_data)} questions through the RAG pipeline...\n")

    # Run evaluation
    all_results = []
    for item in track(qa_data, description="Evaluating..."):
        rag_result = pipeline.run(
            question=item["question"],
            ground_truth=item.get("ground_truth", ""),
        )
        metrics = run_all_metrics(
            answer=rag_result.answer,
            contexts=rag_result.contexts,
        )
        score = overall_score(metrics)
        agentic = run_agentic_workflow(
            question=rag_result.question,
            answer=rag_result.answer,
            contexts=rag_result.contexts,
        )
        all_results.append({
            "question": item["question"][:55] + "...",
            "score": score,
            "agentic_score": agentic.overall_score,
            "metrics": metrics,
            "answer": rag_result.answer[:80] + "...",
        })

    # Summary table
    table = Table(title="RAG Evaluation Summary", box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Question", style="cyan", width=55)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Agentic", justify="right", width=8)
    table.add_column("Status", width=10)

    for i, r in enumerate(all_results, 1):
        status = "✅" if r["score"] >= 0.7 and r["agentic_score"] >= 0.6 else "❌"
        table.add_row(
            str(i),
            r["question"],
            f"{r['score']:.3f}",
            f"{r['agentic_score']:.3f}",
            status,
        )

    console.print(table)

    avg = sum(r["score"] for r in all_results) / len(all_results)
    avg_agentic = sum(r["agentic_score"] for r in all_results) / len(all_results)
    console.print(f"\n[bold]Average Score:[/bold] {avg:.3f}")
    console.print(f"[bold]Average Agentic Score:[/bold] {avg_agentic:.3f}")
    passed = sum(1 for r in all_results if r["score"] >= 0.7)
    console.print(f"[bold]Passed:[/bold] {passed}/{len(all_results)}")


if __name__ == "__main__":
    main()
