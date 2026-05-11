# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Agentic no-reference evaluation workflow (`run_agentic_workflow`) for cases without explicit answers
- New no-reference metrics: `question_coverage`, `context_sufficiency`, and `appropriate_uncertainty`
- Tests for agentic workflow behavior in custom-metric and integration test suites
- Example updates showing agentic workflow usage in basic, RAG, and custom metric demos

### Changed
- Documentation updates across README and docs to explain evaluating without `ground_truth`

## [0.1.0] - 2026-05-11

### Added
- Initial project structure
- LLM client wrapper supporting OpenAI and Anthropic
- Simple RAG pipeline implementation for evaluation
- RAGAS integration: faithfulness, answer relevancy, context recall metrics
- Custom metric helpers (hallucination detection, response length, keyword coverage)
- pytest fixtures with mock mode (no API key required)
- Sample Q&A dataset (10 Python/programming pairs)
- Comprehensive README with PyCon HK 2026 context
- Example scripts for basic eval, RAG eval, and custom metrics
- docs: getting started, metrics explained, PyCon HK 2026 slides outline
- GitHub Actions CI workflow
