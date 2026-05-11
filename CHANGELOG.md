# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
