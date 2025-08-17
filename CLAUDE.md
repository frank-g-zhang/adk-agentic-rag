# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an Agentic RAG (Retrieval-Augmented Generation) agent built with Google's Agent Development Kit (ADK). It implements a production-ready data ingestion pipeline with Vertex AI integration, supporting both Vertex AI Search and Vertex AI Vector Search as datastore options.

## Architecture

The agent follows a modular RAG architecture:
- **Retriever Layer**: Abstracts data retrieval from either Vertex AI Search or Vector Search
- **Compression Layer**: Uses Vertex AI Rank for re-ranking retrieved documents
- **Format Layer**: Templates retrieved documents for LLM consumption
- **Agent Layer**: ADK-based agent with retrieval tools

## Key Components

### Core Agent (`app/agent.py`)
- **Root Agent**: ADK Agent with retrieval capabilities using `gemini-2.0-flash`
- **Embedding Model**: Vertex AI `text-embedding-005` for document embeddings
- **Retrieval Tool**: `retrieve_docs()` function that fetches and formats relevant documents

### Retriever Abstraction (`app/retrievers.py`)
- **Vertex AI Search Retriever**: For managed search datastore (when `cookiecutter.datastore_type == "vertex_ai_search"`)
- **Vector Search Retriever**: For custom vector search index (when `cookiecutter.datastore_type == "vertex_ai_vector_search"`)
- **Mock Fallback**: Graceful degradation with MagicMock for testing

### Document Formatting (`app/templates.py`)
- **Jinja2 Templates**: Standardized document formatting for LLM context
- **Context Structure**: Documents wrapped in `<Document>` tags with indexing

## Environment Configuration

Required environment variables:
```bash
# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=1

# Datastore (Vertex AI Search)
DATA_STORE_REGION=us
DATA_STORE_ID=your-datastore-id

# OR Datastore (Vector Search)
VECTOR_SEARCH_INDEX=your-index-name
VECTOR_SEARCH_INDEX_ENDPOINT=your-endpoint-name
VECTOR_SEARCH_BUCKET=your-bucket-name
```

## Development Commands

### Setup & Installation
```bash
# Install ADK CLI
google-cloud-sdk/bin/gcloud components install adk

# Set up authentication
gcloud auth application-default login

# Install Python dependencies
pip install google-adk langchain-google-community langchain-google-vertexai
```

### Testing
```bash
# Run integration tests
python -m pytest tests/integration/test_agent.py -v

# Test with mock retriever
INTEGRATION_TEST=TRUE python -m pytest tests/integration/test_agent.py
```

### Running the Agent
```bash
# Run locally with ADK
adk run root_agent

# Run with web UI
adk web

# Run specific test
python -c "from app.agent import root_agent; print('Agent loaded successfully')"
```

### Development Workflow
1. **Configuration**: Set up `.env` file with required variables
2. **Datastore Selection**: Choose between Vertex AI Search or Vector Search
3. **Data Ingestion**: Configure data pipeline through Vertex AI Pipelines
4. **Testing**: Use notebooks in `notebooks/` for validation
5. **Deployment**: Use Terraform for infrastructure deployment

## Testing Patterns

- **Integration Tests**: Located in `tests/integration/test_agent.py`
- **Mock Testing**: Uses MagicMock for retriever/compressor simulation
- **Streaming Tests**: Validates SSE event streaming functionality
- **Notebooks**: Interactive testing in `notebooks/adk_app_testing.ipynb`

## Data Pipeline

The agent supports automated data ingestion through:
- **Vertex AI Pipelines**: Scheduled runs and recurring executions
- **Custom Embeddings**: Vertex AI Embeddings integration
- **CI/CD Integration**: Automated deployment with Terraform
- **Scalability**: Supports terabyte-scale data with BigQuery/Dataflow

## Key Files
- `app/agent.py`: Main agent configuration and retrieval logic
- `app/retrievers.py`: Retriever and compressor factory functions
- `app/templates.py`: Document formatting templates
- `tests/integration/test_agent.py`: Integration test suite
- `notebooks/`: Interactive development and testing notebooks