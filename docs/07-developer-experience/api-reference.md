---
title: "API Reference: a11i Platform"
category: "Developer Experience"
tags: ["api", "rest", "sdk", "python", "typescript", "reference"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "./quickstart.md"
  - "./sdk-integration.md"
  - "../03-core-platform/opentelemetry-integration.md"
  - "../02-architecture/data-pipeline.md"
---

# API Reference: a11i Platform

## Table of Contents

- [Introduction](#introduction)
- [REST API](#rest-api)
  - [Traces API](#traces-api)
  - [Metrics API](#metrics-api)
  - [Projects API](#projects-api)
  - [Response Formats](#response-formats)
- [Python SDK](#python-sdk)
  - [Initialization](#initialization)
  - [Decorators](#decorators)
  - [Manual Instrumentation](#manual-instrumentation)
  - [Context Management](#context-management)
  - [Auto-Instrumentation](#auto-instrumentation)
  - [Utility Functions](#utility-functions)
- [TypeScript SDK](#typescript-sdk)
  - [Initialization](#typescript-initialization)
  - [Decorators and Wrappers](#decorators-and-wrappers)
  - [Manual Instrumentation](#typescript-manual-instrumentation)
- [Configuration](#configuration)
  - [Configuration File](#configuration-file)
  - [Environment Variables](#environment-variables)
- [Rate Limits and Headers](#rate-limits-and-headers)
- [Error Handling](#error-handling)
- [Key Takeaways](#key-takeaways)

## Introduction

The a11i platform provides comprehensive APIs and SDKs for instrumenting, querying, and analyzing AI agent telemetry. This reference covers:

- **REST API**: Query traces and metrics via HTTP endpoints
- **Python SDK**: Native Python integration for agent instrumentation
- **TypeScript SDK**: JavaScript/TypeScript support for Node.js applications
- **Configuration**: Environment variables and configuration file options

All APIs follow OpenTelemetry semantic conventions and use standard OTLP (OpenTelemetry Protocol) for data ingestion.

---

## REST API

The a11i REST API provides programmatic access to traces, metrics, and project configuration. All endpoints require authentication via API key.

### Authentication

Include your API key in the `Authorization` header:

```http
Authorization: Bearer a11i_sk_1234567890abcdef
```

Base URL: `https://api.a11i.dev/api/v1`

### Traces API

#### Query Traces

Retrieve traces with filtering and pagination.

**Endpoint:**
```
GET /api/v1/traces
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project` | string | Yes | Project ID to query |
| `start_time` | datetime | No | Start of time range (ISO 8601 format) |
| `end_time` | datetime | No | End of time range (ISO 8601 format) |
| `agent_name` | string | No | Filter by specific agent name |
| `status` | string | No | Filter by status: `ok` or `error` |
| `min_duration_ms` | integer | No | Minimum trace duration in milliseconds |
| `max_duration_ms` | integer | No | Maximum trace duration in milliseconds |
| `limit` | integer | No | Maximum results to return (default: 100, max: 1000) |
| `offset` | integer | No | Pagination offset (default: 0) |

**Example Request:**

```bash
curl -X GET "https://api.a11i.dev/api/v1/traces?project=my-project&start_time=2024-01-15T00:00:00Z&agent_name=research_agent&limit=50" \
  -H "Authorization: Bearer a11i_sk_1234567890abcdef"
```

**Response:**

```json
{
  "data": {
    "traces": [
      {
        "trace_id": "abc123def456",
        "agent_name": "research_agent",
        "start_time": "2024-01-15T10:30:00Z",
        "end_time": "2024-01-15T10:30:45Z",
        "duration_ms": 45000,
        "status": "ok",
        "total_tokens": 5420,
        "cost_usd": 0.25,
        "loop_iterations": 3,
        "span_count": 12,
        "model_name": "gpt-4-turbo",
        "user_id": "user_123"
      },
      {
        "trace_id": "def456ghi789",
        "agent_name": "research_agent",
        "start_time": "2024-01-15T11:15:00Z",
        "end_time": "2024-01-15T11:16:20Z",
        "duration_ms": 80000,
        "status": "error",
        "total_tokens": 8920,
        "cost_usd": 0.42,
        "loop_iterations": 5,
        "span_count": 18,
        "model_name": "gpt-4-turbo",
        "error_message": "Tool execution timeout"
      }
    ],
    "total": 1500,
    "has_more": true
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2024-01-15T12:00:00Z"
  }
}
```

#### Get Trace Details

Retrieve a complete trace with all spans in hierarchical structure.

**Endpoint:**
```
GET /api/v1/traces/{trace_id}
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `trace_id` | string | Yes | Unique trace identifier |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `include_events` | boolean | No | Include span events (default: false) |
| `include_attributes` | boolean | No | Include all span attributes (default: true) |

**Example Request:**

```bash
curl -X GET "https://api.a11i.dev/api/v1/traces/abc123def456?include_events=true" \
  -H "Authorization: Bearer a11i_sk_1234567890abcdef"
```

**Response:**

```json
{
  "data": {
    "trace_id": "abc123def456",
    "project": "my-project",
    "start_time": "2024-01-15T10:30:00Z",
    "duration_ms": 45000,
    "root_span": {
      "span_id": "span001",
      "name": "invoke_agent",
      "kind": "internal",
      "start_time": "2024-01-15T10:30:00Z",
      "duration_ms": 45000,
      "status": "ok",
      "attributes": {
        "gen_ai.operation.name": "invoke_agent",
        "a11i.agent.name": "research_agent",
        "a11i.agent.version": "1.2.0",
        "a11i.user.id": "user_123"
      },
      "events": [
        {
          "timestamp": "2024-01-15T10:30:00Z",
          "name": "agent_started",
          "attributes": {
            "input_query": "Analyze market trends"
          }
        }
      ],
      "children": [
        {
          "span_id": "span002",
          "name": "iteration_1",
          "kind": "internal",
          "start_time": "2024-01-15T10:30:01Z",
          "duration_ms": 12000,
          "attributes": {
            "a11i.loop.iteration": 1,
            "a11i.loop.phase": "think"
          },
          "children": [
            {
              "span_id": "span003",
              "name": "llm_call",
              "kind": "client",
              "start_time": "2024-01-15T10:30:02Z",
              "duration_ms": 8000,
              "attributes": {
                "gen_ai.request.model": "gpt-4-turbo",
                "gen_ai.usage.input_tokens": 850,
                "gen_ai.usage.output_tokens": 420,
                "a11i.cost.usd": 0.08
              }
            }
          ]
        },
        {
          "span_id": "span004",
          "name": "iteration_2",
          "kind": "internal",
          "start_time": "2024-01-15T10:30:15Z",
          "duration_ms": 18000,
          "children": []
        }
      ]
    }
  },
  "meta": {
    "request_id": "req-def456",
    "timestamp": "2024-01-15T12:00:00Z"
  }
}
```

### Metrics API

#### Query Aggregated Metrics

Retrieve aggregated metrics with grouping and time bucketing.

**Endpoint:**
```
GET /api/v1/metrics
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project` | string | Yes | Project ID to query |
| `metric` | string | Yes | Metric name (see table below) |
| `group_by` | string | No | Dimension to group by: `model`, `agent`, `user`, `status` |
| `interval` | string | No | Time bucket: `hour`, `day`, `week`, `month` |
| `start_time` | datetime | No | Start of time range (ISO 8601) |
| `end_time` | datetime | No | End of time range (ISO 8601) |
| `filter` | object | No | Additional filters (JSON object) |

**Supported Metrics:**

| Metric Name | Description | Unit |
|-------------|-------------|------|
| `ai.token_usage` | Total tokens consumed | count |
| `ai.cost` | Estimated cost | USD |
| `ai.latency` | Agent invocation latency | milliseconds |
| `ai.context_saturation` | Context window utilization | percentage |
| `ai.tool_error_rate` | Tool execution failure rate | percentage |
| `ai.loop_iterations` | Agent reasoning iterations | count |
| `ai.success_rate` | Successful agent invocations | percentage |

**Example Request:**

```bash
curl -X GET "https://api.a11i.dev/api/v1/metrics?project=my-project&metric=cost&group_by=agent&interval=day&start_time=2024-01-10T00:00:00Z&end_time=2024-01-16T00:00:00Z" \
  -H "Authorization: Bearer a11i_sk_1234567890abcdef"
```

**Response:**

```json
{
  "data": {
    "metric": "cost",
    "group_by": "agent",
    "interval": "day",
    "data": [
      {
        "timestamp": "2024-01-14T00:00:00Z",
        "groups": {
          "research_agent": 125.50,
          "summarization_agent": 45.30,
          "code_agent": 78.20
        },
        "total": 249.00
      },
      {
        "timestamp": "2024-01-15T00:00:00Z",
        "groups": {
          "research_agent": 142.30,
          "summarization_agent": 52.10,
          "code_agent": 89.40
        },
        "total": 283.80
      }
    ],
    "summary": {
      "total": 532.80,
      "average_per_day": 266.40,
      "min": 249.00,
      "max": 283.80
    }
  },
  "meta": {
    "request_id": "req-ghi789",
    "timestamp": "2024-01-15T12:00:00Z"
  }
}
```

### Projects API

#### List Projects

Retrieve all projects accessible to the authenticated user.

**Endpoint:**
```
GET /api/v1/projects
```

**Response:**

```json
{
  "data": {
    "projects": [
      {
        "id": "my-project",
        "name": "My Production Project",
        "created_at": "2024-01-01T00:00:00Z",
        "environment": "production",
        "trace_count": 125000,
        "monthly_cost": 1250.50
      },
      {
        "id": "staging-project",
        "name": "Staging Environment",
        "created_at": "2024-01-05T00:00:00Z",
        "environment": "staging",
        "trace_count": 8500,
        "monthly_cost": 85.30
      }
    ]
  },
  "meta": {
    "request_id": "req-jkl012",
    "timestamp": "2024-01-15T12:00:00Z"
  }
}
```

### Response Formats

#### Success Response

All successful API responses follow this structure:

```json
{
  "data": {
    // Response payload specific to endpoint
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2024-01-15T12:00:00Z",
    "version": "v1"
  }
}
```

#### Error Response

Error responses include structured error information:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required parameter: project",
    "details": {
      "parameter": "project",
      "expected": "string",
      "received": null
    }
  },
  "meta": {
    "request_id": "req-abc123",
    "timestamp": "2024-01-15T12:00:00Z"
  }
}
```

**Error Codes:**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request or missing parameters |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions for resource |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Server-side error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

---

## Python SDK

The a11i Python SDK provides idiomatic Python integration for agent instrumentation.

### Installation

```bash
pip install a11i-sdk
```

### Initialization

Initialize the a11i SDK at application startup.

```python
import a11i

a11i.init(
    api_key: str = None,           # API key (or A11I_API_KEY env var)
    project: str = "default",       # Project name
    environment: str = "dev",       # Environment tag (dev, staging, prod)
    endpoint: str = None,           # Custom endpoint (default: https://ingest.a11i.dev)
    batch_size: int = 100,          # Export batch size
    flush_interval: float = 5.0,    # Flush interval in seconds
    service_name: str = None,       # Service name for telemetry
    service_version: str = None,    # Service version
)
```

**Example:**

```python
import a11i
import os

# Initialize with environment variables
a11i.init(
    api_key=os.getenv("A11I_API_KEY"),
    project="my-ai-project",
    environment="production",
    service_name="research-agent",
    service_version="1.2.0"
)
```

### Decorators

The Python SDK provides convenient decorators for automatic instrumentation.

#### @observe Decorator

Track any function execution as a span.

```python
from a11i import observe

@observe(
    name: str = None,              # Span name (default: function name)
    capture_input: bool = False,   # Log input arguments as attributes
    capture_output: bool = False,  # Log return value as attribute
    attributes: dict = None,       # Custom attributes to add
    kind: str = "internal",        # Span kind (internal, client, server)
)
def my_function(arg1, arg2):
    """Function to instrument"""
    return result
```

**Example:**

```python
from a11i import observe

@observe(
    name="process_query",
    capture_input=True,
    attributes={"component": "query_processor"}
)
def process_user_query(query: str) -> dict:
    """Process user query and return structured response"""
    # Processing logic
    return {"response": "...", "confidence": 0.95}

# Function is automatically traced
result = process_user_query("What is the weather?")
```

#### @agent_loop Decorator

Instrument agent reasoning loops with iteration tracking.

```python
from a11i import agent_loop

@agent_loop(
    name: str,                     # Agent name (required)
    max_iterations: int = 100,     # Maximum loop iterations
    track_thoughts: bool = True,   # Capture agent thinking process
    track_actions: bool = True,    # Capture agent actions
)
async def my_agent(query: str):
    """Agent implementation with Think→Act→Observe loop"""
    while not done:
        # Agent logic
        yield action
```

**Example:**

```python
from a11i import agent_loop
import asyncio

@agent_loop(
    name="research_agent",
    max_iterations=10,
    track_thoughts=True
)
async def research_agent(topic: str):
    """Research agent that iteratively gathers information"""
    context = []

    for iteration in range(5):
        # Think phase
        thought = await generate_search_query(topic, context)

        # Act phase
        results = await search_web(thought)

        # Observe phase
        context.extend(results)

        yield {"iteration": iteration, "findings": len(results)}

    return {"summary": await summarize(context)}

# Run agent with automatic iteration tracking
result = await research_agent("AI observability trends")
```

#### @tool_call Decorator

Track agent tool executions.

```python
from a11i import tool_call

@tool_call(
    name: str,                     # Tool name (required)
    category: str = None,          # Tool category (retrieval, api, database, etc.)
    capture_input: bool = True,    # Capture tool parameters
    capture_output: bool = True,   # Capture tool results
)
def my_tool(*args, **kwargs):
    """Tool implementation"""
    return result
```

**Example:**

```python
from a11i import tool_call
import requests

@tool_call(
    name="web_search",
    category="retrieval",
    capture_input=True
)
def search_web(query: str, max_results: int = 10) -> list:
    """Search the web and return results"""
    response = requests.get(
        "https://api.search.com/search",
        params={"q": query, "limit": max_results}
    )
    return response.json()["results"]

# Tool execution is automatically traced
results = search_web("AI agents", max_results=5)
```

### Manual Instrumentation

Create spans manually for fine-grained control.

```python
from a11i import tracer

# Context manager approach
with tracer.start_span("custom_operation") as span:
    span.set_attribute("custom.attribute", "value")
    span.set_attribute("iteration", 1)

    # Perform operation
    result = process_data()

    # Add events
    span.add_event("processing_complete", {
        "records_processed": len(result),
        "duration_ms": 150
    })

    # Set status
    if result.is_valid:
        span.set_status("ok")
    else:
        span.set_status("error", "Validation failed")

# Programmatic approach
span = tracer.start_span("another_operation")
span.set_attribute("key", "value")
# ... perform work ...
span.end()
```

**Nested Spans:**

```python
from a11i import tracer

with tracer.start_span("parent_operation") as parent:
    parent.set_attribute("total_items", 100)

    for i in range(10):
        with tracer.start_span(f"process_batch_{i}") as child:
            child.set_attribute("batch_number", i)
            process_batch(i)
```

### Context Management

Manage trace context and agent state.

```python
from a11i import A11iContext

# Set agent name for all subsequent spans
A11iContext.set_agent("my_agent")

# Set custom attributes that persist across spans
A11iContext.set_attribute("user_id", "user123")
A11iContext.set_attribute("session_id", "session456")

# Increment loop iteration counter
iteration = A11iContext.increment_iteration()  # Returns 1, 2, 3, ...

# Get current iteration
current = A11iContext.get_iteration()

# Clear agent context (useful when switching agents)
A11iContext.clear()
```

**Example with Context:**

```python
from a11i import A11iContext, observe

@observe(name="agent_execution")
async def run_agent(user_query: str, user_id: str):
    # Set context that applies to all child spans
    A11iContext.set_agent("research_agent")
    A11iContext.set_attribute("user_id", user_id)
    A11iContext.set_attribute("query_type", "research")

    for i in range(5):
        iteration = A11iContext.increment_iteration()
        result = await process_iteration(user_query, iteration)

    return result

# All spans created during execution inherit the context
await run_agent("What are AI trends?", "user_123")
```

### Auto-Instrumentation

Automatically instrument popular libraries.

```python
from a11i import auto_instrument

# Auto-instrument all supported libraries
auto_instrument()

# Auto-instrument specific libraries
auto_instrument(libraries=["openai", "langchain", "anthropic"])
```

**Supported Libraries:**

- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `langchain` - LangChain framework
- `llama-index` - LlamaIndex framework
- `crewai` - CrewAI framework
- `autogen` - Microsoft AutoGen
- `semantic-kernel` - Microsoft Semantic Kernel
- `requests` - HTTP requests library
- `httpx` - Modern HTTP client

**Example:**

```python
import a11i
from openai import OpenAI

# Initialize and auto-instrument
a11i.init(project="my-project")
a11i.auto_instrument(libraries=["openai"])

# OpenAI calls are automatically traced
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Automatically creates span with token usage, cost, model info
```

### Utility Functions

Utility functions for SDK management.

```python
from a11i import flush, shutdown, get_trace_url, verify_connection

# Force flush pending telemetry (blocks until complete)
flush()

# Graceful shutdown (flushes and closes connections)
shutdown()

# Get URL to view current trace in a11i dashboard
trace_url = get_trace_url()
print(f"View trace: {trace_url}")  # https://app.a11i.dev/traces/abc123

# Verify connection to a11i platform
status = verify_connection()
print(status)
# {"connected": True, "endpoint": "https://ingest.a11i.dev", "latency_ms": 45}
```

**Example with Error Handling:**

```python
import a11i
import atexit

# Initialize SDK
a11i.init(project="my-project")

# Register shutdown handler
atexit.register(a11i.shutdown)

# Verify connection at startup
status = a11i.verify_connection()
if not status["connected"]:
    print(f"Warning: Cannot connect to a11i at {status['endpoint']}")
    print("Telemetry will be buffered locally")

# Your application code
run_agent()

# Flush telemetry before exiting
a11i.flush()
```

---

## TypeScript SDK

The a11i TypeScript SDK provides first-class support for JavaScript and TypeScript applications.

### Installation

```bash
npm install @a11i/sdk
# or
yarn add @a11i/sdk
```

### TypeScript Initialization

Initialize the SDK at application startup.

```typescript
import { init } from '@a11i/sdk';

init({
  apiKey?: string,              // API key (or A11I_API_KEY env var)
  project?: string,             // Project name (default: "default")
  environment?: string,         // Environment (default: "development")
  endpoint?: string,            // Custom endpoint
  batchSize?: number,           // Batch size (default: 100)
  flushInterval?: number,       // Flush interval in ms (default: 5000)
  serviceName?: string,         // Service name
  serviceVersion?: string,      // Service version
});
```

**Example:**

```typescript
import { init } from '@a11i/sdk';

// Initialize with configuration
init({
  apiKey: process.env.A11I_API_KEY,
  project: 'my-typescript-project',
  environment: 'production',
  serviceName: 'typescript-agent',
  serviceVersion: '2.1.0',
  batchSize: 50,
  flushInterval: 3000
});
```

### Decorators and Wrappers

TypeScript SDK supports both decorators (experimental) and wrapper functions.

#### Decorator Syntax (Experimental)

```typescript
import { observe, agentLoop, toolCall } from '@a11i/sdk';

class MyAgent {
  @observe({ name: 'llm_call', captureInput: true })
  async callLLM(prompt: string): Promise<string> {
    const response = await llm.invoke(prompt);
    return response.text;
  }

  @toolCall({ name: 'database_query', category: 'database' })
  async queryDatabase(query: string): Promise<any[]> {
    return await db.execute(query);
  }
}
```

**Note:** Decorators require `experimentalDecorators` in `tsconfig.json`:

```json
{
  "compilerOptions": {
    "experimentalDecorators": true
  }
}
```

#### Wrapper Functions

Wrap existing functions for instrumentation.

```typescript
import { observe, agentLoop, toolCall } from '@a11i/sdk';

// Wrap a function
const tracedFunction = observe(
  async (input: string) => {
    return await processInput(input);
  },
  {
    name: 'process_input',
    captureInput: true,
    captureOutput: true
  }
);

// Use wrapped function
const result = await tracedFunction('user query');

// Agent loop wrapper
const myAgent = agentLoop(
  async function* (query: string) {
    let done = false;
    let iteration = 0;

    while (!done && iteration < 10) {
      const action = await think(query);
      const result = await act(action);
      done = await evaluate(result);

      iteration++;
      yield { iteration, result };
    }
  },
  {
    name: 'research_agent',
    maxIterations: 10,
    trackThoughts: true
  }
);

// Run agent
for await (const step of myAgent('research topic')) {
  console.log(`Iteration ${step.iteration}:`, step.result);
}
```

#### Tool Call Wrapper

```typescript
import { toolCall } from '@a11i/sdk';

// Wrap tool function
const webSearch = toolCall(
  async (query: string, maxResults: number = 10): Promise<SearchResult[]> => {
    const response = await fetch(`https://api.search.com?q=${query}&limit=${maxResults}`);
    return await response.json();
  },
  {
    name: 'web_search',
    category: 'retrieval',
    captureInput: true,
    captureOutput: true
  }
);

// Use wrapped tool
const results = await webSearch('AI observability', 5);
```

### TypeScript Manual Instrumentation

Create spans manually for custom instrumentation.

```typescript
import { tracer } from '@a11i/sdk';

// Context manager pattern
async function processData() {
  const span = tracer.startSpan('process_data');

  try {
    span.setAttribute('data_size', 1000);

    // Perform processing
    const result = await heavyProcessing();

    // Add event
    span.addEvent('processing_milestone', {
      progress: 50,
      records: 500
    });

    // Nested span
    const childSpan = tracer.startSpan('validation');
    await validateResults(result);
    childSpan.end();

    span.setStatus({ code: 'ok' });
    return result;

  } catch (error) {
    span.setStatus({
      code: 'error',
      message: error.message
    });
    throw error;

  } finally {
    span.end();
  }
}
```

**Async Context Propagation:**

```typescript
import { tracer, context } from '@a11i/sdk';

async function parentOperation() {
  return await tracer.withSpan('parent', async (span) => {
    span.setAttribute('parent_attr', 'value');

    // Context is automatically propagated to child operations
    const result = await childOperation();

    return result;
  });
}

async function childOperation() {
  return await tracer.withSpan('child', async (span) => {
    // This span is automatically a child of 'parent' span
    span.setAttribute('child_attr', 'value');

    return 'result';
  });
}
```

---

## Configuration

### Configuration File

Create `a11i.yaml` in your project root for declarative configuration.

```yaml
# a11i.yaml - Configuration file for a11i SDK

# Authentication
api_key: ${A11I_API_KEY}  # Use environment variable (recommended)
# api_key: "a11i_sk_hardcoded_key"  # Or hardcode (not recommended)

# Project settings
project: my-ai-project
environment: production

# Endpoint configuration
endpoint: https://ingest.a11i.dev
otlp_endpoint: https://otlp.a11i.dev:4317

# Telemetry settings
batch_size: 100              # Number of spans to batch before export
flush_interval: 5.0          # Flush interval in seconds
max_queue_size: 10000        # Maximum queue size before dropping spans

# Content capture settings
capture_input: false         # Capture function inputs (may contain PII)
capture_output: false        # Capture function outputs (may contain PII)
max_content_length: 10000    # Maximum content length to capture (bytes)

# Sampling configuration
sampling_rate: 1.0           # 1.0 = 100%, 0.1 = 10%, 0.01 = 1%

# Service identification
service_name: my-ai-service
service_version: 1.2.0

# Default attributes (added to all spans)
default_attributes:
  service.name: my-ai-service
  deployment.region: us-east-1
  deployment.environment: production
  team: ai-platform

# Auto-instrumentation
auto_instrument:
  enabled: true
  libraries:
    - openai
    - anthropic
    - langchain

# Feature flags
features:
  track_token_usage: true
  track_cost: true
  track_context_saturation: true
  redact_pii: false
```

**Load Configuration:**

```python
# Python
import a11i
a11i.init(config_file="a11i.yaml")
```

```typescript
// TypeScript
import { init } from '@a11i/sdk';
init({ configFile: 'a11i.yaml' });
```

### Environment Variables

Configure the SDK using environment variables.

| Variable | Description | Default |
|----------|-------------|---------|
| `A11I_API_KEY` | API key for authentication | None (required) |
| `A11I_PROJECT` | Project name | `"default"` |
| `A11I_ENVIRONMENT` | Environment tag | `"development"` |
| `A11I_ENDPOINT` | API endpoint URL | `"https://ingest.a11i.dev"` |
| `A11I_BATCH_SIZE` | Export batch size | `100` |
| `A11I_FLUSH_INTERVAL` | Flush interval in seconds | `5.0` |
| `A11I_CAPTURE_INPUT` | Capture function inputs | `false` |
| `A11I_CAPTURE_OUTPUT` | Capture function outputs | `false` |
| `A11I_MAX_CONTENT_LENGTH` | Max content length (bytes) | `10000` |
| `A11I_SAMPLING_RATE` | Sampling rate (0.0-1.0) | `1.0` |
| `A11I_AUTO_INSTRUMENT` | Comma-separated library list | `None` |
| `A11I_DEBUG` | Enable debug logging | `false` |
| `A11I_SERVICE_NAME` | Service name for telemetry | None |
| `A11I_SERVICE_VERSION` | Service version | None |

**Example `.env` file:**

```bash
# .env file for a11i configuration

A11I_API_KEY=a11i_sk_1234567890abcdef
A11I_PROJECT=my-production-project
A11I_ENVIRONMENT=production
A11I_BATCH_SIZE=50
A11I_FLUSH_INTERVAL=3.0
A11I_SAMPLING_RATE=1.0
A11I_AUTO_INSTRUMENT=openai,anthropic,langchain
A11I_SERVICE_NAME=research-agent-service
A11I_SERVICE_VERSION=2.1.0
A11I_DEBUG=false
```

**Priority Order:**

Configuration is loaded in the following priority order (highest to lowest):

1. Programmatic configuration (passed to `init()`)
2. Environment variables
3. Configuration file (`a11i.yaml`)
4. Default values

---

## Rate Limits and Headers

### Rate Limits

Rate limits vary by subscription tier.

| Tier | Requests/min | Ingestion/sec | Monthly Traces |
|------|-------------|---------------|----------------|
| **Free** | 60 | 100 spans | 100,000 |
| **Pro** | 300 | 1,000 spans | 1,000,000 |
| **Enterprise** | Unlimited | Custom | Unlimited |

### Rate Limit Headers

All API responses include rate limit information:

```http
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 295
X-RateLimit-Reset: 1705315800
X-RateLimit-Retry-After: 45
```

**Header Descriptions:**

- `X-RateLimit-Limit`: Maximum requests allowed in the current window
- `X-RateLimit-Remaining`: Remaining requests in the current window
- `X-RateLimit-Reset`: Unix timestamp when the rate limit resets
- `X-RateLimit-Retry-After`: Seconds to wait before retrying (only on 429 errors)

### Handling Rate Limits

```python
import requests
import time

def query_traces_with_retry(project: str, **params):
    """Query traces with automatic retry on rate limit"""
    max_retries = 3

    for attempt in range(max_retries):
        response = requests.get(
            "https://api.a11i.dev/api/v1/traces",
            params={"project": project, **params},
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code == 429:
            # Rate limited
            retry_after = int(response.headers.get("X-RateLimit-Retry-After", 60))
            print(f"Rate limited. Retrying in {retry_after} seconds...")
            time.sleep(retry_after)
            continue

        response.raise_for_status()
        return response.json()

    raise Exception("Max retries exceeded")
```

---

## Error Handling

### SDK Error Handling

The SDK handles errors gracefully and provides detailed error messages.

```python
from a11i import init, A11iError, ConnectionError, ConfigurationError

try:
    init(
        api_key="invalid_key",
        project="my-project"
    )
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle invalid configuration

except ConnectionError as e:
    print(f"Connection error: {e}")
    # Handle connection failure

except A11iError as e:
    print(f"General a11i error: {e}")
    # Handle other errors
```

### API Error Handling

Handle API errors using standard HTTP status codes and error responses.

```python
import requests

def safe_api_call():
    try:
        response = requests.get(
            "https://api.a11i.dev/api/v1/traces",
            params={"project": "my-project"},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            error = e.response.json()["error"]
            print(f"Invalid request: {error['message']}")
            print(f"Details: {error['details']}")

        elif e.response.status_code == 401:
            print("Authentication failed. Check your API key.")

        elif e.response.status_code == 429:
            retry_after = e.response.headers.get("X-RateLimit-Retry-After")
            print(f"Rate limited. Retry after {retry_after} seconds.")

        elif e.response.status_code >= 500:
            print("Server error. Please retry later.")

        raise

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        raise
```

### Graceful Degradation

The SDK continues operating even if telemetry export fails.

```python
import a11i

# Initialize with fallback behavior
a11i.init(
    project="my-project",
    fallback_mode="buffer",  # Buffer spans locally on export failure
    max_buffer_size=50000,   # Maximum local buffer size
    offline_mode=False       # Don't fail if connection unavailable
)

# Your application code runs normally
# Telemetry is buffered if export fails
run_agent()

# Attempt to flush buffered telemetry
try:
    a11i.flush()
except Exception as e:
    print(f"Warning: Could not flush telemetry: {e}")
    # Application continues normally
```

---

## Key Takeaways

### REST API
- **Authentication**: All requests require API key in `Authorization` header
- **Filtering**: Comprehensive query parameters for traces and metrics
- **Pagination**: Use `limit` and `offset` for large result sets
- **Time Ranges**: Use ISO 8601 format for `start_time` and `end_time`
- **Rate Limits**: Monitor `X-RateLimit-*` headers and handle 429 responses

### Python SDK
- **Quick Start**: Single `init()` call to get started
- **Decorators**: `@observe`, `@agent_loop`, `@tool_call` for automatic instrumentation
- **Auto-Instrumentation**: `auto_instrument()` for zero-code library integration
- **Context Management**: `A11iContext` for agent state and attributes
- **Utility Functions**: `flush()`, `shutdown()`, `get_trace_url()` for SDK management

### TypeScript SDK
- **Modern JavaScript**: First-class TypeScript support with type definitions
- **Decorators**: Experimental decorator support for classes
- **Wrapper Functions**: Functional approach for instrumenting existing code
- **Async Support**: Native async/await and generator function support

### Configuration
- **Flexible**: Environment variables, config file, or programmatic configuration
- **Hierarchical**: Programmatic > env vars > config file > defaults
- **Secure**: API keys via environment variables (never hardcode)
- **Default Attributes**: Apply common attributes to all spans

### Best Practices
- Use environment variables for sensitive configuration (API keys)
- Enable auto-instrumentation for supported libraries
- Set meaningful `project` and `environment` values
- Monitor rate limits and implement retry logic
- Call `shutdown()` on application exit to flush telemetry
- Use `capture_input`/`capture_output` carefully (may contain PII)
- Set `sampling_rate < 1.0` for high-volume production environments

---

**Related Documentation:**
- [Quickstart Guide](./quickstart.md) - Get started with a11i in 5 minutes
- [SDK Integration](./sdk-integration.md) - Deep-dive SDK integration patterns
- [OpenTelemetry Integration](../03-core-platform/opentelemetry-integration.md) - OTLP protocol details
- [Data Pipeline](../02-architecture/data-pipeline.md) - How telemetry flows through a11i

---

*Document Status: Draft | Last Updated: 2025-11-26 | Maintained by: a11i Documentation Team*
