---
title: "SDK Design"
subtitle: "Python SDK Architecture and Implementation Patterns"
version: "1.0.0"
date: "2025-11-26"
status: "Draft"
category: "Developer Experience"
tags: ["sdk", "api-design", "python", "developer-experience", "instrumentation"]
related_docs:
  - "../03-technical-architecture/api-design.md"
  - "../03-technical-architecture/data-model.md"
  - "./integration-patterns.md"
  - "./quickstart-guide.md"
---

# SDK Design

## Table of Contents

1. [Design Principles](#design-principles)
2. [Decorator API](#decorator-api)
3. [Zero-Code Instrumentation](#zero-code-instrumentation)
4. [Configuration Hierarchy](#configuration-hierarchy)
5. [Context Management](#context-management)
6. [Error Handling](#error-handling)
7. [Testing Utilities](#testing-utilities)
8. [Implementation Patterns](#implementation-patterns)
9. [Best Practices](#best-practices)
10. [Key Takeaways](#key-takeaways)

## Design Principles

The a11i Python SDK is designed around five core principles that guide all API decisions and implementation patterns:

### 1. Decorator-First API

The primary API surface is decorators, providing minimal code intrusion and maximum readability:

```python
from a11i import observe

@observe()
def my_function():
    pass
```

**Rationale**: Decorators allow instrumentation to be added with a single line, keeping business logic clean and maintainable. They provide a clear visual indicator that a function is being traced without cluttering the implementation.

### 2. Zero-Code Option

Common libraries are auto-instrumented without requiring code changes:

```python
import a11i
a11i.auto_instrument()  # Automatically instruments OpenAI, Anthropic, LangChain, etc.
```

**Rationale**: Developers should be able to add basic observability to existing applications without refactoring. This significantly reduces the barrier to adoption.

### 3. Non-Blocking Telemetry

All telemetry export is asynchronous and never blocks the main execution path:

```python
# Telemetry export happens in background
result = my_traced_function()  # Returns immediately
```

**Rationale**: Observability should never impact application performance or reliability. Failed telemetry export should never cause application errors.

### 4. Framework Agnostic

Works with any agent framework (LangChain, LlamaIndex, custom implementations):

```python
# Works with any framework
@observe()
def custom_agent_loop():
    # Your framework logic here
    pass
```

**Rationale**: Organizations use diverse agent frameworks. The SDK should work seamlessly regardless of the underlying implementation.

### 5. Type-Safe

Full type hints for IDE support, autocomplete, and static analysis:

```python
from typing import Optional, Dict, Any

@observe(
    name: str = "",
    capture_input: bool = False,
    capture_output: bool = False,
    attributes: Optional[Dict[str, Any]] = None,
)
def traced_function() -> str:
    pass
```

**Rationale**: Type safety catches errors at development time and provides excellent IDE support. It makes the API self-documenting and reduces cognitive load.

## Decorator API

The decorator API is the primary interface for manual instrumentation, designed for clarity, flexibility, and minimal boilerplate.

### Basic Function Tracing

The simplest use case - trace a function with automatic naming:

```python
from a11i import observe

@observe()
def process_document(doc: str) -> dict:
    """Process a document and extract metadata."""
    # Function implementation
    return {"title": "...", "summary": "..."}
```

**Generated Span**:
- **Name**: `process_document` (auto-derived from function name)
- **Attributes**: `code.function`, `code.filepath`, `code.lineno`
- **Timing**: Automatic start/end timestamps
- **Status**: Automatically set based on exceptions

### Customized Tracing

Advanced usage with explicit configuration:

```python
@observe(
    name="document_processor",
    capture_input=True,
    capture_output=True,
    attributes={
        "component": "preprocessing",
        "version": "2.1.0",
    },
)
def process_document(doc: str) -> dict:
    """Process a document and extract metadata."""
    return {"title": "...", "summary": "..."}
```

**Parameters**:
- `name`: Override span name (useful for better dashboard organization)
- `capture_input`: Capture function arguments as span attributes
- `capture_output`: Capture return value as span attribute
- `attributes`: Additional custom attributes to attach to the span

**Generated Span**:
```json
{
  "name": "document_processor",
  "attributes": {
    "component": "preprocessing",
    "version": "2.1.0",
    "input.doc": "Sample document text...",
    "output": "{\"title\": \"...\", \"summary\": \"...\"}"
  }
}
```

### Agent Loop Tracking

Specialized decorator for iterative agent loops:

```python
from a11i import agent_loop

@agent_loop(
    name="research_agent",
    max_iterations=20,
    track_thoughts=True,
)
async def research_agent(query: str):
    """Research agent with thought tracking."""
    iteration = 0
    done = False

    while not done and iteration < 20:
        # Think phase
        thought = await generate_thought(query)

        # Act phase
        action = await decide_action(thought)
        result = await execute_action(action)

        # Observe phase
        done = await check_completion(result)
        iteration += 1

    return final_answer
```

**Features**:
- Automatic iteration counting and tracking
- Per-iteration spans for detailed analysis
- Thought capture when `track_thoughts=True`
- Max iteration warnings when limit is reached

**Generated Trace Structure**:
```
research_agent (parent span)
├── iteration_0
│   ├── generate_thought
│   ├── decide_action
│   ├── execute_action
│   └── check_completion
├── iteration_1
│   ├── generate_thought
│   ├── decide_action
│   ├── execute_action
│   └── check_completion
└── ...
```

### Tool Call Tracking

Track individual tool invocations with categorization:

```python
from a11i import tool_call

@tool_call(name="web_search", category="retrieval")
def search_web(query: str, max_results: int = 10) -> list:
    """Search the web and return results."""
    results = search_api.query(query, limit=max_results)
    return results

@tool_call(name="calculator", category="computation")
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

@tool_call(name="database_lookup", category="data_access")
async def lookup_user(user_id: str) -> dict:
    """Look up user information from database."""
    return await db.users.find_one({"id": user_id})
```

**Categories**: Standardized tool categories for consistent dashboards:
- `retrieval`: Search, lookup, fetch operations
- `computation`: Calculations, transformations
- `data_access`: Database, API queries
- `generation`: Content creation, synthesis
- `validation`: Checking, verification

**Generated Span Attributes**:
```json
{
  "name": "web_search",
  "attributes": {
    "tool.name": "web_search",
    "tool.category": "retrieval",
    "tool.input.query": "latest AI research",
    "tool.input.max_results": 10,
    "tool.result_count": 10
  }
}
```

### Async Support

All decorators support both sync and async functions:

```python
@observe()
async def async_function():
    await some_operation()

@agent_loop()
async def async_agent_loop():
    while not done:
        await process()

@tool_call(name="async_tool")
async def async_tool(param: str):
    return await external_api.call(param)
```

## Zero-Code Instrumentation

Auto-instrumentation allows adding observability to existing applications without modifying source code.

### Environment Variable Configuration

The simplest approach - configure via environment:

```bash
# Export environment variable before running application
export A11I_AUTO_INSTRUMENT=openai,anthropic,langchain

# Run application - instrumentation happens automatically
python app.py
```

**Supported Libraries**:
- `openai`: OpenAI SDK (ChatCompletion, Embeddings, etc.)
- `anthropic`: Anthropic SDK (Messages API)
- `langchain`: LangChain framework (chains, agents, tools)
- `llama_index`: LlamaIndex framework (query engines, agents)
- `requests`: HTTP client library
- `httpx`: Modern async HTTP client
- `redis`: Redis client operations

### Import-Time Instrumentation

Programmatic instrumentation at application startup:

```python
import a11i

# Option 1: Instrument all detected libraries
a11i.auto_instrument()

# Option 2: Selective instrumentation
a11i.auto_instrument(["openai", "anthropic"])

# Option 3: Exclude specific libraries
a11i.auto_instrument(exclude=["requests"])

# Rest of application code
import openai  # Already instrumented
from langchain import LLMChain  # Already instrumented
```

**Best Practice**: Call `auto_instrument()` as early as possible in application startup, before importing instrumented libraries.

### CLI Wrapper

Command-line wrapper for zero-code instrumentation:

```bash
# Run Python script with automatic instrumentation
a11i-instrument python app.py

# Pass arguments to the script
a11i-instrument python app.py --config config.yaml

# Run Python module
a11i-instrument python -m myapp.main

# Configure libraries to instrument
a11i-instrument --libraries openai,anthropic python app.py
```

**Use Cases**:
- Adding observability to production deployments without code changes
- Testing instrumentation before committing code changes
- Quick debugging sessions

### Implementation Details

The auto-instrumentation system uses Python's import hook mechanism:

```python
import importlib
import functools
from typing import Optional, List, Callable

_INSTRUMENTORS = {
    "openai": "a11i.instrumentors.openai:OpenAIInstrumentor",
    "anthropic": "a11i.instrumentors.anthropic:AnthropicInstrumentor",
    "langchain": "a11i.instrumentors.langchain:LangChainInstrumentor",
    "llama_index": "a11i.instrumentors.llama_index:LlamaIndexInstrumentor",
    "requests": "a11i.instrumentors.requests:RequestsInstrumentor",
    "httpx": "a11i.instrumentors.httpx:HTTPXInstrumentor",
    "redis": "a11i.instrumentors.redis:RedisInstrumentor",
}

def auto_instrument(
    libraries: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> None:
    """Auto-instrument LLM and HTTP libraries.

    Args:
        libraries: Specific libraries to instrument. If None, instruments all detected.
        exclude: Libraries to exclude from instrumentation.

    Example:
        >>> a11i.auto_instrument()  # Instrument all
        >>> a11i.auto_instrument(["openai", "anthropic"])  # Selective
        >>> a11i.auto_instrument(exclude=["requests"])  # All except requests
    """
    exclude = exclude or []
    targets = libraries or list(_INSTRUMENTORS.keys())

    for lib_name in targets:
        if lib_name in exclude or lib_name not in _INSTRUMENTORS:
            continue

        # Check if library is installed
        try:
            module_name = lib_name.replace("_", "-").split(".")[0]
            importlib.import_module(module_name)
        except ImportError:
            # Library not installed, skip silently
            continue

        # Load and apply instrumentor
        module_path, class_name = _INSTRUMENTORS[lib_name].rsplit(":", 1)
        module = importlib.import_module(module_path)
        instrumentor_class = getattr(module, class_name)

        # Instantiate and instrument
        instrumentor = instrumentor_class()
        instrumentor.instrument()
```

### Example Instrumentor

Sample implementation for OpenAI library:

```python
from typing import Any, Callable
from opentelemetry import trace
from opentelemetry.trace import SpanKind
import openai

class OpenAIInstrumentor:
    """Instrumentor for OpenAI SDK."""

    def __init__(self):
        self._original_create = None
        self._tracer = trace.get_tracer(__name__)

    def instrument(self) -> None:
        """Apply instrumentation to OpenAI SDK."""
        # Patch ChatCompletion.create
        self._original_create = openai.ChatCompletion.create
        openai.ChatCompletion.create = self._traced_create

    def _traced_create(self, *args, **kwargs) -> Any:
        """Traced wrapper for ChatCompletion.create."""
        with self._tracer.start_as_current_span(
            "openai.chat.completion",
            kind=SpanKind.CLIENT,
        ) as span:
            # Capture inputs
            span.set_attribute("llm.provider", "openai")
            span.set_attribute("llm.model", kwargs.get("model", "unknown"))
            span.set_attribute("llm.messages.count", len(kwargs.get("messages", [])))

            # Call original function
            result = self._original_create(*args, **kwargs)

            # Capture outputs
            span.set_attribute("llm.usage.prompt_tokens", result.usage.prompt_tokens)
            span.set_attribute("llm.usage.completion_tokens", result.usage.completion_tokens)
            span.set_attribute("llm.usage.total_tokens", result.usage.total_tokens)

            return result
```

## Configuration Hierarchy

The SDK supports flexible configuration with a clear precedence hierarchy: code > environment > file > defaults.

### Configuration Object

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import yaml

@dataclass
class A11iConfig:
    """a11i SDK configuration with hierarchical loading.

    Configuration precedence (highest to lowest):
    1. Programmatic configuration (code)
    2. Environment variables
    3. Configuration file (a11i.yaml)
    4. Default values
    """

    # Authentication
    api_key: Optional[str] = None

    # Project settings
    project: str = "default"
    environment: str = "development"

    # Endpoint configuration
    endpoint: str = "https://ingest.a11i.dev"
    otlp_endpoint: Optional[str] = None

    # Telemetry settings
    batch_size: int = 100
    flush_interval: float = 5.0
    max_queue_size: int = 10000

    # Content capture
    capture_input: bool = False
    capture_output: bool = False
    max_content_length: int = 10000

    # Sampling
    sampling_rate: float = 1.0

    # Custom attributes
    default_attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        config_dict: Optional[Dict] = None,
        config_file: Optional[str] = None
    ) -> "A11iConfig":
        """Load config with hierarchy: code > env > file > defaults.

        Args:
            config_dict: Programmatic configuration (highest priority)
            config_file: Path to YAML config file (or a11i.yaml if exists)

        Returns:
            Configured A11iConfig instance

        Example:
            >>> config = A11iConfig.load({
            ...     "project": "my-agent",
            ...     "capture_input": True,
            ... })
        """
        config = cls()

        # 1. Load from file (lowest priority)
        if config_file:
            config._load_from_file(config_file)
        elif os.path.exists("a11i.yaml"):
            config._load_from_file("a11i.yaml")

        # 2. Load from environment (medium priority)
        config._load_from_env()

        # 3. Load from code (highest priority)
        if config_dict:
            config._load_from_dict(config_dict)

        return config

    def _load_from_file(self, path: str) -> None:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        self._load_from_dict(data)

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            "A11I_API_KEY": "api_key",
            "A11I_PROJECT": "project",
            "A11I_ENVIRONMENT": "environment",
            "A11I_ENDPOINT": "endpoint",
            "A11I_OTLP_ENDPOINT": "otlp_endpoint",
            "A11I_BATCH_SIZE": ("batch_size", int),
            "A11I_FLUSH_INTERVAL": ("flush_interval", float),
            "A11I_MAX_QUEUE_SIZE": ("max_queue_size", int),
            "A11I_CAPTURE_INPUT": ("capture_input", lambda x: x.lower() == "true"),
            "A11I_CAPTURE_OUTPUT": ("capture_output", lambda x: x.lower() == "true"),
            "A11I_MAX_CONTENT_LENGTH": ("max_content_length", int),
            "A11I_SAMPLING_RATE": ("sampling_rate", float),
        }

        for env_var, config_attr in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(config_attr, tuple):
                    attr_name, converter = config_attr
                    setattr(self, attr_name, converter(value))
                else:
                    setattr(self, config_attr, value)

    def _load_from_dict(self, data: Dict) -> None:
        """Load configuration from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
```

### Configuration File Format

Example `a11i.yaml`:

```yaml
# Authentication
api_key: "a11i_prod_..."

# Project identification
project: "research-agent"
environment: "production"

# Telemetry settings
batch_size: 500
flush_interval: 10.0
max_queue_size: 50000

# Content capture
capture_input: true
capture_output: true
max_content_length: 50000

# Sampling
sampling_rate: 0.1  # 10% sampling for high-volume production

# Default attributes attached to all spans
default_attributes:
  team: "ml-platform"
  service: "agent-orchestrator"
  version: "2.1.0"
```

### Environment Variables

Complete list of supported environment variables:

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `A11I_API_KEY` | string | API authentication key | `a11i_prod_abc123...` |
| `A11I_PROJECT` | string | Project identifier | `research-agent` |
| `A11I_ENVIRONMENT` | string | Deployment environment | `production` |
| `A11I_ENDPOINT` | string | Ingestion endpoint URL | `https://ingest.a11i.dev` |
| `A11I_OTLP_ENDPOINT` | string | OTLP endpoint (optional) | `http://localhost:4318` |
| `A11I_BATCH_SIZE` | int | Spans per batch | `100` |
| `A11I_FLUSH_INTERVAL` | float | Seconds between flushes | `5.0` |
| `A11I_MAX_QUEUE_SIZE` | int | Max queued spans | `10000` |
| `A11I_CAPTURE_INPUT` | bool | Capture function inputs | `true` |
| `A11I_CAPTURE_OUTPUT` | bool | Capture function outputs | `true` |
| `A11I_MAX_CONTENT_LENGTH` | int | Max content length | `10000` |
| `A11I_SAMPLING_RATE` | float | Sampling rate (0.0-1.0) | `1.0` |
| `A11I_AUTO_INSTRUMENT` | string | Libraries to auto-instrument | `openai,anthropic` |

### Programmatic Configuration

Highest priority configuration method:

```python
import a11i

# Initialize with configuration
a11i.init({
    "api_key": "a11i_prod_...",
    "project": "research-agent",
    "environment": "production",
    "capture_input": True,
    "capture_output": True,
    "default_attributes": {
        "team": "ml-platform",
        "version": "2.1.0",
    }
})

# Or use configuration object
config = A11iConfig.load({
    "project": "research-agent",
    "capture_input": True,
})
a11i.init(config)
```

## Context Management

Thread-safe context management for tracking agent state across async operations.

### Context Variables

```python
from contextvars import ContextVar
from typing import Optional, Dict, Any
from opentelemetry import trace
from opentelemetry.context import Context

# Thread-safe context variables
_current_agent: ContextVar[Optional[str]] = ContextVar("current_agent", default=None)
_current_iteration: ContextVar[int] = ContextVar("current_iteration", default=0)
_custom_attributes: ContextVar[Dict[str, Any]] = ContextVar("custom_attributes", default={})

class A11iContext:
    """Context manager for a11i telemetry.

    Provides thread-safe storage for agent execution context, including
    agent name, iteration count, and custom attributes.
    """

    @staticmethod
    def set_agent(name: str) -> None:
        """Set current agent name.

        Args:
            name: Agent identifier

        Example:
            >>> A11iContext.set_agent("research_agent")
        """
        _current_agent.set(name)

    @staticmethod
    def get_agent() -> Optional[str]:
        """Get current agent name.

        Returns:
            Current agent name or None
        """
        return _current_agent.get()

    @staticmethod
    def increment_iteration() -> int:
        """Increment and return current iteration count.

        Returns:
            New iteration count

        Example:
            >>> A11iContext.increment_iteration()
            1
            >>> A11iContext.increment_iteration()
            2
        """
        current = _current_iteration.get()
        new_value = current + 1
        _current_iteration.set(new_value)
        return new_value

    @staticmethod
    def get_iteration() -> int:
        """Get current iteration count without incrementing.

        Returns:
            Current iteration count
        """
        return _current_iteration.get()

    @staticmethod
    def reset_iteration() -> None:
        """Reset iteration count to 0."""
        _current_iteration.set(0)

    @staticmethod
    def set_attribute(key: str, value: Any) -> None:
        """Set custom attribute for current context.

        Args:
            key: Attribute key
            value: Attribute value (must be JSON-serializable)

        Example:
            >>> A11iContext.set_attribute("user_id", "user123")
            >>> A11iContext.set_attribute("experiment", "variant_a")
        """
        attrs = _custom_attributes.get().copy()
        attrs[key] = value
        _custom_attributes.set(attrs)

    @staticmethod
    def get_attributes() -> Dict[str, Any]:
        """Get all custom attributes.

        Returns:
            Dictionary of custom attributes
        """
        return _custom_attributes.get()

    @staticmethod
    def clear_attributes() -> None:
        """Clear all custom attributes."""
        _custom_attributes.set({})
```

### Context Manager Usage

The `agent_context` context manager sets agent name and attributes:

```python
class agent_context:
    """Context manager for agent execution.

    Automatically sets agent name and custom attributes for all spans
    created within the context.

    Example:
        >>> with agent_context("research_agent", user_id="user123"):
        ...     result = await my_agent.run()
        ...     # All spans will have agent_name and user_id attributes
    """

    def __init__(self, name: str, **attributes):
        """Initialize agent context.

        Args:
            name: Agent name
            **attributes: Custom attributes to set
        """
        self.name = name
        self.attributes = attributes
        self._previous_agent = None
        self._previous_attrs = None

    def __enter__(self):
        # Save previous state
        self._previous_agent = A11iContext.get_agent()
        self._previous_attrs = A11iContext.get_attributes().copy()

        # Set new state
        A11iContext.set_agent(self.name)
        for key, value in self.attributes.items():
            A11iContext.set_attribute(key, value)

        return self

    def __exit__(self, *args):
        # Restore previous state
        if self._previous_agent is not None:
            A11iContext.set_agent(self._previous_agent)
        A11iContext.clear_attributes()
        for key, value in self._previous_attrs.items():
            A11iContext.set_attribute(key, value)

# Async context manager
class async_agent_context:
    """Async version of agent_context for async/await usage."""

    def __init__(self, name: str, **attributes):
        self.name = name
        self.attributes = attributes
        self._previous_agent = None
        self._previous_attrs = None

    async def __aenter__(self):
        self._previous_agent = A11iContext.get_agent()
        self._previous_attrs = A11iContext.get_attributes().copy()

        A11iContext.set_agent(self.name)
        for key, value in self.attributes.items():
            A11iContext.set_attribute(key, value)

        return self

    async def __aexit__(self, *args):
        if self._previous_agent is not None:
            A11iContext.set_agent(self._previous_agent)
        A11iContext.clear_attributes()
        for key, value in self._previous_attrs.items():
            A11iContext.set_attribute(key, value)
```

### Usage Examples

```python
# Sync usage
with agent_context("research_agent", user_id="user123", session_id="session456"):
    result = research_agent.run("What is quantum computing?")
    # All spans created here will have:
    # - agent_name = "research_agent"
    # - user_id = "user123"
    # - session_id = "session456"

# Async usage
async with async_agent_context("code_generator", task_id="task789"):
    code = await code_generator.generate("implement quicksort")
    # All spans will have agent_name and task_id

# Nested contexts
with agent_context("orchestrator"):
    with agent_context("research_agent", parent="orchestrator"):
        # Inner context overrides agent name
        # parent attribute is preserved
        result = research()
```

## Error Handling

Robust error handling ensures telemetry failures never impact application reliability.

### Error Types

```python
from enum import Enum

class TelemetryError(Exception):
    """Base exception for telemetry errors."""
    pass

class ExportError(TelemetryError):
    """Error during telemetry export.

    Raised when span export to backend fails due to network issues,
    authentication errors, or backend unavailability.
    """
    pass

class ConfigurationError(TelemetryError):
    """Configuration error.

    Raised when SDK configuration is invalid or incomplete.
    """
    pass

class InstrumentationError(TelemetryError):
    """Error during instrumentation setup.

    Raised when auto-instrumentation fails to patch a library.
    """
    pass

class SerializationError(TelemetryError):
    """Error serializing span data.

    Raised when span attributes cannot be serialized to JSON.
    """
    pass
```

### Error Handler

```python
from typing import Optional, Callable
import logging

class TelemetryErrorHandler:
    """Handle telemetry errors gracefully.

    Provides configurable error handling policies to ensure telemetry
    failures never crash the application.
    """

    def __init__(
        self,
        on_error: str = "log",
        callback: Optional[Callable[[Exception, str], None]] = None
    ):
        """Initialize error handler.

        Args:
            on_error: Error handling policy - "log", "ignore", or "raise"
            callback: Optional callback function for custom error handling

        Example:
            >>> handler = TelemetryErrorHandler(on_error="log")
            >>> handler = TelemetryErrorHandler(
            ...     on_error="log",
            ...     callback=lambda e, ctx: metrics.increment("telemetry_errors")
            ... )
        """
        if on_error not in ("log", "ignore", "raise"):
            raise ValueError(f"Invalid on_error policy: {on_error}")

        self.on_error = on_error
        self.callback = callback
        self._error_count = 0
        self._last_error = None
        self._logger = logging.getLogger("a11i")

    def handle(self, error: Exception, context: str = "") -> None:
        """Handle an error according to policy.

        Args:
            error: Exception that occurred
            context: Context string describing where error occurred

        Example:
            >>> try:
            ...     export_spans(spans)
            ... except Exception as e:
            ...     handler.handle(e, "export_spans")
        """
        self._error_count += 1
        self._last_error = error

        # Execute callback if provided
        if self.callback:
            try:
                self.callback(error, context)
            except Exception as cb_error:
                self._logger.error(
                    f"Error in telemetry error callback: {cb_error}"
                )

        # Apply error policy
        if self.on_error == "raise":
            raise error
        elif self.on_error == "log":
            self._logger.warning(
                f"Telemetry error ({context}): {error}",
                exc_info=True
            )
        # "ignore" does nothing

    def get_stats(self) -> dict:
        """Get error statistics.

        Returns:
            Dictionary with error count and last error

        Example:
            >>> handler.get_stats()
            {'error_count': 3, 'last_error': 'Connection timeout'}
        """
        return {
            "error_count": self._error_count,
            "last_error": str(self._last_error) if self._last_error else None,
        }

    def reset_stats(self) -> None:
        """Reset error statistics."""
        self._error_count = 0
        self._last_error = None
```

### Usage in SDK

```python
from a11i.errors import TelemetryErrorHandler, ExportError

# Global error handler
_error_handler = TelemetryErrorHandler(on_error="log")

def export_spans(spans: list) -> None:
    """Export spans with error handling."""
    try:
        # Attempt export
        response = http_client.post("/v1/traces", json=spans)
        response.raise_for_status()
    except Exception as e:
        # Handle error gracefully
        _error_handler.handle(
            ExportError(f"Failed to export {len(spans)} spans: {e}"),
            context="export_spans"
        )

# Custom error handling
def custom_error_callback(error: Exception, context: str) -> None:
    """Custom error handling logic."""
    # Increment metrics
    metrics.increment("a11i.errors", tags={"context": context})

    # Send to error tracking service
    sentry.capture_exception(error)

handler = TelemetryErrorHandler(
    on_error="log",
    callback=custom_error_callback
)
```

## Testing Utilities

Testing utilities for verifying instrumentation and telemetry behavior.

### Mock Tracer

```python
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MockSpan:
    """Mock span for testing."""
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    parent_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        """Add span event."""
        self.events.append({
            "name": name,
            "attributes": attributes or {},
            "timestamp": datetime.now(),
        })

    def set_status(self, status: str) -> None:
        """Set span status."""
        self.status = status

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now()

class MockTracer:
    """Mock tracer for testing instrumentation.

    Captures spans for verification without requiring a backend.

    Example:
        >>> tracer = MockTracer()
        >>> with tracer.start_span("test_span") as span:
        ...     span.set_attribute("key", "value")
        >>> assert len(tracer.get_spans()) == 1
    """

    def __init__(self):
        self.spans: List[MockSpan] = []
        self._current_span: Optional[MockSpan] = None

    def start_span(self, name: str, **kwargs) -> MockSpan:
        """Start a new span.

        Args:
            name: Span name
            **kwargs: Additional span configuration

        Returns:
            MockSpan instance
        """
        span = MockSpan(
            name=name,
            attributes=kwargs.get("attributes", {}),
            parent_id=self._current_span.name if self._current_span else None
        )
        self.spans.append(span)
        self._current_span = span
        return span

    def get_spans(self) -> List[MockSpan]:
        """Get all recorded spans.

        Returns:
            List of MockSpan objects
        """
        return self.spans

    def find_span(self, name: str) -> Optional[MockSpan]:
        """Find span by name.

        Args:
            name: Span name to search for

        Returns:
            First matching MockSpan or None
        """
        return next((s for s in self.spans if s.name == name), None)

    def find_spans(self, **attributes) -> List[MockSpan]:
        """Find spans by attributes.

        Args:
            **attributes: Attribute key-value pairs to match

        Returns:
            List of matching MockSpan objects

        Example:
            >>> spans = tracer.find_spans(tool_name="web_search")
        """
        results = []
        for span in self.spans:
            if all(span.attributes.get(k) == v for k, v in attributes.items()):
                results.append(span)
        return results

    def clear(self) -> None:
        """Clear all recorded spans."""
        self.spans = []
        self._current_span = None

    def get_trace_tree(self) -> Dict[str, Any]:
        """Get hierarchical trace structure.

        Returns:
            Nested dictionary representing trace tree
        """
        root_spans = [s for s in self.spans if s.parent_id is None]

        def build_tree(span: MockSpan) -> Dict:
            children = [s for s in self.spans if s.parent_id == span.name]
            return {
                "name": span.name,
                "attributes": span.attributes,
                "children": [build_tree(child) for child in children]
            }

        return [build_tree(root) for root in root_spans]
```

### Mock LLM

```python
from typing import List, Dict, Any, Optional

class MockLLM:
    """Mock LLM for testing agent behavior.

    Returns predefined responses and records all calls for verification.

    Example:
        >>> llm = MockLLM(["Response 1", "Response 2"])
        >>> llm.invoke("prompt 1")
        'Response 1'
        >>> llm.invoke("prompt 2")
        'Response 2'
        >>> len(llm.calls)
        2
    """

    def __init__(self, responses: List[str]):
        """Initialize mock LLM.

        Args:
            responses: List of responses to return in order
        """
        self.responses = responses
        self._index = 0
        self.calls: List[Dict[str, Any]] = []

    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke LLM with prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Next response from responses list
        """
        response = self.responses[self._index % len(self.responses)]
        self._index += 1

        self.calls.append({
            "prompt": prompt,
            "kwargs": kwargs,
            "response": response,
        })

        return response

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """Async version of invoke."""
        return self.invoke(prompt, **kwargs)

    def get_call_count(self) -> int:
        """Get number of times LLM was called."""
        return len(self.calls)

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get last LLM call."""
        return self.calls[-1] if self.calls else None

    def reset(self) -> None:
        """Reset call history and response index."""
        self.calls = []
        self._index = 0
```

### Test Examples

```python
import pytest
from unittest.mock import patch
from a11i import observe
from a11i.testing import MockTracer, MockLLM

def test_observe_decorator():
    """Test that @observe decorator creates spans."""
    mock_tracer = MockTracer()

    with patch("a11i.tracer", mock_tracer):
        @observe()
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)

    assert result == 10
    assert len(mock_tracer.get_spans()) == 1
    assert mock_tracer.find_span("my_function") is not None

def test_agent_with_tools():
    """Test agent execution with tool calls."""
    mock_tracer = MockTracer()
    mock_llm = MockLLM([
        "I should search for information",
        "Based on the results, the answer is 42"
    ])

    with patch("a11i.tracer", mock_tracer):
        @observe()
        def web_search(query: str) -> str:
            return "Search results..."

        @observe()
        def agent_loop(question: str) -> str:
            thought = mock_llm.invoke(f"Question: {question}")
            results = web_search("quantum computing")
            answer = mock_llm.invoke(f"Results: {results}")
            return answer

        answer = agent_loop("What is quantum computing?")

    # Verify spans
    assert len(mock_tracer.get_spans()) == 2  # agent_loop, web_search
    assert mock_tracer.find_span("agent_loop") is not None
    assert mock_tracer.find_span("web_search") is not None

    # Verify LLM calls
    assert mock_llm.get_call_count() == 2

@pytest.mark.asyncio
async def test_async_instrumentation():
    """Test instrumentation of async functions."""
    mock_tracer = MockTracer()

    with patch("a11i.tracer", mock_tracer):
        @observe()
        async def async_function():
            return "result"

        result = await async_function()

    assert result == "result"
    assert len(mock_tracer.get_spans()) == 1
```

## Implementation Patterns

### Singleton Pattern for SDK Initialization

```python
class A11iSDK:
    """Singleton SDK instance."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init(self, config: Optional[Dict] = None) -> None:
        """Initialize SDK (idempotent)."""
        if self._initialized:
            return

        self.config = A11iConfig.load(config)
        self._setup_tracer()
        self._setup_exporter()
        self._initialized = True

    def _setup_tracer(self) -> None:
        """Configure OpenTelemetry tracer."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)

    def _setup_exporter(self) -> None:
        """Configure span exporter."""
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from a11i.export import A11iSpanExporter

        exporter = A11iSpanExporter(
            endpoint=self.config.endpoint,
            api_key=self.config.api_key,
        )

        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=self.config.max_queue_size,
            schedule_delay_millis=int(self.config.flush_interval * 1000),
            max_export_batch_size=self.config.batch_size,
        )

        trace.get_tracer_provider().add_span_processor(processor)

# Global SDK instance
_sdk = A11iSDK()

def init(config: Optional[Dict] = None) -> None:
    """Initialize a11i SDK."""
    _sdk.init(config)
```

### Lazy Initialization

```python
def get_tracer():
    """Get tracer, initializing SDK if needed."""
    if not _sdk._initialized:
        _sdk.init()
    return _sdk.tracer
```

### Decorator Implementation

```python
import functools
from typing import Callable, Optional, Dict, Any

def observe(
    name: str = "",
    capture_input: bool = False,
    capture_output: bool = False,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator to trace function execution."""

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                # Set attributes
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)

                # Capture input
                if capture_input:
                    _capture_arguments(span, func, args, kwargs)

                # Execute function
                try:
                    result = func(*args, **kwargs)

                    # Capture output
                    if capture_output:
                        span.set_attribute("output", str(result))

                    return result
                except Exception as e:
                    span.set_status("ERROR")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)

                if capture_input:
                    _capture_arguments(span, func, args, kwargs)

                try:
                    result = await func(*args, **kwargs)

                    if capture_output:
                        span.set_attribute("output", str(result))

                    return result
                except Exception as e:
                    span.set_status("ERROR")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
```

## Best Practices

### 1. Initialize Early

Initialize the SDK as early as possible in application startup:

```python
import a11i

# First thing in main()
def main():
    a11i.init({
        "project": "my-agent",
        "api_key": os.getenv("A11I_API_KEY"),
    })

    # Rest of application
    run_agent()
```

### 2. Use Context Managers

Always use context managers for agent execution to ensure proper attribute propagation:

```python
# Good
with agent_context("research_agent", user_id=user_id):
    result = agent.run(query)

# Bad
A11iContext.set_agent("research_agent")
result = agent.run(query)
# Attributes not properly cleaned up
```

### 3. Capture Selectively

Only capture inputs/outputs for debugging, not production:

```python
# Development
@observe(capture_input=True, capture_output=True)
def debug_function(data):
    pass

# Production - no content capture for performance
@observe()
def production_function(data):
    pass
```

### 4. Use Sampling in Production

Enable sampling for high-volume production systems:

```python
a11i.init({
    "sampling_rate": 0.1,  # 10% sampling
})
```

### 5. Handle Errors Gracefully

Never let telemetry errors crash your application:

```python
a11i.init({
    "on_error": "log",  # Log errors, never raise
})
```

### 6. Test Instrumentation

Always test that instrumentation produces expected spans:

```python
def test_my_agent():
    mock_tracer = MockTracer()

    with patch("a11i.tracer", mock_tracer):
        result = my_agent.run("test")

    # Verify expected spans
    assert mock_tracer.find_span("agent_loop") is not None
    assert mock_tracer.find_span("tool_call") is not None
```

## Key Takeaways

> **SDK Design Principles**
>
> 1. **Decorator-First API**: Minimal code intrusion with clear visual indicators
> 2. **Zero-Code Option**: Auto-instrumentation for instant observability
> 3. **Non-Blocking**: Async telemetry export never impacts performance
> 4. **Framework Agnostic**: Works with any agent framework or implementation
> 5. **Type-Safe**: Full type hints for IDE support and static analysis
>
> **Configuration Hierarchy**: Code > Environment > File > Defaults
>
> **Error Handling**: Telemetry failures never crash the application
>
> **Testing**: Mock tracer and LLM utilities for comprehensive testing
>
> **Best Practices**:
> - Initialize SDK early in application startup
> - Use context managers for proper attribute propagation
> - Capture content selectively (debug only, not production)
> - Enable sampling for high-volume systems
> - Always test instrumentation behavior

## Related Documentation

- **[API Design](/home/becker/projects/a11i/docs/03-technical-architecture/api-design.md)**: REST API endpoints and authentication
- **[Data Model](/home/becker/projects/a11i/docs/03-technical-architecture/data-model.md)**: Trace, span, and event schemas
- **[Integration Patterns](/home/becker/projects/a11i/docs/07-developer-experience/integration-patterns.md)**: Framework-specific integration guides
- **[Quickstart Guide](/home/becker/projects/a11i/docs/07-developer-experience/quickstart-guide.md)**: Getting started with a11i SDK

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-26
**Status**: Draft
**Next Review**: 2025-12-26
