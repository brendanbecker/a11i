---
title: SDK and Library Implementation
description: Comprehensive guide to a11i SDKs for Python and TypeScript with decorator patterns, auto-instrumentation, and testing utilities
category: Implementation
tags: [sdk, python, typescript, instrumentation, decorators, testing]
version: 1.0.0
last_updated: 2025-11-26
related:
  - ../03-core-platform/opentelemetry-integration.md
  - ../03-core-platform/span-hierarchy.md
  - ../02-architecture/technology-stack.md
---

# SDK and Library Implementation

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Python SDK](#python-sdk)
   - [Decorator Pattern](#decorator-pattern)
   - [The @observe Decorator](#the-observe-decorator)
   - [The @agent_loop Decorator](#the-agent_loop-decorator)
   - [Auto-Instrumentation](#auto-instrumentation)
   - [Configuration Hierarchy](#configuration-hierarchy)
4. [TypeScript/Node.js SDK](#typescriptnodejs-sdk)
5. [Zero-Code Instrumentation](#zero-code-instrumentation)
6. [Testing and Mocking](#testing-and-mocking)
7. [Best Practices](#best-practices)
8. [Key Takeaways](#key-takeaways)

## Overview

The a11i SDK provides lightweight, developer-friendly libraries for instrumenting AI agent applications with OpenTelemetry-based observability. The SDK is designed to minimize code changes while maximizing observability insights.

### Key Features

- **Decorator-first approach**: Simple `@observe` and `@agent_loop` decorators for Python
- **Auto-instrumentation**: Zero-code option for popular LLM libraries (OpenAI, Anthropic, LangChain)
- **Non-blocking telemetry**: Async-first design with background batching
- **Framework integrations**: Native support for LangChain, LlamaIndex, CrewAI, and more
- **Type-safe APIs**: Full TypeScript support with comprehensive type definitions
- **Testing utilities**: Mock tracers and LLM responses for unit testing

### Supported Languages

| Language | SDK Package | Status | Auto-Instrument |
|----------|-------------|--------|-----------------|
| Python | `a11i-sdk` | ✅ Stable | ✅ Yes |
| TypeScript/Node.js | `@a11i/sdk` | ✅ Stable | ✅ Yes |
| Go | `github.com/a11i/go-sdk` | 🚧 Beta | ⏳ Planned |
| Java | `dev.a11i:a11i-sdk` | 📋 Planned | 📋 Planned |

## Design Philosophy

### Core Principles

1. **Developer Experience First**: Instrumentation should feel natural and require minimal code changes
2. **Non-Intrusive**: Telemetry should never impact application performance or reliability
3. **Async by Default**: All telemetry operations are non-blocking with background batching
4. **Standards-Based**: Built on OpenTelemetry for vendor neutrality and ecosystem compatibility
5. **Progressive Enhancement**: Start simple with auto-instrumentation, add detailed tracing as needed

### Architecture Decisions

**Why Decorators?**
- Natural Python idiom for cross-cutting concerns
- Minimal code changes (single line per function)
- Clear separation of business logic and observability
- Easy to add/remove without refactoring

**Why Auto-Instrumentation?**
- Zero-code option for quick adoption
- Captures baseline metrics immediately
- Reduces implementation friction
- Complements manual instrumentation

**Why Async-First?**
- Modern AI agents are async/await based
- Non-blocking telemetry prevents performance impact
- Efficient batching of telemetry data
- Better resource utilization

## Python SDK

### Installation

```bash
# Core SDK
pip install a11i-sdk

# With auto-instrumentation for specific libraries
pip install a11i-sdk[openai]
pip install a11i-sdk[anthropic]
pip install a11i-sdk[langchain]

# All instrumentations
pip install a11i-sdk[all]
```

### Quick Start

```python
from a11i import configure, observe

# Configure once at application startup
configure(
    api_key="your-api-key",
    project="my-agent-project"
)

# Instrument your functions
@observe()
def my_llm_function(prompt: str) -> str:
    return llm.invoke(prompt)

# That's it! Traces are automatically sent to a11i
```

### Decorator Pattern

The decorator pattern is the primary interface for manual instrumentation in the Python SDK.

#### Basic Usage

```python
from a11i import observe
from opentelemetry import trace

tracer = trace.get_tracer("a11i")

@observe()
def simple_llm_call(input: str) -> str:
    """Automatically traces with OpenTelemetry span."""
    return llm.invoke(input)

@observe(name="custom_operation_name")
def custom_named_operation(data: dict) -> dict:
    """Override the span name."""
    return process_data(data)
```

#### Advanced Options

```python
@observe(
    name="custom_name",           # Override span name (default: function name)
    capture_input=True,            # Capture function arguments
    capture_output=True,           # Capture return value
    span_kind=SpanKind.CLIENT,    # OpenTelemetry span kind
    attributes={                   # Static attributes
        "component": "data_processor",
        "version": "1.0"
    }
)
async def async_llm_call(prompt: str, model: str) -> str:
    """Fully customizable observation with async support."""
    response = await llm.ainvoke(prompt, model=model)
    return response
```

#### Conditional Instrumentation

```python
import os

# Only instrument in production
should_trace = os.getenv("ENVIRONMENT") == "production"

@observe(enabled=should_trace)
def maybe_traced_function():
    """Only traced when enabled=True."""
    pass

# Sampling (trace only 10% of calls)
@observe(sample_rate=0.1)
def sampled_function():
    """Randomly traces 10% of invocations."""
    pass
```

### The @observe Decorator

#### Implementation

```python
import functools
import asyncio
from typing import Callable, Optional, Dict, Any
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

def observe(
    name: Optional[str] = None,
    capture_input: bool = False,
    capture_output: bool = False,
    span_kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
    sample_rate: float = 1.0,
):
    """
    Decorator for automatic span creation and telemetry.

    Args:
        name: Override span name (default: function name)
        capture_input: Capture function arguments as span attributes
        capture_output: Capture return value as span attribute
        span_kind: OpenTelemetry span kind (INTERNAL, CLIENT, SERVER, etc.)
        attributes: Static attributes to add to every span
        enabled: Whether instrumentation is enabled
        sample_rate: Probability of tracing (0.0 to 1.0)

    Returns:
        Decorated function with automatic tracing
    """
    def decorator(func: Callable):
        if not enabled:
            return func  # Return original function if disabled

        tracer = trace.get_tracer("a11i.sdk")
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Sampling logic
            if sample_rate < 1.0:
                import random
                if random.random() > sample_rate:
                    return await func(*args, **kwargs)

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
            ) as span:
                # Set static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Capture input
                if capture_input:
                    # Truncate to prevent huge spans
                    input_str = str(args)[:1000]
                    if len(str(args)) > 1000:
                        input_str += "... (truncated)"
                    span.set_attribute("a11i.input", input_str)

                    # Capture kwargs separately
                    if kwargs:
                        kwargs_str = str(kwargs)[:1000]
                        span.set_attribute("a11i.kwargs", kwargs_str)

                try:
                    result = await func(*args, **kwargs)

                    # Capture output
                    if capture_output:
                        output_str = str(result)[:1000]
                        if len(str(result)) > 1000:
                            output_str += "... (truncated)"
                        span.set_attribute("a11i.output", output_str)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    # Record exception details
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Sampling logic
            if sample_rate < 1.0:
                import random
                if random.random() > sample_rate:
                    return func(*args, **kwargs)

            with tracer.start_as_current_span(
                span_name,
                kind=span_kind,
            ) as span:
                # Set static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Capture input
                if capture_input:
                    input_str = str(args)[:1000]
                    if len(str(args)) > 1000:
                        input_str += "... (truncated)"
                    span.set_attribute("a11i.input", input_str)

                    if kwargs:
                        kwargs_str = str(kwargs)[:1000]
                        span.set_attribute("a11i.kwargs", kwargs_str)

                try:
                    result = func(*args, **kwargs)

                    # Capture output
                    if capture_output:
                        output_str = str(result)[:1000]
                        if len(str(result)) > 1000:
                            output_str += "... (truncated)"
                        span.set_attribute("a11i.output", output_str)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    # Record exception details
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
```

#### Usage Examples

```python
from a11i import observe
from opentelemetry.trace import SpanKind

# Example 1: Basic LLM call
@observe()
def call_openai(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Example 2: Tool execution with input/output capture
@observe(
    name="execute_search_tool",
    capture_input=True,
    capture_output=True,
    span_kind=SpanKind.CLIENT
)
async def search_web(query: str, max_results: int = 5) -> list:
    results = await search_api.query(query, limit=max_results)
    return results

# Example 3: Data processing with custom attributes
@observe(
    name="process_agent_response",
    attributes={
        "component": "response_processor",
        "version": "2.1"
    }
)
def process_response(raw_response: str) -> dict:
    return {
        "text": extract_text(raw_response),
        "confidence": calculate_confidence(raw_response),
        "metadata": extract_metadata(raw_response)
    }
```

### The @agent_loop Decorator

The `@agent_loop` decorator is specifically designed for instrumenting iterative agent execution patterns.

#### Implementation

```python
from typing import Callable, Optional
from dataclasses import dataclass
import asyncio

class MaxIterationsExceeded(Exception):
    """Raised when agent exceeds maximum iterations."""
    pass

@dataclass
class A11iContext:
    """Context object injected into agent loop functions."""
    iteration: int
    max_iterations: int
    span: Any  # OpenTelemetry span
    tracer: Any

    def increment_iteration(self):
        """Increment iteration counter and create iteration span."""
        self.iteration += 1
        if self.iteration > self.max_iterations:
            raise MaxIterationsExceeded(
                f"Agent exceeded maximum iterations: {self.max_iterations}"
            )

        # Create span for this iteration
        iteration_span = self.tracer.start_span(
            f"iteration.{self.iteration}",
            context=trace.set_span_in_context(self.span)
        )
        iteration_span.set_attribute("a11i.iteration.number", self.iteration)
        return iteration_span

def agent_loop(
    name: str,
    max_iterations: int = 100,
    capture_thoughts: bool = True,
    auto_iterate: bool = False,
):
    """
    Decorator for agent loop instrumentation.

    Args:
        name: Agent name
        max_iterations: Maximum iterations before raising exception
        capture_thoughts: Whether to capture intermediate reasoning
        auto_iterate: Automatically inject iteration tracking

    Returns:
        Decorated function with agent loop tracing
    """
    def decorator(func: Callable):
        tracer = trace.get_tracer("a11i.sdk")

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                f"invoke_agent.{name}",
                attributes={
                    "gen_ai.operation.name": "invoke_agent",
                    "a11i.agent.name": name,
                    "a11i.agent.max_iterations": max_iterations,
                }
            ) as root_span:
                # Inject loop tracking context
                ctx = A11iContext(
                    iteration=0,
                    max_iterations=max_iterations,
                    span=root_span,
                    tracer=tracer,
                )
                kwargs["_a11i_ctx"] = ctx

                try:
                    result = await func(*args, **kwargs)

                    # Record final metrics
                    root_span.set_attribute("a11i.agent.total_iterations", ctx.iteration)
                    root_span.set_attribute("a11i.agent.completed", True)
                    root_span.set_status(Status(StatusCode.OK))

                    return result

                except MaxIterationsExceeded as e:
                    root_span.set_attribute("a11i.agent.exceeded_max", True)
                    root_span.set_attribute("a11i.agent.total_iterations", ctx.iteration)
                    root_span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                except Exception as e:
                    root_span.set_attribute("a11i.agent.total_iterations", ctx.iteration)
                    root_span.record_exception(e)
                    root_span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return wrapper
    return decorator
```

#### Usage Examples

```python
from a11i import agent_loop, observe

@agent_loop(name="research_agent", max_iterations=10)
async def research_workflow(query: str, _a11i_ctx=None):
    """
    Full agent loop with iteration tracking.

    The _a11i_ctx parameter is automatically injected by the decorator.
    """
    ctx = _a11i_ctx
    done = False
    results = []

    while not done:
        # Create span for this iteration
        with ctx.increment_iteration():
            # Think phase
            thought = await think(query, context=results)

            # Act phase
            action = await act(thought)

            # Observe phase
            observation = await observe_result(action)
            results.append(observation)

            # Check completion
            done = await check_completion(observation, query)

    # Synthesize final response
    return synthesize_response(results)

# More complex example with sub-steps
@agent_loop(name="code_implementation_agent", max_iterations=20)
async def code_agent(requirements: str, _a11i_ctx=None):
    """Agent that iteratively implements and refines code."""
    ctx = _a11i_ctx
    code = ""

    for _ in range(ctx.max_iterations):
        with ctx.increment_iteration():
            # Generate or refine code
            code = await generate_code(requirements, current_code=code)

            # Test the code
            test_results = await run_tests(code)

            # Check if tests pass
            if test_results.all_passed:
                return code

            # Refine based on failures
            requirements += f"\nFix: {test_results.failures}"

    raise MaxIterationsExceeded("Could not generate passing code")
```

### Auto-Instrumentation

Auto-instrumentation allows you to capture telemetry from popular LLM libraries without modifying application code.

#### Core Implementation

```python
# a11i/auto_instrument.py
"""Auto-instrumentation for common LLM libraries."""

import importlib
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Registry of available instrumentors
INSTRUMENTORS = [
    ("openai", "a11i.instrumentors.openai", "OpenAIInstrumentor"),
    ("anthropic", "a11i.instrumentors.anthropic", "AnthropicInstrumentor"),
    ("langchain", "a11i.instrumentors.langchain", "LangChainInstrumentor"),
    ("llama_index", "a11i.instrumentors.llama_index", "LlamaIndexInstrumentor"),
    ("crewai", "a11i.instrumentors.crewai", "CrewAIInstrumentor"),
]

def auto_instrument(libraries: Optional[List[str]] = None):
    """
    Automatically instrument available libraries.

    Args:
        libraries: Specific libraries to instrument. If None, attempts all.

    Returns:
        List of successfully instrumented libraries
    """
    instrumented = []

    for lib_name, module_path, class_name in INSTRUMENTORS:
        # Skip if specific libraries requested and this isn't one
        if libraries and lib_name not in libraries:
            continue

        try:
            # Check if library is installed
            importlib.import_module(lib_name)

            # Load and apply instrumentor
            module = importlib.import_module(module_path)
            instrumentor_class = getattr(module, class_name)
            instrumentor = instrumentor_class()
            instrumentor.instrument()

            logger.info(f"✅ Instrumented {lib_name}")
            instrumented.append(lib_name)

        except ImportError:
            # Library not installed, skip silently
            logger.debug(f"Skipping {lib_name} (not installed)")

        except Exception as e:
            logger.warning(f"Failed to instrument {lib_name}: {e}")

    return instrumented

# Convenience function for sitecustomize.py
def auto_instrument_all():
    """Instrument all available libraries. For use in sitecustomize.py."""
    return auto_instrument()
```

#### OpenAI Instrumentor Example

```python
# a11i/instrumentors/openai.py
"""OpenAI library auto-instrumentation."""

import functools
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

tracer = trace.get_tracer("a11i.instrumentors.openai")

class OpenAIInstrumentor:
    """Instrumentor for OpenAI Python library."""

    def __init__(self):
        self.instrumented = False

    def instrument(self):
        """Apply instrumentation to OpenAI library."""
        if self.instrumented:
            return

        import openai

        # Instrument sync ChatCompletion.create
        original_create = openai.ChatCompletion.create

        @functools.wraps(original_create)
        def traced_create(*args, **kwargs):
            with tracer.start_as_current_span(
                "openai.chat.completions",
                kind=SpanKind.CLIENT,
            ) as span:
                # Set request attributes
                span.set_attribute("gen_ai.system", "openai")
                span.set_attribute("gen_ai.request.model", kwargs.get("model", "unknown"))
                span.set_attribute("gen_ai.request.temperature", kwargs.get("temperature", 1.0))

                if "messages" in kwargs:
                    span.set_attribute("a11i.message.count", len(kwargs["messages"]))

                try:
                    response = original_create(*args, **kwargs)

                    # Set response attributes
                    span.set_attribute("gen_ai.response.model", response.model)
                    span.set_attribute("gen_ai.usage.input_tokens",
                                      response.usage.prompt_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens",
                                      response.usage.completion_tokens)
                    span.set_attribute("gen_ai.response.finish_reason",
                                      response.choices[0].finish_reason)

                    span.set_status(Status(StatusCode.OK))
                    return response

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        openai.ChatCompletion.create = traced_create

        # Instrument async ChatCompletion.acreate
        original_acreate = openai.ChatCompletion.acreate

        @functools.wraps(original_acreate)
        async def traced_acreate(*args, **kwargs):
            with tracer.start_as_current_span(
                "openai.chat.completions",
                kind=SpanKind.CLIENT,
            ) as span:
                span.set_attribute("gen_ai.system", "openai")
                span.set_attribute("gen_ai.request.model", kwargs.get("model", "unknown"))

                try:
                    response = await original_acreate(*args, **kwargs)

                    span.set_attribute("gen_ai.response.model", response.model)
                    span.set_attribute("gen_ai.usage.input_tokens",
                                      response.usage.prompt_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens",
                                      response.usage.completion_tokens)

                    span.set_status(Status(StatusCode.OK))
                    return response

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        openai.ChatCompletion.acreate = traced_acreate

        self.instrumented = True

    def uninstrument(self):
        """Remove instrumentation (for testing)."""
        # Implementation omitted for brevity
        pass
```

#### Usage

```python
# Option 1: Explicit auto-instrumentation
from a11i import configure, auto_instrument

configure(api_key="your-key")
auto_instrument()  # Instruments all available libraries

# Option 2: Selective instrumentation
auto_instrument(libraries=["openai", "anthropic"])

# Option 3: Import side-effect (in sitecustomize.py)
import a11i
a11i.auto_instrument_all()

# Now all OpenAI/Anthropic calls are automatically traced!
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
# ↑ Automatically traced with no code changes
```

### Configuration Hierarchy

The SDK supports multiple configuration sources with clear precedence.

#### Configuration Priority

1. **Constructor arguments** (highest priority)
2. **Environment variables** (`A11I_*` prefix)
3. **Configuration file** (`a11i.yaml` or custom path)
4. **Defaults** (lowest priority)

#### Configuration Class

```python
# a11i/config.py
from dataclasses import dataclass, field
from typing import Optional
import os
import yaml
from pathlib import Path

@dataclass
class A11iConfig:
    """Configuration for a11i SDK."""

    # Required
    api_key: Optional[str] = None

    # Project settings
    project: str = "default"
    environment: str = "production"

    # Endpoint configuration
    endpoint: str = "https://ingest.a11i.dev"
    use_tls: bool = True

    # Batching and performance
    batch_size: int = 100
    flush_interval: float = 5.0
    max_queue_size: int = 10000

    # Content capture
    capture_content: bool = False
    max_content_length: int = 1000

    # Sampling
    trace_sample_rate: float = 1.0

    # Debug
    debug: bool = False

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("A11I_API_KEY"),
            project=os.getenv("A11I_PROJECT", "default"),
            environment=os.getenv("A11I_ENVIRONMENT", "production"),
            endpoint=os.getenv("A11I_ENDPOINT", "https://ingest.a11i.dev"),
            batch_size=int(os.getenv("A11I_BATCH_SIZE", "100")),
            flush_interval=float(os.getenv("A11I_FLUSH_INTERVAL", "5.0")),
            capture_content=os.getenv("A11I_CAPTURE_CONTENT", "false").lower() == "true",
            trace_sample_rate=float(os.getenv("A11I_TRACE_SAMPLE_RATE", "1.0")),
            debug=os.getenv("A11I_DEBUG", "false").lower() == "true",
        )

    @classmethod
    def from_file(cls, path: str = "a11i.yaml"):
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def load(cls, config_path: Optional[str] = None, **overrides):
        """
        Load configuration with full precedence chain.

        Priority: overrides > env vars > config file > defaults
        """
        # Start with defaults
        config = cls()

        # Apply config file if exists
        if config_path:
            file_config = cls.from_file(config_path)
        else:
            # Try default locations
            for default_path in ["a11i.yaml", ".a11i.yaml", "config/a11i.yaml"]:
                if Path(default_path).exists():
                    file_config = cls.from_file(default_path)
                    break
            else:
                file_config = cls()

        # Merge file config
        for key, value in file_config.__dict__.items():
            if value != cls.__dataclass_fields__[key].default:
                setattr(config, key, value)

        # Apply environment variables
        env_config = cls.from_env()
        for key, value in env_config.__dict__.items():
            if value is not None:
                setattr(config, key, value)

        # Apply overrides
        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)

        return config
```

#### Configuration File Example

```yaml
# a11i.yaml
api_key: "${A11I_API_KEY}"  # Can reference env vars
project: "my-agent-project"
environment: "production"

# Performance tuning
batch_size: 200
flush_interval: 10.0
max_queue_size: 50000

# Content capture
capture_content: true
max_content_length: 2000

# Sampling (trace 50% of requests)
trace_sample_rate: 0.5

# Debug mode
debug: false
```

#### Usage Examples

```python
from a11i import configure

# Option 1: Constructor arguments (highest priority)
configure(
    api_key="your-api-key",
    project="my-project",
    batch_size=500
)

# Option 2: Environment variables only
# export A11I_API_KEY=your-key
# export A11I_PROJECT=my-project
configure()  # Auto-loads from environment

# Option 3: Configuration file
configure(config_path="config/production.yaml")

# Option 4: Mixed (overrides have priority)
configure(
    config_path="a11i.yaml",
    api_key="override-key",  # Overrides value from file
    debug=True
)
```

## TypeScript/Node.js SDK

The TypeScript SDK provides similar functionality with idiomatic TypeScript patterns.

### Installation

```bash
npm install @a11i/sdk

# With auto-instrumentation
npm install @a11i/sdk-openai
npm install @a11i/sdk-anthropic
```

### Basic Usage

```typescript
import { configure, observe, agentLoop } from "@a11i/sdk";
import { trace } from "@opentelemetry/api";

// Configure once at startup
configure({
  apiKey: process.env.A11I_API_KEY,
  project: "my-agent-project"
});

// Decorator approach (requires experimental decorators)
class MyAgent {
  @observe({ name: "llm_call" })
  async callLLM(prompt: string): Promise<string> {
    return await llm.invoke(prompt);
  }

  @observe({ captureInput: true, captureOutput: true })
  async processResponse(response: string): Promise<object> {
    return {
      text: extractText(response),
      confidence: calculateConfidence(response)
    };
  }
}

// Wrapper function approach (no decorators needed)
const observedCall = observe(
  async (prompt: string) => {
    return await llm.invoke(prompt);
  },
  { name: "llm_call", captureInput: true }
);

// Agent loop
const researchAgent = agentLoop(
  async function research(query: string, ctx?: A11iContext) {
    let done = false;
    const results = [];

    while (!done) {
      ctx?.incrementIteration();

      const thought = await think(query, results);
      const action = await act(thought);
      const observation = await observeResult(action);

      results.push(observation);
      done = checkCompletion(observation, query);
    }

    return synthesizeResponse(results);
  },
  { name: "research_agent", maxIterations: 10 }
);
```

### TypeScript Types

```typescript
// Full type definitions
interface ObserveOptions {
  name?: string;
  captureInput?: boolean;
  captureOutput?: boolean;
  spanKind?: SpanKind;
  attributes?: Record<string, any>;
  enabled?: boolean;
  sampleRate?: number;
}

interface AgentLoopOptions {
  name: string;
  maxIterations?: number;
  captureThoughts?: boolean;
}

interface A11iContext {
  iteration: number;
  maxIterations: number;
  incrementIteration(): void;
}

interface A11iConfig {
  apiKey?: string;
  project?: string;
  environment?: string;
  endpoint?: string;
  batchSize?: number;
  flushInterval?: number;
  captureContent?: boolean;
  traceSampleRate?: number;
  debug?: boolean;
}

// Decorator signatures
function observe(options?: ObserveOptions): MethodDecorator;
function observe<T extends (...args: any[]) => any>(
  fn: T,
  options?: ObserveOptions
): T;

function agentLoop(options: AgentLoopOptions): MethodDecorator;
function agentLoop<T extends (...args: any[]) => any>(
  fn: T,
  options: AgentLoopOptions
): T;

function configure(config: A11iConfig): void;
```

### Auto-Instrumentation

```typescript
// Auto-instrument in application startup
import "@a11i/sdk-openai/auto";
import "@a11i/sdk-anthropic/auto";

// Or programmatic
import { autoInstrument } from "@a11i/sdk";

autoInstrument({
  libraries: ["openai", "anthropic"]
});

// Now all LLM calls are automatically traced
import OpenAI from "openai";
const openai = new OpenAI();

const response = await openai.chat.completions.create({
  model: "gpt-4",
  messages: [{ role: "user", content: "Hello" }]
});
// ↑ Automatically traced!
```

## Zero-Code Instrumentation

For rapid adoption, a11i supports completely zero-code instrumentation.

### Environment Variables Only

```bash
# Set configuration via environment
export A11I_API_KEY=your-api-key
export A11I_PROJECT=my-agent-project
export OTEL_EXPORTER_OTLP_ENDPOINT=https://ingest.a11i.dev

# Run your application (no code changes needed)
python app.py
```

### CLI Wrapper

```bash
# Install CLI tool
pip install a11i-cli

# Wrap any Python application
a11i-instrument python app.py

# With options
a11i-instrument \
  --project my-project \
  --libraries openai,anthropic \
  python app.py

# Works with any command
a11i-instrument uvicorn main:app --host 0.0.0.0
```

### sitecustomize.py Approach

```python
# sitecustomize.py (in Python path or project root)
"""Auto-instrumentation via Python import hook."""

import os

# Only auto-instrument if enabled
if os.getenv("A11I_AUTO_INSTRUMENT", "").lower() == "true":
    try:
        import a11i

        # Configure from environment
        a11i.configure()

        # Auto-instrument all available libraries
        a11i.auto_instrument_all()

        print("✅ a11i auto-instrumentation enabled")
    except Exception as e:
        print(f"⚠️  Failed to auto-instrument: {e}")
```

```bash
# Enable via environment variable
export A11I_AUTO_INSTRUMENT=true
export A11I_API_KEY=your-key

# Now any Python script is automatically instrumented
python your_script.py
```

### Docker Integration

```dockerfile
FROM python:3.11-slim

# Install application and a11i SDK
COPY requirements.txt .
RUN pip install -r requirements.txt a11i-sdk[all]

# Add sitecustomize.py for auto-instrumentation
COPY sitecustomize.py /usr/local/lib/python3.11/site-packages/

# Set environment variables
ENV A11I_AUTO_INSTRUMENT=true
ENV A11I_API_KEY=${A11I_API_KEY}
ENV A11I_PROJECT=my-agent

# Run application (automatically instrumented)
CMD ["python", "app.py"]
```

## Testing and Mocking

The a11i SDK includes comprehensive testing utilities for unit testing instrumented code.

### MockTracer

```python
from a11i.testing import MockTracer, MockLLM, A11iTestCase

class TestMyAgent(A11iTestCase):
    """Base class provides mock tracer setup."""

    def setUp(self):
        super().setUp()
        self.mock_llm = MockLLM(responses=[
            "I'll search for that information",
            "Based on my search, here are the results...",
            "Let me synthesize the findings...",
        ])

    async def test_agent_traces_correctly(self):
        """Test that agent creates expected trace structure."""
        agent = MyAgent(llm=self.mock_llm)

        # Execute agent with mock tracer
        result = await agent.run("Find information about quantum computing")

        # Assert on traces
        spans = self.get_spans()
        self.assertEqual(len(spans), 4)  # root + 3 iterations

        # Assert root span
        root_span = spans[0]
        self.assertEqual(root_span.name, "invoke_agent.my_agent")
        self.assertEqual(root_span.attributes["a11i.agent.name"], "my_agent")
        self.assertEqual(root_span.attributes["a11i.agent.total_iterations"], 3)

        # Assert iteration spans
        for i, span in enumerate(spans[1:], start=1):
            self.assertEqual(span.name, f"iteration.{i}")
            self.assertEqual(span.attributes["a11i.iteration.number"], i)

    async def test_error_handling_traces(self):
        """Test that errors are properly recorded in traces."""
        agent = MyAgent(llm=self.mock_llm)

        # Configure mock to raise error
        self.mock_llm.set_error(ValueError("API rate limit exceeded"))

        with self.assertRaises(ValueError):
            await agent.run("query")

        # Verify error recorded in span
        spans = self.get_spans()
        root_span = spans[0]
        self.assertEqual(root_span.status.status_code, StatusCode.ERROR)
        self.assertIn("API rate limit exceeded", str(root_span.events))
```

### MockLLM

```python
from a11i.testing import MockLLM

# Simple response sequence
mock = MockLLM(responses=[
    "Response 1",
    "Response 2",
    "Response 3"
])

# Returns responses in sequence
assert await mock.invoke("prompt") == "Response 1"
assert await mock.invoke("prompt") == "Response 2"

# Conditional responses
mock = MockLLM(response_map={
    "What is AI?": "AI is artificial intelligence",
    "What is ML?": "ML is machine learning",
})

assert await mock.invoke("What is AI?") == "AI is artificial intelligence"

# Error injection
mock = MockLLM(responses=["OK"])
mock.set_error(ValueError("Rate limited"), after=1)

assert await mock.invoke("") == "OK"  # First call succeeds
with pytest.raises(ValueError):
    await mock.invoke("")  # Second call raises error
```

### Test Utilities

```python
from a11i.testing import (
    assert_span_exists,
    assert_attribute_equals,
    assert_span_count,
    get_span_by_name,
)

async def test_search_tool_instrumentation():
    """Test search tool creates correct spans."""
    agent = SearchAgent()

    await agent.search("Python programming")

    # Helper assertions
    assert_span_exists("execute_search_tool")
    assert_span_count(expected=2)  # root + search

    search_span = get_span_by_name("execute_search_tool")
    assert_attribute_equals(
        search_span,
        "a11i.tool.name",
        "web_search"
    )
```

### Integration Testing

```python
import pytest
from a11i import configure
from a11i.testing import InMemoryExporter

@pytest.fixture
def a11i_exporter():
    """Fixture that captures real telemetry in-memory."""
    exporter = InMemoryExporter()
    configure(
        api_key="test-key",
        exporter=exporter,  # Use in-memory instead of OTLP
    )
    yield exporter
    exporter.clear()

async def test_end_to_end_agent(a11i_exporter):
    """Test actual agent with real instrumentation."""
    agent = RealAgent()

    result = await agent.run("test query")

    # Verify telemetry was captured
    spans = a11i_exporter.get_spans()
    assert len(spans) > 0

    # Verify span hierarchy
    root_spans = [s for s in spans if s.parent is None]
    assert len(root_spans) == 1
```

## Best Practices

### When to Use Manual vs Auto-Instrumentation

**Use Auto-Instrumentation When:**
- Getting started quickly
- Instrumenting third-party libraries
- Capturing baseline metrics
- Minimal code changes required

**Use Manual Instrumentation When:**
- Custom business logic needs tracing
- Fine-grained control over span attributes
- Capturing domain-specific context
- Agent-specific patterns (loops, tools, etc.)

### Performance Considerations

```python
# ❌ Bad: Capturing large inputs
@observe(capture_input=True)
def process_document(document: str):  # Could be megabytes!
    return analyze(document)

# ✅ Good: Capture metadata instead
@observe(attributes={"document.size": len(document)})
def process_document(document: str):
    return analyze(document)

# ❌ Bad: Synchronous instrumentation in hot path
@observe()  # Creates span overhead on every call
def tokenize_word(word: str):
    return word.split()

# ✅ Good: Sample or instrument at higher level
@observe(sample_rate=0.01)  # Only trace 1% of calls
def tokenize_word(word: str):
    return word.split()
```

### Error Handling

```python
# ✅ Good: Let exceptions propagate
@observe()
def risky_operation():
    return might_fail()  # Exception auto-recorded in span

# ❌ Bad: Swallowing exceptions
@observe()
def risky_operation():
    try:
        return might_fail()
    except Exception:
        return None  # Error not visible in traces!

# ✅ Better: Explicit error handling with context
@observe()
def risky_operation():
    try:
        return might_fail()
    except Exception as e:
        # Add context before re-raising
        trace.get_current_span().set_attribute("error.type", "retryable")
        raise
```

### Span Naming Conventions

```python
# ✅ Good: Descriptive, hierarchical names
@observe(name="agent.think")
@observe(name="agent.act.execute_tool")
@observe(name="llm.openai.chat_completion")

# ❌ Bad: Generic or inconsistent names
@observe(name="function1")
@observe(name="do_stuff")
@observe(name="AGENT_EXECUTION")  # Inconsistent casing
```

## Key Takeaways

| Aspect | Summary |
|--------|---------|
| **Decorator Pattern** | Primary interface for Python: `@observe()` and `@agent_loop()` |
| **Auto-Instrumentation** | Zero-code option via `auto_instrument()` for OpenAI, Anthropic, LangChain, etc. |
| **Configuration** | Flexible hierarchy: code > env vars > config file > defaults |
| **Non-Blocking** | All telemetry operations are async with background batching |
| **TypeScript Support** | Full SDK available for Node.js with TypeScript types |
| **Testing** | Comprehensive mocking utilities for unit testing instrumented code |
| **Performance** | Minimal overhead with sampling, batching, and async design |
| **Standards-Based** | Built on OpenTelemetry for vendor neutrality |

### Quick Reference

```python
# Minimal setup
from a11i import configure, observe, agent_loop

configure(api_key="your-key", project="my-project")

@observe()
def my_function(x):
    return process(x)

@agent_loop(name="my_agent", max_iterations=10)
async def my_agent_loop(query, _a11i_ctx=None):
    # Agent implementation
    pass
```

### Next Steps

- **Integration Guide**: See how to integrate with specific frameworks (LangChain, CrewAI, etc.)
- **Span Hierarchy**: Review span structure and semantic conventions
- **Deployment**: Learn about deployment patterns and environment configuration
- **Troubleshooting**: Common issues and debugging techniques

---

**Related Documentation:**
- [OpenTelemetry Integration](/home/becker/projects/a11i/docs/03-core-platform/opentelemetry-integration.md)
- [Span Hierarchy and Agent Loop Tracing](/home/becker/projects/a11i/docs/03-core-platform/span-hierarchy.md)
- [Technology Stack](/home/becker/projects/a11i/docs/02-architecture/technology-stack.md)
