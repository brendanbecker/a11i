---
title: OpenTelemetry Integration
description: Comprehensive guide to OpenTelemetry observability for a11i agents with GenAI semantic conventions
category: Core Platform
tags: [observability, opentelemetry, tracing, metrics, genai]
version: 1.0.0
last_updated: 2025-11-26
---

# OpenTelemetry Integration

## Table of Contents

- [Overview](#overview)
- [GenAI Semantic Conventions](#genai-semantic-conventions)
  - [Standard Attributes Reference](#standard-attributes-reference)
  - [a11i Custom Extensions](#a11i-custom-extensions)
- [Implementation Patterns](#implementation-patterns)
  - [Context Propagation](#context-propagation)
  - [Multi-Agent Correlation](#multi-agent-correlation)
  - [Instrumentation Examples](#instrumentation-examples)
- [Provider-Specific Attributes](#provider-specific-attributes)
  - [OpenAI](#openai)
  - [AWS Bedrock](#aws-bedrock)
  - [Azure AI](#azure-ai)
  - [Anthropic](#anthropic)
- [Migration Strategy for Evolving Standards](#migration-strategy-for-evolving-standards)
  - [Stability Levels](#stability-levels)
  - [Forward Compatibility Patterns](#forward-compatibility-patterns)
- [OTel Collector Configuration](#otel-collector-configuration)
- [Query Patterns and Analytics](#query-patterns-and-analytics)
- [Key Takeaways](#key-takeaways)
- [Cross-References](#cross-references)

## Overview

OpenTelemetry (OTel) provides standardized observability for a11i's multi-agent system. This document covers the implementation of OTel tracing with GenAI semantic conventions (v1.38+), enabling comprehensive monitoring of agent execution, LLM interactions, tool usage, and cross-agent workflows.

**Why OpenTelemetry for a11i:**

- **Distributed Tracing**: Track requests across multiple agents and LLM providers
- **Standard Semantics**: Use GenAI conventions for consistent instrumentation
- **Provider Agnostic**: Works with OpenAI, Anthropic, AWS Bedrock, Azure AI, and others
- **Cost Tracking**: Monitor token usage and estimated costs across agent operations
- **Performance Analysis**: Identify bottlenecks in agent loops and tool execution

**Architecture Components:**

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Agent A   │─────▶│  OTel SDK    │─────▶│ Collector   │
│  (Python)   │      │  (in-process)│      │  (Gateway)  │
└─────────────┘      └──────────────┘      └─────────────┘
                             │                     │
┌─────────────┐              │                     │
│   Agent B   │──────────────┘                     │
│  (Python)   │                                    │
└─────────────┘                                    │
                                                   ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│ LLM Proxy   │─────▶│  OTel SDK    │      │ ClickHouse  │
│  (litellm)  │      │  (sidecar)   │─────▶│  (Backend)  │
└─────────────┘      └──────────────┘      └─────────────┘
```

## GenAI Semantic Conventions

### Standard Attributes Reference

The GenAI semantic conventions (introduced in OTel v1.38) provide standardized attributes for LLM and agent observability. These attributes are **required** for a11i instrumentation.

| Attribute | Type | Description | Example | Required |
|-----------|------|-------------|---------|----------|
| `gen_ai.system` | string | LLM vendor identifier | `"openai"`, `"anthropic"`, `"aws.bedrock"` | ✓ |
| `gen_ai.request.model` | string | Model requested by client | `"gpt-4-turbo-2024-04-09"` | ✓ |
| `gen_ai.response.model` | string | Actual model used by provider | `"gpt-4-turbo-2024-04-09"` | ✓ |
| `gen_ai.response.id` | string | Provider-assigned request ID | `"chatcmpl-123"` | ✓ |
| `gen_ai.operation.name` | string | Type of GenAI operation | `"invoke_agent"`, `"chat"`, `"execute_tool"` | ✓ |
| `gen_ai.usage.input_tokens` | int | Prompt/input token count | `500` | ✓ |
| `gen_ai.usage.output_tokens` | int | Completion/output token count | `150` | ✓ |
| `gen_ai.usage.total_tokens` | int | Total tokens consumed | `650` | - |
| `gen_ai.agent.id` | string | Unique agent identifier | `"agent-001"` | ✓ |
| `gen_ai.agent.name` | string | Human-readable agent name | `"research_assistant"` | - |
| `gen_ai.conversation.id` | string | Conversation/session identifier | `"conv-abc123"` | - |
| `gen_ai.request.temperature` | double | Sampling temperature | `0.7` | - |
| `gen_ai.request.top_p` | double | Nucleus sampling parameter | `0.95` | - |
| `gen_ai.request.max_tokens` | int | Maximum tokens to generate | `2048` | - |
| `gen_ai.request.frequency_penalty` | double | Frequency penalty parameter | `0.5` | - |
| `gen_ai.request.presence_penalty` | double | Presence penalty parameter | `0.5` | - |

**Operation Types:**

- `chat` - Standard chat completion
- `text_completion` - Legacy completion endpoint
- `invoke_agent` - Agent invocation with tools
- `execute_tool` - Tool/function execution
- `embedding` - Text embedding generation
- `image_generation` - Image generation (DALL-E, Stable Diffusion)

### a11i Custom Extensions

a11i extends the GenAI conventions with domain-specific attributes for agent loop monitoring, context management, and cost tracking.

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `a11i.agent.loop_iteration` | int | Current iteration in agent loop | `3` |
| `a11i.agent.loop_phase` | string | Current phase of agent loop | `"think"`, `"act"`, `"observe"` |
| `a11i.agent.loop_complete` | boolean | Whether loop has completed | `false` |
| `a11i.context.saturation` | double | Context window usage ratio (0.0-1.0) | `0.72` |
| `a11i.context.tokens_used` | int | Tokens used from context window | `7200` |
| `a11i.context.tokens_remaining` | int | Remaining context capacity | `2800` |
| `a11i.context.compression_applied` | boolean | Whether context was compressed | `true` |
| `a11i.tool.category` | string | Tool type classification | `"retrieval"`, `"api"`, `"computation"` |
| `a11i.tool.name` | string | Tool function name | `"search_documents"` |
| `a11i.tool.execution_time_ms` | int | Tool execution duration | `450` |
| `a11i.tool.success` | boolean | Whether tool executed successfully | `true` |
| `a11i.cost.estimate_usd` | double | Estimated cost in USD | `0.0042` |
| `a11i.cost.input_cost_usd` | double | Input token cost | `0.0025` |
| `a11i.cost.output_cost_usd` | double | Output token cost | `0.0017` |
| `a11i.delegation.parent_agent_id` | string | Parent agent identifier (if delegated) | `"orchestrator-001"` |
| `a11i.delegation.depth` | int | Delegation depth in agent hierarchy | `2` |

**Agent Loop Phases:**

- `think` - Planning and reasoning phase
- `act` - Action selection and execution (tool calling)
- `observe` - Result processing and state update
- `decide` - Termination decision

## Implementation Patterns

### Context Propagation

a11i uses W3C Trace Context for distributed tracing across agents, LLM proxies, and tool services. The `traceparent` header propagates trace context through HTTP requests.

**W3C Trace Context Format:**

```
traceparent: 00-{trace_id}-{span_id}-{flags}
```

- `00` - Version (always "00" for current spec)
- `trace_id` - 32 hex characters (128-bit trace identifier)
- `span_id` - 16 hex characters (64-bit span identifier)
- `flags` - 2 hex characters (sampling and other flags)

**Example:**

```
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
```

**Python Implementation:**

```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import requests

# Initialize tracer provider
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure OTLP exporter
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Inject context into outgoing LLM request
def make_llm_request(prompt: str, model: str = "gpt-4-turbo-2024-04-09"):
    """Make LLM request with trace context propagation."""
    headers = {"Content-Type": "application/json"}

    # Inject W3C trace context into headers
    inject(headers)

    with tracer.start_as_current_span("llm_request") as span:
        # Set GenAI semantic convention attributes
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", "chat")

        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        )

        # Record response metadata
        data = response.json()
        span.set_attribute("gen_ai.response.id", data.get("id"))
        span.set_attribute("gen_ai.response.model", data.get("model"))

        # Record token usage
        usage = data.get("usage", {})
        span.set_attribute("gen_ai.usage.input_tokens", usage.get("prompt_tokens", 0))
        span.set_attribute("gen_ai.usage.output_tokens", usage.get("completion_tokens", 0))
        span.set_attribute("gen_ai.usage.total_tokens", usage.get("total_tokens", 0))

        return response.json()

# Extract context in LLM proxy/sidecar
def handle_incoming_request(request):
    """Extract trace context from incoming request."""
    # Extract W3C trace context from headers
    ctx = extract(request.headers)

    with tracer.start_as_current_span("proxy_llm_call", context=ctx) as span:
        # Process request with inherited trace context
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("proxy.backend", "litellm")

        # Forward to actual LLM provider
        result = forward_to_llm(request)

        return result
```

**FastAPI Middleware for Automatic Propagation:**

```python
from fastapi import FastAPI, Request
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

app = FastAPI()

# Automatic instrumentation with context propagation
FastAPIInstrumentor.instrument_app(app)

@app.post("/agent/execute")
async def execute_agent(request: Request):
    # Trace context automatically extracted from request headers
    with tracer.start_as_current_span("agent_execution") as span:
        span.set_attribute("gen_ai.agent.name", "research_assistant")
        span.set_attribute("a11i.agent.loop_iteration", 1)

        # Agent logic here
        result = await agent.execute()

        return result
```

### Multi-Agent Correlation

When agents delegate work to other agents, trace context must be propagated to maintain correlation across the entire agent hierarchy.

**Parent-Child Agent Delegation:**

```python
from opentelemetry import trace
from opentelemetry.trace import Link

tracer = trace.get_tracer(__name__)

class OrchestratorAgent:
    """Parent agent that delegates tasks to worker agents."""

    def execute_task(self, task: dict):
        """Execute task with child agent delegation."""
        with tracer.start_as_current_span("orchestrator_task") as parent_span:
            parent_span.set_attribute("gen_ai.agent.id", "orchestrator-001")
            parent_span.set_attribute("gen_ai.agent.name", "orchestrator")
            parent_span.set_attribute("a11i.agent.loop_phase", "think")

            # Get current span context to pass to child
            parent_context = trace.get_current_span().get_span_context()

            # Delegate to worker agent
            worker = WorkerAgent(agent_id="worker-001")
            result = worker.execute(
                task=task,
                parent_trace_context=parent_context
            )

            parent_span.set_attribute("a11i.delegation.child_count", 1)

            return result


class WorkerAgent:
    """Child agent that receives delegated work."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def execute(self, task: dict, parent_trace_context=None):
        """Execute delegated task with trace correlation."""

        # Create span linked to parent trace
        links = []
        if parent_trace_context:
            links.append(Link(parent_trace_context))

        with tracer.start_as_current_span(
            "worker_task",
            links=links
        ) as span:
            span.set_attribute("gen_ai.agent.id", self.agent_id)
            span.set_attribute("gen_ai.agent.name", "worker")
            span.set_attribute("a11i.delegation.parent_agent_id", "orchestrator-001")
            span.set_attribute("a11i.delegation.depth", 1)
            span.set_attribute("a11i.agent.loop_phase", "act")

            # Execute actual work
            result = self._do_work(task)

            return result
```

**Parallel Agent Execution:**

```python
import asyncio
from opentelemetry.trace import Link

async def parallel_agent_execution(tasks: list):
    """Execute multiple agents in parallel with trace correlation."""

    with tracer.start_as_current_span("parallel_execution") as parent_span:
        parent_span.set_attribute("gen_ai.operation.name", "parallel_invoke_agent")
        parent_span.set_attribute("a11i.delegation.child_count", len(tasks))

        # Get parent context for all child agents
        parent_context = trace.get_current_span().get_span_context()

        # Create tasks with trace links
        async_tasks = []
        for i, task in enumerate(tasks):
            async_tasks.append(
                execute_agent_with_context(task, parent_context, index=i)
            )

        # Execute all agents in parallel
        results = await asyncio.gather(*async_tasks)

        parent_span.set_attribute("a11i.delegation.completed_count", len(results))

        return results


async def execute_agent_with_context(task, parent_context, index: int):
    """Execute single agent with parent trace link."""

    links = [Link(parent_context)]

    with tracer.start_as_current_span(
        f"parallel_agent_{index}",
        links=links
    ) as span:
        span.set_attribute("gen_ai.agent.id", f"parallel-agent-{index}")
        span.set_attribute("a11i.delegation.depth", 1)
        span.set_attribute("a11i.agent.loop_phase", "act")

        # Execute agent logic
        result = await AgentExecutor.execute(task)

        return result
```

### Instrumentation Examples

**Complete Agent Loop with OTel:**

```python
from opentelemetry import trace
from typing import Optional
import time

tracer = trace.get_tracer(__name__)

class A11iAgent:
    """Agent with comprehensive OTel instrumentation."""

    def __init__(self, agent_id: str, agent_name: str, model: str = "gpt-4-turbo-2024-04-09"):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.model = model
        self.conversation_id = f"conv-{agent_id}-{int(time.time())}"

    def execute(self, task: str, max_iterations: int = 10) -> dict:
        """Execute agent task with full tracing."""

        with tracer.start_as_current_span("agent_execution") as execution_span:
            # Set agent metadata
            execution_span.set_attribute("gen_ai.agent.id", self.agent_id)
            execution_span.set_attribute("gen_ai.agent.name", self.agent_name)
            execution_span.set_attribute("gen_ai.conversation.id", self.conversation_id)
            execution_span.set_attribute("gen_ai.request.model", self.model)

            iteration = 0
            complete = False
            total_cost = 0.0

            while not complete and iteration < max_iterations:
                iteration += 1

                # Think phase
                thought = self._think(task, iteration)

                # Act phase
                action_result = self._act(thought, iteration)

                # Observe phase
                observation = self._observe(action_result, iteration)

                # Decide phase
                complete = self._decide(observation, iteration)

                # Update cost tracking
                total_cost += action_result.get("cost_usd", 0.0)

            # Record execution summary
            execution_span.set_attribute("a11i.agent.loop_iteration", iteration)
            execution_span.set_attribute("a11i.agent.loop_complete", complete)
            execution_span.set_attribute("a11i.cost.estimate_usd", total_cost)

            return {
                "success": complete,
                "iterations": iteration,
                "cost_usd": total_cost
            }

    def _think(self, task: str, iteration: int) -> str:
        """Think/planning phase."""

        with tracer.start_as_current_span("agent_think") as span:
            span.set_attribute("a11i.agent.loop_phase", "think")
            span.set_attribute("a11i.agent.loop_iteration", iteration)
            span.set_attribute("gen_ai.operation.name", "chat")

            # Call LLM for reasoning
            response = self._call_llm(
                prompt=f"Plan approach for: {task}",
                operation="think"
            )

            return response["content"]

    def _act(self, thought: str, iteration: int) -> dict:
        """Action/tool execution phase."""

        with tracer.start_as_current_span("agent_act") as span:
            span.set_attribute("a11i.agent.loop_phase", "act")
            span.set_attribute("a11i.agent.loop_iteration", iteration)

            # Determine action from thought
            action = self._parse_action(thought)

            if action["type"] == "tool_call":
                result = self._execute_tool(action["tool_name"], action["arguments"])
            else:
                result = {"result": "no action needed"}

            return result

    def _observe(self, action_result: dict, iteration: int) -> str:
        """Observation/result processing phase."""

        with tracer.start_as_current_span("agent_observe") as span:
            span.set_attribute("a11i.agent.loop_phase", "observe")
            span.set_attribute("a11i.agent.loop_iteration", iteration)
            span.set_attribute("gen_ai.operation.name", "chat")

            # Process action result
            response = self._call_llm(
                prompt=f"Analyze result: {action_result}",
                operation="observe"
            )

            return response["content"]

    def _decide(self, observation: str, iteration: int) -> bool:
        """Decision/termination phase."""

        with tracer.start_as_current_span("agent_decide") as span:
            span.set_attribute("a11i.agent.loop_phase", "decide")
            span.set_attribute("a11i.agent.loop_iteration", iteration)
            span.set_attribute("gen_ai.operation.name", "chat")

            # Decide if task is complete
            response = self._call_llm(
                prompt=f"Is task complete based on: {observation}?",
                operation="decide"
            )

            complete = "complete" in response["content"].lower()
            span.set_attribute("a11i.agent.loop_complete", complete)

            return complete

    def _call_llm(self, prompt: str, operation: str) -> dict:
        """Call LLM with full instrumentation."""

        with tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.request.model", self.model)
            span.set_attribute("gen_ai.operation.name", operation)
            span.set_attribute("gen_ai.request.temperature", 0.7)

            # Simulate LLM call
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "model": self.model,
                "content": f"Response to: {prompt[:50]}...",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                }
            }

            # Record response metadata
            span.set_attribute("gen_ai.response.id", response["id"])
            span.set_attribute("gen_ai.response.model", response["model"])
            span.set_attribute("gen_ai.usage.input_tokens", response["usage"]["prompt_tokens"])
            span.set_attribute("gen_ai.usage.output_tokens", response["usage"]["completion_tokens"])

            # Calculate and record cost (GPT-4 Turbo pricing)
            input_cost = (response["usage"]["prompt_tokens"] / 1000) * 0.01
            output_cost = (response["usage"]["completion_tokens"] / 1000) * 0.03
            total_cost = input_cost + output_cost

            span.set_attribute("a11i.cost.input_cost_usd", round(input_cost, 6))
            span.set_attribute("a11i.cost.output_cost_usd", round(output_cost, 6))
            span.set_attribute("a11i.cost.estimate_usd", round(total_cost, 6))

            # Record context saturation
            context_window = 128000  # GPT-4 Turbo context window
            tokens_used = response["usage"]["total_tokens"]
            saturation = tokens_used / context_window

            span.set_attribute("a11i.context.tokens_used", tokens_used)
            span.set_attribute("a11i.context.tokens_remaining", context_window - tokens_used)
            span.set_attribute("a11i.context.saturation", round(saturation, 4))

            return response

    def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute tool with instrumentation."""

        with tracer.start_as_current_span("tool_execution") as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("a11i.tool.name", tool_name)
            span.set_attribute("a11i.tool.category", self._get_tool_category(tool_name))

            start_time = time.time()

            # Simulate tool execution
            result = {
                "tool": tool_name,
                "result": f"Executed {tool_name} with {arguments}",
                "success": True
            }

            execution_time_ms = int((time.time() - start_time) * 1000)

            span.set_attribute("a11i.tool.execution_time_ms", execution_time_ms)
            span.set_attribute("a11i.tool.success", result["success"])

            return result

    def _get_tool_category(self, tool_name: str) -> str:
        """Categorize tool for observability."""
        categories = {
            "search": "retrieval",
            "query": "retrieval",
            "fetch": "api",
            "post": "api",
            "calculate": "computation",
            "analyze": "computation"
        }

        for keyword, category in categories.items():
            if keyword in tool_name.lower():
                return category

        return "unknown"

    def _parse_action(self, thought: str) -> dict:
        """Parse action from thought (simplified)."""
        return {
            "type": "tool_call",
            "tool_name": "search_documents",
            "arguments": {"query": "example"}
        }
```

## Provider-Specific Attributes

### OpenAI

OpenAI-specific extensions for service tiers and organization tracking:

```python
def call_openai(prompt: str, service_tier: str = "default"):
    """Call OpenAI with provider-specific attributes."""

    with tracer.start_as_current_span("openai_call") as span:
        # Standard GenAI attributes
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "gpt-4-turbo-2024-04-09")
        span.set_attribute("gen_ai.operation.name", "chat")

        # OpenAI-specific attributes
        span.set_attribute("openai.request.service_tier", service_tier)
        span.set_attribute("openai.organization", os.getenv("OPENAI_ORG_ID"))

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-2024-04-09",
            messages=[{"role": "user", "content": prompt}],
            extra_headers={"OpenAI-Service-Tier": service_tier}
        )

        # Record actual service tier used
        span.set_attribute("openai.response.service_tier",
                         response.get("service_tier", "default"))

        return response
```

**OpenAI Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `openai.request.service_tier` | string | Requested service tier: "auto", "default" |
| `openai.response.service_tier` | string | Actual service tier used |
| `openai.organization` | string | Organization ID |
| `openai.request.user` | string | End-user identifier for abuse monitoring |

### AWS Bedrock

AWS Bedrock-specific attributes for guardrails and knowledge bases:

```python
import boto3
from opentelemetry import trace

def call_bedrock(prompt: str, guardrail_id: Optional[str] = None):
    """Call AWS Bedrock with provider-specific attributes."""

    bedrock = boto3.client('bedrock-runtime')

    with tracer.start_as_current_span("bedrock_call") as span:
        # Standard GenAI attributes
        span.set_attribute("gen_ai.system", "aws.bedrock")
        span.set_attribute("gen_ai.request.model", "anthropic.claude-3-sonnet-20240229-v1:0")
        span.set_attribute("gen_ai.operation.name", "invoke_agent")

        # AWS Bedrock-specific attributes
        if guardrail_id:
            span.set_attribute("aws.bedrock.guardrail.id", guardrail_id)

        span.set_attribute("aws.region", os.getenv("AWS_REGION", "us-east-1"))

        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.7
            })
        )

        # Record response metadata
        response_body = json.loads(response['body'].read())
        span.set_attribute("gen_ai.response.id", response_body.get("id"))
        span.set_attribute("gen_ai.usage.input_tokens",
                         response_body.get("usage", {}).get("input_tokens", 0))
        span.set_attribute("gen_ai.usage.output_tokens",
                         response_body.get("usage", {}).get("output_tokens", 0))

        return response_body
```

**AWS Bedrock Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `aws.bedrock.guardrail.id` | string | Guardrail identifier for content filtering |
| `aws.bedrock.knowledge_base.id` | string | Knowledge base identifier for RAG |
| `aws.bedrock.agent.id` | string | Bedrock Agent identifier |
| `aws.region` | string | AWS region |

### Azure AI

Azure AI-specific attributes for deployment and endpoint tracking:

```python
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

def call_azure_ai(prompt: str, deployment_name: str):
    """Call Azure AI with provider-specific attributes."""

    endpoint = os.getenv("AZURE_AI_ENDPOINT")
    credential = AzureKeyCredential(os.getenv("AZURE_AI_KEY"))

    client = ChatCompletionsClient(endpoint=endpoint, credential=credential)

    with tracer.start_as_current_span("azure_ai_call") as span:
        # Standard GenAI attributes
        span.set_attribute("gen_ai.system", "azure.ai")
        span.set_attribute("gen_ai.request.model", deployment_name)
        span.set_attribute("gen_ai.operation.name", "chat")

        # Azure AI-specific attributes
        span.set_attribute("azure.ai.deployment.name", deployment_name)
        span.set_attribute("azure.ai.service.endpoint", endpoint)
        span.set_attribute("azure.subscription.id", os.getenv("AZURE_SUBSCRIPTION_ID"))

        response = client.complete(
            messages=[{"role": "user", "content": prompt}],
            model=deployment_name
        )

        # Record response metadata
        span.set_attribute("gen_ai.response.model", deployment_name)
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)

        return response
```

**Azure AI Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `azure.ai.deployment.name` | string | Azure AI deployment name |
| `azure.ai.service.endpoint` | string | Azure AI service endpoint URL |
| `azure.subscription.id` | string | Azure subscription identifier |
| `azure.resource.group` | string | Azure resource group |

### Anthropic

Anthropic-specific attributes for Claude models:

```python
import anthropic

def call_anthropic(prompt: str):
    """Call Anthropic Claude with provider-specific attributes."""

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with tracer.start_as_current_span("anthropic_call") as span:
        # Standard GenAI attributes
        span.set_attribute("gen_ai.system", "anthropic")
        span.set_attribute("gen_ai.request.model", "claude-sonnet-4-5-20250929")
        span.set_attribute("gen_ai.operation.name", "chat")

        # Anthropic-specific thinking tokens
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            thinking={
                "type": "enabled",
                "budget_tokens": 1000
            }
        )

        # Record standard usage
        span.set_attribute("gen_ai.response.id", response.id)
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.output_tokens)

        # Record thinking tokens separately (Anthropic-specific)
        if hasattr(response.usage, 'thinking_tokens'):
            span.set_attribute("anthropic.usage.thinking_tokens",
                             response.usage.thinking_tokens)

        return response
```

**Anthropic Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `anthropic.usage.thinking_tokens` | int | Extended thinking tokens (Claude Sonnet 4.5+) |
| `anthropic.response.stop_reason` | string | Completion stop reason |

## Migration Strategy for Evolving Standards

The GenAI semantic conventions are currently in **Development** status and may change before reaching **Stable** status. a11i must handle convention evolution gracefully.

### Stability Levels

**Stable for Adoption (Safe to Use):**

- ✓ `gen_ai.operation.name` (required)
- ✓ `gen_ai.system` (required - was `gen_ai.provider.name`)
- ✓ `gen_ai.request.model` / `gen_ai.response.model`
- ✓ `gen_ai.usage.*` attributes (token counts)
- ✓ `gen_ai.agent.id` / `gen_ai.agent.name`
- ✓ `gen_ai.conversation.id`

**Experimental (Expect Changes):**

- ⚠ `gen_ai.input.messages` - Full message content capture
- ⚠ `gen_ai.output.messages` - Response content capture
- ⚠ Agent-specific extensions beyond basic ID/name
- ⚠ Fine-grained request parameters (may be consolidated)

**Deprecated/Renamed:**

- ✗ `gen_ai.provider.name` → Use `gen_ai.system` instead
- ✗ `llm.model` → Use `gen_ai.request.model`
- ✗ `llm.usage.prompt_tokens` → Use `gen_ai.usage.input_tokens`

**Opt-in for Latest Experimental:**

```bash
# Enable experimental GenAI conventions
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental

# Or in Python
import os
os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"
```

### Forward Compatibility Patterns

#### Pattern 1: Dual-Emit During Transitions

Emit both old and new attribute names during transition periods:

```python
def set_model_attribute(span, model: str):
    """Set model attribute with dual-emit for compatibility."""

    # New convention (stable)
    span.set_attribute("gen_ai.request.model", model)

    # Legacy convention (for older collectors/backends)
    span.set_attribute("llm.model", model)

    # Provider-specific (if applicable)
    if "gpt" in model:
        span.set_attribute("openai.model", model)


def set_usage_attributes(span, usage: dict):
    """Set usage attributes with dual-emit."""

    # New conventions (stable)
    span.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
    span.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])
    span.set_attribute("gen_ai.usage.total_tokens", usage["total_tokens"])

    # Legacy conventions
    span.set_attribute("llm.usage.prompt_tokens", usage["prompt_tokens"])
    span.set_attribute("llm.usage.completion_tokens", usage["completion_tokens"])
    span.set_attribute("llm.usage.total_tokens", usage["total_tokens"])
```

#### Pattern 2: OTel Collector Attribute Transforms

Use the OTel Collector to transform attributes for downstream compatibility:

```yaml
# otel-collector-config.yaml
processors:
  attributes/genai_compatibility:
    actions:
      # Transform new conventions to legacy for older backends
      - key: llm.model
        action: insert
        from_attribute: gen_ai.request.model

      - key: llm.usage.prompt_tokens
        action: insert
        from_attribute: gen_ai.usage.input_tokens

      - key: llm.usage.completion_tokens
        action: insert
        from_attribute: gen_ai.usage.output_tokens

      # Transform legacy to new for future-proofing
      - key: gen_ai.system
        action: insert
        from_attribute: gen_ai.provider.name
        if: gen_ai.system == nil

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [attributes/genai_compatibility, batch]
      exporters: [clickhouse, jaeger]
```

#### Pattern 3: Query-Time Coalescing

Use database queries that coalesce old and new attribute names:

```sql
-- ClickHouse query with attribute coalescing
SELECT
    trace_id,
    span_id,
    coalesce(
        attributes['gen_ai.request.model'],
        attributes['llm.model']
    ) as model,
    coalesce(
        toInt32(attributes['gen_ai.usage.input_tokens']),
        toInt32(attributes['llm.usage.prompt_tokens'])
    ) as input_tokens,
    coalesce(
        toInt32(attributes['gen_ai.usage.output_tokens']),
        toInt32(attributes['llm.usage.completion_tokens'])
    ) as output_tokens,
    coalesce(
        attributes['gen_ai.system'],
        attributes['gen_ai.provider.name']
    ) as provider
FROM otel_traces
WHERE operation_name = 'llm_call'
ORDER BY start_time DESC
LIMIT 100;
```

#### Pattern 4: Abstraction Layer Wrapper

Create a wrapper abstraction to isolate attribute naming from business logic:

```python
from dataclasses import dataclass
from typing import Optional
from opentelemetry import trace

@dataclass
class GenAIAttributes:
    """Abstraction for GenAI semantic conventions with version handling."""

    # Convention version to use
    version: str = "v1.38"

    def get_system_attr(self) -> str:
        """Get provider/system attribute name."""
        if self.version >= "v1.38":
            return "gen_ai.system"
        return "gen_ai.provider.name"

    def get_request_model_attr(self) -> str:
        """Get request model attribute name."""
        if self.version >= "v1.38":
            return "gen_ai.request.model"
        return "llm.model"

    def get_usage_input_attr(self) -> str:
        """Get input tokens attribute name."""
        if self.version >= "v1.38":
            return "gen_ai.usage.input_tokens"
        return "llm.usage.prompt_tokens"

    def get_usage_output_attr(self) -> str:
        """Get output tokens attribute name."""
        if self.version >= "v1.38":
            return "gen_ai.usage.output_tokens"
        return "llm.usage.completion_tokens"

    def set_model(self, span, model: str, dual_emit: bool = True):
        """Set model attribute with version awareness."""
        span.set_attribute(self.get_request_model_attr(), model)

        if dual_emit and self.version >= "v1.38":
            # Also emit legacy for compatibility
            span.set_attribute("llm.model", model)

    def set_usage(self, span, input_tokens: int, output_tokens: int,
                  dual_emit: bool = True):
        """Set usage attributes with version awareness."""
        span.set_attribute(self.get_usage_input_attr(), input_tokens)
        span.set_attribute(self.get_usage_output_attr(), output_tokens)
        span.set_attribute("gen_ai.usage.total_tokens", input_tokens + output_tokens)

        if dual_emit and self.version >= "v1.38":
            # Also emit legacy for compatibility
            span.set_attribute("llm.usage.prompt_tokens", input_tokens)
            span.set_attribute("llm.usage.completion_tokens", output_tokens)
            span.set_attribute("llm.usage.total_tokens", input_tokens + output_tokens)


# Usage
attrs = GenAIAttributes(version="v1.38")

with tracer.start_as_current_span("llm_call") as span:
    attrs.set_model(span, "gpt-4-turbo-2024-04-09")
    attrs.set_usage(span, input_tokens=500, output_tokens=150)
```

## OTel Collector Configuration

The OpenTelemetry Collector acts as a central gateway for trace data, handling batching, transformation, and export to backend systems.

**Complete Collector Configuration:**

```yaml
# otel-collector-config.yaml

receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  # Batch spans for efficiency
  batch:
    timeout: 5s
    send_batch_size: 1024
    send_batch_max_size: 2048

  # Add resource attributes
  resource:
    attributes:
      - key: service.name
        value: a11i
        action: upsert
      - key: deployment.environment
        from_attribute: DEPLOYMENT_ENV
        action: insert

  # Transform GenAI attributes for compatibility
  attributes/genai:
    actions:
      # Ensure gen_ai.system exists (backfill from legacy)
      - key: gen_ai.system
        action: insert
        from_attribute: gen_ai.provider.name
        if: gen_ai.system == nil

      # Calculate total_tokens if missing
      - key: gen_ai.usage.total_tokens
        action: insert
        value: attributes['gen_ai.usage.input_tokens'] + attributes['gen_ai.usage.output_tokens']
        if: gen_ai.usage.total_tokens == nil

  # Filter sensitive data
  attributes/filter:
    actions:
      # Remove PII from prompts
      - key: gen_ai.input.messages
        action: delete
        if: attributes['a11i.privacy.filter_prompts'] == "true"

      - key: gen_ai.output.messages
        action: delete
        if: attributes['a11i.privacy.filter_completions'] == "true"

  # Memory limiter to prevent OOM
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128

exporters:
  # Primary backend: ClickHouse
  clickhouse:
    endpoint: tcp://clickhouse:9000
    database: a11i_observability
    ttl: 720h  # 30 days retention
    logs_table_name: otel_logs
    traces_table_name: otel_traces
    metrics_table_name: otel_metrics
    timeout: 10s
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s

  # Secondary backend: Jaeger for trace visualization
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  # Logging exporter for debugging
  logging:
    loglevel: debug
    sampling_initial: 5
    sampling_thereafter: 200

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors:
        - memory_limiter
        - resource
        - attributes/genai
        - attributes/filter
        - batch
      exporters: [clickhouse, jaeger, logging]

  telemetry:
    logs:
      level: info
    metrics:
      level: detailed
      address: 0.0.0.0:8888
```

**Docker Compose Deployment:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.110.0
    container_name: a11i-otel-collector
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
      - "8888:8888"   # Metrics
    environment:
      - DEPLOYMENT_ENV=production
    depends_on:
      - clickhouse
      - jaeger

  clickhouse:
    image: clickhouse/clickhouse-server:24.1
    container_name: a11i-clickhouse
    ports:
      - "9000:9000"   # Native protocol
      - "8123:8123"   # HTTP interface
    volumes:
      - clickhouse-data:/var/lib/clickhouse
      - ./clickhouse-init.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      - CLICKHOUSE_DB=a11i_observability
      - CLICKHOUSE_USER=a11i
      - CLICKHOUSE_PASSWORD=changeme

  jaeger:
    image: jaegertracing/all-in-one:1.53
    container_name: a11i-jaeger
    ports:
      - "16686:16686"  # Jaeger UI
      - "14250:14250"  # gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true

volumes:
  clickhouse-data:
```

**ClickHouse Schema Initialization:**

```sql
-- clickhouse-init.sql
CREATE DATABASE IF NOT EXISTS a11i_observability;

USE a11i_observability;

-- Traces table with GenAI attributes
CREATE TABLE IF NOT EXISTS otel_traces (
    timestamp DateTime64(9),
    trace_id String,
    span_id String,
    parent_span_id String,
    trace_state String,
    span_name LowCardinality(String),
    span_kind LowCardinality(String),
    service_name LowCardinality(String),
    resource_attributes Map(String, String),
    scope_name String,
    scope_version String,
    attributes Map(String, String),
    duration_ns UInt64,
    status_code LowCardinality(String),
    status_message String,
    events Nested (
        timestamp DateTime64(9),
        name LowCardinality(String),
        attributes Map(String, String)
    ),
    links Nested (
        trace_id String,
        span_id String,
        trace_state String,
        attributes Map(String, String)
    )
) ENGINE = MergeTree()
PARTITION BY toDate(timestamp)
ORDER BY (service_name, span_name, timestamp)
TTL timestamp + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- Materialized view for GenAI operations
CREATE MATERIALIZED VIEW IF NOT EXISTS genai_operations_mv
ENGINE = SummingMergeTree()
PARTITION BY toDate(timestamp)
ORDER BY (service_name, provider, model, operation, hour)
AS SELECT
    toStartOfHour(timestamp) as hour,
    service_name,
    attributes['gen_ai.system'] as provider,
    attributes['gen_ai.request.model'] as model,
    attributes['gen_ai.operation.name'] as operation,
    attributes['gen_ai.agent.id'] as agent_id,
    count() as request_count,
    sum(toInt64OrNull(attributes['gen_ai.usage.input_tokens'])) as total_input_tokens,
    sum(toInt64OrNull(attributes['gen_ai.usage.output_tokens'])) as total_output_tokens,
    sum(toFloat64OrNull(attributes['a11i.cost.estimate_usd'])) as total_cost_usd,
    avg(duration_ns / 1000000) as avg_duration_ms
FROM otel_traces
WHERE attributes['gen_ai.operation.name'] != ''
GROUP BY hour, service_name, provider, model, operation, agent_id;

-- Index for fast trace lookup
CREATE INDEX IF NOT EXISTS idx_trace_id ON otel_traces (trace_id) TYPE bloom_filter(0.01);
CREATE INDEX IF NOT EXISTS idx_agent_id ON otel_traces (attributes['gen_ai.agent.id']) TYPE bloom_filter(0.01);
```

## Query Patterns and Analytics

**Example Queries for a11i Observability:**

```sql
-- 1. Agent execution summary by agent ID
SELECT
    attributes['gen_ai.agent.id'] as agent_id,
    attributes['gen_ai.agent.name'] as agent_name,
    count() as executions,
    avg(duration_ns / 1000000000) as avg_duration_seconds,
    sum(toInt64OrNull(attributes['gen_ai.usage.total_tokens'])) as total_tokens,
    sum(toFloat64OrNull(attributes['a11i.cost.estimate_usd'])) as total_cost_usd
FROM otel_traces
WHERE span_name = 'agent_execution'
    AND timestamp >= now() - INTERVAL 24 HOUR
GROUP BY agent_id, agent_name
ORDER BY total_cost_usd DESC;

-- 2. Agent loop iteration analysis
SELECT
    attributes['gen_ai.agent.id'] as agent_id,
    toInt32OrNull(attributes['a11i.agent.loop_iteration']) as iteration,
    attributes['a11i.agent.loop_phase'] as phase,
    count() as phase_count,
    avg(duration_ns / 1000000) as avg_duration_ms
FROM otel_traces
WHERE attributes['a11i.agent.loop_iteration'] != ''
    AND timestamp >= now() - INTERVAL 1 HOUR
GROUP BY agent_id, iteration, phase
ORDER BY agent_id, iteration, phase;

-- 3. LLM provider cost breakdown
SELECT
    attributes['gen_ai.system'] as provider,
    attributes['gen_ai.request.model'] as model,
    count() as request_count,
    sum(toInt64OrNull(attributes['gen_ai.usage.input_tokens'])) as total_input_tokens,
    sum(toInt64OrNull(attributes['gen_ai.usage.output_tokens'])) as total_output_tokens,
    sum(toFloat64OrNull(attributes['a11i.cost.estimate_usd'])) as total_cost_usd,
    avg(toFloat64OrNull(attributes['a11i.cost.estimate_usd'])) as avg_cost_per_request
FROM otel_traces
WHERE attributes['gen_ai.system'] != ''
    AND timestamp >= now() - INTERVAL 7 DAY
GROUP BY provider, model
ORDER BY total_cost_usd DESC;

-- 4. Tool execution performance
SELECT
    attributes['a11i.tool.name'] as tool_name,
    attributes['a11i.tool.category'] as tool_category,
    count() as execution_count,
    countIf(attributes['a11i.tool.success'] = 'true') as success_count,
    countIf(attributes['a11i.tool.success'] = 'false') as failure_count,
    avg(toInt32OrNull(attributes['a11i.tool.execution_time_ms'])) as avg_execution_ms,
    quantile(0.95)(toInt32OrNull(attributes['a11i.tool.execution_time_ms'])) as p95_execution_ms
FROM otel_traces
WHERE attributes['a11i.tool.name'] != ''
    AND timestamp >= now() - INTERVAL 24 HOUR
GROUP BY tool_name, tool_category
ORDER BY execution_count DESC;

-- 5. Context saturation monitoring
SELECT
    attributes['gen_ai.agent.id'] as agent_id,
    attributes['gen_ai.request.model'] as model,
    avg(toFloat64OrNull(attributes['a11i.context.saturation'])) as avg_saturation,
    max(toFloat64OrNull(attributes['a11i.context.saturation'])) as max_saturation,
    countIf(toFloat64OrNull(attributes['a11i.context.saturation']) > 0.8) as high_saturation_count
FROM otel_traces
WHERE attributes['a11i.context.saturation'] != ''
    AND timestamp >= now() - INTERVAL 24 HOUR
GROUP BY agent_id, model
ORDER BY max_saturation DESC;

-- 6. Multi-agent delegation tree
WITH RECURSIVE delegation_tree AS (
    -- Root agents (no parent)
    SELECT
        trace_id,
        span_id,
        attributes['gen_ai.agent.id'] as agent_id,
        attributes['gen_ai.agent.name'] as agent_name,
        '' as parent_agent_id,
        0 as depth,
        duration_ns / 1000000 as duration_ms
    FROM otel_traces
    WHERE attributes['a11i.delegation.parent_agent_id'] = ''
        AND span_name = 'agent_execution'
        AND timestamp >= now() - INTERVAL 1 HOUR

    UNION ALL

    -- Child agents
    SELECT
        t.trace_id,
        t.span_id,
        t.attributes['gen_ai.agent.id'] as agent_id,
        t.attributes['gen_ai.agent.name'] as agent_name,
        t.attributes['a11i.delegation.parent_agent_id'] as parent_agent_id,
        toInt32OrNull(t.attributes['a11i.delegation.depth']) as depth,
        t.duration_ns / 1000000 as duration_ms
    FROM otel_traces t
    INNER JOIN delegation_tree dt ON t.trace_id = dt.trace_id
    WHERE t.attributes['a11i.delegation.parent_agent_id'] != ''
        AND t.span_name = 'agent_execution'
)
SELECT * FROM delegation_tree
ORDER BY trace_id, depth;

-- 7. Error rate by operation
SELECT
    attributes['gen_ai.operation.name'] as operation,
    attributes['gen_ai.system'] as provider,
    count() as total_requests,
    countIf(status_code = 'ERROR') as error_count,
    (error_count / total_requests) * 100 as error_rate_pct,
    groupArray(status_message) as error_messages
FROM otel_traces
WHERE attributes['gen_ai.operation.name'] != ''
    AND timestamp >= now() - INTERVAL 24 HOUR
GROUP BY operation, provider
ORDER BY error_rate_pct DESC;
```

## Key Takeaways

> **OpenTelemetry GenAI Conventions for a11i**
>
> - **Use GenAI v1.38+ Conventions**: Adopt `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*` attributes as the standard
> - **Extend with a11i Attributes**: Add `a11i.*` custom attributes for agent loop monitoring, context tracking, and cost analysis
> - **Propagate W3C Trace Context**: Use `traceparent` headers to correlate multi-agent workflows and LLM proxy calls
> - **Implement Dual-Emit Pattern**: Emit both new and legacy attribute names during transition periods for backward compatibility
> - **Use OTel Collector for Transformation**: Centralize attribute mapping and filtering in the collector pipeline
> - **Monitor Context Saturation**: Track `a11i.context.saturation` to prevent context overflow and optimize prompt sizing
> - **Track Costs Continuously**: Record `a11i.cost.estimate_usd` on every LLM call for real-time cost monitoring
> - **Plan for Convention Evolution**: Use abstraction layers and collector transforms to handle semantic convention changes
> - **Leverage ClickHouse Analytics**: Use materialized views and aggregation queries for real-time observability dashboards
> - **Correlate Multi-Agent Trees**: Use span links and delegation attributes to visualize agent collaboration patterns

## Cross-References

- **[Agent Loop Architecture](../02-architecture/agent-loop.md)** - Understanding agent phases for loop instrumentation
- **[LLM Proxy Integration](./llm-proxy.md)** - Configuring litellm proxy with OTel sidecar instrumentation
- **[ClickHouse Storage](./clickhouse-storage.md)** - Schema design and query optimization for trace data
- **[Cost Tracking](./cost-tracking.md)** - Detailed cost calculation and attribution patterns
- **[Multi-Agent Coordination](../02-architecture/multi-agent.md)** - Agent delegation and collaboration patterns
- **[Observability Dashboard](./observability-dashboard.md)** - Building Grafana dashboards from OTel data
- **[Security & Privacy](../04-security/data-privacy.md)** - Filtering sensitive data from traces

---

**Next Steps:**

1. Implement OTel SDK instrumentation in agent base classes
2. Deploy OTel Collector with ClickHouse exporter
3. Create ClickHouse schema with GenAI attribute indexes
4. Build Grafana dashboards for agent monitoring
5. Configure cost tracking with real-time alerting
6. Set up trace retention and sampling policies
