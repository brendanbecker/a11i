---
title: Framework Integrations
description: Comprehensive guide to integrating a11i with popular AI agent frameworks including LangChain, Semantic Kernel, AutoGen, CrewAI, DSPy, Haystack, and LlamaIndex
category: Developer Experience
tags: [frameworks, langchain, semantic-kernel, autogen, crewai, dspy, haystack, llamaindex, integrations, instrumentation]
version: 1.0.0
last_updated: 2025-11-26
related:
  - ../04-implementation/sdk-library.md
  - ../03-core-platform/opentelemetry-integration.md
  - ../03-core-platform/span-hierarchy.md
---

# Framework Integrations

## Table of Contents

1. [Overview](#overview)
2. [Integration Matrix](#integration-matrix)
3. [LangChain Integration](#langchain-integration)
4. [Semantic Kernel Integration](#semantic-kernel-integration)
5. [AutoGen Integration](#autogen-integration)
6. [CrewAI Integration](#crewai-integration)
7. [DSPy Integration](#dspy-integration)
8. [Haystack Integration](#haystack-integration)
9. [LlamaIndex Integration](#llamaindex-integration)
10. [Custom Agent Frameworks](#custom-agent-frameworks)
11. [Auto-Instrumentation Setup](#auto-instrumentation-setup)
12. [Best Practices](#best-practices)
13. [Key Takeaways](#key-takeaways)

## Overview

a11i provides native integrations with the most popular AI agent frameworks, enabling comprehensive observability with minimal code changes. Each integration is designed to capture framework-specific patterns while maintaining consistency with OpenTelemetry semantic conventions.

### Design Philosophy

**Framework-Native Patterns:**
- Each integration uses the framework's native extension mechanisms (callbacks, hooks, middleware)
- Minimal disruption to existing codebases
- Preserves framework idioms and best practices

**Consistent Observability:**
- Unified span hierarchy across all frameworks
- Standard OpenTelemetry semantic conventions
- Cross-framework trace correlation

**Progressive Enhancement:**
- Auto-instrumentation for quick adoption
- Manual instrumentation for fine-grained control
- Hybrid approaches supported

### Integration Approaches

| Approach | Description | Use Case |
|----------|-------------|----------|
| **CallbackHandler** | Framework-native callback interfaces | LangChain, LlamaIndex |
| **Native OTel** | Built-in OpenTelemetry support | Semantic Kernel |
| **TracerProvider** | Direct OpenTelemetry integration | AutoGen, Custom agents |
| **Hooks/Middleware** | Framework extension points | CrewAI, Haystack |
| **OpenInference** | Specialized LLM tracing protocol | DSPy |
| **Decorators** | Function-level instrumentation | Any Python framework |

## Integration Matrix

Comprehensive overview of framework support status and integration patterns:

| Framework | Pattern | Auto-Instrument | Code Example | Status |
|-----------|---------|-----------------|--------------|--------|
| **LangChain** | CallbackHandler | ✅ Yes | Full | ✅ Stable |
| **Semantic Kernel** | Native OTel | ✅ Yes | Full | ✅ Stable |
| **AutoGen** | TracerProvider | 🔧 Partial | Basic | 🔧 Beta |
| **CrewAI** | Hooks | 🔧 Partial | Basic | 🔧 Beta |
| **DSPy** | OpenInference | ✅ Yes | Full | ✅ Stable |
| **Haystack** | Tracer Interface | 🔧 Partial | Basic | 🔧 Beta |
| **LlamaIndex** | Callbacks | ✅ Yes | Full | ✅ Stable |
| **Custom Agents** | Decorators | ✅ Yes | Full | ✅ Stable |

**Status Legend:**
- ✅ **Stable**: Production-ready with full feature support
- 🔧 **Beta**: Functional but may have limited features
- 📋 **Planned**: On roadmap, not yet available

## LangChain Integration

LangChain is instrumented via a custom `CallbackHandler` that integrates with LangChain's callback system.

### Architecture

LangChain's callback system provides hooks for:
- **Chain Execution**: Start/end of chain invocations
- **LLM Calls**: Request/response for language model interactions
- **Tool Execution**: Tool selection and execution
- **Agent Actions**: Multi-step agent decision-making

### Implementation

```python
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

class A11iCallbackHandler(BaseCallbackHandler):
    """a11i tracing callback handler for LangChain.

    This handler integrates with LangChain's callback system to provide
    comprehensive observability for chains, LLM calls, tools, and agents.
    """

    def __init__(self, tracer_name: str = "a11i.langchain"):
        """Initialize callback handler.

        Args:
            tracer_name: Name for the OpenTelemetry tracer
        """
        self.tracer = trace.get_tracer(tracer_name)
        self._spans: Dict[UUID, trace.Span] = {}
        self._run_to_span: Dict[UUID, UUID] = {}

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        """Start span for chain execution.

        Creates a new span when a LangChain chain begins execution,
        properly handling parent-child relationships.
        """
        span_name = serialized.get("name", "chain")

        # Get parent span if exists
        parent_span = self._spans.get(parent_run_id) if parent_run_id else None
        context = trace.set_span_in_context(parent_span) if parent_span else None

        span = self.tracer.start_span(
            f"langchain.{span_name}",
            context=context,
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute("a11i.framework", "langchain")
        span.set_attribute("gen_ai.operation.name", "chain")
        span.set_attribute("langchain.chain.type", serialized.get("_type", "unknown"))

        # Capture input if available
        if inputs:
            input_str = str(inputs)[:1000]
            if len(str(inputs)) > 1000:
                input_str += "... (truncated)"
            span.set_attribute("a11i.chain.input", input_str)

        self._spans[run_id] = span

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        run_id: UUID,
        **kwargs,
    ):
        """End chain span.

        Completes the span with output data and success status.
        """
        span = self._spans.pop(run_id, None)
        if span:
            # Capture output
            if outputs:
                output_str = str(outputs)[:1000]
                if len(str(outputs)) > 1000:
                    output_str += "... (truncated)"
                span.set_attribute("a11i.chain.output", output_str)

            span.set_status(Status(StatusCode.OK))
            span.end()

    def on_chain_error(
        self,
        error: Exception,
        run_id: UUID,
        **kwargs,
    ):
        """Record chain error.

        Captures exception details and marks span as failed.
        """
        span = self._spans.pop(run_id, None)
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        """Start span for LLM call.

        Captures LLM invocation with model parameters and prompt information.
        """
        parent_span = self._spans.get(parent_run_id) if parent_run_id else None
        context = trace.set_span_in_context(parent_span) if parent_span else None

        span = self.tracer.start_span(
            "langchain.llm",
            context=context,
            kind=SpanKind.CLIENT,
        )

        # Extract model configuration
        invocation_params = kwargs.get("invocation_params", {})
        model_name = invocation_params.get("model_name", "unknown")
        temperature = invocation_params.get("temperature", 1.0)
        max_tokens = invocation_params.get("max_tokens")

        # Set standard Gen AI semantic conventions
        span.set_attribute("gen_ai.system", "langchain")
        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("gen_ai.request.temperature", temperature)
        if max_tokens:
            span.set_attribute("gen_ai.request.max_tokens", max_tokens)
        span.set_attribute("gen_ai.operation.name", "chat")

        # Capture prompt information
        span.set_attribute("a11i.prompt.count", len(prompts))
        if prompts:
            # Capture first prompt (truncated)
            first_prompt = prompts[0][:500]
            if len(prompts[0]) > 500:
                first_prompt += "... (truncated)"
            span.set_attribute("a11i.prompt.preview", first_prompt)

        self._spans[run_id] = span

    def on_llm_end(
        self,
        response: LLMResult,
        run_id: UUID,
        **kwargs,
    ):
        """End LLM span with usage data.

        Captures token usage, response content, and completion status.
        """
        span = self._spans.pop(run_id, None)
        if span:
            # Capture token usage
            if response.llm_output:
                usage = response.llm_output.get("token_usage", {})
                span.set_attribute("gen_ai.usage.input_tokens",
                                  usage.get("prompt_tokens", 0))
                span.set_attribute("gen_ai.usage.output_tokens",
                                  usage.get("completion_tokens", 0))
                span.set_attribute("gen_ai.usage.total_tokens",
                                  usage.get("total_tokens", 0))

            # Capture response preview
            if response.generations:
                first_gen = response.generations[0][0].text[:500]
                if len(response.generations[0][0].text) > 500:
                    first_gen += "... (truncated)"
                span.set_attribute("a11i.response.preview", first_gen)

            span.set_status(Status(StatusCode.OK))
            span.end()

    def on_llm_error(
        self,
        error: Exception,
        run_id: UUID,
        **kwargs,
    ):
        """Record LLM error."""
        span = self._spans.pop(run_id, None)
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        """Start span for tool execution.

        Captures tool invocation with input parameters.
        """
        parent_span = self._spans.get(parent_run_id) if parent_run_id else None
        context = trace.set_span_in_context(parent_span) if parent_span else None

        tool_name = serialized.get("name", "tool")

        span = self.tracer.start_span(
            f"langchain.tool.{tool_name}",
            context=context,
            kind=SpanKind.CLIENT,
        )
        span.set_attribute("a11i.framework", "langchain")
        span.set_attribute("a11i.tool.name", tool_name)
        span.set_attribute("gen_ai.operation.name", "execute_tool")

        # Capture tool input
        tool_input = input_str[:1000]
        if len(input_str) > 1000:
            tool_input += "... (truncated)"
        span.set_attribute("a11i.tool.input", tool_input)

        self._spans[run_id] = span

    def on_tool_end(
        self,
        output: str,
        run_id: UUID,
        **kwargs,
    ):
        """End tool span.

        Captures tool output and marks as successful.
        """
        span = self._spans.pop(run_id, None)
        if span:
            # Capture tool output
            tool_output = str(output)[:1000]
            if len(str(output)) > 1000:
                tool_output += "... (truncated)"
            span.set_attribute("a11i.tool.output", tool_output)

            span.set_status(Status(StatusCode.OK))
            span.end()

    def on_tool_error(
        self,
        error: Exception,
        run_id: UUID,
        **kwargs,
    ):
        """Record tool error."""
        span = self._spans.pop(run_id, None)
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()

    def on_agent_action(
        self,
        action: AgentAction,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        """Record agent action decision.

        Captures the agent's reasoning and chosen action.
        """
        parent_span = self._spans.get(parent_run_id)
        if parent_span:
            # Add event to parent span
            parent_span.add_event(
                "agent_action",
                attributes={
                    "action.tool": action.tool,
                    "action.input": str(action.tool_input)[:500],
                    "action.log": action.log[:500] if action.log else "",
                }
            )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        """Record agent completion.

        Captures final agent output and reasoning.
        """
        parent_span = self._spans.get(parent_run_id)
        if parent_span:
            parent_span.add_event(
                "agent_finish",
                attributes={
                    "finish.output": str(finish.return_values)[:500],
                    "finish.log": finish.log[:500] if finish.log else "",
                }
            )
```

### Usage Examples

#### Basic LangChain Chain

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from a11i.integrations.langchain import A11iCallbackHandler
from a11i import configure

# Configure a11i
configure(api_key="your-api-key", project="langchain-demo")

# Create callback handler
handler = A11iCallbackHandler()

# Create chain with callback
llm = ChatOpenAI(temperature=0.7, callbacks=[handler])
prompt = PromptTemplate.from_template("Explain {topic} in simple terms")
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])

# Run chain - automatically traced
result = chain.run(topic="quantum computing")
print(result)
```

#### LangChain Agent with Tools

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from a11i.integrations.langchain import A11iCallbackHandler

# Define tools
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

def calculate(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

tools = [
    Tool(name="Search", func=search_web, description="Search the web"),
    Tool(name="Calculator", func=calculate, description="Perform calculations"),
]

# Create agent with callback
handler = A11iCallbackHandler()
llm = ChatOpenAI(model="gpt-4", callbacks=[handler])
agent = create_openai_functions_agent(llm, tools)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[handler],
    verbose=True
)

# Execute agent - full trace captured
result = agent_executor.invoke({
    "input": "What is 15% of 240, and then search for that number's significance"
})
```

#### Async LangChain with Streaming

```python
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from a11i.integrations.langchain import A11iCallbackHandler

# Combine streaming with tracing
handler = A11iCallbackHandler()
streaming = StreamingStdOutCallbackHandler()

llm = ChatOpenAI(
    streaming=True,
    callbacks=[handler, streaming]  # Multiple callbacks
)

async def stream_response(prompt: str):
    response = await llm.apredict(prompt)
    return response

# Traced and streamed
await stream_response("Write a poem about observability")
```

### Auto-Instrumentation

```python
# Automatic instrumentation for LangChain
from a11i import auto_instrument

# Instrument all LangChain operations
auto_instrument(frameworks=["langchain"])

# Now all LangChain chains/agents are automatically traced
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI()
result = llm.predict("Hello!")  # Automatically traced!
```

## Semantic Kernel Integration

Semantic Kernel has native OpenTelemetry support, making integration seamless.

### Architecture

Semantic Kernel natively emits OpenTelemetry traces for:
- **Semantic Functions**: Prompt templates and executions
- **Native Functions**: Code-based skills
- **Planners**: Multi-step planning and execution
- **Memory**: Vector store operations

### Implementation

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup a11i-compatible tracing
provider = TracerProvider(
    resource=Resource.create({
        "service.name": "semantic-kernel-app",
        "a11i.project": "my-project",
    })
)
exporter = OTLPSpanExporter(endpoint="http://a11i-collector:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# Semantic Kernel automatically uses the configured tracer
kernel = sk.Kernel()
kernel.add_chat_service(
    "chat",
    OpenAIChatCompletion("gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
)

# All kernel operations are automatically traced
result = await kernel.invoke_prompt("Explain AI observability")
```

### Usage Examples

#### Semantic Functions

```python
import semantic_kernel as sk
from semantic_kernel.core_skills import TextSkill

kernel = sk.Kernel()
kernel.add_chat_service("chat", OpenAIChatCompletion("gpt-4", api_key))

# Define semantic function (automatically traced)
summarize = kernel.create_semantic_function(
    "Summarize the following text:\n{{$input}}",
    max_tokens=100,
    temperature=0.3,
)

# Execute - fully traced with OpenTelemetry
text = "Long article text here..."
summary = await summarize.invoke_async(text)
```

#### Planner with Multiple Steps

```python
from semantic_kernel.planning import SequentialPlanner

kernel = sk.Kernel()
kernel.add_chat_service("chat", OpenAIChatCompletion("gpt-4", api_key))

# Import skills
kernel.import_skill(TextSkill(), "text")
kernel.import_skill(MathSkill(), "math")

# Create planner (plans are automatically traced)
planner = SequentialPlanner(kernel)

# Generate and execute plan
ask = "What is 25% of 840, then write a haiku about that number"
plan = await planner.create_plan_async(ask)

# Execute plan - each step is traced as separate span
result = await plan.invoke_async()

# Trace hierarchy:
# invoke_plan
#   ├─ step_1: math.percentage (25% of 840)
#   └─ step_2: text.generate (haiku about 210)
```

#### Memory Operations

```python
from semantic_kernel.memory import VolatileMemoryStore

kernel = sk.Kernel()
kernel.add_chat_service("chat", OpenAIChatCompletion("gpt-4", api_key))

# Memory operations are automatically traced
memory = VolatileMemoryStore()
kernel.register_memory(memory)

# Save to memory - traced
await kernel.memory.save_information_async(
    collection="facts",
    id="fact1",
    text="AI observability helps monitor autonomous agents"
)

# Search memory - traced
results = await kernel.memory.search_async(
    collection="facts",
    query="How to monitor AI agents?",
    limit=3
)
```

### Configuration

```python
# Configure Semantic Kernel with a11i
from opentelemetry.sdk.resources import Resource

resource = Resource.create({
    "service.name": "my-semantic-kernel-app",
    "service.version": "1.0.0",
    "a11i.project": "sk-project",
    "a11i.environment": "production",
})

provider = TracerProvider(resource=resource)
# ... setup exporter as shown above
```

## AutoGen Integration

AutoGen integration uses OpenTelemetry TracerProvider for multi-agent conversations.

### Architecture

AutoGen's multi-agent conversations require custom instrumentation:
- **Agent Creation**: Track agent initialization
- **Message Exchange**: Capture agent-to-agent messages
- **Tool Calls**: Monitor function executions
- **Conversation Flow**: Trace complete multi-agent dialogues

### Implementation

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from typing import Dict, Any, Optional

class A11iAutogenTracer:
    """a11i tracing for AutoGen multi-agent systems."""

    def __init__(self, tracer_name: str = "a11i.autogen"):
        self.tracer = trace.get_tracer(tracer_name)
        self._conversation_span = None
        self._message_spans = {}

    def start_conversation(self, conversation_id: str, agents: list):
        """Start tracing a multi-agent conversation."""
        self._conversation_span = self.tracer.start_span(
            f"autogen.conversation.{conversation_id}",
            kind=SpanKind.INTERNAL,
        )
        self._conversation_span.set_attribute("a11i.framework", "autogen")
        self._conversation_span.set_attribute("autogen.agent.count", len(agents))
        self._conversation_span.set_attribute("autogen.agents",
                                              ",".join([a.name for a in agents]))

    def trace_message(
        self,
        sender: str,
        recipient: str,
        message: str,
        message_id: Optional[str] = None,
    ):
        """Trace a message between agents."""
        if not self._conversation_span:
            return

        context = trace.set_span_in_context(self._conversation_span)
        span = self.tracer.start_span(
            f"autogen.message",
            context=context,
            kind=SpanKind.INTERNAL,
        )

        span.set_attribute("autogen.message.sender", sender)
        span.set_attribute("autogen.message.recipient", recipient)

        # Capture message content (truncated)
        content = message[:1000]
        if len(message) > 1000:
            content += "... (truncated)"
        span.set_attribute("autogen.message.content", content)

        if message_id:
            self._message_spans[message_id] = span

        return span

    def end_message(self, message_id: Optional[str] = None, span=None):
        """End a message span."""
        target_span = span or (self._message_spans.pop(message_id, None)
                               if message_id else None)
        if target_span:
            target_span.set_status(Status(StatusCode.OK))
            target_span.end()

    def end_conversation(self, total_messages: int, success: bool = True):
        """End conversation tracing."""
        if self._conversation_span:
            self._conversation_span.set_attribute("autogen.total_messages",
                                                  total_messages)
            if success:
                self._conversation_span.set_status(Status(StatusCode.OK))
            else:
                self._conversation_span.set_status(Status(StatusCode.ERROR))
            self._conversation_span.end()
```

### Usage Examples

#### Two-Agent Conversation

```python
from autogen import AssistantAgent, UserProxyAgent
from a11i.integrations.autogen import A11iAutogenTracer

# Create tracer
tracer = A11iAutogenTracer()

# Create agents
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")},
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"},
)

# Start conversation tracing
tracer.start_conversation("code_generation", [assistant, user_proxy])

# Initiate chat with custom message handler
def message_handler(sender, recipient, message):
    """Custom handler to trace messages."""
    span = tracer.trace_message(sender.name, recipient.name, message["content"])
    # Original handler logic here
    tracer.end_message(span=span)

# Execute conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate Fibonacci numbers"
)

# End tracing
tracer.end_conversation(total_messages=10, success=True)
```

#### Group Chat with Multiple Agents

```python
from autogen import GroupChat, GroupChatManager

tracer = A11iAutogenTracer()

# Create multiple agents
researcher = AssistantAgent(name="researcher", llm_config=llm_config)
coder = AssistantAgent(name="coder", llm_config=llm_config)
reviewer = AssistantAgent(name="reviewer", llm_config=llm_config)
user = UserProxyAgent(name="user", human_input_mode="NEVER")

agents = [researcher, coder, reviewer, user]

# Create group chat
group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=20
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# Trace group conversation
tracer.start_conversation("research_and_code", agents)

# Execute group chat
user.initiate_chat(
    manager,
    message="Research best practices for Python async programming, "
            "then implement an example"
)

tracer.end_conversation(total_messages=len(group_chat.messages))
```

## CrewAI Integration

CrewAI integration uses custom hooks to trace crew execution and task delegation.

### Implementation

```python
from crewai import Agent, Task, Crew
from crewai.agents import CrewAgentExecutor
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from typing import Optional

class A11iCrewCallbacks:
    """a11i tracing callbacks for CrewAI.

    Integrates with CrewAI's lifecycle hooks to provide observability
    for crews, agents, and tasks.
    """

    def __init__(self):
        self.tracer = trace.get_tracer("a11i.crewai")
        self._crew_span = None
        self._agent_spans = {}
        self._task_spans = {}

    def on_crew_start(self, crew: Crew):
        """Start tracing crew execution."""
        self._crew_span = self.tracer.start_span(
            f"crewai.crew.{crew.name or 'unnamed'}",
            kind=SpanKind.INTERNAL,
        )
        self._crew_span.set_attribute("a11i.framework", "crewai")
        self._crew_span.set_attribute("crewai.agents.count", len(crew.agents))
        self._crew_span.set_attribute("crewai.tasks.count", len(crew.tasks))
        self._crew_span.set_attribute("crewai.process", crew.process)

    def on_crew_end(self, crew: Crew, result: Any):
        """End crew tracing."""
        if self._crew_span:
            # Capture result
            result_str = str(result)[:1000]
            if len(str(result)) > 1000:
                result_str += "... (truncated)"
            self._crew_span.set_attribute("crewai.result", result_str)

            self._crew_span.set_status(Status(StatusCode.OK))
            self._crew_span.end()

    def on_agent_start(self, agent: Agent):
        """Start tracing agent execution."""
        if not self._crew_span:
            return

        context = trace.set_span_in_context(self._crew_span)
        span = self.tracer.start_span(
            f"crewai.agent.{agent.role}",
            context=context,
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute("crewai.agent.role", agent.role)
        span.set_attribute("crewai.agent.goal", agent.goal)
        span.set_attribute("crewai.agent.backstory", agent.backstory[:500])

        self._agent_spans[id(agent)] = span

    def on_agent_end(self, agent: Agent, result: Any):
        """End agent tracing."""
        span = self._agent_spans.pop(id(agent), None)
        if span:
            span.set_status(Status(StatusCode.OK))
            span.end()

    def on_task_start(self, task: Task):
        """Start tracing task execution."""
        if not self._crew_span:
            return

        context = trace.set_span_in_context(self._crew_span)
        span = self.tracer.start_span(
            f"crewai.task",
            context=context,
            kind=SpanKind.INTERNAL,
        )

        # Capture task details
        description = task.description[:500]
        if len(task.description) > 500:
            description += "... (truncated)"
        span.set_attribute("crewai.task.description", description)

        if task.expected_output:
            span.set_attribute("crewai.task.expected_output",
                             task.expected_output[:500])

        self._task_spans[id(task)] = span

    def on_task_end(self, task: Task, result: Any):
        """End task tracing."""
        span = self._task_spans.pop(id(task), None)
        if span:
            # Capture result
            result_str = str(result)[:1000]
            if len(str(result)) > 1000:
                result_str += "... (truncated)"
            span.set_attribute("crewai.task.result", result_str)

            span.set_status(Status(StatusCode.OK))
            span.end()

    def on_tool_start(self, tool_name: str, tool_input: Any):
        """Trace tool execution."""
        if not self._crew_span:
            return

        context = trace.set_span_in_context(self._crew_span)
        span = self.tracer.start_span(
            f"crewai.tool.{tool_name}",
            context=context,
            kind=SpanKind.CLIENT,
        )
        span.set_attribute("a11i.tool.name", tool_name)
        span.set_attribute("a11i.tool.input", str(tool_input)[:500])
        return span

    def on_tool_end(self, span, result: Any):
        """End tool tracing."""
        if span:
            span.set_attribute("a11i.tool.output", str(result)[:500])
            span.set_status(Status(StatusCode.OK))
            span.end()
```

### Usage Examples

```python
from crewai import Agent, Task, Crew, Process
from a11i.integrations.crewai import A11iCrewCallbacks

# Create callbacks
callbacks = A11iCrewCallbacks()

# Define agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="Expert in AI research with 10 years experience",
    verbose=True
)

writer = Agent(
    role="Tech Content Writer",
    goal="Craft compelling content about AI advances",
    backstory="Experienced tech writer with deep AI knowledge",
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research latest AI observability trends",
    agent=researcher,
    expected_output="Comprehensive research report"
)

writing_task = Task(
    description="Write article about AI observability",
    agent=writer,
    expected_output="Publication-ready article"
)

# Create crew with callbacks
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True
)

# Instrument with callbacks
callbacks.on_crew_start(crew)

# Execute crew - fully traced
result = crew.kickoff()

callbacks.on_crew_end(crew, result)
```

## DSPy Integration

DSPy works with OpenInference which is OpenTelemetry-compatible.

### Implementation

```python
import dspy
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Setup a11i-compatible tracing
resource = Resource.create({
    "service.name": "dspy-application",
    "a11i.project": "my-project",
})

provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint="http://a11i-collector:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# Instrument DSPy
DSPyInstrumentor().instrument(tracer_provider=provider)

# Configure DSPy
lm = dspy.OpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
dspy.settings.configure(lm=lm)

# All DSPy modules are now automatically traced!
```

### Usage Examples

#### RAG Pipeline

```python
import dspy

class RAG(dspy.Module):
    """Retrieval-Augmented Generation module (fully traced)."""

    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Create and use RAG
rag = RAG()
result = rag("What is a11i?")  # Fully traced!

# Trace hierarchy:
# dspy.RAG.forward
#   ├─ dspy.Retrieve (vector search)
#   └─ dspy.ChainOfThought.forward
#       └─ llm.openai.chat_completion
```

#### Multi-Step Reasoning

```python
import dspy

class MultiHopQA(dspy.Module):
    """Multi-hop question answering with intermediate reasoning."""

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_query = dspy.ChainOfThought("context, question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # First retrieval
        context1 = self.retrieve(question).passages

        # Generate follow-up query
        follow_up = self.generate_query(context=context1, question=question)

        # Second retrieval
        context2 = self.retrieve(follow_up.search_query).passages

        # Final answer
        return self.generate_answer(
            context=context1 + context2,
            question=question
        )

# Execute - complete trace captured
qa = MultiHopQA()
answer = qa("Who invented the company that makes the iPhone?")
```

## Haystack Integration

Haystack integration uses the framework's Tracer interface.

### Implementation

```python
from haystack import Pipeline
from haystack.tracing import OpenTelemetryTracer
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryBM25Retriever
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Setup a11i tracer
resource = Resource.create({
    "service.name": "haystack-app",
    "a11i.project": "my-project",
})

exporter = OTLPSpanExporter(endpoint="http://a11i-collector:4317")
tracer = OpenTelemetryTracer(
    tracer_provider=trace.get_tracer_provider(),
    service_name="haystack-app",
)

# Configure Haystack pipeline with tracer
pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store))
pipeline.add_component("generator", OpenAIGenerator(model="gpt-4"))
pipeline.connect("retriever", "generator")

# Attach tracer
pipeline.tracer = tracer

# Run traced pipeline
result = pipeline.run({
    "retriever": {"query": "What is observability?"},
    "generator": {"prompt": "Based on context: {documents}"}
})
```

### Usage Examples

```python
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

# Create RAG pipeline
pipeline = Pipeline()
pipeline.add_component("prompt_builder", PromptBuilder(template="..."))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4"))
pipeline.connect("prompt_builder", "llm")

# Attach a11i tracer
pipeline.tracer = tracer

# Execute - fully traced
result = pipeline.run({"prompt_builder": {"question": "What is a11i?"}})
```

## LlamaIndex Integration

LlamaIndex integration uses callback handlers similar to LangChain.

### Implementation

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from typing import Any, Dict, Optional

class A11iLlamaIndexHandler(BaseCallbackHandler):
    """a11i callback handler for LlamaIndex.

    Captures LlamaIndex operations including queries, retrievals,
    embeddings, and LLM calls.
    """

    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.tracer = trace.get_tracer("a11i.llamaindex")
        self._spans = {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs,
    ):
        """Start event tracing."""
        # Create span for this event
        span = self.tracer.start_span(
            f"llamaindex.{event_type.value}",
            kind=self._get_span_kind(event_type),
        )

        span.set_attribute("a11i.framework", "llamaindex")
        span.set_attribute("event_type", event_type.value)

        # Add event-specific attributes
        if payload:
            self._add_event_attributes(span, event_type, payload)

        self._spans[event_id] = span

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs,
    ):
        """End event tracing."""
        span = self._spans.pop(event_id, None)
        if not span:
            return

        # Capture results
        if payload:
            self._add_result_attributes(span, event_type, payload)

        span.set_status(Status(StatusCode.OK))
        span.end()

    def _get_span_kind(self, event_type: CBEventType) -> SpanKind:
        """Determine appropriate span kind for event type."""
        if event_type in [CBEventType.LLM, CBEventType.EMBEDDING]:
            return SpanKind.CLIENT
        return SpanKind.INTERNAL

    def _add_event_attributes(
        self,
        span: trace.Span,
        event_type: CBEventType,
        payload: Dict[str, Any]
    ):
        """Add event-specific attributes to span."""
        if event_type == CBEventType.QUERY:
            if EventPayload.QUERY_STR in payload:
                query = str(payload[EventPayload.QUERY_STR])[:500]
                span.set_attribute("llamaindex.query", query)

        elif event_type == CBEventType.RETRIEVE:
            if EventPayload.QUERY_STR in payload:
                query = str(payload[EventPayload.QUERY_STR])[:500]
                span.set_attribute("llamaindex.retrieve.query", query)

        elif event_type == CBEventType.LLM:
            if EventPayload.MESSAGES in payload:
                messages = payload[EventPayload.MESSAGES]
                span.set_attribute("gen_ai.request.messages", len(messages))
            if EventPayload.PROMPT in payload:
                prompt = str(payload[EventPayload.PROMPT])[:500]
                span.set_attribute("gen_ai.prompt", prompt)

    def _add_result_attributes(
        self,
        span: trace.Span,
        event_type: CBEventType,
        payload: Dict[str, Any]
    ):
        """Add result attributes to span."""
        if event_type == CBEventType.LLM:
            if EventPayload.RESPONSE in payload:
                response = str(payload[EventPayload.RESPONSE])[:500]
                span.set_attribute("gen_ai.response", response)

            # Token usage
            if EventPayload.PROMPT_TOKENS in payload:
                span.set_attribute("gen_ai.usage.input_tokens",
                                 payload[EventPayload.PROMPT_TOKENS])
            if EventPayload.COMPLETION_TOKENS in payload:
                span.set_attribute("gen_ai.usage.output_tokens",
                                 payload[EventPayload.COMPLETION_TOKENS])

        elif event_type == CBEventType.RETRIEVE:
            if EventPayload.NODES in payload:
                nodes = payload[EventPayload.NODES]
                span.set_attribute("llamaindex.retrieved.count", len(nodes))

    def start_trace(self, trace_id: Optional[str] = None):
        """Required by LlamaIndex interface."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, Any]] = None
    ):
        """Required by LlamaIndex interface."""
        pass
```

### Usage Examples

#### Query Engine

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.callbacks import CallbackManager
from a11i.integrations.llamaindex import A11iLlamaIndexHandler

# Setup callback
handler = A11iLlamaIndexHandler()
callback_manager = CallbackManager([handler])
Settings.callback_manager = callback_manager

# Load documents and create index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine (automatically traced)
query_engine = index.as_query_engine()

# Execute query - fully traced
response = query_engine.query("What is a11i?")

# Trace hierarchy:
# llamaindex.QUERY
#   ├─ llamaindex.RETRIEVE (vector search)
#   ├─ llamaindex.SYNTHESIZE
#   └─ llamaindex.LLM (response generation)
```

#### Chat Engine with Memory

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# Create chat engine with memory
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
)

# Multi-turn conversation (each turn traced)
response1 = chat_engine.chat("What is AI observability?")
response2 = chat_engine.chat("How does it help with agents?")
response3 = chat_engine.chat("What are the key metrics?")
```

## Custom Agent Frameworks

For proprietary or custom agent frameworks, use a11i's decorator patterns.

### Implementation

```python
from a11i import observe, agent_loop, configure

# Configure a11i
configure(api_key="your-key", project="custom-agent")

class CustomAgent:
    """Custom agent implementation with a11i instrumentation."""

    def __init__(self, name: str, llm):
        self.name = name
        self.llm = llm

    @agent_loop(name="custom_agent", max_iterations=10)
    async def run(self, goal: str, _a11i_ctx=None):
        """Main agent loop (automatically traced)."""
        ctx = _a11i_ctx
        done = False
        context = []

        while not done:
            with ctx.increment_iteration():
                # Think phase
                thought = await self.think(goal, context)

                # Act phase
                action = await self.act(thought)

                # Observe phase
                observation = await self.observe(action)
                context.append(observation)

                # Check completion
                done = await self.is_complete(observation, goal)

        return self.synthesize(context)

    @observe(name="agent.think", capture_input=True)
    async def think(self, goal: str, context: list) -> str:
        """Reasoning phase."""
        prompt = f"Goal: {goal}\nContext: {context}\nWhat should I do next?"
        return await self.llm.invoke(prompt)

    @observe(name="agent.act", capture_output=True)
    async def act(self, thought: str) -> dict:
        """Action execution phase."""
        # Parse thought and execute action
        action = self.parse_action(thought)
        result = await self.execute_action(action)
        return result

    @observe(name="agent.observe")
    async def observe(self, action_result: dict) -> str:
        """Observation phase."""
        # Process action result
        return self.format_observation(action_result)

    @observe(name="agent.tool.execute", span_kind=SpanKind.CLIENT)
    async def execute_action(self, action: dict) -> dict:
        """Execute a tool/action."""
        tool_name = action["tool"]
        tool_input = action["input"]

        # Execute tool
        result = await self.tools[tool_name](tool_input)
        return {"tool": tool_name, "result": result}
```

### Usage Example

```python
# Create custom agent
agent = CustomAgent(name="research_agent", llm=my_llm)

# Run agent - fully traced
result = await agent.run(goal="Research AI observability best practices")

# Trace hierarchy:
# invoke_agent.custom_agent
#   ├─ iteration.1
#   │   ├─ agent.think
#   │   │   └─ llm.openai.chat_completion
#   │   ├─ agent.act
#   │   │   └─ agent.tool.execute (web_search)
#   │   └─ agent.observe
#   ├─ iteration.2
#   │   ├─ agent.think
#   │   ├─ agent.act
#   │   └─ agent.observe
#   └─ iteration.3
#       ├─ agent.think
#       ├─ agent.act
#       └─ agent.observe
```

## Auto-Instrumentation Setup

One-line setup for automatic instrumentation of all supported frameworks.

### Universal Auto-Instrumentation

```python
from a11i import configure, auto_instrument

# Configure a11i
configure(
    api_key="your-api-key",
    project="my-agent-project",
    environment="production"
)

# Auto-detect and instrument all available frameworks
auto_instrument()

# Now all supported frameworks are automatically traced!
from langchain import LLMChain
from llama_index.core import VectorStoreIndex
import dspy

# All operations are traced with zero additional code
```

### Selective Auto-Instrumentation

```python
from a11i import auto_instrument

# Instrument only specific frameworks
auto_instrument(frameworks=["langchain", "openai", "llama_index"])

# Or exclude specific frameworks
auto_instrument(exclude=["autogen"])
```

### Framework-Specific Setup

```python
# LangChain only
from a11i.integrations.langchain import auto_instrument_langchain
auto_instrument_langchain()

# LlamaIndex only
from a11i.integrations.llamaindex import auto_instrument_llamaindex
auto_instrument_llamaindex()

# DSPy only (via OpenInference)
from a11i.integrations.dspy import auto_instrument_dspy
auto_instrument_dspy()
```

### Environment-Based Auto-Instrumentation

```bash
# Enable via environment variable
export A11I_AUTO_INSTRUMENT=true
export A11I_API_KEY=your-key
export A11I_PROJECT=my-project

# Add to sitecustomize.py
# (automatically loads on Python startup)
```

```python
# sitecustomize.py
import os
if os.getenv("A11I_AUTO_INSTRUMENT") == "true":
    import a11i
    a11i.configure()
    a11i.auto_instrument()
```

## Best Practices

### Framework Selection Guidelines

| Framework | Best For | a11i Integration Quality |
|-----------|----------|-------------------------|
| **LangChain** | General-purpose chains and agents | ✅ Excellent (native callbacks) |
| **Semantic Kernel** | Microsoft ecosystem, planners | ✅ Excellent (native OTel) |
| **LlamaIndex** | RAG applications, document QA | ✅ Excellent (native callbacks) |
| **DSPy** | Prompt optimization, research | ✅ Excellent (OpenInference) |
| **CrewAI** | Multi-agent collaboration | 🔧 Good (custom hooks) |
| **AutoGen** | Complex multi-agent conversations | 🔧 Good (custom tracing) |
| **Haystack** | NLP pipelines, production systems | 🔧 Good (tracer interface) |

### Performance Optimization

```python
# ❌ Bad: Over-instrumentation
@observe()  # Unnecessary overhead
def tokenize(text: str):
    return text.split()

# ✅ Good: Instrument at appropriate level
@observe()
def process_document(doc: str):
    tokens = tokenize(doc)  # Not instrumented
    return analyze_tokens(tokens)

# ✅ Good: Use sampling for high-frequency calls
@observe(sample_rate=0.01)  # Trace 1% of calls
def frequent_operation():
    pass
```

### Error Handling

```python
# ✅ Good: Let framework handle errors naturally
@observe()
async def agent_step():
    return await llm.invoke(prompt)  # Exceptions auto-captured

# ❌ Bad: Swallowing errors
@observe()
async def agent_step():
    try:
        return await llm.invoke(prompt)
    except Exception:
        return None  # Error not visible!
```

### Context Propagation

```python
# ✅ Good: Proper context propagation across async boundaries
from opentelemetry import trace, context

@observe()
async def parent_operation():
    # Context automatically propagated to child
    result = await child_operation()
    return result

@observe()
async def child_operation():
    # Receives parent context automatically
    pass

# For manual context management:
async def manual_context():
    token = context.attach(trace.set_span_in_context(current_span))
    try:
        await async_operation()
    finally:
        context.detach(token)
```

### Span Naming Conventions

```python
# ✅ Good: Hierarchical, descriptive names
@observe(name="agent.research.web_search")
@observe(name="llm.openai.chat_completion")
@observe(name="tool.calculator.evaluate")

# ❌ Bad: Generic or inconsistent names
@observe(name="function1")
@observe(name="DO_STUFF")
@observe(name="op")
```

### Attribute Best Practices

```python
# ✅ Good: Structured, queryable attributes
@observe(attributes={
    "agent.name": "researcher",
    "agent.type": "autonomous",
    "task.category": "research",
    "task.priority": "high"
})

# ❌ Bad: Unstructured or huge attributes
@observe(attributes={
    "data": huge_json_blob,  # Too large
    "misc": "stuff",  # Too vague
})
```

## Key Takeaways

| Framework | Integration Pattern | Setup Effort | Observability Depth |
|-----------|-------------------|--------------|---------------------|
| **LangChain** | CallbackHandler | 5 minutes | ★★★★★ Complete |
| **Semantic Kernel** | Native OTel | 5 minutes | ★★★★★ Complete |
| **LlamaIndex** | Callbacks | 5 minutes | ★★★★★ Complete |
| **DSPy** | OpenInference | 10 minutes | ★★★★★ Complete |
| **AutoGen** | TracerProvider | 15 minutes | ★★★★☆ Good |
| **CrewAI** | Hooks | 15 minutes | ★★★★☆ Good |
| **Haystack** | Tracer Interface | 10 minutes | ★★★★☆ Good |
| **Custom** | Decorators | 30 minutes | ★★★★★ Complete |

### Quick Reference

**Fastest Setup:**
```python
from a11i import configure, auto_instrument
configure(api_key="key")
auto_instrument()
```

**Best for LangChain:**
```python
from a11i.integrations.langchain import A11iCallbackHandler
handler = A11iCallbackHandler()
chain = LLMChain(llm=llm, callbacks=[handler])
```

**Best for Custom Agents:**
```python
from a11i import observe, agent_loop

@agent_loop(name="my_agent")
async def run(goal, _a11i_ctx=None):
    # Agent implementation
    pass
```

### Implementation Checklist

- [ ] Choose framework integration approach (auto vs manual)
- [ ] Configure a11i with API key and project
- [ ] Add framework-specific callbacks/handlers
- [ ] Test trace generation with simple example
- [ ] Verify spans appear in a11i dashboard
- [ ] Add custom attributes for business context
- [ ] Implement error handling and recording
- [ ] Optimize performance with sampling if needed
- [ ] Document integration for team members

### Next Steps

- **View Traces**: Check the a11i dashboard to see captured traces
- **Custom Metrics**: Add business-specific metrics and attributes
- **Alerting**: Configure alerts for agent failures or performance issues
- **Cost Analysis**: Review token usage and cost attribution
- **Optimization**: Use trace data to identify optimization opportunities

---

**Related Documentation:**
- [SDK and Library Implementation](/home/becker/projects/a11i/docs/04-implementation/sdk-library.md) - Core SDK documentation
- [OpenTelemetry Integration](/home/becker/projects/a11i/docs/03-core-platform/opentelemetry-integration.md) - OTel configuration
- [Span Hierarchy](/home/becker/projects/a11i/docs/03-core-platform/span-hierarchy.md) - Trace structure patterns

---

*Document Status: Stable | Last Updated: 2025-11-26 | Maintained by: a11i Documentation Team*
