---
title: "Core Metrics: The Five Pillars of AI Agent Observability"
category: "Core Platform"
tags: ["metrics", "observability", "token-usage", "cost-tracking", "context-saturation", "tool-errors", "loop-detection"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "../02-architecture/system-architecture.md"
  - "../02-architecture/data-pipeline.md"
  - "./instrumentation-sdk.md"
  - "./alerting-system.md"
  - "../04-use-cases/cost-optimization.md"
---

# Core Metrics: The Five Pillars of AI Agent Observability

## Table of Contents

- [Introduction](#introduction)
- [Metrics Philosophy](#metrics-philosophy)
- [The Five Core Metrics](#the-five-core-metrics)
  - [1. ai.token_usage_counter](#1-aitoken_usage_counter)
  - [2. ai.cost_estimate_counter](#2-aicost_estimate_counter)
  - [3. ai.context_saturation_gauge](#3-aicontext_saturation_gauge)
  - [4. ai.tool_error_rate](#4-aitool_error_rate)
  - [5. ai.loop_velocity_gauge](#5-ailoop_velocity_gauge)
- [Additional Metrics](#additional-metrics)
  - [Latency Metrics](#latency-metrics)
  - [Quality Metrics](#quality-metrics)
  - [System Health Metrics](#system-health-metrics)
- [Metrics Dashboard](#metrics-dashboard)
- [Implementation Guidelines](#implementation-guidelines)
- [Key Takeaways](#key-takeaways)

## Introduction

AI agents are fundamentally different from traditional software systems. They consume variable resources (tokens), make probabilistic decisions, and can fail silently without explicit errors. Traditional infrastructure metrics (CPU, memory, request latency) are necessary but insufficient for understanding agent behavior, controlling costs, and preventing cognitive failures.

**a11i defines five core metrics that capture the unique characteristics of AI agent systems:**

1. **Token Usage** - Resource consumption foundation
2. **Cost Estimation** - Financial observability and accountability
3. **Context Saturation** - Cognitive capacity and performance
4. **Tool Error Rate** - Execution fidelity and reliability
5. **Loop Velocity** - Runaway detection and efficiency

These metrics provide the essential signals for operating production AI agents with confidence, predictability, and cost control.

## Metrics Philosophy

### Why These Five Metrics?

The five core metrics were selected based on analysis of real-world AI agent failures and operational challenges:

**Token Usage**: Every LLM operation consumes tokens, making token counting the foundation of all other metrics. Without accurate token tracking, cost attribution and context management are impossible.

**Cost Estimation**: Token usage without cost context is operationally meaningless. Engineering teams need to understand the financial impact of agent behaviors to make informed optimization decisions.

**Context Saturation**: The "Lost in the Middle" effect and quadratic attention complexity mean agents degrade in quality and performance as context windows fill. Tracking saturation provides early warning of cognitive decline.

**Tool Error Rate**: Agents are only as reliable as their tools. High tool error rates indicate either unstable infrastructure or poor prompt engineering, both requiring immediate attention.

**Loop Velocity**: Infinite reasoning loops are a unique failure mode of autonomous agents. Traditional timeout-based detection is insufficient; velocity tracking enables early intervention before costs spiral.

### Semantic Conventions

All metrics follow **OpenTelemetry Semantic Conventions** with a11i-specific extensions:

```yaml
# Standard OTel attributes
gen_ai.system: "openai" | "anthropic" | "bedrock"
gen_ai.request.model: "gpt-4-turbo" | "claude-3-5-sonnet"
gen_ai.response.id: "chatcmpl-123"
gen_ai.usage.input_tokens: 1234
gen_ai.usage.output_tokens: 567

# a11i agent-specific extensions
a11i.agent.name: "research-assistant"
a11i.agent.framework: "langchain" | "crewai" | "autogen"
a11i.agent.loop.iteration: 3
a11i.agent.thought.phase: "think" | "act" | "observe"
a11i.tenant.id: "acme-corp"
a11i.workflow.id: "data-analysis-pipeline"
```

## The Five Core Metrics

### 1. ai.token_usage_counter

#### What It Measures

The total number of tokens consumed in both prompts (input) and completions (output) for each LLM request. This is the fundamental unit of resource consumption in AI agent systems.

**Metric Type:** Counter (monotonically increasing)

**Unit:** `tokens`

**Dimensions:**
- `type`: `input` | `output`
- `model`: LLM model identifier (e.g., `gpt-4-turbo`, `claude-3-5-sonnet`)
- `agent_name`: Name of the agent making the request
- `tenant_id`: Multi-tenant identifier for cost attribution

#### Why It Matters

Token usage is the foundation of all AI agent observability:

- **Cost Basis**: Direct correlation to LLM API billing
- **Context Management**: Understanding how context windows fill over time
- **Resource Utilization**: Identifying inefficient prompting patterns
- **Capacity Planning**: Forecasting infrastructure needs based on usage trends

**Critical Insight:** Input tokens and output tokens have different costs (typically 3-5x difference) and represent different optimization opportunities. Tracking them separately is essential.

#### Implementation

```python
from opentelemetry import metrics

# Initialize meter
meter = metrics.get_meter("a11i")

# Create token counter
token_counter = meter.create_counter(
    name="ai.token_usage",
    description="Token usage for LLM requests",
    unit="tokens"
)

# On each LLM call
def track_llm_call(response, agent_name: str, tenant_id: str):
    """Track token usage from LLM response."""

    # Extract token counts (provider-specific)
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    model = response.model

    # Record input tokens
    token_counter.add(
        input_tokens,
        attributes={
            "type": "input",
            "model": model,
            "agent_name": agent_name,
            "tenant_id": tenant_id,
            "gen_ai.system": "openai",  # OTel convention
            "gen_ai.request.model": model
        }
    )

    # Record output tokens
    token_counter.add(
        output_tokens,
        attributes={
            "type": "output",
            "model": model,
            "agent_name": agent_name,
            "tenant_id": tenant_id,
            "gen_ai.system": "openai",
            "gen_ai.request.model": model
        }
    )
```

#### Calculation Methods

**Provider-Returned Counts (Preferred):**

All major LLM providers return token counts in their API responses. This is the most accurate method and should be used whenever available:

```python
# OpenAI
tokens = {
    "input": response.usage.prompt_tokens,
    "output": response.usage.completion_tokens,
    "total": response.usage.total_tokens
}

# Anthropic
tokens = {
    "input": response.usage.input_tokens,
    "output": response.usage.output_tokens
}

# AWS Bedrock
tokens = {
    "input": response["usage"]["inputTokens"],
    "output": response["usage"]["outputTokens"]
}
```

**Local Tokenization (Fallback):**

For providers that don't return counts, or for pre-flight estimation:

```python
import tiktoken  # OpenAI tokenizer

def count_tokens_local(text: str, model: str) -> int:
    """Count tokens using local tokenizer."""

    # Model-specific encoding
    encoding_map = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
    }

    encoding_name = encoding_map.get(model, "cl100k_base")
    encoding = tiktoken.get_encoding(encoding_name)

    return len(encoding.encode(text))

# For non-OpenAI models, use SentencePiece or provider tokenizer
```

**Caching for Performance:**

Repeated prompts (system prompts, few-shot examples) should be cached:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def count_tokens_cached(text: str, model: str) -> int:
    """Cached token counting for repeated prompts."""
    return count_tokens_local(text, model)
```

#### Visualization

**Time Series - Token Consumption Over Time:**

```
Tokens per Hour
│
15K ┤                                    ╭─╮
    │                               ╭────╯ ╰─╮
12K ┤                          ╭────╯        ╰─╮
    │                     ╭────╯                ╰─╮
9K  ┤                ╭────╯                       ╰─╮
    │           ╭────╯
6K  ┤      ╭────╯                    Input Tokens ━━━
    │ ╭────╯                         Output Tokens ━━━
3K  ┤─╯
    │
0   └─────┬─────┬─────┬─────┬─────┬─────┬─────┬────
         00:00 04:00 08:00 12:00 16:00 20:00 24:00
```

**Breakdown by Model:**

```
Token Distribution by Model (Last 24h)

GPT-4-Turbo     ████████████████████ 45% (2.3M tokens)
Claude-3.5      ████████████████     35% (1.8M tokens)
GPT-3.5-Turbo   ████████             20% (1.0M tokens)
```

**Input vs Output Ratio:**

Ideal ratio varies by use case. High output ratio may indicate inefficient prompting.

```
Agent Token Usage Patterns

Research Agent    Input ████████ Output ████  (2:1 ratio)
Code Generator    Input ███      Output ██████████ (1:3 ratio)
Summarizer        Input █████████ Output ██  (4:1 ratio)
```

#### Alert Conditions

```yaml
alerts:
  # High token usage spike
  - name: "Token Usage Spike"
    condition: "rate(ai.token_usage[5m]) > 2 * rate(ai.token_usage[1h])"
    severity: "warning"
    description: "Token usage increased 2x above hourly baseline"

  # Agent exceeding token budget
  - name: "Agent Token Budget Exceeded"
    condition: "sum(ai.token_usage{agent_name='$agent'}) > $budget"
    severity: "critical"
    description: "Agent exceeded daily token budget"

  # Unusual input/output ratio
  - name: "Abnormal Token Ratio"
    condition: |
      (sum(ai.token_usage{type='output'}) /
       sum(ai.token_usage{type='input'})) > 10
    severity: "warning"
    description: "Output tokens 10x higher than input (possible loop)"
```

#### Advanced Analytics

**Token Efficiency Score:**

```python
def calculate_token_efficiency(
    output_quality: float,  # 0-1 score
    total_tokens: int
) -> float:
    """
    Calculate token efficiency metric.
    Higher is better: more value per token.
    """
    return output_quality / (total_tokens / 1000)

# Example: 0.9 quality using 5000 tokens = 0.18 efficiency
# Compare across agents/prompts to optimize
```

**Token Waste Detection:**

```python
def detect_token_waste(conversation_history: List[Message]) -> dict:
    """Identify wasted tokens in conversation history."""

    waste_analysis = {
        "repeated_context": 0,
        "unused_tool_schemas": 0,
        "redundant_examples": 0
    }

    # Detect repeated context across turns
    for i, msg in enumerate(conversation_history[:-1]):
        if msg.content in conversation_history[i+1].content:
            waste_analysis["repeated_context"] += len(msg.tokens)

    # Detect unused tool schemas
    tools_defined = set(get_tool_names(conversation_history))
    tools_called = set(get_called_tools(conversation_history))
    unused_tools = tools_defined - tools_called
    waste_analysis["unused_tool_schemas"] = (
        len(unused_tools) * AVG_TOOL_SCHEMA_TOKENS
    )

    return waste_analysis
```

---

### 2. ai.cost_estimate_counter

#### What It Measures

The estimated dollar cost of each LLM request, calculated from token usage multiplied by the provider's rate card. This metric enables financial visibility and cost attribution across agents, teams, and workflows.

**Metric Type:** Counter (monotonically increasing)

**Unit:** `USD` (or configured currency)

**Dimensions:**
- `provider`: LLM provider (e.g., `openai`, `anthropic`)
- `model`: Specific model used
- `agent_name`: Agent making the request
- `tenant_id`: Tenant for multi-tenant cost allocation
- `workflow_id`: Business workflow for cost attribution

#### Why It Matters

Token counts without cost context are operationally insufficient. Engineering and product teams need to understand:

- **Financial Impact**: What does each agent interaction cost?
- **Budget Management**: Are we staying within allocated budgets?
- **Cost Attribution**: Which teams, workflows, or users drive costs?
- **Optimization ROI**: What's the financial benefit of optimizations?

**Critical Business Value:** Cost metrics enable chargeback/showback models, allowing platform teams to attribute AI spending to business units and justify infrastructure investments.

#### Implementation

```python
from decimal import Decimal

# Cost Registry (hot-reloadable from config)
COST_REGISTRY = {
    # Provider, Model, Type: Cost per 1K tokens
    ("openai", "gpt-4-turbo", "input"): Decimal("0.01"),
    ("openai", "gpt-4-turbo", "output"): Decimal("0.03"),
    ("openai", "gpt-4o", "input"): Decimal("0.000003"),
    ("openai", "gpt-4o", "output"): Decimal("0.000015"),
    ("anthropic", "claude-3-5-sonnet-20241022", "input"): Decimal("0.003"),
    ("anthropic", "claude-3-5-sonnet-20241022", "output"): Decimal("0.015"),
    ("anthropic", "claude-3-haiku-20240307", "input"): Decimal("0.00025"),
    ("anthropic", "claude-3-haiku-20240307", "output"): Decimal("0.00125"),
}

def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int
) -> Decimal:
    """Calculate cost for LLM request."""

    # Get rates from registry
    input_rate = COST_REGISTRY.get(
        (provider, model, "input"),
        Decimal("0")  # Unknown model = $0
    )
    output_rate = COST_REGISTRY.get(
        (provider, model, "output"),
        Decimal("0")
    )

    # Calculate costs
    input_cost = (Decimal(input_tokens) * input_rate) / Decimal("1000")
    output_cost = (Decimal(output_tokens) * output_rate) / Decimal("1000")

    return input_cost + output_cost

# Track cost metric
cost_counter = meter.create_counter(
    name="ai.cost_estimate",
    description="Estimated cost of LLM requests in USD",
    unit="USD"
)

def track_llm_cost(response, agent_name: str, tenant_id: str, workflow_id: str):
    """Track cost for LLM request."""

    cost = calculate_cost(
        provider="openai",
        model=response.model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens
    )

    cost_counter.add(
        float(cost),
        attributes={
            "provider": "openai",
            "model": response.model,
            "agent_name": agent_name,
            "tenant_id": tenant_id,
            "workflow_id": workflow_id,
            "gen_ai.system": "openai",
            "gen_ai.request.model": response.model
        }
    )
```

#### Multi-Tenancy Cost Attribution

Cost tracking becomes exponentially more valuable with proper attribution:

```python
# Cost hierarchy for attribution
COST_DIMENSIONS = [
    "tenant_id",      # Top-level organization
    "team_id",        # Engineering team
    "workflow_id",    # Business workflow
    "agent_name",     # Specific agent
    "user_id",        # End user (for user-facing agents)
]

def track_attributed_cost(
    cost: Decimal,
    tenant_id: str,
    team_id: str = None,
    workflow_id: str = None,
    agent_name: str = None,
    user_id: str = None
):
    """Track cost with full attribution hierarchy."""

    attributes = {"tenant_id": tenant_id}

    if team_id:
        attributes["team_id"] = team_id
    if workflow_id:
        attributes["workflow_id"] = workflow_id
    if agent_name:
        attributes["agent_name"] = agent_name
    if user_id:
        attributes["user_id"] = user_id

    cost_counter.add(float(cost), attributes=attributes)
```

#### Visualization

**Cost Sunburst Chart (Hierarchical Attribution):**

```
                    ┌──────────────────┐
                    │   Total: $1,245  │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐      ┌─────▼─────┐     ┌─────▼─────┐
    │ Tenant A  │      │ Tenant B  │     │ Tenant C  │
    │   $456    │      │   $678    │     │   $111    │
    └─────┬─────┘      └─────┬─────┘     └───────────┘
          │                  │
    ┌─────┼─────┐      ┌─────┼─────┐
    │           │      │           │
 ┌──▼──┐    ┌──▼──┐ ┌──▼──┐    ┌──▼──┐
 │Team1│    │Team2│ │Team3│    │Team4│
 │$234 │    │$222 │ │$345 │    │$333 │
 └─────┘    └─────┘ └─────┘    └─────┘
```

**Cost Trends Over Time:**

```
Daily Cost Trends

$200 ┤                                        ╭●
     │                                   ╭────╯
$150 ┤                              ╭────╯
     │                         ╭────╯
$100 ┤                    ╭────╯
     │               ╭────╯                Budget: $150/day
$50  ┤          ╭────╯                    ━━━━━━━━━━━━━━━━
     │     ╭────╯
$0   └─────┴────┴────┴────┴────┴────┴────┴────┴────
         Mon  Tue  Wed  Thu  Fri  Sat  Sun
```

**Top Cost Drivers (Last 24h):**

```
Agent                     Cost      % of Total   Requests
────────────────────────────────────────────────────────────
code-generator           $45.67      36.7%       1,234
research-assistant       $32.10      25.8%       2,456
data-analyzer           $18.92      15.2%         567
summarizer              $12.45      10.0%       5,678
document-qa             $15.24      12.3%       3,890
```

#### Alert Conditions

```yaml
alerts:
  # Budget exceeded
  - name: "Daily Budget Exceeded"
    condition: "sum(ai.cost_estimate{tenant_id='$tenant'}) > $daily_budget"
    severity: "critical"
    description: "Tenant exceeded daily cost budget"
    actions:
      - "notify: finance-team@company.com"
      - "throttle: reduce_agent_quota"

  # Anomaly detection
  - name: "Cost Spike Detected"
    condition: |
      sum(ai.cost_estimate[1h]) >
      2 * avg(ai.cost_estimate[24h] offset 1h)
    severity: "warning"
    description: "Hourly cost 2x above 24h average"

  # Runaway request
  - name: "Single Request High Cost"
    condition: "ai.cost_estimate > $5.00"
    severity: "warning"
    description: "Individual request cost exceeds threshold"
    metadata:
      trace_id: "$trace_id"
      agent_name: "$agent_name"

  # Model selection opportunity
  - name: "Expensive Model for Simple Task"
    condition: |
      ai.cost_estimate{model='gpt-4-turbo'} AND
      ai.token_usage{type='input'} < 500
    severity: "info"
    description: "GPT-4 used for <500 token task (consider GPT-3.5)"
```

#### Advanced Analytics

**Cost Optimization Recommendations:**

```python
def analyze_cost_optimization_opportunities(
    time_range: tuple
) -> List[Recommendation]:
    """Identify cost optimization opportunities."""

    recommendations = []

    # Query cost data
    query = """
        SELECT
            agent_name,
            model,
            avg(input_tokens) as avg_input,
            avg(output_tokens) as avg_output,
            sum(cost_usd) as total_cost,
            count(*) as request_count
        FROM llm_requests
        WHERE timestamp BETWEEN %(start)s AND %(end)s
        GROUP BY agent_name, model
    """

    results = execute_query(query, time_range)

    for row in results:
        # Opportunity 1: Expensive model for small prompts
        if row.model == "gpt-4-turbo" and row.avg_input < 1000:
            potential_savings = calculate_savings(
                current_model="gpt-4-turbo",
                suggested_model="gpt-3.5-turbo",
                request_count=row.request_count,
                avg_tokens=(row.avg_input, row.avg_output)
            )

            recommendations.append({
                "type": "model_downgrade",
                "agent": row.agent_name,
                "current_model": "gpt-4-turbo",
                "suggested_model": "gpt-3.5-turbo",
                "potential_savings_monthly": potential_savings,
                "confidence": "high"
            })

        # Opportunity 2: High output tokens (context optimization)
        if row.avg_output > row.avg_input * 2:
            recommendations.append({
                "type": "context_optimization",
                "agent": row.agent_name,
                "issue": "High output/input ratio suggests excessive context",
                "suggestion": "Review prompt engineering and context pruning",
                "potential_savings_monthly": row.total_cost * 0.3
            })

    return recommendations
```

**Cost Forecasting:**

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_monthly_cost(historical_data: pd.DataFrame) -> dict:
    """Forecast next month's cost using exponential smoothing."""

    # Aggregate daily costs
    daily_costs = historical_data.groupby('date')['cost_usd'].sum()

    # Fit exponential smoothing model
    model = ExponentialSmoothing(
        daily_costs,
        seasonal_periods=7,  # Weekly seasonality
        trend='add',
        seasonal='add'
    )
    fitted = model.fit()

    # Forecast next 30 days
    forecast = fitted.forecast(steps=30)

    return {
        "forecast_total": forecast.sum(),
        "forecast_daily_avg": forecast.mean(),
        "confidence_interval_95": (
            forecast.sum() * 0.85,
            forecast.sum() * 1.15
        ),
        "trend": "increasing" if forecast[-1] > forecast[0] else "decreasing"
    }
```

---

### 3. ai.context_saturation_gauge

#### What It Measures

The percentage of an LLM's context window currently utilized by the prompt and expected completion. Context saturation indicates how close an agent is to its cognitive capacity limit.

**Metric Type:** Gauge (current value, can increase or decrease)

**Unit:** `percentage` (0-100%)

**Dimensions:**
- `model`: LLM model with specific context limit
- `agent_name`: Agent managing the context
- `session_id`: Conversation or workflow session
- `phase`: `planning` | `execution` | `completion`

#### Why It Matters

Context window saturation is a leading indicator of agent cognitive failure:

- **Lost in the Middle Effect**: Models perform poorly on information in the middle of long contexts
- **Quadratic Attention Complexity**: Latency increases dramatically as context grows
- **Quality Degradation**: Responses become less coherent and relevant at high saturation
- **Cost Inefficiency**: Paying for tokens that don't contribute to quality

**Critical Performance Threshold:** Research shows significant quality degradation above 80% saturation, with severe issues above 95%.

#### Implementation

```python
# Context limits registry
CONTEXT_LIMITS = {
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-haiku-20240307": 200000,
    "gemini-1.5-pro": 2000000,
}

def calculate_context_saturation(
    model: str,
    prompt_tokens: int,
    expected_completion_tokens: int = 0,
    reserved_tokens: int = 100  # Safety margin
) -> dict:
    """
    Calculate context saturation metrics.

    Args:
        model: LLM model identifier
        prompt_tokens: Current prompt size in tokens
        expected_completion_tokens: Expected completion size (if known)
        reserved_tokens: Safety margin for token count discrepancies

    Returns:
        Dictionary with saturation metrics and risk indicators
    """
    max_context = CONTEXT_LIMITS.get(model, 4096)

    # Total tokens including expected completion
    total_tokens = prompt_tokens + expected_completion_tokens + reserved_tokens

    # Calculate saturation percentage
    saturation = (total_tokens / max_context) * 100

    # Determine risk level
    if saturation >= 95:
        risk_level = "critical"
    elif saturation >= 80:
        risk_level = "high"
    elif saturation >= 60:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "saturation_pct": saturation,
        "tokens_used": total_tokens,
        "tokens_available": max_context - total_tokens,
        "max_context": max_context,
        "risk_level": risk_level,
        "at_risk": saturation > 80,
        "critical": saturation > 95
    }

# Track saturation metric
saturation_gauge = meter.create_gauge(
    name="ai.context_saturation",
    description="Context window utilization percentage",
    unit="percentage"
)

def track_context_saturation(
    model: str,
    prompt_tokens: int,
    agent_name: str,
    session_id: str
):
    """Record context saturation metric."""

    metrics = calculate_context_saturation(model, prompt_tokens)

    saturation_gauge.set(
        metrics["saturation_pct"],
        attributes={
            "model": model,
            "agent_name": agent_name,
            "session_id": session_id,
            "risk_level": metrics["risk_level"],
            "gen_ai.request.model": model
        }
    )

    # Also track absolute token counts
    token_gauge = meter.create_gauge(
        name="ai.context_tokens_used",
        description="Tokens currently in context",
        unit="tokens"
    )

    token_gauge.set(
        metrics["tokens_used"],
        attributes={
            "model": model,
            "agent_name": agent_name,
            "session_id": session_id
        }
    )
```

#### Visualization

**Context Saturation Heatmap:**

```
Agent Context Saturation (Last Hour)

Time  │ Agent-1  Agent-2  Agent-3  Agent-4  Agent-5
──────┼──────────────────────────────────────────────
10:00 │   25%     34%      67%      12%      89%
10:15 │   31%     42%      73%      18%      94%
10:30 │   28%     38%      78%      15%      97%
10:45 │   35%     45%      84%      22%      99%
11:00 │   32%     41%      88%      19%      95%

Legend: < 60%  60-80%  80-95%  > 95%
         🟢      🟡      🟠      🔴
```

**Saturation Timeline for Single Agent:**

```
Context Saturation Over Conversation

100% ┤                                            ●
     │                                       ╭────╯
 80% ┤                                  ╭────╯    CRITICAL
     │                             ╭────╯
 60% ┤                        ╭────╯             AT RISK
     │                   ╭────╯
 40% ┤              ╭────╯                       HEALTHY
     │         ╭────╯
 20% ┤    ╭────╯
     │────╯
  0% └────┴────┴────┴────┴────┴────┴────┴────
      T1   T2   T3   T4   T5   T6   T7   T8
                    (conversation turns)
```

**Context Composition Breakdown:**

```
Context Window Composition (Turn 5)

System Prompt       ████████ 12% (2.4K tokens)
Conversation Hist   ████████████████████ 35% (7.0K tokens)
Tool Schemas        ██████████ 18% (3.6K tokens)
Retrieved Context   ████████████████ 28% (5.6K tokens)
Current Query       ███ 7% (1.4K tokens)
                    ─────────────────────────────
Total: 20K / 128K tokens (15.6% saturation)
```

#### Alert Conditions

```yaml
alerts:
  # Context at risk
  - name: "Context Saturation High"
    condition: "ai.context_saturation > 80"
    severity: "warning"
    description: "Context window exceeds 80% capacity"
    actions:
      - "trigger: context_pruning_strategy"
      - "log: context_composition_details"

  # Context critical
  - name: "Context Saturation Critical"
    condition: "ai.context_saturation > 95"
    severity: "critical"
    description: "Context window at maximum capacity"
    actions:
      - "trigger: emergency_context_reset"
      - "alert: on-call-engineer"

  # Rapid saturation increase
  - name: "Context Growing Rapidly"
    condition: |
      (ai.context_saturation - ai.context_saturation offset 5m) > 20
    severity: "warning"
    description: "Context grew >20% in 5 minutes"

  # Sustained high saturation
  - name: "Sustained High Saturation"
    condition: |
      avg(ai.context_saturation[15m]) > 85
    severity: "warning"
    description: "Context sustained above 85% for 15 minutes"
```

#### Advanced Analytics

**Context Pruning Effectiveness:**

```python
def analyze_context_pruning(
    before_tokens: int,
    after_tokens: int,
    quality_before: float,
    quality_after: float
) -> dict:
    """Measure effectiveness of context pruning strategies."""

    tokens_removed = before_tokens - after_tokens
    reduction_pct = (tokens_removed / before_tokens) * 100

    # Quality delta (negative means quality decreased)
    quality_delta = quality_after - quality_before

    # Efficiency score: how much we reduced without hurting quality
    if quality_delta >= 0:
        # No quality loss = excellent
        efficiency = reduction_pct
    else:
        # Quality loss = penalize efficiency
        efficiency = reduction_pct * (1 + quality_delta)

    return {
        "tokens_removed": tokens_removed,
        "reduction_percentage": reduction_pct,
        "quality_impact": quality_delta,
        "efficiency_score": efficiency,
        "recommendation": (
            "Effective pruning" if efficiency > 20 and quality_delta >= -0.05
            else "Tune pruning strategy"
        )
    }
```

**Saturation vs Performance Correlation:**

```python
import pandas as pd
from scipy.stats import pearsonr

def correlate_saturation_performance(
    saturation_data: pd.DataFrame,
    performance_data: pd.DataFrame
) -> dict:
    """
    Analyze correlation between context saturation and agent performance.

    Args:
        saturation_data: DataFrame with [timestamp, saturation_pct]
        performance_data: DataFrame with [timestamp, latency_ms, quality_score]

    Returns:
        Correlation analysis results
    """
    # Merge datasets on timestamp
    merged = pd.merge_asof(
        saturation_data.sort_values('timestamp'),
        performance_data.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )

    # Calculate correlations
    saturation_latency_corr, p_latency = pearsonr(
        merged['saturation_pct'],
        merged['latency_ms']
    )

    saturation_quality_corr, p_quality = pearsonr(
        merged['saturation_pct'],
        merged['quality_score']
    )

    return {
        "saturation_latency_correlation": saturation_latency_corr,
        "saturation_quality_correlation": saturation_quality_corr,
        "latency_significant": p_latency < 0.05,
        "quality_significant": p_quality < 0.05,
        "interpretation": {
            "latency": (
                "Higher saturation significantly increases latency"
                if saturation_latency_corr > 0.5 and p_latency < 0.05
                else "No significant saturation-latency correlation"
            ),
            "quality": (
                "Higher saturation significantly decreases quality"
                if saturation_quality_corr < -0.3 and p_quality < 0.05
                else "No significant saturation-quality correlation"
            )
        }
    }
```

---

### 4. ai.tool_error_rate

#### What It Measures

The percentage of tool executions that result in errors, distinguishing between infrastructure failures (500 errors, timeouts) and semantic failures (empty results, validation errors).

**Metric Type:** Gauge (calculated as ratio)

**Unit:** `percentage` (0-100%)

**Dimensions:**
- `tool_name`: Specific tool being executed
- `error_type`: `infrastructure` | `semantic` | `validation`
- `agent_name`: Agent calling the tool
- `severity`: `critical` | `error` | `warning`

#### Why It Matters

Agents are only as reliable as their tools. Tool errors cascade into agent failures:

- **Infrastructure Errors**: Backend systems are unstable or misconfigured
- **Semantic Errors**: Tool returns unexpected or invalid data
- **Validation Errors**: Agent calls tool with incorrect parameters
- **Silent Failures**: Tool succeeds but returns empty/useless results

**Key Insight:** High infrastructure error rates indicate system problems. High validation error rates indicate prompt engineering problems.

#### Implementation

```python
from enum import Enum

class ToolErrorType(Enum):
    INFRASTRUCTURE = "infrastructure"  # 500, timeout, network
    SEMANTIC = "semantic"              # Empty results, wrong format
    VALIDATION = "validation"          # Invalid parameters
    SUCCESS = "success"

def classify_tool_result(
    status_code: int,
    response_body: any,
    expected_schema: dict = None
) -> ToolErrorType:
    """Classify tool execution result."""

    # Infrastructure errors
    if status_code >= 500:
        return ToolErrorType.INFRASTRUCTURE
    if status_code == 408:  # Timeout
        return ToolErrorType.INFRASTRUCTURE

    # Validation errors (client-side)
    if status_code >= 400 and status_code < 500:
        return ToolErrorType.VALIDATION

    # Semantic errors (successful call, bad data)
    if status_code == 200:
        if response_body in [None, "", "null", []]:
            return ToolErrorType.SEMANTIC

        if expected_schema:
            try:
                validate_schema(response_body, expected_schema)
            except ValidationError:
                return ToolErrorType.SEMANTIC

        return ToolErrorType.SUCCESS

    return ToolErrorType.INFRASTRUCTURE

# Track tool executions
tool_execution_counter = meter.create_counter(
    name="ai.tool_executions",
    description="Tool execution attempts",
    unit="executions"
)

tool_error_counter = meter.create_counter(
    name="ai.tool_errors",
    description="Tool execution errors",
    unit="errors"
)

def track_tool_execution(
    tool_name: str,
    status_code: int,
    response_body: any,
    agent_name: str,
    latency_ms: float
):
    """Track tool execution and errors."""

    error_type = classify_tool_result(status_code, response_body)

    # Count total executions
    tool_execution_counter.add(
        1,
        attributes={
            "tool_name": tool_name,
            "agent_name": agent_name,
            "status": error_type.value
        }
    )

    # Count errors if applicable
    if error_type != ToolErrorType.SUCCESS:
        tool_error_counter.add(
            1,
            attributes={
                "tool_name": tool_name,
                "agent_name": agent_name,
                "error_type": error_type.value,
                "severity": (
                    "critical" if error_type == ToolErrorType.INFRASTRUCTURE
                    else "warning"
                )
            }
        )

    # Track latency
    tool_latency_histogram = meter.create_histogram(
        name="ai.tool_latency",
        description="Tool execution latency",
        unit="milliseconds"
    )

    tool_latency_histogram.record(
        latency_ms,
        attributes={
            "tool_name": tool_name,
            "status": error_type.value
        }
    )
```

#### Calculation

**Error Rate Formula:**

```
Tool Error Rate = (Failed Executions / Total Executions) × 100%

Where:
- Failed Executions = Infrastructure + Semantic + Validation errors
- Total Executions = All tool calls in time window
```

**Per-Tool Error Rate Query:**

```sql
SELECT
    tool_name,
    COUNT(*) as total_executions,
    SUM(CASE WHEN error_type != 'success' THEN 1 ELSE 0 END) as errors,
    (errors * 100.0 / total_executions) as error_rate_pct,
    SUM(CASE WHEN error_type = 'infrastructure' THEN 1 ELSE 0 END) as infra_errors,
    SUM(CASE WHEN error_type = 'semantic' THEN 1 ELSE 0 END) as semantic_errors,
    SUM(CASE WHEN error_type = 'validation' THEN 1 ELSE 0 END) as validation_errors
FROM tool_executions
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY tool_name
ORDER BY error_rate_pct DESC;
```

#### Visualization

**Tool Error Rate Dashboard:**

```
Tool Reliability (Last Hour)

Tool Name           Error Rate   Total Calls   Error Breakdown
──────────────────────────────────────────────────────────────────
database_query        2.3%         4,567       Infra: 45  Semantic: 60
web_search           12.5%         2,890       Infra: 120 Semantic: 241
file_read             0.8%         6,123       Infra: 12  Validation: 37
api_call             15.2%         1,234       Infra: 187 Semantic: 0
code_execution        5.4%           890       Infra: 23  Validation: 25

Overall Error Rate: 6.7%
```

**Error Type Distribution:**

```
Error Type Breakdown

Infrastructure (42%)  ████████████████████
Semantic (38%)        ██████████████████
Validation (20%)      ██████████

Total Errors: 481 (6.7% of 7,204 executions)
```

**Tool Dependency Graph with Error Rates:**

```
Agent Workflow: Data Analysis Pipeline

┌──────────────┐
│ fetch_data   │  2.1% error rate
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ validate_csv │  0.3% error rate
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ analyze_data │  8.7% error rate  ← HIGH ERROR RATE
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ generate_viz │  1.2% error rate
└──────────────┘
```

#### Alert Conditions

```yaml
alerts:
  # High overall error rate
  - name: "Tool Error Rate High"
    condition: |
      (sum(ai.tool_errors) / sum(ai.tool_executions)) * 100 > 20
    severity: "critical"
    description: "Tool error rate exceeds 20%"

  # Infrastructure errors
  - name: "Infrastructure Errors Detected"
    condition: |
      sum(ai.tool_errors{error_type='infrastructure'}) > 10
    severity: "critical"
    description: "Multiple infrastructure errors indicate system problem"
    actions:
      - "page: on-call-sre"
      - "create: incident-ticket"

  # Specific tool failing
  - name: "Tool Consistently Failing"
    condition: |
      (sum(ai.tool_errors{tool_name='$tool'}) /
       sum(ai.tool_executions{tool_name='$tool'})) > 50
    severity: "critical"
    description: "Tool failing >50% of executions"
    actions:
      - "disable: tool_temporarily"
      - "alert: tool-owner"

  # Validation errors (prompt engineering issue)
  - name: "High Validation Errors"
    condition: |
      sum(ai.tool_errors{error_type='validation'}) > 50
    severity: "warning"
    description: "Agent calling tools with invalid parameters"
    actions:
      - "review: agent_prompts"
      - "check: tool_schema_documentation"
```

#### Advanced Analytics

**Tool Reliability Score:**

```python
def calculate_tool_reliability_score(
    tool_stats: dict,
    time_window: str = "24h"
) -> dict:
    """
    Calculate comprehensive reliability score for tool.

    Score components:
    - Success rate (50%)
    - Latency percentiles (25%)
    - Consistency (low variance) (25%)
    """

    success_rate = (
        tool_stats['successful_calls'] / tool_stats['total_calls']
    ) * 100

    # Latency score (inverse of P95 latency)
    latency_score = max(0, 100 - (tool_stats['p95_latency_ms'] / 10))

    # Consistency score (inverse of coefficient of variation)
    if tool_stats['mean_latency'] > 0:
        cv = tool_stats['stddev_latency'] / tool_stats['mean_latency']
        consistency_score = max(0, 100 - (cv * 100))
    else:
        consistency_score = 100

    # Weighted composite score
    reliability_score = (
        success_rate * 0.5 +
        latency_score * 0.25 +
        consistency_score * 0.25
    )

    # Grade
    if reliability_score >= 95:
        grade = "A"
    elif reliability_score >= 85:
        grade = "B"
    elif reliability_score >= 70:
        grade = "C"
    elif reliability_score >= 50:
        grade = "D"
    else:
        grade = "F"

    return {
        "reliability_score": reliability_score,
        "grade": grade,
        "success_rate": success_rate,
        "latency_score": latency_score,
        "consistency_score": consistency_score,
        "recommendation": (
            "Production ready" if grade in ["A", "B"]
            else "Requires optimization" if grade == "C"
            else "Not production ready"
        )
    }
```

**Error Pattern Detection:**

```python
from collections import Counter

def detect_error_patterns(
    error_logs: List[dict],
    min_pattern_occurrences: int = 3
) -> List[dict]:
    """Detect recurring error patterns."""

    patterns = []

    # Group errors by type and tool
    error_groups = {}
    for error in error_logs:
        key = (error['tool_name'], error['error_type'], error['status_code'])
        if key not in error_groups:
            error_groups[key] = []
        error_groups[key].append(error)

    # Analyze each group for patterns
    for key, errors in error_groups.items():
        if len(errors) < min_pattern_occurrences:
            continue

        tool_name, error_type, status_code = key

        # Extract error messages
        error_messages = [e.get('error_message', '') for e in errors]
        message_counts = Counter(error_messages)

        # Find most common error message
        most_common_msg, count = message_counts.most_common(1)[0]

        if count >= min_pattern_occurrences:
            patterns.append({
                "tool_name": tool_name,
                "error_type": error_type,
                "status_code": status_code,
                "occurrences": count,
                "error_message": most_common_msg,
                "first_seen": min(e['timestamp'] for e in errors),
                "last_seen": max(e['timestamp'] for e in errors),
                "affected_agents": list(set(e['agent_name'] for e in errors)),
                "severity": "high" if count > 10 else "medium"
            })

    return sorted(patterns, key=lambda x: x['occurrences'], reverse=True)
```

---

### 5. ai.loop_velocity_gauge

#### What It Measures

The rate of agent reasoning iterations, measured as the time delta between consecutive Think→Act→Observe cycles. Low velocity (rapid iterations) can indicate runaway loops or inefficient reasoning.

**Metric Type:** Gauge (time between iterations)

**Unit:** `milliseconds`

**Dimensions:**
- `agent_name`: Agent being monitored
- `session_id`: Conversation or workflow session
- `loop_type`: `normal` | `rapid` | `stuck`
- `iteration_number`: Current iteration count

#### Why It Matters

Infinite reasoning loops are a unique failure mode of autonomous agents:

- **Financial Risk**: Each iteration costs money (tokens) without progress
- **Performance Degradation**: Rapid loops waste compute and delay results
- **Silent Failure**: Agent appears to be working but makes no progress
- **Cognitive Thrashing**: Repeating the same actions without learning

**Critical Detection Threshold:** Iterations faster than 200ms typically indicate runaway behavior rather than productive reasoning.

#### Implementation

```python
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Iteration:
    """Single agent iteration record."""
    timestamp: float
    iteration_number: int
    thought: str
    action: str
    observation: str
    tokens_used: int

class LoopVelocityTracker:
    """Track agent loop velocity and detect runaway patterns."""

    def __init__(self, window_size: int = 10):
        self.iterations: deque[Iteration] = deque(maxlen=window_size)
        self.velocities: deque[float] = deque(maxlen=window_size - 1)

    def record_iteration(
        self,
        thought: str,
        action: str,
        observation: str,
        tokens_used: int
    ) -> dict:
        """Record new iteration and calculate velocity metrics."""

        current_time = time.time()
        iteration_number = len(self.iterations) + 1

        iteration = Iteration(
            timestamp=current_time,
            iteration_number=iteration_number,
            thought=thought,
            action=action,
            observation=observation,
            tokens_used=tokens_used
        )

        # Calculate velocity if we have previous iteration
        if self.iterations:
            prev_iteration = self.iterations[-1]
            velocity_ms = (current_time - prev_iteration.timestamp) * 1000
            self.velocities.append(velocity_ms)

        self.iterations.append(iteration)

        # Analyze loop patterns
        analysis = self._analyze_loop_patterns()

        return analysis

    def _analyze_loop_patterns(self) -> dict:
        """Analyze iteration patterns for loop detection."""

        if len(self.velocities) == 0:
            return {"status": "initializing"}

        avg_velocity = sum(self.velocities) / len(self.velocities)
        min_velocity = min(self.velocities)
        max_velocity = max(self.velocities)

        # Detect rapid iterations
        rapid_iterations = sum(1 for v in self.velocities if v < 200)

        # Detect tool repetition
        tool_sequence = [it.action for it in self.iterations]
        has_tool_loop = self._detect_tool_loop(tool_sequence)

        # Detect semantic similarity (soft loops)
        if len(self.iterations) >= 3:
            has_semantic_loop = self._detect_semantic_loop([
                it.thought for it in self.iterations
            ])
        else:
            has_semantic_loop = False

        # Determine loop type
        if rapid_iterations >= 3 or has_tool_loop or has_semantic_loop:
            loop_type = "stuck"
        elif avg_velocity < 500:
            loop_type = "rapid"
        else:
            loop_type = "normal"

        return {
            "status": "running",
            "avg_velocity_ms": avg_velocity,
            "min_velocity_ms": min_velocity,
            "max_velocity_ms": max_velocity,
            "rapid_iterations": rapid_iterations,
            "loop_type": loop_type,
            "has_tool_loop": has_tool_loop,
            "has_semantic_loop": has_semantic_loop,
            "iteration_count": len(self.iterations),
            "total_tokens": sum(it.tokens_used for it in self.iterations),
            "alert": loop_type == "stuck"
        }

    def _detect_tool_loop(self, tool_sequence: List[str]) -> bool:
        """Detect if agent is calling same tools repeatedly."""

        if len(tool_sequence) < 4:
            return False

        # Check for pattern: A → B → A → B
        for i in range(len(tool_sequence) - 3):
            if (tool_sequence[i] == tool_sequence[i + 2] and
                tool_sequence[i + 1] == tool_sequence[i + 3]):
                return True

        # Check for same tool called >3 times consecutively
        for i in range(len(tool_sequence) - 2):
            if (tool_sequence[i] == tool_sequence[i + 1] ==
                tool_sequence[i + 2]):
                return True

        return False

    def _detect_semantic_loop(
        self,
        thoughts: List[str],
        similarity_threshold: float = 0.95
    ) -> bool:
        """Detect semantic similarity between consecutive thoughts."""

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if len(thoughts) < 2:
            return False

        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(thoughts)
        except ValueError:
            return False

        # Check consecutive pairs
        for i in range(len(thoughts) - 1):
            similarity = cosine_similarity(
                tfidf_matrix[i:i+1],
                tfidf_matrix[i+1:i+2]
            )[0][0]

            if similarity > similarity_threshold:
                return True

        return False

# Track loop velocity metric
loop_velocity_gauge = meter.create_gauge(
    name="ai.loop_velocity",
    description="Time between agent iterations in milliseconds",
    unit="milliseconds"
)

# Track iteration count
iteration_counter = meter.create_counter(
    name="ai.iterations",
    description="Agent iteration count",
    unit="iterations"
)

def track_agent_iteration(
    agent_name: str,
    session_id: str,
    thought: str,
    action: str,
    observation: str,
    tokens_used: int,
    tracker: LoopVelocityTracker
):
    """Track agent iteration and loop metrics."""

    analysis = tracker.record_iteration(thought, action, observation, tokens_used)

    if analysis["status"] == "running":
        # Record velocity
        loop_velocity_gauge.set(
            analysis["avg_velocity_ms"],
            attributes={
                "agent_name": agent_name,
                "session_id": session_id,
                "loop_type": analysis["loop_type"]
            }
        )

        # Count iteration
        iteration_counter.add(
            1,
            attributes={
                "agent_name": agent_name,
                "session_id": session_id,
                "loop_type": analysis["loop_type"]
            }
        )

        # Alert if stuck
        if analysis["alert"]:
            logger.warning(
                f"Agent {agent_name} in runaway loop",
                extra={
                    "session_id": session_id,
                    "iteration_count": analysis["iteration_count"],
                    "total_tokens": analysis["total_tokens"],
                    "has_tool_loop": analysis["has_tool_loop"],
                    "has_semantic_loop": analysis["has_semantic_loop"]
                }
            )
```

#### Visualization

**Loop Velocity Timeline:**

```
Agent Iteration Velocity

Velocity (ms)
│
2000 ┤ ●                                        HEALTHY
     │   ●     ●                               (>1000ms)
1500 ┤      ●     ●        ●
     │               ●  ●     ●
1000 ┤                          ●───────────── THRESHOLD
     │                             ●
 500 ┤                                ●        RAPID
     │                                   ●     (<500ms)
 200 ┤                                      ●── RUNAWAY
     │                                         (<200ms)
   0 └────┴────┴────┴────┴────┴────┴────┴────
        1    2    3    4    5    6    7    8
                  (iteration number)
```

**Loop Detection Dashboard:**

```
Active Agent Sessions - Loop Status

Agent Name        Status     Velocity    Iterations   Tokens   Alert
───────────────────────────────────────────────────────────────────
research-agent    NORMAL     1,234ms          5        12.4K    -
code-gen          RAPID        456ms         12        45.2K    ⚠
data-analyzer     STUCK        123ms         28       156.7K    🚨
summarizer        NORMAL     2,145ms          3         8.1K    -
doc-qa            NORMAL       892ms          7        18.9K    -
```

**Tool Loop Pattern Visualization:**

```
Tool Call Sequence - Runaway Loop Detected

Iteration  Tool Called         Pattern
──────────────────────────────────────────
1          web_search          Initial
2          extract_data        Normal
3          web_search          ← Repeat
4          extract_data        ← Repeat
5          web_search          ← LOOP DETECTED
6          extract_data        ← LOOP DETECTED
7          web_search          ← LOOP CONTINUES
8          extract_data        ← LOOP CONTINUES

Pattern: web_search → extract_data → [REPEAT 4x]
Cost: $2.34 with no progress
```

#### Alert Conditions

```yaml
alerts:
  # Rapid iterations
  - name: "Rapid Agent Iterations"
    condition: "ai.loop_velocity < 200"
    for: "5 iterations"
    severity: "warning"
    description: "Agent iterating faster than 200ms"

  # Runaway loop detected
  - name: "Agent Runaway Loop"
    condition: |
      ai.loop_velocity < 200 AND
      ai.iterations{loop_type='stuck'} > 5
    severity: "critical"
    description: "Agent in confirmed runaway loop"
    actions:
      - "terminate: agent_session"
      - "alert: engineering-team"
      - "save: debug_trace"

  # Tool repetition
  - name: "Tool Repetition Detected"
    condition: "detect_tool_loop(tool_sequence) == true"
    severity: "critical"
    description: "Agent calling same tools repeatedly"

  # High iteration count
  - name: "Excessive Iterations"
    condition: "ai.iterations{session_id='$session'} > 50"
    severity: "warning"
    description: "Session exceeded 50 iterations"
    actions:
      - "review: agent_reasoning_trace"
      - "check: task_complexity"

  # Cost runaway
  - name: "Runaway Cost from Loops"
    condition: |
      ai.iterations{loop_type='stuck'} > 10 AND
      ai.cost_estimate{session_id='$session'} > 5.0
    severity: "critical"
    description: "Runaway loop cost exceeds $5"
    actions:
      - "terminate: immediately"
      - "refund: user_if_applicable"
```

#### Advanced Analytics

**Loop Efficiency Analysis:**

```python
def analyze_loop_efficiency(iterations: List[Iteration]) -> dict:
    """Determine if iterations are productive or wasteful."""

    # Calculate progress metrics
    unique_tools = len(set(it.action for it in iterations))
    unique_thoughts = len(set(it.thought for it in iterations))
    total_iterations = len(iterations)

    # Tool diversity (higher = more exploration)
    tool_diversity = unique_tools / total_iterations if total_iterations > 0 else 0

    # Thought diversity (higher = more creative thinking)
    thought_diversity = unique_thoughts / total_iterations if total_iterations > 0 else 0

    # Calculate token efficiency
    total_tokens = sum(it.tokens_used for it in iterations)
    tokens_per_iteration = total_tokens / total_iterations if total_iterations > 0 else 0

    # Velocity trend (are we slowing down or speeding up?)
    if len(iterations) >= 3:
        early_velocity = iterations[1].timestamp - iterations[0].timestamp
        late_velocity = iterations[-1].timestamp - iterations[-2].timestamp
        velocity_trend = "increasing" if late_velocity > early_velocity else "decreasing"
    else:
        velocity_trend = "unknown"

    # Efficiency score
    efficiency = (tool_diversity + thought_diversity) / 2

    if efficiency > 0.7:
        assessment = "Highly productive - good exploration"
    elif efficiency > 0.4:
        assessment = "Moderately productive"
    elif efficiency > 0.2:
        assessment = "Low productivity - some repetition"
    else:
        assessment = "Unproductive - likely stuck in loop"

    return {
        "total_iterations": total_iterations,
        "tool_diversity": tool_diversity,
        "thought_diversity": thought_diversity,
        "efficiency_score": efficiency,
        "assessment": assessment,
        "total_tokens": total_tokens,
        "tokens_per_iteration": tokens_per_iteration,
        "velocity_trend": velocity_trend
    }
```

**Runaway Prevention System:**

```python
class RunawayPreventionSystem:
    """Proactive system to prevent agent runaway loops."""

    def __init__(
        self,
        max_iterations: int = 30,
        max_cost: float = 10.0,
        max_time_seconds: int = 300
    ):
        self.max_iterations = max_iterations
        self.max_cost = max_cost
        self.max_time_seconds = max_time_seconds
        self.session_trackers = {}

    def check_session(
        self,
        session_id: str,
        current_iteration: int,
        current_cost: float,
        elapsed_time: float,
        velocity_tracker: LoopVelocityTracker
    ) -> dict:
        """
        Check if session should be terminated to prevent runaway.

        Returns decision with reasoning.
        """
        violations = []

        # Check iteration count
        if current_iteration > self.max_iterations:
            violations.append({
                "type": "max_iterations",
                "threshold": self.max_iterations,
                "current": current_iteration,
                "severity": "high"
            })

        # Check cost
        if current_cost > self.max_cost:
            violations.append({
                "type": "max_cost",
                "threshold": self.max_cost,
                "current": current_cost,
                "severity": "critical"
            })

        # Check time
        if elapsed_time > self.max_time_seconds:
            violations.append({
                "type": "max_time",
                "threshold": self.max_time_seconds,
                "current": elapsed_time,
                "severity": "medium"
            })

        # Check loop patterns
        analysis = velocity_tracker._analyze_loop_patterns()
        if analysis.get("loop_type") == "stuck":
            violations.append({
                "type": "runaway_loop",
                "details": analysis,
                "severity": "critical"
            })

        # Make termination decision
        critical_violations = [v for v in violations if v["severity"] == "critical"]

        if critical_violations:
            decision = "terminate"
            reason = f"Critical violations: {', '.join(v['type'] for v in critical_violations)}"
        elif len(violations) >= 2:
            decision = "warn_and_monitor"
            reason = f"Multiple violations: {', '.join(v['type'] for v in violations)}"
        elif violations:
            decision = "monitor"
            reason = f"Minor violation: {violations[0]['type']}"
        else:
            decision = "continue"
            reason = "All checks passed"

        return {
            "decision": decision,
            "reason": reason,
            "violations": violations,
            "current_metrics": {
                "iterations": current_iteration,
                "cost": current_cost,
                "elapsed_time": elapsed_time,
                "loop_type": analysis.get("loop_type", "unknown")
            }
        }
```

---

## Additional Metrics

Beyond the five core metrics, a11i tracks supplementary metrics that provide deeper insights into agent performance and behavior.

### Latency Metrics

#### Time to First Token (TTFT)

**What:** Latency from request submission to first token streamed back

**Why:** Critical for perceived responsiveness in user-facing agents

**Implementation:**

```python
ttft_histogram = meter.create_histogram(
    name="ai.latency.ttft",
    description="Time to first token",
    unit="milliseconds"
)

start_time = time.time()
response = await llm.stream(prompt)
first_token_time = time.time()

ttft_ms = (first_token_time - start_time) * 1000

ttft_histogram.record(
    ttft_ms,
    attributes={
        "model": model,
        "provider": provider,
        "agent_name": agent_name
    }
)
```

#### Inter-Token Latency (ITL)

**What:** Average time between consecutive tokens in streaming responses

**Why:** Indicates stream smoothness and user experience quality

#### Time Per Output Token (TPOT)

**What:** Average generation time per output token

**Why:** Efficiency metric for comparing models and configurations

#### End-to-End Latency

**What:** Total time from agent receiving task to delivering result

**Why:** Overall agent performance from user perspective

### Quality Metrics

#### Hallucination Detection

**What:** Automated detection of factually incorrect or unsupported claims

**Implementation:**

```python
# Integration with evaluation frameworks
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

def track_hallucination_metrics(
    question: str,
    answer: str,
    context: List[str]
) -> dict:
    """Detect potential hallucinations."""

    result = evaluate({
        "question": question,
        "answer": answer,
        "context": context
    }, metrics=[faithfulness, answer_relevancy])

    # Record metrics
    hallucination_gauge = meter.create_gauge(
        name="ai.quality.faithfulness",
        description="Faithfulness score (0-1)",
        unit="score"
    )

    hallucination_gauge.set(
        result["faithfulness"],
        attributes={
            "agent_name": agent_name,
            "task_type": task_type
        }
    )

    return result
```

#### User Feedback Scores

**What:** Explicit user ratings (thumbs up/down, 1-5 stars)

**Why:** Ground truth for quality monitoring

#### Groundedness Score

**What:** For RAG systems, measures how well answer is supported by retrieved context

**Why:** Prevents hallucination in knowledge-based agents

### System Health Metrics

#### Cache Hit Rate

**What:** Percentage of requests served from cache vs fresh LLM calls

**Why:** Cost optimization and latency improvement

```python
cache_hit_counter = meter.create_counter(
    name="ai.cache.hits",
    description="Cache hit count",
    unit="hits"
)

cache_miss_counter = meter.create_counter(
    name="ai.cache.misses",
    description="Cache miss count",
    unit="misses"
)

# Calculate hit rate
cache_hit_rate = hits / (hits + misses) * 100
```

#### Provider Availability

**What:** Uptime percentage for each LLM provider

**Why:** Reliability monitoring and failover planning

#### Rate Limit Events

**What:** Count of rate limit errors (429 responses)

**Why:** Capacity planning and request throttling

---

## Metrics Dashboard

### Comprehensive Agent Observability Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│                    a11i Agent Observability                         │
│                         Last 24 Hours                               │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────┬──────────────────────┬──────────────────────┐
│   Token Usage        │   Cost Tracking      │  Context Health      │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ Total: 15.2M tokens  │ Total: $456.78       │ Avg Saturation: 42%  │
│ Input: 9.8M (64%)    │ GPT-4: $234.56 (51%) │ At Risk: 3 agents    │
│ Output: 5.4M (36%)   │ Claude: $189.23(41%) │ Critical: 1 agent    │
│                      │ GPT-3.5: $32.99 (8%) │                      │
│ ████████████████     │ Budget: $500/day     │ Risk Distribution:   │
│ Trend: ↑ 12% vs prev │ Remaining: $43.22    │ 🟢 12  🟡 4  🔴 1   │
└──────────────────────┴──────────────────────┴──────────────────────┘

┌──────────────────────┬──────────────────────────────────────────────┐
│  Tool Reliability    │  Loop Velocity                               │
├──────────────────────┼──────────────────────────────────────────────┤
│ Overall: 93.2%       │ Active Sessions: 23                          │
│ Success: 14,567      │                                              │
│ Errors: 1,067 (6.8%) │ Loop Status:                                 │
│                      │ • Normal: 19 sessions                        │
│ Error Breakdown:     │ • Rapid: 3 sessions  ⚠                      │
│ • Infrastructure: 42%│ • Stuck: 1 session   🚨                     │
│ • Semantic: 38%      │                                              │
│ • Validation: 20%    │ Avg Velocity: 1,234ms                        │
│                      │ Runaway Prevented: 2 today                   │
└──────────────────────┴──────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      Agent Performance Heatmap                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Agent              Token  Cost    Context  Tool    Loop    Overall  │
│                    Usage          Sat.     Errors  Velocity         │
│ ─────────────────────────────────────────────────────────────────── │
│ research-agent     🟢     🟢      🟡       🟢      🟢       🟢       │
│ code-generator     🟡     🟡      🟢       🟢      🟡       🟡       │
│ data-analyzer      🟢     🟢      🟢       🔴      🟢       🟡       │
│ summarizer         🟢     🟢      🟢       🟢      🟢       🟢       │
│ doc-qa             🔴     🔴      🔴       🟡      🔴       🔴       │
│                                                                      │
│ Legend: 🟢 Healthy  🟡 Warning  🔴 Critical                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         Recent Alerts                                │
├─────────────────────────────────────────────────────────────────────┤
│ 🚨 CRITICAL  11:23  Agent "doc-qa" runaway loop - TERMINATED        │
│ ⚠  WARNING   10:45  Context saturation 87% for "research-agent"     │
│ ⚠  WARNING   09:12  Daily budget 85% consumed                       │
│ ℹ️  INFO      08:30  High tool error rate for "api_call" (15%)      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Guidelines

### Getting Started

**Step 1: Instrument LLM Calls**

Start with basic token and cost tracking:

```python
from a11i import init_observability, track_llm_call

# Initialize a11i SDK
init_observability(
    service_name="my-agent",
    exporter_endpoint="http://localhost:4317"
)

# Wrap your LLM calls
response = openai.chat.completions.create(...)
track_llm_call(response, agent_name="my-agent")
```

**Step 2: Add Context Tracking**

Monitor context window utilization:

```python
from a11i import track_context_saturation

messages = build_conversation_history()
prompt_tokens = count_tokens(messages)

track_context_saturation(
    model="gpt-4-turbo",
    prompt_tokens=prompt_tokens,
    agent_name="my-agent"
)
```

**Step 3: Instrument Tool Calls**

Track tool execution and errors:

```python
from a11i import track_tool_execution

result = await execute_tool(tool_name, params)

track_tool_execution(
    tool_name=tool_name,
    status_code=result.status_code,
    response_body=result.data,
    agent_name="my-agent"
)
```

**Step 4: Monitor Loop Velocity**

Detect runaway iterations:

```python
from a11i import LoopVelocityTracker

tracker = LoopVelocityTracker()

for iteration in agent_loop():
    analysis = tracker.record_iteration(
        thought=iteration.thought,
        action=iteration.action,
        observation=iteration.observation,
        tokens_used=iteration.tokens
    )

    if analysis.get("alert"):
        break  # Terminate runaway loop
```

### Best Practices

1. **Always use provider-returned token counts** when available for cost accuracy
2. **Set up alerts on all five core metrics** before going to production
3. **Track attribution dimensions** (tenant, team, workflow) from day one
4. **Monitor metric correlations** (e.g., saturation vs quality) to understand causation
5. **Review cost optimization recommendations** weekly to improve efficiency
6. **Test runaway prevention** in staging before production deployment

---

## Key Takeaways

> **The Five Core Metrics in Practice**
>
> 1. **Token Usage** provides the foundation for all other metrics. Accurate token counting is non-negotiable.
>
> 2. **Cost Estimation** transforms token usage into business metrics. Multi-tenant attribution enables chargeback models and ROI analysis.
>
> 3. **Context Saturation** is the earliest warning signal for agent cognitive failures. Monitor actively and optimize aggressively.
>
> 4. **Tool Error Rates** distinguish infrastructure problems from prompt engineering problems. Different error types require different solutions.
>
> 5. **Loop Velocity** catches the silent failures that traditional monitoring misses. Proactive termination prevents runaway costs.
>
> **Together, these five metrics provide comprehensive observability into AI agent systems, enabling confident production deployments with predictable costs and reliable performance.**

**Implementation Priority:**

- **Week 1**: Token usage + Cost estimation (financial visibility)
- **Week 2**: Context saturation (quality/performance)
- **Week 3**: Tool error rate (reliability)
- **Week 4**: Loop velocity (runaway prevention)

**Success Metrics:**

- 100% LLM call coverage for token/cost tracking
- <5% error rate on metrics collection
- <10ms latency overhead from instrumentation
- Zero runaway loops in production (prevented by automation)

---

**Related Documentation:**

- [System Architecture](/home/becker/projects/a11i/docs/02-architecture/system-architecture.md) - Platform architecture
- [Instrumentation SDK](/home/becker/projects/a11i/docs/03-core-platform/instrumentation-sdk.md) - SDK implementation
- [Alerting System](/home/becker/projects/a11i/docs/03-core-platform/alerting-system.md) - Alert configuration
- [Cost Optimization](/home/becker/projects/a11i/docs/04-use-cases/cost-optimization.md) - Cost reduction strategies

---

*Document Status: Draft | Last Updated: 2025-11-26 | Maintained by: a11i Documentation Team*
