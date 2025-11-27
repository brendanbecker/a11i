---
title: "Monitoring & Alerting: Production AI Agent Operations"
category: "Operations"
tags: ["monitoring", "alerting", "dashboards", "prometheus", "grafana", "anomaly-detection", "runbooks", "sre"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "../03-core-platform/core-metrics.md"
  - "../02-architecture/system-architecture.md"
  - "../02-architecture/data-pipeline.md"
  - "./deployment-operations.md"
  - "./incident-response.md"
---

# Monitoring & Alerting: Production AI Agent Operations

## Table of Contents

- [Introduction](#introduction)
- [Monitoring Philosophy](#monitoring-philosophy)
- [Key Dashboards](#key-dashboards)
  - [Usage & Performance Dashboard](#usage--performance-dashboard)
  - [Agent Health Dashboard](#agent-health-dashboard)
  - [Cost & ROI Dashboard](#cost--roi-dashboard)
  - [Latency & Throughput Dashboard](#latency--throughput-dashboard)
- [Alert Conditions](#alert-conditions)
  - [Prometheus Rules Configuration](#prometheus-rules-configuration)
  - [Alert Severity Levels](#alert-severity-levels)
  - [Alert Routing](#alert-routing)
- [Anomaly Detection](#anomaly-detection)
  - [Statistical Anomaly Detection](#statistical-anomaly-detection)
  - [ML-Based Anomaly Detection](#ml-based-anomaly-detection)
  - [Pattern Recognition](#pattern-recognition)
- [Runbook Examples](#runbook-examples)
  - [Stuck Agent Loop](#stuck-agent-loop)
  - [Cost Anomaly](#cost-anomaly)
  - [High Tool Failure Rate](#high-tool-failure-rate)
  - [Context Saturation Crisis](#context-saturation-crisis)
- [Grafana Dashboard Configuration](#grafana-dashboard-configuration)
- [Integration with Incident Response](#integration-with-incident-response)
- [Key Takeaways](#key-takeaways)

## Introduction

Production AI agents require specialized monitoring and alerting strategies that go beyond traditional infrastructure monitoring. While CPU, memory, and request rates remain important, **AI agents fail in fundamentally different ways**:

- **Silent Divergence**: Agents complete tasks successfully but produce incorrect or harmful outputs
- **Cognitive Degradation**: Performance degrades as context windows saturate without explicit errors
- **Runaway Loops**: Agents consume unbounded resources in circular reasoning patterns
- **Cost Spirals**: Inefficient agent behavior drives exponential LLM API costs

**a11i's monitoring and alerting system** provides comprehensive observability into these unique failure modes, enabling SRE teams to:

1. **Detect failures early** before they impact users or budgets
2. **Diagnose root causes** through correlated metrics and traces
3. **Prevent incidents** with proactive anomaly detection
4. **Optimize performance** based on data-driven insights
5. **Control costs** through budget enforcement and optimization recommendations

This document covers the complete monitoring and alerting architecture, from dashboard design to runbook procedures, enabling confident production operations of autonomous AI agents.

## Monitoring Philosophy

### Layered Monitoring Strategy

a11i implements a **three-layer monitoring strategy** that progressively increases in sophistication:

```
┌─────────────────────────────────────────────────────────┐
│              Layer 3: Predictive & Proactive            │
│  • Anomaly detection (statistical + ML)                 │
│  • Cost forecasting and budget predictions              │
│  • Performance trend analysis                           │
│  • Capacity planning recommendations                    │
└─────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────┐
│              Layer 2: Agent-Specific Metrics            │
│  • Context saturation monitoring                        │
│  • Tool error rate tracking                             │
│  • Loop velocity detection                              │
│  • Agent cognition quality metrics                      │
└─────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────┐
│         Layer 1: Infrastructure & LLM Metrics           │
│  • Request rate, latency, error rate (RED)              │
│  • Token usage and cost tracking                        │
│  • Provider availability and rate limits                │
│  • System resource utilization                          │
└─────────────────────────────────────────────────────────┘
```

**Layer 1** provides foundation infrastructure monitoring compatible with traditional APM tools. **Layer 2** adds agent-native observability using the five core metrics. **Layer 3** implements predictive analytics to prevent incidents before they occur.

### Monitoring Principles

1. **Metric-Driven, Not Log-Driven**: Structured metrics enable aggregation and alerting; logs provide debugging context
2. **Early Warning Signals**: Monitor leading indicators (context saturation) not just lagging indicators (errors)
3. **Cost as First-Class Metric**: Financial observability is not optional for production AI agents
4. **Proactive, Not Reactive**: Anomaly detection and predictive alerts prevent incidents
5. **Actionable Alerts**: Every alert must have a clear runbook and remediation path
6. **Multi-Tenancy Aware**: All metrics support tenant/team/workflow attribution

## Key Dashboards

### Usage & Performance Dashboard

The **Usage & Performance Dashboard** provides high-level visibility into overall platform activity and health. This is typically the primary dashboard for engineering leadership and product teams.

#### Dashboard Components

**1. Request Volume Metrics**

```
┌─────────────────────────────────────────────────────────┐
│               Requests per Hour/Day                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Requests/hr                                             │
│ 2500 ┤                                      ╭●          │
│      │                                 ╭────╯           │
│ 2000 ┤                            ╭────╯                │
│      │                       ╭────╯                     │
│ 1500 ┤                  ╭────╯                          │
│      │             ╭────╯                               │
│ 1000 ┤        ╭────╯                                    │
│      │   ╭────╯                                         │
│  500 ┤───╯                                              │
│      │                                                  │
│    0 └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──              │
│        00 02 04 06 08 10 12 14 16 18 20 22              │
│                                                         │
│ Total Today: 34,567 requests (+12% vs yesterday)       │
└─────────────────────────────────────────────────────────┘
```

**2. Token Usage Trends**

```
┌─────────────────────────────────────────────────────────┐
│              Token Consumption by Type                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Input Tokens    ████████████████████████ 15.2M (65%)   │
│ Output Tokens   ████████████████ 8.4M (35%)            │
│                                                         │
│ Total: 23.6M tokens                                     │
│ Peak Hour: 14:00-15:00 (2.8M tokens)                    │
│ Trend: ↑ 8% vs last 24h                                │
└─────────────────────────────────────────────────────────┘
```

**3. Average Latency Metrics**

Tracks three critical latency metrics:

- **TTFT (Time to First Token)**: Perceived responsiveness
- **E2E (End-to-End)**: Total request duration
- **TPOT (Time Per Output Token)**: Generation efficiency

```
┌─────────────────────────────────────────────────────────┐
│                  Latency Percentiles                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Metric      P50      P90      P95      P99             │
│ ──────────────────────────────────────────────────────  │
│ TTFT        234ms    456ms    678ms    1,234ms         │
│ E2E         2.3s     5.6s     8.9s     15.2s           │
│ TPOT        45ms     89ms     123ms    234ms           │
│                                                         │
│ SLO Target: P95 TTFT < 1s ✓                            │
│ SLO Target: P99 E2E < 30s ✓                            │
└─────────────────────────────────────────────────────────┘
```

**4. Error Rate**

```
┌─────────────────────────────────────────────────────────┐
│               Error Rate (Last 24h)                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Error % │                                               │
│   5%  ┤                                                 │
│       │                                                 │
│   4%  ┤     ●                                           │
│       │   ●   ●                                         │
│   3%  ┤ ●       ●                                       │
│       │           ●   ●                                 │
│   2%  ┤                 ●   ●   ●   ●   ●   ●   ●   ● ←│
│       │                                                 │
│   1%  ┤                                                 │
│       │                                                 │
│   0%  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─    │
│        00    04    08    12    16    20    24           │
│                                                         │
│ Current: 1.8% | Target: <2% ✓ | 24h Avg: 2.1%          │
└─────────────────────────────────────────────────────────┘
```

**5. Top Models by Usage**

```
┌─────────────────────────────────────────────────────────┐
│            Model Usage Distribution                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Model                 Requests    Tokens      Cost      │
│ ───────────────────────────────────────────────────────│
│ gpt-4-turbo          14,567      8.9M        $234.56   │
│ claude-3-5-sonnet    12,345      7.2M        $189.23   │
│ gpt-3.5-turbo         5,678      4.1M         $32.99   │
│ gpt-4o                1,234      2.8M         $45.67   │
│ claude-3-haiku          789      0.6M          $8.34   │
│                                                         │
│ Most Expensive: gpt-4-turbo (46% of spend)              │
│ Most Efficient: claude-3-haiku ($0.011/req)            │
└─────────────────────────────────────────────────────────┘
```

#### Prometheus Queries

```promql
# Request rate (requests per second)
sum(rate(ai_requests_total[5m]))

# Token usage rate (tokens per hour)
sum(increase(ai_token_usage_counter[1h]))

# Average latency P95
histogram_quantile(0.95,
  sum(rate(ai_request_duration_seconds_bucket[5m])) by (le)
)

# Error rate percentage
sum(rate(ai_errors_total[5m])) /
sum(rate(ai_requests_total[5m])) * 100

# Top models by request count
topk(5, sum by (model) (increase(ai_requests_total{model!=""}[24h])))
```

---

### Agent Health Dashboard

The **Agent Health Dashboard** provides deep visibility into agent-specific metrics, focusing on the unique operational characteristics of autonomous AI agents.

#### Dashboard Components

**1. Active Agents Status**

```
┌─────────────────────────────────────────────────────────┐
│              Active Agent Sessions                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Agent Name         Status    Sessions   Success Rate   │
│ ───────────────────────────────────────────────────────│
│ research-agent     🟢 UP          23         97.8%     │
│ code-generator     🟢 UP          18         94.2%     │
│ data-analyzer      🟡 DEGRADED     7         88.5%     │
│ summarizer         🟢 UP          45         99.1%     │
│ doc-qa             🔴 DOWN         0          0.0%     │
│                                                         │
│ Total: 93 active sessions across 5 agent types          │
└─────────────────────────────────────────────────────────┘
```

**2. Success vs Failure Rate**

Tracks both explicit failures (errors) and semantic failures (detected via quality metrics or user feedback).

```
┌─────────────────────────────────────────────────────────┐
│          Agent Success Rate (Last 7 Days)               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Success │                                               │
│  100% ┤ ●───●───●───●───●───●───●                      │
│       │                                                 │
│   95% ┤                                                 │
│       │                                                 │
│   90% ┤                                                 │
│       │                                                 │
│   85% ┤                                                 │
│       │                                                 │
│   80% └─┴───┴───┴───┴───┴───┴───┴───                   │
│        Mon Tue Wed Thu Fri Sat Sun                      │
│                                                         │
│ ✓ Explicit Success (200 OK): 96.4%                     │
│ ⚠ Task Completion Quality: 91.2%                       │
│ ✗ Total Failure Rate: 3.6%                             │
└─────────────────────────────────────────────────────────┘
```

**3. Loop Iteration Distribution**

Identifies agents with abnormal iteration patterns.

```
┌─────────────────────────────────────────────────────────┐
│        Agent Loop Iteration Distribution                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Iterations     Count    Agents                          │
│ ───────────────────────────────────────────────────────│
│ 1-5           ████████████████████████████  1,234      │
│ 6-10          ████████████████              678        │
│ 11-15         ████████                      345        │
│ 16-20         ████                          156        │
│ 21-30         ██                             89        │
│ 31-50         ●                              23  ⚠     │
│ 51+           ●                               5  🚨    │
│                                                         │
│ Avg Iterations: 6.7 | Max: 87 (runaway loop detected) │
│ Terminated: 3 sessions (excessive iterations)           │
└─────────────────────────────────────────────────────────┘
```

**4. Context Saturation Levels**

Real-time heatmap of context window utilization.

```
┌─────────────────────────────────────────────────────────┐
│          Context Saturation by Agent                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Agent              Current   Peak    Avg    Status     │
│ ───────────────────────────────────────────────────────│
│ research-agent       34%     67%     42%    🟢 Healthy │
│ code-generator       56%     78%     61%    🟡 Watch   │
│ data-analyzer        89%     95%     82%    🔴 Critical│
│ summarizer           23%     45%     28%    🟢 Healthy │
│ doc-qa               12%     34%     18%    🟢 Healthy │
│                                                         │
│ Saturation Zones: <60% 🟢 | 60-80% 🟡 | >80% 🔴       │
└─────────────────────────────────────────────────────────┘
```

**5. Tool Error Rates**

Breakdown by tool and error type.

```
┌─────────────────────────────────────────────────────────┐
│              Tool Reliability Matrix                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Tool Name        Calls   Error%  Infra  Semantic Valid │
│ ───────────────────────────────────────────────────────│
│ database_query   4,567    2.3%    45      60       0   │
│ web_search       2,890   12.5%   120     241       0   │
│ file_read        6,123    0.8%    12       0      37   │
│ api_call         1,234   15.2%   187       0       0   │
│ code_execution     890    5.4%    23       0      25   │
│                                                         │
│ Overall Error Rate: 6.7% | Target: <5% ⚠               │
│ Action Required: Investigate api_call infrastructure   │
└─────────────────────────────────────────────────────────┘
```

#### Prometheus Queries

```promql
# Active agent sessions
count(ai_agent_session_active == 1)

# Agent success rate
sum(rate(ai_agent_success_total[5m])) /
sum(rate(ai_agent_requests_total[5m])) * 100

# Loop iteration histogram
histogram_quantile(0.95,
  sum(rate(ai_agent_loop_iteration_bucket[5m])) by (le)
)

# Context saturation by agent
avg(ai_context_saturation_gauge) by (agent_name)

# Tool error rate
sum(rate(ai_tool_error_total[5m])) /
sum(rate(ai_tool_calls_total[5m])) * 100
```

---

### Cost & ROI Dashboard

The **Cost & ROI Dashboard** enables financial observability and cost optimization for AI agent operations.

#### Dashboard Components

**1. Cost per Hour/Day**

```
┌─────────────────────────────────────────────────────────┐
│              Daily Cost Trends                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Cost                                                    │
│ $600 ┤                                                  │
│      │                                    Budget ━━━━━━│
│ $500 ┤ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│      │                                        ╭●        │
│ $400 ┤                                   ╭────╯         │
│      │                              ╭────╯              │
│ $300 ┤                         ╭────╯                   │
│      │                    ╭────╯                        │
│ $200 ┤               ╭────╯                             │
│      │          ╭────╯                                  │
│ $100 ┤     ╭────╯                                       │
│      │ ────╯                                            │
│   $0 └─┴───┴───┴───┴───┴───┴───┴───                    │
│       Mon Tue Wed Thu Fri Sat Sun                       │
│                                                         │
│ Today: $456.78 | Budget: $500/day | Remaining: $43.22  │
│ Projected Month: $13,703 | Budget: $15,000 ✓           │
└─────────────────────────────────────────────────────────┘
```

**2. Cost by Model**

```
┌─────────────────────────────────────────────────────────┐
│            Cost Distribution by Model                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ GPT-4-Turbo         ████████████████████████  $234.56  │
│ (51% of spend)                                          │
│                                                         │
│ Claude-3.5-Sonnet   ███████████████████       $189.23  │
│ (41% of spend)                                          │
│                                                         │
│ GPT-3.5-Turbo       ████                       $32.99  │
│ (7% of spend)                                           │
│                                                         │
│ Other Models        ●                           $8.45  │
│ (2% of spend)                                           │
│                                                         │
│ Total: $465.23 | Avg per Request: $0.013                │
└─────────────────────────────────────────────────────────┘
```

**3. Cost by Team/User**

Multi-tenant cost attribution for chargeback/showback.

```
┌─────────────────────────────────────────────────────────┐
│           Cost Attribution by Team                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Team              Cost      % Total   Requests  $/Req  │
│ ───────────────────────────────────────────────────────│
│ Engineering      $189.45     41.5%     8,567   $0.022  │
│ Product          $134.67     29.5%     12,345  $0.011  │
│ Research          $87.34     19.1%     3,456   $0.025  │
│ Support           $45.23      9.9%     6,789   $0.007  │
│                                                         │
│ Total: $456.69 across 4 teams                           │
│ Cost Allocation: Enabled ✓                              │
└─────────────────────────────────────────────────────────┘
```

**4. Budget Utilization**

```
┌─────────────────────────────────────────────────────────┐
│           Monthly Budget Status                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Budget: $15,000/month                                   │
│ Spent:  $13,245 (88.3%)                                 │
│ Remaining: $1,755 (11.7%)                               │
│                                                         │
│ ████████████████████████████████████░░░░                │
│                                                         │
│ Days Remaining: 4                                       │
│ Avg Daily Spend: $478.75                                │
│ Projected Overage: $0 (within budget) ✓                │
│                                                         │
│ Alert Threshold: 90% ($13,500) - Not Triggered         │
└─────────────────────────────────────────────────────────┘
```

**5. ROI by Feature**

Links cost to business value metrics.

```
┌─────────────────────────────────────────────────────────┐
│          ROI Analysis by Workflow                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Workflow          Cost    Value    ROI    Status       │
│ ───────────────────────────────────────────────────────│
│ Code Generation  $234    $4,680   20x    🟢 Excellent │
│ Data Analysis    $189    $2,835   15x    🟢 Excellent │
│ Documentation     $87    $1,131   13x    🟢 Good      │
│ Customer Support  $45      $540   12x    🟢 Good      │
│ Research          $67      $335    5x    🟡 Marginal  │
│                                                         │
│ Overall ROI: 14.2x (Cost: $622 → Value: $8,831)        │
└─────────────────────────────────────────────────────────┘
```

#### Prometheus Queries

```promql
# Cost per hour
sum(increase(ai_cost_estimate_counter[1h]))

# Cost by model
sum by (model) (increase(ai_cost_estimate_counter[24h]))

# Cost by team
sum by (team_id) (increase(ai_cost_estimate_counter[24h]))

# Budget utilization percentage
sum(increase(ai_cost_estimate_counter[30d])) / 15000 * 100

# Cost per request
sum(increase(ai_cost_estimate_counter[24h])) /
sum(increase(ai_requests_total[24h]))
```

---

### Latency & Throughput Dashboard

The **Latency & Throughput Dashboard** monitors performance and system capacity metrics.

#### Dashboard Components

**1. Latency Percentiles (P50/P90/P99)**

```
┌─────────────────────────────────────────────────────────┐
│         Latency Distribution Over Time                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Latency (s)                                             │
│  30s ┤                                                  │
│      │                                                  │
│  25s ┤                                    P99 ─ ─ ─ ─  │
│      │                              ─ ─ ─              │
│  20s ┤                        ─ ─ ─                     │
│      │                  ─ ─ ─                           │
│  15s ┤            ─ ─ ─                                 │
│      │      ─ ─ ─                                       │
│  10s ┤ ─ ─ ─                  P90 ───────────────      │
│      │                  ──────                          │
│   5s ┤            ──────              P50 ●●●●●●●●●●   │
│      │      ●●●●●●                                      │
│   0s └─┴───┴───┴───┴───┴───┴───┴───                    │
│       00  04  08  12  16  20  24                        │
│                                                         │
│ P50: 2.3s | P90: 8.7s | P99: 18.4s | Max: 29.1s        │
└─────────────────────────────────────────────────────────┘
```

**2. Throughput (Requests/sec)**

```
┌─────────────────────────────────────────────────────────┐
│              Throughput Trends                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Req/sec                                                 │
│   50 ┤                                      ╭●          │
│      │                                 ╭────╯           │
│   40 ┤                            ╭────╯                │
│      │                       ╭────╯                     │
│   30 ┤                  ╭────╯                          │
│      │             ╭────╯                               │
│   20 ┤        ╭────╯                                    │
│      │   ╭────╯                                         │
│   10 ┤───╯                                              │
│      │                                                  │
│    0 └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──              │
│        00 02 04 06 08 10 12 14 16 18 20 22              │
│                                                         │
│ Current: 38.7 req/s | Peak: 52.3 req/s | Avg: 28.4 req/s│
│ Capacity: 100 req/s | Utilization: 39% 🟢              │
└─────────────────────────────────────────────────────────┘
```

**3. Queue Depth**

Monitors backlog of pending requests.

```
┌─────────────────────────────────────────────────────────┐
│              Request Queue Depth                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Queue Size                                              │
│  100 ┤                                                  │
│      │                                                  │
│   80 ┤                                                  │
│      │                                                  │
│   60 ┤                                                  │
│      │                                                  │
│   40 ┤                              ●                   │
│      │                          ●       ●               │
│   20 ┤          ●   ●   ●   ●               ●   ●   ● │
│      │  ●   ●                                          │
│    0 └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──              │
│        00 02 04 06 08 10 12 14 16 18 20 22              │
│                                                         │
│ Current: 12 queued | Max: 43 | Avg: 18                  │
│ Alert Threshold: 50 ✓                                   │
└─────────────────────────────────────────────────────────┘
```

**4. Backend Response Times**

Tracks LLM provider latency separately from agent processing time.

```
┌─────────────────────────────────────────────────────────┐
│         Provider Latency Comparison                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Provider            P50      P95      P99    Avg       │
│ ───────────────────────────────────────────────────────│
│ OpenAI (GPT-4)      1.2s     3.4s     7.8s   1.8s      │
│ Anthropic (Claude)  0.9s     2.1s     5.2s   1.3s  ✓   │
│ AWS Bedrock         1.5s     4.2s     9.1s   2.2s      │
│ Azure OpenAI        1.3s     3.8s     8.4s   2.0s      │
│                                                         │
│ Fastest Provider: Anthropic (1.3s avg)                  │
│ Target: P95 < 5s ✓                                      │
└─────────────────────────────────────────────────────────┘
```

#### Prometheus Queries

```promql
# P50, P90, P99 latency
histogram_quantile(0.50, sum(rate(ai_request_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.90, sum(rate(ai_request_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.99, sum(rate(ai_request_duration_seconds_bucket[5m])) by (le))

# Throughput (requests per second)
sum(rate(ai_requests_total[1m]))

# Queue depth
ai_request_queue_depth

# Provider latency by provider
histogram_quantile(0.95,
  sum(rate(ai_provider_latency_seconds_bucket[5m])) by (provider, le)
)
```

---

## Alert Conditions

### Prometheus Rules Configuration

Complete Prometheus alerting rules for production AI agent monitoring.

```yaml
# /etc/prometheus/rules/a11i-agent-alerts.yml
groups:
  - name: a11i-agent-alerts
    interval: 30s
    rules:

      # ===== STUCK LOOP DETECTION =====

      - alert: AgentStuckLoop
        expr: |
          max by (agent_id, trace_id, session_id) (
            ai_agent_loop_iteration
          ) > 20
        for: 2m
        labels:
          severity: warning
          category: agent_cognition
        annotations:
          summary: "Agent {{ $labels.agent_id }} appears stuck (>20 iterations)"
          description: |
            Agent {{ $labels.agent_id }} has exceeded 20 iterations in session {{ $labels.session_id }}.
            This may indicate a runaway loop or circular reasoning pattern.

            Current iteration count: {{ $value }}
            Session: {{ $labels.session_id }}
            Trace: {{ $labels.trace_id }}
          runbook_url: "https://docs.a11i.dev/runbooks/stuck-loop"
          dashboard_url: "https://grafana.a11i.dev/d/agent-health?var-agent={{ $labels.agent_id }}"

      - alert: AgentRunawayLoop
        expr: |
          max by (agent_id, session_id) (
            ai_agent_loop_iteration
          ) > 50
        labels:
          severity: critical
          category: agent_cognition
          auto_remediate: "true"
        annotations:
          summary: "CRITICAL: Agent {{ $labels.agent_id }} in runaway loop (>50 iterations)"
          description: |
            Agent {{ $labels.agent_id }} has entered a confirmed runaway loop with {{ $value }} iterations.
            Automatic termination initiated.
          runbook_url: "https://docs.a11i.dev/runbooks/stuck-loop"
          action: "terminate_session"

      # ===== COST ANOMALY DETECTION =====

      - alert: CostAnomaly
        expr: |
          sum(rate(ai_cost_estimate_counter[1h])) >
          avg_over_time(sum(rate(ai_cost_estimate_counter[1h]))[7d:1h]) * 2
        for: 15m
        labels:
          severity: critical
          category: cost_management
        annotations:
          summary: "Cost spike detected: 2x higher than weekly average"
          description: |
            Hourly cost rate is {{ $value | humanize }}x above the 7-day average.
            This indicates a potential cost anomaly requiring investigation.

            Current hourly rate: ${{ $value }}/hr
            7-day average: ${{ query "avg_over_time(sum(rate(ai_cost_estimate_counter[1h]))[7d:1h])" | first | value }}/hr
          runbook_url: "https://docs.a11i.dev/runbooks/cost-anomaly"
          dashboard_url: "https://grafana.a11i.dev/d/cost-roi"

      - alert: DailyBudgetExceeded
        expr: |
          sum by (tenant_id) (increase(ai_cost_estimate_counter{tenant_id!=""}[24h]))
          > on(tenant_id) group_left tenant_daily_budget
        labels:
          severity: critical
          category: cost_management
        annotations:
          summary: "Daily budget exceeded for tenant {{ $labels.tenant_id }}"
          description: |
            Tenant {{ $labels.tenant_id }} has exceeded their daily cost budget.

            Spent: ${{ $value }}
            Budget: ${{ query "tenant_daily_budget{tenant_id=\"" $labels.tenant_id "\"}" | first | value }}
          runbook_url: "https://docs.a11i.dev/runbooks/budget-exceeded"
          action: "throttle_tenant_requests"

      - alert: SingleRequestHighCost
        expr: |
          ai_cost_estimate_counter > 5.0
        labels:
          severity: warning
          category: cost_management
        annotations:
          summary: "Individual request cost exceeds $5.00"
          description: |
            A single request cost ${{ $value }}, exceeding the threshold.

            Agent: {{ $labels.agent_name }}
            Model: {{ $labels.model }}
            Trace: {{ $labels.trace_id }}
          runbook_url: "https://docs.a11i.dev/runbooks/high-cost-request"

      # ===== TOOL FAILURE RATE =====

      - alert: HighToolFailureRate
        expr: |
          sum(rate(ai_tool_error_total[5m])) /
          sum(rate(ai_tool_calls_total[5m])) > 0.2
        for: 5m
        labels:
          severity: warning
          category: tool_reliability
        annotations:
          summary: "Tool error rate >20%"
          description: |
            Tool execution error rate is {{ $value | humanizePercentage }}, exceeding the 20% threshold.

            This indicates either infrastructure issues or prompt engineering problems.
          runbook_url: "https://docs.a11i.dev/runbooks/tool-failures"
          dashboard_url: "https://grafana.a11i.dev/d/agent-health?tab=tools"

      - alert: ToolConsistentlyFailing
        expr: |
          sum by (tool_name) (rate(ai_tool_error_total[5m])) /
          sum by (tool_name) (rate(ai_tool_calls_total[5m])) > 0.5
        for: 10m
        labels:
          severity: critical
          category: tool_reliability
        annotations:
          summary: "Tool {{ $labels.tool_name }} failing >50% of executions"
          description: |
            Tool {{ $labels.tool_name }} is failing {{ $value | humanizePercentage }} of executions.
            Tool may need to be disabled temporarily.
          runbook_url: "https://docs.a11i.dev/runbooks/tool-failures"
          action: "consider_disabling_tool"

      - alert: InfrastructureToolErrors
        expr: |
          sum(rate(ai_tool_error_total{error_type="infrastructure"}[5m])) > 10
        for: 5m
        labels:
          severity: critical
          category: infrastructure
        annotations:
          summary: "High rate of infrastructure tool errors"
          description: |
            {{ $value }} infrastructure errors/minute detected.
            This indicates backend system issues requiring immediate attention.
          runbook_url: "https://docs.a11i.dev/runbooks/tool-failures"
          action: "page_sre"

      # ===== CONTEXT SATURATION =====

      - alert: HighContextSaturation
        expr: |
          histogram_quantile(0.95,
            sum(rate(ai_context_saturation_bucket[5m])) by (le)
          ) > 90
        for: 10m
        labels:
          severity: warning
          category: agent_cognition
        annotations:
          summary: "95th percentile context saturation >90%"
          description: |
            Context windows are saturated at {{ $value }}% (P95).
            This will degrade agent performance and increase latency.
          runbook_url: "https://docs.a11i.dev/runbooks/context-saturation"
          dashboard_url: "https://grafana.a11i.dev/d/agent-health?tab=context"

      - alert: CriticalContextSaturation
        expr: |
          ai_context_saturation_gauge > 95
        labels:
          severity: critical
          category: agent_cognition
        annotations:
          summary: "Agent {{ $labels.agent_name }} context at {{ $value }}%"
          description: |
            Agent {{ $labels.agent_name }} context window is critically saturated.

            Saturation: {{ $value }}%
            Session: {{ $labels.session_id }}

            Immediate context pruning required.
          runbook_url: "https://docs.a11i.dev/runbooks/context-saturation"
          action: "trigger_emergency_pruning"

      # ===== QUALITY DEGRADATION =====

      - alert: QualityDegradation
        expr: |
          avg(rate(ai_feedback_negative[1h])) /
          avg(rate(ai_feedback_total[1h])) > 0.15
        for: 30m
        labels:
          severity: warning
          category: quality
        annotations:
          summary: "Negative feedback rate >15%"
          description: |
            User feedback indicates {{ $value | humanizePercentage }} negative responses.
            This suggests agent quality degradation requiring investigation.
          runbook_url: "https://docs.a11i.dev/runbooks/quality-issues"

      # ===== HIGH LATENCY =====

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(ai_request_duration_seconds_bucket[5m])) by (le)
          ) > 30
        for: 5m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "P99 latency >30s"
          description: |
            Request latency P99 is {{ $value }}s, exceeding the 30s threshold.
            This impacts user experience and may indicate performance issues.
          runbook_url: "https://docs.a11i.dev/runbooks/high-latency"

      - alert: TTFTDegradation
        expr: |
          histogram_quantile(0.95,
            sum(rate(ai_latency_ttft_bucket[5m])) by (le)
          ) > 2
        for: 10m
        labels:
          severity: warning
          category: performance
        annotations:
          summary: "Time to First Token P95 >2s"
          description: |
            TTFT latency is {{ $value }}s (P95), degrading perceived responsiveness.
          runbook_url: "https://docs.a11i.dev/runbooks/high-latency"

      # ===== SYSTEM HEALTH =====

      - alert: HighErrorRate
        expr: |
          sum(rate(ai_errors_total[5m])) /
          sum(rate(ai_requests_total[5m])) > 0.05
        for: 10m
        labels:
          severity: critical
          category: reliability
        annotations:
          summary: "Error rate >5%"
          description: |
            System error rate is {{ $value | humanizePercentage }}, exceeding the 5% threshold.
          runbook_url: "https://docs.a11i.dev/runbooks/high-error-rate"

      - alert: ProviderUnavailable
        expr: |
          sum by (provider) (rate(ai_provider_errors{error_type="unavailable"}[5m])) > 5
        labels:
          severity: critical
          category: infrastructure
        annotations:
          summary: "LLM provider {{ $labels.provider }} unavailable"
          description: |
            Provider {{ $labels.provider }} is returning availability errors.
            Failover to backup provider recommended.
          runbook_url: "https://docs.a11i.dev/runbooks/provider-unavailable"
          action: "trigger_failover"

      - alert: RateLimitThrottling
        expr: |
          sum(rate(ai_provider_errors{error_type="rate_limit"}[5m])) > 10
        for: 5m
        labels:
          severity: warning
          category: capacity
        annotations:
          summary: "High rate limit errors ({{ $value }}/min)"
          description: |
            Frequent rate limit errors indicate capacity constraints.
            Consider request throttling or quota increase.
          runbook_url: "https://docs.a11i.dev/runbooks/rate-limits"
```

### Alert Severity Levels

a11i uses a four-tier severity model:

| Severity | SLA Response | Auto-Remediate | Examples |
|----------|--------------|----------------|----------|
| **Critical** | 15 minutes | Yes (if configured) | Runaway loops, budget exceeded, provider down |
| **Warning** | 1 hour | No (manual review) | High context saturation, elevated error rate |
| **Info** | 4 hours | No | Cost optimization opportunities, performance trends |
| **Debug** | Best effort | No | Metric collection issues, non-critical anomalies |

### Alert Routing

Alert routing configuration for Alertmanager:

```yaml
# /etc/alertmanager/config.yml
route:
  receiver: 'default-receiver'
  group_by: ['alertname', 'category']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

  routes:
    # Critical alerts → Page on-call
    - match:
        severity: critical
      receiver: 'pagerduty-oncall'
      group_wait: 10s
      repeat_interval: 15m

    # Cost alerts → Finance team
    - match:
        category: cost_management
      receiver: 'slack-finance'
      group_wait: 1m

    # Quality alerts → Product team
    - match:
        category: quality
      receiver: 'slack-product'

    # Infrastructure alerts → SRE team
    - match:
        category: infrastructure
      receiver: 'pagerduty-sre'
      group_wait: 5s

    # Agent cognition alerts → Engineering team
    - match:
        category: agent_cognition
      receiver: 'slack-engineering'

receivers:
  - name: 'default-receiver'
    slack_configs:
      - api_url: '<slack_webhook_url>'
        channel: '#a11i-alerts'
        title: '[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty-oncall'
    pagerduty_configs:
      - service_key: '<pagerduty_service_key>'
        severity: '{{ .CommonLabels.severity }}'

  - name: 'slack-finance'
    slack_configs:
      - api_url: '<slack_webhook_url>'
        channel: '#finance-alerts'

  - name: 'slack-engineering'
    slack_configs:
      - api_url: '<slack_webhook_url>'
        channel: '#engineering-alerts'
```

---

## Anomaly Detection

### Statistical Anomaly Detection

Statistical methods provide robust baseline anomaly detection without machine learning overhead.

```python
from typing import List, Dict, Optional
import numpy as np
from scipy import stats
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis."""
    is_anomaly: bool
    severity: str  # "low", "medium", "high", "critical"
    method: str
    score: float
    threshold: float
    details: Dict
    timestamp: datetime

class AnomalyDetector:
    """Statistical anomaly detection for AI metrics."""

    def __init__(self, window_size: int = 168):  # 1 week of hourly data
        self.window_size = window_size

    def detect_zscore_anomaly(
        self,
        current_value: float,
        historical: List[float],
        threshold: float = 3.0,
    ) -> AnomalyResult:
        """
        Detect anomaly using Z-score (standard deviations from mean).

        Best for: Normal distributions, detecting absolute outliers
        Sensitivity: Medium (threshold=3 catches ~0.3% of normal data)
        """
        if len(historical) < 10:
            return AnomalyResult(
                is_anomaly=False,
                severity="low",
                method="zscore",
                score=0.0,
                threshold=threshold,
                details={"reason": "insufficient_data", "data_points": len(historical)},
                timestamp=datetime.now()
            )

        mean = np.mean(historical)
        std = np.std(historical)

        if std == 0:
            return AnomalyResult(
                is_anomaly=False,
                severity="low",
                method="zscore",
                score=0.0,
                threshold=threshold,
                details={"reason": "zero_variance", "mean": mean},
                timestamp=datetime.now()
            )

        zscore = (current_value - mean) / std
        abs_zscore = abs(zscore)

        # Determine severity based on Z-score
        if abs_zscore > 5:
            severity = "critical"
        elif abs_zscore > 4:
            severity = "high"
        elif abs_zscore > threshold:
            severity = "medium"
        else:
            severity = "low"

        return AnomalyResult(
            is_anomaly=abs_zscore > threshold,
            severity=severity,
            method="zscore",
            score=abs_zscore,
            threshold=threshold,
            details={
                "zscore": zscore,
                "mean": mean,
                "std": std,
                "direction": "high" if zscore > 0 else "low",
                "current_value": current_value
            },
            timestamp=datetime.now()
        )

    def detect_iqr_anomaly(
        self,
        current_value: float,
        historical: List[float],
        multiplier: float = 1.5,
    ) -> AnomalyResult:
        """
        Detect anomaly using Interquartile Range (IQR).

        Best for: Non-normal distributions, robust to outliers
        Sensitivity: Low (1.5x IQR catches ~0.7% of normal data)
        """
        if len(historical) < 10:
            return AnomalyResult(
                is_anomaly=False,
                severity="low",
                method="iqr",
                score=0.0,
                threshold=multiplier,
                details={"reason": "insufficient_data"},
                timestamp=datetime.now()
            )

        q1 = np.percentile(historical, 25)
        q3 = np.percentile(historical, 75)
        iqr = q3 - q1

        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)

        is_anomaly = current_value < lower_bound or current_value > upper_bound

        # Calculate severity based on distance from bounds
        if current_value < lower_bound:
            distance = (lower_bound - current_value) / iqr if iqr > 0 else 0
            direction = "low"
        elif current_value > upper_bound:
            distance = (current_value - upper_bound) / iqr if iqr > 0 else 0
            direction = "high"
        else:
            distance = 0
            direction = "normal"

        if distance > 3:
            severity = "critical"
        elif distance > 2:
            severity = "high"
        elif distance > 1:
            severity = "medium"
        else:
            severity = "low"

        return AnomalyResult(
            is_anomaly=is_anomaly,
            severity=severity,
            method="iqr",
            score=distance,
            threshold=multiplier,
            details={
                "current_value": current_value,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "direction": direction,
                "distance_from_bound": distance
            },
            timestamp=datetime.now()
        )

    def detect_trend_change(
        self,
        recent_values: List[float],
        historical_values: List[float],
        significance_level: float = 0.05,
    ) -> AnomalyResult:
        """
        Detect if recent trend differs significantly from historical trend.

        Best for: Identifying gradual changes in behavior
        Sensitivity: Based on statistical significance (p-value)
        """
        if len(recent_values) < 3 or len(historical_values) < 10:
            return AnomalyResult(
                is_anomaly=False,
                severity="low",
                method="trend_change",
                score=0.0,
                threshold=significance_level,
                details={"reason": "insufficient_data"},
                timestamp=datetime.now()
            )

        # Calculate slopes using linear regression
        recent_slope = self._calculate_slope(recent_values)
        historical_slope = self._calculate_slope(historical_values[-24:])  # Last 24h

        # T-test for difference in means
        try:
            _, p_value = stats.ttest_ind(
                recent_values,
                historical_values[-len(recent_values):],
            )
        except Exception:
            p_value = 1.0

        is_anomaly = p_value < significance_level

        # Determine severity based on slope change magnitude
        slope_change = abs(recent_slope - historical_slope)
        if slope_change > abs(historical_slope) * 2:
            severity = "high"
        elif slope_change > abs(historical_slope):
            severity = "medium"
        else:
            severity = "low"

        return AnomalyResult(
            is_anomaly=is_anomaly,
            severity=severity if is_anomaly else "low",
            method="trend_change",
            score=1 - p_value,  # Convert p-value to confidence score
            threshold=significance_level,
            details={
                "recent_slope": recent_slope,
                "historical_slope": historical_slope,
                "slope_change": slope_change,
                "p_value": p_value,
                "recent_mean": np.mean(recent_values),
                "historical_mean": np.mean(historical_values)
            },
            timestamp=datetime.now()
        )

    def detect_composite_anomaly(
        self,
        current_value: float,
        historical: List[float],
        recent_window: int = 6,
    ) -> AnomalyResult:
        """
        Composite anomaly detection using multiple methods.

        Combines Z-score, IQR, and trend detection for robust anomaly identification.
        """
        # Run all detection methods
        zscore_result = self.detect_zscore_anomaly(current_value, historical)
        iqr_result = self.detect_iqr_anomaly(current_value, historical)

        recent_values = historical[-recent_window:] + [current_value]
        trend_result = self.detect_trend_change(recent_values, historical)

        # Voting system: if 2+ methods agree, it's an anomaly
        votes = sum([
            zscore_result.is_anomaly,
            iqr_result.is_anomaly,
            trend_result.is_anomaly
        ])

        is_anomaly = votes >= 2

        # Take highest severity
        severities = ["low", "medium", "high", "critical"]
        max_severity = max(
            [zscore_result.severity, iqr_result.severity, trend_result.severity],
            key=lambda s: severities.index(s)
        )

        # Composite score (weighted average)
        composite_score = (
            zscore_result.score * 0.4 +
            iqr_result.score * 0.4 +
            trend_result.score * 0.2
        )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            severity=max_severity if is_anomaly else "low",
            method="composite",
            score=composite_score,
            threshold=0.0,  # No single threshold for composite
            details={
                "votes": votes,
                "zscore": zscore_result.details,
                "iqr": iqr_result.details,
                "trend": trend_result.details,
                "methods_triggered": [
                    m for m, r in [
                        ("zscore", zscore_result),
                        ("iqr", iqr_result),
                        ("trend", trend_result)
                    ] if r.is_anomaly
                ]
            },
            timestamp=datetime.now()
        )

    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate linear regression slope."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        try:
            slope, _, _, _, _ = stats.linregress(x, values)
            return slope
        except Exception:
            return 0.0


# Usage Example
def monitor_cost_anomalies():
    """Example: Monitor for cost anomalies."""
    detector = AnomalyDetector()

    # Fetch historical cost data (last 7 days, hourly)
    historical_costs = fetch_historical_costs(days=7)

    # Get current hour's cost
    current_cost = fetch_current_cost()

    # Run anomaly detection
    result = detector.detect_composite_anomaly(
        current_value=current_cost,
        historical=historical_costs
    )

    if result.is_anomaly:
        # Trigger alert
        alert_manager.send_alert(
            title=f"Cost Anomaly Detected: {result.severity.upper()}",
            description=f"Current cost ${current_cost:.2f} is anomalous",
            severity=result.severity,
            details=result.details,
            runbook_url="https://docs.a11i.dev/runbooks/cost-anomaly"
        )
```

### ML-Based Anomaly Detection

For advanced deployments, machine learning provides adaptive anomaly detection.

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from typing import Dict, List

class MLAnomalyDetector:
    """ML-powered anomaly detection for AI agent metrics."""

    def __init__(self, contamination: float = 0.05):
        """
        Initialize ML anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (default 5%)
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

    def train(self, historical_data: pd.DataFrame):
        """
        Train anomaly detector on historical metrics.

        Args:
            historical_data: DataFrame with metric columns
        """
        features = self._extract_features(historical_data)

        # Normalize features
        features_scaled = self.scaler.fit_transform(features)

        # Train isolation forest
        self.model.fit(features_scaled)
        self.is_trained = True

        print(f"✓ ML Anomaly Detector trained on {len(features)} samples")

    def detect(self, current_metrics: Dict[str, float]) -> Dict:
        """
        Detect if current metrics are anomalous.

        Args:
            current_metrics: Dictionary of current metric values

        Returns:
            Detection result with anomaly status and score
        """
        if not self.is_trained:
            return {
                "is_anomaly": False,
                "reason": "model_not_trained",
                "confidence": 0.0
            }

        # Convert metrics to feature vector
        features = self._metrics_to_features(current_metrics)
        features_scaled = self.scaler.transform([features])

        # Predict anomaly
        prediction = self.model.predict(features_scaled)[0]
        score = self.model.decision_function(features_scaled)[0]

        # Convert score to confidence (0-1 range)
        # Negative scores = anomalies, lower = more anomalous
        confidence = 1 / (1 + np.exp(score))  # Sigmoid normalization

        is_anomaly = prediction == -1

        # Determine severity based on score
        if score < -0.5:
            severity = "critical"
        elif score < -0.3:
            severity = "high"
        elif score < -0.1:
            severity = "medium"
        else:
            severity = "low"

        return {
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "anomaly_score": -score,  # Higher = more anomalous
            "severity": severity if is_anomaly else "low",
            "features": dict(zip(self.feature_names, features)),
            "timestamp": datetime.now()
        }

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from DataFrame."""
        self.feature_names = [
            "token_usage_input",
            "token_usage_output",
            "cost_usd",
            "latency_p99_ms",
            "error_rate_pct",
            "loop_iterations_avg",
            "context_saturation_avg",
            "tool_error_rate",
        ]

        # Calculate features from raw data
        features = df[self.feature_names].values
        return features

    def _metrics_to_features(self, metrics: Dict[str, float]) -> List[float]:
        """Convert current metrics to feature vector."""
        return [metrics.get(f, 0.0) for f in self.feature_names]

    def save_model(self, filepath: str):
        """Save trained model to disk."""
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from disk."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")


# Training Pipeline Example
def train_anomaly_detector():
    """Train ML anomaly detector on historical data."""

    # Fetch historical metrics (last 30 days)
    query = """
        SELECT
            timestamp,
            SUM(CASE WHEN type='input' THEN tokens ELSE 0 END) as token_usage_input,
            SUM(CASE WHEN type='output' THEN tokens ELSE 0 END) as token_usage_output,
            SUM(cost_usd) as cost_usd,
            percentile(latency_ms, 0.99) as latency_p99_ms,
            AVG(error_rate) as error_rate_pct,
            AVG(loop_iterations) as loop_iterations_avg,
            AVG(context_saturation) as context_saturation_avg,
            AVG(tool_error_rate) as tool_error_rate
        FROM ai_metrics
        WHERE timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY date_trunc('hour', timestamp)
        ORDER BY timestamp
    """

    historical_data = pd.read_sql(query, database_connection)

    # Train detector
    detector = MLAnomalyDetector(contamination=0.05)
    detector.train(historical_data)

    # Save model
    detector.save_model("/models/anomaly_detector_v1.pkl")

    return detector
```

### Pattern Recognition

Detect recurring patterns that indicate systemic issues.

```python
from collections import Counter, defaultdict
from typing import List, Tuple
import re

class PatternDetector:
    """Detect recurring patterns in agent behavior and errors."""

    def __init__(self, min_occurrences: int = 3):
        self.min_occurrences = min_occurrences

    def detect_error_patterns(
        self,
        error_logs: List[Dict],
    ) -> List[Dict]:
        """
        Detect recurring error patterns.

        Returns list of patterns sorted by frequency.
        """
        patterns = []

        # Group errors by type and tool
        error_groups = defaultdict(list)
        for error in error_logs:
            key = (
                error.get('tool_name', 'unknown'),
                error.get('error_type', 'unknown'),
                error.get('status_code', 0)
            )
            error_groups[key].append(error)

        # Analyze each group
        for key, errors in error_groups.items():
            if len(errors) < self.min_occurrences:
                continue

            tool_name, error_type, status_code = key

            # Extract error messages
            error_messages = [e.get('error_message', '') for e in errors]
            message_counts = Counter(error_messages)

            # Find most common error message
            if message_counts:
                most_common_msg, count = message_counts.most_common(1)[0]

                if count >= self.min_occurrences:
                    patterns.append({
                        "type": "error_pattern",
                        "tool_name": tool_name,
                        "error_type": error_type,
                        "status_code": status_code,
                        "occurrences": count,
                        "error_message": most_common_msg,
                        "first_seen": min(e['timestamp'] for e in errors),
                        "last_seen": max(e['timestamp'] for e in errors),
                        "affected_agents": list(set(e.get('agent_name') for e in errors)),
                        "severity": self._assess_severity(count, len(errors))
                    })

        return sorted(patterns, key=lambda x: x['occurrences'], reverse=True)

    def detect_cost_patterns(
        self,
        cost_events: List[Dict],
    ) -> List[Dict]:
        """Detect patterns in cost spikes."""
        patterns = []

        # Group by agent and model
        cost_groups = defaultdict(list)
        for event in cost_events:
            key = (event['agent_name'], event['model'])
            cost_groups[key].append(event)

        for (agent_name, model), events in cost_groups.items():
            if len(events) < self.min_occurrences:
                continue

            # Calculate cost statistics
            costs = [e['cost_usd'] for e in events]
            avg_cost = np.mean(costs)
            max_cost = max(costs)

            # Identify high-cost events
            high_cost_events = [e for e in events if e['cost_usd'] > avg_cost * 2]

            if len(high_cost_events) >= self.min_occurrences:
                patterns.append({
                    "type": "cost_pattern",
                    "agent_name": agent_name,
                    "model": model,
                    "occurrences": len(high_cost_events),
                    "avg_cost": avg_cost,
                    "max_cost": max_cost,
                    "total_cost": sum(costs),
                    "pattern": "recurring_high_cost",
                    "severity": "high" if max_cost > avg_cost * 5 else "medium"
                })

        return patterns

    def detect_loop_patterns(
        self,
        loop_events: List[Dict],
    ) -> List[Dict]:
        """Detect patterns in agent loop behavior."""
        patterns = []

        # Analyze tool call sequences
        for event in loop_events:
            tool_sequence = event.get('tool_sequence', [])

            # Detect alternating pattern (A → B → A → B)
            if self._has_alternating_pattern(tool_sequence):
                patterns.append({
                    "type": "loop_pattern",
                    "pattern": "alternating_tools",
                    "agent_name": event['agent_name'],
                    "session_id": event['session_id'],
                    "tool_sequence": tool_sequence,
                    "iterations": len(tool_sequence),
                    "severity": "high"
                })

            # Detect repetition pattern (A → A → A)
            elif self._has_repetition_pattern(tool_sequence):
                patterns.append({
                    "type": "loop_pattern",
                    "pattern": "tool_repetition",
                    "agent_name": event['agent_name'],
                    "session_id": event['session_id'],
                    "tool_sequence": tool_sequence,
                    "iterations": len(tool_sequence),
                    "severity": "high"
                })

        return patterns

    def _assess_severity(self, occurrence_count: int, total_count: int) -> str:
        """Assess severity based on occurrence frequency."""
        ratio = occurrence_count / total_count if total_count > 0 else 0

        if ratio > 0.5:
            return "critical"
        elif ratio > 0.3:
            return "high"
        elif ratio > 0.1:
            return "medium"
        else:
            return "low"

    def _has_alternating_pattern(self, sequence: List[str]) -> bool:
        """Detect alternating pattern in tool calls."""
        if len(sequence) < 4:
            return False

        for i in range(len(sequence) - 3):
            if (sequence[i] == sequence[i + 2] and
                sequence[i + 1] == sequence[i + 3] and
                sequence[i] != sequence[i + 1]):
                return True

        return False

    def _has_repetition_pattern(self, sequence: List[str]) -> bool:
        """Detect repetition pattern in tool calls."""
        if len(sequence) < 3:
            return False

        for i in range(len(sequence) - 2):
            if sequence[i] == sequence[i + 1] == sequence[i + 2]:
                return True

        return False
```

---

## Runbook Examples

### Stuck Agent Loop

**Runbook: Agent Stuck in Reasoning Loop**

```yaml
runbook:
  name: "Stuck Agent Loop"
  id: "RB-001"
  severity: "critical"
  category: "agent_cognition"

  description: |
    Agent is taking excessive iterations (>20) without converging on a solution.
    This indicates a runaway loop consuming resources without progress.

  symptoms:
    - Agent loop iteration count > 20
    - Same tool called repeatedly with similar parameters
    - No observable progress in task completion
    - Context saturation increasing steadily
    - Cost accumulating rapidly without results

  detection:
    alert: "AgentStuckLoop"
    prometheus_query: |
      max by (agent_id, session_id) (ai_agent_loop_iteration) > 20
    dashboard: "https://grafana.a11i.dev/d/agent-health"

  investigation:
    steps:
      - step: 1
        action: "Check agent trace for iteration pattern"
        command: |
          curl -s "https://api.a11i.dev/traces?session_id=${SESSION_ID}" | \
            jq '.iterations[] | {iteration: .number, action: .action, tokens: .tokens}'

      - step: 2
        action: "Examine tool outputs for errors or empty results"
        command: |
          curl -s "https://api.a11i.dev/traces/${TRACE_ID}/tools" | \
            jq '.tool_calls[] | select(.status != "success")'

      - step: 3
        action: "Review context saturation level"
        query: |
          ai_context_saturation_gauge{session_id="${SESSION_ID}"}

      - step: 4
        action: "Check for semantic similarity between iterations"
        command: |
          python /scripts/analyze_iteration_similarity.py --session ${SESSION_ID}

  remediation:
    automatic:
      - condition: "iteration_count > 50"
        action: "Terminate agent session"
        command: |
          curl -X POST "https://api.a11i.dev/sessions/${SESSION_ID}/terminate" \
            -H "Authorization: Bearer ${API_KEY}" \
            -d '{"reason": "runaway_loop", "save_trace": true}'

      - condition: "cost > $10"
        action: "Terminate and alert financial team"
        command: |
          curl -X POST "https://api.a11i.dev/sessions/${SESSION_ID}/terminate"
          slack-notify --channel finance --message "High-cost runaway loop terminated"

    manual:
      - priority: 1
        action: "Review agent prompt and system instructions"
        details: |
          Check for circular logic in prompt:
          - Are stop conditions clearly defined?
          - Is task decomposition working correctly?
          - Are tool descriptions accurate?

      - priority: 2
        action: "Check tool configuration and availability"
        details: |
          Verify tools are:
          - Returning expected data formats
          - Not timing out silently
          - Providing actionable results

      - priority: 3
        action: "Implement iteration limits"
        details: |
          Add hard iteration limit to agent:
          ```python
          MAX_ITERATIONS = 30
          if iteration_count > MAX_ITERATIONS:
              raise MaxIterationsExceeded()
          ```

      - priority: 4
        action: "Add circuit breaker for repeated tool calls"
        details: |
          Prevent same tool being called >3 times consecutively:
          ```python
          if tool_call_history[-3:] == [tool_name] * 3:
              break  # Exit loop
          ```

  prevention:
    - "Set MAX_ITERATIONS limit on all agents (recommended: 30)"
    - "Implement semantic similarity checking between iterations"
    - "Add tool call circuit breakers"
    - "Monitor context saturation and prune aggressively"
    - "Use streaming feedback to detect lack of progress"

  escalation:
    - time: "10 minutes"
      action: "Page on-call engineer"
      contact: "oncall-engineering@company.com"

    - time: "30 minutes"
      action: "Page team lead"
      contact: "team-lead@company.com"

  postmortem:
    required: true
    template: "https://docs.a11i.dev/postmortem-template"
    questions:
      - "What caused the loop? (prompt, tool, context)"
      - "Why didn't automatic termination trigger sooner?"
      - "What prevented the agent from making progress?"
      - "How can we detect this pattern earlier?"
```

### Cost Anomaly

**Runbook: Unexpected Cost Spike**

```yaml
runbook:
  name: "Cost Anomaly"
  id: "RB-002"
  severity: "critical"
  category: "cost_management"

  description: |
    Hourly cost rate is significantly higher than historical baseline,
    indicating potential runaway usage or inefficient agent behavior.

  symptoms:
    - Cost rate >2x weekly average
    - Sudden spike in token usage
    - Increased usage of expensive models (GPT-4)
    - High number of requests from single agent/user

  detection:
    alert: "CostAnomaly"
    prometheus_query: |
      sum(rate(ai_cost_estimate_counter[1h])) >
      avg_over_time(sum(rate(ai_cost_estimate_counter[1h]))[7d:1h]) * 2

  investigation:
    steps:
      - step: 1
        action: "Identify top cost drivers"
        query: |
          topk(10,
            sum by (agent_name, model) (increase(ai_cost_estimate_counter[1h]))
          )

      - step: 2
        action: "Check for runaway loops contributing to cost"
        query: |
          sum by (session_id) (ai_cost_estimate_counter) > 5

      - step: 3
        action: "Analyze token usage patterns"
        query: |
          sum by (type) (increase(ai_token_usage_counter[1h]))

      - step: 4
        action: "Review recent model changes"
        command: |
          git log --since="2 hours ago" --grep="model" -- agents/

  remediation:
    automatic:
      - condition: "cost_rate > $100/hour"
        action: "Enable request throttling"
        command: |
          kubectl scale deployment/a11i-gateway --replicas=1
          kubectl annotate service/a11i-api rate-limit="10req/min"

    manual:
      - priority: 1
        action: "Terminate high-cost sessions"
        details: |
          Identify and terminate sessions costing >$10:
          ```bash
          curl -s "https://api.a11i.dev/sessions?min_cost=10" | \
            jq -r '.sessions[].id' | \
            xargs -I {} curl -X POST "https://api.a11i.dev/sessions/{}/terminate"
          ```

      - priority: 2
        action: "Review recent prompt changes"
        details: |
          Check for changes that increased token usage:
          - Longer system prompts
          - Additional few-shot examples
          - Verbose output formats

      - priority: 3
        action: "Implement cost guardrails"
        details: |
          Add per-request cost limits:
          ```python
          MAX_COST_PER_REQUEST = 1.0  # $1
          if estimated_cost > MAX_COST_PER_REQUEST:
              raise CostLimitExceeded()
          ```

  prevention:
    - "Set daily budget limits per tenant"
    - "Implement cost-based request throttling"
    - "Monitor cost trends with anomaly detection"
    - "Use cheaper models for simple tasks"
    - "Optimize context window usage"

  escalation:
    - time: "15 minutes"
      action: "Notify finance team"
      contact: "finance@company.com"

    - time: "1 hour"
      action: "Page engineering leadership"
      contact: "eng-leadership@company.com"
```

### High Tool Failure Rate

**Runbook: Tool Execution Failures**

```yaml
runbook:
  name: "High Tool Failure Rate"
  id: "RB-003"
  severity: "warning"
  category: "tool_reliability"

  description: |
    Tool execution error rate exceeds 20%, indicating infrastructure
    issues or prompt engineering problems.

  symptoms:
    - Tool error rate >20%
    - Infrastructure errors (5xx, timeouts)
    - Validation errors (400, invalid parameters)
    - Semantic errors (empty results, wrong format)

  detection:
    alert: "HighToolFailureRate"
    prometheus_query: |
      sum(rate(ai_tool_error_total[5m])) /
      sum(rate(ai_tool_calls_total[5m])) > 0.2

  investigation:
    steps:
      - step: 1
        action: "Identify failing tools"
        query: |
          topk(5,
            sum by (tool_name) (rate(ai_tool_error_total[5m])) /
            sum by (tool_name) (rate(ai_tool_calls_total[5m]))
          )

      - step: 2
        action: "Classify error types"
        query: |
          sum by (error_type) (rate(ai_tool_error_total[5m]))

      - step: 3
        action: "Check tool backend health"
        command: |
          kubectl get pods -l component=tool-backend
          kubectl logs -l component=tool-backend --tail=100

      - step: 4
        action: "Review tool parameter validation"
        command: |
          curl -s "https://api.a11i.dev/tools/${TOOL_NAME}/errors?limit=20" | \
            jq '.errors[] | {params: .parameters, error: .error_message}'

  remediation:
    automatic:
      - condition: "infrastructure_errors > 10/min"
        action: "Trigger failover to backup system"
        command: |
          kubectl set env deployment/tool-backend FAILOVER_MODE=true

    manual:
      - priority: 1
        action: "Fix infrastructure issues"
        details: |
          If error_type == "infrastructure":
          - Check backend service health
          - Review error logs for root cause
          - Scale up resources if capacity issue
          - Contact on-call SRE if persistent

      - priority: 2
        action: "Fix validation errors"
        details: |
          If error_type == "validation":
          - Review tool schema documentation
          - Update agent prompts with correct examples
          - Add parameter validation before tool calls
          - Improve few-shot examples

      - priority: 3
        action: "Fix semantic errors"
        details: |
          If error_type == "semantic":
          - Review tool output format
          - Add result validation
          - Improve error handling in tools
          - Consider tool redesign if persistent

  prevention:
    - "Monitor tool reliability scores"
    - "Implement circuit breakers for failing tools"
    - "Add comprehensive tool parameter validation"
    - "Maintain tool schema documentation"
    - "Test tools independently before agent integration"
```

### Context Saturation Crisis

**Runbook: Critical Context Saturation**

```yaml
runbook:
  name: "Context Saturation Crisis"
  id: "RB-004"
  severity: "critical"
  category: "agent_cognition"

  description: |
    Agent context window is critically saturated (>95%), leading to
    performance degradation and potential failures.

  symptoms:
    - Context saturation >95%
    - Increasing latency
    - Quality degradation
    - "Context length exceeded" errors

  detection:
    alert: "CriticalContextSaturation"
    prometheus_query: |
      ai_context_saturation_gauge > 95

  investigation:
    steps:
      - step: 1
        action: "Analyze context composition"
        command: |
          curl -s "https://api.a11i.dev/sessions/${SESSION_ID}/context" | \
            jq '{
              system_prompt: .system_prompt_tokens,
              conversation: .conversation_tokens,
              tools: .tool_schema_tokens,
              retrieved: .retrieved_context_tokens,
              total: .total_tokens,
              limit: .context_limit
            }'

      - step: 2
        action: "Check context growth rate"
        query: |
          rate(ai_context_tokens_used[5m])

      - step: 3
        action: "Review pruning effectiveness"
        query: |
          ai_context_pruning_effectiveness

  remediation:
    automatic:
      - condition: "saturation > 95%"
        action: "Trigger emergency context pruning"
        command: |
          curl -X POST "https://api.a11i.dev/sessions/${SESSION_ID}/prune" \
            -d '{"strategy": "aggressive", "target_saturation": 60}'

    manual:
      - priority: 1
        action: "Prune conversation history"
        details: |
          Remove older conversation turns:
          ```python
          # Keep only last N turns
          MAX_HISTORY_TURNS = 5
          conversation_history = conversation_history[-MAX_HISTORY_TURNS:]
          ```

      - priority: 2
        action: "Optimize tool schemas"
        details: |
          - Remove unused tool definitions
          - Simplify tool descriptions
          - Use tool schema compression

      - priority: 3
        action: "Implement semantic summarization"
        details: |
          Summarize older context before pruning:
          ```python
          old_context = conversation_history[:-5]
          summary = summarize(old_context)
          conversation_history = [summary] + conversation_history[-5:]
          ```

  prevention:
    - "Implement proactive context pruning at 80% saturation"
    - "Use semantic summarization for long conversations"
    - "Optimize tool schema sizes"
    - "Monitor context growth rates"
    - "Set context saturation alerts at 70%, 80%, 90%"
```

---

## Grafana Dashboard Configuration

Complete Grafana dashboard configuration for a11i monitoring.

```json
{
  "dashboard": {
    "title": "a11i Agent Health & Performance",
    "tags": ["a11i", "ai-agents", "observability"],
    "timezone": "browser",
    "refresh": "30s",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(ai_requests_total[5m]))",
            "legendFormat": "Requests/sec",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "color": {"mode": "palette-classic"}
          }
        }
      },
      {
        "id": 2,
        "title": "Error Rate",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(ai_errors_total[5m])) / sum(rate(ai_requests_total[5m])) * 100",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 1, "color": "yellow"},
                {"value": 5, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 3,
        "title": "Context Saturation Heatmap",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "sum(rate(ai_context_saturation_bucket[5m])) by (le)",
            "format": "heatmap",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "custom": {
              "hideFrom": {"tooltip": false, "viz": false, "legend": false}
            }
          }
        },
        "options": {
          "calculate": true,
          "cellGap": 1,
          "color": {
            "exponent": 0.5,
            "fill": "dark-orange",
            "mode": "scheme",
            "scale": "exponential",
            "scheme": "Oranges",
            "steps": 64
          },
          "exemplars": {"color": "rgba(255,0,255,0.7)"}
        }
      },
      {
        "id": 4,
        "title": "Cost per Hour",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16},
        "targets": [
          {
            "expr": "sum(increase(ai_cost_estimate_counter[1h]))",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "mappings": [],
            "color": {"mode": "thresholds"},
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 20, "color": "yellow"},
                {"value": 50, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 5,
        "title": "Token Usage by Type",
        "type": "piechart",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 20},
        "targets": [
          {
            "expr": "sum by (type) (increase(ai_token_usage_counter[24h]))",
            "legendFormat": "{{type}}",
            "refId": "A"
          }
        ],
        "options": {
          "legend": {"displayMode": "table", "placement": "right"},
          "pieType": "pie",
          "tooltip": {"mode": "single"}
        }
      },
      {
        "id": 6,
        "title": "Tool Error Rate by Tool",
        "type": "table",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 20},
        "targets": [
          {
            "expr": "sum by (tool_name) (rate(ai_tool_error_total[1h])) / sum by (tool_name) (rate(ai_tool_calls_total[1h])) * 100",
            "format": "table",
            "instant": true,
            "refId": "A"
          }
        ],
        "transformations": [
          {
            "id": "organize",
            "options": {
              "excludeByName": {"Time": true},
              "indexByName": {"tool_name": 0, "Value": 1},
              "renameByName": {"tool_name": "Tool", "Value": "Error Rate (%)"}
            }
          }
        ]
      },
      {
        "id": 7,
        "title": "Latency Percentiles",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 28},
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(ai_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.90, sum(rate(ai_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P90",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(ai_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P99",
            "refId": "C"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "custom": {"lineWidth": 2, "fillOpacity": 10}
          }
        }
      }
    ],
    "templating": {
      "list": [
        {
          "name": "agent_name",
          "type": "query",
          "query": "label_values(ai_requests_total, agent_name)",
          "refresh": 1
        },
        {
          "name": "model",
          "type": "query",
          "query": "label_values(ai_requests_total, model)",
          "refresh": 1
        }
      ]
    }
  }
}
```

---

## Integration with Incident Response

### Automated Incident Creation

Link alerts to incident management systems:

```python
from typing import Dict
import requests

class IncidentManager:
    """Integrate monitoring alerts with incident management."""

    def __init__(self, pagerduty_key: str, slack_webhook: str):
        self.pagerduty_key = pagerduty_key
        self.slack_webhook = slack_webhook

    def create_incident(
        self,
        title: str,
        description: str,
        severity: str,
        alert_details: Dict
    ):
        """Create incident from alert."""

        # Create PagerDuty incident for critical alerts
        if severity == "critical":
            self._create_pagerduty_incident(title, description, alert_details)

        # Post to Slack
        self._post_slack_notification(title, description, severity, alert_details)

    def _create_pagerduty_incident(
        self,
        title: str,
        description: str,
        alert_details: Dict
    ):
        """Create PagerDuty incident."""
        payload = {
            "incident": {
                "type": "incident",
                "title": title,
                "service": {"id": "PXXXXXX", "type": "service_reference"},
                "urgency": "high",
                "body": {
                    "type": "incident_body",
                    "details": description
                },
                "incident_key": alert_details.get("trace_id", "unknown")
            }
        }

        headers = {
            "Authorization": f"Token token={self.pagerduty_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.pagerduty.com/incidents",
            json=payload,
            headers=headers
        )

        return response.json()

    def _post_slack_notification(
        self,
        title: str,
        description: str,
        severity: str,
        alert_details: Dict
    ):
        """Post Slack notification."""

        color_map = {
            "critical": "#FF0000",
            "warning": "#FFA500",
            "info": "#0000FF"
        }

        payload = {
            "attachments": [
                {
                    "color": color_map.get(severity, "#808080"),
                    "title": title,
                    "text": description,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": severity.upper(),
                            "short": True
                        },
                        {
                            "title": "Agent",
                            "value": alert_details.get("agent_name", "unknown"),
                            "short": True
                        }
                    ],
                    "footer": "a11i Monitoring",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }

        requests.post(self.slack_webhook, json=payload)
```

---

## Key Takeaways

> **Production AI Agent Monitoring Best Practices**
>
> 1. **Layer Your Monitoring**: Combine infrastructure metrics (Layer 1), agent-specific metrics (Layer 2), and predictive analytics (Layer 3) for comprehensive coverage.
>
> 2. **Alert on Leading Indicators**: Context saturation and loop velocity predict failures before they occur. Don't wait for explicit errors.
>
> 3. **Cost is a First-Class Metric**: Financial observability is not optional. Budget alerts and anomaly detection prevent cost surprises.
>
> 4. **Automate Remediation**: Critical issues like runaway loops should trigger automatic termination. Manual intervention is too slow.
>
> 5. **Maintain Runbooks**: Every alert must have a clear runbook with investigation steps and remediation actions.
>
> 6. **Use Composite Anomaly Detection**: Combine statistical methods (Z-score, IQR) with ML-based detection for robust anomaly identification.
>
> 7. **Monitor Patterns, Not Just Events**: Recurring error patterns and tool loops indicate systemic issues requiring architectural fixes.
>
> 8. **Integrate with Incident Response**: Alerts should automatically create incidents and notify appropriate teams based on severity.

**Implementation Priorities:**

**Week 1:**
- Deploy core dashboards (Usage, Agent Health, Cost)
- Configure basic Prometheus alerts (stuck loops, high costs)
- Set up alert routing to Slack

**Week 2:**
- Implement statistical anomaly detection
- Create runbooks for top 5 alert conditions
- Integrate with incident management (PagerDuty)

**Week 3:**
- Deploy ML-based anomaly detection
- Add pattern recognition for errors and loops
- Configure advanced Grafana dashboards

**Week 4:**
- Implement automated remediation for critical alerts
- Establish SLAs and on-call rotation
- Conduct tabletop exercises for incident response

---

**Related Documentation:**

- [Core Metrics](/home/becker/projects/a11i/docs/03-core-platform/core-metrics.md) - Metric definitions and semantics
- [System Architecture](/home/becker/projects/a11i/docs/02-architecture/system-architecture.md) - Platform architecture overview
- [Data Pipeline](/home/becker/projects/a11i/docs/02-architecture/data-pipeline.md) - Metrics collection and storage
- [Deployment Operations](/home/becker/projects/a11i/docs/06-operations/deployment-operations.md) - Deployment procedures
- [Incident Response](/home/becker/projects/a11i/docs/06-operations/incident-response.md) - Incident handling procedures

---

*Document Status: Draft | Last Updated: 2025-11-26 | Maintained by: a11i Documentation Team*
