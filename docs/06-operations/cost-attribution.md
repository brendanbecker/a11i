---
title: "Cost Attribution: Multi-Dimensional LLM Cost Tracking for a11i"
category: "Operations"
tags: ["cost", "attribution", "chargeback", "showback", "budgets", "forecasting", "roi", "finops"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "../02-architecture/data-pipeline.md"
  - "../03-core-platform/span-hierarchy.md"
  - "../03-core-platform/core-metrics.md"
  - "./alerting-rules.md"
---

# Cost Attribution: Multi-Dimensional LLM Cost Tracking

## Table of Contents

- [Introduction](#introduction)
- [Rate Card Management](#rate-card-management)
- [Per-Request Cost Calculation](#per-request-cost-calculation)
- [Multi-Dimensional Cost Tracking](#multi-dimensional-cost-tracking)
- [Financial Dashboards and Reports](#financial-dashboards-and-reports)
- [Budget Alerts and Guardrails](#budget-alerts-and-guardrails)
- [Cost Forecasting](#cost-forecasting)
- [Cost Optimization Insights](#cost-optimization-insights)
- [Implementation Architecture](#implementation-architecture)
- [Key Takeaways](#key-takeaways)

## Introduction

LLM costs can quickly spiral out of control without proper attribution and governance. Organizations need to:

- **Track costs accurately** across providers, models, teams, and features
- **Allocate costs fairly** for internal chargeback and showback
- **Prevent budget overruns** with real-time alerts and throttling
- **Optimize spending** by identifying inefficiencies and opportunities
- **Forecast expenses** to support financial planning and budgeting

**a11i provides comprehensive cost attribution** by calculating per-request costs using hot-reloadable rate cards, tracking costs across multiple dimensions (user, team, feature, conversation), and enabling sophisticated financial analysis through materialized views and dashboards.

### Cost Attribution Dimensions

a11i tracks costs across five key dimensions:

| Dimension | Use Case | Example Query |
|-----------|----------|---------------|
| **Per Request** | Real-time cost visibility | "What did this specific API call cost?" |
| **Per Session** | Conversation economics | "How much did this customer support chat cost?" |
| **Per User (Showback)** | Usage transparency | "Show employees their individual LLM usage" |
| **Per Team (Chargeback)** | Internal billing | "Bill engineering team for their AI assistant costs" |
| **Per Feature (ROI)** | Business value analysis | "What's the ROI of our code completion feature?" |

This multi-dimensional approach enables both **granular cost control** (preventing individual runaway requests) and **strategic business analysis** (measuring feature-level profitability).

## Rate Card Management

### Hot-Reloadable Rate Cards

Rate cards are stored in YAML configuration files that can be reloaded without service restarts, enabling rapid response to provider pricing changes.

**Configuration Structure:**

```yaml
# rate-cards.yaml (hot-reloadable)
rate_cards:
  openai:
    gpt-4o:
      input_per_1k: 0.003
      output_per_1k: 0.015
      effective_date: "2024-01-01"
      version: "2024-01-01"

    gpt-4-turbo:
      input_per_1k: 0.01
      output_per_1k: 0.03
      effective_date: "2023-11-01"

    gpt-3.5-turbo:
      input_per_1k: 0.0005
      output_per_1k: 0.0015
      effective_date: "2023-06-01"

    o1-preview:
      input_per_1k: 0.015
      output_per_1k: 0.06
      effective_date: "2024-09-01"

  anthropic:
    claude-sonnet-4-20250514:
      input_per_1k: 0.003
      output_per_1k: 0.015
      effective_date: "2025-05-14"

    claude-3-5-sonnet:
      input_per_1k: 0.003
      output_per_1k: 0.015
      effective_date: "2024-06-20"

    claude-3-opus:
      input_per_1k: 0.015
      output_per_1k: 0.075
      effective_date: "2024-03-04"

    claude-3-haiku:
      input_per_1k: 0.00025
      output_per_1k: 0.00125
      effective_date: "2024-03-04"

  aws_bedrock:
    anthropic.claude-3-sonnet:
      input_per_1k: 0.003
      output_per_1k: 0.015
      effective_date: "2024-03-01"

    amazon.titan-text-express:
      input_per_1k: 0.0002
      output_per_1k: 0.0006
      effective_date: "2023-09-01"

    meta.llama3-70b:
      input_per_1k: 0.00099
      output_per_1k: 0.00099
      effective_date: "2024-04-01"

  azure_openai:
    gpt-4o:
      input_per_1k: 0.003
      output_per_1k: 0.015
      effective_date: "2024-01-01"
      # Azure pricing may vary by region
      region_overrides:
        eastus: {input_per_1k: 0.003, output_per_1k: 0.015}
        westeurope: {input_per_1k: 0.0033, output_per_1k: 0.0165}

  custom_overrides:
    # Enterprise customers with volume discounts
    org_abc123:
      gpt-4o:
        input_per_1k: 0.0025  # 17% discount
        output_per_1k: 0.0125
        reason: "Enterprise volume commitment: 100M tokens/month"
        effective_date: "2024-01-01"
        expires: "2024-12-31"

    org_startup_xyz:
      claude-3-5-sonnet:
        input_per_1k: 0.0027  # 10% startup discount
        output_per_1k: 0.0135
        reason: "Startup program participant"
        effective_date: "2024-06-01"
        expires: "2025-06-01"

# Rate card versioning for historical accuracy
version: "2025.11.26"
last_updated: "2025-11-26T10:00:00Z"
update_source: "https://openai.com/pricing"
```

### Rate Card Reload Service

```python
import yaml
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import asyncio

class RateCardManager:
    """Manage hot-reloadable rate cards with versioning."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.rate_cards: Dict = {}
        self.version: str = ""
        self.last_loaded: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Load rate cards from YAML configuration."""
        async with self._lock:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.rate_cards = config.get("rate_cards", {})
            self.version = config.get("version", "unknown")
            self.last_loaded = datetime.utcnow()

            print(f"Loaded rate cards version {self.version} at {self.last_loaded}")

    async def reload_if_changed(self) -> bool:
        """Reload rate cards if configuration file has changed."""
        current_mtime = self.config_path.stat().st_mtime

        if self.last_loaded is None or current_mtime > self.last_loaded.timestamp():
            await self.load()
            return True

        return False

    def get_rate(
        self,
        provider: str,
        model: str,
        org_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Optional[Dict]:
        """Get rate card for provider/model with optional overrides."""
        # Check org-specific override first
        if org_id and org_id in self.rate_cards.get("custom_overrides", {}):
            org_rates = self.rate_cards["custom_overrides"][org_id]
            if model in org_rates:
                return org_rates[model]

        # Check provider rates
        provider_rates = self.rate_cards.get(provider, {})
        model_rates = provider_rates.get(model)

        if not model_rates:
            return None

        # Apply regional overrides for Azure
        if region and "region_overrides" in model_rates:
            regional = model_rates["region_overrides"].get(region)
            if regional:
                return {**model_rates, **regional}

        return model_rates
```

## Per-Request Cost Calculation

### Cost Calculator Implementation

```python
from typing import Optional

class CostCalculator:
    """Calculate LLM request costs with multi-provider support."""

    def __init__(self, rate_card_manager: RateCardManager):
        self.rate_cards = rate_card_manager

    def calculate(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        org_id: Optional[str] = None,
        region: Optional[str] = None,
        cached_tokens: int = 0,  # For prompt caching
    ) -> dict:
        """Calculate cost for a single request.

        Args:
            provider: LLM provider (openai, anthropic, aws_bedrock, etc.)
            model: Model identifier (gpt-4o, claude-3-5-sonnet, etc.)
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            org_id: Organization ID for custom pricing overrides
            region: Azure region for regional pricing
            cached_tokens: Number of cached tokens (charged at reduced rate)

        Returns:
            Dictionary with cost breakdown:
            {
                "total_cost_usd": 0.00456,
                "input_cost_usd": 0.0012,
                "output_cost_usd": 0.00336,
                "cached_cost_usd": 0.0,
                "rate_card_version": "2025.11.26",
                "effective_date": "2024-01-01",
                "applied_org_discount": False
            }
        """
        # Get applicable rate card
        rates = self.rate_cards.get_rate(provider, model, org_id, region)

        if not rates:
            # Unknown model - log warning and return zero cost
            print(f"WARNING: No rate card found for {provider}/{model}")
            return {
                "total_cost_usd": 0.0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "cached_cost_usd": 0.0,
                "rate_card_version": "unknown",
                "error": "no_rate_card"
            }

        # Calculate costs
        cost_breakdown = self._compute_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            rates=rates,
        )

        # Add metadata
        cost_breakdown.update({
            "rate_card_version": self.rate_cards.version,
            "effective_date": rates.get("effective_date"),
            "applied_org_discount": org_id in self.rate_cards.rate_cards.get("custom_overrides", {}),
        })

        return cost_breakdown

    def _compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        rates: dict,
    ) -> dict:
        """Compute cost components from token counts and rates."""
        # Standard input cost
        input_cost = (input_tokens / 1000.0) * rates["input_per_1k"]

        # Output cost
        output_cost = (output_tokens / 1000.0) * rates["output_per_1k"]

        # Cached tokens (if supported) - typically 10% of input rate
        cached_cost = 0.0
        if cached_tokens > 0:
            cache_rate = rates.get("cached_per_1k", rates["input_per_1k"] * 0.1)
            cached_cost = (cached_tokens / 1000.0) * cache_rate

        total_cost = input_cost + output_cost + cached_cost

        return {
            "total_cost_usd": round(total_cost, 6),
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "cached_cost_usd": round(cached_cost, 6),
        }
```

### Integration with Span Processing

```python
class SpanProcessor:
    """Process spans and attach cost attributes."""

    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calc = cost_calculator

    async def process_llm_span(self, span: dict) -> dict:
        """Process LLM span and add cost attributes."""
        attributes = span.get("attributes", {})

        # Extract token counts from span attributes
        input_tokens = attributes.get("llm.usage.input_tokens", 0)
        output_tokens = attributes.get("llm.usage.output_tokens", 0)
        cached_tokens = attributes.get("llm.usage.cached_tokens", 0)

        # Extract model information
        provider = attributes.get("llm.provider", "")
        model = attributes.get("llm.model", "")

        # Extract tenant/org context
        org_id = attributes.get("tenant.id")
        region = attributes.get("cloud.region")

        # Calculate cost
        cost_info = self.cost_calc.calculate(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            org_id=org_id,
            region=region,
            cached_tokens=cached_tokens,
        )

        # Add cost attributes to span
        attributes.update({
            "llm.cost.total_usd": cost_info["total_cost_usd"],
            "llm.cost.input_usd": cost_info["input_cost_usd"],
            "llm.cost.output_usd": cost_info["output_cost_usd"],
            "llm.cost.cached_usd": cost_info["cached_cost_usd"],
            "llm.cost.rate_version": cost_info["rate_card_version"],
            "llm.cost.org_discount": cost_info.get("applied_org_discount", False),
        })

        span["attributes"] = attributes
        return span
```

## Multi-Dimensional Cost Tracking

### Dimension Tagging Strategy

Every LLM request span is tagged with contextual dimensions that enable multi-dimensional cost analysis:

```python
# Dimensions attached to every span
span_dimensions = {
    # Identity dimensions
    "tenant.id": "acme_corp",
    "user.id": "user_12345",
    "team.id": "engineering",
    "department.id": "product_development",

    # Feature dimensions
    "feature.id": "code_completion",
    "feature.name": "AI Code Assistant",
    "product.area": "developer_tools",

    # Session dimensions
    "conversation.id": "conv_abc123",
    "session.id": "session_xyz789",
    "workflow.id": "onboarding_flow",

    # Technical dimensions
    "llm.provider": "openai",
    "llm.model": "gpt-4o",
    "cloud.region": "us-east-1",
    "deployment.env": "production",

    # Business dimensions
    "customer.tier": "enterprise",
    "billing.plan": "pro",
    "cost_center": "CC-1234",
}
```

### ClickHouse Materialized View for Cost Aggregation

```sql
-- Multi-dimensional cost aggregation materialized view
CREATE MATERIALIZED VIEW cost_attribution_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (tenant_id, timestamp, model, user_id, team_id, feature_id)
POPULATE
AS SELECT
    -- Time dimension
    toStartOfHour(start_time) as timestamp,
    toDate(start_time) as date,

    -- Identity dimensions
    tenant_id,
    user_id,
    team_id,
    department_id,

    -- Feature dimensions
    feature_id,
    feature_name,
    product_area,

    -- Session dimensions
    conversation_id,
    session_id,

    -- Technical dimensions
    provider,
    model,
    cloud_region,
    deployment_env,

    -- Business dimensions
    customer_tier,
    billing_plan,
    cost_center,

    -- Aggregated metrics
    sum(input_tokens) as total_input_tokens,
    sum(output_tokens) as total_output_tokens,
    sum(cached_tokens) as total_cached_tokens,
    sum(cost_usd) as total_cost_usd,
    sum(input_cost_usd) as total_input_cost_usd,
    sum(output_cost_usd) as total_output_cost_usd,

    count() as request_count,
    avg(duration_ms) as avg_latency_ms,
    quantile(0.95)(duration_ms) as p95_latency_ms,

    -- Error tracking
    countIf(status = 'error') as error_count,
    sum(if(status = 'error', cost_usd, 0)) as error_cost_usd

FROM agent_traces
WHERE span_kind = 'llm_request'
GROUP BY
    timestamp, date,
    tenant_id, user_id, team_id, department_id,
    feature_id, feature_name, product_area,
    conversation_id, session_id,
    provider, model, cloud_region, deployment_env,
    customer_tier, billing_plan, cost_center;

-- Create indexes for common query patterns
ALTER TABLE cost_attribution_mv ADD INDEX idx_user (user_id) TYPE minmax GRANULARITY 3;
ALTER TABLE cost_attribution_mv ADD INDEX idx_team (team_id) TYPE minmax GRANULARITY 3;
ALTER TABLE cost_attribution_mv ADD INDEX idx_feature (feature_id) TYPE bloom_filter GRANULARITY 3;
```

### Per-Session Cost Tracking

```sql
-- Session-level cost aggregation view
CREATE MATERIALIZED VIEW session_costs_mv
ENGINE = SummingMergeTree()
ORDER BY (tenant_id, session_id, conversation_id)
AS SELECT
    tenant_id,
    session_id,
    conversation_id,
    user_id,
    feature_id,

    min(timestamp) as session_start,
    max(timestamp) as session_end,
    dateDiff('second', session_start, session_end) as session_duration_sec,

    sum(total_cost_usd) as session_cost_usd,
    sum(request_count) as total_requests,
    sum(total_input_tokens + total_output_tokens) as total_tokens,

    avg(avg_latency_ms) as session_avg_latency_ms,
    max(p95_latency_ms) as session_max_p95_latency

FROM cost_attribution_mv
GROUP BY tenant_id, session_id, conversation_id, user_id, feature_id;
```

## Financial Dashboards and Reports

### Chargeback Report: Cost by Team

```sql
-- Monthly team chargeback report
SELECT
    team_id,
    department_id,
    cost_center,

    sum(total_cost_usd) as monthly_cost_usd,
    sum(request_count) as total_requests,
    sum(total_input_tokens + total_output_tokens) as total_tokens,

    -- Per-request metrics
    monthly_cost_usd / total_requests as avg_cost_per_request,
    total_tokens / total_requests as avg_tokens_per_request,

    -- Model breakdown
    groupArray((model, sum(total_cost_usd))) as cost_by_model,

    -- Cost trend (compare to previous month)
    monthly_cost_usd - (
        SELECT sum(total_cost_usd)
        FROM cost_attribution_mv
        WHERE team_id = t.team_id
          AND timestamp >= date_sub(MONTH, 1, toStartOfMonth(now()))
          AND timestamp < toStartOfMonth(now())
    ) as cost_change_usd,

    cost_change_usd / monthly_cost_usd * 100 as cost_change_percent

FROM cost_attribution_mv t
WHERE timestamp >= toStartOfMonth(now())
  AND timestamp < toStartOfDay(now() + INTERVAL 1 DAY)
GROUP BY team_id, department_id, cost_center
ORDER BY monthly_cost_usd DESC;
```

### Showback Report: Cost by User

```sql
-- User cost leaderboard (showback report)
SELECT
    user_id,
    team_id,

    -- Current month costs
    sum(total_cost_usd) as cost_this_month,
    sum(request_count) as requests_this_month,
    cost_this_month / requests_this_month as avg_cost_per_request,

    -- Usage patterns
    countDistinct(conversation_id) as unique_conversations,
    countDistinct(feature_id) as features_used,

    -- Model preferences
    topK(3)(model) as top_models,

    -- Time distribution
    countIf(toHour(timestamp) >= 9 AND toHour(timestamp) < 17) as business_hours_requests,
    countIf(toHour(timestamp) < 9 OR toHour(timestamp) >= 17) as after_hours_requests,

    -- Efficiency metrics
    sum(error_count) as total_errors,
    total_errors / requests_this_month * 100 as error_rate_percent

FROM cost_attribution_mv
WHERE timestamp >= toStartOfMonth(now())
  AND timestamp < toStartOfDay(now() + INTERVAL 1 DAY)
GROUP BY user_id, team_id
ORDER BY cost_this_month DESC
LIMIT 100;
```

### ROI Analysis: Cost by Feature

```sql
-- Feature-level ROI analysis
WITH feature_revenue AS (
    -- Assume external revenue data joined from business metrics
    SELECT
        feature_id,
        sum(revenue_usd) as monthly_revenue
    FROM business_metrics.feature_revenue
    WHERE month = toStartOfMonth(now())
    GROUP BY feature_id
)
SELECT
    c.feature_id,
    c.feature_name,
    c.product_area,

    -- Costs
    sum(c.total_cost_usd) as monthly_cost_usd,
    sum(c.request_count) as total_requests,

    -- Revenue (if available)
    r.monthly_revenue,

    -- ROI calculation
    (r.monthly_revenue - monthly_cost_usd) as profit_usd,
    (r.monthly_revenue / NULLIF(monthly_cost_usd, 0)) as roi_multiple,
    ((r.monthly_revenue - monthly_cost_usd) / NULLIF(r.monthly_revenue, 0) * 100) as profit_margin_percent,

    -- Usage metrics
    countDistinct(c.user_id) as active_users,
    monthly_cost_usd / active_users as cost_per_user,

    -- Quality metrics
    avg(c.avg_latency_ms) as avg_response_time_ms,
    sum(c.error_count) / sum(c.request_count) * 100 as error_rate_percent

FROM cost_attribution_mv c
LEFT JOIN feature_revenue r ON c.feature_id = r.feature_id
WHERE c.timestamp >= toStartOfMonth(now())
  AND c.timestamp < toStartOfDay(now() + INTERVAL 1 DAY)
GROUP BY c.feature_id, c.feature_name, c.product_area, r.monthly_revenue
ORDER BY roi_multiple DESC;
```

### Model Comparison Dashboard

```sql
-- Model cost-effectiveness comparison
SELECT
    provider,
    model,

    -- Volume metrics
    sum(total_cost_usd) as total_spend,
    sum(request_count) as total_requests,
    sum(total_input_tokens + total_output_tokens) as total_tokens,

    -- Cost efficiency
    total_spend / total_tokens * 1000 as cost_per_1k_tokens,
    total_spend / total_requests as avg_cost_per_request,

    -- Performance metrics
    avg(avg_latency_ms) as avg_latency,
    avg(p95_latency_ms) as p95_latency,

    -- Quality metrics
    sum(error_count) / total_requests * 100 as error_rate_percent,

    -- Usage share
    total_spend / (SELECT sum(total_cost_usd) FROM cost_attribution_mv WHERE timestamp >= toStartOfMonth(now())) * 100 as spend_share_percent

FROM cost_attribution_mv
WHERE timestamp >= toStartOfMonth(now())
GROUP BY provider, model
ORDER BY total_spend DESC;
```

## Budget Alerts and Guardrails

### Budget Configuration

```yaml
# budget-alerts.yaml
budgets:
  # Team monthly budget
  - name: team_engineering_monthly
    enabled: true
    scope:
      team_id: engineering
    limit_usd: 10000
    period: monthly
    reset_day: 1  # First day of month
    alerts:
      - threshold_percent: 50
        channels: [email]
        recipients: ["eng-leads@company.com"]
        message: "Engineering team has reached 50% of monthly AI budget"

      - threshold_percent: 80
        channels: [email, slack]
        recipients: ["eng-leads@company.com"]
        slack_webhook: "https://hooks.slack.com/..."
        message: "⚠️ Engineering team at 80% of monthly AI budget"

      - threshold_percent: 100
        channels: [email, slack, pagerduty]
        recipients: ["eng-leads@company.com", "finance@company.com"]
        message: "🚨 Engineering team has EXCEEDED monthly AI budget"
        action: warn  # Options: warn, throttle, block

  # Project daily budget
  - name: project_chatbot_daily
    enabled: true
    scope:
      feature_id: customer_chatbot
    limit_usd: 500
    period: daily
    alerts:
      - threshold_percent: 90
        channels: [slack]
        slack_webhook: "https://hooks.slack.com/..."
        message: "Customer chatbot approaching daily budget limit"
        action: throttle  # Reduce rate limits

  # User budget (prevent runaway costs)
  - name: user_daily_limit
    enabled: true
    scope:
      user_id: "*"  # Apply to all users
    limit_usd: 100
    period: daily
    alerts:
      - threshold_percent: 100
        channels: [email]
        action: block  # Block further requests
        message: "Daily user AI usage limit exceeded"

  # Organization-wide monthly budget
  - name: org_monthly_budget
    enabled: true
    scope:
      tenant_id: acme_corp
    limit_usd: 50000
    period: monthly
    alerts:
      - threshold_percent: 75
        channels: [email, slack]
        recipients: ["cto@company.com", "cfo@company.com"]

      - threshold_percent: 90
        channels: [email, slack, pagerduty]
        severity: high
        action: throttle

      - threshold_percent: 100
        channels: [email, slack, pagerduty]
        severity: critical
        action: block
```

### Budget Alert Service Implementation

```python
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio

class BudgetAlertService:
    """Monitor budgets and trigger alerts when thresholds are exceeded."""

    def __init__(self, db_client, alert_manager, budget_config: Dict):
        self.db = db_client
        self.alerts = alert_manager
        self.budgets = budget_config["budgets"]
        self.triggered_alerts = {}  # Track to prevent duplicate alerts

    async def check_all_budgets(self):
        """Check all configured budgets and trigger alerts."""
        for budget in self.budgets:
            if not budget.get("enabled", True):
                continue

            try:
                await self.check_budget(budget)
            except Exception as e:
                print(f"Error checking budget {budget['name']}: {e}")

    async def check_budget(self, budget: Dict):
        """Check a single budget and trigger alerts if needed."""
        # Get current spend for budget scope and period
        current_spend = await self.get_current_spend(budget)
        limit = budget["limit_usd"]

        # Calculate utilization
        utilization_percent = (current_spend / limit) * 100

        # Check each alert threshold
        for alert in budget["alerts"]:
            threshold = alert["threshold_percent"]

            if utilization_percent >= threshold:
                # Check if we've already alerted for this threshold
                alert_key = f"{budget['name']}:{threshold}"

                if not self._should_alert(alert_key, budget["period"]):
                    continue

                # Trigger alert
                await self.trigger_alert(
                    budget=budget,
                    alert=alert,
                    current_spend=current_spend,
                    limit=limit,
                    utilization_percent=utilization_percent,
                )

                # Execute action if specified
                if "action" in alert:
                    await self.execute_action(
                        action=alert["action"],
                        budget=budget,
                        utilization_percent=utilization_percent,
                    )

                # Record that we've alerted
                self.triggered_alerts[alert_key] = datetime.utcnow()

    async def get_current_spend(self, budget: Dict) -> float:
        """Get current spend for budget scope and period."""
        period_start = self._get_period_start(budget["period"], budget.get("reset_day", 1))
        scope = budget["scope"]

        # Build query based on scope
        where_clauses = ["timestamp >= %(start)s"]
        params = {"start": period_start}

        for key, value in scope.items():
            if value == "*":
                continue  # Wildcard - no filter
            where_clauses.append(f"{key} = %({key})s")
            params[key] = value

        query = f"""
            SELECT sum(total_cost_usd) as spend
            FROM cost_attribution_mv
            WHERE {' AND '.join(where_clauses)}
        """

        result = await self.db.query_one(query, params)
        return result["spend"] or 0.0

    def _get_period_start(self, period: str, reset_day: int = 1) -> datetime:
        """Calculate period start time."""
        now = datetime.utcnow()

        if period == "daily":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)

        elif period == "weekly":
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        elif period == "monthly":
            return now.replace(day=reset_day, hour=0, minute=0, second=0, microsecond=0)

        else:
            raise ValueError(f"Unknown period: {period}")

    def _should_alert(self, alert_key: str, period: str) -> bool:
        """Check if we should alert (prevent duplicate alerts in same period)."""
        last_alert = self.triggered_alerts.get(alert_key)

        if not last_alert:
            return True

        # Alert once per period
        period_start = self._get_period_start(period)
        return last_alert < period_start

    async def trigger_alert(
        self,
        budget: Dict,
        alert: Dict,
        current_spend: float,
        limit: float,
        utilization_percent: float,
    ):
        """Send alert notifications through configured channels."""
        message = alert.get("message", f"Budget {budget['name']} at {utilization_percent:.1f}%")

        alert_data = {
            "budget_name": budget["name"],
            "current_spend": current_spend,
            "limit": limit,
            "utilization_percent": utilization_percent,
            "threshold_percent": alert["threshold_percent"],
            "period": budget["period"],
            "scope": budget["scope"],
        }

        # Send through each configured channel
        for channel in alert.get("channels", []):
            if channel == "email":
                await self.alerts.send_email(
                    recipients=alert.get("recipients", []),
                    subject=f"Budget Alert: {budget['name']}",
                    body=message,
                    data=alert_data,
                )

            elif channel == "slack":
                await self.alerts.send_slack(
                    webhook=alert.get("slack_webhook"),
                    message=message,
                    data=alert_data,
                )

            elif channel == "pagerduty":
                await self.alerts.send_pagerduty(
                    severity=alert.get("severity", "warning"),
                    message=message,
                    data=alert_data,
                )

    async def execute_action(
        self,
        action: str,
        budget: Dict,
        utilization_percent: float,
    ):
        """Execute budget enforcement action."""
        scope = budget["scope"]

        if action == "warn":
            # Just log warning
            print(f"WARNING: Budget {budget['name']} at {utilization_percent:.1f}%")

        elif action == "throttle":
            # Reduce rate limits for scope
            await self.apply_throttling(scope, reduction_factor=0.5)

        elif action == "block":
            # Block all requests for scope
            await self.block_requests(scope)

    async def apply_throttling(self, scope: Dict, reduction_factor: float):
        """Reduce rate limits for budget scope."""
        # Implementation would integrate with rate limiting system
        print(f"Applying {reduction_factor}x throttling to scope: {scope}")

    async def block_requests(self, scope: Dict):
        """Block all requests for budget scope."""
        # Implementation would integrate with API gateway
        print(f"Blocking requests for scope: {scope}")
```

## Cost Forecasting

### Monthly Forecast Implementation

```python
import statistics
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class CostForecaster:
    """Forecast future costs based on historical usage patterns."""

    def __init__(self, db_client):
        self.db = db_client

    async def forecast_monthly(
        self,
        tenant_id: str,
        scope: Optional[Dict] = None,
    ) -> Dict:
        """Forecast month-end cost based on current trend.

        Args:
            tenant_id: Tenant identifier
            scope: Optional scope filters (team_id, feature_id, etc.)

        Returns:
            Forecast data including projected spend and confidence intervals
        """
        # Get daily costs for current month
        daily_costs = await self.get_daily_costs(tenant_id, scope)

        if not daily_costs:
            return {
                "forecast": None,
                "confidence": "none",
                "error": "no_data"
            }

        days_elapsed = len(daily_costs)
        days_in_month = self._days_in_current_month()

        # Need at least 3 days for reasonable forecast
        if days_elapsed < 3:
            return {
                "current_spend": sum(daily_costs),
                "forecast": None,
                "confidence": "insufficient_data",
                "days_elapsed": days_elapsed,
            }

        # Calculate linear projection
        avg_daily = sum(daily_costs) / days_elapsed
        linear_projected = avg_daily * days_in_month

        # Calculate confidence interval
        std_dev = statistics.stdev(daily_costs) if len(daily_costs) > 1 else 0
        # Standard error of the mean for remaining days
        remaining_days = days_in_month - days_elapsed
        confidence_95 = 1.96 * std_dev * math.sqrt(remaining_days)

        # Detect trend (increasing/decreasing)
        trend = self._calculate_trend(daily_costs)

        # Adjust projection based on trend
        if trend["direction"] == "increasing":
            trend_adjusted = linear_projected * (1 + trend["rate"])
        elif trend["direction"] == "decreasing":
            trend_adjusted = linear_projected * (1 - trend["rate"])
        else:
            trend_adjusted = linear_projected

        # Determine confidence level
        cv = std_dev / avg_daily if avg_daily > 0 else 0  # Coefficient of variation
        if days_elapsed >= 14 and cv < 0.3:
            confidence = "high"
        elif days_elapsed >= 7 and cv < 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "current_spend": sum(daily_costs),
            "days_elapsed": days_elapsed,
            "days_in_month": days_in_month,
            "avg_daily_cost": avg_daily,

            # Forecasts
            "forecast_linear": round(linear_projected, 2),
            "forecast_trend_adjusted": round(trend_adjusted, 2),
            "low_estimate_95": round(trend_adjusted - confidence_95, 2),
            "high_estimate_95": round(trend_adjusted + confidence_95, 2),

            # Trend analysis
            "trend_direction": trend["direction"],
            "trend_rate": trend["rate"],

            # Confidence
            "confidence": confidence,
            "coefficient_of_variation": round(cv, 3),
            "std_dev": round(std_dev, 2),
        }

    async def get_daily_costs(
        self,
        tenant_id: str,
        scope: Optional[Dict] = None,
    ) -> List[float]:
        """Get daily cost totals for current month."""
        where_clauses = [
            "tenant_id = %(tenant_id)s",
            "date >= toStartOfMonth(now())",
            "date < toStartOfDay(now() + INTERVAL 1 DAY)",
        ]
        params = {"tenant_id": tenant_id}

        if scope:
            for key, value in scope.items():
                where_clauses.append(f"{key} = %({key})s")
                params[key] = value

        query = f"""
            SELECT
                date,
                sum(total_cost_usd) as daily_cost
            FROM cost_attribution_mv
            WHERE {' AND '.join(where_clauses)}
            GROUP BY date
            ORDER BY date
        """

        results = await self.db.query(query, params)
        return [r["daily_cost"] for r in results]

    def _calculate_trend(self, daily_costs: List[float]) -> Dict:
        """Calculate trend direction and rate using simple linear regression."""
        n = len(daily_costs)
        if n < 2:
            return {"direction": "stable", "rate": 0.0}

        # Simple linear regression
        x = list(range(n))
        y = daily_costs

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return {"direction": "stable", "rate": 0.0}

        slope = numerator / denominator

        # Determine direction and rate
        if abs(slope) < y_mean * 0.05:  # Less than 5% change
            direction = "stable"
            rate = 0.0
        elif slope > 0:
            direction = "increasing"
            rate = min(slope / y_mean, 0.5)  # Cap at 50% increase
        else:
            direction = "decreasing"
            rate = min(abs(slope) / y_mean, 0.5)  # Cap at 50% decrease

        return {
            "direction": direction,
            "rate": round(rate, 3),
            "slope": slope,
        }

    def _days_in_current_month(self) -> int:
        """Get number of days in current month."""
        now = datetime.utcnow()
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)

        return (next_month - now.replace(day=1)).days
```

## Cost Optimization Insights

### Optimization Opportunity Detection

```python
class CostOptimizationService:
    """Identify cost optimization opportunities and recommendations."""

    def __init__(self, db_client):
        self.db = db_client

    async def analyze_inefficiencies(
        self,
        tenant_id: str,
        lookback_days: int = 30,
    ) -> List[Dict]:
        """Find cost optimization opportunities.

        Returns list of opportunities with:
        - type: Category of optimization
        - description: Human-readable description
        - potential_savings: Estimated monthly savings in USD
        - recommendation: Actionable recommendation
        - affected_requests: Number of requests that could be optimized
        """
        opportunities = []

        # 1. Model downgrade opportunities
        expensive_simple = await self.find_expensive_simple_queries(tenant_id, lookback_days)
        if expensive_simple["count"] > 0:
            opportunities.append({
                "type": "model_downgrade",
                "priority": "high",
                "description": f"{expensive_simple['count']} simple queries using expensive models",
                "potential_savings_monthly": expensive_simple["potential_savings"],
                "current_monthly_cost": expensive_simple["current_cost"],
                "recommendation": (
                    f"Consider using GPT-3.5-turbo or Claude Haiku for queries with "
                    f"< 100 output tokens and simple prompts. "
                    f"Potential savings: ${expensive_simple['potential_savings']:.2f}/month"
                ),
                "affected_requests": expensive_simple["count"],
                "example_prompts": expensive_simple["examples"][:5],
            })

        # 2. Prompt caching opportunities
        duplicates = await self.find_duplicate_prompts(tenant_id, lookback_days)
        if duplicates["count"] > 100:  # Worth caching if >100 duplicates
            opportunities.append({
                "type": "prompt_caching",
                "priority": "medium",
                "description": f"{duplicates['count']} duplicate prompts detected",
                "potential_savings_monthly": duplicates["potential_savings"],
                "current_monthly_cost": duplicates["duplicate_cost"],
                "recommendation": (
                    f"Enable prompt caching for {len(duplicates['top_prompts'])} "
                    f"frequently repeated system prompts. "
                    f"Potential savings: ${duplicates['potential_savings']:.2f}/month (90% cache hit rate)"
                ),
                "affected_requests": duplicates["count"],
                "top_duplicate_prompts": duplicates["top_prompts"][:10],
            })

        # 3. Prompt optimization (reduce verbosity)
        verbose = await self.find_verbose_prompts(tenant_id, lookback_days)
        if verbose["avg_tokens"] > 1000:
            opportunities.append({
                "type": "prompt_optimization",
                "priority": "medium",
                "description": f"System prompts averaging {verbose['avg_tokens']} tokens",
                "potential_savings_monthly": verbose["potential_savings"],
                "recommendation": (
                    f"System prompts are {verbose['avg_tokens']} tokens on average. "
                    f"Consider condensing to ~500 tokens. "
                    f"Potential savings: ${verbose['potential_savings']:.2f}/month"
                ),
                "affected_requests": verbose["count"],
                "verbose_prompts": verbose["examples"][:5],
            })

        # 4. Error retry optimization
        errors = await self.find_expensive_errors(tenant_id, lookback_days)
        if errors["cost"] > 100:  # $100+ wasted on errors
            opportunities.append({
                "type": "error_reduction",
                "priority": "high",
                "description": f"${errors['cost']:.2f} spent on failed requests",
                "potential_savings_monthly": errors["cost"],
                "recommendation": (
                    f"Failed requests cost ${errors['cost']:.2f}/month. "
                    f"Top error: {errors['top_error_type']} ({errors['top_error_count']} occurrences). "
                    f"Implement better input validation and error handling."
                ),
                "affected_requests": errors["count"],
                "error_breakdown": errors["by_type"],
            })

        # 5. Batch processing opportunities
        batchable = await self.find_batchable_requests(tenant_id, lookback_days)
        if batchable["count"] > 1000:
            opportunities.append({
                "type": "batching",
                "priority": "low",
                "description": f"{batchable['count']} sequential requests that could be batched",
                "potential_savings_monthly": batchable["latency_savings_hours"] * 50,  # Assume $50/hour value
                "recommendation": (
                    f"{batchable['count']} requests are made sequentially but could be batched. "
                    f"This would save {batchable['latency_savings_hours']:.1f} hours of wall-clock time monthly."
                ),
                "affected_requests": batchable["count"],
            })

        # Sort by potential savings
        opportunities.sort(key=lambda x: x["potential_savings_monthly"], reverse=True)

        return opportunities

    async def find_expensive_simple_queries(
        self,
        tenant_id: str,
        lookback_days: int,
    ) -> Dict:
        """Find queries using expensive models for simple tasks."""
        query = """
            SELECT
                count() as query_count,
                sum(total_cost_usd) as current_cost,
                avg(total_output_tokens) as avg_output_tokens,
                groupArray(5)(prompt_template) as example_prompts
            FROM agent_traces
            WHERE tenant_id = %(tenant_id)s
              AND start_time >= now() - INTERVAL %(days)s DAY
              AND model IN ('gpt-4o', 'claude-3-opus', 'gpt-4-turbo')
              AND total_output_tokens < 100
              AND prompt_complexity_score < 0.3  -- Simple prompts
        """

        result = await self.db.query_one(query, {
            "tenant_id": tenant_id,
            "days": lookback_days,
        })

        # Calculate potential savings if downgraded to cheaper model
        # Assume 80% cost reduction (e.g., GPT-4o -> GPT-3.5-turbo)
        monthly_factor = 30 / lookback_days
        potential_savings = result["current_cost"] * 0.8 * monthly_factor

        return {
            "count": result["query_count"],
            "current_cost": result["current_cost"] * monthly_factor,
            "potential_savings": potential_savings,
            "examples": result["example_prompts"],
        }

    async def find_duplicate_prompts(
        self,
        tenant_id: str,
        lookback_days: int,
    ) -> Dict:
        """Find frequently repeated prompts that could benefit from caching."""
        query = """
            WITH prompt_hashes AS (
                SELECT
                    cityHash64(system_prompt) as prompt_hash,
                    any(system_prompt) as prompt_text,
                    count() as occurrence_count,
                    sum(input_tokens) as total_input_tokens,
                    sum(input_cost_usd) as total_input_cost
                FROM agent_traces
                WHERE tenant_id = %(tenant_id)s
                  AND start_time >= now() - INTERVAL %(days)s DAY
                GROUP BY prompt_hash
                HAVING occurrence_count > 10
            )
            SELECT
                sum(occurrence_count) as total_duplicates,
                sum(total_input_cost) as duplicate_cost,
                topK(10)((occurrence_count, prompt_text)) as top_prompts
            FROM prompt_hashes
        """

        result = await self.db.query_one(query, {
            "tenant_id": tenant_id,
            "days": lookback_days,
        })

        # Prompt caching reduces input costs by ~90%
        monthly_factor = 30 / lookback_days
        potential_savings = result["duplicate_cost"] * 0.9 * monthly_factor

        return {
            "count": result["total_duplicates"],
            "duplicate_cost": result["duplicate_cost"] * monthly_factor,
            "potential_savings": potential_savings,
            "top_prompts": result["top_prompts"],
        }

    async def find_verbose_prompts(
        self,
        tenant_id: str,
        lookback_days: int,
    ) -> Dict:
        """Find overly verbose system prompts."""
        query = """
            SELECT
                count() as prompt_count,
                avg(system_prompt_tokens) as avg_tokens,
                sum(input_cost_usd) as total_system_cost,
                groupArray(5)(system_prompt) as examples
            FROM agent_traces
            WHERE tenant_id = %(tenant_id)s
              AND start_time >= now() - INTERVAL %(days)s DAY
              AND system_prompt_tokens > 1000
        """

        result = await self.db.query_one(query, {
            "tenant_id": tenant_id,
            "days": lookback_days,
        })

        # Assume 50% reduction in system prompt length is achievable
        monthly_factor = 30 / lookback_days
        potential_savings = result["total_system_cost"] * 0.5 * monthly_factor

        return {
            "count": result["prompt_count"],
            "avg_tokens": result["avg_tokens"],
            "potential_savings": potential_savings,
            "examples": result["examples"],
        }

    async def find_expensive_errors(
        self,
        tenant_id: str,
        lookback_days: int,
    ) -> Dict:
        """Find cost wasted on error requests."""
        query = """
            SELECT
                count() as error_count,
                sum(cost_usd) as error_cost,
                topK(1)(error_type) as top_error,
                groupArray((error_type, count())) as error_breakdown
            FROM agent_traces
            WHERE tenant_id = %(tenant_id)s
              AND start_time >= now() - INTERVAL %(days)s DAY
              AND status = 'error'
              AND cost_usd > 0
            GROUP BY tenant_id
        """

        result = await self.db.query_one(query, {
            "tenant_id": tenant_id,
            "days": lookback_days,
        })

        monthly_factor = 30 / lookback_days

        return {
            "count": result["error_count"],
            "cost": result["error_cost"] * monthly_factor,
            "top_error_type": result["top_error"][0] if result["top_error"] else "unknown",
            "top_error_count": result["top_error"][1] if result["top_error"] else 0,
            "by_type": result["error_breakdown"],
        }

    async def find_batchable_requests(
        self,
        tenant_id: str,
        lookback_days: int,
    ) -> Dict:
        """Find sequential requests that could be batched."""
        query = """
            WITH sequential_requests AS (
                SELECT
                    session_id,
                    count() as request_count,
                    sum(duration_ms) as total_duration_ms
                FROM agent_traces
                WHERE tenant_id = %(tenant_id)s
                  AND start_time >= now() - INTERVAL %(days)s DAY
                  AND is_streaming = false
                GROUP BY session_id
                HAVING request_count >= 3
            )
            SELECT
                sum(request_count) as batchable_count,
                sum(total_duration_ms) / 1000 / 3600 as latency_hours
            FROM sequential_requests
        """

        result = await self.db.query_one(query, {
            "tenant_id": tenant_id,
            "days": lookback_days,
        })

        monthly_factor = 30 / lookback_days

        return {
            "count": result["batchable_count"],
            "latency_savings_hours": result["latency_hours"] * monthly_factor * 0.5,  # 50% latency reduction
        }
```

## Implementation Architecture

### Cost Attribution Data Flow

```
┌─────────────────┐
│  LLM Request    │
│  (SDK/Sidecar)  │
└────────┬────────┘
         │ 1. Execute request
         │ 2. Record token usage
         ▼
┌─────────────────────────┐
│  Span Processor         │
│  ─────────────────────  │
│  • Extract tokens       │
│  • Load rate card       │
│  • Calculate cost       │
│  • Attach attributes    │
└────────┬────────────────┘
         │ 3. Enriched span with cost
         ▼
┌─────────────────────────┐
│  ClickHouse Ingestion   │
│  ─────────────────────  │
│  • agent_traces table   │
│  • Real-time inserts    │
└────────┬────────────────┘
         │ 4. Materialized views aggregate
         ▼
┌──────────────────────────────────┐
│  cost_attribution_mv             │
│  ──────────────────────────────  │
│  • Hourly aggregation            │
│  • Multi-dimensional grouping    │
│  • Sum tokens, costs, requests   │
└────────┬─────────────────────────┘
         │ 5. Financial queries
         ▼
┌──────────────────────────────────┐
│  Dashboards & Reports            │
│  ──────────────────────────────  │
│  • Team chargeback               │
│  • User showback                 │
│  • Feature ROI                   │
│  • Budget alerts                 │
│  • Cost forecasts                │
└──────────────────────────────────┘
```

### Integration Points

1. **Rate Card Hot Reload**: Background task checks for config changes every 60 seconds
2. **Span Enrichment**: Cost calculation happens during span processing before storage
3. **Budget Monitoring**: Scheduled job (every 15 minutes) checks budget thresholds
4. **Forecast Updates**: Daily job generates cost forecasts for all tenants
5. **Optimization Analysis**: Weekly job identifies cost optimization opportunities

## Key Takeaways

> **Cost Attribution Essentials**
>
> 1. **Hot-Reloadable Rate Cards**: YAML-based rate configuration enables rapid response to pricing changes without service restarts
>
> 2. **Per-Request Cost Tagging**: Every LLM span includes calculated cost attributes based on token usage and current rate cards
>
> 3. **Multi-Dimensional Tracking**: Costs tracked across 5 key dimensions - request, session, user, team, and feature - enabling both granular control and strategic analysis
>
> 4. **ClickHouse Materialized Views**: Pre-aggregated cost data enables sub-second financial queries across billions of requests
>
> 5. **Real-Time Budget Alerts**: Threshold-based alerts with configurable actions (warn/throttle/block) prevent budget overruns
>
> 6. **Trend-Based Forecasting**: Linear regression with confidence intervals provides month-end cost projections after 3+ days of data
>
> 7. **Automated Optimization Insights**: ML-powered detection identifies model downgrade opportunities, caching candidates, and prompt optimization potential
>
> 8. **Chargeback & Showback**: Team-level billing allocation and user-level usage transparency create accountability
>
> 9. **Feature-Level ROI**: Linking costs to business features enables data-driven product decisions
>
> 10. **Enterprise Pricing Overrides**: Org-specific rate cards support volume discounts and custom pricing agreements

**Related Documentation:**
- [Core Metrics](../03-core-platform/core-metrics.md) - Token usage metrics and tracking
- [Data Pipeline](../02-architecture/data-pipeline.md) - ClickHouse materialized views and aggregations
- [Span Hierarchy](../03-core-platform/span-hierarchy.md) - Span attributes and context propagation
- [Multi-Tenancy](../05-security-compliance/multi-tenancy.md) - Tenant isolation for cost attribution
