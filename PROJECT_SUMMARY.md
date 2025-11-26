# Project Summary: a11i (Analyzabiliti)

**Status:** Design / RFC Phase
**Owner:** @brendanbecker

## Vision
**"Turn black-box agents into glass-box infrastructure."**

`a11i` (pronounced "Ally") stands for **Analyzabiliti**. It is a lightweight instrumentation layer designed to bring standard SRE observability practices to AI Agent workflows.

It treats "Agent Activity" as a metric stream, allowing platform engineers to monitor costs, latency, tool usage, and context saturation just like they monitor CPU and Memory.

## The Core Problem
* **The Black Box:** Once an agent starts running a task (like an `arx` runbook), SREs are blind. They don't know if it's looping, hallucinating, or burning through $50 of API credits until it finishes or fails.
* **Context Saturation:** Performance degrades as the context window fills up, but there is no "gauge" to warn operators when an agent is becoming "demented" due to overflow.
* **Tool Blindness:** We don't track how often agents fail to use tools correctly (e.g., syntax errors in generated JSON), which is a leading indicator of model drift.

## The Solution
`a11i` acts as an OTel-native (OpenTelemetry) proxy or middleware wrapper around your agent's LLM calls.

1.  **Intercept:** Captures request/response cycles transparently.
2.  **Measure:** Calculates token usage, cost, duration, and "Context Density."
3.  **Trace:** Spans every tool execution (e.g., `arx step`, `kubectl` call).
4.  **Export:** Pushes data to standard OTLP backends (Prometheus, Grafana, Honeycomb, Jaeger).

## Design Philosophy
1.  **OTel Native:** Do not invent a new format. AI traces should look just like microservice traces.
2.  **Low Overhead:** Minimal latency impact on the agent loop.
3.  **Privacy Aware:** Configurable redaction of prompt text (PII) while preserving the metrics (token counts/latency).
4.  **Drop-in Compatibility:** Works as a wrapper for standard Python/Node SDKs or as a local reverse proxy.

## Target Audience
* **Platform Teams:** deploying internal "DevBots" who need to chargeback costs to specific teams.
* **AI Engineers:** debugging complex agent loops where one bad tool output causes a cascade of failures.
* **SREs:** who want to set up PagerDuty alerts for "Agent Error Rate > 5%".

## Integration Map
* **arx:** `a11i` wraps the `arx` execution loop to provide traces of every runbook step.
* **SREcodex:** Tracks which specific Skills are most popular or most error-prone.
* **RaggedWiki:** Monitors retrieval quality (e.g., "How many chunks were retrieved but ignored?").
