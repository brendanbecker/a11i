---
title: "Glossary: AI Agent Observability Terms"
category: "Reference"
tags: ["glossary", "terminology", "definitions", "reference"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "/docs/03-core-platform/core-metrics.md"
  - "/docs/03-core-platform/streaming-handling.md"
  - "/docs/05-security-compliance/pii-redaction.md"
  - "/docs/02-architecture/technology-stack.md"
---

# Glossary: AI Agent Observability Terms

## Table of Contents

- [Overview](#overview)
- [A](#a)
- [B-C](#b-c)
- [D-G](#d-g)
- [H-L](#h-l)
- [M-O](#m-o)
- [P](#p)
- [Q-R](#q-r)
- [S](#s)
- [T](#t)
- [U-Z](#u-z)
- [Key Takeaways](#key-takeaways)

## Overview

This glossary provides comprehensive definitions of key terms used throughout the a11i platform documentation. Terms are organized alphabetically and include cross-references to related concepts and documentation.

**Usage Notes:**
- **Bold terms** indicate primary definitions
- *Italicized terms* are synonyms or related concepts
- See also: links point to related glossary entries
- → indicates cross-references to detailed documentation

---

## A

### Agent Loop
The iterative execution pattern of autonomous AI agents following the **Think-Act-Observe** cycle. An agent loop consists of:

1. **Think Phase**: Reasoning about the current state and planning next actions
2. **Act Phase**: Executing selected tools or generating responses
3. **Observe Phase**: Processing results and updating internal state
4. **Decide Phase**: Determining whether the goal is achieved or iteration should continue

**Example**: A research agent might think about search strategies, act by calling a search API, observe the results, and decide whether more research is needed.

**See also**: Think-Act-Observe Pattern, Loop Velocity, Agent Loop Iteration

**Metrics**: `a11i.agent.loop_iteration`, `a11i.agent.loop_phase`, `a11i.agent.loop_complete`

→ Related: `/docs/03-core-platform/opentelemetry-integration.md`

---

### Agentic Systems
Software architectures where autonomous AI agents make independent decisions, use tools, and collaborate to achieve complex goals without explicit step-by-step instructions. Unlike traditional software with deterministic execution paths, agentic systems operate probabilistically based on context, memory, and reasoning.

**Characteristics**:
- Autonomous decision-making
- Tool/function calling capabilities
- Memory and state management
- Multi-step reasoning chains
- Collaboration between agents

**Example**: An agentic customer service system where multiple specialized agents (routing, knowledge retrieval, response generation) collaborate to resolve customer inquiries.

**See also**: Multi-Agent Systems, Agent Loop

→ Related: `/docs/01-overview/executive-summary.md`

---

### Agent Loop Iteration
A single cycle through the Think-Act-Observe-Decide pattern. Tracked via the `a11i.agent.loop_iteration` attribute (integer starting at 1).

**See also**: Agent Loop, Loop Velocity

---

## B-C

### Chain of Thought (CoT)
A reasoning technique where LLMs verbalize intermediate steps before arriving at a final answer, improving performance on complex reasoning tasks. In a11i, CoT traces are captured as sequential spans showing the complete reasoning process.

**Example**:
```
User: "What is 27 * 34?"

Chain of Thought:
1. "Let me break this down: 27 * 34"
2. "First, 27 * 30 = 810"
3. "Then, 27 * 4 = 108"
4. "Adding: 810 + 108 = 918"

Final Answer: "918"
```

**See also**: Think-Act-Observe Pattern, Hallucination

**Metrics**: Captured via nested spans with `gen_ai.operation.name` = "chat"

---

### ClickHouse
Columnar OLAP database providing high-compression storage (10x-20x) and fast analytical queries for a11i's trace and metrics data. Selected for 92% compression ratios, sub-second query performance, and ability to handle 300M+ spans/day on a single instance.

**Key Features**:
- Columnar compression (MergeTree engine)
- Materialized views for pre-aggregation
- Distributed tables with sharding/replication
- Native OpenTelemetry Collector exporter
- TTL-based retention management

**See also**: Storage Layer, OLAP, Time-Series Database

→ Related: `/docs/02-architecture/technology-stack.md#layer-3-storage-layer`

---

### Context Saturation
The percentage of an LLM's context window currently occupied by tokens. Measured as:

```
context_saturation = tokens_used / context_window_size
```

**Critical Thresholds**:
- **< 50%**: Healthy - plenty of headroom
- **50-80%**: Warning - monitor closely
- **> 80%**: Critical - degraded performance likely
- **> 95%**: Emergency - context overflow imminent

**Example**: GPT-4 Turbo has a 128K context window. Using 96K tokens = 75% saturation (warning threshold).

**See also**: Context Window, Token, RAG

**Metrics**: `a11i.context.saturation` (gauge 0.0-1.0), `a11i.context.tokens_used`, `a11i.context.tokens_remaining`

→ Related: `/docs/03-core-platform/core-metrics.md`

---

### Context Window
The maximum number of tokens an LLM can process in a single request, including both input (prompt) and output (completion). Context windows vary by model:

| Model | Context Window | Use Case |
|-------|---------------|----------|
| GPT-3.5 Turbo | 16,384 tokens | Short conversations |
| GPT-4 Turbo | 128,000 tokens | Long documents, complex reasoning |
| Claude 3 Opus | 200,000 tokens | Very long context |
| Gemini 1.5 Pro | 1,000,000 tokens | Extreme long context |

**Implications**:
- Exceeding context window causes request failure
- Larger windows = higher costs and slower inference
- Context management critical for long-running agents

**See also**: Context Saturation, Token, RAG

---

## D-G

### E2E Latency (End-to-End Latency)
Total time from sending a request to receiving the complete response. For streaming responses:

```
E2E Latency = TTFT + (ITL_avg × (token_count - 1))
```

**Example**:
- TTFT: 500ms
- Average ITL: 25ms
- Output tokens: 100
- E2E = 500 + (25 × 99) = 2,975ms (~3 seconds)

**See also**: TTFT, ITL, TPOT

→ Related: `/docs/03-core-platform/streaming-handling.md#streaming-metrics`

---

### GenAI Semantic Conventions
OpenTelemetry standard attributes for instrumenting generative AI applications, introduced in OTel v1.38+. Defines standard naming for LLM operations, models, tokens, and costs.

**Core Attributes**:
- `gen_ai.system`: Provider name (e.g., "openai", "anthropic")
- `gen_ai.request.model`: Model requested
- `gen_ai.usage.input_tokens`: Prompt token count
- `gen_ai.usage.output_tokens`: Completion token count
- `gen_ai.operation.name`: Operation type ("chat", "invoke_agent", etc.)

**See also**: OpenTelemetry, OTLP, Span

→ Related: `/docs/03-core-platform/opentelemetry-integration.md#genai-semantic-conventions`

---

## H-L

### Hallucination
When an LLM generates factually incorrect information with high confidence. Hallucinations are a critical failure mode for AI agents as they can provide incorrect answers, fabricate data, or misrepresent facts without any indication of uncertainty.

**Types**:
- **Factual hallucinations**: Incorrect facts ("Paris is the capital of Germany")
- **Temporal hallucinations**: Wrong dates or sequences
- **Reasoning hallucinations**: Logical errors that appear sound
- **Source hallucinations**: Citations to non-existent sources

**Detection Strategies**:
- External fact verification
- Consistency checking across multiple responses
- Confidence scoring and uncertainty quantification
- Human-in-the-loop validation for critical decisions

**See also**: Chain of Thought, Silent Divergence

---

### ITL (Inter-Token Latency)
Time elapsed between consecutive tokens in a streaming LLM response. Measured in milliseconds and typically ranges from 10-100ms depending on model size, load, and infrastructure.

**Formula**:
```
ITL[i] = timestamp[i] - timestamp[i-1]
```

**Statistical Measures**:
- **Average ITL**: Mean time between tokens
- **P95 ITL**: 95th percentile (smoothness indicator)
- **Max ITL**: Longest gap (stall detection)

**Alert Thresholds**:
- Normal: < 100ms average
- Warning: > 200ms average
- Critical: > 500ms P95 (indicates stalls or throttling)

**Example**:
```
Token timestamps: [50ms, 73ms, 96ms, 119ms, 145ms]
ITL values:       [23ms, 23ms, 23ms, 26ms]
Average ITL:      23.75ms
```

**See also**: TTFT, TPOT, E2E Latency

**Metrics**: `gen_ai.itl.avg_ms`, `gen_ai.itl.p95_ms`, `gen_ai.itl.max_ms`

→ Related: `/docs/03-core-platform/streaming-handling.md#streaming-metrics`

---

### LLM (Large Language Model)
Neural network-based AI model trained on vast amounts of text data to understand and generate human-like language. LLMs power modern AI agents through capabilities like reasoning, code generation, and tool use.

**Common LLM Providers**:
- **OpenAI**: GPT-4, GPT-3.5 Turbo, GPT-4o
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku), Claude Sonnet 4.5
- **Google**: Gemini, PaLM
- **AWS Bedrock**: Hosted models from multiple providers
- **Azure AI**: OpenAI models via Azure infrastructure

**Key Characteristics**:
- **Scale**: Billions to trillions of parameters
- **Emergent abilities**: Reasoning, tool use appear at scale
- **Cost**: Pay-per-token pricing model
- **Latency**: 50ms-3s depending on model and load

**See also**: Token, Tokenization, TTFT, Context Window

---

### Loop Velocity
The rate at which an agent completes Think-Act-Observe iterations, measured in iterations per second or average iteration duration. Critical metric for detecting infinite loops or stuck agents.

**Formula**:
```
loop_velocity = iterations_completed / elapsed_time_seconds
```

**Alert Conditions**:
- **Zero velocity**: Agent stuck in infinite loop
- **Extremely high velocity**: Agent cycling without making progress
- **Decreasing velocity**: Performance degradation over time

**Example**:
```
Agent completes 5 iterations in 30 seconds
Loop velocity = 5 / 30 = 0.167 iterations/second
Average iteration duration = 6 seconds
```

**See also**: Agent Loop, Agent Loop Iteration

**Metrics**: `a11i.loop_velocity_gauge` (iterations/second)

→ Related: `/docs/03-core-platform/core-metrics.md`

---

## M-O

### NATS JetStream
Cloud-native message queue system providing high-throughput, low-latency streaming for a11i's telemetry pipeline. Selected as default queue for Kubernetes deployments due to <1ms latency, operational simplicity, and horizontal scalability.

**Key Features**:
- Sub-millisecond message delivery latency
- Built-in persistence with configurable retention
- No external dependencies (no ZooKeeper)
- Kubernetes Operator for native deployments
- At-most-once, at-least-once, and exactly-once semantics

**Comparison with Kafka**:

| Feature | NATS JetStream | Apache Kafka |
|---------|---------------|--------------|
| Latency | <1ms | 2-10ms |
| Throughput | Moderate (10M msg/s) | Very High (100M+ msg/s) |
| Ops Complexity | Low | High |
| Dependencies | None | ZooKeeper |

**See also**: Message Queue, Apache Kafka, Streaming

→ Related: `/docs/02-architecture/technology-stack.md#layer-2-message-queue-layer`

---

### OLAP (Online Analytical Processing)
Database architecture optimized for analytical queries over large datasets, as opposed to OLTP (Online Transaction Processing) optimized for transactional operations. ClickHouse is a11i's OLAP database for trace and metrics storage.

**OLAP Characteristics**:
- Columnar storage for fast aggregations
- High compression ratios (10x-20x)
- Optimized for read-heavy workloads
- Supports complex analytical queries

**OLTP Characteristics** (for comparison):
- Row-based storage for fast writes
- Low latency for individual records
- ACID transaction support
- Optimized for concurrent updates

**See also**: ClickHouse, Storage Layer, PostgreSQL

---

### OpenLLMetry
Open-source (Apache 2.0) instrumentation SDK by Traceloop providing automatic OpenTelemetry tracing for LLM applications. Foundation for a11i's instrumentation layer, extended with agent-specific semantics.

**Auto-Instrumented Frameworks**:
- OpenAI, Anthropic, Cohere, Google AI
- LangChain, LangGraph, LlamaIndex
- Vector databases (Pinecone, Weaviate, Qdrant, ChromaDB)
- Bedrock, Azure AI, Hugging Face

**Captured Attributes**:
- Token counts (prompt, completion, total)
- Model parameters (temperature, top_p, etc.)
- Latency breakdown
- Cost estimation
- Request/response metadata

**See also**: OpenTelemetry, GenAI Semantic Conventions, Instrumentation

→ Related: `/docs/02-architecture/technology-stack.md#layer-4-instrumentation-layer`

---

### OpenTelemetry (OTel)
Vendor-neutral, open-source observability framework providing standardized APIs, SDKs, and protocols for collecting distributed traces, metrics, and logs. a11i is built on OpenTelemetry to ensure interoperability and avoid vendor lock-in.

**Core Components**:
- **API**: Language-agnostic specification
- **SDK**: Implementation libraries (Python, Go, Java, JS, etc.)
- **Protocol (OTLP)**: Wire format for telemetry data
- **Collector**: Agent for receiving, processing, and exporting telemetry
- **Semantic Conventions**: Standard naming for attributes

**Benefits for a11i**:
- Vendor neutrality (works with any backend)
- Ecosystem compatibility (Jaeger, Prometheus, Grafana, etc.)
- Community-driven standards (GenAI conventions)
- Future-proof architecture

**See also**: OTLP, Span, Trace, GenAI Semantic Conventions

→ Related: `/docs/03-core-platform/opentelemetry-integration.md`

---

### OTLP (OpenTelemetry Protocol)
Standard wire protocol for transmitting telemetry data between OpenTelemetry SDKs and collectors/backends. Supports both gRPC (default) and HTTP transport.

**Protocol Versions**:
- **OTLP/gRPC**: Port 4317, binary protocol, lower latency
- **OTLP/HTTP**: Port 4318, JSON or protobuf, firewall-friendly

**Data Types**:
- Traces (spans with hierarchical relationships)
- Metrics (gauges, counters, histograms)
- Logs (structured log records)

**See also**: OpenTelemetry, Collector, Span

---

## P

### Passthrough-with-Tapping
Architectural pattern for observing streaming LLM responses without introducing latency. The pattern immediately forwards each chunk to the client (passthrough) while simultaneously buffering for telemetry (tapping) in a non-blocking side channel.

**Mechanism**:
```python
async for chunk in llm_stream:
    buffer.append(chunk)  # Non-blocking tap
    yield chunk          # Immediate passthrough
```

**Performance Characteristics**:
- Added latency: <100 nanoseconds per chunk
- TTFT impact: Unmeasurable (0.00016%)
- Memory overhead: ~15MB per stream

**See also**: Streaming, TTFT, ITL

→ Related: `/docs/03-core-platform/streaming-handling.md#passthrough-with-tapping-pattern`

---

### PII (Personally Identifiable Information)
Information that can be used to identify, contact, or locate an individual. a11i implements edge-based PII redaction to ensure sensitive data never leaves customer infrastructure.

**Common PII Types Detected**:
- **Personal Identifiers**: SSN, passport numbers, driver's licenses
- **Contact Information**: Email addresses, phone numbers, physical addresses
- **Financial Data**: Credit card numbers, bank accounts, tax IDs
- **Medical Data**: Medical record numbers, health plan IDs
- **Network Data**: IP addresses, MAC addresses
- **Biometric Data**: Fingerprints, facial recognition data

**Redaction Strategies**:
- **Masking**: Replace with `****` or `<REDACTED>`
- **Pseudonymization**: Replace with consistent fake values (e.g., `User_001`)
- **Hashing**: One-way hash for deterministic anonymization
- **Removal**: Delete entirely from telemetry

**Compliance Regulations**:
- GDPR (EU): Right to erasure, data minimization
- HIPAA (US): Protected Health Information (PHI) safeguards
- CCPA (California): Consumer data privacy rights
- SOC 2: Data handling controls

**See also**: Presidio, Pseudonymization, Compliance

**Metrics**: `gen_ai.pii_detected` (boolean), `gen_ai.output.redacted` (boolean)

→ Related: `/docs/05-security-compliance/pii-redaction.md`

---

### Presidio
Microsoft's open-source (MIT license) PII detection and anonymization framework combining Named Entity Recognition (NER) and regex pattern matching. a11i uses Presidio for ML-powered PII redaction at the edge.

**Detection Capabilities**:
- **NER-based**: Person names, locations, organizations
- **Pattern-based**: SSN, credit cards, emails, phone numbers
- **Custom recognizers**: Domain-specific PII patterns
- **Confidence scoring**: Tunable thresholds (0.0-1.0)

**Anonymization Strategies**:
- Redaction (removal)
- Replacement with placeholders
- Hashing (deterministic)
- Encryption (reversible)

**Performance**:
- Latency: 5-20ms per span (typical)
- Supported languages: English (primary), 20+ others
- Accuracy: >95% precision on structured PII

**See also**: PII, Redaction, NER

→ Related: `/docs/05-security-compliance/pii-redaction.md#microsoft-presidio-integration`

---

### Pseudonymization
Technique of replacing PII with consistent, realistic-looking fake values that preserve debugging context while protecting privacy. Unlike masking (which removes all meaning), pseudonymization maintains relationships and patterns.

**Example**:
```
Original:  "Alice (alice@acme.com) called Bob (555-1234)"
Masked:    "<REDACTED> (<REDACTED>) called <REDACTED> (<REDACTED>)"
Pseudonymized: "User_001 (user1@redacted.example.com) called User_002 (+1-555-0002)"
```

**Benefits**:
- Maintains debugging context and relationships
- Deterministic (same PII → same pseudonym across sessions)
- Compliant with GDPR when properly salted
- Enables pattern analysis without exposing real data

**Implementation**:
```python
pseudonym = HMAC-SHA256(original_value, secret_salt)
consistent_id = hash_to_readable_format(pseudonym, entity_type)
```

**See also**: PII, Masking, Hashing

→ Related: `/docs/05-security-compliance/pii-redaction.md#pseudonymization-patterns`

---

## Q-R

### RAG (Retrieval-Augmented Generation)
Technique combining LLM generation with external knowledge retrieval. Instead of relying solely on the LLM's training data, RAG systems query vector databases or search engines to inject relevant context into prompts.

**Architecture**:
```
User Query → Embedding → Vector Search → Retrieved Docs →
→ Construct Prompt → LLM → Generated Response
```

**Benefits**:
- Access to up-to-date information (beyond training cutoff)
- Domain-specific knowledge integration
- Reduced hallucinations through grounding in facts
- Lower costs vs. fine-tuning large models

**Observability Challenges**:
- Context window consumption from retrieved docs
- Quality of retrieval (relevance, diversity)
- Token efficiency (how much context is useful)
- Multi-hop reasoning across documents

**See also**: Context Window, Context Saturation, Vector Database

**Metrics**: `a11i.context.retrieval_count`, `a11i.context.retrieval_quality`

---

## S

### Silent Divergence
Failure mode where an AI agent completes successfully (no errors) but produces incorrect or harmful outputs. Unlike traditional software failures (500 errors, exceptions), silent divergence provides no explicit signal that something went wrong.

**Examples**:
- Agent retrieves irrelevant documents but generates confident response
- Planning agent selects suboptimal strategy without indication
- Tool calls use incorrect parameters but return valid-looking results
- Multi-agent workflow arrives at wrong conclusion through flawed reasoning

**Detection Strategies**:
- Output validation against expected patterns
- Consistency checking across multiple runs
- Human-in-the-loop verification
- Automated quality scoring
- Anomaly detection on agent behavior patterns

**See also**: Hallucination, Agent Loop, Observability

---

### Span
Fundamental unit of distributed tracing representing a single operation in a workflow. In OpenTelemetry, spans form tree structures (traces) showing parent-child relationships between operations.

**Span Attributes**:
- **Identifiers**: `trace_id`, `span_id`, `parent_span_id`
- **Timing**: `start_time`, `end_time`, `duration`
- **Metadata**: `span_name`, `span_kind`, `attributes`
- **Status**: `status_code` (OK, ERROR), `status_message`

**Span Hierarchy Example**:
```
Trace: Agent Execution
├─ Span: agent_think (200ms)
│  └─ Span: llm_call (180ms)
├─ Span: agent_act (500ms)
│  ├─ Span: tool_search (450ms)
│  └─ Span: parse_results (50ms)
└─ Span: agent_observe (150ms)
   └─ Span: llm_call (130ms)
```

**GenAI Span Attributes**:
- `gen_ai.system`: "openai"
- `gen_ai.request.model`: "gpt-4-turbo-2024-04-09"
- `gen_ai.usage.input_tokens`: 500
- `gen_ai.usage.output_tokens`: 150
- `a11i.cost.estimate_usd`: 0.0042

**See also**: Trace, OpenTelemetry, Distributed Tracing

→ Related: `/docs/03-core-platform/opentelemetry-integration.md`

---

### Streaming
Incremental delivery of LLM responses token-by-token as they are generated, rather than waiting for the complete response. Streaming significantly improves perceived latency and user experience.

**Protocol**: Server-Sent Events (SSE) or custom JSON streaming

**Benefits**:
- Reduced perceived latency (TTFT vs E2E)
- Improved user experience (immediate feedback)
- Ability to interrupt long generations
- Progressive rendering of responses

**Observability Challenges**:
- Complete response unknown until stream ends
- PII patterns may span chunk boundaries
- Token counts only available at completion
- Must not add latency to hot path

**See also**: TTFT, ITL, Passthrough-with-Tapping

→ Related: `/docs/03-core-platform/streaming-handling.md`

---

## T

### Think-Act-Observe Pattern
Core architectural pattern for autonomous AI agents describing the iterative decision-making cycle:

**Phases**:

1. **Think**: Agent reasons about current state and plans next action(s)
   - Analyze current context and goals
   - Consider available tools and strategies
   - Plan action sequence

2. **Act**: Agent executes selected action (LLM call, tool invocation)
   - Call external tools or APIs
   - Generate responses
   - Modify state

3. **Observe**: Agent processes action results and updates state
   - Parse tool outputs
   - Update working memory
   - Evaluate progress toward goal

4. **Decide** (implicit): Determine whether to continue iterating or complete

**Example Trace**:
```
Think:   "Need to find weather for Paris"
         [LLM reasoning: 150ms]
Act:     Call weather_api(city="Paris")
         [Tool execution: 450ms]
Observe: Parse result: "Sunny, 22°C"
         [LLM synthesis: 100ms]
Decide:  Goal achieved (weather retrieved)
```

**See also**: Agent Loop, Chain of Thought, Agentic Systems

**Metrics**: `a11i.agent.loop_phase` (enum: think, act, observe, decide)

---

### Token
Basic unit of text processed by LLMs. Tokens can represent whole words, parts of words, or individual characters depending on the tokenization algorithm.

**Tokenization Examples** (using GPT-4's cl100k_base encoding):
- "Hello" → 1 token
- "Hello, world!" → 4 tokens ["Hello", ",", " world", "!"]
- "Tokenization" → 3 tokens ["Token", "ization"]
- "1234567890" → 3 tokens ["1234", "5678", "90"]

**Rough Estimates**:
- English: ~4 characters per token (0.75 tokens per word)
- Code: ~2-3 characters per token
- Non-English languages: Varies (often less efficient)

**Cost Implications**:
- LLM pricing based on token consumption
- Input tokens typically cheaper than output tokens
- Context window limits measured in tokens

**See also**: Tokenization, Context Window, TPOT

→ Related: `/docs/04-implementation/tokenization.md`

---

### Tokenization
Process of converting text into tokens that LLMs can process. Different models use different tokenization algorithms (BPE, WordPiece, SentencePiece, Unigram).

**Common Tokenizers**:
- **tiktoken** (OpenAI): GPT-3.5, GPT-4, GPT-4o
  - `cl100k_base`: GPT-4, GPT-3.5-turbo
  - `o200k_base`: GPT-4o
- **SentencePiece** (Google): LLaMA, Mistral, Falcon
- **HuggingFace Transformers**: Universal fallback

**Why It Matters for Observability**:
- Accurate token counting for cost attribution
- Context window management and saturation alerts
- Performance metrics (TPOT, throughput)

**Example**:
```python
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode("Hello, world!")
print(f"Token count: {len(tokens)}")  # 4
print(f"Tokens: {tokens}")  # [9906, 11, 1917, 0]
```

**See also**: Token, Context Window, tiktoken

→ Related: `/docs/04-implementation/tokenization.md`

---

### TPOT (Time Per Output Token)
Average time to generate a single output token, calculated by dividing total generation time by number of output tokens. TPOT accounts for both TTFT and streaming speed, providing a holistic measure of generation efficiency.

**Formula**:
```
TPOT = total_duration_ms / output_token_count
```

**Example**:
```
Request to completion: 2,400ms
Output tokens: 120
TPOT = 2400 / 120 = 20ms per token
```

**Interpretation**:
- **< 30ms**: Excellent (fast model, efficient infrastructure)
- **30-50ms**: Good (typical for cloud providers)
- **50-100ms**: Acceptable (may indicate load)
- **> 100ms**: Poor (investigate throttling or capacity issues)

**Comparison with TTFT and ITL**:
- TTFT: Time to first token (user's initial wait)
- ITL: Time between consecutive tokens (smoothness)
- TPOT: Average efficiency across entire response

**See also**: TTFT, ITL, E2E Latency

**Metrics**: `gen_ai.tpot_ms` (float, milliseconds per token)

→ Related: `/docs/03-core-platform/streaming-handling.md#streaming-metrics`

---

### Trace
Collection of related spans forming a complete workflow execution path. Traces show the hierarchical relationship between operations, enabling end-to-end visibility into distributed systems.

**Trace Structure**:
```
Trace ID: 4bf92f3577b34da6a3ce929d0e0e4736
├─ Root Span: agent_execution (2.5s)
│  ├─ Child Span: agent_think (200ms)
│  │  └─ Child Span: llm_call_planning (180ms)
│  ├─ Child Span: agent_act (500ms)
│  │  ├─ Child Span: tool_search (450ms)
│  │  └─ Child Span: parse_results (50ms)
│  └─ Child Span: agent_observe (150ms)
│     └─ Child Span: llm_call_synthesis (130ms)
```

**W3C Trace Context Propagation**:
```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             ││ └─────────── trace_id ────────────┘ └── span_id ──┘ └flags
             └version
```

**See also**: Span, OpenTelemetry, Distributed Tracing

---

### TTFT (Time to First Token)
Critical user experience metric measuring the elapsed time from sending a request to receiving the first token of the response. TTFT is the primary indicator of perceived responsiveness in streaming LLM applications.

**Formula**:
```
TTFT = first_chunk_arrival_time - request_sent_time
```

**Target Values**:
- **Excellent**: < 500ms
- **Good**: < 1,000ms
- **Acceptable**: < 2,000ms
- **Poor**: > 3,000ms

**Factors Affecting TTFT**:
- Model size (larger models = longer TTFT)
- Queue time (provider load and capacity)
- Cold starts (first request after idle period)
- Prompt length (more tokens to process)
- Network latency (geographic distance)

**Example**:
```
Request sent:     0ms
First chunk received: 487ms
TTFT = 487ms (Good)
```

**a11i Guarantee**: Passthrough-with-tapping adds <1ms to TTFT (unmeasurable in production)

**See also**: ITL, TPOT, E2E Latency, Streaming

**Metrics**: `gen_ai.ttft_ms` (float, milliseconds)

→ Related: `/docs/03-core-platform/streaming-handling.md#streaming-metrics`

---

## U-Z

### Vector Database
Specialized database optimized for storing and querying high-dimensional vector embeddings, commonly used in RAG systems for semantic search.

**Common Vector Databases**:
- **Pinecone**: Fully managed, cloud-native
- **Weaviate**: Open-source, GraphQL API
- **Qdrant**: Rust-based, high performance
- **ChromaDB**: Embedded, developer-friendly
- **Milvus**: Scalable, enterprise-focused

**Use Cases in AI Agents**:
- Semantic search over knowledge bases
- Long-term memory retrieval
- Document similarity matching
- Context injection for RAG

**See also**: RAG, Embeddings, Context Window

---

### Windowed Buffer Scanning
Technique for detecting PII patterns that span multiple chunks in streaming responses. Maintains a rolling window of recent content to catch patterns split across chunk boundaries.

**Mechanism**:
```python
window = ""  # Rolling buffer
for chunk in stream:
    window += chunk
    if len(window) > window_size * 2:
        window = window[-window_size:]  # Trim to max size

    # Scan window for PII
    pii_detected = scan_patterns(window)
```

**Buffer Size Recommendations**:
- **64 chars**: Short PII (email, phone)
- **256 chars**: Recommended default (multi-line patterns)
- **512 chars**: High-security applications

**Example**:
```
Chunk 1: "My credit card is 4532"
Chunk 2: "-0123-4567-"
Chunk 3: "8901"

Window buffer: "4532-0123-4567-8901"
Pattern detected: CREDIT_CARD
```

**See also**: PII, Streaming, Presidio

→ Related: `/docs/05-security-compliance/pii-redaction.md#windowed-buffer-scanning-for-streaming`

---

## Key Takeaways

> **Essential Terminology for a11i**
>
> **Core Agent Concepts**:
> - **Agent Loop**: Think-Act-Observe-Decide cycle of autonomous agents
> - **Agentic Systems**: Probabilistic, autonomous decision-making vs deterministic software
> - **Chain of Thought**: Verbalized reasoning improving performance on complex tasks
>
> **Performance Metrics**:
> - **TTFT**: Time to first token - primary UX metric (<1s target)
> - **ITL**: Inter-token latency - smoothness of generation (<100ms target)
> - **TPOT**: Time per output token - efficiency measure (<50ms target)
> - **Loop Velocity**: Agent iteration rate - infinite loop detection
>
> **Context Management**:
> - **Context Window**: Maximum tokens processable by LLM (16K-1M)
> - **Context Saturation**: % of window consumed (alert at >80%)
> - **RAG**: Retrieval-Augmented Generation for knowledge injection
>
> **Privacy & Security**:
> - **PII**: Personally Identifiable Information requiring redaction
> - **Presidio**: Microsoft's ML-powered PII detection framework
> - **Pseudonymization**: Replacing PII with consistent fake values for debugging
> - **Windowed Buffer**: Scanning technique for streaming PII detection
>
> **Technical Infrastructure**:
> - **OpenTelemetry**: Vendor-neutral observability standard
> - **GenAI Semantic Conventions**: Standard LLM tracing attributes
> - **Passthrough-with-Tapping**: Zero-latency streaming observation pattern
> - **ClickHouse**: OLAP database with 92% compression for traces
> - **NATS JetStream**: <1ms latency message queue for telemetry
>
> **Failure Modes**:
> - **Hallucination**: Confident but factually incorrect outputs
> - **Silent Divergence**: Completing successfully with wrong results
> - **Context Saturation**: Degraded performance from full context windows
> - **Infinite Loops**: Agents cycling without progress

---

**Related Documentation:**
- [Core Metrics](/home/becker/projects/a11i/docs/03-core-platform/core-metrics.md) - Metric definitions and calculations
- [Streaming Handling](/home/becker/projects/a11i/docs/03-core-platform/streaming-handling.md) - TTFT, ITL, TPOT details
- [OpenTelemetry Integration](/home/becker/projects/a11i/docs/03-core-platform/opentelemetry-integration.md) - GenAI semantic conventions
- [PII Redaction](/home/becker/projects/a11i/docs/05-security-compliance/pii-redaction.md) - Presidio and privacy implementation
- [Technology Stack](/home/becker/projects/a11i/docs/02-architecture/technology-stack.md) - Infrastructure components

---

*Document Status: Draft | Last Updated: 2025-11-26 | Maintained by: a11i Documentation Team*
