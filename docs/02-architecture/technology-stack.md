---
title: Technology Stack
description: Comprehensive technology selection and decision rationale for a11i observability platform
category: Architecture
tags: [technology, stack, decisions, components]
version: 1.0.0
last_updated: 2025-11-26
status: draft
related_docs:
  - /docs/02-architecture/system-design.md
  - /docs/02-architecture/data-flow.md
  - /docs/03-components/proxy-gateway.md
  - /docs/03-components/message-queue.md
  - /docs/03-components/storage-layer.md
---

# Technology Stack

## Table of Contents

1. [Overview](#overview)
2. [Stack Architecture Diagram](#stack-architecture-diagram)
3. [Layer 1: Proxy/Gateway Layer](#layer-1-proxygateway-layer)
4. [Layer 2: Message Queue Layer](#layer-2-message-queue-layer)
5. [Layer 3: Storage Layer](#layer-3-storage-layer)
6. [Layer 4: Instrumentation Layer](#layer-4-instrumentation-layer)
7. [Layer 5: PII Redaction](#layer-5-pii-redaction)
8. [Layer 6: Tokenization Libraries](#layer-6-tokenization-libraries)
9. [Layer 7: Dashboards & Visualization](#layer-7-dashboards--visualization)
10. [Layer 8: Programming Languages](#layer-8-programming-languages)
11. [Complete Stack Summary](#complete-stack-summary)
12. [Key Takeaways](#key-takeaways)
13. [Migration Path](#migration-path)
14. [References](#references)

## Overview

The a11i observability platform requires a carefully selected technology stack that balances performance, operational complexity, ecosystem maturity, and cost. This document provides comprehensive decision matrices for each layer of the stack, comparing available options and providing clear recommendations with rationale.

### Design Principles

Our technology selection follows these core principles:

- **Cloud-Native First**: Kubernetes-optimized components with minimal operational overhead
- **Performance**: Sub-5ms latency for hot path, high throughput for data ingestion
- **Standards-Based**: OpenTelemetry compliance, industry-standard protocols
- **Operational Simplicity**: Prefer managed solutions and battle-tested components
- **Cost Efficiency**: Optimize for storage compression and compute efficiency
- **Flexibility**: Support for multiple LLM providers and deployment patterns

## Stack Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dashboard & Visualization                    │
│         Grafana (metrics/traces) + Custom React (UX)            │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│          Go/Python REST + GraphQL (agent-specific)               │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  ClickHouse (hot/warm) + S3/MinIO (cold) + PostgreSQL (meta)   │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    Message Queue Layer                           │
│         NATS JetStream (default) | Apache Kafka (scale)         │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                  OpenTelemetry Collector                         │
│        Custom Processors: PII Redaction, Token Counting         │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                     Proxy/Gateway Layer                          │
│   Go Custom (dev) → Envoy AI Gateway (production-ready)         │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                         SDK Layer                                │
│              Python SDK + TypeScript/Node SDK                    │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    Instrumentation Layer                         │
│         OpenLLMetry Foundation + a11i Extensions                 │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 1: Proxy/Gateway Layer

### Decision Matrix

| Component | Language | Latency | Features | Complexity | LLM Support | Maturity |
|-----------|----------|---------|----------|------------|-------------|----------|
| **Envoy AI Gateway** | C++ | <3ms | Comprehensive | Medium | Native | High |
| **LiteLLM** | Python | 5-15ms | Good | Low | Excellent | Medium |
| **Custom Go** | Go | <5ms | Flexible | Medium | Custom | N/A |
| **Custom Rust** | Rust | <2ms | Flexible | High | Custom | N/A |

### Detailed Comparison

#### Envoy AI Gateway (Recommended for Production)

**Strengths:**
- Purpose-built for LLM traffic patterns
- Native OpenTelemetry integration with semantic conventions
- Built-in support for OpenAI, Anthropic, Bedrock, Vertex AI
- Token-based rate limiting and quota management
- Sub-3ms internal latency overhead
- Model Context Protocol (MCP) support
- Smart routing with automatic fallback
- Active Envoy community and enterprise support

**Trade-offs:**
- Requires C++ expertise for custom extensions
- Configuration complexity for advanced features
- Larger binary size and memory footprint

**Use Cases:**
- Production deployments with multiple LLM providers
- Environments requiring advanced traffic management
- Teams with existing Envoy expertise

#### LiteLLM (Alternative for Simpler Deployments)

**Strengths:**
- Python-native universal gateway
- 100+ model provider support
- OpenAI-compatible API normalization
- Built-in cost tracking per request
- Can run as proxy server or embedded library
- Rapid development and iteration
- Excellent documentation

**Trade-offs:**
- Higher latency (5-15ms) due to Python overhead
- Limited horizontal scalability compared to Go/Rust
- Less mature for high-throughput production use

**Use Cases:**
- Development and testing environments
- Smaller deployments (<100 req/s)
- Teams prioritizing ease of use over performance

#### Custom Go Proxy (Recommended for Initial Development)

**Strengths:**
- Full control over implementation and business logic
- Go's goroutines ideal for concurrent streaming
- Mature OpenTelemetry libraries (go.opentelemetry.io)
- Strong standard library for HTTP/2 and SSE
- Easy to containerize and deploy in Kubernetes
- Fast compilation and iteration cycle

**Trade-offs:**
- Requires building and maintaining custom code
- Need to implement provider-specific adapters
- Ongoing maintenance burden

**Use Cases:**
- Early development phase with evolving requirements
- Domain-specific logic not available in off-the-shelf solutions
- Teams with Go expertise

#### Custom Rust Proxy

**Strengths:**
- Superior memory safety without garbage collection
- Zero-cost abstractions for maximum performance
- Excellent for CPU-bound transformations
- Growing async ecosystem (Tokio, async-std)

**Trade-offs:**
- Steeper learning curve and slower development
- Smaller ecosystem for LLM-specific libraries
- Longer compilation times

**Use Cases:**
- Extreme performance requirements (<1ms latency)
- Safety-critical applications
- Teams with Rust expertise

### Recommendation

**Phase 1 (Development):** Start with **Custom Go Proxy**
- Rapid iteration on a11i-specific features
- Learn exact requirements before committing to platform
- Build expertise in OpenTelemetry integration

**Phase 2 (Production):** Migrate to **Envoy AI Gateway**
- Proven performance and reliability
- Comprehensive LLM provider support
- Enterprise-grade traffic management
- Keep Go proxy for custom business logic not supported by Envoy

**Alternative Path:** Use **LiteLLM** if team is Python-focused and scale requirements are moderate (<100 req/s).

## Layer 2: Message Queue Layer

### Decision Matrix

| Queue | Latency | Throughput | Ops Complexity | K8s Native | Persistence | Best For |
|-------|---------|------------|----------------|------------|-------------|----------|
| **NATS JetStream** | <1ms | Moderate | Low | Yes | Yes | K8s, simplicity |
| **Apache Kafka** | 2-10ms | Very High | High | No | Yes | Enterprise scale |
| **Redpanda** | <5ms | High | Medium | Partial | Yes | Kafka compat |
| **RabbitMQ** | 1-5ms | Moderate | Medium | Partial | Yes | Traditional apps |

### Detailed Comparison

#### NATS JetStream (Recommended for K8s)

**Strengths:**
- Cloud-native design, Kubernetes Operator available
- Sub-millisecond latency for message delivery
- No JVM overhead, written in Go
- Built-in persistence with configurable retention
- Horizontal scalability through clustering
- Simple operational model (no ZooKeeper)
- Support for at-most-once, at-least-once, exactly-once semantics
- Native support for pub/sub, request/reply, queue groups

**Trade-offs:**
- Smaller ecosystem compared to Kafka
- Less mature tooling for stream processing
- Lower maximum throughput than Kafka

**Use Cases:**
- Kubernetes-native deployments
- Low-latency requirements (<5ms)
- Simpler operational requirements
- Small to medium scale (< 1M msg/s)

#### Apache Kafka (Enterprise Standard)

**Strengths:**
- Industry standard for high-scale streaming
- Rich ecosystem (Kafka Connect, KSQL, Schema Registry)
- Proven at massive scale (trillions of messages/day)
- Strong durability guarantees and replication
- Extensive monitoring and management tools
- Large community and vendor support

**Trade-offs:**
- High operational complexity (ZooKeeper dependency)
- JVM tuning and garbage collection management
- Heavier resource requirements
- Higher latency (2-10ms typical)

**Use Cases:**
- Enterprise-scale deployments (>1M msg/s)
- Complex stream processing requirements
- Organizations with existing Kafka expertise
- Multi-datacenter replication needs

#### Redpanda (Kafka-Compatible Alternative)

**Strengths:**
- Kafka API compatible, drop-in replacement
- C++ implementation, no JVM overhead
- No ZooKeeper dependency
- Lower resource usage than Kafka
- Simpler operational model
- Built-in schema registry

**Trade-offs:**
- Younger project, less battle-tested
- Smaller ecosystem and community
- Some Kafka ecosystem tools may not work

**Use Cases:**
- Teams wanting Kafka compatibility without operational overhead
- Cloud deployments where resource costs matter
- Migration from Kafka with operational pain

### Recommendation

**Default Choice:** **NATS JetStream**
- Best fit for Kubernetes-native architecture
- Lowest operational overhead
- Sufficient throughput for most observability workloads
- Excellent latency characteristics

**Enterprise Scale:** **Apache Kafka**
- When message volume exceeds 1M/s consistently
- When stream processing with KSQL is valuable
- When organization has existing Kafka infrastructure

**Migration Path:** Start with NATS JetStream. Design abstractions to allow migration to Kafka if scale demands it. The queue layer should be pluggable behind an interface.

## Layer 3: Storage Layer

### Decision Matrix

| Storage | Compression | Query Speed | Text Search | Cardinality | Cost | Best For |
|---------|-------------|-------------|-------------|-------------|------|----------|
| **ClickHouse** | 92% | Very Fast | Basic | Excellent | Low | Analytics |
| **TimescaleDB** | 70% | Fast | Via extension | Good | Medium | Moderate scale |
| **OpenSearch** | 50% | Moderate | Excellent | Poor | High | Search-heavy |
| **PostgreSQL** | 60% | Moderate | Good | Moderate | Medium | Metadata |

### Detailed Comparison

#### ClickHouse (Recommended for Traces/Spans)

**Strengths:**
- Columnar compression: 10x-20x reduction (92% typical)
- Vectorized query execution for aggregations
- Excellent for high-cardinality dimensions (user_id, trace_id)
- Native OpenTelemetry Collector exporter
- Unified storage for metrics, logs, traces
- Production proven: 300M spans/day on single instance
- Materialized views for pre-aggregation
- Distributed architecture with sharding/replication

**Trade-offs:**
- Limited UPDATE/DELETE performance (not OLTP)
- Basic full-text search capabilities
- Requires understanding of table engines and partitioning
- Eventually consistent in distributed mode

**Use Cases:**
- High-volume trace and span storage
- Time-series metrics aggregation
- Analytical queries over observability data
- Cost-sensitive deployments needing high compression

**Schema Example:**
```sql
CREATE TABLE spans (
    trace_id String,
    span_id String,
    parent_span_id String,
    name LowCardinality(String),
    timestamp DateTime64(9),
    duration_ns UInt64,
    attributes Map(String, String),
    resource_attributes Map(String, String),
    INDEX idx_trace_id trace_id TYPE bloom_filter GRANULARITY 1
) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (service_name, timestamp, trace_id)
TTL timestamp + INTERVAL 30 DAY;
```

#### TimescaleDB (PostgreSQL-Based Alternative)

**Strengths:**
- Built on PostgreSQL, familiar SQL semantics
- Automatic partitioning via hypertables
- Continuous aggregates for rollups
- Good compression with native and custom algorithms
- ACID compliance and strong consistency
- Rich ecosystem of PostgreSQL extensions
- Good for moderate scale

**Trade-offs:**
- Lower compression than ClickHouse (70% vs 92%)
- Slower for large analytical queries
- Higher storage costs at scale
- More limited horizontal scalability

**Use Cases:**
- Teams with strong PostgreSQL expertise
- Requirements for ACID transactions
- Moderate scale deployments (<10M spans/day)
- Need for relational joins with metadata

#### OpenSearch/Elasticsearch

**Strengths:**
- Excellent full-text search capabilities
- Rich query DSL for complex searches
- Good visualization ecosystem (Kibana/Dashboards)
- Familiar to many operations teams
- Strong for log analysis use cases

**Trade-offs:**
- Poor compression (50% typical)
- High storage and compute costs
- Struggles with high-cardinality fields
- More complex cluster management
- Higher memory requirements

**Use Cases:**
- Primarily log-focused observability
- Heavy text search requirements
- Existing OpenSearch/Elastic infrastructure
- Teams prioritizing search over analytics

#### PostgreSQL (Metadata Store)

**Strengths:**
- Reliable, battle-tested RDBMS
- ACID guarantees for critical metadata
- Rich constraint and validation support
- Excellent for relational data
- Strong backup and recovery tools

**Trade-offs:**
- Not optimized for time-series data
- Limited horizontal scalability
- Not suitable for high-volume telemetry

**Use Cases:**
- User accounts and authentication
- Agent configurations and metadata
- API keys and access control
- Alerting rules and dashboard definitions

### Recommendation

**Tiered Storage Strategy:**

1. **Hot Storage (0-7 days):** ClickHouse with high-performance SSD
   - Full-resolution traces and spans
   - Fast queries for recent data
   - All attributes and tags preserved

2. **Warm Storage (8-30 days):** ClickHouse with standard disk
   - Full-resolution data, slower queries acceptable
   - Cost-optimized storage tier

3. **Cold Storage (31-365 days):** S3/MinIO with Parquet
   - Sampled traces (10% retention)
   - Pre-aggregated metrics only
   - Archive for compliance and long-term analysis

4. **Metadata Store:** PostgreSQL
   - User accounts, teams, API keys
   - Dashboard and alert configurations
   - Agent metadata and relationships

## Layer 4: Instrumentation Layer

### Decision Matrix

| Framework | Language | Coverage | Overhead | Standards | Customization |
|-----------|----------|----------|----------|-----------|---------------|
| **OpenLLMetry** | Python/JS | Excellent | Low | OTel | Medium |
| **LangChain Callbacks** | Python/JS | Good | Low | Custom | High |
| **OpenAI SDK** | Python/JS | Limited | Minimal | Custom | Low |
| **Custom OTel** | Any | Complete | Variable | OTel | Complete |

### Detailed Comparison

#### OpenLLMetry (Recommended Foundation)

**Strengths:**
- Traceloop's SDK with GenAI semantic conventions
- Apache 2.0 license, open-source
- Standard OTLP output compatible with any backend
- Auto-instrumentation for popular frameworks:
  - OpenAI, Anthropic, Cohere APIs
  - LangChain, LlamaIndex
  - Vector databases (Pinecone, Weaviate, Qdrant)
- Rich span attributes:
  - Token counts (prompt/completion)
  - Model parameters
  - Cost estimation
  - Latency breakdown

**Integration Example:**
```python
from traceloop.sdk import Traceloop

Traceloop.init(
    app_name="a11i-app",
    api_endpoint="http://otel-collector:4318",
    disable_batch=False
)

# Automatic instrumentation
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
# Trace created automatically with token counts, latency, cost
```

**Trade-offs:**
- Limited to supported frameworks
- May not capture custom agent logic
- Requires extending for a11i-specific metrics

#### Custom a11i Extensions

**Additional Instrumentation Needed:**
- Agent loop iterations (think/act/observe cycles)
- Tool call chains and dependencies
- Context window utilization over time
- Working memory state transitions
- Decision point metadata
- Feedback loop metrics

**Extension Pattern:**
```python
from opentelemetry import trace
from a11i.instrumentation import AgentTracer

tracer = AgentTracer("agent-name")

@tracer.trace_agent_loop()
def agent_iteration(state):
    with tracer.span("think") as span:
        thought = think(state)
        span.set_attribute("thought.type", thought.type)
        span.set_attribute("thought.confidence", thought.confidence)

    with tracer.span("act") as span:
        action = act(thought)
        span.set_attribute("action.type", action.type)
        span.set_attribute("action.tool", action.tool_name)

    with tracer.span("observe") as span:
        observation = observe(action)
        span.set_attribute("observation.success", observation.success)

    return observation
```

### Recommendation

**Layered Instrumentation Strategy:**

1. **Foundation:** OpenLLMetry for standard LLM framework instrumentation
2. **Extensions:** Custom a11i spans for agent-specific behavior
3. **Metrics:** Prometheus metrics for system health and resource usage
4. **Logs:** Structured logging with correlation IDs matching trace_id

## Layer 5: PII Redaction

### Decision Matrix

| Solution | Coverage | Accuracy | Performance | Customization | Maintenance |
|----------|----------|----------|-------------|---------------|-------------|
| **Presidio** | 50+ types | High | Good | Excellent | Low |
| **Custom Regex** | Limited | Moderate | Excellent | Complete | High |
| **AWS Comprehend** | Good | High | Moderate | Low | None |
| **Google DLP** | Excellent | Very High | Moderate | Medium | None |

### Detailed Comparison

#### Microsoft Presidio (Recommended)

**Strengths:**
- Open-source (MIT license)
- 50+ built-in entity recognizers:
  - Credit cards, SSN, emails, phone numbers
  - Names, locations, organizations
  - Medical records, IP addresses
  - Custom patterns via regex or NER models
- Multiple anonymization strategies:
  - Redaction (removal)
  - Replacement (synthetic data)
  - Hashing (deterministic)
  - Encryption (reversible)
- Configurable confidence thresholds
- Integration as OTel Collector processor

**Architecture:**
```
Span Data → Presidio Analyzer → Confidence Scores → Anonymizer → Redacted Span
```

**Implementation Example:**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = "John Doe's SSN is 123-45-6789"
results = analyzer.analyze(text=text, language='en')
anonymized = anonymizer.anonymize(text=text, analyzer_results=results)

# Output: "<PERSON>'s SSN is <SSN>"
```

**Trade-offs:**
- Python-based, adds latency (10-50ms per span)
- May have false positives requiring tuning
- NER models require GPU for best performance

**Use Cases:**
- Comprehensive PII protection required
- Compliance with GDPR, HIPAA, CCPA
- Mixed content types (structured and unstructured)

#### Custom Regex (Lightweight Alternative)

**Strengths:**
- Zero external dependencies
- Sub-millisecond performance
- Complete control over patterns
- Easy to debug and test

**Trade-offs:**
- Limited to known patterns
- High false positive/negative rates
- Requires ongoing maintenance
- No semantic understanding

**Use Cases:**
- Simple, well-defined PII patterns
- Performance-critical hot path
- Predictable input formats

### Recommendation

**Hybrid Approach:**

1. **Hot Path (Proxy/Collector):** Lightweight regex for known patterns
   - API keys, tokens, passwords (HIGH confidence patterns)
   - Minimal latency impact

2. **Processing Pipeline:** Presidio for comprehensive analysis
   - Full entity recognition
   - Acceptable latency (10-50ms)
   - Configurable per-tenant

3. **Configuration:**
   - Tenant-level PII policies
   - Allow-list for specific attributes
   - Audit log of redactions

## Layer 6: Tokenization Libraries

### Decision Matrix

| Tokenizer | Model Coverage | Speed | Accuracy | Ecosystem |
|-----------|----------------|-------|----------|-----------|
| **tiktoken** | OpenAI | Excellent | Exact | Python/Rust |
| **SentencePiece** | LLaMA/Mistral | Excellent | Exact | C++/Python |
| **HuggingFace** | Universal | Good | High | Python/JS |
| **Custom BPE** | Specific | Excellent | Exact | Any |

### Detailed Comparison

#### tiktoken (OpenAI Models)

**Strengths:**
- Official OpenAI tokenizer library
- Encoding support:
  - `cl100k_base`: GPT-4, GPT-3.5-turbo
  - `o200k_base`: GPT-4o
  - `p50k_base`: Codex models
- Written in Rust, Python bindings
- Exact token counts matching API billing
- Fast: 1M tokens/second

**Usage Example:**
```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode("Hello, world!")
num_tokens = len(tokens)  # Exact count for billing

# For token limits
max_tokens = 8192
available = max_tokens - num_tokens
```

**Trade-offs:**
- OpenAI models only
- No support for Anthropic/other providers

#### SentencePiece (Open Models)

**Strengths:**
- LLaMA, Mistral, Falcon tokenization
- BPE and Unigram models supported
- Cross-platform (C++, Python, Go)
- Unicode normalization built-in

**Usage Example:**
```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('llama-tokenizer.model')

tokens = sp.encode('Hello, world!')
num_tokens = len(tokens)
```

**Trade-offs:**
- Requires model file for each tokenizer
- Slightly slower than tiktoken

#### HuggingFace Transformers (Universal Fallback)

**Strengths:**
- AutoTokenizer supports 100+ models
- Automatic model download
- Consistent API across models
- Python and JavaScript support

**Usage Example:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokens = tokenizer.encode("Hello, world!")
num_tokens = len(tokens)
```

**Trade-offs:**
- Slower than specialized tokenizers
- Requires internet for model download
- Heavier dependency

### Recommendation

**Multi-Tokenizer Strategy:**

```python
class TokenCounter:
    """Adaptive token counter supporting multiple model families"""

    def __init__(self):
        self.encodings = {
            'openai': tiktoken.encoding_for_model,
            'anthropic': self._anthropic_counter,
            'llama': self._load_sentencepiece
        }

    def count_tokens(self, text: str, model: str) -> int:
        family = self._detect_family(model)
        encoder = self.encodings[family]
        return len(encoder(text))
```

**Implementation Priority:**
1. **tiktoken** for OpenAI (GPT-3.5, GPT-4, GPT-4o)
2. **Anthropic SDK** for Claude (uses their API)
3. **SentencePiece** for LLaMA/Mistral/Falcon
4. **HuggingFace** as universal fallback

## Layer 7: Dashboards & Visualization

### Decision Matrix

| Solution | Setup Time | Customization | Agent-Specific | Cost |
|----------|------------|---------------|----------------|------|
| **Grafana** | Low | High | Medium | Open-source |
| **Custom React** | High | Complete | Excellent | Development |
| **Datadog APM** | Very Low | Low | Low | $$$$ |
| **Jaeger UI** | Low | Low | Low | Open-source |

### Detailed Comparison

#### Grafana (Recommended for Foundation)

**Strengths:**
- Industry standard, familiar to most teams
- Pre-built GenAI and LLM dashboards available
- Full LGTM stack integration (Loki, Grafana, Tempo, Mimir)
- Rich plugin ecosystem
- AlertManager integration
- Multi-tenancy support
- Open-source with enterprise option

**Pre-Built Dashboards:**
- LLM request rates and latencies
- Token consumption over time
- Cost tracking per model/user
- Error rate and failure analysis
- P50/P95/P99 latency percentiles

**Trade-offs:**
- Limited agent-specific visualizations
- Generic observability focus
- Requires custom panels for advanced UX

**Use Cases:**
- Standard observability metrics and traces
- Team collaboration on shared dashboards
- Alerting and on-call workflows

#### Custom React Dashboard (Agent-Specific UX)

**Strengths:**
- Complete control over user experience
- Agent-specific visualizations:
  - Chain of Thought waterfall view
  - Tool call dependency graphs
  - Context window utilization heatmaps
  - Cost sunburst charts by project/user/model
  - Memory state timelines
- Rich interactivity and drill-downs
- Branded experience

**Key Visualizations:**

1. **Chain of Thought Waterfall:**
   ```
   Think (GPT-4) ████████░░░░░░░░ 850ms | 1,200 tokens | $0.048
   ├─ Act (Tool: search) ████░░░░░░░░ 450ms
   └─ Observe (GPT-4) ████████████░░░░ 1,200ms | 2,100 tokens | $0.084
   ```

2. **Cost Sunburst:**
   ```
   Total: $1,234.56
   ├─ Project A: $678.90
   │  ├─ GPT-4: $456.78
   │  └─ Claude: $222.12
   └─ Project B: $555.66
   ```

3. **Context Window Utilization:**
   ```
   ████████████████░░░░ 16,384 / 32,768 tokens (50%)
   Warning: Approaching 80% threshold
   ```

**Trade-offs:**
- High initial development effort
- Ongoing maintenance burden
- Requires frontend expertise

**Use Cases:**
- Product differentiation
- Agent developer-focused features
- Custom analytics and insights

### Recommendation

**Dual-Dashboard Strategy:**

1. **Grafana** for operational observability
   - System health and resource usage
   - SLO monitoring and alerting
   - Team-wide visibility

2. **Custom React Dashboard** for agent-specific UX
   - Developer productivity features
   - Cost optimization insights
   - Agent behavior analysis

**Phase 1:** Start with Grafana only
**Phase 2:** Add custom dashboards as product matures

## Layer 8: Programming Languages

### Language Selection by Component

| Component | Language | Rationale |
|-----------|----------|-----------|
| **Proxy/Gateway** | Go | Concurrency, performance, OTel ecosystem |
| **SDK (Python)** | Python | Primary AI/ML ecosystem, LangChain/LlamaIndex |
| **SDK (Node)** | TypeScript | Enterprise web apps, Next.js, NestJS |
| **Backend API** | Go | Type safety, performance, easy deployment |
| **OTel Processors** | Go | Native OTel Collector extension language |
| **UI/Dashboard** | TypeScript/React | Modern frontend, rich ecosystem |
| **Data Pipeline** | Python | pandas, dask, data science tools |
| **ML/Analytics** | Python | scikit-learn, TensorFlow, PyTorch |

### Language-Specific Considerations

#### Go for Infrastructure

**Strengths:**
- Single binary deployment, no runtime dependencies
- Excellent concurrency with goroutines
- Strong standard library (HTTP/2, TLS, context)
- Native OpenTelemetry support
- Fast compilation and startup
- Cross-platform builds

**Use Cases:**
- Proxy/gateway services
- API backend
- OTel Collector processors
- CLI tools

#### Python for AI/ML Integration

**Strengths:**
- Dominant AI/ML ecosystem
- LangChain, LlamaIndex, AutoGen integration
- Rich data processing libraries
- Familiar to data scientists

**Use Cases:**
- Python SDK
- Data pipelines
- ML model integration
- Analytics notebooks

#### TypeScript for Web

**Strengths:**
- Type safety for large codebases
- Rich React ecosystem
- Node.js for backend services
- Strong tooling (VS Code, ESLint)

**Use Cases:**
- Custom dashboard UI
- Node.js/TypeScript SDK
- Web-based admin interfaces

### Recommendation

**Polyglot Architecture with Clear Boundaries:**
- **Infrastructure:** Go for performance and operational simplicity
- **Integration:** Python for AI/ML ecosystem compatibility
- **Frontend:** TypeScript/React for modern web UX
- **Cross-Language:** gRPC/Protocol Buffers for type-safe communication

## Complete Stack Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION & UI LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│ Grafana                  │ Operational dashboards, alerts       │
│ Custom React Dashboard   │ Agent-specific UX, cost analytics    │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│ Go REST API              │ High-performance backend             │
│ GraphQL (optional)       │ Flexible querying for UI             │
│ Authentication           │ JWT + API keys (PostgreSQL)          │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
┌─────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│ ClickHouse (hot/warm)    │ Traces, spans, metrics (0-30 days)   │
│ S3/MinIO (cold)          │ Archive storage (31-365 days)        │
│ PostgreSQL               │ Metadata, users, configs             │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
┌─────────────────────────────────────────────────────────────────┐
│                   MESSAGE QUEUE LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│ NATS JetStream (default) │ K8s-native, <1ms latency             │
│ Apache Kafka (scale)     │ Enterprise scale, stream processing  │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
┌─────────────────────────────────────────────────────────────────┐
│                 OPENTELEMETRY COLLECTOR                          │
├─────────────────────────────────────────────────────────────────┤
│ PII Redaction            │ Presidio integration                 │
│ Token Counting           │ tiktoken, SentencePiece              │
│ Batching & Compression   │ OTLP optimization                    │
│ Multi-tenant Routing     │ Namespace isolation                  │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
┌─────────────────────────────────────────────────────────────────┐
│                    PROXY/GATEWAY LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│ Go Custom Proxy (dev)    │ Rapid iteration, full control        │
│ Envoy AI Gateway (prod)  │ Production-ready, <3ms latency       │
│ Features                 │ Rate limiting, fallback, MCP         │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
┌─────────────────────────────────────────────────────────────────┐
│                          SDK LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│ Python SDK               │ AI/ML ecosystem integration          │
│ TypeScript/Node SDK      │ Web app integration                  │
│ Auto-instrumentation     │ Zero-code observability              │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
┌─────────────────────────────────────────────────────────────────┐
│                   INSTRUMENTATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│ OpenLLMetry              │ Standard LLM instrumentation         │
│ a11i Extensions          │ Agent loops, tool chains             │
│ Prometheus Metrics       │ System health, resource usage        │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Dependencies

```yaml
foundation:
  opentelemetry: "1.22+"
  kubernetes: "1.27+"
  helm: "3.12+"

proxy_gateway:
  development:
    go: "1.21+"
    otel_go: "1.22+"
  production:
    envoy_ai_gateway: "latest"
    litellm: "1.17+" # alternative

message_queue:
  default:
    nats_jetstream: "2.10+"
  enterprise:
    apache_kafka: "3.6+"
    redpanda: "23.3+" # alternative

storage:
  analytics:
    clickhouse: "23.11+"
  metadata:
    postgresql: "15+"
  archive:
    s3_compatible: "any"
    minio: "RELEASE.2024-01-01+" # optional

instrumentation:
  openllmetry: "0.22+"
  tiktoken: "0.5+"
  sentencepiece: "0.1.99+"
  presidio_analyzer: "2.2+"
  presidio_anonymizer: "2.2+"

visualization:
  grafana: "10.2+"
  react: "18+"
  typescript: "5.3+"

languages:
  go: "1.21+"
  python: "3.11+"
  node: "20+"
```

## Key Takeaways

### Architecture Principles

1. **Start Simple, Scale Intentionally**
   - Begin with Go proxy + NATS + ClickHouse
   - Migrate to Envoy AI Gateway when production-ready
   - Add Kafka only when NATS becomes a bottleneck

2. **Cloud-Native First**
   - All components designed for Kubernetes
   - Minimal operational overhead
   - Horizontal scalability built-in

3. **Standards-Based Integration**
   - OpenTelemetry for all instrumentation
   - OTLP for data transport
   - Prometheus for metrics
   - Standard protocols reduce vendor lock-in

4. **Cost Optimization**
   - ClickHouse compression saves 10x on storage
   - Tiered storage strategy (hot/warm/cold)
   - Token counting for accurate cost attribution
   - Sampling policies for high-volume traces

5. **Security & Compliance**
   - PII redaction in OTel Collector
   - Multi-tenant isolation
   - Encryption at rest and in transit
   - Audit logging for compliance

### Performance Targets

| Metric | Target | Component |
|--------|--------|-----------|
| Proxy latency | <5ms | Go proxy / Envoy AI Gateway |
| Queue latency | <1ms | NATS JetStream |
| Query latency (hot) | <100ms | ClickHouse |
| Storage compression | >90% | ClickHouse |
| Data retention (hot) | 7 days | ClickHouse SSD |
| Data retention (warm) | 30 days | ClickHouse HDD |
| Data retention (cold) | 365 days | S3/MinIO |

### Operational Metrics

| Component | CPU (typical) | Memory (typical) | Storage |
|-----------|---------------|------------------|---------|
| Go Proxy | 0.5 cores | 512 MB | Minimal |
| NATS JetStream | 1 core | 2 GB | 100 GB (configurable) |
| ClickHouse | 4 cores | 16 GB | 1 TB (hot) + archive |
| OTel Collector | 1 core | 1 GB | Minimal |
| Grafana | 0.5 cores | 512 MB | 10 GB |
| PostgreSQL | 2 cores | 4 GB | 100 GB |

### Decision Framework

When evaluating technology choices:

1. **Alignment with Principles**
   - Does it support cloud-native deployment?
   - Is it standards-based and interoperable?
   - What is the operational complexity?

2. **Performance Characteristics**
   - What is the latency overhead?
   - How does it scale horizontally?
   - What are the resource requirements?

3. **Ecosystem Maturity**
   - How active is the community?
   - What is the vendor support story?
   - Are there production success stories?

4. **Team Capability**
   - Do we have expertise in this technology?
   - What is the learning curve?
   - Can we hire talent for it?

5. **Total Cost of Ownership**
   - What are the licensing costs?
   - What are the infrastructure costs?
   - What is the operational burden?

## Migration Path

### Phase 1: MVP (Months 1-3)

**Minimal Viable Stack:**
- Go custom proxy (rapid iteration)
- NATS JetStream (simplicity)
- ClickHouse (cost-effective storage)
- Grafana (quick dashboards)
- OpenLLMetry (standard instrumentation)

**Capabilities:**
- Basic trace collection and visualization
- Token counting and cost tracking
- Simple dashboards for developers
- Manual PII redaction

### Phase 2: Production-Ready (Months 4-6)

**Enhanced Stack:**
- Envoy AI Gateway (production proxy)
- Enhanced ClickHouse with replication
- Presidio PII redaction
- Multi-tenant isolation
- Custom Grafana dashboards

**Capabilities:**
- Production-grade reliability
- Automatic PII redaction
- Per-tenant observability
- SLO monitoring and alerting

### Phase 3: Scale & Polish (Months 7-12)

**Full-Featured Stack:**
- Kafka (if needed for scale)
- Custom React dashboard
- Advanced analytics
- ML-based anomaly detection
- Cost optimization recommendations

**Capabilities:**
- Enterprise scale (>1M spans/s)
- Rich agent-specific UX
- Predictive insights
- Automated cost optimization

## References

### Documentation Links

- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
- [Envoy AI Gateway](https://www.envoyproxy.io/docs/envoy/latest/)
- [NATS JetStream](https://docs.nats.io/nats-concepts/jetstream)
- [ClickHouse Documentation](https://clickhouse.com/docs/en/intro)
- [OpenLLMetry SDK](https://github.com/traceloop/openllmetry)
- [Microsoft Presidio](https://microsoft.github.io/presidio/)
- [tiktoken](https://github.com/openai/tiktoken)

### Related a11i Documentation

- [System Design](/home/becker/projects/a11i/docs/02-architecture/system-design.md) - Overall architecture
- [Data Flow](/home/becker/projects/a11i/docs/02-architecture/data-flow.md) - How data moves through the system
- [Proxy/Gateway Component](/home/becker/projects/a11i/docs/03-components/proxy-gateway.md) - Detailed proxy design
- [Message Queue Component](/home/becker/projects/a11i/docs/03-components/message-queue.md) - Queue architecture
- [Storage Layer Component](/home/becker/projects/a11i/docs/03-components/storage-layer.md) - Storage design

### Performance Benchmarks

- ClickHouse: [300M spans/day case study](https://clickhouse.com/blog/storing-opentelemetry-data-in-clickhouse)
- NATS JetStream: [Performance testing results](https://github.com/nats-io/nats-server/blob/main/JETSTREAM-PERFORMANCE.md)
- Envoy: [Proxy performance benchmarks](https://www.envoyproxy.io/docs/envoy/latest/faq/performance/how_fast_is_envoy)

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-26
**Status:** Draft - Pending review
**Next Review:** Before Phase 1 implementation kickoff
