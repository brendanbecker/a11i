---
title: "Architectural Decision Log"
category: "Reference"
tags: ["adr", "decisions", "architecture", "rationale"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "/docs/02-architecture/system-architecture.md"
  - "/docs/02-architecture/technology-stack.md"
  - "/docs/05-security-compliance/compliance-framework.md"
  - "/docs/01-overview/competitive-landscape.md"
---

# Architectural Decision Log

## Table of Contents

- [Overview](#overview)
- [Decision Template](#decision-template)
- [Active Decisions](#active-decisions)
  - [ADR-001: ClickHouse for Trace Storage](#adr-001-clickhouse-for-trace-storage)
  - [ADR-002: Go for Proxy Implementation](#adr-002-go-for-proxy-implementation)
  - [ADR-003: NATS JetStream for Message Queue](#adr-003-nats-jetstream-for-message-queue)
  - [ADR-004: OpenLLMetry as Instrumentation Foundation](#adr-004-openllmetry-as-instrumentation-foundation)
  - [ADR-005: Hybrid Instrumentation Model](#adr-005-hybrid-instrumentation-model)
  - [ADR-006: Open-Core Licensing Model](#adr-006-open-core-licensing-model)
  - [ADR-007: Edge-Based PII Redaction](#adr-007-edge-based-pii-redaction)
  - [ADR-008: Passthrough-with-Tapping for Streaming](#adr-008-passthrough-with-tapping-for-streaming)
- [Superseded Decisions](#superseded-decisions)
- [Key Takeaways](#key-takeaways)

---

## Overview

This document records significant architectural decisions made during the design and development of the a11i platform. Each decision is documented with context, rationale, consequences, and alternatives considered to provide transparency and enable future teams to understand the reasoning behind key choices.

**Purpose**:
- Document why decisions were made, not just what decisions
- Provide context for future architectural evolution
- Enable informed changes when assumptions change
- Onboard new team members with decision history

**Usage**:
- Reference ADR numbers when discussing architecture: "Per ADR-003..."
- Update status when decisions are superseded or deprecated
- Add new ADRs when making significant architectural choices
- Link to ADRs from related technical documentation

---

## Decision Template

Use this template when creating new ADRs:

```markdown
## ADR-XXX: [Decision Title]

**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Superseded | Deprecated
**Supersedes**: ADR-XXX (if applicable)
**Superseded by**: ADR-XXX (if applicable)

### Context

What is the issue we're facing? What forces are at play? What technical, business, or organizational factors are relevant?

### Decision

What is the decision we're making? Be specific and actionable.

### Consequences

What becomes easier or harder as a result of this decision?

**Positive Consequences:**
- ...

**Negative Consequences:**
- ...

**Risks:**
- ...

**Mitigation Strategies:**
- ...

### Alternatives Considered

What other options did we evaluate and why were they rejected?

1. **Alternative 1**: Description
   - Pros: ...
   - Cons: ...
   - Rejected because: ...

2. **Alternative 2**: Description
   - Pros: ...
   - Cons: ...
   - Rejected because: ...

### References

- Links to relevant documentation
- Research materials
- Competitive analysis
- Performance benchmarks
```

---

## Active Decisions

---

## ADR-001: ClickHouse for Trace Storage

**Date**: 2025-11-20
**Status**: Accepted
**Decision Owner**: Platform Architecture Team

### Context

a11i requires high-performance analytical storage for telemetry data (traces, spans, metrics) with the following requirements:

**Requirements**:
- Handle 300M+ spans/day on single instance
- Sub-second query performance for dashboard queries
- High compression ratios to minimize storage costs
- Time-series optimized for retention and TTL management
- Native OpenTelemetry Collector exporter support
- Cost-effective at scale (10x lower storage costs than alternatives)

**Constraints**:
- Must scale horizontally for enterprise deployments
- Must support distributed architecture with replication
- Must integrate with standard observability tooling (Grafana, Jaeger)
- Must handle high-cardinality dimensions (user_id, trace_id, agent_id)

### Decision

**We will use ClickHouse as the primary storage backend for traces, spans, and metrics.**

**Architecture**:
- **MergeTree engine** for time-series data with automatic merging
- **Columnar storage** for 10x-20x compression (92% typical compression ratio)
- **Distributed tables** with sharding by tenant_id for horizontal scaling
- **Materialized views** for pre-aggregated dashboards and reports
- **TTL-based retention** with hot/warm/cold storage tiers
- **Native OTLP exporter** from OpenTelemetry Collector

**Tiered Storage Strategy**:
- Hot (0-7 days): ClickHouse on SSD, sub-second queries
- Warm (8-30 days): ClickHouse on HDD, second-range queries
- Cold (31-365 days): S3/MinIO archive, archival access

### Consequences

**Positive Consequences**:
- **92% compression**: Typical 10x-20x reduction in storage costs vs row-based databases
- **Sub-second queries**: P95 query latency <500ms for trace reconstruction
- **300M spans/day**: Proven capacity on single instance, scales horizontally
- **Low operational cost**: Storage costs ~$0.10/GB/month (vs $0.30+/GB for alternatives)
- **Standard integrations**: Native Grafana support, Jaeger backend compatibility

**Negative Consequences**:
- **Limited UPDATE/DELETE**: Not suitable for OLTP workloads (acceptable for append-only telemetry)
- **Learning curve**: ClickHouse query language and table engines require expertise
- **Eventually consistent**: Distributed mode has eventual consistency (acceptable for observability)

**Risks**:
- **Version compatibility**: ClickHouse evolves rapidly; breaking changes possible
- **Operational complexity**: Distributed ClickHouse clusters require expertise
- **Query optimization**: Complex queries may require manual optimization

**Mitigation Strategies**:
- Pin ClickHouse version and test upgrades in staging
- Use managed ClickHouse Cloud for simplified operations (alternative to self-hosting)
- Provide query templates and materialized views for common use cases
- Monitor query performance and create indexes proactively

### Alternatives Considered

#### 1. PostgreSQL + TimescaleDB

**Description**: Use PostgreSQL with TimescaleDB extension for time-series data

**Pros**:
- Familiar SQL semantics and tooling
- ACID compliance and strong consistency
- Rich PostgreSQL ecosystem (extensions, monitoring tools)
- Good for moderate scale (<10M spans/day)

**Cons**:
- Lower compression: ~60% vs ClickHouse's 92%
- Slower analytical queries: 5-10x slower than ClickHouse on aggregations
- Higher storage costs: 3x more expensive at scale
- Limited horizontal scalability compared to ClickHouse

**Rejected because**: Storage costs and query performance don't meet requirements at target scale (300M+ spans/day). Better suited for metadata storage (user accounts, configurations) which is how we're using PostgreSQL.

---

#### 2. Elasticsearch / OpenSearch

**Description**: Use Elasticsearch or OpenSearch for full-text search and analytics

**Pros**:
- Excellent full-text search capabilities
- Rich query DSL for complex searches
- Familiar to many operations teams (ELK stack)
- Strong visualization ecosystem (Kibana)

**Cons**:
- Poor compression: ~50% vs ClickHouse's 92%
- 5-10x higher storage costs than ClickHouse
- Struggles with high-cardinality fields (trace_id, span_id, user_id)
- Higher memory requirements: 2-3x more RAM needed
- More complex cluster management (master nodes, data nodes, coordination)

**Rejected because**: Storage costs prohibitive at scale. High-cardinality trace data doesn't benefit from Elasticsearch's text search strengths. Better suited for log aggregation where full-text search is critical.

---

#### 3. Apache Druid

**Description**: Use Apache Druid for OLAP and real-time analytics

**Pros**:
- Designed for OLAP workloads
- Real-time and batch ingestion
- Good compression and query performance
- Built-in rollup and aggregation

**Cons**:
- Complex architecture: Multiple node types (Historical, Broker, Coordinator, etc.)
- Operational overhead: Requires ZooKeeper and deep storage
- Smaller ecosystem: Fewer integrations and community resources
- Steeper learning curve for team

**Rejected because**: Operational complexity outweighs benefits. ClickHouse provides similar performance with simpler architecture and better OpenTelemetry ecosystem integration.

---

### References

- [ClickHouse OpenTelemetry Case Study](https://clickhouse.com/blog/storing-opentelemetry-data-in-clickhouse) - 300M spans/day benchmark
- [ClickHouse vs TimescaleDB Benchmark](https://clickhouse.com/benchmark/timescale/) - Query performance comparison
- [Technology Stack Documentation](/home/becker/projects/a11i/docs/02-architecture/technology-stack.md#layer-3-storage-layer)

---

## ADR-002: Go for Proxy Implementation

**Date**: 2025-11-20
**Status**: Accepted
**Decision Owner**: Platform Architecture Team

### Context

a11i requires a high-performance proxy/gateway component to intercept LLM API traffic for observability. The proxy must:

**Requirements**:
- Sub-5ms latency overhead (P99)
- Handle streaming responses without TTFT impact
- Support concurrent connections (1000+ simultaneous requests)
- Integrate with OpenTelemetry natively
- Deploy easily in Kubernetes environments

**Development Constraints**:
- Initial development phase requires rapid iteration
- Team has varying language expertise (Python, Go, JavaScript)
- Proxy will evolve significantly during MVP phase

### Decision

**We will implement the proxy in Go, with a migration path to Envoy AI Gateway for production.**

**Development Phase (Phase 1: 0-6 months)**:
- **Custom Go Proxy** for rapid iteration and learning
- Goroutines for concurrent streaming connection handling
- Native OpenTelemetry Go libraries (`go.opentelemetry.io/otel`)
- Single binary deployment with no runtime dependencies

**Production Phase (Phase 2: 6+ months)**:
- **Migrate to Envoy AI Gateway** when production-ready
- Proven at scale (C++ implementation, <3ms latency)
- Built-in LLM provider support and Model Context Protocol (MCP)
- Keep Go proxy for custom business logic not supported by Envoy

### Consequences

**Positive Consequences**:
- **Fast iteration**: Go compilation is fast (seconds), enabling rapid testing
- **Low latency**: <5ms overhead achievable with goroutines
- **Single binary**: No runtime dependencies, easy Docker/K8s deployment
- **Native OTel**: Excellent OpenTelemetry library support in Go ecosystem
- **Production path**: Clear migration to Envoy when ready

**Negative Consequences**:
- **Development overhead**: Must build custom proxy features initially
- **Maintenance burden**: Ongoing maintenance of Go proxy codebase
- **Migration cost**: Eventually migrate to Envoy (planned, not a surprise)

**Risks**:
- **Feature creep**: May over-invest in Go proxy instead of migrating to Envoy
- **Performance gaps**: Custom implementation may miss edge cases

**Mitigation Strategies**:
- Set clear deadline for Envoy migration (6-month MVP milestone)
- Use Go proxy to validate requirements before Envoy migration
- Document lessons learned from Go proxy for Envoy configuration
- Keep Go proxy scope minimal (passthrough, basic instrumentation only)

### Alternatives Considered

#### 1. Python with LiteLLM

**Description**: Use Python with LiteLLM as universal LLM gateway

**Pros**:
- Rapid development: Python is fast to prototype
- 100+ LLM provider support out-of-box
- OpenAI-compatible API normalization
- Active community and development

**Cons**:
- **High latency**: 5-15ms overhead due to Python GIL and interpreter overhead
- **Concurrency limits**: Python async isn't true parallelism
- **Higher memory**: 50-100MB base footprint vs Go's 10-20MB
- **Deployment complexity**: Requires Python runtime, dependencies

**Rejected because**: Latency requirements (<5ms) not achievable with Python. Better suited for development/testing environments or as SDK library, not production proxy.

---

#### 2. Rust

**Description**: Implement proxy in Rust for maximum performance

**Pros**:
- **Superior performance**: Zero-cost abstractions, no GC pauses
- **Memory safety**: Compile-time safety without runtime overhead
- **Excellent async**: Tokio async runtime is battle-tested

**Cons**:
- **Slower development**: Rust's learning curve and borrow checker slow iteration
- **Longer compilation**: Rust compile times 5-10x slower than Go
- **Smaller ecosystem**: Fewer LLM-specific libraries
- **Team expertise**: Steeper onboarding for team

**Rejected because**: Development velocity more important than marginal performance gains during MVP phase. Go provides "good enough" performance (<5ms) with faster iteration. Consider Rust for performance-critical components later if needed.

---

#### 3. Direct Envoy AI Gateway

**Description**: Start with Envoy AI Gateway from day one

**Pros**:
- Production-ready: Proven at scale
- <3ms latency: Best-in-class performance
- Comprehensive features: LLM routing, MCP support, rate limiting

**Cons**:
- **Configuration complexity**: Envoy configuration is verbose and complex
- **Limited flexibility**: Hard to customize for a11i-specific logic
- **Learning curve**: C++ codebase difficult to modify
- **Uncertain maturity**: AI Gateway features are relatively new (2024)

**Rejected because**: Too constraining during MVP phase when requirements are evolving rapidly. Better to learn requirements with flexible Go proxy, then migrate to Envoy once patterns are established.

---

### References

- [Go Concurrency Performance](https://go.dev/blog/concurrency-is-not-parallelism)
- [Envoy AI Gateway Announcement](https://www.envoyproxy.io/docs/envoy/latest/)
- [Technology Stack Documentation](/home/becker/projects/a11i/docs/02-architecture/technology-stack.md#layer-1-proxygateway-layer)

---

## ADR-003: NATS JetStream for Message Queue

**Date**: 2025-11-21
**Status**: Accepted
**Decision Owner**: Platform Architecture Team

### Context

a11i's data pipeline requires a message queue between the OpenTelemetry Collector and storage layer (ClickHouse) to:

**Requirements**:
- Buffer telemetry data during traffic spikes
- Decouple ingestion from storage for resilience
- Support at-least-once delivery semantics
- Handle 10M+ messages/day (moderate scale)
- Deploy easily in Kubernetes environments
- Low operational overhead (<1 FTE for operations)

**Constraints**:
- Must be cloud-native and Kubernetes-friendly
- Must support persistence for durability
- Must scale to enterprise volumes (100M+ msg/day) if needed

### Decision

**We will use NATS JetStream as the default message queue, with Apache Kafka as an enterprise upgrade path.**

**Default Deployment (Small to Medium Scale)**:
- **NATS JetStream** for <100M messages/day
- <1ms message delivery latency
- File-based persistence with configurable retention
- Kubernetes Operator for native deployments
- No external dependencies (no ZooKeeper)

**Enterprise Scale Option (>100M messages/day)**:
- **Apache Kafka** for proven massive scale
- Rich ecosystem (Kafka Connect, KSQL, Schema Registry)
- Multi-datacenter replication
- Extensive monitoring and management tools

**Migration Path**: Design queue abstraction layer allowing migration from NATS to Kafka without application changes.

### Consequences

**Positive Consequences**:
- **Low latency**: <1ms P99 message delivery (vs Kafka's 2-10ms)
- **Simple operations**: Single binary, no ZooKeeper dependency
- **K8s native**: Operator provides seamless Kubernetes integration
- **Low resource usage**: ~1GB memory and 2 CPU cores for typical workloads
- **Flexible migration**: Can upgrade to Kafka when scale demands it

**Negative Consequences**:
- **Smaller ecosystem**: Fewer integrations and tools vs Kafka
- **Limited stream processing**: No native equivalent to Kafka Streams or KSQL
- **Lower maximum throughput**: ~10M msg/s vs Kafka's 100M+ msg/s

**Risks**:
- **Premature scale**: May need to migrate to Kafka sooner than expected
- **Feature gaps**: Missing Kafka ecosystem features (Connect, KSQL)

**Mitigation Strategies**:
- Abstract queue interface to enable transparent migration
- Monitor queue throughput and set alerts at 50% capacity
- Document Kafka migration procedure and test in staging
- Budget for Kafka migration in year 2 roadmap

### Alternatives Considered

#### 1. Apache Kafka

**Description**: Use Kafka from day one for maximum scalability

**Pros**:
- **Massive scale**: Proven at 100M+ messages/second
- **Rich ecosystem**: Kafka Connect, KSQL, Schema Registry
- **Battle-tested**: Used by LinkedIn, Uber, Netflix at extreme scale
- **Stream processing**: Native support for complex stream transformations

**Cons**:
- **Operational complexity**: Requires ZooKeeper, complex cluster management
- **Higher latency**: 2-10ms typical (vs NATS's <1ms)
- **Resource intensive**: 4-8GB memory per broker minimum
- **Overkill for MVP**: Massive over-engineering for initial scale

**Rejected because**: Operational complexity not justified at MVP scale (<10M msg/day). Will use Kafka when we outgrow NATS (100M+ msg/day), but likely 12-24 months away.

---

#### 2. RabbitMQ

**Description**: Use RabbitMQ as general-purpose message broker

**Pros**:
- **Mature ecosystem**: Long history, well-understood
- **Flexible routing**: Complex routing patterns (topic, fanout, direct)
- **Good documentation**: Extensive tutorials and guides
- **Management UI**: Built-in web interface for monitoring

**Cons**:
- **Lower throughput**: ~50K msg/s per broker (vs NATS's 10M msg/s)
- **Erlang dependency**: Requires Erlang VM and understanding
- **Clustering complexity**: RabbitMQ clusters can be fragile
- **Not optimized for throughput**: Better for job queues than streaming

**Rejected because**: Insufficient throughput for telemetry workloads. RabbitMQ excels at routing and job queues, but NATS is purpose-built for high-throughput messaging.

---

#### 3. Amazon SQS / Google Pub/Sub

**Description**: Use cloud provider's managed message queue service

**Pros**:
- **Zero operations**: Fully managed, no infrastructure to maintain
- **Unlimited scale**: Automatically scales to any load
- **Pay-per-use**: No fixed costs for idle infrastructure

**Cons**:
- **Vendor lock-in**: Tied to specific cloud provider
- **Higher latency**: ~20-100ms vs NATS's <1ms
- **Higher costs**: $0.50/million messages adds up at scale
- **No self-hosting**: Can't deploy on-premises for enterprise compliance

**Rejected because**: a11i must support self-hosting for compliance (HIPAA, SOC2). Cloud provider lock-in conflicts with open-source positioning. May offer as optional integration for cloud-native deployments.

---

### References

- [NATS JetStream Performance](https://github.com/nats-io/nats-server/blob/main/JETSTREAM-PERFORMANCE.md)
- [Kafka vs NATS Comparison](https://nats.io/blog/comparing-nats-to-kafka/)
- [Technology Stack Documentation](/home/becker/projects/a11i/docs/02-architecture/technology-stack.md#layer-2-message-queue-layer)

---

## ADR-004: OpenLLMetry as Instrumentation Foundation

**Date**: 2025-11-21
**Status**: Accepted
**Decision Owner**: Platform Architecture Team

### Context

a11i requires automatic instrumentation for LLM applications with minimal code changes. Requirements include:

**Instrumentation Requirements**:
- Auto-instrument popular LLM frameworks (LangChain, LlamaIndex, OpenAI, Anthropic)
- Capture token counts, costs, and latency without manual spans
- Standard OpenTelemetry semantic conventions for LLM operations
- Support multiple LLM providers without custom integrations
- Extensible for a11i-specific agent semantics

**Ecosystem Constraints**:
- Must work with existing OpenTelemetry infrastructure
- Must avoid vendor lock-in or proprietary formats
- Must integrate with Python (primary AI/ML language) and JavaScript

### Decision

**We will use OpenLLMetry (by Traceloop) as the foundation for automatic instrumentation, extended with a11i-specific agent semantics.**

**Architecture**:
```
a11i SDK
├─ OpenLLMetry Core (Apache 2.0)
│  ├─ Auto-instrumentation for LLM providers
│  ├─ Auto-instrumentation for frameworks (LangChain, LlamaIndex)
│  └─ Standard GenAI semantic conventions
└─ a11i Extensions
   ├─ Agent loop instrumentation (Think-Act-Observe)
   ├─ Tool call tracking
   ├─ Context saturation monitoring
   └─ Multi-agent correlation
```

**Integration Approach**:
```python
from traceloop.sdk import Traceloop
from a11i import A11iAgentExtensions

# Initialize OpenLLMetry foundation
Traceloop.init(app_name="my-agent")

# Add a11i agent-specific extensions
A11iAgentExtensions.init()

# Auto-instrumentation now active for both LLM calls and agent loops
```

### Consequences

**Positive Consequences**:
- **Apache 2.0 license**: No licensing restrictions or fees
- **Standard conventions**: Uses OpenTelemetry GenAI semantic conventions
- **Broad coverage**: Auto-instruments 20+ frameworks and providers
- **Active development**: Maintained by Traceloop with regular updates
- **Zero code changes**: Auto-instrumentation via import
- **Extensible**: Can add a11i-specific spans on top of base instrumentation

**Negative Consequences**:
- **Dependency on third-party**: Reliant on Traceloop's development velocity
- **Generic semantics**: Base OpenLLMetry doesn't capture agent-specific concepts
- **Potential breaking changes**: OpenLLMetry evolves; may require adaptation

**Risks**:
- **Project abandonment**: Traceloop could discontinue OpenLLMetry
- **Diverging standards**: GenAI conventions may evolve differently than OpenLLMetry
- **Performance overhead**: Auto-instrumentation adds latency

**Mitigation Strategies**:
- **Fork readiness**: Maintain ability to fork OpenLLMetry if needed (Apache 2.0 allows)
- **Vendor neutrality**: Build on OTel standards, not OpenLLMetry-specific features
- **Performance monitoring**: Measure and optimize instrumentation overhead
- **Contribution**: Contribute agent-specific features back to OpenLLMetry upstream

### Alternatives Considered

#### 1. Custom OpenTelemetry Instrumentation

**Description**: Build all instrumentation from scratch using raw OpenTelemetry SDK

**Pros**:
- **Complete control**: No dependencies, full customization
- **Optimized performance**: Only instrument what's needed
- **Agent-first design**: Purpose-built for agent semantics

**Cons**:
- **High development cost**: 6-12 months to match OpenLLMetry's coverage
- **Maintenance burden**: Must track updates to LLM provider APIs
- **Reinventing wheel**: Duplicating work already done by OpenLLMetry
- **Slower time-to-market**: Delays MVP delivery

**Rejected because**: Development cost too high for MVP phase. OpenLLMetry provides 80% of needed instrumentation; building from scratch provides <20% additional value at 10x cost.

---

#### 2. LangSmith SDK

**Description**: Use LangChain's LangSmith for instrumentation

**Pros**:
- **Deep LangChain integration**: Unmatched visibility into LangChain internals
- **Rich visualization**: Sophisticated trace UI with LangGraph state
- **Integrated features**: Prompt management, datasets, evaluations

**Cons**:
- **Vendor lock-in**: Proprietary format, difficult migration
- **LangChain-only**: Minimal value for non-LangChain users
- **Not OTel-native**: Exports to OTel, but fundamentally proprietary
- **Paid service**: Requires LangSmith subscription for hosting

**Rejected because**: Conflicts with a11i's OpenTelemetry-native and framework-agnostic positioning. LangSmith is a competitor, not a foundation to build on.

---

#### 3. Proprietary Instrumentation (Like Datadog APM)

**Description**: Build proprietary agent for richer data collection

**Pros**:
- **Richer data**: Can capture more than OTel semantic conventions
- **Optimized protocols**: Custom binary protocol vs OTLP
- **Integrated stack**: Tight coupling between agent and backend

**Cons**:
- **Vendor lock-in**: Customers can't use alternative backends
- **No ecosystem**: Can't integrate with Grafana, Prometheus, Jaeger
- **Higher development cost**: Must build and maintain proprietary agent
- **Market rejection**: Developers prefer open standards

**Rejected because**: Violates a11i's core principle of OpenTelemetry-native architecture. Vendor lock-in is a competitive disadvantage vs open-source alternatives.

---

### References

- [OpenLLMetry GitHub](https://github.com/traceloop/openllmetry) - Apache 2.0 licensed
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Technology Stack Documentation](/home/becker/projects/a11i/docs/02-architecture/technology-stack.md#layer-4-instrumentation-layer)

---

## ADR-005: Hybrid Instrumentation Model

**Date**: 2025-11-21
**Status**: Accepted
**Decision Owner**: Product & Platform Teams

### Context

a11i must provide comprehensive observability for AI agents while maximizing ease of adoption. Two fundamental approaches exist:

**Instrumentation Approaches**:

1. **SDK-based (deep visibility)**:
   - Requires code integration
   - Captures internal agent state and reasoning
   - Framework-specific integrations needed
   - Higher implementation effort

2. **Proxy-based (zero-code)**:
   - No code changes required
   - Only sees external LLM API traffic
   - Limited visibility into agent internals
   - Fastest time-to-value

**Market Analysis**:
- **LangSmith**: SDK-only (deep but requires code changes)
- **Helicone**: Proxy-only (fast but limited visibility)
- **a11i Opportunity**: Combine both approaches

### Decision

**We will provide a hybrid instrumentation model combining proxy and SDK approaches, allowing users to choose based on their needs.**

**Instrumentation Tiers**:

| Tier | Approach | Time to Value | Visibility Depth | Use Case |
|------|----------|---------------|------------------|----------|
| **Level 1: Proxy Only** | Sidecar | 5 minutes | LLM calls only | Quick evaluation, legacy systems |
| **Level 2: SDK Basic** | OpenLLMetry | 15 minutes | LLM + basic tracing | Standard LLM apps |
| **Level 3: Framework Integration** | LangChain/CrewAI hooks | 30 minutes | Full agent lifecycle | Modern agent frameworks |
| **Level 4: Custom** | Manual spans | 2-4 hours | Complete control | Proprietary agents |

**Architecture**:
```
┌─────────────────────────────────────────┐
│        a11i Platform Backend            │
└────────────▲───────────▲────────────────┘
             │           │
       ┌─────┴───┐  ┌────┴──────┐
       │ Proxy   │  │ SDK       │
       │ (OTLP)  │  │ (OTLP)    │
       └─────┬───┘  └────┬──────┘
             │           │
       ┌─────┴───────────┴──────┐
       │   Agent Application    │
       │  (Can use both!)       │
       └────────────────────────┘
```

**Progressive Enhancement**:
1. Start with proxy for immediate visibility
2. Add SDK for deeper instrumentation
3. Both send to same trace_id for correlation
4. Users choose based on needs

### Consequences

**Positive Consequences**:
- **Fastest time-to-value**: Proxy provides 5-minute quick-start
- **Maximum depth**: SDK provides full agent lifecycle visibility
- **Flexible adoption**: Users choose level based on effort/value tradeoff
- **Competitive advantage**: Unique positioning vs SDK-only or proxy-only competitors
- **Migration path**: Users can start with proxy, add SDK later

**Negative Consequences**:
- **Complexity**: Must maintain both proxy and SDK codebases
- **Coordination challenge**: Ensuring proxy and SDK produce compatible telemetry
- **Documentation overhead**: Must explain both approaches and when to use each
- **Testing complexity**: Must test both paths and hybrid scenarios

**Risks**:
- **Duplicate telemetry**: Risk of double-counting if proxy and SDK both instrument same LLM call
- **Configuration confusion**: Users may not understand which approach to use
- **Maintenance burden**: Two instrumentation paths = 2x maintenance

**Mitigation Strategies**:
- **Automatic deduplication**: SDK detects proxy presence and skips LLM instrumentation
- **Clear decision tree**: Documentation provides simple flowchart for approach selection
- **Shared codebase**: Maximize code reuse between proxy and SDK
- **Integration testing**: Test hybrid scenarios in CI/CD pipeline

### Alternatives Considered

#### 1. SDK-Only (LangSmith Model)

**Description**: Provide only SDK-based instrumentation

**Pros**:
- **Deep visibility**: Full access to agent internals
- **Rich semantics**: Can capture agent-specific concepts (planning, memory, etc.)
- **Framework integration**: Tight coupling with LangChain, LlamaIndex, etc.

**Cons**:
- **Higher friction**: Requires code changes, slower adoption
- **Framework dependency**: Only works well with supported frameworks
- **Legacy system challenges**: Difficult to instrument existing codebases

**Rejected because**: Too high barrier to adoption. Proxy-based competitors (Helicone) offer faster time-to-value. Hybrid model provides both fast adoption AND deep visibility.

---

#### 2. Proxy-Only (Helicone Model)

**Description**: Provide only proxy-based instrumentation

**Pros**:
- **Zero code changes**: Fastest possible adoption (5 minutes)
- **Framework agnostic**: Works with any LLM client
- **Simple deployment**: Single container deployment

**Cons**:
- **Limited visibility**: Can only see LLM API calls, not agent internals
- **No agent semantics**: Can't capture Think-Act-Observe loops
- **Black box debugging**: Can't observe reasoning or tool calls

**Rejected because**: Insufficient for agent-native observability mission. Surface-level LLM monitoring doesn't differentiate a11i in market. Hybrid model provides proxy benefits PLUS deep visibility.

---

#### 3. eBPF-Based Instrumentation

**Description**: Use eBPF (Extended Berkeley Packet Filter) for kernel-level instrumentation

**Pros**:
- **Zero application changes**: True zero-code instrumentation
- **Language agnostic**: Works with any language (Python, Go, Java, etc.)
- **Deep visibility**: Can intercept system calls and network traffic

**Cons**:
- **Linux-only**: Doesn't work on macOS or Windows
- **Kernel expertise required**: Complex to develop and debug
- **Limited semantic visibility**: Can see HTTP traffic but not LLM-specific semantics
- **Security concerns**: Kernel-level instrumentation raises security questions

**Rejected because**: Too experimental and Linux-specific. Proxy provides similar zero-code benefits with better cross-platform support and easier troubleshooting. May revisit for advanced use cases in future.

---

### References

- [Helicone Architecture](https://docs.helicone.ai/introduction) - Proxy-based approach
- [LangSmith Documentation](https://docs.smith.langchain.com/) - SDK-based approach
- [Hybrid Instrumentation Model](/home/becker/projects/a11i/docs/02-architecture/system-architecture.md#hybrid-instrumentation-model)

---

## ADR-006: Open-Core Licensing Model

**Date**: 2025-11-22
**Status**: Accepted
**Decision Owner**: Executive Team, Legal

### Context

a11i requires a monetization strategy that balances:

**Goals**:
- Build community trust through open source
- Generate revenue to sustain development
- Enable self-hosting for compliance (HIPAA, SOC2)
- Compete with open-source alternatives (Langfuse, Arize Phoenix)
- Prevent cloud providers from undercutting with managed offerings

**Market Context**:
- **Langfuse**: MIT core, proprietary enterprise features (SSO, RBAC)
- **Arize Phoenix**: MIT core, $50K-$100K/year commercial platform
- **LangSmith**: Fully proprietary (competitive threat from vendor lock-in)

**Legal Constraints**:
- Must allow self-hosting without fees (enterprise requirement)
- Must prevent cloud providers from offering without contributing back
- Must enable commercial fork protection

### Decision

**We will use an open-core model with Apache 2.0 license for core platform and proprietary licensing for enterprise features.**

**Open-Source Core (Apache 2.0)**:
- OpenTelemetry SDK and instrumentation
- Proxy/gateway implementation
- Basic observability dashboards
- ClickHouse schema and queries
- Documentation and getting-started guides

**Proprietary Enterprise Features**:
- SSO integration (SAML, OIDC)
- Advanced RBAC (role-based access control, attribute-based access control)
- SCIM user provisioning
- Multi-tenancy with tenant isolation
- Advanced cost optimization recommendations
- SLA guarantees and enterprise support
- Audit logging and compliance reporting

**Monetization Model**:
- **Open Source**: Free forever, unlimited usage
- **Cloud (Managed SaaS)**: $500-$5,000/month based on volume
- **Enterprise (Self-Hosted)**: $25,000-$100,000/year for enterprise features + support

### Consequences

**Positive Consequences**:
- **Community trust**: Apache 2.0 builds trust, encourages contributions
- **Competitive positioning**: Undercuts proprietary solutions (LangSmith) on lock-in
- **Self-hosting compliance**: Enterprises can meet HIPAA/SOC2 requirements
- **Fork protection**: Proprietary enterprise features prevent cloud provider commoditization
- **Sustainable revenue**: Clear path to monetization without restricting core value

**Negative Consequences**:
- **Feature split complexity**: Must maintain clear boundary between open-source and proprietary
- **Competitive pressure**: Open-source competitors can match core features
- **Sales friction**: Must explain why enterprise features aren't open-source
- **Cloud provider risk**: AWS/GCP could still offer managed open-source core

**Risks**:
- **Community backlash**: Perception of "bait and switch" if core features moved to proprietary
- **Limited differentiation**: Core features insufficient to compete with proprietary alternatives
- **Pricing pressure**: Enterprise customers demand open-source pricing

**Mitigation Strategies**:
- **Clear feature roadmap**: Commit to keeping core observability features open-source
- **Community governance**: Establish contributor guidelines and feature prioritization
- **Value articulation**: Clearly communicate value of enterprise features (compliance, support, SLAs)
- **Contributor program**: Recognize and reward community contributions
- **Commercial support**: Offer paid support for open-source version (alternative to enterprise)

### Alternatives Considered

#### 1. Fully Open-Source (MIT or Apache 2.0)

**Description**: Release everything as open-source, monetize only through managed cloud and support

**Pros**:
- **Maximum community trust**: No proprietary features or lock-in
- **Fastest adoption**: No sales friction or licensing concerns
- **Contributor magnet**: Attracts maximum contributors

**Cons**:
- **Weak monetization**: Hard to justify $50K+ enterprise pricing for managed hosting alone
- **Cloud provider risk**: AWS/GCP can undercut pricing with managed offerings
- **Limited defensibility**: No moat against competitors cloning and undercutting

**Rejected because**: Insufficient defensibility for venture-backed company. Cloud providers could offer managed a11i at commodity pricing, making it difficult to build sustainable business. Open-core provides better balance.

---

#### 2. AGPL (Aggressive Copyleft)

**Description**: Use AGPL license requiring cloud providers to open-source their modifications

**Pros**:
- **Fork protection**: Cloud providers must contribute back changes
- **Commercial forcing function**: Cloud providers must negotiate commercial license
- **Community contributions**: Ensures improvements flow back to community

**Cons**:
- **Enterprise resistance**: Many enterprises ban AGPL software (license contamination concerns)
- **Reduced adoption**: Developers avoid AGPL due to legal complexity
- **Contributor friction**: AGPL deters corporate contributors
- **Perception issues**: Seen as "anti-cloud" or "hostile to business"

**Rejected because**: AGPL significantly reduces enterprise adoption. Many Fortune 500 companies have blanket bans on AGPL software. Apache 2.0 open-core provides fork protection through proprietary enterprise features without AGPL friction.

---

#### 3. Fully Proprietary (Like LangSmith)

**Description**: Keep all code proprietary, monetize through SaaS and enterprise licenses

**Pros**:
- **Maximum defensibility**: No risk of commoditization
- **Clear pricing**: No confusion about open-source vs proprietary
- **Simple sales**: Traditional enterprise software model

**Cons**:
- **Vendor lock-in**: Customers hesitate due to lock-in concerns
- **Limited ecosystem**: Proprietary format prevents integration ecosystem
- **Competitive disadvantage**: Open-source alternatives have adoption advantage
- **Community**: No community contributions or ecosystem

**Rejected because**: Violates a11i's core positioning as OpenTelemetry-native and avoiding vendor lock-in. Proprietary approach conflicts with market trends toward open observability standards.

---

### References

- [Langfuse Licensing Model](https://github.com/langfuse/langfuse) - MIT core + proprietary enterprise
- [GitLab Open-Core Success Story](https://about.gitlab.com/company/strategy/)
- [Why Open-Core Works](https://opencoreventures.com/blog/2019-10-open-core-definition/)

---

## ADR-007: Edge-Based PII Redaction

**Date**: 2025-11-22
**Status**: Accepted
**Decision Owner**: Security & Compliance Team

### Context

AI agents frequently process sensitive data (PII, PHI, financial data) that must be protected in telemetry. Traditional observability platforms log raw data, creating compliance and security risks.

**Requirements**:
- GDPR Article 17 compliance (right to erasure)
- HIPAA compliance (PHI protection)
- SOC 2 compliance (data handling controls)
- Breach resilience (backend compromise doesn't expose customer PII)
- Debugging capability (must maintain context for troubleshooting)

**Architectural Options**:
- **Edge (SDK/Sidecar)**: Redact before data leaves customer VPC
- **Storage (Database)**: Redact after storage, mask on query
- **Query-time (Application)**: Show redacted view to users

### Decision

**We will implement PII redaction at the edge (within customer infrastructure) before telemetry data leaves the customer's VPC.**

**Architecture**:
```
Customer VPC (Trusted Zone)
├─ AI Agent Application (processes raw data)
├─ a11i SDK/Sidecar (PII detection & redaction)
└─ Redacted OTLP → a11i Backend (never sees PII)
```

**Implementation**:
- **Microsoft Presidio**: ML-powered PII detection (NER + regex)
- **Windowed buffer scanning**: Detect PII spanning streaming chunks
- **Pseudonymization**: Replace PII with consistent fake values for debugging
- **Configurable policies**: Per-tenant PII handling rules

**Redaction Location**: Within a11i SDK (Python) or Sidecar (Go) as final step before OTLP export

### Consequences

**Positive Consequences**:
- **Privacy by design**: PII never leaves customer control (strongest possible guarantee)
- **Breach resilience**: a11i backend compromise doesn't expose customer PII
- **Compliance simplified**: GDPR/HIPAA requirements met architecturally, not procedurally
- **Customer trust**: "Your data never leaves your VPC" is powerful sales message
- **Regulatory approval**: Easier regulatory sign-off (healthcare, finance)

**Negative Consequences**:
- **Performance impact**: PII detection adds 5-20ms latency per span
- **Customer resources**: PII redaction runs on customer infrastructure (CPU, memory)
- **Debugging complexity**: a11i support team can't see original PII for troubleshooting
- **Consistency challenges**: PII policies must be configured consistently across deployments

**Risks**:
- **False negatives**: Some PII may not be detected (GDPR violation risk)
- **Performance degradation**: Presidio can add significant latency on large texts
- **Configuration errors**: Misconfigured PII policies could leak sensitive data

**Mitigation Strategies**:
- **Default-strict policies**: Enable aggressive PII detection by default
- **Performance optimization**: Quick regex pre-check before expensive ML analysis
- **Monitoring**: Alert on high PII detection latency (>20ms P95)
- **Audit mode**: Optional logging of redaction events (without original PII)
- **Regular testing**: Compliance team tests PII detection accuracy quarterly

### Alternatives Considered

#### 1. Storage-Based Redaction (Redact at Database)

**Description**: Store raw telemetry in database, redact with stored procedures or database views

**Pros**:
- **Zero performance impact**: No latency added to hot path
- **Centralized control**: Single place to update PII patterns
- **Easy rollback**: Can re-redact data if policies change

**Cons**:
- **PII at rest**: Raw PII stored in database (GDPR/HIPAA violation risk)
- **Breach exposure**: Database compromise exposes all customer PII
- **Compliance gaps**: Doesn't satisfy "data minimization" requirements
- **Temporary storage**: Even 1 second of PII storage may violate regulations

**Rejected because**: Violates fundamental principle of data minimization. Even temporary PII storage creates compliance risk. Edge-based redaction provides strongest possible guarantee.

---

#### 2. Query-Time Redaction (Application Layer)

**Description**: Store raw data, redact dynamically when displaying to users

**Pros**:
- **No performance impact**: Redaction only when data accessed (rare)
- **Flexible policies**: Can change redaction rules without re-processing data
- **Full debugging**: Support team can see original data when needed

**Cons**:
- **PII at rest**: Raw PII permanently stored
- **Access control risk**: One broken permission check exposes PII
- **Compliance failure**: Doesn't meet GDPR "right to erasure" requirements
- **Audit complexity**: Hard to prove PII never accessed improperly

**Rejected because**: PII storage violates compliance requirements. Query-time redaction is masking, not deletion. Better suited for internal tools, not compliant observability platform.

---

#### 3. No Redaction (User Responsibility)

**Description**: Don't implement PII redaction; make users responsible

**Pros**:
- **Zero performance impact**: No redaction overhead
- **Zero development cost**: No PII detection code to build
- **Maximum debugging**: Full unredacted data for troubleshooting

**Cons**:
- **Compliance liability**: Users may unknowingly violate GDPR/HIPAA
- **Market limitation**: Can't sell to healthcare, finance, or EU customers
- **Competitive disadvantage**: Competitors offer PII redaction
- **Reputation risk**: One PII breach tanks company reputation

**Rejected because**: Unacceptable compliance and business risk. PII redaction is table-stakes for enterprise observability. Market demands privacy-first architecture.

---

### References

- [Microsoft Presidio Documentation](https://microsoft.github.io/presidio/)
- [GDPR Article 17: Right to Erasure](https://gdpr-info.eu/art-17-gdpr/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [PII Redaction Documentation](/home/becker/projects/a11i/docs/05-security-compliance/pii-redaction.md)

---

## ADR-008: Passthrough-with-Tapping for Streaming

**Date**: 2025-11-22
**Status**: Accepted
**Decision Owner**: Platform Architecture Team

### Context

Streaming LLM responses require specialized observability handling to avoid adding latency. The challenge:

**Requirements**:
- **Zero TTFT impact**: Must not delay Time to First Token
- **Complete telemetry**: Must capture full response for analysis
- **PII detection**: Must scan for PII across chunk boundaries
- **Graceful degradation**: Stream continues even if telemetry fails
- **Cost attribution**: Must calculate token counts and costs

**Performance Requirements**:
- TTFT overhead: <1ms (unmeasurable)
- Memory overhead: <20MB per stream
- CPU overhead: <5%

**Observability Challenges**:
- Full response only available after stream completes
- PII patterns may span multiple chunks
- Token counts only available at stream end
- Traditional buffering would destroy streaming benefits

### Decision

**We will use the passthrough-with-tapping pattern for streaming observability, where chunks are immediately forwarded to the client while simultaneously buffered for telemetry.**

**Pattern**:
```python
async for chunk in llm_stream:
    buffer.append(chunk)  # Non-blocking tap (~20ns)
    yield chunk          # Immediate passthrough (~50ns)

# After stream completes (asynchronous):
asyncio.create_task(analyze_and_emit(buffer))
```

**Mechanism**:
1. **Hot path (user-facing)**: Chunks forwarded immediately with <100ns overhead
2. **Side channel (telemetry)**: Chunks buffered for analysis in parallel
3. **Async processing**: Telemetry emitted after stream completes (fire-and-forget)

**Implementation**:
- **Bounded buffers**: Max 10,000 chunks to prevent OOM
- **Windowed scanning**: PII detection on rolling 256-char window
- **Async export**: OTLP export happens asynchronously, never blocks client

### Consequences

**Positive Consequences**:
- **Zero TTFT impact**: <100 nanoseconds per chunk (0.00016% overhead)
- **Complete telemetry**: Full response captured for analysis
- **Streaming PII detection**: Windowed buffer catches cross-chunk patterns
- **Graceful degradation**: Client receives stream even if telemetry fails
- **Memory efficient**: Bounded buffers prevent memory exhaustion

**Negative Consequences**:
- **Memory overhead**: ~15MB per concurrent stream
- **Buffer management**: Must handle buffer overflow scenarios
- **Complexity**: More complex than blocking analysis

**Risks**:
- **Memory leaks**: Unbounded buffers could exhaust memory
- **Dropped telemetry**: Buffer overflow causes telemetry loss
- **Latency creep**: Poor implementation could add measurable latency

**Mitigation Strategies**:
- **Bounded buffers**: Hard limit at 10,000 chunks (configurable)
- **Buffer overflow alerts**: Alert when buffers fill >80%
- **Performance monitoring**: Track per-chunk append time (<100ns)
- **Load testing**: Test with 1000+ concurrent streams

### Alternatives Considered

#### 1. Blocking Analysis (Wait for Complete Response)

**Description**: Buffer all chunks, analyze after stream completes, then send to client

**Pros**:
- **Simple implementation**: Standard request/response pattern
- **Complete context**: Analyze full response at once
- **Easy PII detection**: No chunk boundary issues

**Cons**:
- **Destroys streaming UX**: TTFT = E2E latency (2-5 seconds)
- **User experience failure**: Negates primary benefit of streaming
- **Competitive disadvantage**: Users will use alternatives with zero latency

**Rejected because**: Destroys fundamental value proposition of streaming. Users would notice 2-5 second delay (vs native <1 second TTFT). Unacceptable UX degradation.

---

#### 2. Sampling (Only Analyze Subset of Chunks)

**Description**: Only buffer and analyze every Nth chunk to reduce overhead

**Pros**:
- **Lower overhead**: Reduced memory and CPU usage
- **Still captures most of response**: 10% sampling still useful

**Cons**:
- **Incomplete telemetry**: Missing chunks makes debugging harder
- **Broken PII detection**: May miss PII that only appears in skipped chunks
- **Inaccurate costs**: Token count estimates will be wrong

**Rejected because**: Incomplete telemetry defeats purpose of observability. PII detection failure creates compliance risk. Passthrough-with-tapping provides complete telemetry with same low overhead.

---

#### 3. Deferred Analysis (Background Job)

**Description**: Store raw chunks to disk, analyze later in background job

**Pros**:
- **Zero real-time overhead**: All analysis happens offline
- **No memory constraints**: Can analyze arbitrary-length streams

**Cons**:
- **Delayed telemetry**: Traces not available for minutes/hours
- **Disk I/O overhead**: Writing chunks to disk adds latency
- **Storage costs**: Must store raw chunks before processing
- **Complexity**: Requires job queue and worker infrastructure

**Rejected because**: Real-time telemetry required for dashboards and alerting. Delayed telemetry makes debugging active issues impossible. Passthrough-with-tapping provides real-time telemetry without deferred processing.

---

### References

- [Streaming Handling Documentation](/home/becker/projects/a11i/docs/03-core-platform/streaming-handling.md)
- [Passthrough-with-Tapping Pattern](/home/becker/projects/a11i/docs/03-core-platform/streaming-handling.md#passthrough-with-tapping-pattern)
- [Performance Benchmarks](/home/becker/projects/a11i/docs/03-core-platform/streaming-handling.md#performance-analysis)

---

## Superseded Decisions

This section will contain decisions that have been replaced by newer ADRs. Currently empty as all decisions are active.

---

## Key Takeaways

> **Critical Architectural Decisions**
>
> **Storage & Infrastructure**:
> - **ClickHouse** (ADR-001): 92% compression, 300M+ spans/day capacity, <500ms P95 queries
> - **NATS JetStream** (ADR-003): <1ms latency, K8s-native, simple operations (upgrade to Kafka at >100M msg/day)
> - **Go Proxy** (ADR-002): <5ms overhead, rapid iteration (migrate to Envoy for production)
>
> **Instrumentation Strategy**:
> - **Hybrid Model** (ADR-005): Proxy (zero-code) + SDK (deep visibility) = competitive advantage
> - **OpenLLMetry Foundation** (ADR-004): Apache 2.0, auto-instruments 20+ frameworks, extended with a11i agent semantics
> - **Passthrough-with-Tapping** (ADR-008): <100ns per chunk overhead, zero TTFT impact, complete telemetry
>
> **Security & Privacy**:
> - **Edge-Based PII Redaction** (ADR-007): Redact before data leaves customer VPC, strongest compliance guarantee
> - **Microsoft Presidio**: ML-powered detection (NER + regex), 5-20ms latency, 95%+ accuracy on structured PII
>
> **Business Model**:
> - **Open-Core** (ADR-006): Apache 2.0 core + proprietary enterprise features (SSO, RBAC, advanced analytics)
> - **Self-hosting**: Enable compliance (HIPAA, SOC2) while protecting against cloud provider commoditization
>
> **Migration Paths Established**:
> - Go Proxy → Envoy AI Gateway (when production-ready, 6-12 months)
> - NATS JetStream → Apache Kafka (when >100M msg/day, 12-24 months)
> - OpenLLMetry → Potential fork if needed (Apache 2.0 allows)

---

**Related Documentation:**
- [System Architecture](/home/becker/projects/a11i/docs/02-architecture/system-architecture.md) - Overall platform design
- [Technology Stack](/home/becker/projects/a11i/docs/02-architecture/technology-stack.md) - Technology decisions and rationale
- [Competitive Landscape](/home/becker/projects/a11i/docs/01-overview/competitive-landscape.md) - Market positioning
- [Compliance Framework](/home/becker/projects/a11i/docs/05-security-compliance/compliance-framework.md) - GDPR, HIPAA, SOC2 requirements

---

*Document Status: Draft | Last Updated: 2025-11-26 | Maintained by: a11i Architecture Team*
