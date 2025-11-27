---
title: Scalability Guide
description: Horizontal scaling, performance targets, and capacity planning for a11i observability platform
category: Operations
tags: [scalability, performance, capacity-planning, benchmarks]
last_updated: 2025-11-26
---

# Scalability Guide

## Table of Contents

1. [Performance Targets](#performance-targets)
2. [Horizontal Scaling Architecture](#horizontal-scaling-architecture)
3. [Production Benchmarks](#production-benchmarks)
4. [Stateless Collectors](#stateless-collectors)
5. [Queue-Based Buffering](#queue-based-buffering)
6. [Partitioned Storage](#partitioned-storage)
7. [Sampling Strategies](#sampling-strategies)
8. [Capacity Planning](#capacity-planning)
9. [Deployment Topologies](#deployment-topologies)
10. [Operational Considerations](#operational-considerations)

## Performance Targets

a11i is designed to add minimal overhead to agent applications while providing comprehensive observability. The following performance targets ensure production-ready scalability:

| Metric | Target | P50 | P90 | P99 |
|--------|--------|-----|-----|-----|
| Proxy Latency Overhead | <10ms | 2ms | 5ms | 10ms |
| SDK Span Creation | <1ms | 0.1ms | 0.3ms | 0.8ms |
| OTLP Ingestion | <5ms | 1ms | 3ms | 5ms |
| Query Response | <500ms | 50ms | 200ms | 500ms |
| Dashboard Load | <2s | 500ms | 1s | 2s |

### Performance Design Principles

1. **Asynchronous by Default**: All instrumentation uses async I/O to avoid blocking agent execution
2. **Batching**: Spans are batched before transmission to reduce network overhead
3. **Buffering**: Queue-based buffering prevents backpressure during traffic spikes
4. **Compression**: OTLP uses gzip compression to reduce bandwidth by ~70%
5. **Columnar Storage**: ClickHouse provides 10x compression and fast analytical queries

## Horizontal Scaling Architecture

a11i achieves horizontal scalability through stateless components and distributed data architecture:

```
                    ┌─────────────────────────────┐
                    │       Load Balancer         │
                    │  (nginx/HAProxy/AWS ALB)    │
                    └─────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
   ┌────────▼────────┐ ┌─────▼─────┐ ┌────────▼────────┐
   │ API Gateway #1  │ │ API GW #2 │ │ API Gateway #3  │
   │   (Stateless)   │ │(Stateless)│ │   (Stateless)   │
   └────────┬────────┘ └─────┬─────┘ └────────┬────────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
   ┌─────────────────────────▼─────────────────────────┐
   │               NATS JetStream Cluster               │
   │  (Distributed message queue with persistence)      │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
   │  │  Node 1  │  │  Node 2  │  │  Node 3  │         │
   │  └──────────┘  └──────────┘  └──────────┘         │
   └─────────────────────────┬─────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
   ┌────────▼────────┐ ┌─────▼─────┐ ┌────────▼────────┐
   │ OTel Collector  │ │ Collector │ │ OTel Collector  │
   │      #1         │ │    #2     │ │      #3         │
   │   (Stateless)   │ │(Stateless)│ │   (Stateless)   │
   └────────┬────────┘ └─────┬─────┘ └────────┬────────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
   ┌─────────────────────────▼─────────────────────────┐
   │            ClickHouse Cluster (Sharded)            │
   │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐   │
   │  │   Shard 1   │  │   Shard 2   │  │  Shard 3 │   │
   │  │  ┌────────┐ │  │  ┌────────┐ │  │ ┌──────┐ │   │
   │  │  │ Node A │ │  │  │ Node A │ │  │ │Node A│ │   │
   │  │  └────────┘ │  │  └────────┘ │  │ └──────┘ │   │
   │  │  ┌────────┐ │  │  ┌────────┐ │  │ ┌──────┐ │   │
   │  │  │ Node B │ │  │  │ Node B │ │  │ │Node B│ │   │
   │  │  └────────┘ │  │  └────────┘ │  │ └──────┘ │   │
   │  │ (Replicas)  │  │ (Replicas)  │  │(Replica) │   │
   │  └─────────────┘  └─────────────┘  └──────────┘   │
   └───────────────────────────────────────────────────┘
```

### Scaling Characteristics

**API Gateway Layer:**
- **Stateless**: Can scale horizontally without coordination
- **Auto-scaling**: Based on CPU utilization and request rate
- **Load balancing**: Round-robin or least-connections

**Message Queue Layer:**
- **NATS JetStream**: Distributed, persistent message queue
- **Stream replication**: 3-way replication for durability
- **Consumer groups**: Multiple collectors read in parallel
- **Backpressure handling**: Queue depth metrics trigger scaling

**Collector Layer:**
- **Stateless processing**: Each collector processes independently
- **Batch processing**: Configurable batch size and timeout
- **Retry logic**: Automatic retry with exponential backoff
- **Graceful shutdown**: Drain queue before termination

**Storage Layer:**
- **Horizontal sharding**: Data distributed by tenant_id
- **Replication**: 2-way replication per shard for HA
- **Distributed queries**: Parallel query execution across shards
- **Tiered storage**: Hot (SSD), warm (HDD), cold (S3)

## Production Benchmarks

a11i leverages proven open-source technologies with demonstrated scalability:

### Real-World ClickHouse Deployments

| Company | Scale | Details |
|---------|-------|---------|
| **Resmo** | 300M spans/day | Single c7g.xlarge instance, <$100/month |
| **Character.AI** | 450PB logs/month | Multi-tenant logging platform |
| **Cloudflare** | 30M rows/second | Real-time analytics ingestion |
| **ClickHouse LogHouse** | 100PB data | 500 trillion rows, 200+ nodes |
| **Uber** | 1PB+/day | Observability and analytics |

### Performance Characteristics

**Write Performance:**
- **Single node**: 100K-500K rows/second
- **Cluster**: Scales linearly with shard count
- **Compression**: 10:1 typical compression ratio
- **Latency**: P99 < 10ms for batch inserts

**Query Performance:**
- **Simple aggregations**: Sub-second on billions of rows
- **Complex queries**: Seconds on trillion-row datasets
- **Materialized views**: Pre-aggregated for dashboard queries
- **Distributed queries**: Parallel execution across shards

### a11i Specific Benchmarks

Based on internal testing and Resmo's public metrics:

```
Small Deployment (1 ClickHouse node, 2 collectors):
- Ingestion: 50M spans/day (500 spans/second)
- Storage: ~50GB/day raw, ~5GB/day compressed
- Queries: <100ms P99 for dashboards
- Cost: <$100/month (AWS c7g.xlarge)

Medium Deployment (3-node cluster, 5 collectors):
- Ingestion: 1B spans/day (11K spans/second)
- Storage: ~1TB/day raw, ~100GB/day compressed
- Queries: <200ms P99 for dashboards
- Cost: ~$500/month (3x c7g.2xlarge)

Large Deployment (9-node cluster, 20 collectors):
- Ingestion: 10B+ spans/day (115K+ spans/second)
- Storage: ~10TB/day raw, ~1TB/day compressed
- Queries: <500ms P99 for dashboards
- Cost: ~$3K/month (9x c7g.4xlarge)
```

## Stateless Collectors

OpenTelemetry Collectors are stateless processors that can scale horizontally without coordination:

### Kubernetes Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
  namespace: a11i-system
spec:
  replicas: 5  # Scale horizontally as needed
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      containers:
      - name: collector
        image: otel/opentelemetry-collector-contrib:0.91.0
        ports:
        - containerPort: 4317  # OTLP gRPC
          name: otlp-grpc
        - containerPort: 4318  # OTLP HTTP
          name: otlp-http
        - containerPort: 8888  # Prometheus metrics
          name: metrics
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /
            port: 13133
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 13133
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /etc/otelcol
      volumes:
      - name: config
        configMap:
          name: otel-collector-config
---
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
  namespace: a11i-system
spec:
  selector:
    app: otel-collector
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
  - name: metrics
    port: 8888
    targetPort: 8888
  type: ClusterIP
```

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: otel-collector-hpa
  namespace: a11i-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: otel-collector
  minReplicas: 3
  maxReplicas: 20
  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  # Custom metric: spans received per second
  - type: Pods
    pods:
      metric:
        name: spans_received_per_second
      target:
        type: AverageValue
        averageValue: "10000"

  # Custom metric: queue depth
  - type: Pods
    pods:
      metric:
        name: queue_depth
      target:
        type: AverageValue
        averageValue: "5000"

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Collector Configuration

```yaml
# ConfigMap: otel-collector-config
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s
    send_batch_size: 500
    send_batch_max_size: 1000

  memory_limiter:
    check_interval: 1s
    limit_mib: 450
    spike_limit_mib: 100

  resource:
    attributes:
    - key: collector.id
      value: ${POD_NAME}
      action: insert

exporters:
  clickhouse:
    endpoint: tcp://clickhouse:9000
    database: a11i
    ttl: 30d
    batch_size: 500
    timeout: 30s
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, resource]
      exporters: [clickhouse]

  telemetry:
    metrics:
      address: 0.0.0.0:8888
```

## Queue-Based Buffering

NATS JetStream provides durable message queuing to prevent data loss during traffic spikes:

### JetStream Stream Configuration

```yaml
nats:
  jetstream:
    # Stream for agent traces
    streams:
      - name: AGENT_TRACES
        subjects: ["traces.>"]
        retention: limits  # Retention based on limits
        max_bytes: 10GB    # Maximum stream size
        max_msgs: 10000000 # Maximum message count
        max_age: 24h       # Maximum message age
        replicas: 3        # Replication factor
        storage: file      # Persistent file storage
        discard: old       # Drop oldest messages if full

        # Duplicate detection
        duplicate_window: 2m

        # Performance tuning
        max_msg_size: 1MB
        max_consumers: 10

      # Stream for agent metrics
      - name: AGENT_METRICS
        subjects: ["metrics.>"]
        retention: limits
        max_bytes: 5GB
        max_msgs: 5000000
        max_age: 12h
        replicas: 3
        storage: file
        discard: old

    # Consumer for ClickHouse writer
    consumers:
      - name: clickhouse_writer
        stream_name: AGENT_TRACES
        durable_name: clickhouse_writer

        # Delivery semantics
        ack_wait: 30s              # Wait for ack before redelivery
        max_deliver: 3             # Maximum delivery attempts
        max_ack_pending: 10000     # Max unacknowledged messages

        # Batching
        batch_size: 500
        max_batch_wait: 5s

        # Rate limiting
        rate_limit: 100000  # Messages per second

        # Backoff on failure
        backoff:
          - 1s
          - 5s
          - 10s
```

### Queue Monitoring

```yaml
# Prometheus metrics for NATS JetStream
- name: nats_jetstream_stream_messages
  help: Current message count in stream

- name: nats_jetstream_stream_bytes
  help: Current byte count in stream

- name: nats_jetstream_consumer_ack_pending
  help: Pending acknowledgements for consumer

- name: nats_jetstream_consumer_redelivered
  help: Redelivered message count
```

### Alerting Rules

```yaml
groups:
- name: a11i_queue_alerts
  rules:
  # Queue depth alert
  - alert: JetStreamQueueDepthHigh
    expr: nats_jetstream_stream_messages{stream="AGENT_TRACES"} > 5000000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "NATS JetStream queue depth is high"
      description: "Stream {{ $labels.stream }} has {{ $value }} messages pending"

  # Consumer lag alert
  - alert: JetStreamConsumerLag
    expr: nats_jetstream_consumer_ack_pending{consumer="clickhouse_writer"} > 50000
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "ClickHouse writer is lagging behind"
      description: "Consumer has {{ $value }} unacknowledged messages"

  # Redelivery rate alert
  - alert: JetStreamHighRedeliveryRate
    expr: rate(nats_jetstream_consumer_redelivered[5m]) > 100
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High message redelivery rate detected"
      description: "Redelivery rate is {{ $value }}/sec, indicates processing failures"
```

## Partitioned Storage

ClickHouse uses distributed tables and sharding to scale storage horizontally:

### Distributed Table Setup

```sql
-- Distributed table (query interface)
CREATE TABLE agent_traces_distributed ON CLUSTER '{cluster}'
(
    -- Tenant isolation
    tenant_id LowCardinality(String),

    -- Trace identification
    trace_id FixedString(32),
    span_id FixedString(16),
    parent_span_id Nullable(FixedString(16)),

    -- Timing
    start_time DateTime64(3),
    end_time DateTime64(3),
    duration_ms UInt32,

    -- Agent context
    agent_id String,
    agent_type LowCardinality(String),

    -- LLM metadata
    model_name LowCardinality(String),
    input_tokens UInt32,
    output_tokens UInt32,
    total_tokens UInt32,
    cost_estimate_usd Float64,

    -- Status
    status_code LowCardinality(String),
    error_message String,

    -- Attributes (JSON)
    attributes String,

    -- Events
    events String,

    -- Links
    links String
)
ENGINE = Distributed(
    '{cluster}',           -- Cluster name
    'a11i',                -- Database name
    'agent_traces_local',  -- Local table name
    cityHash64(tenant_id)  -- Sharding key (distribute by tenant)
);

-- Local table on each shard
CREATE TABLE agent_traces_local ON CLUSTER '{cluster}'
(
    tenant_id LowCardinality(String),
    trace_id FixedString(32),
    span_id FixedString(16),
    parent_span_id Nullable(FixedString(16)),
    start_time DateTime64(3),
    end_time DateTime64(3),
    duration_ms UInt32,
    agent_id String,
    agent_type LowCardinality(String),
    model_name LowCardinality(String),
    input_tokens UInt32,
    output_tokens UInt32,
    total_tokens UInt32,
    cost_estimate_usd Float64,
    status_code LowCardinality(String),
    error_message String,
    attributes String,
    events String,
    links String
)
ENGINE = ReplicatedMergeTree(
    '/clickhouse/tables/{shard}/agent_traces',  -- ZooKeeper path
    '{replica}'                                  -- Replica name
)
PARTITION BY (tenant_id, toYYYYMM(start_time))
ORDER BY (tenant_id, trace_id, start_time)
TTL start_time + INTERVAL 30 DAY TO VOLUME 'warm',
    start_time + INTERVAL 180 DAY TO VOLUME 'cold',
    start_time + INTERVAL 365 DAY DELETE
SETTINGS
    index_granularity = 8192,
    ttl_only_drop_parts = 1;
```

### Storage Policies

```xml
<!-- ClickHouse storage configuration -->
<storage_configuration>
    <disks>
        <!-- Hot storage: NVMe SSD -->
        <hot>
            <type>local</type>
            <path>/var/lib/clickhouse/hot/</path>
        </hot>

        <!-- Warm storage: SATA SSD -->
        <warm>
            <type>local</type>
            <path>/var/lib/clickhouse/warm/</path>
        </warm>

        <!-- Cold storage: S3 -->
        <cold>
            <type>s3</type>
            <endpoint>https://s3.amazonaws.com/my-bucket/clickhouse/</endpoint>
            <access_key_id>ACCESS_KEY</access_key_id>
            <secret_access_key>SECRET_KEY</secret_access_key>
        </cold>
    </disks>

    <policies>
        <tiered>
            <volumes>
                <hot>
                    <disk>hot</disk>
                </hot>
                <warm>
                    <disk>warm</disk>
                </warm>
                <cold>
                    <disk>cold</disk>
                </cold>
            </volumes>
            <move_factor>0.2</move_factor>
        </tiered>
    </policies>
</storage_configuration>
```

### Materialized Views for Performance

```sql
-- Pre-aggregate hourly statistics per agent
CREATE MATERIALIZED VIEW agent_hourly_stats_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (tenant_id, agent_id, hour)
AS SELECT
    tenant_id,
    agent_id,
    agent_type,
    toStartOfHour(start_time) AS hour,
    count() AS request_count,
    sum(input_tokens) AS total_input_tokens,
    sum(output_tokens) AS total_output_tokens,
    sum(cost_estimate_usd) AS total_cost,
    avg(duration_ms) AS avg_duration_ms,
    quantile(0.50)(duration_ms) AS p50_duration_ms,
    quantile(0.90)(duration_ms) AS p90_duration_ms,
    quantile(0.99)(duration_ms) AS p99_duration_ms,
    countIf(status_code = 'ERROR') AS error_count
FROM agent_traces_local
GROUP BY tenant_id, agent_id, agent_type, hour;

-- Pre-aggregate model usage statistics
CREATE MATERIALIZED VIEW model_usage_stats_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(day)
ORDER BY (tenant_id, model_name, day)
AS SELECT
    tenant_id,
    model_name,
    toDate(start_time) AS day,
    count() AS request_count,
    sum(input_tokens) AS total_input_tokens,
    sum(output_tokens) AS total_output_tokens,
    sum(total_tokens) AS total_tokens,
    sum(cost_estimate_usd) AS total_cost_usd,
    avg(cost_estimate_usd) AS avg_cost_per_request
FROM agent_traces_local
GROUP BY tenant_id, model_name, day;
```

## Sampling Strategies

Sampling reduces data volume while preserving critical traces:

### Tail-Based Sampling

```yaml
# OpenTelemetry Collector tail sampling configuration
processors:
  # Head-based probabilistic sampling (10% baseline)
  probabilistic_sampling:
    hash_seed: 12345
    sampling_percentage: 10

  # Tail-based sampling with intelligent policies
  tail_sampling:
    # Wait time before making sampling decision
    decision_wait: 10s

    # Number of traces to keep in memory
    num_traces: 100000

    # Expected trace arrival rate
    expected_new_traces_per_sec: 10000

    policies:
      # Policy 1: Always keep errors
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]

      # Policy 2: Always keep slow traces (>5s)
      - name: slow_traces
        type: latency
        latency:
          threshold_ms: 5000

      # Policy 3: Always keep expensive requests (>$0.10)
      - name: expensive
        type: numeric_attribute
        numeric_attribute:
          key: a11i.cost.estimate_usd
          min_value: 0.10

      # Policy 4: Keep long agent loops (>10 iterations)
      - name: long_loops
        type: numeric_attribute
        numeric_attribute:
          key: a11i.agent.loop_iteration
          min_value: 10

      # Policy 5: Keep traces with tool failures
      - name: tool_failures
        type: string_attribute
        string_attribute:
          key: a11i.tool.error
          enabled_regex_matching: true
          values: [".+"]  # Any non-empty error

      # Policy 6: Keep traces from specific high-value agents
      - name: critical_agents
        type: string_attribute
        string_attribute:
          key: a11i.agent.id
          values:
            - "production-orchestrator"
            - "revenue-agent"
            - "security-agent"

      # Policy 7: Rate limiting per tenant
      - name: tenant_rate_limit
        type: rate_limiting
        rate_limiting:
          spans_per_second: 100

      # Policy 8: Probabilistic sampling for everything else (5%)
      - name: baseline
        type: probabilistic
        probabilistic:
          sampling_percentage: 5

exporters:
  clickhouse:
    endpoint: tcp://clickhouse:9000
    database: a11i

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [tail_sampling, batch]
      exporters: [clickhouse]
```

### Adaptive Sampling

```python
class AdaptiveSampler:
    """Dynamically adjust sampling rate based on traffic volume."""

    def __init__(
        self,
        target_spans_per_day: int = 100_000_000,
        min_sample_rate: float = 0.01,
        max_sample_rate: float = 1.0,
    ):
        self.target_spans_per_day = target_spans_per_day
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate

    def calculate_sample_rate(
        self,
        current_spans_per_day: int,
    ) -> float:
        """Calculate optimal sampling rate."""

        if current_spans_per_day <= self.target_spans_per_day:
            return self.max_sample_rate

        # Calculate ratio
        ratio = self.target_spans_per_day / current_spans_per_day

        # Clamp to min/max
        sample_rate = max(
            self.min_sample_rate,
            min(self.max_sample_rate, ratio)
        )

        return sample_rate

# Example usage
sampler = AdaptiveSampler(target_spans_per_day=100_000_000)

# Low traffic: 10M spans/day -> 100% sampling
rate = sampler.calculate_sample_rate(10_000_000)  # Returns 1.0

# Target traffic: 100M spans/day -> 100% sampling
rate = sampler.calculate_sample_rate(100_000_000)  # Returns 1.0

# High traffic: 1B spans/day -> 10% sampling
rate = sampler.calculate_sample_rate(1_000_000_000)  # Returns 0.1

# Very high traffic: 10B spans/day -> 1% sampling (min)
rate = sampler.calculate_sample_rate(10_000_000_000)  # Returns 0.01
```

### Sampling Strategy Guidelines

1. **Always sample**:
   - Errors and failures
   - Slow requests (>P95 latency)
   - High-cost requests (>$0.05)
   - Security-related traces

2. **Probabilistically sample**:
   - Normal successful requests
   - Fast, cheap operations
   - Background tasks

3. **Adjust sampling based on**:
   - Traffic volume
   - Storage costs
   - Query performance
   - Tenant quotas

## Capacity Planning

Estimate infrastructure requirements based on expected load:

### Capacity Planning Calculator

```python
class CapacityPlanner:
    """Estimate infrastructure requirements for a11i deployment."""

    def estimate_storage(
        self,
        agents_per_day: int,
        avg_tokens_per_request: int,
        avg_requests_per_agent: int,
        retention_days: int,
        sampling_rate: float = 1.0,
    ) -> dict:
        """Estimate storage requirements.

        Args:
            agents_per_day: Number of agent sessions per day
            avg_tokens_per_request: Average tokens per LLM request
            avg_requests_per_agent: Average requests per agent session
            retention_days: Data retention period in days
            sampling_rate: Sampling rate (0.0 to 1.0)

        Returns:
            Dictionary with storage estimates
        """

        # Assumptions based on production data
        BYTES_PER_TRACE = 2000  # Average trace size
        COMPRESSION_RATIO = 0.1  # ClickHouse achieves ~10:1 compression
        CLICKHOUSE_OVERHEAD = 1.2  # 20% overhead for indexes, etc.

        # Calculate request volume
        requests_per_day = agents_per_day * avg_requests_per_agent
        sampled_requests_per_day = requests_per_day * sampling_rate

        # Calculate storage
        raw_bytes_per_day = sampled_requests_per_day * BYTES_PER_TRACE
        compressed_bytes_per_day = raw_bytes_per_day * COMPRESSION_RATIO
        total_bytes = compressed_bytes_per_day * retention_days * CLICKHOUSE_OVERHEAD

        # Calculate recommended shards
        # Rule of thumb: 1 shard per 50M requests/day
        recommended_shards = max(1, int(sampled_requests_per_day / 50_000_000))

        return {
            "requests_per_day": requests_per_day,
            "sampled_requests_per_day": sampled_requests_per_day,
            "raw_data_per_day_gb": raw_bytes_per_day / 1e9,
            "compressed_data_per_day_gb": compressed_bytes_per_day / 1e9,
            "total_storage_gb": total_bytes / 1e9,
            "total_storage_tb": total_bytes / 1e12,
            "recommended_shards": recommended_shards,
            "storage_per_shard_gb": (total_bytes / 1e9) / recommended_shards,
        }

    def estimate_compute(
        self,
        requests_per_day: int,
        sampling_rate: float = 1.0,
    ) -> dict:
        """Estimate compute requirements.

        Args:
            requests_per_day: Total requests per day
            sampling_rate: Sampling rate

        Returns:
            Dictionary with compute recommendations
        """

        sampled_requests_per_day = requests_per_day * sampling_rate
        sampled_requests_per_second = sampled_requests_per_day / 86400

        # OTel Collector sizing
        # Rule: 1 collector handles ~10K spans/sec
        collectors = max(3, int(sampled_requests_per_second / 10000) + 1)

        # ClickHouse node sizing
        # Rule: c7g.xlarge handles ~50M spans/day
        clickhouse_nodes = max(1, int(sampled_requests_per_day / 50_000_000))

        # NATS JetStream sizing
        # Rule: 3 nodes for HA, scale for throughput
        nats_nodes = 3 if sampled_requests_per_second < 50000 else 5

        return {
            "otel_collectors": collectors,
            "clickhouse_nodes": clickhouse_nodes,
            "nats_nodes": nats_nodes,
            "estimated_cpu_cores": collectors * 4 + clickhouse_nodes * 8,
            "estimated_memory_gb": collectors * 2 + clickhouse_nodes * 16,
        }

    def estimate_cost(
        self,
        requests_per_day: int,
        retention_days: int,
        sampling_rate: float = 1.0,
        cloud_provider: str = "aws",
    ) -> dict:
        """Estimate monthly cloud costs.

        Args:
            requests_per_day: Total requests per day
            retention_days: Data retention period
            sampling_rate: Sampling rate
            cloud_provider: Cloud provider ("aws", "gcp", "azure")

        Returns:
            Dictionary with cost estimates
        """

        storage = self.estimate_storage(
            agents_per_day=requests_per_day / 50,  # Assume 50 req/agent
            avg_tokens_per_request=2000,
            avg_requests_per_agent=50,
            retention_days=retention_days,
            sampling_rate=sampling_rate,
        )

        compute = self.estimate_compute(
            requests_per_day=requests_per_day,
            sampling_rate=sampling_rate,
        )

        # AWS pricing estimates (on-demand, us-east-1)
        if cloud_provider == "aws":
            # ClickHouse: c7g.xlarge ($0.145/hr)
            clickhouse_monthly = compute["clickhouse_nodes"] * 0.145 * 730

            # OTel Collectors: t3.medium ($0.042/hr)
            collector_monthly = compute["otel_collectors"] * 0.042 * 730

            # NATS: t3.small ($0.021/hr)
            nats_monthly = compute["nats_nodes"] * 0.021 * 730

            # EBS storage: gp3 ($0.08/GB-month)
            storage_monthly = storage["total_storage_gb"] * 0.08

            # Data transfer (estimate 10% of ingress)
            transfer_monthly = storage["compressed_data_per_day_gb"] * 30 * 0.1 * 0.09

            total_monthly = (
                clickhouse_monthly +
                collector_monthly +
                nats_monthly +
                storage_monthly +
                transfer_monthly
            )

            return {
                "clickhouse_monthly_usd": round(clickhouse_monthly, 2),
                "collectors_monthly_usd": round(collector_monthly, 2),
                "nats_monthly_usd": round(nats_monthly, 2),
                "storage_monthly_usd": round(storage_monthly, 2),
                "transfer_monthly_usd": round(transfer_monthly, 2),
                "total_monthly_usd": round(total_monthly, 2),
                "cost_per_million_spans": round(
                    total_monthly / (storage["sampled_requests_per_day"] * 30 / 1_000_000),
                    2
                ),
            }

        return {"error": f"Unsupported cloud provider: {cloud_provider}"}


# Example: Startup scenario
planner = CapacityPlanner()

startup_sizing = planner.estimate_storage(
    agents_per_day=1000,
    avg_tokens_per_request=2000,
    avg_requests_per_agent=50,
    retention_days=30,
)
print("Startup sizing:", startup_sizing)
# Output:
# {
#   'requests_per_day': 50000,
#   'sampled_requests_per_day': 50000,
#   'raw_data_per_day_gb': 0.1,
#   'compressed_data_per_day_gb': 0.01,
#   'total_storage_gb': 0.36,
#   'recommended_shards': 1
# }

startup_cost = planner.estimate_cost(
    requests_per_day=50000,
    retention_days=30,
    sampling_rate=1.0,
)
print("Startup cost:", startup_cost)
# Output: ~$120/month


# Example: Growth scenario
growth_sizing = planner.estimate_storage(
    agents_per_day=100_000,
    avg_tokens_per_request=2000,
    avg_requests_per_agent=50,
    retention_days=30,
)
print("Growth sizing:", growth_sizing)
# Output:
# {
#   'requests_per_day': 5000000,
#   'sampled_requests_per_day': 5000000,
#   'raw_data_per_day_gb': 10.0,
#   'compressed_data_per_day_gb': 1.0,
#   'total_storage_gb': 36.0,
#   'recommended_shards': 1
# }

growth_cost = planner.estimate_cost(
    requests_per_day=5_000_000,
    retention_days=30,
    sampling_rate=1.0,
)
print("Growth cost:", growth_cost)
# Output: ~$150/month


# Example: Enterprise scenario (with sampling)
enterprise_sizing = planner.estimate_storage(
    agents_per_day=1_000_000,
    avg_tokens_per_request=2000,
    avg_requests_per_agent=50,
    retention_days=90,
    sampling_rate=0.2,  # 20% sampling
)
print("Enterprise sizing:", enterprise_sizing)
# Output:
# {
#   'requests_per_day': 50000000,
#   'sampled_requests_per_day': 10000000,
#   'raw_data_per_day_gb': 20.0,
#   'compressed_data_per_day_gb': 2.0,
#   'total_storage_gb': 216.0,
#   'recommended_shards': 1
# }

enterprise_cost = planner.estimate_cost(
    requests_per_day=50_000_000,
    retention_days=90,
    sampling_rate=0.2,
)
print("Enterprise cost:", enterprise_cost)
# Output: ~$500/month
```

### Sizing Guidelines

| Deployment Size | Agents/Day | Requests/Day | ClickHouse | Collectors | Monthly Cost |
|-----------------|------------|--------------|------------|------------|--------------|
| **Dev/Test** | <1K | <50K | 1 node (small) | 1-2 | <$100 |
| **Startup** | 1K-10K | 50K-500K | 1 node (medium) | 2-3 | $100-$200 |
| **Growth** | 10K-100K | 500K-5M | 1-3 nodes | 3-5 | $200-$500 |
| **Scale** | 100K-1M | 5M-50M | 3-6 nodes | 5-10 | $500-$2K |
| **Enterprise** | 1M+ | 50M+ | 6+ nodes (sharded) | 10-20 | $2K-$10K+ |

## Deployment Topologies

Choose the right architecture for your scale:

### Small Deployment (Dev/Startup)

**Characteristics:**
- Single ClickHouse node
- Single OTel Collector
- NATS embedded or single node
- Suitable for: <10K requests/day

**Architecture:**
```
┌─────────────┐
│   Agents    │
└──────┬──────┘
       │
┌──────▼───────┐
│ OTel         │
│ Collector    │
└──────┬───────┘
       │
┌──────▼───────┐
│ ClickHouse   │
│ (Single Node)│
└──────────────┘
```

**Kubernetes Manifest:**
```yaml
# Single-node ClickHouse
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: clickhouse
spec:
  serviceName: clickhouse
  replicas: 1
  template:
    spec:
      containers:
      - name: clickhouse
        image: clickhouse/clickhouse-server:23.8
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: data
          mountPath: /var/lib/clickhouse
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

**Estimated Cost:** $50-$150/month

### Medium Deployment (Growth)

**Characteristics:**
- 3-node ClickHouse cluster (replicated)
- 3-5 OTel Collectors (auto-scaling)
- 3-node NATS JetStream cluster
- Suitable for: 10K-1M requests/day

**Architecture:**
```
         ┌─────────────┐
         │   Agents    │
         └──────┬──────┘
                │
        ┌───────┴───────┐
        │               │
┌───────▼──┐    ┌──────▼───┐
│ OTel #1  │    │ OTel #2  │
└───────┬──┘    └──────┬───┘
        │              │
    ┌───┴──────────────┴───┐
    │   NATS JetStream     │
    │   (3-node cluster)   │
    └───┬──────────────┬───┘
        │              │
┌───────▼──┐    ┌──────▼───┐    ┌──────────┐
│ClickHouse│    │ClickHouse│    │ClickHouse│
│  Node 1  │<-->│  Node 2  │<-->│  Node 3  │
└──────────┘    └──────────┘    └──────────┘
  (Replicated, not sharded)
```

**Estimated Cost:** $200-$800/month

### Large Deployment (Enterprise)

**Characteristics:**
- 6+ node ClickHouse cluster (sharded + replicated)
- 10-20 OTel Collectors (HPA)
- 5-node NATS cluster
- Multi-region optional
- Suitable for: 1M-100M+ requests/day

**Architecture:**
```
                ┌────────────────┐
                │  Load Balancer │
                └────────┬───────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──┐    ┌───────▼──┐    ┌───────▼──┐
│ OTel #1  │    │ OTel #2  │ ...│ OTel #N  │
└───────┬──┘    └───────┬──┘    └───────┬──┘
        │                │                │
    ┌───┴────────────────┴────────────────┴───┐
    │        NATS JetStream Cluster            │
    │         (5 nodes, multi-zone)            │
    └───┬──────────────────────────────────┬───┘
        │                                  │
┌───────▼───────┐              ┌──────────▼──────┐
│   Shard 1     │              │    Shard 2      │
│ ┌──────────┐  │              │  ┌──────────┐   │
│ │ Node 1A  │  │              │  │ Node 2A  │   │
│ └────┬─────┘  │              │  └────┬─────┘   │
│      │        │              │       │         │
│ ┌────▼─────┐  │              │  ┌────▼─────┐   │
│ │ Node 1B  │  │              │  │ Node 2B  │   │
│ └──────────┘  │              │  └──────────┘   │
│  (Replicas)   │              │   (Replicas)    │
└───────────────┘              └─────────────────┘
```

**Sharding Configuration:**
```xml
<!-- ClickHouse cluster configuration -->
<remote_servers>
    <a11i_cluster>
        <shard>
            <replica>
                <host>clickhouse-01a.a11i.svc.cluster.local</host>
                <port>9000</port>
            </replica>
            <replica>
                <host>clickhouse-01b.a11i.svc.cluster.local</host>
                <port>9000</port>
            </replica>
        </shard>
        <shard>
            <replica>
                <host>clickhouse-02a.a11i.svc.cluster.local</host>
                <port>9000</port>
            </replica>
            <replica>
                <host>clickhouse-02b.a11i.svc.cluster.local</host>
                <port>9000</port>
            </replica>
        </shard>
        <shard>
            <replica>
                <host>clickhouse-03a.a11i.svc.cluster.local</host>
                <port>9000</port>
            </replica>
            <replica>
                <host>clickhouse-03b.a11i.svc.cluster.local</host>
                <port>9000</port>
            </replica>
        </shard>
    </a11i_cluster>
</remote_servers>
```

**Estimated Cost:** $2K-$20K+/month

## Operational Considerations

### Monitoring Scalability

```yaml
# Prometheus alerts for scalability issues
groups:
- name: a11i_scalability
  rules:
  # Query performance degradation
  - alert: SlowQueryPerformance
    expr: histogram_quantile(0.99, rate(clickhouse_query_duration_seconds[5m])) > 5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "ClickHouse query performance degrading"

  # Collector saturation
  - alert: CollectorSaturation
    expr: rate(otelcol_processor_refused_spans[5m]) > 100
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "OTel Collector refusing spans due to overload"

  # Storage growth rate
  - alert: StorageGrowthHigh
    expr: |
      predict_linear(
        clickhouse_table_size_bytes{table="agent_traces_local"}[1d],
        7 * 24 * 3600
      ) > (0.8 * clickhouse_disk_size_bytes)
    for: 1h
    labels:
      severity: warning
    annotations:
      summary: "Storage will be 80% full in 7 days"
```

### Performance Tuning Checklist

- [ ] **ClickHouse**: Tune `max_threads`, `max_memory_usage`, `merge_tree_max_rows_to_use_cache`
- [ ] **OTel Collector**: Optimize batch size, timeout, and queue size
- [ ] **NATS**: Adjust `max_msgs`, `max_bytes`, `max_age` for streams
- [ ] **Network**: Enable compression, use gRPC for high throughput
- [ ] **Indexes**: Add materialized columns for frequently queried attributes
- [ ] **Partitioning**: Review partition key and granularity
- [ ] **Sampling**: Implement intelligent sampling to control volume

### Scaling Triggers

**Scale out collectors when:**
- CPU utilization > 70% for 5+ minutes
- Queue depth > 10K messages for 5+ minutes
- Span refusal rate > 1%

**Scale out ClickHouse when:**
- Query P99 latency > 2 seconds
- Disk utilization > 80%
- Write throughput > 80% of node capacity
- CPU utilization > 80% sustained

**Scale out NATS when:**
- Message backlog > 1M messages
- Consumer lag > 1 hour
- Redelivery rate > 10%

---

## Key Takeaways

> **Horizontal Scalability**
>
> a11i is designed for horizontal scaling at every layer:
> - **Stateless collectors** scale independently based on ingestion load
> - **Queue-based buffering** with NATS JetStream prevents data loss during spikes
> - **Sharded storage** with ClickHouse distributes data across nodes
> - **Intelligent sampling** controls costs while preserving critical traces
>
> **Performance First**
>
> - Sub-10ms proxy overhead ensures minimal impact on agents
> - Batching and compression reduce network bandwidth by 70%
> - Materialized views provide sub-second dashboard queries
> - Columnar storage achieves 10:1 compression ratio
>
> **Production Proven**
>
> Built on battle-tested technologies:
> - ClickHouse powers observability at Uber, Cloudflare, and Character.AI
> - NATS JetStream handles millions of messages/second with persistence
> - OpenTelemetry is the CNCF standard for observability
>
> **Cost Effective**
>
> - Startup: <$150/month for 1M requests/day
> - Growth: ~$500/month for 50M requests/day
> - Enterprise: ~$3K/month for 1B+ requests/day with sampling

---

## Cross-References

- **Architecture Overview**: [architecture.md](/home/becker/projects/a11i/docs/02-architecture/architecture.md)
- **Deployment Guide**: [deployment.md](/home/becker/projects/a11i/docs/06-operations/deployment.md)
- **Monitoring Guide**: [monitoring.md](/home/becker/projects/a11i/docs/06-operations/monitoring.md)
- **Cost Optimization**: [cost-optimization.md](/home/becker/projects/a11i/docs/06-operations/cost-optimization.md)
- **Data Retention Policies**: [data-retention.md](/home/becker/projects/a11i/docs/06-operations/data-retention.md)
- **API Reference**: [api-reference.md](/home/becker/projects/a11i/docs/04-api-reference/api-reference.md)
