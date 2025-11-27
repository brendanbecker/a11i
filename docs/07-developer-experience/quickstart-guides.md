---
title: Quickstart Guides
description: Get started with a11i in minutes with these step-by-step guides
category: Developer Experience
tags: [quickstart, getting-started, tutorial, setup]
difficulty: beginner
estimated_time: 5-30 minutes
last_updated: 2025-11-26
---

# Quickstart Guides

Get started with a11i in minutes. Choose the guide that matches your use case.

## Table of Contents

1. [Guide 1: Monitor LangChain in 5 Minutes](#guide-1-monitor-langchain-in-5-minutes)
2. [Guide 2: Add a11i to Existing Application](#guide-2-add-a11i-to-existing-application)
3. [Guide 3: Self-Host a11i Locally](#guide-3-self-host-a11i-locally)
4. [Guide 4: Deploy to Kubernetes](#guide-4-deploy-to-kubernetes)
5. [Verification and Troubleshooting](#verification-and-troubleshooting)

---

## Guide 1: Monitor LangChain in 5 Minutes

**Time Required:** 5 minutes
**Prerequisites:** Python 3.8+, OpenAI API key
**Best For:** Quick proof of concept with LangChain applications

### Step 1: Install Dependencies

```bash
# Install a11i SDK and LangChain
pip install a11i langchain openai
```

### Step 2: Set Environment Variables

```bash
# Set your a11i API key
export A11I_API_KEY=your_key_here

# Set your OpenAI API key
export OPENAI_API_KEY=your_openai_key

# Optional: Set project name
export A11I_PROJECT=langchain-demo
```

**For Windows (PowerShell):**
```powershell
$env:A11I_API_KEY="your_key_here"
$env:OPENAI_API_KEY="your_openai_key"
```

### Step 3: Create Your LangChain Application

Create a file called `app.py`:

```python
# app.py - LangChain agent with a11i monitoring
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from a11i.integrations.langchain import A11iCallbackHandler

# Initialize a11i callback handler
handler = A11iCallbackHandler(
    project="langchain-demo",
    environment="development",
    session_id="demo-session-1"
)

# Create LLM with a11i monitoring
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    callbacks=[handler]
)

# Define simple tools
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def word_count(text: str) -> str:
    """Counts words in text"""
    return f"Word count: {len(text.split())}"

tools = [
    Tool(
        name="calculator",
        func=calculator,
        description="Calculates mathematical expressions. Input should be a valid Python expression."
    ),
    Tool(
        name="word_count",
        func=word_count,
        description="Counts the number of words in a given text."
    )
]

# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent and executor
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[handler],
    verbose=True
)

# Run the agent
if __name__ == "__main__":
    result = executor.invoke({
        "input": "What is 25 * 4? Then count the words in 'Hello world from a11i'."
    })

    print("\n" + "="*50)
    print("RESULT:", result["output"])
    print("="*50)

    # Get trace URL
    print(f"\nView trace at: {handler.get_trace_url()}")
```

### Step 4: Run and View Traces

```bash
# Run the application
python app.py

# Expected output:
# > Entering new AgentExecutor chain...
# ...tool calls and reasoning...
#
# ==================================================
# RESULT: 25 * 4 equals 100. The phrase 'Hello world from a11i' contains 4 words.
# ==================================================
#
# View trace at: https://app.a11i.dev/traces/abc123...
```

### Step 5: Explore Your Traces

1. Open the trace URL in your browser
2. View the agent's reasoning chain
3. Inspect LLM calls, tool usage, and latencies
4. Analyze token usage and costs

### What You'll See in a11i

- **Agent Execution Flow**: Visual graph of agent reasoning
- **LLM Calls**: Every GPT-4 call with prompts and responses
- **Tool Invocations**: Calculator and word_count tool usage
- **Performance Metrics**: Latency per step, total execution time
- **Cost Tracking**: Token usage and estimated costs

### Next Steps

- Add more complex tools
- Implement multi-agent workflows
- Set up alerts for slow traces
- Track production usage patterns

---

## Guide 2: Add a11i to Existing Application

**Time Required:** 10-15 minutes
**Prerequisites:** Existing LLM application
**Best For:** Instrumenting production applications with minimal code changes

### Step 1: Install a11i SDK

**Python:**
```bash
pip install a11i

# For specific integrations
pip install a11i[openai]      # OpenAI support
pip install a11i[anthropic]   # Anthropic Claude support
pip install a11i[langchain]   # LangChain support
pip install a11i[all]         # All integrations
```

**Node.js:**
```bash
npm install @a11i/sdk

# Or with Yarn
yarn add @a11i/sdk
```

### Step 2: Initialize a11i

Add initialization code to your application startup:

**Python:**
```python
# app.py or __init__.py
import a11i

# Initialize with API key from environment variable
a11i.init(
    api_key="your_key_here",  # Or use A11I_API_KEY env var
    project="my-ai-project",
    environment="production",

    # Optional configuration
    auto_instrument=True,  # Automatically instrument supported libraries
    capture_input=True,    # Capture LLM inputs
    capture_output=True,   # Capture LLM outputs
    sample_rate=1.0,       # Sample 100% of traces (reduce for high volume)
)

print("a11i initialized successfully")
```

**TypeScript/JavaScript:**
```typescript
// index.ts or app.ts
import { A11i } from '@a11i/sdk';

const a11i = new A11i({
  apiKey: process.env.A11I_API_KEY || 'your_key_here',
  project: 'my-ai-project',
  environment: 'production',

  // Optional configuration
  autoInstrument: true,
  captureInput: true,
  captureOutput: true,
  sampleRate: 1.0,
});

await a11i.initialize();
console.log('a11i initialized successfully');
```

### Step 3: Instrument Your LLM Calls

#### Option A: Automatic Instrumentation (Recommended)

**Python:**
```python
from a11i import auto_instrument

# Automatically trace all supported libraries
auto_instrument([
    "openai",      # OpenAI SDK
    "anthropic",   # Anthropic SDK
    "langchain",   # LangChain
    "llamaindex",  # LlamaIndex
])

# Your existing code works without changes
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# This call is automatically traced!
```

**TypeScript:**
```typescript
import { autoInstrument } from '@a11i/sdk';

// Automatically trace all OpenAI calls
autoInstrument(['openai']);

// Your existing code works without changes
import OpenAI from 'openai';

const openai = new OpenAI();
const response = await openai.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello!' }],
});
// This call is automatically traced!
```

#### Option B: Manual Instrumentation (More Control)

**Python:**
```python
from a11i import observe, span
import openai

@observe(name="chat_completion", capture_io=True)
def call_gpt(messages: list, model: str = "gpt-4") -> str:
    """Call GPT with automatic tracing"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

@observe(name="rag_pipeline")
def rag_query(question: str) -> str:
    """RAG pipeline with nested spans"""

    # Retrieve relevant documents
    with span("retrieve_documents") as s:
        docs = vector_db.search(question, top_k=5)
        s.set_attribute("num_docs", len(docs))

    # Generate context
    with span("generate_context"):
        context = "\n".join([d.content for d in docs])

    # Generate answer
    with span("generate_answer"):
        messages = [
            {"role": "system", "content": "Answer based on context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        answer = call_gpt(messages)

    return answer

# Use the instrumented function
result = rag_query("What is a11i?")
```

**TypeScript:**
```typescript
import { observe, span } from '@a11i/sdk';
import OpenAI from 'openai';

const openai = new OpenAI();

@observe({ name: 'chat_completion', captureIO: true })
async function callGPT(messages: any[], model = 'gpt-4'): Promise<string> {
  const response = await openai.chat.completions.create({
    model,
    messages,
  });
  return response.choices[0].message.content;
}

@observe({ name: 'rag_pipeline' })
async function ragQuery(question: string): Promise<string> {
  // Retrieve documents
  const docs = await span('retrieve_documents', async (s) => {
    const results = await vectorDB.search(question, { topK: 5 });
    s.setAttribute('num_docs', results.length);
    return results;
  });

  // Generate context
  const context = await span('generate_context', async () => {
    return docs.map(d => d.content).join('\n');
  });

  // Generate answer
  const answer = await span('generate_answer', async () => {
    const messages = [
      { role: 'system', content: 'Answer based on context.' },
      { role: 'user', content: `Context:\n${context}\n\nQuestion: ${question}` }
    ];
    return await callGPT(messages);
  });

  return answer;
}

// Use the instrumented function
const result = await ragQuery('What is a11i?');
```

### Step 4: Track Agent Loops (Optional)

For agentic workflows with multiple iterations:

**Python:**
```python
from a11i import agent_loop, span

@agent_loop(name="research_agent", max_iterations=10)
async def research_agent(query: str) -> str:
    """Research agent with automatic loop tracking"""
    done = False
    iteration = 0
    context = ""

    while not done and iteration < 10:
        # Think
        with span("think") as s:
            thought = await think(query, context)
            s.set_attribute("thought", thought)

        # Act
        with span("act") as s:
            action = await decide_action(thought)
            s.set_attribute("action_type", action.type)

        # Observe
        with span("observe") as s:
            observation = await execute_action(action)
            context += f"\n{observation}"
            s.set_attribute("observation", observation)

        # Check if done
        done = is_complete(observation)
        iteration += 1

    # Generate final answer
    with span("final_answer"):
        answer = await generate_answer(query, context)

    return answer

# a11i automatically tracks:
# - Number of iterations
# - Time per iteration
# - Total tokens used
# - Cost per iteration
# - Loop exit reason (max iterations, completion, error)
result = await research_agent("Explain quantum computing")
```

**TypeScript:**
```typescript
import { agentLoop, span } from '@a11i/sdk';

@agentLoop({ name: 'research_agent', maxIterations: 10 })
async function researchAgent(query: string): Promise<string> {
  let done = false;
  let iteration = 0;
  let context = '';

  while (!done && iteration < 10) {
    // Think
    const thought = await span('think', async (s) => {
      const result = await think(query, context);
      s.setAttribute('thought', result);
      return result;
    });

    // Act
    const action = await span('act', async (s) => {
      const result = await decideAction(thought);
      s.setAttribute('action_type', result.type);
      return result;
    });

    // Observe
    const observation = await span('observe', async (s) => {
      const result = await executeAction(action);
      context += `\n${result}`;
      s.setAttribute('observation', result);
      return result;
    });

    done = isComplete(observation);
    iteration++;
  }

  // Generate final answer
  const answer = await span('final_answer', async () => {
    return await generateAnswer(query, context);
  });

  return answer;
}

const result = await researchAgent('Explain quantum computing');
```

### Step 5: Deploy and Monitor

```bash
# Your application now sends traces to a11i
python app.py

# View in a11i dashboard
# - Real-time traces
# - Performance analytics
# - Cost tracking
# - Error monitoring
```

### Migration Checklist

- [ ] Install a11i SDK
- [ ] Initialize a11i in application startup
- [ ] Choose instrumentation strategy (auto vs manual)
- [ ] Add instrumentation to critical paths
- [ ] Test in development environment
- [ ] Set appropriate sample rate for production
- [ ] Configure alerts and dashboards
- [ ] Monitor cost and performance impact

---

## Guide 3: Self-Host a11i Locally

**Time Required:** 15-20 minutes
**Prerequisites:** Docker and Docker Compose installed
**Best For:** Data privacy requirements, development environments, air-gapped deployments

### Step 1: Clone the Repository

```bash
# Clone a11i repository
git clone https://github.com/a11i/a11i.git
cd a11i

# Or download specific release
wget https://github.com/a11i/a11i/releases/download/v1.0.0/a11i-v1.0.0.tar.gz
tar -xzf a11i-v1.0.0.tar.gz
cd a11i-v1.0.0
```

### Step 2: Review Docker Compose Configuration

The repository includes a complete `docker-compose.yaml`:

```yaml
# docker-compose.yaml
version: '3.8'

services:
  # a11i Proxy - Routes LLM requests and adds tracing
  proxy:
    image: a11i/proxy:latest
    container_name: a11i-proxy
    ports:
      - "8080:8080"
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317
      - UPSTREAM_OPENAI=https://api.openai.com
      - UPSTREAM_ANTHROPIC=https://api.anthropic.com
      - LOG_LEVEL=info
    depends_on:
      - collector
    restart: unless-stopped

  # OpenTelemetry Collector - Receives and processes traces
  collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: a11i-collector
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
      - "8888:8888"  # Prometheus metrics
    volumes:
      - ./config/otel-collector.yaml:/etc/otel-collector-config.yaml
    command: ["--config=/etc/otel-collector-config.yaml"]
    depends_on:
      - clickhouse
    restart: unless-stopped

  # ClickHouse - Time-series database for trace storage
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    container_name: a11i-clickhouse
    ports:
      - "8123:8123"  # HTTP interface
      - "9000:9000"  # Native protocol
    volumes:
      - ./data/clickhouse:/var/lib/clickhouse
      - ./config/clickhouse/init.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      - CLICKHOUSE_DB=a11i
      - CLICKHOUSE_USER=a11i
      - CLICKHOUSE_PASSWORD=a11i_password
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    restart: unless-stopped

  # a11i UI - Web dashboard for viewing traces
  ui:
    image: a11i/ui:latest
    container_name: a11i-ui
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=http://a11i:a11i_password@clickhouse:8123/a11i
      - COLLECTOR_ENDPOINT=http://collector:4318
      - API_BASE_URL=http://localhost:3000
    depends_on:
      - clickhouse
      - collector
    restart: unless-stopped

  # Redis - Cache for UI and query acceleration (optional)
  redis:
    image: redis:alpine
    container_name: a11i-redis
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    restart: unless-stopped

volumes:
  clickhouse-data:
  redis-data:

networks:
  default:
    name: a11i-network
```

### Step 3: Configure OpenTelemetry Collector

Create the collector configuration:

```bash
# Create config directory
mkdir -p config
```

Create `config/otel-collector.yaml`:

```yaml
# config/otel-collector.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 5s
    send_batch_size: 200
    send_batch_max_size: 500

  memory_limiter:
    check_interval: 1s
    limit_mib: 512

  # Add custom attributes
  attributes:
    actions:
      - key: deployment.environment
        value: self-hosted
        action: insert

exporters:
  clickhouse:
    endpoint: tcp://clickhouse:9000?database=a11i
    username: a11i
    password: a11i_password
    traces_table_name: otel_traces
    timeout: 10s
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s

  # Debug exporter for troubleshooting
  logging:
    loglevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, attributes]
      exporters: [clickhouse, logging]

  telemetry:
    logs:
      level: info
    metrics:
      address: 0.0.0.0:8888
```

### Step 4: Initialize ClickHouse Schema

Create `config/clickhouse/init.sql`:

```sql
-- config/clickhouse/init.sql
CREATE DATABASE IF NOT EXISTS a11i;

USE a11i;

-- Traces table
CREATE TABLE IF NOT EXISTS otel_traces (
    timestamp DateTime64(9) CODEC(Delta, ZSTD(1)),
    trace_id String CODEC(ZSTD(1)),
    span_id String CODEC(ZSTD(1)),
    parent_span_id String CODEC(ZSTD(1)),
    trace_state String CODEC(ZSTD(1)),
    span_name LowCardinality(String) CODEC(ZSTD(1)),
    span_kind LowCardinality(String) CODEC(ZSTD(1)),
    service_name LowCardinality(String) CODEC(ZSTD(1)),
    resource_attributes Map(String, String) CODEC(ZSTD(1)),
    span_attributes Map(String, String) CODEC(ZSTD(1)),
    duration UInt64 CODEC(ZSTD(1)),
    status_code LowCardinality(String) CODEC(ZSTD(1)),
    status_message String CODEC(ZSTD(1)),
    events Nested(
        timestamp DateTime64(9),
        name String,
        attributes Map(String, String)
    ) CODEC(ZSTD(1)),
    links Nested(
        trace_id String,
        span_id String,
        trace_state String,
        attributes Map(String, String)
    ) CODEC(ZSTD(1))
) ENGINE = MergeTree()
PARTITION BY toDate(timestamp)
ORDER BY (service_name, span_name, toUnixTimestamp(timestamp), trace_id)
TTL timestamp + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- Materialized view for fast trace lookups
CREATE MATERIALIZED VIEW IF NOT EXISTS traces_by_trace_id
ENGINE = MergeTree()
ORDER BY (trace_id, timestamp)
AS SELECT
    trace_id,
    timestamp,
    span_id,
    span_name,
    duration,
    status_code
FROM otel_traces;

-- Metrics table for aggregations
CREATE TABLE IF NOT EXISTS trace_metrics (
    timestamp DateTime CODEC(Delta, ZSTD(1)),
    service_name LowCardinality(String),
    span_name LowCardinality(String),
    count UInt64,
    total_duration UInt64,
    min_duration UInt64,
    max_duration UInt64,
    p50_duration UInt64,
    p95_duration UInt64,
    p99_duration UInt64,
    error_count UInt64
) ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (service_name, span_name, timestamp)
TTL timestamp + INTERVAL 90 DAY;
```

### Step 5: Start Services

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# Expected output:
# NAME                STATUS              PORTS
# a11i-proxy          Up 10 seconds       0.0.0.0:8080->8080/tcp
# a11i-collector      Up 10 seconds       0.0.0.0:4317-4318->4317-4318/tcp
# a11i-clickhouse     Up 10 seconds       0.0.0.0:8123->8123/tcp, 0.0.0.0:9000->9000/tcp
# a11i-ui             Up 10 seconds       0.0.0.0:3000->3000/tcp
# a11i-redis          Up 10 seconds       0.0.0.0:6379->6379/tcp

# View logs
docker-compose logs -f

# Check specific service
docker-compose logs -f proxy
```

### Step 6: Configure Your Application

Update your application to use the self-hosted a11i:

**Python:**
```python
# Configure to use local a11i
import os

# Option 1: Route through proxy (recommended)
os.environ["OPENAI_BASE_URL"] = "http://localhost:8080/v1"
os.environ["OPENAI_API_KEY"] = "your_openai_key"  # Proxy forwards this

# Option 2: Send traces directly to collector
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"

# Initialize a11i SDK
import a11i

a11i.init(
    # No API key needed for self-hosted
    endpoint="http://localhost:4317",
    project="my-project",
    environment="development",
)

# Your code works the same
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_BASE_URL automatically
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello from self-hosted a11i!"}]
)

print(response.choices[0].message.content)
```

**Environment Variables:**
```bash
# .env file for your application
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_API_KEY=your_actual_openai_key
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
A11I_PROJECT=my-project
A11I_ENVIRONMENT=development
```

### Step 7: Access the Dashboard

```bash
# Open browser to view traces
open http://localhost:3000

# Or curl to test
curl http://localhost:3000/health

# Check ClickHouse
curl "http://localhost:8123/?query=SELECT count() FROM a11i.otel_traces"
```

### Step 8: Production Hardening (Optional)

For production deployments, add:

```yaml
# docker-compose.prod.yaml
version: '3.8'

services:
  proxy:
    environment:
      - RATE_LIMIT=1000  # Requests per minute
      - AUTH_ENABLED=true
      - API_KEYS_FILE=/secrets/api_keys.json
    volumes:
      - ./secrets:/secrets:ro

  clickhouse:
    environment:
      - CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD}  # Use secret
    volumes:
      - /var/lib/clickhouse:/var/lib/clickhouse  # Persistent storage

  ui:
    environment:
      - AUTH_PROVIDER=oauth2
      - OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID}
      - OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.a11i.rule=Host(`a11i.yourcompany.com`)"
      - "traefik.http.routers.a11i.tls=true"
```

### Maintenance Commands

```bash
# View disk usage
docker-compose exec clickhouse clickhouse-client --query="
  SELECT
    table,
    formatReadableSize(sum(bytes)) as size
  FROM system.parts
  WHERE database = 'a11i'
  GROUP BY table
"

# Optimize tables (compress old data)
docker-compose exec clickhouse clickhouse-client --query="
  OPTIMIZE TABLE a11i.otel_traces FINAL
"

# Backup data
docker-compose exec clickhouse clickhouse-client --query="
  BACKUP TABLE a11i.otel_traces TO Disk('backups', 'traces_backup.zip')
"

# Stop services
docker-compose down

# Stop and remove all data
docker-compose down -v
```

---

## Guide 4: Deploy to Kubernetes

**Time Required:** 20-30 minutes
**Prerequisites:** Kubernetes cluster, kubectl, Helm 3.x
**Best For:** Production deployments, high availability, scalability

### Step 1: Add Helm Repository

```bash
# Add a11i Helm charts repository
helm repo add a11i https://charts.a11i.dev

# Update repository
helm repo update

# Verify repository
helm search repo a11i

# Expected output:
# NAME                    CHART VERSION   APP VERSION     DESCRIPTION
# a11i/a11i               1.0.0           1.0.0           Complete a11i observability stack
# a11i/a11i-proxy         1.0.0           1.0.0           a11i LLM proxy with tracing
# a11i/a11i-collector     1.0.0           0.88.0          OpenTelemetry collector for a11i
```

### Step 2: Create Namespace and Configuration

```bash
# Create dedicated namespace
kubectl create namespace a11i

# Label namespace
kubectl label namespace a11i name=a11i monitoring=enabled
```

Create `values.yaml` for custom configuration:

```yaml
# values.yaml - Production configuration for a11i

global:
  domain: a11i.your-domain.com
  environment: production

# a11i Proxy configuration
proxy:
  replicas: 3
  image:
    repository: a11i/proxy
    tag: "1.0.0"
    pullPolicy: IfNotPresent

  resources:
    requests:
      memory: "128Mi"
      cpu: "100m"
    limits:
      memory: "256Mi"
      cpu: "500m"

  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

  service:
    type: ClusterIP
    port: 8080
    annotations:
      service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

  env:
    - name: OTEL_EXPORTER_OTLP_ENDPOINT
      value: "http://a11i-collector:4317"
    - name: LOG_LEVEL
      value: "info"
    - name: RATE_LIMIT_ENABLED
      value: "true"
    - name: RATE_LIMIT_REQUESTS_PER_MINUTE
      value: "1000"

# OpenTelemetry Collector configuration
collector:
  replicas: 3
  image:
    repository: otel/opentelemetry-collector-contrib
    tag: "0.88.0"

  resources:
    requests:
      memory: "256Mi"
      cpu: "200m"
    limits:
      memory: "512Mi"
      cpu: "1000m"

  config:
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318

    processors:
      batch:
        timeout: 5s
        send_batch_size: 200
        send_batch_max_size: 500

      memory_limiter:
        check_interval: 1s
        limit_mib: 400
        spike_limit_mib: 100

      k8sattributes:
        auth_type: "serviceAccount"
        passthrough: false
        extract:
          metadata:
            - k8s.pod.name
            - k8s.pod.uid
            - k8s.deployment.name
            - k8s.namespace.name
            - k8s.node.name
          labels:
            - tag_name: app
              key: app.kubernetes.io/name
              from: pod

    exporters:
      clickhouse:
        endpoint: tcp://a11i-clickhouse:9000?database=a11i
        username: a11i
        password: ${CLICKHOUSE_PASSWORD}
        traces_table_name: otel_traces
        timeout: 10s
        retry_on_failure:
          enabled: true
          initial_interval: 5s
          max_interval: 30s
          max_elapsed_time: 300s

    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, k8sattributes, batch]
          exporters: [clickhouse]

# ClickHouse configuration
storage:
  type: clickhouse

  clickhouse:
    enabled: true
    replicas: 3

    image:
      repository: clickhouse/clickhouse-server
      tag: "23.8-alpine"

    resources:
      requests:
        memory: "2Gi"
        cpu: "1000m"
      limits:
        memory: "4Gi"
        cpu: "2000m"

    persistence:
      enabled: true
      storageClass: "gp3"  # AWS EBS gp3
      size: 100Gi

    zookeeper:
      enabled: true
      replicas: 3

    config:
      users:
        a11i:
          password: ${CLICKHOUSE_PASSWORD}
          networks:
            - "::1"
            - "127.0.0.1"
            - "10.0.0.0/8"
          profile: default
          quota: default

      profiles:
        default:
          max_memory_usage: 10000000000
          use_uncompressed_cache: 0
          load_balancing: random

      quotas:
        default:
          interval:
            duration: 3600
            queries: 0
            errors: 0
            result_rows: 0
            read_rows: 0
            execution_time: 0

# UI configuration
ui:
  enabled: true
  replicas: 2

  image:
    repository: a11i/ui
    tag: "1.0.0"

  resources:
    requests:
      memory: "128Mi"
      cpu: "100m"
    limits:
      memory: "256Mi"
      cpu: "500m"

  env:
    - name: DATABASE_URL
      value: "http://a11i:${CLICKHOUSE_PASSWORD}@a11i-clickhouse:8123/a11i"
    - name: COLLECTOR_ENDPOINT
      value: "http://a11i-collector:4318"

  service:
    type: ClusterIP
    port: 3000

# Ingress configuration
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"

  hosts:
    - host: a11i.your-domain.com
      paths:
        - path: /
          pathType: Prefix
          service: ui
        - path: /api
          pathType: Prefix
          service: proxy

  tls:
    - secretName: a11i-tls
      hosts:
        - a11i.your-domain.com

# Redis for caching (optional but recommended)
redis:
  enabled: true
  architecture: replication
  master:
    persistence:
      enabled: true
      size: 8Gi
  replica:
    replicaCount: 2
    persistence:
      enabled: true
      size: 8Gi

# Monitoring configuration
monitoring:
  enabled: true

  serviceMonitor:
    enabled: true
    interval: 30s
    scrapeTimeout: 10s

  prometheusRule:
    enabled: true
    rules:
      - alert: A11iProxyHighErrorRate
        expr: rate(a11i_proxy_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in a11i proxy"

      - alert: A11iCollectorHighMemory
        expr: container_memory_usage_bytes{pod=~"a11i-collector.*"} / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Collector memory usage > 90%"
```

### Step 3: Create Secrets

```bash
# Create secret for ClickHouse password
kubectl create secret generic a11i-clickhouse \
  --from-literal=password=$(openssl rand -base64 32) \
  -n a11i

# Create secret for API keys (if using authentication)
kubectl create secret generic a11i-api-keys \
  --from-file=api_keys.json=./api_keys.json \
  -n a11i

# Verify secrets
kubectl get secrets -n a11i
```

### Step 4: Install a11i

```bash
# Install with custom values
helm install a11i a11i/a11i \
  -f values.yaml \
  -n a11i \
  --set storage.clickhouse.config.users.a11i.password="$(kubectl get secret a11i-clickhouse -n a11i -o jsonpath='{.data.password}' | base64 -d)"

# Watch installation progress
kubectl get pods -n a11i -w

# Expected output (after 2-3 minutes):
# NAME                                READY   STATUS    RESTARTS   AGE
# a11i-proxy-7d4f8b9c5d-abc12         1/1     Running   0          2m
# a11i-proxy-7d4f8b9c5d-def34         1/1     Running   0          2m
# a11i-proxy-7d4f8b9c5d-ghi56         1/1     Running   0          2m
# a11i-collector-6b8c9d4e5f-jkl78     1/1     Running   0          2m
# a11i-collector-6b8c9d4e5f-mno90     1/1     Running   0          2m
# a11i-collector-6b8c9d4e5f-pqr12     1/1     Running   0          2m
# a11i-clickhouse-0                   1/1     Running   0          2m
# a11i-clickhouse-1                   1/1     Running   0          2m
# a11i-clickhouse-2                   1/1     Running   0          2m
# a11i-ui-5c6d7e8f9g-stu34            1/1     Running   0          2m
# a11i-ui-5c6d7e8f9g-vwx56            1/1     Running   0          2m
```

### Step 5: Verify Installation

```bash
# Check deployment status
helm status a11i -n a11i

# Verify services
kubectl get svc -n a11i

# Expected output:
# NAME                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)
# a11i-proxy            ClusterIP   10.100.1.10     <none>        8080/TCP
# a11i-collector        ClusterIP   10.100.1.11     <none>        4317/TCP,4318/TCP
# a11i-clickhouse       ClusterIP   10.100.1.12     <none>        8123/TCP,9000/TCP
# a11i-ui               ClusterIP   10.100.1.13     <none>        3000/TCP

# Check ingress
kubectl get ingress -n a11i

# Test proxy endpoint
kubectl port-forward -n a11i svc/a11i-proxy 8080:8080 &
curl http://localhost:8080/health

# Test UI
kubectl port-forward -n a11i svc/a11i-ui 3000:3000 &
curl http://localhost:3000/health
```

### Step 6: Configure Agent Deployment

Deploy your agent with a11i sidecar pattern:

```yaml
# agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-agent
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-agent
  template:
    metadata:
      labels:
        app: my-agent
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
    spec:
      containers:
        # Main agent container
        - name: agent
          image: my-agent:latest
          ports:
            - containerPort: 5000
              name: http
          env:
            # Route LLM requests through local proxy
            - name: OPENAI_BASE_URL
              value: "http://localhost:8080/v1"
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: openai-secret
                  key: api-key

            # Or send traces directly to collector
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "http://a11i-collector.a11i.svc:4317"
            - name: A11I_PROJECT
              value: "my-agent"
            - name: A11I_ENVIRONMENT
              value: "production"

          resources:
            requests:
              memory: "256Mi"
              cpu: "200m"
            limits:
              memory: "512Mi"
              cpu: "500m"

        # a11i proxy sidecar (optional - for request interception)
        - name: a11i-proxy
          image: a11i/proxy:1.0.0
          ports:
            - containerPort: 8080
              name: proxy
          env:
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "http://a11i-collector.a11i.svc:4317"
            - name: UPSTREAM_OPENAI
              value: "https://api.openai.com"
            - name: LOG_LEVEL
              value: "info"

          resources:
            requests:
              memory: "64Mi"
              cpu: "50m"
            limits:
              memory: "128Mi"
              cpu: "200m"

---
apiVersion: v1
kind: Service
metadata:
  name: my-agent
  namespace: default
spec:
  selector:
    app: my-agent
  ports:
    - port: 80
      targetPort: 5000
      name: http
```

Deploy the agent:

```bash
# Deploy agent
kubectl apply -f agent-deployment.yaml

# Verify deployment
kubectl get pods -l app=my-agent
kubectl logs -l app=my-agent -c agent --tail=50
kubectl logs -l app=my-agent -c a11i-proxy --tail=50
```

### Step 7: Access the Dashboard

```bash
# Get ingress URL
kubectl get ingress -n a11i

# Or port-forward for testing
kubectl port-forward -n a11i svc/a11i-ui 3000:3000

# Open browser
open https://a11i.your-domain.com
# or
open http://localhost:3000
```

### Step 8: Production Operations

**Scaling:**
```bash
# Scale proxy
kubectl scale deployment a11i-proxy -n a11i --replicas=5

# Scale collector
kubectl scale deployment a11i-collector -n a11i --replicas=5

# Scale ClickHouse (requires more complex procedure)
# See https://clickhouse.com/docs/en/operations/scaling
```

**Monitoring:**
```bash
# Check metrics
kubectl port-forward -n a11i svc/a11i-collector 8888:8888
curl http://localhost:8888/metrics

# View Prometheus metrics in Grafana
# Import dashboard: https://grafana.com/grafana/dashboards/a11i
```

**Backup:**
```bash
# Create ClickHouse backup
kubectl exec -n a11i a11i-clickhouse-0 -- clickhouse-client --query="
  BACKUP TABLE a11i.otel_traces TO Disk('backups', 'backup-$(date +%Y%m%d).zip')
"

# Export to S3
kubectl exec -n a11i a11i-clickhouse-0 -- clickhouse-client --query="
  INSERT INTO FUNCTION s3(
    's3://my-bucket/a11i-backups/traces-$(date +%Y%m%d).parquet',
    'aws_access_key_id', 'aws_secret_access_key'
  )
  SELECT * FROM a11i.otel_traces WHERE timestamp >= today() - 7
"
```

**Upgrade:**
```bash
# Update to new version
helm upgrade a11i a11i/a11i \
  -f values.yaml \
  -n a11i \
  --set image.tag=1.1.0

# Rollback if needed
helm rollback a11i -n a11i
```

---

## Verification and Troubleshooting

### Verification Checklist

After completing any quickstart, verify your setup:

#### Python Verification

```python
# verify_a11i.py
from a11i import verify_connection, get_trace_url
import openai

# Test 1: Verify a11i connection
print("Test 1: Verify a11i connection")
status = verify_connection()
print(f"  Status: {status}")
assert status['connected'], "Not connected to a11i"
print("  ✓ Connected to a11i")

# Test 2: Send test trace
print("\nTest 2: Send test trace")
from a11i import observe

@observe(name="test_trace")
def test_function():
    return "Hello from a11i"

result = test_function()
trace_url = get_trace_url()
print(f"  Result: {result}")
print(f"  Trace URL: {trace_url}")
print("  ✓ Trace sent successfully")

# Test 3: LLM call with tracing
print("\nTest 3: LLM call with tracing")
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say 'a11i works!'"}],
    max_tokens=10
)
print(f"  Response: {response.choices[0].message.content}")
print(f"  Trace URL: {get_trace_url()}")
print("  ✓ LLM call traced successfully")

print("\n✅ All verification tests passed!")
print(f"View your traces at: {trace_url}")
```

Run verification:
```bash
python verify_a11i.py
```

#### TypeScript Verification

```typescript
// verify-a11i.ts
import { verifyConnection, getTraceUrl, observe } from '@a11i/sdk';
import OpenAI from 'openai';

async function verify() {
  // Test 1: Verify connection
  console.log('Test 1: Verify a11i connection');
  const status = await verifyConnection();
  console.log('  Status:', status);
  if (!status.connected) throw new Error('Not connected to a11i');
  console.log('  ✓ Connected to a11i');

  // Test 2: Send test trace
  console.log('\nTest 2: Send test trace');
  const testFunc = observe({ name: 'test_trace' })(async () => {
    return 'Hello from a11i';
  });
  const result = await testFunc();
  console.log('  Result:', result);
  console.log('  Trace URL:', getTraceUrl());
  console.log('  ✓ Trace sent successfully');

  // Test 3: LLM call with tracing
  console.log('\nTest 3: LLM call with tracing');
  const openai = new OpenAI();
  const response = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [{ role: 'user', content: "Say 'a11i works!'" }],
    max_tokens: 10,
  });
  console.log('  Response:', response.choices[0].message.content);
  console.log('  Trace URL:', getTraceUrl());
  console.log('  ✓ LLM call traced successfully');

  console.log('\n✅ All verification tests passed!');
  console.log(`View your traces at: ${getTraceUrl()}`);
}

verify().catch(console.error);
```

### Common Issues and Solutions

#### Issue 1: Traces Not Appearing

**Symptoms:**
- No traces in dashboard
- `verify_connection()` returns `{'connected': False}`

**Solutions:**
```bash
# Check environment variables
echo $A11I_API_KEY
echo $OTEL_EXPORTER_OTLP_ENDPOINT

# Test endpoint connectivity
curl -v http://localhost:4317  # For self-hosted
curl -v https://api.a11i.dev/health  # For cloud

# Check SDK initialization
python -c "import a11i; a11i.init(); print(a11i.verify_connection())"

# Enable debug logging
export A11I_LOG_LEVEL=debug
python your_app.py
```

#### Issue 2: High Latency with Proxy

**Symptoms:**
- LLM calls taking longer than usual
- Proxy adding significant overhead

**Solutions:**
```yaml
# Adjust proxy configuration
proxy:
  resources:
    limits:
      cpu: "1000m"  # Increase CPU

  env:
    - name: TIMEOUT
      value: "30s"
    - name: BUFFER_SIZE
      value: "10MB"

# Or bypass proxy for non-critical calls
OPENAI_BASE_URL=https://api.openai.com  # Direct to OpenAI
```

#### Issue 3: ClickHouse Disk Space

**Symptoms:**
- `ClickHouse: Disk is full` errors
- Traces not being stored

**Solutions:**
```bash
# Check disk usage
docker-compose exec clickhouse du -sh /var/lib/clickhouse

# Reduce TTL to clean up old data
docker-compose exec clickhouse clickhouse-client --query="
  ALTER TABLE a11i.otel_traces MODIFY TTL timestamp + INTERVAL 7 DAY
"

# Manually clean old data
docker-compose exec clickhouse clickhouse-client --query="
  ALTER TABLE a11i.otel_traces DELETE WHERE timestamp < now() - INTERVAL 30 DAY
"

# Increase storage (Kubernetes)
kubectl edit pvc a11i-clickhouse-data-a11i-clickhouse-0 -n a11i
# Update spec.resources.requests.storage
```

#### Issue 4: Collector Memory Issues

**Symptoms:**
- Collector OOMKilled
- Traces being dropped

**Solutions:**
```yaml
# Increase memory limits
collector:
  resources:
    limits:
      memory: "1Gi"  # Increase from 512Mi

  config:
    processors:
      memory_limiter:
        limit_mib: 800  # 80% of limit
        spike_limit_mib: 200

      batch:
        send_batch_size: 100  # Reduce batch size
        timeout: 2s  # Send more frequently
```

#### Issue 5: Authentication Errors

**Symptoms:**
- `401 Unauthorized` errors
- API key rejected

**Solutions:**
```bash
# Verify API key is set correctly
echo $A11I_API_KEY | wc -c  # Should be 32+ characters

# Check API key in dashboard
# Visit https://app.a11i.dev/settings/api-keys

# Regenerate if needed
a11i auth refresh

# For self-hosted, check secrets
kubectl get secret a11i-api-keys -n a11i -o yaml
```

### Debug Checklist

- [ ] Environment variables set correctly
- [ ] API key valid (for cloud) or endpoint reachable (for self-hosted)
- [ ] SDK initialized before making LLM calls
- [ ] Network connectivity to collector/proxy
- [ ] Sufficient resources (CPU, memory, disk)
- [ ] Correct ports exposed and accessible
- [ ] Firewall rules allow traffic
- [ ] TLS certificates valid (for HTTPS ingress)

---

## Key Takeaways

### Quick Reference

| Setup Type | Time | Best For | Pros | Cons |
|------------|------|----------|------|------|
| **LangChain Integration** | 5 min | Quick POC with LangChain | Minimal code changes | LangChain-specific |
| **Existing App** | 10-15 min | Adding to production apps | Framework agnostic, flexible | Requires code changes |
| **Self-Hosted** | 15-20 min | Data privacy, air-gapped | Full control, no data leaves network | Manage infrastructure |
| **Kubernetes** | 20-30 min | Production at scale | HA, scalable, enterprise-ready | Complex setup |

### Next Steps by Use Case

**For Development:**
1. Start with Guide 1 (LangChain) or Guide 2 (Existing App)
2. Use cloud-hosted a11i for simplicity
3. Focus on instrumentation and trace exploration

**For Production:**
1. Start with Guide 3 (Self-Hosted) for testing
2. Move to Guide 4 (Kubernetes) for deployment
3. Set up monitoring, alerting, and backups
4. Configure authentication and access control

**For Enterprise:**
1. Deploy Guide 4 (Kubernetes) with HA configuration
2. Integrate with existing observability stack (Prometheus, Grafana)
3. Set up data retention and compliance policies
4. Establish operational runbooks

### Related Documentation

- [Architecture Overview](/home/becker/projects/a11i/docs/02-architecture/system-overview.md) - Understand how a11i components work together
- [Integration Patterns](/home/becker/projects/a11i/docs/03-integration/integration-patterns.md) - Advanced integration strategies
- [Deployment Guide](/home/becker/projects/a11i/docs/04-deployment/deployment-strategies.md) - Detailed deployment configurations
- [Security Best Practices](/home/becker/projects/a11i/docs/05-security/security-model.md) - Secure your a11i deployment
- [Monitoring and Operations](/home/becker/projects/a11i/docs/06-operations/monitoring-alerting.md) - Production operations guide

### Support Resources

- **Documentation:** https://docs.a11i.dev
- **GitHub:** https://github.com/a11i/a11i
- **Discord Community:** https://discord.gg/a11i
- **Email Support:** support@a11i.dev

---

**Last Updated:** 2025-11-26
**Version:** 1.0.0
**Maintainer:** a11i Documentation Team
