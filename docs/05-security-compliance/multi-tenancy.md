---
title: Multi-Tenancy Architecture
category: Security & Compliance
tags: [multi-tenancy, isolation, security, api-keys, data-partitioning]
version: 1.0.0
last_updated: 2025-11-26
applies_to: [api, clickhouse, kubernetes]
related_docs:
  - ../02-architecture/clickhouse-schema.md
  - ./rbac.md
  - ./encryption.md
  - ../04-operations/monitoring.md
---

# Multi-Tenancy Architecture

## Table of Contents
- [Overview](#overview)
- [Isolation Levels](#isolation-levels)
- [Implementation by Level](#implementation-by-level)
- [Tenant Context Enforcement](#tenant-context-enforcement)
- [API Key Management](#api-key-management)
- [Data Partitioning](#data-partitioning)
- [Cross-Tenant Protection](#cross-tenant-protection)
- [Network Isolation](#network-isolation)
- [Monitoring and Auditing](#monitoring-and-auditing)
- [Key Takeaways](#key-takeaways)

## Overview

a11i implements a flexible multi-tenancy architecture that provides different levels of data isolation based on customer requirements, regulatory needs, and cost considerations. The system supports three distinct isolation models, from complete physical separation to efficient logical partitioning.

**Core Principles:**
- **Security by Default**: Tenant isolation enforced at multiple layers
- **Flexible Tiers**: Choose isolation level based on requirements
- **Cost Optimization**: Balance security needs with infrastructure costs
- **Audit Trail**: Complete visibility into cross-tenant access attempts
- **Performance**: Efficient queries even with tenant filtering

**Architecture Layers:**
```
┌─────────────────────────────────────────────┐
│  API Layer: Tenant Context Middleware       │
├─────────────────────────────────────────────┤
│  Application: Repository Pattern with       │
│               Automatic Tenant Scoping       │
├─────────────────────────────────────────────┤
│  Database: ClickHouse with Partitioning     │
│            and Row-Level Security            │
├─────────────────────────────────────────────┤
│  Infrastructure: Network Policies (Premium) │
└─────────────────────────────────────────────┘
```

## Isolation Levels

a11i supports three levels of tenant isolation, each with different tradeoffs:

| Level | Implementation | Use Case | Cost | Isolation | Performance |
|-------|---------------|----------|------|-----------|-------------|
| **Database per tenant** | Separate ClickHouse cluster | Enterprise, regulated industries | High | Complete physical | Excellent |
| **Schema per tenant** | Separate databases, shared infra | Business tier | Medium | Strong logical | Very Good |
| **Row-level security** | tenant_id in all queries | Self-serve, free tier | Low | Logical | Good |

### Level Comparison

**Database Per Tenant (Enterprise):**
- Completely isolated ClickHouse instance or cluster
- Dedicated encryption keys per tenant
- Separate backup schedules and retention policies
- Independent resource allocation and scaling
- Highest security compliance (HIPAA, PCI-DSS, SOC 2)
- Highest infrastructure cost
- Best for: Regulated industries, large enterprises, strict compliance requirements

**Schema Per Tenant (Business):**
- Same ClickHouse cluster, separate databases
- Shared infrastructure with database-level separation
- Per-tenant backup and restore capabilities
- Shared compute resources with quotas
- Good balance of isolation and cost
- Best for: Mid-market businesses, moderate compliance needs, growth stage companies

**Row-Level Security (Default):**
- Single database with `tenant_id` column in all tables
- Application-layer query filtering
- Shared infrastructure and resources
- Most cost-effective option
- Suitable for most SaaS use cases
- Best for: Startups, self-serve tiers, non-regulated data

## Implementation by Level

### Row-Level Security (Default Implementation)

This is the standard implementation for most tenants, providing efficient logical isolation through consistent tenant filtering.

#### Schema Design

```sql
-- ClickHouse schema with tenant isolation
CREATE TABLE agent_traces (
    tenant_id LowCardinality(String),  -- Required for all tables
    trace_id FixedString(32),
    span_id FixedString(16),
    parent_span_id Nullable(FixedString(16)),
    agent_id String,
    conversation_id String,
    start_time DateTime64(3, 'UTC'),
    end_time DateTime64(3, 'UTC'),
    duration_ms UInt32,
    model LowCardinality(String),
    input_tokens UInt32,
    output_tokens UInt32,
    cost_usd Float64,
    status LowCardinality(String),
    error_message Nullable(String),
    -- Metadata stored as JSON
    metadata String,
    -- ... other columns
)
ENGINE = MergeTree()
PARTITION BY (tenant_id, toYYYYMM(start_time))
ORDER BY (tenant_id, trace_id, start_time)
SETTINGS index_granularity = 8192;

-- Create secondary index for efficient tenant filtering
ALTER TABLE agent_traces
ADD INDEX idx_tenant_time (tenant_id, start_time) TYPE minmax GRANULARITY 4;

-- Materialized view for tenant-scoped metrics
CREATE MATERIALIZED VIEW tenant_metrics
ENGINE = SummingMergeTree()
PARTITION BY (tenant_id, toYYYYMM(timestamp))
ORDER BY (tenant_id, timestamp, model)
AS SELECT
    tenant_id,
    toStartOfHour(start_time) as timestamp,
    model,
    agent_id,
    count() as trace_count,
    sum(input_tokens) as total_input_tokens,
    sum(output_tokens) as total_output_tokens,
    sum(cost_usd) as total_cost,
    avg(duration_ms) as avg_duration_ms,
    quantile(0.95)(duration_ms) as p95_duration_ms
FROM agent_traces
GROUP BY tenant_id, timestamp, model, agent_id;

-- Tenant metadata table
CREATE TABLE tenants (
    tenant_id String,
    name String,
    tier LowCardinality(String),  -- free, business, enterprise
    created_at DateTime,
    updated_at DateTime,
    settings String,  -- JSON settings
    retention_days UInt16,
    max_agents UInt32,
    max_monthly_traces UInt64,
    status LowCardinality(String)  -- active, suspended, deleted
)
ENGINE = MergeTree()
ORDER BY tenant_id;
```

#### Tenant Quotas

```sql
-- Table for tracking tenant resource usage
CREATE TABLE tenant_quotas (
    tenant_id String,
    period_start DateTime,
    period_end DateTime,
    traces_used UInt64,
    traces_limit UInt64,
    storage_bytes_used UInt64,
    storage_bytes_limit UInt64,
    api_calls_used UInt64,
    api_calls_limit UInt64
)
ENGINE = ReplacingMergeTree(period_end)
ORDER BY (tenant_id, period_start);

-- Materialized view to update quotas in real-time
CREATE MATERIALIZED VIEW tenant_quota_updates
TO tenant_quotas
AS SELECT
    tenant_id,
    toStartOfMonth(start_time) as period_start,
    toStartOfMonth(start_time) + INTERVAL 1 MONTH as period_end,
    count() as traces_used,
    any(t.max_monthly_traces) as traces_limit,
    0 as storage_bytes_used,  -- Updated separately
    0 as storage_bytes_limit,
    0 as api_calls_used,
    0 as api_calls_limit
FROM agent_traces
JOIN tenants t ON agent_traces.tenant_id = t.tenant_id
GROUP BY tenant_id, period_start, period_end;
```

### Schema Per Tenant (Business Tier)

For business tier customers, each tenant gets a dedicated database within a shared ClickHouse cluster.

```python
class TenantDatabaseManager:
    """Manages per-tenant databases in shared ClickHouse cluster."""

    def __init__(self, admin_client):
        self.admin = admin_client

    async def create_tenant_database(self, tenant_id: str):
        """Create dedicated database for tenant."""
        db_name = f"a11i_tenant_{tenant_id}"

        # Create database
        await self.admin.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")

        # Create all tables in tenant database
        await self.create_tenant_schema(db_name)

        # Create tenant-specific user with restricted permissions
        await self.create_tenant_user(tenant_id, db_name)

    async def create_tenant_schema(self, db_name: str):
        """Create all required tables in tenant database."""
        # Same schema as row-level security, but without tenant_id column
        tables = [
            ("agent_traces", self.get_traces_schema()),
            ("agent_spans", self.get_spans_schema()),
            ("tenant_metrics", self.get_metrics_schema()),
        ]

        for table_name, schema in tables:
            await self.admin.execute(
                f"CREATE TABLE IF NOT EXISTS {db_name}.{table_name} {schema}"
            )

    async def create_tenant_user(self, tenant_id: str, db_name: str):
        """Create user with access only to their database."""
        user = f"tenant_{tenant_id}"
        password = secrets.token_urlsafe(32)

        await self.admin.execute(f"""
            CREATE USER IF NOT EXISTS {user}
            IDENTIFIED WITH sha256_password BY '{password}'
            SETTINGS PROFILE 'tenant_profile'
        """)

        # Grant permissions only to this database
        await self.admin.execute(f"""
            GRANT SELECT, INSERT ON {db_name}.* TO {user}
        """)

        return password
```

### Database Per Tenant (Enterprise Tier)

Enterprise customers receive completely isolated ClickHouse instances.

```yaml
# Helm values for dedicated tenant cluster
apiVersion: v1
kind: ConfigMap
metadata:
  name: tenant-{{ tenant_id }}-config
data:
  tenant_id: "{{ tenant_id }}"
  tier: "enterprise"
  clickhouse_cluster_size: "3"
  clickhouse_storage_size: "1Ti"
  backup_schedule: "0 2 * * *"
  retention_days: "730"  # 2 years for enterprise

---
# Dedicated ClickHouse StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: clickhouse-{{ tenant_id }}
  labels:
    app: clickhouse
    tenant: "{{ tenant_id }}"
    tier: enterprise
spec:
  serviceName: clickhouse-{{ tenant_id }}
  replicas: 3
  selector:
    matchLabels:
      app: clickhouse
      tenant: "{{ tenant_id }}"
  template:
    metadata:
      labels:
        app: clickhouse
        tenant: "{{ tenant_id }}"
    spec:
      # Dedicated nodes for enterprise tenant
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: tenant
                    operator: In
                    values: ["{{ tenant_id }}"]
      containers:
        - name: clickhouse
          image: clickhouse/clickhouse-server:23.8
          resources:
            requests:
              memory: "32Gi"
              cpu: "8"
            limits:
              memory: "64Gi"
              cpu: "16"
          volumeMounts:
            - name: data
              mountPath: /var/lib/clickhouse
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: ssd-encrypted
        resources:
          requests:
            storage: 1Ti
```

## Tenant Context Enforcement

### Context Management

The system uses thread-safe context variables to track the current tenant across async operations.

```python
from contextvars import ContextVar
from functools import wraps
from typing import Optional
import logging

# Thread-safe tenant context
current_tenant: ContextVar[Optional[str]] = ContextVar('current_tenant', default=None)

logger = logging.getLogger(__name__)


class TenantContext:
    """Manages tenant context for a request."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.token = None

    def __enter__(self):
        self.token = current_tenant.set(self.tenant_id)
        logger.debug(f"Set tenant context: {self.tenant_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_tenant.reset(self.token)
        logger.debug(f"Cleared tenant context: {self.tenant_id}")


def get_current_tenant() -> str:
    """Get current tenant ID from context."""
    tenant = current_tenant.get()
    if not tenant:
        raise TenantNotFoundError("No tenant context set")
    return tenant


def require_tenant(func):
    """Decorator to ensure tenant context is set."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tenant = current_tenant.get()
        if not tenant:
            logger.error("Function called without tenant context", extra={
                "function": func.__name__,
                "args": args,
            })
            raise TenantNotFoundError("Tenant context required")
        return await func(*args, **kwargs)
    return wrapper
```

### Middleware Integration

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable


class TenantMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for tenant context management."""

    def __init__(self, app, api_key_manager):
        super().__init__(app)
        self.api_key_manager = api_key_manager

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Extract tenant from request and set context."""
        # Extract API key from header
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            # Try Authorization header as fallback
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                api_key = auth[7:]

        if not api_key:
            return Response(
                content="Missing API key",
                status_code=401,
            )

        # Resolve tenant from API key
        tenant_info = await self.api_key_manager.verify_key(api_key)
        if not tenant_info:
            logger.warning("Invalid API key", extra={
                "key_prefix": api_key[:10],
                "ip": request.client.host,
            })
            return Response(
                content="Invalid API key",
                status_code=401,
            )

        tenant_id = tenant_info["tenant_id"]

        # Set tenant context for request
        with TenantContext(tenant_id):
            # Add tenant info to request state
            request.state.tenant_id = tenant_id
            request.state.api_key_id = tenant_info["key_id"]

            # Log request with tenant context
            logger.info("API request", extra={
                "tenant_id": tenant_id,
                "path": request.url.path,
                "method": request.method,
            })

            response = await call_next(request)

            # Add tenant ID to response headers (for debugging)
            response.headers["X-Tenant-ID"] = tenant_id

            return response
```

## API Key Management

### Secure Key Generation and Storage

```python
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional


class APIKeyManager:
    """Secure API key management with hashing and rotation."""

    def __init__(self, db_client):
        self.db = db_client
        self.cache = {}  # In-memory cache for hot paths

    def generate_key(
        self,
        tenant_id: str,
        name: str,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, str]:
        """Generate new API key. Returns (key, key_id)."""
        # Generate secure random key with prefix
        key = f"a11i_{secrets.token_urlsafe(32)}"

        # Hash for storage (never store plaintext keys)
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Generate key ID for management
        key_id = f"key_{secrets.token_urlsafe(8)}"

        # Calculate expiration if specified
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Store key metadata
        self.db.execute("""
            INSERT INTO api_keys (
                key_id, key_hash, tenant_id, name,
                created_at, expires_at, last_used_at,
                revoked, revoked_at, revoked_reason
            ) VALUES (
                {key_id}, {key_hash}, {tenant_id}, {name},
                now(), {expires_at}, NULL,
                false, NULL, NULL
            )
        """, {
            "key_id": key_id,
            "key_hash": key_hash,
            "tenant_id": tenant_id,
            "name": name,
            "expires_at": expires_at,
        })

        logger.info("API key created", extra={
            "tenant_id": tenant_id,
            "key_id": key_id,
            "name": name,
        })

        # Return plaintext key (only time it's available)
        return key, key_id

    async def verify_key(self, key: str) -> Optional[dict]:
        """Verify API key and return tenant info."""
        # Hash the provided key
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Check cache first
        if key_hash in self.cache:
            cached = self.cache[key_hash]
            if cached["valid_until"] > datetime.utcnow():
                return cached["tenant_info"]

        # Query database
        result = await self.db.query_one("""
            SELECT
                k.tenant_id,
                k.key_id,
                k.revoked,
                k.expires_at,
                t.status as tenant_status,
                t.tier
            FROM api_keys k
            JOIN tenants t ON k.tenant_id = t.tenant_id
            WHERE k.key_hash = {hash}
        """, hash=key_hash)

        if not result:
            logger.warning("Unknown API key", extra={"key_hash": key_hash[:16]})
            return None

        # Check if key is revoked
        if result["revoked"]:
            logger.warning("Revoked API key used", extra={
                "tenant_id": result["tenant_id"],
                "key_id": result["key_id"],
            })
            return None

        # Check if key is expired
        if result["expires_at"] and result["expires_at"] < datetime.utcnow():
            logger.warning("Expired API key used", extra={
                "tenant_id": result["tenant_id"],
                "key_id": result["key_id"],
            })
            return None

        # Check tenant status
        if result["tenant_status"] != "active":
            logger.warning("Inactive tenant API key used", extra={
                "tenant_id": result["tenant_id"],
                "status": result["tenant_status"],
            })
            return None

        # Update last used timestamp (async, non-blocking)
        self._update_last_used(result["key_id"])

        tenant_info = {
            "tenant_id": result["tenant_id"],
            "key_id": result["key_id"],
            "tier": result["tier"],
        }

        # Cache for 5 minutes
        self.cache[key_hash] = {
            "tenant_info": tenant_info,
            "valid_until": datetime.utcnow() + timedelta(minutes=5),
        }

        return tenant_info

    def _update_last_used(self, key_id: str):
        """Update last_used_at timestamp (fire and forget)."""
        # Use ALTER TABLE UPDATE for ClickHouse
        self.db.execute_async("""
            ALTER TABLE api_keys
            UPDATE last_used_at = now()
            WHERE key_id = {key_id}
        """, key_id=key_id)

    def revoke_key(self, key_id: str, tenant_id: str, reason: str = "User revoked"):
        """Revoke an API key."""
        self.db.execute("""
            ALTER TABLE api_keys
            UPDATE
                revoked = true,
                revoked_at = now(),
                revoked_reason = {reason}
            WHERE key_id = {key_id}
              AND tenant_id = {tenant_id}
        """, key_id=key_id, tenant_id=tenant_id, reason=reason)

        # Clear from cache
        self.cache = {k: v for k, v in self.cache.items()
                      if v["tenant_info"]["key_id"] != key_id}

        logger.info("API key revoked", extra={
            "tenant_id": tenant_id,
            "key_id": key_id,
            "reason": reason,
        })

    async def rotate_key(
        self,
        old_key_id: str,
        tenant_id: str,
        name: str,
    ) -> tuple[str, str]:
        """Rotate API key (create new, mark old as rotated)."""
        # Generate new key
        new_key, new_key_id = self.generate_key(tenant_id, name)

        # Mark old key as rotated (don't revoke immediately for grace period)
        self.db.execute("""
            ALTER TABLE api_keys
            UPDATE rotated_to = {new_key_id}
            WHERE key_id = {old_key_id}
              AND tenant_id = {tenant_id}
        """, new_key_id=new_key_id, old_key_id=old_key_id, tenant_id=tenant_id)

        logger.info("API key rotated", extra={
            "tenant_id": tenant_id,
            "old_key_id": old_key_id,
            "new_key_id": new_key_id,
        })

        return new_key, new_key_id
```

### API Keys Table Schema

```sql
CREATE TABLE api_keys (
    key_id String,
    key_hash FixedString(64),  -- SHA-256 hash
    tenant_id String,
    name String,
    created_at DateTime,
    expires_at Nullable(DateTime),
    last_used_at Nullable(DateTime),
    revoked Bool DEFAULT false,
    revoked_at Nullable(DateTime),
    revoked_reason Nullable(String),
    rotated_to Nullable(String),  -- Key ID of replacement key
    metadata String  -- JSON for additional fields
)
ENGINE = ReplacingMergeTree(revoked_at)
ORDER BY (key_hash, key_id)
SETTINGS index_granularity = 8192;

-- Index for tenant key management
ALTER TABLE api_keys
ADD INDEX idx_tenant_keys (tenant_id, created_at) TYPE minmax GRANULARITY 4;
```

## Data Partitioning

### Partition Strategy

ClickHouse partitioning enables efficient queries, data retention, and tenant isolation.

```sql
-- Agent traces partitioned by tenant and month
CREATE TABLE agent_traces (
    tenant_id LowCardinality(String),
    trace_id FixedString(32),
    start_time DateTime64(3, 'UTC'),
    -- ... other columns
)
ENGINE = MergeTree()
PARTITION BY (tenant_id, toYYYYMM(start_time))
ORDER BY (tenant_id, trace_id, start_time)
SETTINGS index_granularity = 8192;
```

### Tenant-Specific TTL

Different retention policies based on tenant tier:

```sql
-- Free tier: 30 days
-- Business tier: 90 days
-- Enterprise tier: 365+ days (configurable)

-- Apply TTL based on tenant tier
ALTER TABLE agent_traces
MODIFY TTL
    start_time + INTERVAL 30 DAY
    WHERE tenant_id IN (
        SELECT tenant_id FROM tenants WHERE tier = 'free'
    ),
    start_time + INTERVAL 90 DAY
    WHERE tenant_id IN (
        SELECT tenant_id FROM tenants WHERE tier = 'business'
    ),
    start_time + toIntervalDay(
        (SELECT retention_days FROM tenants t WHERE t.tenant_id = agent_traces.tenant_id)
    )
    WHERE tenant_id IN (
        SELECT tenant_id FROM tenants WHERE tier = 'enterprise'
    );
```

### Partition Management

```python
class TenantPartitionManager:
    """Manages ClickHouse partitions for tenant data lifecycle."""

    def __init__(self, db_client):
        self.db = db_client

    async def list_tenant_partitions(self, tenant_id: str) -> list[dict]:
        """List all partitions for a tenant."""
        result = await self.db.query("""
            SELECT
                partition,
                name,
                rows,
                bytes_on_disk,
                min_time,
                max_time
            FROM system.parts
            WHERE table = 'agent_traces'
              AND partition LIKE {pattern}
              AND active = 1
            ORDER BY partition
        """, pattern=f"{tenant_id}-%")

        return result

    async def drop_tenant_data(self, tenant_id: str):
        """Drop all partitions for a tenant (account deletion)."""
        partitions = await self.list_tenant_partitions(tenant_id)

        for partition in partitions:
            await self.db.execute(f"""
                ALTER TABLE agent_traces
                DROP PARTITION '{partition['partition']}'
            """)

        logger.info("Tenant data dropped", extra={
            "tenant_id": tenant_id,
            "partitions_dropped": len(partitions),
        })

    async def export_tenant_data(self, tenant_id: str, output_path: str):
        """Export all tenant data (GDPR data portability)."""
        await self.db.execute("""
            INSERT INTO FUNCTION
            file('{path}/traces.parquet', 'Parquet')
            SELECT * FROM agent_traces
            WHERE tenant_id = {tenant_id}
        """, path=output_path, tenant_id=tenant_id)

        logger.info("Tenant data exported", extra={
            "tenant_id": tenant_id,
            "output_path": output_path,
        })
```

## Cross-Tenant Protection

### Query Validation

```python
class TenantIsolationValidator:
    """Validates tenant isolation at multiple layers."""

    @staticmethod
    def validate_query(query: str, tenant_id: str) -> bool:
        """Ensure query includes tenant filter."""
        query_lower = query.lower()

        # Check that tenant_id appears in WHERE clause
        if "tenant_id" not in query_lower:
            logger.error("Query missing tenant_id filter", extra={
                "tenant_id": tenant_id,
                "query": query[:200],
            })
            raise SecurityError("Query must include tenant_id filter")

        # Check for attempts to bypass filtering
        dangerous_patterns = [
            "drop",
            "alter",
            "truncate",
            "-- tenant_id",  # Commented out filter
        ]

        for pattern in dangerous_patterns:
            if pattern in query_lower:
                logger.error("Dangerous query pattern detected", extra={
                    "tenant_id": tenant_id,
                    "pattern": pattern,
                    "query": query[:200],
                })
                raise SecurityError(f"Query contains forbidden pattern: {pattern}")

        return True

    @staticmethod
    def validate_result(results: list[dict], tenant_id: str) -> list[dict]:
        """Verify all results belong to tenant."""
        for i, result in enumerate(results):
            result_tenant = result.get("tenant_id")
            if result_tenant != tenant_id:
                logger.critical("Cross-tenant data leak detected", extra={
                    "expected_tenant": tenant_id,
                    "actual_tenant": result_tenant,
                    "row_index": i,
                })
                raise SecurityError(
                    f"Cross-tenant data leak: expected {tenant_id}, "
                    f"got {result_tenant}"
                )

        return results

    @staticmethod
    def sanitize_tenant_id(tenant_id: str) -> str:
        """Sanitize tenant ID to prevent injection."""
        # Only allow alphanumeric, hyphens, underscores
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', tenant_id):
            raise ValueError(f"Invalid tenant ID format: {tenant_id}")

        if len(tenant_id) > 64:
            raise ValueError(f"Tenant ID too long: {len(tenant_id)}")

        return tenant_id
```

### Repository Pattern with Automatic Scoping

```python
class TenantScopedRepository:
    """Repository that automatically scopes all queries to current tenant."""

    def __init__(self, db_client):
        self.db = db_client
        self.validator = TenantIsolationValidator()

    @property
    def tenant_id(self) -> str:
        """Get current tenant from context."""
        tenant = get_current_tenant()
        return self.validator.sanitize_tenant_id(tenant)

    async def get_traces(
        self,
        start_time: datetime,
        end_time: datetime,
        agent_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get traces - automatically scoped to current tenant."""
        query = """
            SELECT
                trace_id,
                agent_id,
                conversation_id,
                start_time,
                end_time,
                duration_ms,
                model,
                input_tokens,
                output_tokens,
                cost_usd,
                status
            FROM agent_traces
            WHERE tenant_id = {tenant_id}
              AND start_time >= {start}
              AND start_time < {end}
              {agent_filter}
            ORDER BY start_time DESC
            LIMIT {limit}
            OFFSET {offset}
        """

        agent_filter = ""
        if agent_id:
            agent_filter = "AND agent_id = {agent_id}"

        results = await self.db.query(
            query.replace("{agent_filter}", agent_filter),
            tenant_id=self.tenant_id,
            start=start_time,
            end=end_time,
            agent_id=agent_id,
            limit=limit,
            offset=offset,
        )

        # Validate results belong to tenant
        return self.validator.validate_result(results, self.tenant_id)

    async def insert_trace(self, trace: dict):
        """Insert trace - tenant_id automatically set."""
        # Override tenant_id to ensure it matches context
        trace["tenant_id"] = self.tenant_id

        await self.db.execute("""
            INSERT INTO agent_traces FORMAT JSONEachRow
        """, [trace])

        logger.debug("Trace inserted", extra={
            "tenant_id": self.tenant_id,
            "trace_id": trace["trace_id"],
        })

    async def get_tenant_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        group_by: str = "model",
    ) -> list[dict]:
        """Get aggregated metrics for tenant."""
        allowed_group_by = ["model", "agent_id", "status"]
        if group_by not in allowed_group_by:
            raise ValueError(f"Invalid group_by: {group_by}")

        query = f"""
            SELECT
                {group_by},
                count() as trace_count,
                sum(input_tokens) as total_input_tokens,
                sum(output_tokens) as total_output_tokens,
                sum(cost_usd) as total_cost,
                avg(duration_ms) as avg_duration_ms
            FROM agent_traces
            WHERE tenant_id = {{tenant_id}}
              AND start_time >= {{start}}
              AND start_time < {{end}}
            GROUP BY {group_by}
            ORDER BY total_cost DESC
        """

        results = await self.db.query(
            query,
            tenant_id=self.tenant_id,
            start=start_time,
            end=end_time,
        )

        return results
```

## Network Isolation

### Kubernetes Network Policies (Premium)

For enterprise customers, network-level isolation prevents unauthorized communication.

```yaml
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tenant-isolation-{{ tenant_id }}
  namespace: a11i
spec:
  podSelector:
    matchLabels:
      tenant: "{{ tenant_id }}"
  policyTypes:
    - Ingress
    - Egress

  # Ingress: Only from API gateway and monitoring
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: a11i-api
      ports:
        - protocol: TCP
          port: 8000

    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
        - podSelector:
            matchLabels:
              app: prometheus
      ports:
        - protocol: TCP
          port: 9090

  # Egress: Only to tenant-specific ClickHouse and shared services
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: clickhouse
              tenant: "{{ tenant_id }}"
      ports:
        - protocol: TCP
          port: 9000
        - protocol: TCP
          port: 8123

    # DNS resolution
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
        - podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53

---
# Deny all by default, then whitelist
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: a11i
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
```

### Service Mesh Isolation (Istio)

```yaml
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: tenant-{{ tenant_id }}-access
  namespace: a11i
spec:
  selector:
    matchLabels:
      app: clickhouse-{{ tenant_id }}

  action: ALLOW

  rules:
    # Only API pods with matching tenant label
    - from:
        - source:
            principals: ["cluster.local/ns/a11i/sa/a11i-api"]
            requestPrincipals: ["tenant-{{ tenant_id }}"]

      to:
        - operation:
            ports: ["9000", "8123"]
            methods: ["POST", "GET"]
```

## Monitoring and Auditing

### Cross-Tenant Access Monitoring

```python
class TenantAccessAuditor:
    """Audit and alert on suspicious cross-tenant access patterns."""

    def __init__(self, metrics_client):
        self.metrics = metrics_client

    def record_access(
        self,
        tenant_id: str,
        resource_type: str,
        action: str,
        success: bool,
    ):
        """Record tenant access for auditing."""
        self.metrics.increment(
            "tenant_access_total",
            tags={
                "tenant_id": tenant_id,
                "resource_type": resource_type,
                "action": action,
                "success": str(success).lower(),
            },
        )

    def record_isolation_violation(
        self,
        tenant_id: str,
        attempted_tenant: str,
        resource_type: str,
    ):
        """Record attempted cross-tenant access."""
        logger.critical("Tenant isolation violation", extra={
            "tenant_id": tenant_id,
            "attempted_tenant": attempted_tenant,
            "resource_type": resource_type,
        })

        self.metrics.increment(
            "tenant_isolation_violation_total",
            tags={
                "tenant_id": tenant_id,
                "resource_type": resource_type,
            },
        )

        # Trigger immediate alert
        self.alert_security_team({
            "severity": "critical",
            "type": "tenant_isolation_violation",
            "tenant_id": tenant_id,
            "attempted_tenant": attempted_tenant,
            "resource_type": resource_type,
            "timestamp": datetime.utcnow().isoformat(),
        })
```

### Prometheus Metrics

```yaml
# Record metrics for tenant isolation monitoring
- record: tenant:api_requests:rate5m
  expr: rate(tenant_access_total[5m])

- record: tenant:isolation_violations:rate5m
  expr: rate(tenant_isolation_violation_total[5m])

# Alert on isolation violations
- alert: TenantIsolationViolation
  expr: tenant:isolation_violations:rate5m > 0
  for: 0m
  labels:
    severity: critical
  annotations:
    summary: "Tenant isolation violation detected"
    description: "{{ $labels.tenant_id }} attempted to access data from another tenant"
```

## Key Takeaways

**Critical Success Factors:**

1. **Defense in Depth**: Tenant isolation enforced at API, application, database, and network layers
2. **Context Management**: Thread-safe tenant context prevents cross-tenant data leaks
3. **Secure by Default**: All queries automatically scoped to current tenant via repository pattern
4. **Audit Everything**: Complete audit trail of tenant access and isolation violations
5. **Flexible Tiers**: Choose isolation level based on security requirements and budget

**Implementation Checklist:**

- [ ] **API Keys**: Generate secure keys with SHA-256 hashing, never store plaintext
- [ ] **Middleware**: Set tenant context from API key on every request
- [ ] **Repository Pattern**: All data access goes through tenant-scoped repositories
- [ ] **Query Validation**: Verify tenant_id filter in all database queries
- [ ] **Result Validation**: Check all returned data belongs to current tenant
- [ ] **Partitioning**: Partition tables by tenant_id for efficient queries and TTL
- [ ] **Network Policies**: Implement network isolation for premium tiers
- [ ] **Monitoring**: Track tenant access patterns and alert on violations
- [ ] **Testing**: Automated tests verify tenant isolation cannot be bypassed

**Security Principles:**

- **Never Trust, Always Verify**: Validate tenant context at every layer
- **Fail Secure**: Reject requests without valid tenant context
- ** Principle of Least Privilege**: Each tenant sees only their data
- **Audit Trail**: Log all access attempts for security review
- **Automated Testing**: Continuous validation of isolation boundaries

**Related Documentation:**
- [ClickHouse Schema Design](../02-architecture/clickhouse-schema.md) - Database schema with tenant isolation
- [RBAC & Permissions](./rbac.md) - Role-based access control within tenants
- [Encryption at Rest](./encryption.md) - Data encryption for sensitive tenant data
- [Monitoring & Alerting](../04-operations/monitoring.md) - Tenant access monitoring

