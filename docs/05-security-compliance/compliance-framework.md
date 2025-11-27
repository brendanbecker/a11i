---
title: Compliance Framework
description: Comprehensive compliance framework covering SOC 2, GDPR, HIPAA, CCPA, and EU AI Act requirements
status: designed
last_updated: 2025-11-26
owner: Security & Compliance Team
related_docs:
  - security-architecture.md
  - privacy-controls.md
  - audit-logging.md
  - data-retention.md
---

# Compliance Framework

## Table of Contents

1. [Overview](#overview)
2. [Compliance Matrix](#compliance-matrix)
3. [SOC 2 Type II Compliance](#soc-2-type-ii-compliance)
4. [GDPR Compliance](#gdpr-compliance)
5. [Data Retention Policies](#data-retention-policies)
6. [Encryption Standards](#encryption-standards)
7. [Audit Logging](#audit-logging)
8. [Regional Data Residency](#regional-data-residency)
9. [HIPAA Compliance](#hipaa-compliance)
10. [CCPA Compliance](#ccpa-compliance)
11. [EU AI Act Compliance](#eu-ai-act-compliance)
12. [Key Takeaways](#key-takeaways)

## Overview

The a11i platform implements a comprehensive compliance framework to meet the requirements of major regulatory standards and industry certifications. This document outlines our approach to compliance across multiple frameworks, including technical implementations, operational procedures, and evidence collection processes.

**Core Compliance Principles:**

- **Privacy by Design**: Data protection and privacy controls built into system architecture
- **Security First**: Multi-layered security controls with defense in depth
- **Transparency**: Clear audit trails and user visibility into data processing
- **Data Minimization**: Collect and retain only necessary data for operational purposes
- **User Rights**: Comprehensive implementation of data subject rights across all frameworks

## Compliance Matrix

| Standard | Requirements | Implementation | Status |
|----------|--------------|----------------|--------|
| **SOC 2 Type II** | Security, availability, confidentiality | Encryption, RBAC, audit logs, uptime SLAs | ✅ Designed |
| **GDPR** | Data minimization, right to deletion, consent | PII redaction, data retention, deletion API | ✅ Designed |
| **HIPAA** | PHI protection, access controls, audit trails | Encryption, RBAC, audit logging, BAA | 🔧 Partial |
| **CCPA** | Consumer data rights, opt-out | Privacy controls, data export | ✅ Designed |
| **EU AI Act** | AI system transparency, risk assessment | Audit trails, model tracking | 📋 Planned |

## SOC 2 Type II Compliance

SOC 2 Type II certification demonstrates our commitment to security, availability, and confidentiality through both design (Type I) and operational effectiveness over time (Type II).

### Trust Service Criteria Coverage

#### CC6.1 - Logical and Physical Access Controls

**Security Controls:**

- **Encryption at Rest**: AES-256 encryption for all stored data
- **Encryption in Transit**: TLS 1.3 for all network communications
- **Access Control**: Role-based access control (RBAC) with least privilege principle
- **Network Security**: VPC isolation, security groups, and firewall rules
- **Vulnerability Management**: Automated scanning and patch management

**Implementation:**

```yaml
# SOC 2 Control Mapping - CC6.1
controls:
  CC6.1:
    name: "Logical and Physical Access Controls"
    implementation:
      - RBAC with least privilege principle
      - MFA required for all administrative access
      - API key rotation policy (90 days)
      - Session timeout (30 minutes)
      - IP allowlisting for production access

    evidence:
      - Access control policy document
      - User access reviews (quarterly)
      - MFA enrollment reports
      - API key rotation logs
      - Failed login attempt logs

    testing_procedures:
      - Verify MFA enforcement for admin accounts
      - Review access grant/revoke procedures
      - Test session timeout functionality
      - Validate IP allowlist enforcement
```

#### CC6.2 - System Operations and Monitoring

**Availability Controls:**

- **99.9% Uptime SLA**: Multi-region deployment with automated failover
- **Disaster Recovery**: RPO of 1 hour, RTO of 4 hours
- **Incident Response**: 24/7 monitoring and on-call procedures
- **Capacity Planning**: Auto-scaling based on demand

**Implementation:**

```yaml
# SOC 2 Control Mapping - CC6.2
controls:
  CC6.2:
    name: "System Operations and Monitoring"
    implementation:
      - Prometheus + Grafana monitoring
      - PagerDuty alerting with escalation
      - Automated health checks (30s intervals)
      - Incident response runbooks
      - Change management process

    evidence:
      - Uptime reports (monthly)
      - Incident response logs
      - Change tickets and approvals
      - Monitoring dashboards
      - Capacity planning reports

    testing_procedures:
      - Verify alert escalation paths
      - Test failover procedures
      - Review incident response times
      - Validate monitoring coverage
```

#### CC6.3 - Change Management

**Confidentiality Controls:**

- **Data Classification**: Automatic PII detection and classification
- **PII Redaction**: Configurable redaction of sensitive data
- **Access Logging**: Complete audit trail of data access
- **Non-Disclosure Agreements**: Required for all personnel with data access

**Implementation:**

```yaml
# SOC 2 Control Mapping - CC6.3
controls:
  CC6.3:
    name: "Change Management"
    implementation:
      - Version control (Git) for all code changes
      - Mandatory code review (minimum 2 approvers)
      - Automated CI/CD pipeline with testing gates
      - Deployment approval workflow
      - Rollback procedures

    evidence:
      - Git commit history and tags
      - Pull request approval logs
      - CI/CD pipeline execution logs
      - Deployment approval records
      - Rollback incident reports

    testing_procedures:
      - Verify code review enforcement
      - Test automated testing gates
      - Review emergency change procedures
      - Validate rollback capabilities
```

### Evidence Collection and Reporting

**Continuous Monitoring:**

```python
class SOC2EvidenceCollector:
    """Automated collection of SOC 2 compliance evidence."""

    async def collect_access_control_evidence(self, period_start, period_end):
        """Collect evidence for CC6.1 - Access Controls."""
        evidence = {
            "control": "CC6.1",
            "period": {"start": period_start, "end": period_end},
            "artifacts": []
        }

        # User access reviews
        access_reviews = await self.get_access_reviews(period_start, period_end)
        evidence["artifacts"].append({
            "type": "user_access_review",
            "count": len(access_reviews),
            "data": access_reviews
        })

        # MFA enrollment status
        mfa_status = await self.get_mfa_enrollment_status()
        evidence["artifacts"].append({
            "type": "mfa_enrollment",
            "total_users": mfa_status["total"],
            "enrolled": mfa_status["enrolled"],
            "percentage": mfa_status["percentage"]
        })

        # Failed login attempts
        failed_logins = await self.get_failed_login_attempts(period_start, period_end)
        evidence["artifacts"].append({
            "type": "failed_logins",
            "count": len(failed_logins),
            "data": failed_logins
        })

        return evidence
```

## GDPR Compliance

The General Data Protection Regulation (GDPR) establishes comprehensive data protection requirements for EU citizens. Our implementation ensures full compliance with all GDPR articles relevant to AI observability platforms.

### Article 17 - Right to Deletion (Right to be Forgotten)

**Implementation:**

```python
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

@dataclass
class DeletionReport:
    """Report of GDPR deletion operation."""
    request_id: str
    user_id: str
    org_id: str
    traces_found: int = 0
    traces_deleted: int = 0
    metrics_anonymized: int = 0
    completion_time: Optional[datetime] = None
    status: str = "pending"

class GDPRComplianceService:
    """GDPR compliance operations."""

    def __init__(self, db, audit_log, storage):
        self.db = db
        self.audit_log = audit_log
        self.storage = storage

    async def delete_user_data(
        self,
        user_id: str,
        org_id: str,
        request_id: str,
    ) -> DeletionReport:
        """Delete all user data per GDPR Article 17.

        This implements the "right to be forgotten" by:
        1. Identifying all data associated with the user
        2. Deleting personally identifiable information
        3. Anonymizing aggregated metrics
        4. Creating comprehensive audit trail

        Args:
            user_id: Unique identifier of the user
            org_id: Organization identifier
            request_id: Unique identifier for this deletion request

        Returns:
            DeletionReport with detailed results
        """
        report = DeletionReport(
            request_id=request_id,
            user_id=user_id,
            org_id=org_id,
        )

        try:
            # Find all traces containing user data
            traces = await self.find_user_traces(user_id, org_id)
            report.traces_found = len(traces)

            # Delete traces (soft delete with grace period)
            for trace in traces:
                await self.soft_delete_trace(trace.trace_id)
                report.traces_deleted += 1

            # Anonymize user in metrics (preserve aggregates)
            metrics_updated = await self.anonymize_user_metrics(user_id, org_id)
            report.metrics_anonymized = metrics_updated

            # Delete user profile and settings
            await self.delete_user_profile(user_id, org_id)
            await self.delete_user_settings(user_id, org_id)

            # Delete authentication credentials
            await self.delete_user_credentials(user_id)

            # Mark completion
            report.completion_time = datetime.utcnow()
            report.status = "completed"

            # Create compliance audit record
            await self.audit_log.record(
                action="gdpr_deletion",
                user_id=user_id,
                org_id=org_id,
                request_id=request_id,
                report=report.to_dict(),
                retention_period="7_years",  # Legal requirement
            )

            return report

        except Exception as e:
            report.status = "failed"
            report.error = str(e)
            await self.audit_log.record(
                action="gdpr_deletion_failed",
                user_id=user_id,
                request_id=request_id,
                error=str(e),
            )
            raise

    async def export_user_data(
        self,
        user_id: str,
        org_id: str,
    ) -> bytes:
        """Export all user data per GDPR Article 20 (data portability).

        Provides user with complete copy of their data in machine-readable format.

        Args:
            user_id: Unique identifier of the user
            org_id: Organization identifier

        Returns:
            JSON export of all user data
        """
        data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "org_id": org_id,
            "data": {}
        }

        # User profile
        data["data"]["user_profile"] = await self.get_user_profile(user_id)

        # All traces
        traces = await self.get_user_traces(user_id, org_id)
        data["data"]["traces"] = [t.to_dict() for t in traces]

        # User settings and preferences
        data["data"]["settings"] = await self.get_user_settings(user_id)

        # API keys and access tokens (masked)
        data["data"]["api_keys"] = await self.get_masked_api_keys(user_id)

        # Audit log of user actions
        data["data"]["activity_log"] = await self.get_user_activity_log(user_id)

        # Log export for audit
        await self.audit_log.record(
            action="gdpr_export",
            user_id=user_id,
            org_id=org_id,
            record_count=len(traces),
        )

        return json.dumps(data, indent=2).encode()

    async def process_data_subject_request(
        self,
        request_type: str,
        user_id: str,
        org_id: str,
    ) -> dict:
        """Process GDPR data subject access request (DSAR).

        Request types:
        - access: Provide copy of data (Article 15)
        - rectification: Correct inaccurate data (Article 16)
        - deletion: Delete data (Article 17)
        - restriction: Restrict processing (Article 18)
        - portability: Export data (Article 20)
        - objection: Object to processing (Article 21)
        """
        request_id = f"dsar-{datetime.utcnow().timestamp()}"

        handlers = {
            "access": self.export_user_data,
            "deletion": self.delete_user_data,
            "portability": self.export_user_data,
        }

        handler = handlers.get(request_type)
        if not handler:
            raise ValueError(f"Unsupported request type: {request_type}")

        return await handler(user_id, org_id)
```

### Article 30 - Records of Processing Activities

```yaml
# GDPR Records of Processing Activities
processing_activities:
  - name: "AI Agent Trace Collection"
    purpose: "Monitor and debug AI agent interactions"
    legal_basis: "Legitimate interest (service provision)"
    data_categories:
      - Agent prompts and completions
      - Metadata (timestamps, IDs)
      - Performance metrics
    data_subjects: "End users of customer AI applications"
    recipients: "Customer organization administrators"
    retention: "30 days (default), up to 365 days (enterprise)"
    security_measures:
      - Encryption at rest and in transit
      - RBAC access controls
      - Audit logging

  - name: "User Authentication and Authorization"
    purpose: "Secure access to platform"
    legal_basis: "Contract performance"
    data_categories:
      - Email address
      - Hashed passwords
      - API keys
    data_subjects: "Platform users"
    recipients: "Internal security team only"
    retention: "Duration of account + 90 days"
    security_measures:
      - bcrypt password hashing
      - MFA support
      - Session management
```

## Data Retention Policies

Automated data retention ensures compliance with legal requirements while optimizing storage costs and respecting user privacy.

```yaml
# retention-config.yaml
retention:
  # Default retention periods
  default:
    traces: 30d
    metrics: 90d
    audit_logs: 365d
    api_logs: 30d
    error_logs: 90d

  # Tier-based retention
  tiers:
    free:
      traces: 7d
      metrics: 30d
      audit_logs: 90d
      storage_limit: "1GB"

    professional:
      traces: 30d
      metrics: 90d
      audit_logs: 365d
      storage_limit: "100GB"

    enterprise:
      traces: custom  # Per contract, up to 7 years
      metrics: custom
      audit_logs: 7y  # GDPR compliance requirement
      storage_limit: custom

  # Compliance-driven retention
  compliance:
    gdpr_audit_logs: 7y  # Article 30 requirement
    financial_records: 7y  # Tax compliance
    hipaa_audit_logs: 6y  # HIPAA requirement
    security_incidents: 7y  # SOC 2 requirement

  # Enforcement configuration
  enforcement:
    schedule: "0 2 * * *"  # Daily at 2 AM UTC
    method: "soft_delete_then_purge"
    soft_delete_period: 7d  # Grace period before hard delete
    batch_size: 10000  # Records per batch
    notification:
      enabled: true
      days_before: [30, 7, 1]  # Warn before deletion

  # Data lifecycle stages
  lifecycle:
    - stage: "active"
      duration: "per_tier"
      storage_class: "hot"

    - stage: "archived"
      duration: "90d"
      storage_class: "cold"
      compression: true

    - stage: "soft_deleted"
      duration: "7d"
      storage_class: "cold"
      recoverable: true

    - stage: "purged"
      irreversible: true
```

**Automated Retention Enforcement:**

```python
class RetentionEnforcer:
    """Automated data retention enforcement."""

    async def enforce_retention_policies(self):
        """Daily execution of retention policies."""
        policies = await self.load_retention_policies()

        for policy in policies:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - policy.retention_period

            # Find data eligible for deletion
            eligible_data = await self.find_eligible_data(
                data_type=policy.data_type,
                cutoff_date=cutoff_date,
            )

            # Send notifications before deletion
            if policy.notification_enabled:
                await self.send_retention_notifications(
                    eligible_data=eligible_data,
                    days_before=policy.notification_days,
                )

            # Execute soft delete
            for batch in self.batch(eligible_data, policy.batch_size):
                await self.soft_delete_batch(batch)
                await asyncio.sleep(1)  # Rate limiting

            # Purge soft-deleted data after grace period
            purge_cutoff = datetime.utcnow() - timedelta(days=7)
            await self.purge_soft_deleted(
                data_type=policy.data_type,
                cutoff_date=purge_cutoff,
            )

            # Log retention execution
            await self.audit_log.record(
                action="retention_enforcement",
                policy=policy.name,
                records_deleted=len(eligible_data),
                cutoff_date=cutoff_date,
            )
```

## Encryption Standards

### Encryption at Rest

**ClickHouse Database Encryption:**

```yaml
# ClickHouse encryption configuration
clickhouse:
  encryption:
    method: "aes_256_gcm"
    key_management: "aws_kms"  # or HashiCorp Vault
    key_rotation: 90d

    # Master key configuration
    master_key:
      service: "aws_kms"
      key_id: "arn:aws:kms:us-east-1:ACCOUNT:key/KEY_ID"
      region: "us-east-1"

  # Column-level encryption for sensitive fields
  sensitive_columns:
    - table: "agent_traces"
      column: "prompt_content"
      encryption: "envelope"  # Envelope encryption with tenant key
      searchable: false  # Cannot query encrypted content

    - table: "agent_traces"
      column: "completion_content"
      encryption: "envelope"
      searchable: false

    - table: "users"
      column: "email"
      encryption: "deterministic"  # Allows exact match queries
      searchable: true

    - table: "api_keys"
      column: "key_hash"
      encryption: "one_way"  # Bcrypt hashing
      searchable: false

  # Backup encryption
  backups:
    encryption: "aes_256_gcm"
    key_source: "aws_kms"
    separate_key: true  # Different key from primary data
```

**Object Storage Encryption:**

```yaml
# S3 bucket encryption
s3:
  encryption:
    default: "SSE-KMS"
    kms_key_id: "arn:aws:kms:us-east-1:ACCOUNT:key/KEY_ID"

  bucket_policies:
    - name: "trace-exports"
      encryption: "AES256"
      versioning: true
      lifecycle:
        - transition_to_glacier: 90d
        - expiration: 365d

    - name: "model-artifacts"
      encryption: "SSE-KMS"
      versioning: true
      retention: "7y"
```

### Encryption in Transit

**TLS Configuration:**

```yaml
# TLS/SSL configuration
tls:
  min_version: "1.3"

  # Cipher suites (in preference order)
  cipher_suites:
    - TLS_AES_256_GCM_SHA384
    - TLS_CHACHA20_POLY1305_SHA256
    - TLS_AES_128_GCM_SHA256

  # Certificate management
  certificates:
    provider: "letsencrypt"
    rotation: 30d
    auto_renewal: true

  # HSTS configuration
  hsts:
    enabled: true
    max_age: 31536000  # 1 year
    include_subdomains: true
    preload: true

  # Certificate pinning for critical services
  pinning:
    enabled: true
    backup_pins: 2
    max_age: 60d
```

## Audit Logging

Comprehensive audit logging for compliance reporting and security monitoring.

```python
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

class AuditAction(Enum):
    """Enumeration of auditable actions."""

    # Authentication and authorization
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    MFA_ENABLED = "auth.mfa_enabled"
    MFA_DISABLED = "auth.mfa_disabled"
    PASSWORD_CHANGED = "auth.password_changed"
    API_KEY_CREATED = "auth.api_key_created"
    API_KEY_REVOKED = "auth.api_key_revoked"

    # Data access
    TRACE_VIEW = "trace.view"
    TRACE_SEARCH = "trace.search"
    TRACE_EXPORT = "trace.export"
    TRACE_DELETE = "trace.delete"
    METRICS_VIEW = "metrics.view"
    METRICS_EXPORT = "metrics.export"

    # Administrative actions
    USER_CREATE = "user.create"
    USER_UPDATE = "user.update"
    USER_DELETE = "user.delete"
    ROLE_ASSIGN = "role.assign"
    ROLE_REVOKE = "role.revoke"
    ORG_SETTINGS_CHANGE = "org.settings_change"
    RETENTION_POLICY_CHANGE = "retention.policy_change"

    # Compliance actions
    GDPR_DELETION = "gdpr.deletion"
    GDPR_EXPORT = "gdpr.export"
    GDPR_RECTIFICATION = "gdpr.rectification"
    DATA_RETENTION_PURGE = "retention.purge"
    SECURITY_INCIDENT = "security.incident"
    ACCESS_REVIEW = "compliance.access_review"

class AuditLogger:
    """Compliance audit logging with tamper-proof records."""

    def __init__(self, db, encryption_service):
        self.db = db
        self.encryption = encryption_service

    async def log(
        self,
        action: AuditAction,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        org_id: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        result: str = "success",
    ):
        """Record audit log entry.

        Creates immutable audit record with cryptographic hash for tamper detection.

        Args:
            action: Type of action being audited
            actor_id: User or service performing the action
            resource_type: Type of resource being accessed
            resource_id: Unique identifier of the resource
            org_id: Organization context
            details: Additional contextual information
            ip_address: Source IP address
            user_agent: User agent string
            result: Action result (success, failure, partial)
        """
        timestamp = datetime.utcnow()

        entry = {
            "timestamp": timestamp,
            "action": action.value,
            "actor_id": actor_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "org_id": org_id,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "result": result,
        }

        # Add cryptographic hash for tamper detection
        entry["hash"] = self._compute_hash(entry)

        # Insert with write-once guarantee
        await self.db.insert("audit_logs", entry, immutable=True)

        # Stream to SIEM if configured
        if self.siem_enabled:
            await self.stream_to_siem(entry)

    async def query_logs(
        self,
        org_id: str,
        start_time: datetime,
        end_time: datetime,
        actions: Optional[list[AuditAction]] = None,
        actor_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Query audit logs for compliance reporting.

        Provides filtered access to audit logs with automatic retention
        enforcement and access control.
        """
        query = """
            SELECT
                timestamp,
                action,
                actor_id,
                resource_type,
                resource_id,
                details,
                ip_address,
                result
            FROM audit_logs
            WHERE org_id = {org_id:String}
              AND timestamp >= {start:DateTime}
              AND timestamp < {end:DateTime}
        """

        params = {
            "org_id": org_id,
            "start": start_time,
            "end": end_time,
        }

        if actions:
            query += " AND action IN {actions:Array(String)}"
            params["actions"] = [a.value for a in actions]

        if actor_id:
            query += " AND actor_id = {actor_id:String}"
            params["actor_id"] = actor_id

        if resource_type:
            query += " AND resource_type = {resource_type:String}"
            params["resource_type"] = resource_type

        query += " ORDER BY timestamp DESC LIMIT {limit:UInt32}"
        params["limit"] = limit

        return await self.db.query(query, params)

    async def generate_compliance_report(
        self,
        org_id: str,
        report_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> bytes:
        """Generate compliance report in PDF format.

        Report types:
        - soc2: SOC 2 control evidence
        - gdpr: GDPR Article 30 records
        - hipaa: HIPAA access logs
        - security: Security incident summary
        """
        logs = await self.query_logs(
            org_id=org_id,
            start_time=start_date,
            end_time=end_date,
        )

        report = self._format_report(
            report_type=report_type,
            logs=logs,
            start_date=start_date,
            end_date=end_date,
        )

        # Log report generation
        await self.log(
            action=AuditAction.METRICS_EXPORT,
            actor_id="system",
            resource_type="compliance_report",
            resource_id=f"{report_type}-{start_date.date()}",
            org_id=org_id,
            details={"report_type": report_type, "record_count": len(logs)},
        )

        return report

    def _compute_hash(self, entry: dict) -> str:
        """Compute cryptographic hash of audit entry."""
        # Sort keys for deterministic hashing
        canonical = json.dumps(entry, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
```

## Regional Data Residency

Multi-region deployment ensures data residency compliance for global customers.

```yaml
# Multi-region deployment configuration
regions:
  us-east-1:
    name: "US East (Virginia)"
    storage: "clickhouse-us-east-1.a11i.internal"
    collectors: "collector-us-east-1.a11i.dev"
    api: "api-us-east-1.a11i.dev"
    compliance:
      - SOC2
      - CCPA
    customers:
      - default
      - us_customers

  eu-west-1:
    name: "EU West (Ireland)"
    storage: "clickhouse-eu-west-1.a11i.internal"
    collectors: "collector-eu-west-1.a11i.dev"
    api: "api-eu.a11i.dev"
    compliance:
      - SOC2
      - GDPR
      - EU_AI_ACT
    gdpr_compliant: true
    data_processing_agreement: required
    customers:
      - eu_customers
      - gdpr_required

  ap-southeast-1:
    name: "Asia Pacific (Singapore)"
    storage: "clickhouse-ap-southeast-1.a11i.internal"
    collectors: "collector-ap-southeast-1.a11i.dev"
    api: "api-apac.a11i.dev"
    compliance:
      - SOC2
      - PDPA  # Singapore Personal Data Protection Act
    customers:
      - apac_customers

# Tenant region assignment
tenant_routing:
  default_region: "us-east-1"

  routing_rules:
    - condition: "customer.country IN ['DE', 'FR', 'IT', 'ES', 'NL', 'BE']"
      region: "eu-west-1"
      reason: "GDPR compliance"

    - condition: "customer.country IN ['SG', 'MY', 'TH', 'ID']"
      region: "ap-southeast-1"
      reason: "Data residency preference"

    - condition: "customer.compliance_requirements CONTAINS 'GDPR'"
      region: "eu-west-1"
      reason: "GDPR compliance requirement"

  cross_region:
    enabled: false  # Strict regional isolation
    replication: false  # No cross-region replication

# Regional data isolation
isolation:
  network:
    - VPC per region
    - No cross-region VPC peering
    - Regional NAT gateways

  storage:
    - Regional ClickHouse clusters
    - Regional S3 buckets
    - No cross-region backup replication

  encryption:
    - Regional KMS keys
    - No cross-region key sharing
```

## HIPAA Compliance

Healthcare customers requiring Protected Health Information (PHI) handling must have Business Associate Agreement (BAA) in place.

```yaml
hipaa:
  # PHI handling requirements
  phi_handling:
    scope: "AI agent prompts/completions may contain PHI"
    requirements:
      - Business Associate Agreement (BAA) required
      - Enhanced encryption requirements
      - Extended audit log retention (6 years)
      - Restricted data access
      - Additional training for personnel

  # Administrative safeguards
  administrative:
    access_control:
      - Unique user identification (§164.312(a)(2)(i))
      - Emergency access procedures (§164.312(a)(2)(ii))
      - Automatic logoff after 30 minutes (§164.312(a)(2)(iii))
      - Encryption and decryption (§164.312(a)(2)(iv))

    workforce_training:
      - HIPAA awareness training (annual)
      - Security reminders (quarterly)
      - Protection from malicious software
      - Log-in monitoring
      - Password management

    incident_response:
      - Incident reporting procedures
      - Breach notification protocol (within 60 days)
      - Incident investigation and documentation

  # Physical safeguards
  physical:
    facility_access:
      - Data center SOC 2 Type II certification
      - Visitor logs and escort requirements
      - Badge access controls

    workstation_security:
      - Screen lock requirements
      - Clean desk policy
      - Device encryption

  # Technical safeguards
  technical:
    access_control:
      implementation:
        - Role-based access control (RBAC)
        - Principle of least privilege
        - Access review (quarterly)
      evidence:
        - Access control lists
        - User access logs
        - Quarterly access reviews

    audit_controls:
      implementation:
        - Comprehensive audit logging
        - 6-year retention for audit logs
        - Tamper-proof logging (write-once storage)
        - Regular audit log reviews
      evidence:
        - Audit log exports
        - Log review reports
        - Anomaly detection alerts

    integrity_controls:
      implementation:
        - Data backup and recovery (daily backups)
        - Checksums for data integrity verification
        - Version control for configuration
      evidence:
        - Backup success logs
        - Recovery test results
        - Integrity verification reports

    transmission_security:
      implementation:
        - TLS 1.3 for all data in transit
        - End-to-end encryption for sensitive data
        - VPN for remote access
      evidence:
        - TLS configuration audits
        - Network traffic analysis
        - VPN access logs

  # Breach notification
  breach_notification:
    discovery_to_notification: "60 days"
    notification_recipients:
      - Affected individuals
      - HHS Office for Civil Rights
      - Media (if > 500 individuals)

    documentation_requirements:
      - Date of breach discovery
      - Description of breach
      - Types of PHI involved
      - Steps individuals should take
      - What organization is doing
```

## CCPA Compliance

California Consumer Privacy Act requirements for California residents.

```yaml
ccpa:
  consumer_rights:
    - name: "Right to Know"
      implementation: "Data export API (GDPR Article 20 compatible)"
      endpoint: "/api/privacy/export"

    - name: "Right to Delete"
      implementation: "Data deletion API (GDPR Article 17 compatible)"
      endpoint: "/api/privacy/delete"

    - name: "Right to Opt-Out"
      implementation: "Do Not Sell preference"
      endpoint: "/api/privacy/opt-out"
      note: "a11i does not sell personal information"

    - name: "Right to Non-Discrimination"
      implementation: "No service degradation for privacy requests"
      enforcement: "Policy and training"

  data_categories_collected:
    - Identifiers (email, user ID, IP address)
    - Commercial information (subscription tier, usage)
    - Internet activity (API usage, trace data)
    - Geolocation data (country, region)

  business_purpose:
    - Service provision and maintenance
    - Usage analytics and improvements
    - Security and fraud prevention
    - Regulatory compliance

  disclosure_requirements:
    privacy_policy_update: "Annual or upon material change"
    consumer_request_response: "45 days (extendable to 90 days)"
    verification_method: "Email verification + account authentication"
```

## EU AI Act Compliance

Emerging AI regulation compliance (high-risk AI systems).

```yaml
eu_ai_act:
  risk_classification: "limited_risk"  # Transparency obligations

  transparency_requirements:
    - name: "AI System Disclosure"
      implementation: "Clear labeling of AI-generated content in UI"
      status: "planned"

    - name: "Technical Documentation"
      implementation: "Comprehensive model documentation"
      includes:
        - Model architecture and training data
        - Performance metrics and limitations
        - Bias testing results
      status: "planned"

    - name: "Audit Trail"
      implementation: "Complete logging of AI system operations"
      retention: "As per GDPR (7 years for audit logs)"
      status: "designed"

  risk_management:
    - Risk assessment documentation
    - Continuous monitoring and testing
    - Post-market monitoring plan

  human_oversight:
    - Human-in-the-loop for high-stakes decisions
    - Override mechanisms
    - Monitoring dashboards

  data_governance:
    - Training data documentation
    - Bias detection and mitigation
    - Data quality assurance procedures
```

## Key Takeaways

> **Compliance Summary**
>
> - **Multi-Framework Approach**: a11i is designed to meet requirements across SOC 2, GDPR, HIPAA, CCPA, and emerging EU AI Act regulations
> - **Privacy by Design**: Data protection controls built into system architecture, not added as afterthoughts
> - **Automated Enforcement**: Retention policies, encryption, and audit logging automated to ensure consistent compliance
> - **Regional Isolation**: Strict regional data residency ensures compliance with local data protection laws
> - **User Rights**: Comprehensive implementation of data subject rights (access, deletion, portability, rectification)
> - **Audit Trail**: Tamper-proof audit logging with 6-7 year retention for regulatory compliance
> - **Continuous Monitoring**: Automated evidence collection for SOC 2 and other compliance frameworks
> - **Encryption Everywhere**: AES-256 at rest, TLS 1.3 in transit, envelope encryption for sensitive fields
>
> **Next Steps**:
> 1. Complete SOC 2 Type II certification (in progress)
> 2. Obtain HIPAA compliance attestation for healthcare customers
> 3. Prepare for EU AI Act implementation (2025-2026)
> 4. Annual compliance audits and policy reviews

## Cross-References

**Related Documentation:**

- [Security Architecture](/home/becker/projects/a11i/docs/05-security-compliance/security-architecture.md) - Detailed security controls and threat model
- [Privacy Controls](/home/becker/projects/a11i/docs/05-security-compliance/privacy-controls.md) - PII redaction and data minimization
- [Audit Logging](/home/becker/projects/a11i/docs/05-security-compliance/audit-logging.md) - Detailed audit logging implementation
- [Data Retention](/home/becker/projects/a11i/docs/05-security-compliance/data-retention.md) - Comprehensive retention policies
- [Multi-Tenancy](/home/becker/projects/a11i/docs/02-core-architecture/multi-tenancy.md) - Tenant isolation and data segregation
- [RBAC](/home/becker/projects/a11i/docs/05-security-compliance/rbac.md) - Role-based access control implementation

**External Resources:**

- [SOC 2 Trust Service Criteria](https://www.aicpa.org/interestareas/frc/assuranceadvisoryservices/sorhome)
- [GDPR Full Text](https://gdpr-info.eu/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [CCPA Official Site](https://oag.ca.gov/privacy/ccpa)
- [EU AI Act](https://artificialintelligenceact.eu/)

---

*This documentation is maintained by the Security & Compliance Team. Last updated: 2025-11-26*
