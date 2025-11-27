---
title: "Enterprise Audit Logging"
category: "Enterprise"
tags: ["audit", "logging", "compliance", "security", "siem", "gdpr", "hipaa", "soc2"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "./enterprise-features.md"
  - "./sso-integration.md"
  - "../05-security-compliance/rbac-auth.md"
  - "../05-security-compliance/compliance-framework.md"
---

# Enterprise Audit Logging

## Table of Contents

1. [Overview](#overview)
2. [What Actions to Audit](#what-actions-to-audit)
3. [Audit Log Format Specification](#audit-log-format-specification)
4. [Storage and Retention Policies](#storage-and-retention-policies)
5. [Compliance Reporting Features](#compliance-reporting-features)
6. [Access Logs](#access-logs)
7. [Change Logs](#change-logs)
8. [Log Export Capabilities](#log-export-capabilities)
9. [SIEM Integration](#siem-integration)
10. [Key Takeaways](#key-takeaways)

---

## Overview

Enterprise audit logging provides comprehensive, tamper-proof records of all security-relevant events in the a11i platform. Audit logs are essential for security monitoring, compliance reporting, incident investigation, and forensic analysis.

### Why Audit Logging Matters

**For Security Teams:**
- Detect and investigate security incidents
- Monitor for suspicious access patterns
- Track privileged user activities
- Identify unauthorized access attempts
- Support forensic analysis after breaches

**For Compliance:**
- SOC 2 Trust Service Criteria CC6.1 (audit logging requirements)
- GDPR Article 30 (records of processing activities)
- HIPAA §164.312(b) (audit controls)
- PCI DSS Requirement 10 (track and monitor all access)
- ISO 27001 A.12.4.1 (event logging)

**For Operations:**
- Troubleshoot user access issues
- Track configuration changes
- Monitor system usage patterns
- Generate activity reports
- Support customer inquiries about data access

### Enterprise vs. Standard Logging

| Feature | Open Source | Professional | Enterprise |
|---------|-------------|--------------|------------|
| **Retention** | 90 days | 365 days | 7 years (customizable) |
| **Event Types** | Basic (auth, data access) | Enhanced | Complete (all actions) |
| **Export** | Manual CSV | API access | Automated exports, SIEM integration |
| **Tamper Protection** | Basic | Enhanced | Cryptographic hash chain |
| **Compliance Reports** | None | Quarterly | Custom schedules |
| **SIEM Integration** | None | None | Real-time streaming |
| **Log Analysis** | Manual | Dashboard | AI-powered anomaly detection |
| **Storage Location** | Shared | Shared | Dedicated or customer-owned |

---

## What Actions to Audit

Enterprise audit logging captures comprehensive events across all system components.

### Authentication Events

**User Authentication:**
- Login attempts (successful and failed)
- Logout events
- Password changes and resets
- MFA enrollment and verification
- Session creation and termination
- Account lockouts
- SSO/SAML authentication flows

**Service Authentication:**
- API key creation and revocation
- Service account access
- OAuth token issuance
- JWT token validation failures

**Example Audit Log:**

```json
{
  "event_id": "evt_2TkL9mN3pQ",
  "timestamp": "2025-11-26T14:23:45.123Z",
  "action": "auth.login_success",
  "actor": {
    "user_id": "usr_abc123",
    "email": "alice@company.com",
    "ip_address": "203.0.113.42",
    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
  },
  "authentication": {
    "method": "saml",
    "idp": "okta",
    "session_id": "sess_xyz789",
    "mfa_verified": true
  },
  "result": "success",
  "metadata": {
    "organization_id": "org_acme",
    "location": {
      "country": "US",
      "region": "California",
      "city": "San Francisco"
    }
  }
}
```

### Authorization Decisions

**Permission Checks:**
- Access control evaluations
- Role assignments and revocations
- Permission grants and denials
- Custom role creation and modification
- Group membership changes

**Example Audit Log:**

```json
{
  "event_id": "evt_7PqR8sT4uV",
  "timestamp": "2025-11-26T14:25:12.456Z",
  "action": "authz.permission_check",
  "actor": {
    "user_id": "usr_abc123",
    "email": "alice@company.com"
  },
  "resource": {
    "type": "project",
    "id": "proj_customer_api",
    "name": "Customer API Observability"
  },
  "permission": "traces.export",
  "result": "granted",
  "reason": "user has role workspace_admin on workspace ws_engineering",
  "metadata": {
    "organization_id": "org_acme",
    "workspace_id": "ws_engineering"
  }
}
```

### Data Access Events

**Trace and Metrics Access:**
- Trace queries and searches
- Dashboard views
- Metric queries
- Data exports
- Report generation
- API data retrieval

**Example Audit Log:**

```json
{
  "event_id": "evt_3WxY9zA1bC",
  "timestamp": "2025-11-26T14:30:22.789Z",
  "action": "data.traces_export",
  "actor": {
    "user_id": "usr_abc123",
    "email": "alice@company.com"
  },
  "resource": {
    "type": "project",
    "id": "proj_customer_api",
    "name": "Customer API Observability"
  },
  "data_access": {
    "query": {
      "time_range": "last_7_days",
      "filters": ["status:error", "user_id:*"],
      "trace_count": 15234
    },
    "export_format": "json",
    "file_size_bytes": 45678912,
    "contains_pii": true
  },
  "result": "success",
  "metadata": {
    "organization_id": "org_acme"
  }
}
```

### Administrative Actions

**User Management:**
- User creation, updates, deletion
- Role assignments
- Organization membership changes
- Invitation sends and accepts

**Organization Configuration:**
- Settings changes
- Feature flag modifications
- Integration configurations
- Billing changes
- Data retention policy updates

**Example Audit Log:**

```json
{
  "event_id": "evt_4DfG5hJ6kL",
  "timestamp": "2025-11-26T15:00:00.000Z",
  "action": "admin.role_assign",
  "actor": {
    "user_id": "usr_def456",
    "email": "bob@company.com",
    "role": "org_admin"
  },
  "resource": {
    "type": "user",
    "id": "usr_ghi789",
    "email": "carol@company.com"
  },
  "changes": {
    "before": {
      "roles": [
        {
          "role": "project_viewer",
          "scope": "project",
          "resource_id": "proj_customer_api"
        }
      ]
    },
    "after": {
      "roles": [
        {
          "role": "project_viewer",
          "scope": "project",
          "resource_id": "proj_customer_api"
        },
        {
          "role": "workspace_admin",
          "scope": "workspace",
          "resource_id": "ws_engineering"
        }
      ]
    }
  },
  "result": "success",
  "metadata": {
    "organization_id": "org_acme",
    "reason": "Promotion to team lead"
  }
}
```

### Compliance-Specific Events

**GDPR Data Subject Requests:**
- Right to access requests
- Right to deletion (right to be forgotten)
- Data portability exports
- Consent management changes

**HIPAA Access Controls:**
- PHI access events
- Encryption key access
- Backup/restoration activities
- Breach notification events

**Example Audit Log:**

```json
{
  "event_id": "evt_5MnO6pQ7rS",
  "timestamp": "2025-11-26T16:15:30.123Z",
  "action": "compliance.gdpr_deletion_request",
  "actor": {
    "user_id": "usr_jkl012",
    "email": "privacy-officer@company.com",
    "role": "compliance_manager"
  },
  "resource": {
    "type": "user",
    "id": "usr_mno345",
    "email": "former-employee@company.com"
  },
  "compliance": {
    "regulation": "gdpr",
    "request_type": "deletion",
    "request_id": "dsar_abc123",
    "legal_basis": "Article 17 - Right to Erasure",
    "data_categories": [
      "traces",
      "user_profile",
      "audit_logs"
    ],
    "records_affected": 45231,
    "completion_date": "2025-11-27T16:15:30.123Z"
  },
  "result": "success",
  "metadata": {
    "organization_id": "org_acme",
    "retention_override": "7_years_for_audit"
  }
}
```

---

## Audit Log Format Specification

All audit logs follow a standardized JSON schema to ensure consistency, machine-readability, and SIEM compatibility.

### Core Schema

```typescript
/**
 * Audit Log Event Schema
 * Version: 2.0
 * Compliant with: SOC 2, GDPR Article 30, HIPAA §164.312(b)
 */
interface AuditLogEvent {
  // Required fields
  event_id: string;              // Unique event identifier (evt_*)
  timestamp: string;              // ISO 8601 timestamp with timezone
  action: string;                 // Action enumeration (category.action)
  actor: Actor;                   // Who performed the action
  result: "success" | "failure" | "partial";  // Action outcome

  // Conditional fields
  resource?: Resource;            // Resource being acted upon
  changes?: ChangeSet;            // Before/after state (for modifications)
  authentication?: AuthContext;   // Authentication details
  authorization?: AuthzContext;   // Authorization details
  data_access?: DataAccessContext; // Data access details
  compliance?: ComplianceContext; // Compliance-specific details

  // Metadata
  metadata: {
    organization_id: string;
    workspace_id?: string;
    project_id?: string;
    correlation_id?: string;      // Link related events
    [key: string]: any;            // Extensible metadata
  };

  // Tamper protection
  hash: string;                   // SHA-256 hash of event content
  previous_hash?: string;         // Hash of previous event (hash chain)
}

interface Actor {
  user_id?: string;               // User ID (if user action)
  service_id?: string;            // Service ID (if automated action)
  email?: string;                 // User email
  ip_address: string;             // Source IP address
  user_agent?: string;            // User agent string
  role?: string;                  // Actor's role at time of action
  session_id?: string;            // Session identifier
}

interface Resource {
  type: string;                   // Resource type (user, project, trace, etc.)
  id: string;                     // Resource unique identifier
  name?: string;                  // Human-readable name
  parent_id?: string;             // Parent resource ID
}

interface ChangeSet {
  before: object;                 // State before change
  after: object;                  // State after change
  diff?: object;                  // Computed diff (optional)
}

interface AuthContext {
  method: "password" | "saml" | "oidc" | "api_key" | "service_account";
  idp?: string;                   // Identity provider (for SSO)
  session_id?: string;
  mfa_verified?: boolean;
  saml_assertion_id?: string;
  oauth_token_id?: string;
}

interface AuthzContext {
  permission: string;             // Permission checked
  granted: boolean;
  reason?: string;                // Why granted/denied
  evaluated_policies?: string[];  // Policies evaluated
}

interface DataAccessContext {
  query?: object;                 // Query parameters
  trace_count?: number;           // Number of traces accessed
  metric_count?: number;          // Number of metrics accessed
  export_format?: string;         // Export format (csv, json, parquet)
  file_size_bytes?: number;
  contains_pii?: boolean;         // PII detected in data
  redaction_applied?: boolean;
}

interface ComplianceContext {
  regulation: "gdpr" | "hipaa" | "ccpa" | "sox" | "pci_dss";
  request_type?: string;
  request_id?: string;
  legal_basis?: string;
  data_categories?: string[];
  records_affected?: number;
  completion_date?: string;
}
```

### Action Taxonomy

Actions follow a hierarchical naming convention: `category.action_detail`

**Categories:**

| Category | Description | Examples |
|----------|-------------|----------|
| `auth` | Authentication events | `auth.login_success`, `auth.logout`, `auth.mfa_enabled` |
| `authz` | Authorization decisions | `authz.permission_check`, `authz.access_denied` |
| `data` | Data access events | `data.traces_view`, `data.traces_export`, `data.metrics_query` |
| `admin` | Administrative actions | `admin.user_create`, `admin.role_assign`, `admin.settings_change` |
| `compliance` | Compliance-specific | `compliance.gdpr_export`, `compliance.audit_report_generated` |
| `system` | System events | `system.backup_completed`, `system.security_scan` |
| `integration` | External integrations | `integration.siem_export`, `integration.webhook_triggered` |

**Full Action Enumeration:**

```typescript
enum AuditAction {
  // Authentication (auth.*)
  AUTH_LOGIN_SUCCESS = "auth.login_success",
  AUTH_LOGIN_FAILED = "auth.login_failed",
  AUTH_LOGOUT = "auth.logout",
  AUTH_PASSWORD_CHANGED = "auth.password_changed",
  AUTH_PASSWORD_RESET_REQUESTED = "auth.password_reset_requested",
  AUTH_MFA_ENABLED = "auth.mfa_enabled",
  AUTH_MFA_DISABLED = "auth.mfa_disabled",
  AUTH_MFA_VERIFIED = "auth.mfa_verified",
  AUTH_SSO_LOGIN = "auth.sso_login",
  AUTH_API_KEY_CREATED = "auth.api_key_created",
  AUTH_API_KEY_REVOKED = "auth.api_key_revoked",
  AUTH_SESSION_EXPIRED = "auth.session_expired",
  AUTH_ACCOUNT_LOCKED = "auth.account_locked",

  // Authorization (authz.*)
  AUTHZ_PERMISSION_CHECK = "authz.permission_check",
  AUTHZ_ACCESS_GRANTED = "authz.access_granted",
  AUTHZ_ACCESS_DENIED = "authz.access_denied",
  AUTHZ_ROLE_ASSIGNED = "authz.role_assigned",
  AUTHZ_ROLE_REVOKED = "authz.role_revoked",

  // Data Access (data.*)
  DATA_TRACES_VIEW = "data.traces_view",
  DATA_TRACES_SEARCH = "data.traces_search",
  DATA_TRACES_EXPORT = "data.traces_export",
  DATA_TRACES_DELETE = "data.traces_delete",
  DATA_METRICS_VIEW = "data.metrics_view",
  DATA_METRICS_EXPORT = "data.metrics_export",
  DATA_DASHBOARD_VIEW = "data.dashboard_view",
  DATA_REPORT_GENERATED = "data.report_generated",

  // Administration (admin.*)
  ADMIN_USER_CREATED = "admin.user_created",
  ADMIN_USER_UPDATED = "admin.user_updated",
  ADMIN_USER_DELETED = "admin.user_deleted",
  ADMIN_USER_INVITED = "admin.user_invited",
  ADMIN_ROLE_ASSIGNED = "admin.role_assign",
  ADMIN_ROLE_REVOKED = "admin.role_revoke",
  ADMIN_CUSTOM_ROLE_CREATED = "admin.custom_role_created",
  ADMIN_ORG_SETTINGS_CHANGED = "admin.org_settings_changed",
  ADMIN_RETENTION_POLICY_CHANGED = "admin.retention_policy_changed",
  ADMIN_INTEGRATION_CONFIGURED = "admin.integration_configured",
  ADMIN_SSO_CONFIGURED = "admin.sso_configured",

  // Compliance (compliance.*)
  COMPLIANCE_GDPR_EXPORT = "compliance.gdpr_export",
  COMPLIANCE_GDPR_DELETION = "compliance.gdpr_deletion",
  COMPLIANCE_GDPR_RECTIFICATION = "compliance.gdpr_rectification",
  COMPLIANCE_HIPAA_PHI_ACCESS = "compliance.hipaa_phi_access",
  COMPLIANCE_AUDIT_REPORT_GENERATED = "compliance.audit_report_generated",
  COMPLIANCE_DATA_RETENTION_PURGE = "compliance.data_retention_purge",

  // System (system.*)
  SYSTEM_BACKUP_STARTED = "system.backup_started",
  SYSTEM_BACKUP_COMPLETED = "system.backup_completed",
  SYSTEM_BACKUP_FAILED = "system.backup_failed",
  SYSTEM_SECURITY_SCAN = "system.security_scan",
  SYSTEM_CONFIG_CHANGED = "system.config_changed",
  SYSTEM_CERTIFICATE_RENEWED = "system.certificate_renewed",

  // Integration (integration.*)
  INTEGRATION_SIEM_EXPORT = "integration.siem_export",
  INTEGRATION_WEBHOOK_TRIGGERED = "integration.webhook_triggered",
  INTEGRATION_API_CALLED = "integration.api_called",
}
```

---

## Storage and Retention Policies

### Storage Architecture

**Multi-Tier Storage:**

```yaml
# Audit log storage tiers
storage_tiers:
  # Hot tier - Recent logs with fast query access
  hot:
    duration: "90d"
    storage_class: "nvme_ssd"
    compression: false
    indexing: "full_text_search"
    query_latency: "<100ms p95"

  # Warm tier - Less frequently accessed
  warm:
    duration: "365d"
    storage_class: "ssd"
    compression: true
    compression_ratio: "5:1"
    indexing: "partial"
    query_latency: "<1s p95"

  # Cold tier - Long-term compliance storage
  cold:
    duration: "7y"  # 7 years for compliance
    storage_class: "object_storage"
    compression: true
    compression_ratio: "10:1"
    indexing: "metadata_only"
    query_latency: "<5s p95"

  # Archive tier - Immutable compliance archives
  archive:
    duration: "indefinite"  # Compliance-driven
    storage_class: "glacier"
    compression: true
    write_once: true  # WORM storage
    retrieval_time: "hours"
```

### Retention Policies

**Enterprise Retention:**

```python
from datetime import timedelta
from enum import Enum

class RetentionPeriod(Enum):
    """Audit log retention periods."""

    # Standard retention
    AUTHENTICATION = timedelta(days=365)  # 1 year
    AUTHORIZATION = timedelta(days=365)   # 1 year
    DATA_ACCESS = timedelta(days=730)     # 2 years
    ADMINISTRATIVE = timedelta(days=2555) # 7 years

    # Compliance-driven retention
    GDPR_LOGS = timedelta(days=2555)      # 7 years (Article 30)
    HIPAA_LOGS = timedelta(days=2190)     # 6 years (§164.316(b)(2)(i))
    SOC2_LOGS = timedelta(days=2555)      # 7 years
    FINANCIAL = timedelta(days=2555)      # 7 years (SOX)

    # Custom enterprise retention
    ENTERPRISE_CUSTOM = None               # Configurable per customer

class AuditLogRetentionManager:
    """Manage audit log retention lifecycle."""

    async def apply_retention_policy(self):
        """Apply retention policies to audit logs."""

        # Get all retention policies
        policies = await self.get_retention_policies()

        for policy in policies:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - policy.retention_period

            # Find eligible logs
            eligible_logs = await self.find_logs_for_deletion(
                log_category=policy.category,
                before_date=cutoff_date,
                exclude_compliance=True  # Never delete compliance logs
            )

            # Move to cold storage before deletion
            if policy.transition_to_archive:
                await self.transition_to_archive(eligible_logs)

            # Soft delete with grace period
            await self.soft_delete_logs(
                logs=eligible_logs,
                grace_period=timedelta(days=30)
            )

            # Generate retention report
            await self.generate_retention_report(
                policy=policy,
                logs_deleted=len(eligible_logs)
            )

    async def transition_to_archive(
        self,
        logs: list,
        archive_type: str = "compliance"
    ):
        """Transition logs to immutable archive storage."""

        # Batch logs by time period
        batches = self.batch_logs_by_period(logs, period="month")

        for batch in batches:
            # Create archive file
            archive_file = await self.create_archive_file(
                logs=batch,
                format="parquet",  # Compressed columnar format
                compression="gzip"
            )

            # Calculate archive checksum
            checksum = self.calculate_checksum(archive_file)

            # Upload to WORM storage
            archive_id = await self.upload_to_worm_storage(
                file=archive_file,
                bucket=f"audit-archives-{archive_type}",
                retention_lock="7_years"
            )

            # Record archive metadata
            await self.record_archive_metadata(
                archive_id=archive_id,
                log_count=len(batch),
                time_range=(batch[0].timestamp, batch[-1].timestamp),
                checksum=checksum,
                retrieval_instructions="contact_compliance_team"
            )
```

### Tamper Protection

**Cryptographic Hash Chain:**

```python
import hashlib
import json
from typing import Optional

class TamperProofAuditLog:
    """Implement tamper-proof audit logging with hash chain."""

    def __init__(self):
        self.previous_hash: Optional[str] = None

    def compute_event_hash(self, event: dict) -> str:
        """Compute cryptographic hash of audit event.

        Creates tamper-evident hash chain by including
        previous event's hash in current event hash.
        """
        # Create canonical JSON representation
        canonical_event = json.dumps(
            event,
            sort_keys=True,
            separators=(',', ':')
        )

        # Include previous hash to create chain
        hash_input = f"{canonical_event}|{self.previous_hash or ''}"

        # Compute SHA-256 hash
        event_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        return event_hash

    async def log_event(self, event: dict):
        """Log event with tamper protection."""

        # Compute event hash
        event['hash'] = self.compute_event_hash(event)
        event['previous_hash'] = self.previous_hash

        # Store event (write-once)
        await self.store_event_immutable(event)

        # Update chain
        self.previous_hash = event['hash']

        # Periodic hash chain verification
        if self.should_verify_chain():
            await self.verify_hash_chain()

    async def verify_hash_chain(self) -> bool:
        """Verify integrity of audit log hash chain.

        Detects any tampering or corruption of audit logs.
        """
        # Retrieve all events in chronological order
        events = await self.get_all_events_ordered()

        previous_hash = None

        for event in events:
            # Recompute hash
            expected_hash = self.compute_event_hash_static(
                event,
                previous_hash
            )

            # Compare with stored hash
            if expected_hash != event['hash']:
                await self.alert_tampering_detected(
                    event_id=event['event_id'],
                    expected_hash=expected_hash,
                    actual_hash=event['hash']
                )
                return False

            previous_hash = event['hash']

        return True
```

---

## Compliance Reporting Features

### SOC 2 Compliance Reports

**Automated SOC 2 Evidence Collection:**

```python
from datetime import datetime, timedelta
from typing import List, Dict

class SOC2ComplianceReporter:
    """Generate SOC 2 compliance reports from audit logs."""

    async def generate_soc2_report(
        self,
        org_id: str,
        report_period_start: datetime,
        report_period_end: datetime
    ) -> Dict:
        """Generate comprehensive SOC 2 audit report.

        Covers Trust Service Criteria:
        - CC6.1: Logical and Physical Access Controls
        - CC6.2: System Operations
        - CC6.3: Change Management
        - CC7.2: System Monitoring
        """

        report = {
            "report_type": "soc2_type_ii",
            "organization_id": org_id,
            "period": {
                "start": report_period_start.isoformat(),
                "end": report_period_end.isoformat()
            },
            "controls": {}
        }

        # CC6.1: Access Controls
        report["controls"]["cc6_1"] = await self.collect_cc6_1_evidence(
            org_id, report_period_start, report_period_end
        )

        # CC6.2: System Operations
        report["controls"]["cc6_2"] = await self.collect_cc6_2_evidence(
            org_id, report_period_start, report_period_end
        )

        # CC6.3: Change Management
        report["controls"]["cc6_3"] = await self.collect_cc6_3_evidence(
            org_id, report_period_start, report_period_end
        )

        # CC7.2: System Monitoring
        report["controls"]["cc7_2"] = await self.collect_cc7_2_evidence(
            org_id, report_period_start, report_period_end
        )

        return report

    async def collect_cc6_1_evidence(
        self,
        org_id: str,
        start: datetime,
        end: datetime
    ) -> Dict:
        """CC6.1: Logical and Physical Access Controls evidence."""

        # Authentication events
        auth_events = await audit_log_repository.query(
            org_id=org_id,
            actions=[
                "auth.login_success",
                "auth.login_failed",
                "auth.mfa_enabled",
                "auth.password_changed"
            ],
            time_range=(start, end)
        )

        # Failed login attempts
        failed_logins = [e for e in auth_events if e.action == "auth.login_failed"]

        # MFA enrollment rate
        total_users = await user_repository.count_users(org_id)
        mfa_enabled_users = await user_repository.count_mfa_enabled(org_id)
        mfa_rate = (mfa_enabled_users / total_users * 100) if total_users > 0 else 0

        # Access reviews
        access_reviews = await audit_log_repository.query(
            org_id=org_id,
            actions=["admin.role_assign", "admin.role_revoke"],
            time_range=(start, end)
        )

        return {
            "control": "CC6.1 - Logical and Physical Access Controls",
            "evidence": {
                "total_authentication_events": len(auth_events),
                "failed_login_attempts": len(failed_logins),
                "mfa_enrollment_rate_percent": round(mfa_rate, 2),
                "access_reviews_performed": len(access_reviews),
                "password_changes": len([e for e in auth_events if e.action == "auth.password_changed"])
            },
            "compliance_status": "pass" if mfa_rate >= 95 else "warning",
            "recommendations": [
                "Enforce MFA for all users" if mfa_rate < 100 else None,
                "Investigate failed login patterns" if len(failed_logins) > 100 else None
            ]
        }
```

### GDPR Article 30 Reports

**Records of Processing Activities:**

```python
class GDPRArticle30Reporter:
    """Generate GDPR Article 30 records of processing activities."""

    async def generate_article_30_record(
        self,
        org_id: str,
        year: int
    ) -> Dict:
        """Generate annual Article 30 processing record."""

        # Query all data processing events
        processing_events = await audit_log_repository.query(
            org_id=org_id,
            actions=[
                "data.traces_view",
                "data.traces_export",
                "data.metrics_view",
                "compliance.gdpr_export",
                "compliance.gdpr_deletion"
            ],
            time_range=(
                datetime(year, 1, 1),
                datetime(year, 12, 31, 23, 59, 59)
            )
        )

        return {
            "record_type": "gdpr_article_30",
            "organization_id": org_id,
            "year": year,

            "processing_activities": [
                {
                    "activity_name": "AI Agent Observability Data Processing",
                    "purpose": "Monitor and debug AI agent interactions",
                    "legal_basis": "Legitimate interest (service provision)",
                    "data_categories": [
                        "Agent prompts and completions",
                        "Performance metrics",
                        "User identifiers (email, user ID)"
                    ],
                    "data_subjects": "End users of customer AI applications",
                    "recipients": "Customer organization administrators",
                    "retention_period": "30-365 days (configurable)",
                    "security_measures": [
                        "AES-256 encryption at rest",
                        "TLS 1.3 in transit",
                        "RBAC access controls",
                        "Comprehensive audit logging"
                    ],
                    "processing_volume": {
                        "traces_processed": sum(1 for e in processing_events if "traces" in e.action),
                        "data_exports": sum(1 for e in processing_events if "export" in e.action),
                        "deletion_requests": sum(1 for e in processing_events if "deletion" in e.action)
                    }
                }
            ],

            "data_subject_requests": await self.get_dsar_summary(org_id, year),

            "data_breaches": await self.get_breach_summary(org_id, year),

            "dpo_contact": "dpo@a11i.dev"
        }
```

### HIPAA Access Reports

```python
class HIPAAAccessReporter:
    """Generate HIPAA access reports for PHI."""

    async def generate_hipaa_access_report(
        self,
        org_id: str,
        month: int,
        year: int
    ) -> Dict:
        """Monthly HIPAA access report for PHI-containing traces."""

        # Query PHI access events
        phi_access_events = await audit_log_repository.query(
            org_id=org_id,
            actions=["data.traces_view", "data.traces_export"],
            filters={"contains_pii": True},
            time_range=(
                datetime(year, month, 1),
                datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
            )
        )

        return {
            "report_type": "hipaa_phi_access",
            "organization_id": org_id,
            "period": f"{year}-{month:02d}",

            "summary": {
                "total_phi_access_events": len(phi_access_events),
                "unique_users_accessing_phi": len(set(e.actor.user_id for e in phi_access_events)),
                "phi_exports": sum(1 for e in phi_access_events if "export" in e.action)
            },

            "access_by_user": await self.summarize_by_user(phi_access_events),

            "access_by_resource": await self.summarize_by_resource(phi_access_events),

            "security_incidents": await self.get_security_incidents(org_id, year, month),

            "compliance_notes": "All PHI access events logged per §164.312(b) requirements"
        }
```

---

## Access Logs

Track all data access events with detailed query and export information.

**Access Log Examples:**

```json
{
  "event_id": "evt_access_001",
  "timestamp": "2025-11-26T10:30:00.000Z",
  "action": "data.traces_view",
  "actor": {
    "user_id": "usr_alice",
    "email": "alice@company.com",
    "ip_address": "203.0.113.10",
    "role": "project_editor"
  },
  "resource": {
    "type": "project",
    "id": "proj_prod_api",
    "name": "Production API"
  },
  "data_access": {
    "query": {
      "time_range": "last_1_hour",
      "filters": ["status:error"],
      "trace_count": 127
    },
    "ui_view": "trace_explorer",
    "contains_pii": false
  },
  "result": "success",
  "metadata": {
    "organization_id": "org_acme",
    "session_duration_seconds": 145
  }
}
```

---

## Change Logs

Track all configuration and administrative changes with before/after states.

**Change Log Example:**

```json
{
  "event_id": "evt_change_001",
  "timestamp": "2025-11-26T11:00:00.000Z",
  "action": "admin.retention_policy_changed",
  "actor": {
    "user_id": "usr_bob",
    "email": "bob@company.com",
    "role": "org_admin"
  },
  "resource": {
    "type": "organization",
    "id": "org_acme",
    "name": "Acme Corporation"
  },
  "changes": {
    "before": {
      "retention_policy": {
        "traces": "30d",
        "metrics": "90d",
        "audit_logs": "365d"
      }
    },
    "after": {
      "retention_policy": {
        "traces": "90d",
        "metrics": "365d",
        "audit_logs": "2555d"
      }
    },
    "diff": {
      "traces": "30d → 90d",
      "metrics": "90d → 365d",
      "audit_logs": "365d → 2555d (7 years)"
    }
  },
  "result": "success",
  "metadata": {
    "organization_id": "org_acme",
    "reason": "HIPAA compliance requirement"
  }
}
```

---

## Log Export Capabilities

### Manual Export

**Export via UI:**

```
Settings → Audit Logs → Export
- Select time range
- Choose format (CSV, JSON, Parquet)
- Apply filters (action types, users, resources)
- Click "Export"
```

### API Export

```python
import requests

# Export audit logs via API
response = requests.get(
    "https://api.a11i.dev/v1/audit-logs/export",
    headers={
        "Authorization": "Bearer <API_KEY>",
        "Content-Type": "application/json"
    },
    params={
        "start_date": "2025-11-01T00:00:00Z",
        "end_date": "2025-11-30T23:59:59Z",
        "format": "json",
        "actions": ["auth.login_success", "data.traces_export"],
        "limit": 10000
    }
)

# Save to file
with open("audit_logs_november_2025.json", "wb") as f:
    f.write(response.content)
```

### Scheduled Exports

```yaml
# Automated export configuration
automated_exports:
  - name: "Monthly compliance export"
    schedule: "0 0 1 * *"  # First day of each month
    format: "parquet"
    destination: "s3://customer-compliance-bucket/audit-logs/"
    retention: "7y"

  - name: "Daily security export"
    schedule: "0 2 * * *"  # Daily at 2 AM
    format: "json"
    filters:
      actions: ["auth.login_failed", "authz.access_denied"]
    destination: "https://customer-siem.company.com/ingest"
```

---

## SIEM Integration

Stream audit logs in real-time to Security Information and Event Management (SIEM) systems.

### Supported SIEM Platforms

| SIEM Platform | Protocol | Status |
|---------------|----------|--------|
| **Splunk** | HTTP Event Collector (HEC) | ✓ Certified |
| **Datadog** | Datadog Logs API | ✓ Certified |
| **Sumo Logic** | HTTP Source | ✓ Supported |
| **Elastic (SIEM)** | Elasticsearch Bulk API | ✓ Supported |
| **Azure Sentinel** | Log Analytics API | ✓ Supported |
| **AWS Security Lake** | S3 + Glue Catalog | ✓ Supported |

### Configuration Example (Splunk)

```yaml
# SIEM Integration: Splunk
siem_integration:
  enabled: true
  provider: "splunk"

  connection:
    hec_url: "https://splunk.company.com:8088/services/collector"
    hec_token: "abcd1234-5678-90ef-ghij-klmnopqrstuv"
    verify_ssl: true

  streaming:
    batch_size: 100
    batch_interval: "30s"
    retry_attempts: 3
    retry_backoff: "exponential"

  filters:
    # Only send security-relevant events
    include_actions:
      - "auth.*"
      - "authz.access_denied"
      - "data.traces_export"
      - "admin.*"
      - "compliance.*"
      - "system.security_scan"

  enrichment:
    add_organization_name: true
    add_geolocation: true
    add_severity_level: true

  format:
    type: "json"
    timestamp_field: "timestamp"
    source: "a11i"
    sourcetype: "a11i:audit"
    index: "security"
```

### Real-Time Streaming

```python
import asyncio
import httpx
from typing import List

class SIEMStreamer:
    """Stream audit logs to SIEM in real-time."""

    def __init__(self, config: dict):
        self.config = config
        self.batch: List[dict] = []
        self.client = httpx.AsyncClient()

    async def stream_event(self, event: dict):
        """Add event to batch and flush if needed."""
        self.batch.append(self.format_for_siem(event))

        if len(self.batch) >= self.config["batch_size"]:
            await self.flush_batch()

    async def flush_batch(self):
        """Send batched events to SIEM."""
        if not self.batch:
            return

        try:
            # Format for Splunk HEC
            payload = {
                "event": self.batch,
                "source": "a11i",
                "sourcetype": "a11i:audit"
            }

            # Send to SIEM
            response = await self.client.post(
                self.config["hec_url"],
                headers={
                    "Authorization": f"Splunk {self.config['hec_token']}"
                },
                json=payload,
                timeout=30.0
            )

            response.raise_for_status()

            # Clear batch
            self.batch = []

        except Exception as e:
            # Log error and retry
            await self.handle_streaming_error(e)

    def format_for_siem(self, event: dict) -> dict:
        """Format audit event for SIEM ingestion."""
        return {
            "time": event["timestamp"],
            "host": "a11i-platform",
            "source": "a11i-audit",
            "sourcetype": "a11i:audit:json",
            "event": event,
            # Splunk-specific fields
            "severity": self.map_severity(event),
            "category": self.map_category(event)
        }
```

---

## Key Takeaways

> **Audit Logging Summary**
>
> **Comprehensive Coverage:**
> - All authentication and authorization events logged
> - Complete data access tracking with query details
> - Administrative changes with before/after states
> - Compliance-specific events (GDPR, HIPAA, SOC 2)
>
> **Enterprise Features:**
> - 7-year retention for compliance (customizable)
> - Tamper-proof logs with cryptographic hash chains
> - Multi-tier storage with automated lifecycle management
> - Real-time SIEM integration for security monitoring
> - Automated compliance report generation
>
> **Compliance Support:**
> - SOC 2 Trust Service Criteria evidence collection
> - GDPR Article 30 processing records
> - HIPAA PHI access reports
> - PCI DSS access tracking
> - ISO 27001 event logging requirements
>
> **Export & Integration:**
> - Manual export (CSV, JSON, Parquet)
> - API-based programmatic access
> - Scheduled automated exports
> - Real-time streaming to SIEM platforms
> - Batch archives for long-term storage
>
> **Security:**
> - Write-once storage for immutability
> - Cryptographic hash chains for tamper detection
> - Encrypted at rest and in transit
> - Role-based access to audit logs
> - Separate audit log retention from data retention

**Best Practices:**
1. Enable SIEM integration for real-time security monitoring
2. Set retention periods based on compliance requirements
3. Regularly review access patterns for anomalies
4. Generate compliance reports quarterly
5. Test audit log recovery procedures
6. Maintain separate backup of audit archives
7. Restrict audit log access to security team

---

**Related Documentation:**
- [Enterprise Features Overview](./enterprise-features.md)
- [SSO Integration Guide](./sso-integration.md)
- [RBAC and Authentication](../05-security-compliance/rbac-auth.md)
- [Compliance Framework](../05-security-compliance/compliance-framework.md)

---

*Document Status: Draft | Last Updated: 2025-11-26 | Maintained by: Security & Compliance Team*
