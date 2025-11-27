---
title: "Enterprise Features"
category: "Enterprise"
tags: ["enterprise", "features", "pricing", "sso", "rbac", "audit", "sla", "support"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "./sso-integration.md"
  - "./audit-logging.md"
  - "../05-security-compliance/rbac-auth.md"
  - "../05-security-compliance/compliance-framework.md"
  - "../02-architecture/deployment-modes.md"
---

# Enterprise Features

## Table of Contents

1. [Overview](#overview)
2. [Feature Comparison Matrix](#feature-comparison-matrix)
3. [Enterprise-Exclusive Features](#enterprise-exclusive-features)
4. [SSO and SAML Support](#sso-and-saml-support)
5. [Advanced RBAC with Custom Roles](#advanced-rbac-with-custom-roles)
6. [Comprehensive Audit Logging](#comprehensive-audit-logging)
7. [White-Labeling Options](#white-labeling-options)
8. [SLA Commitments](#sla-commitments)
9. [Priority Support Tiers](#priority-support-tiers)
10. [Data Retention Customization](#data-retention-customization)
11. [Dedicated Infrastructure Options](#dedicated-infrastructure-options)
12. [Enterprise Pricing Model](#enterprise-pricing-model)
13. [Key Takeaways](#key-takeaways)

---

## Overview

The **a11i Enterprise Edition** is designed for organizations with demanding requirements for security, compliance, scalability, and operational control. Enterprise features enable large-scale AI observability deployments with advanced access controls, dedicated support, and customizable infrastructure options.

### Enterprise Value Proposition

**For Enterprise Organizations:**
- **Security & Compliance**: SOC 2, GDPR, HIPAA compliance with advanced audit logging
- **Operational Control**: Custom data retention, dedicated infrastructure, regional data residency
- **Identity Integration**: SSO/SAML integration with existing identity providers (Okta, Azure AD, Google Workspace)
- **Advanced Access Management**: Custom RBAC roles, fine-grained permissions, multi-tier hierarchy
- **Guaranteed Performance**: 99.9% uptime SLA with financial penalties for non-compliance
- **Dedicated Support**: Named Technical Account Manager, priority incident response, custom onboarding

### Who Needs Enterprise?

| Use Case | Recommended Plan |
|----------|------------------|
| **Regulated Industries** (Healthcare, Finance, Government) | Enterprise - HIPAA/SOC 2 compliance required |
| **Large Engineering Teams** (500+ developers) | Enterprise - Advanced RBAC and SSO integration |
| **Multi-National Organizations** | Enterprise - Regional data residency options |
| **High-Volume Production** (>10M traces/month) | Enterprise - Dedicated infrastructure and custom retention |
| **Custom Integration Requirements** | Enterprise - Technical Account Manager and custom development |
| **Startups and Small Teams** (<50 developers) | Open Source or Professional - Cost-effective for smaller scale |

---

## Feature Comparison Matrix

| Feature Category | Open Source | Professional | Enterprise |
|------------------|-------------|--------------|------------|
| **Core Observability** |
| AI Agent Tracing | ✓ | ✓ | ✓ |
| Think→Act→Observe Loop Tracking | ✓ | ✓ | ✓ |
| Five Core Metrics (tokens, cost, context, tools, loops) | ✓ | ✓ | ✓ |
| OpenTelemetry-Native | ✓ | ✓ | ✓ |
| Multi-Provider Support (OpenAI, Anthropic, etc.) | ✓ | ✓ | ✓ |
| **Data & Storage** |
| Trace Retention | 7 days | 30 days | Custom (up to 7 years) |
| Metrics Retention | 30 days | 90 days | Custom (up to 7 years) |
| Audit Log Retention | 90 days | 365 days | 7 years (compliance) |
| Storage Limit | 1 GB | 100 GB | Unlimited |
| Data Export | CSV only | CSV, JSON | CSV, JSON, Parquet, custom |
| Regional Data Residency | US only | US, EU | US, EU, APAC, custom |
| **Access Control & Security** |
| Basic Authentication (email/password) | ✓ | ✓ | ✓ |
| API Keys | ✓ | ✓ | ✓ |
| Multi-Factor Authentication (MFA) | - | ✓ | ✓ |
| SSO/SAML Integration | - | - | ✓ |
| OIDC/OAuth Support | - | - | ✓ |
| SCIM User Provisioning | - | - | ✓ |
| Custom RBAC Roles | - | Limited | Unlimited |
| Fine-Grained Permissions | - | ✓ | ✓ |
| IP Allowlisting | - | ✓ | ✓ |
| Service Accounts | - | ✓ | ✓ |
| **Compliance & Audit** |
| SOC 2 Type II Certified Infrastructure | - | ✓ | ✓ |
| GDPR Compliance Tools | ✓ | ✓ | ✓ |
| HIPAA Compliance (BAA Available) | - | - | ✓ |
| Comprehensive Audit Logging | Basic | Enhanced | Complete |
| SIEM Integration | - | - | ✓ |
| Compliance Reporting | - | Monthly | Custom schedules |
| Data Residency Guarantees | - | - | ✓ |
| **Platform & Deployment** |
| Cloud-Hosted (SaaS) | ✓ | ✓ | ✓ |
| Self-Hosted (On-Premises) | ✓ | - | ✓ |
| Dedicated VPC | - | - | ✓ |
| Single-Tenant Deployment | - | - | ✓ |
| Custom Infrastructure | - | - | ✓ |
| Air-Gapped Deployment | - | - | ✓ (custom) |
| **Performance & Scale** |
| Uptime SLA | Best effort | 99.5% | 99.9% |
| Support Response Time | Community | 24 hours | 1 hour (P1), 4 hours (P2) |
| Rate Limits (traces/min) | 1,000 | 10,000 | Custom (unlimited) |
| API Rate Limits | 100 req/min | 1,000 req/min | Custom (unlimited) |
| Concurrent Users | 10 | 100 | Unlimited |
| Projects per Workspace | 5 | 50 | Unlimited |
| **Customization** |
| White-Labeling | - | - | ✓ |
| Custom Domain | - | ✓ | ✓ |
| Custom Branding | - | - | ✓ |
| Custom Integrations | - | Limited | ✓ |
| Custom Metrics | Limited | ✓ | ✓ |
| **Support & Services** |
| Community Support | ✓ | - | - |
| Email Support | - | ✓ | ✓ |
| Technical Account Manager | - | - | ✓ |
| Dedicated Slack Channel | - | - | ✓ |
| Onboarding & Training | Self-service | 2 hours | Customized (unlimited) |
| Architecture Review | - | - | ✓ |
| Custom Development | - | - | ✓ (consulting) |
| Quarterly Business Reviews | - | - | ✓ |
| **Pricing** |
| Base Cost | Free | Starting at $499/mo | Custom (volume-based) |
| Per-User Pricing | - | $49/user/mo | Volume discounts |
| Volume Pricing | - | Tiers | Custom contracts |
| Annual Commitment | - | Optional (15% discount) | Required |

---

## Enterprise-Exclusive Features

### 1. Enhanced Security Posture

**Enterprise Security Controls:**

```yaml
# Enterprise security configuration
enterprise_security:
  authentication:
    mfa:
      enforcement: "required"
      methods: ["totp", "sms", "hardware_token", "biometric"]
      session_timeout: 30  # minutes

    sso:
      providers:
        - type: "saml"
          idp: ["okta", "azure_ad", "google_workspace", "onelogin", "ping_identity"]
        - type: "oidc"
          providers: ["auth0", "okta", "azure_ad", "google"]

    password_policy:
      min_length: 16
      require_uppercase: true
      require_lowercase: true
      require_numbers: true
      require_special: true
      max_age_days: 90
      history: 12  # Cannot reuse last 12 passwords

  network_security:
    ip_allowlisting:
      enabled: true
      lists:
        - name: "corporate_network"
          cidrs: ["10.0.0.0/8", "172.16.0.0/12"]
        - name: "vpn_access"
          cidrs: ["203.0.113.0/24"]

    rate_limiting:
      enabled: true
      custom_limits: true
      per_user: 10000  # requests per minute
      per_org: 100000

  encryption:
    at_rest:
      algorithm: "aes_256_gcm"
      key_management: "customer_managed_keys"  # Enterprise can provide own KMS keys

    in_transit:
      tls_version: "1.3"
      mutual_tls: true  # Optional mTLS for service-to-service

  compliance:
    certifications:
      - "soc2_type_ii"
      - "iso_27001"
      - "hipaa"  # With BAA
      - "pci_dss"  # For payment data
```

### 2. Advanced Operational Features

**Enterprise Operational Tools:**

```python
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class EnterpriseFeatures:
    """Enterprise-specific feature configuration."""

    # Custom data retention
    retention_policies: dict = {
        "traces": "2y",  # 2 years instead of default 30 days
        "metrics": "3y",  # 3 years for compliance
        "audit_logs": "7y",  # Legal requirement
        "api_logs": "1y",
        "error_logs": "1y",
    }

    # Dedicated infrastructure
    infrastructure: dict = {
        "deployment_type": "dedicated_vpc",
        "region": "us-east-1",
        "backup_region": "us-west-2",
        "compute": {
            "collectors": "c5.2xlarge x 5",
            "api_servers": "c5.4xlarge x 3",
            "database": "i3en.6xlarge x 6 (ClickHouse cluster)",
        },
        "storage": {
            "hot_tier": "5 TB NVMe SSD",
            "warm_tier": "20 TB EBS",
            "cold_tier": "100 TB S3 Glacier",
        }
    }

    # White-labeling
    branding: dict = {
        "custom_domain": "observability.customer-company.com",
        "logo": "https://cdn.customer.com/logo.png",
        "color_scheme": {
            "primary": "#1a73e8",
            "secondary": "#34a853",
            "accent": "#fbbc04",
        },
        "email_templates": "custom",
        "ui_customization": True,
    }

    # SLA configuration
    sla: dict = {
        "uptime_target": "99.9%",
        "monitoring_interval": "30s",
        "incident_response": {
            "p1_response_time": "15min",
            "p2_response_time": "1h",
            "p3_response_time": "4h",
        },
        "financial_penalties": {
            "99.5_to_99.9": "10% monthly credit",
            "99.0_to_99.5": "25% monthly credit",
            "below_99.0": "50% monthly credit",
        }
    }

class EnterpriseManagementService:
    """Enterprise-specific management and administration."""

    async def provision_dedicated_infrastructure(
        self,
        org_id: str,
        config: EnterpriseFeatures
    ):
        """Provision dedicated infrastructure for enterprise customer.

        Creates isolated VPC, dedicated compute/storage resources,
        and configures region-specific deployment.
        """
        # Create dedicated VPC
        vpc_id = await self.create_vpc(
            region=config.infrastructure["region"],
            cidr_block="10.100.0.0/16",
            tags={"org_id": org_id, "type": "dedicated"}
        )

        # Deploy ClickHouse cluster
        cluster_id = await self.deploy_clickhouse_cluster(
            vpc_id=vpc_id,
            instance_type=config.infrastructure["compute"]["database"],
            replica_count=6,
            shard_count=3,
            backup_region=config.infrastructure["backup_region"]
        )

        # Deploy API servers with load balancer
        api_cluster = await self.deploy_api_cluster(
            vpc_id=vpc_id,
            instance_type=config.infrastructure["compute"]["api_servers"],
            count=3,
            load_balancer=True
        )

        # Configure custom domain with SSL
        await self.configure_custom_domain(
            domain=config.branding["custom_domain"],
            load_balancer=api_cluster.lb_dns,
            ssl_cert="auto_provision"
        )

        # Set up cross-region backup
        await self.configure_backup_replication(
            source_region=config.infrastructure["region"],
            backup_region=config.infrastructure["backup_region"],
            schedule="every_6_hours"
        )

        return {
            "vpc_id": vpc_id,
            "cluster_id": cluster_id,
            "api_endpoint": f"https://{config.branding['custom_domain']}",
            "status": "provisioned"
        }

    async def configure_white_labeling(
        self,
        org_id: str,
        branding: dict
    ):
        """Configure white-label branding for enterprise customer."""

        # Upload custom assets
        await self.upload_branding_assets(
            org_id=org_id,
            logo=branding["logo"],
            favicon=branding.get("favicon"),
            color_scheme=branding["color_scheme"]
        )

        # Customize email templates
        if branding.get("email_templates") == "custom":
            await self.deploy_custom_email_templates(
                org_id=org_id,
                templates_path=branding.get("templates_repository")
            )

        # Configure custom domain
        await self.setup_custom_domain(
            org_id=org_id,
            domain=branding["custom_domain"],
            ssl_validation="dns"
        )

        # Apply UI customization
        if branding.get("ui_customization"):
            await self.apply_ui_customization(
                org_id=org_id,
                theme=branding["color_scheme"],
                custom_css=branding.get("custom_css")
            )
```

---

## SSO and SAML Support

Enterprise customers can integrate a11i with their existing identity providers for centralized authentication and user provisioning.

**Supported Identity Providers:**

| Provider | Protocol | Features | Status |
|----------|----------|----------|--------|
| **Okta** | SAML 2.0, OIDC | SSO, SCIM provisioning, group mapping | ✓ Certified |
| **Azure Active Directory** | SAML 2.0, OIDC | SSO, SCIM provisioning, conditional access | ✓ Certified |
| **Google Workspace** | SAML 2.0, OIDC | SSO, group mapping | ✓ Certified |
| **OneLogin** | SAML 2.0 | SSO, SCIM provisioning | ✓ Supported |
| **Auth0** | OIDC | SSO, custom claims | ✓ Supported |
| **Ping Identity** | SAML 2.0 | SSO, federation | ✓ Supported |
| **Custom SAML 2.0** | SAML 2.0 | SSO | ✓ Supported |

**Key Benefits:**

- **Centralized Authentication**: Users authenticate through corporate identity provider
- **Just-in-Time Provisioning**: Automatic user account creation on first login
- **Group-Based Access**: Map identity provider groups to a11i roles
- **Automated De-provisioning**: Disable access when users leave organization
- **Audit Trail**: Complete authentication and authorization logging

**Example Configuration:**

See [SSO Integration Guide](./sso-integration.md) for detailed setup instructions.

---

## Advanced RBAC with Custom Roles

Enterprise customers can define unlimited custom roles with fine-grained permissions tailored to their organizational structure.

### Custom Role Creation

```python
from a11i.enterprise import CustomRoleManager
from a11i.rbac import Permission

# Initialize role manager
role_manager = CustomRoleManager(org_id="acme-corp")

# Create security auditor role
security_auditor = await role_manager.create_role(
    name="Security Auditor",
    description="Can view and export traces for security investigations",
    permissions=[
        Permission.TRACES_READ,
        Permission.TRACES_EXPORT,
        Permission.TRACES_SEARCH,
        Permission.DASHBOARDS_READ,
        Permission.ORG_AUDIT_LOGS,
        Permission.METRICS_VIEW,
    ],
    scope="organization",
    max_export_size="unlimited",  # Enterprise feature
    data_retention_override="7y"   # Enterprise feature
)

# Create compliance manager role
compliance_manager = await role_manager.create_role(
    name="Compliance Manager",
    description="Manages compliance settings and generates reports",
    permissions=[
        Permission.ORG_AUDIT_LOGS,
        Permission.COMPLIANCE_REPORTS,
        Permission.RETENTION_POLICY_MANAGE,
        Permission.DATA_RESIDENCY_CONFIGURE,
        Permission.GDPR_REQUESTS_MANAGE,
    ],
    scope="organization",
    generate_compliance_reports=True
)

# Create development team lead role
dev_team_lead = await role_manager.create_role(
    name="Development Team Lead",
    description="Manages team workspace and projects",
    permissions=[
        Permission.WORKSPACE_SETTINGS,
        Permission.PROJECT_CREATE,
        Permission.MEMBERS_INVITE,
        Permission.MEMBERS_ROLE_ASSIGN,
        Permission.TRACES_READ,
        Permission.TRACES_WRITE,
        Permission.DASHBOARDS_WRITE,
        Permission.ALERTS_WRITE,
        Permission.API_KEYS_MANAGE,
    ],
    scope="workspace"
)
```

### Permission Categories

**Enterprise-Exclusive Permissions:**

| Permission Category | Permissions | Description |
|---------------------|-------------|-------------|
| **Compliance** | `COMPLIANCE_REPORTS`, `GDPR_REQUESTS_MANAGE`, `RETENTION_POLICY_MANAGE` | Compliance management and reporting |
| **Advanced Audit** | `AUDIT_LOGS_EXPORT`, `SIEM_INTEGRATION_CONFIGURE` | Enhanced audit capabilities |
| **Infrastructure** | `DEDICATED_VPC_MANAGE`, `DATA_RESIDENCY_CONFIGURE` | Infrastructure management |
| **Customization** | `WHITE_LABEL_CONFIGURE`, `CUSTOM_DOMAIN_MANAGE` | Branding and customization |
| **Advanced Data** | `UNLIMITED_EXPORT`, `CUSTOM_RETENTION_CONFIGURE` | Advanced data management |

---

## Comprehensive Audit Logging

Enterprise customers receive enhanced audit logging with extended retention, SIEM integration, and compliance reporting capabilities.

**Enterprise Audit Features:**

```yaml
# Enterprise audit configuration
enterprise_audit:
  retention: "7y"  # 7-year retention for compliance

  event_types:
    # All standard events plus enterprise-specific
    - authentication_events
    - authorization_decisions
    - data_access_events
    - administrative_actions
    - compliance_actions
    - infrastructure_changes
    - security_incidents
    - anomaly_detections

  siem_integration:
    enabled: true
    providers:
      - splunk
      - datadog
      - sumo_logic
      - elastic_siem
      - azure_sentinel
    format: "cef"  # Common Event Format
    streaming: true
    batch_interval: "30s"

  compliance_reporting:
    enabled: true
    schedules:
      - type: "soc2"
        frequency: "quarterly"
        recipients: ["compliance@customer.com"]
      - type: "gdpr_article_30"
        frequency: "annual"
        recipients: ["dpo@customer.com"]
      - type: "hipaa_access_report"
        frequency: "monthly"
        recipients: ["privacy-officer@customer.com"]

  alerting:
    enabled: true
    rules:
      - name: "Unusual data export volume"
        condition: "export_size > 10GB"
        notification: "security@customer.com"
      - name: "Failed login attempts"
        condition: "failed_logins > 5 in 10min"
        notification: "security@customer.com"
      - name: "Privileged role assignment"
        condition: "role IN ['org_admin', 'billing_admin']"
        notification: "iam@customer.com"
```

See [Audit Logging Guide](./audit-logging.md) for complete documentation.

---

## White-Labeling Options

Enterprise customers can fully customize the a11i platform with their own branding, creating a seamless experience for end users.

### White-Label Components

**1. Visual Branding:**
- Custom logo and favicon
- Custom color scheme and theme
- Custom fonts and typography
- Custom CSS for advanced styling

**2. Domain and URLs:**
- Custom domain (e.g., `observability.company.com`)
- Custom email domain for notifications
- Custom API endpoints
- Custom documentation URLs

**3. Email Templates:**
- Branded email templates
- Custom sender name and address
- Localized email content
- Custom email footer

**4. Documentation:**
- Custom documentation portal
- White-labeled API documentation
- Custom help center URL
- Embedded support chat with branding

### Implementation Example

```yaml
# white-label-config.yaml
white_label:
  organization: "acme-corp"

  branding:
    company_name: "Acme Corporation"
    logo_url: "https://cdn.acme.com/logo.svg"
    favicon_url: "https://cdn.acme.com/favicon.ico"

    theme:
      primary_color: "#0052CC"
      secondary_color: "#00B8D9"
      accent_color: "#FFAB00"
      background_color: "#F4F5F7"
      text_color: "#172B4D"

    fonts:
      heading: "Inter, sans-serif"
      body: "Inter, sans-serif"
      monospace: "Fira Code, monospace"

  domain:
    primary: "ai-observability.acme.com"
    api: "api.ai-observability.acme.com"
    docs: "docs.ai-observability.acme.com"

  email:
    from_name: "Acme AI Observability"
    from_address: "ai-observability@acme.com"
    reply_to: "support@acme.com"
    templates: "custom"  # Use custom email templates

  support:
    help_center_url: "https://help.acme.com/ai-observability"
    contact_email: "ai-support@acme.com"
    chat_enabled: true
    chat_widget_color: "#0052CC"
```

---

## SLA Commitments

Enterprise customers receive guaranteed uptime with financial penalties for non-compliance.

### Uptime SLA: 99.9%

**Monthly Uptime Guarantee:**

| Uptime Percentage | Monthly Downtime | Service Credit |
|-------------------|------------------|----------------|
| **99.9% - 100%** | < 43.2 minutes | 0% |
| **99.5% - 99.9%** | 43.2 - 216 minutes | 10% |
| **99.0% - 99.5%** | 216 - 432 minutes | 25% |
| **< 99.0%** | > 432 minutes | 50% |

**SLA Exclusions:**
- Scheduled maintenance (with 7-day notice)
- Customer-caused outages (misconfiguration, abuse)
- Force majeure events
- Third-party service failures (AWS, GCP outages)

### Performance SLA

**Response Time Targets:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| **API Response Time (p95)** | < 200ms | Per-endpoint monitoring |
| **Query Response Time (p95)** | < 2 seconds | Dashboard query execution |
| **Trace Ingestion Latency (p99)** | < 10ms | From API to storage |
| **Alert Delivery Time** | < 60 seconds | From threshold breach to notification |

### Support SLA

**Incident Response Times:**

| Priority | Definition | Response Time | Resolution Target |
|----------|------------|---------------|-------------------|
| **P1 (Critical)** | Complete service outage | 15 minutes | 4 hours |
| **P2 (High)** | Significant feature degradation | 1 hour | 8 hours |
| **P3 (Medium)** | Minor feature issue | 4 hours | 48 hours |
| **P4 (Low)** | General questions, feature requests | 24 hours | Best effort |

---

## Priority Support Tiers

Enterprise customers receive dedicated support resources and proactive engagement.

### Enterprise Support Components

**1. Technical Account Manager (TAM):**
- Named technical contact
- Quarterly business reviews
- Architecture guidance
- Roadmap input and early access to features
- Escalation point for critical issues

**2. Dedicated Slack Channel:**
- Direct communication with engineering team
- Real-time incident response
- Feature discussions and feedback
- Integration support

**3. Onboarding and Training:**
- Custom onboarding plan
- Administrator training (up to 40 hours)
- Developer training workshops
- Best practices review
- Architecture design session

**4. Proactive Monitoring:**
- Dedicated monitoring by a11i SRE team
- Anomaly detection and alerting
- Capacity planning recommendations
- Performance optimization suggestions

### Support Comparison

| Support Feature | Open Source | Professional | Enterprise |
|-----------------|-------------|--------------|------------|
| **Response Times** |
| P1 (Critical) | Community | 8 hours | 15 minutes |
| P2 (High) | Community | 24 hours | 1 hour |
| P3 (Medium) | Community | 48 hours | 4 hours |
| **Support Channels** |
| Community Forum | ✓ | ✓ | ✓ |
| Email Support | - | ✓ | ✓ |
| Dedicated Slack | - | - | ✓ |
| Phone Support | - | - | ✓ (P1/P2 only) |
| **Resources** |
| Documentation | ✓ | ✓ | ✓ + custom docs |
| Video Tutorials | ✓ | ✓ | ✓ |
| Onboarding | Self-service | 2 hours | Custom (unlimited) |
| Technical Account Manager | - | - | ✓ |
| Architecture Review | - | - | ✓ (quarterly) |
| Quarterly Business Review | - | - | ✓ |
| **Development** |
| Feature Requests | Community voting | Prioritized | Roadmap influence |
| Custom Integrations | DIY | Guidance | Development support |
| Bug Fixes | Community | Standard | Priority |

---

## Data Retention Customization

Enterprise customers can configure custom data retention policies to meet compliance and operational requirements.

### Configurable Retention Periods

```python
from a11i.enterprise import RetentionPolicyManager
from datetime import timedelta

# Initialize retention manager
retention_mgr = RetentionPolicyManager(org_id="acme-corp")

# Configure custom retention for regulatory compliance
await retention_mgr.set_policy(
    data_type="traces",
    retention_period=timedelta(days=730),  # 2 years
    reason="Financial services compliance requirement"
)

await retention_mgr.set_policy(
    data_type="metrics",
    retention_period=timedelta(days=1095),  # 3 years
    reason="SOX compliance"
)

await retention_mgr.set_policy(
    data_type="audit_logs",
    retention_period=timedelta(days=2555),  # 7 years
    reason="GDPR Article 30 requirement"
)

# Configure lifecycle transitions
await retention_mgr.configure_lifecycle(
    data_type="traces",
    stages=[
        {"age": "0-30d", "storage_class": "hot", "compression": False},
        {"age": "30-90d", "storage_class": "warm", "compression": True},
        {"age": "90-730d", "storage_class": "cold", "compression": True},
    ]
)
```

### Retention Policy Options

| Data Type | Open Source | Professional | Enterprise |
|-----------|-------------|--------------|------------|
| **Traces** | 7 days | 30 days | Up to 7 years |
| **Metrics** | 30 days | 90 days | Up to 7 years |
| **Audit Logs** | 90 days | 365 days | Up to 10 years |
| **Exports** | Not retained | 7 days | Custom |
| **Compliance Logs** | N/A | N/A | Indefinite (with archival) |

---

## Dedicated Infrastructure Options

Enterprise customers can choose from multiple infrastructure deployment models to meet security, compliance, and performance requirements.

### Deployment Models

**1. Dedicated VPC (Shared Region):**
- Isolated VPC within a11i's AWS/GCP account
- Dedicated compute and storage resources
- Shared region with other customers (isolated network)
- Best for: Cost-effective isolation with compliance requirements

**2. Single-Tenant Cluster:**
- Completely isolated infrastructure
- No shared resources with other customers
- Dedicated region or availability zone
- Best for: High security requirements, regulated industries

**3. Customer VPC Deployment:**
- Deployed within customer's AWS/GCP account
- Customer maintains infrastructure ownership
- a11i provides software and management
- Best for: Maximum control and data sovereignty

**4. On-Premises Deployment:**
- Deployed in customer's data center
- Air-gapped or VPN-connected options
- Customer-managed infrastructure
- Best for: Highly regulated environments, government

### Infrastructure Sizing

**Small Enterprise (< 1M traces/month):**
```yaml
infrastructure:
  collectors: "3x c5.xlarge (AWS) or n2-standard-4 (GCP)"
  api_servers: "2x c5.2xlarge or n2-standard-8"
  database: "3x i3en.2xlarge or n2-highmem-8 (ClickHouse)"
  storage: "1 TB hot + 5 TB warm"
  estimated_cost: "$5,000/month"
```

**Medium Enterprise (1M - 10M traces/month):**
```yaml
infrastructure:
  collectors: "5x c5.2xlarge or n2-standard-8"
  api_servers: "3x c5.4xlarge or n2-standard-16"
  database: "6x i3en.6xlarge or n2-highmem-32 (ClickHouse)"
  storage: "5 TB hot + 20 TB warm + 50 TB cold"
  estimated_cost: "$15,000/month"
```

**Large Enterprise (> 10M traces/month):**
```yaml
infrastructure:
  collectors: "10x c5.4xlarge or n2-standard-16 (auto-scaling)"
  api_servers: "5x c5.9xlarge or n2-standard-32"
  database: "12x i3en.12xlarge or n2-highmem-64 (ClickHouse)"
  storage: "20 TB hot + 100 TB warm + 500 TB cold"
  estimated_cost: "$50,000+/month (custom pricing)"
```

---

## Enterprise Pricing Model

Enterprise pricing is customized based on usage, features, and infrastructure requirements.

### Pricing Components

**1. Base Platform Fee:**
- Includes core platform features, SSO, RBAC, audit logging
- Starts at $5,000/month (annual commitment)
- Scales with organization size and features

**2. Usage-Based Pricing:**
- Traces ingested: $0.50 per 1M traces
- Storage: $100 per TB-month (hot), $20 per TB-month (cold)
- API calls: Included up to 10M/month, then $1 per 1M

**3. Infrastructure Costs:**
- Dedicated VPC: Included in base fee
- Single-Tenant: +$10,000/month
- Customer VPC: +$15,000/month
- On-Premises: Custom (software license model)

**4. Support & Services:**
- Technical Account Manager: Included
- Additional training: $2,500 per day
- Custom development: $250/hour
- Architecture consulting: $350/hour

### Sample Pricing Scenarios

**Small Enterprise (50 developers, 1M traces/month):**
```
Base Platform Fee:        $5,000/month
Traces (1M @ $0.50):        $500/month
Storage (2TB hot):          $200/month
Support (TAM):           Included
------------------------------------------
Total:                   $5,700/month
Annual (15% discount):  $58,140/year
```

**Medium Enterprise (200 developers, 5M traces/month):**
```
Base Platform Fee:       $12,000/month
Traces (5M @ $0.50):      $2,500/month
Storage (10TB hot):       $1,000/month
Dedicated VPC:            Included
Support (TAM):            Included
------------------------------------------
Total:                   $15,500/month
Annual (15% discount):  $158,100/year
```

**Large Enterprise (1000+ developers, 20M traces/month):**
```
Base Platform Fee:       $25,000/month
Traces (20M @ $0.40):     $8,000/month  (volume discount)
Storage (50TB):           $3,500/month
Single-Tenant:           $10,000/month
Support (TAM + QBR):      Included
------------------------------------------
Total:                   $46,500/month
Annual (20% discount):  $446,400/year
```

### Volume Discounts

| Annual Commit | Discount |
|---------------|----------|
| $100K - $250K | 15% |
| $250K - $500K | 20% |
| $500K - $1M | 25% |
| > $1M | Custom (up to 35%) |

---

## Key Takeaways

> **Enterprise Feature Summary**
>
> **Security & Compliance:**
> - SSO/SAML integration with major identity providers (Okta, Azure AD, Google Workspace)
> - Custom RBAC roles with unlimited fine-grained permissions
> - 7-year audit log retention for regulatory compliance
> - HIPAA, SOC 2, GDPR compliance certifications
>
> **Infrastructure & Performance:**
> - 99.9% uptime SLA with financial penalties
> - Dedicated VPC, single-tenant, or on-premises deployment options
> - Custom data retention up to 7 years
> - Regional data residency guarantees
>
> **Support & Services:**
> - Named Technical Account Manager
> - 15-minute P1 incident response
> - Dedicated Slack channel with engineering team
> - Quarterly business reviews and architecture guidance
> - Custom onboarding and unlimited training
>
> **Customization:**
> - Complete white-labeling with custom domain and branding
> - SIEM integration for audit logs
> - Custom integrations and development support
> - Early access to features and roadmap input
>
> **When to Choose Enterprise:**
> - Regulated industries (healthcare, finance, government)
> - Large engineering organizations (500+ developers)
> - Multi-national companies requiring data residency
> - High-volume production workloads (>10M traces/month)
> - Organizations with custom integration requirements

**Next Steps:**
1. [Review SSO Integration Guide](./sso-integration.md) for identity provider setup
2. [Review Audit Logging Guide](./audit-logging.md) for compliance requirements
3. Contact sales@a11i.dev for custom pricing and enterprise evaluation
4. Schedule architecture review with enterprise solutions team

---

**Related Documentation:**
- [SSO Integration Guide](./sso-integration.md)
- [Audit Logging Guide](./audit-logging.md)
- [RBAC and Authentication](../05-security-compliance/rbac-auth.md)
- [Compliance Framework](../05-security-compliance/compliance-framework.md)
- [Deployment Modes](../02-architecture/deployment-modes.md)

---

*Document Status: Draft | Last Updated: 2025-11-26 | Maintained by: Enterprise Sales & Solutions Team*
