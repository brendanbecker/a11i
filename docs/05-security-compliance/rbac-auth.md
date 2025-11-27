---
title: "RBAC and Authentication"
section: "05-security-compliance"
subsection: "rbac-auth"
version: "1.0"
last_updated: "2025-11-26"
status: "active"
owner: "Security Team"
applies_to: ["all_environments", "enterprise", "team", "developer"]
related_docs:
  - "../04-deployment-operations/api-gateway.md"
  - "./data-privacy.md"
  - "./audit-logging.md"
tags: ["rbac", "authentication", "authorization", "sso", "permissions", "security"]
---

# RBAC and Authentication

## Table of Contents

1. [Overview](#overview)
2. [Three-Tier Hierarchy Model](#three-tier-hierarchy-model)
3. [Role Definitions](#role-definitions)
4. [Permission Matrix](#permission-matrix)
5. [Authorization Implementation](#authorization-implementation)
6. [SSO Integration](#sso-integration)
7. [Custom Roles (Enterprise)](#custom-roles-enterprise)
8. [Security Best Practices](#security-best-practices)
9. [Key Takeaways](#key-takeaways)
10. [References](#references)

---

## Overview

The a11i platform implements a comprehensive Role-Based Access Control (RBAC) system that provides fine-grained authorization across a three-tier organizational hierarchy. This document describes the authentication and authorization mechanisms, role definitions, permission models, and enterprise SSO integration capabilities.

### Key Features

- **Three-Tier Hierarchy**: Organization → Workspace → Project structure for flexible access control
- **Granular Permissions**: Fine-grained permission system with clear role boundaries
- **SSO Integration**: Enterprise-grade SAML 2.0 and OIDC/OAuth support
- **Custom Roles**: Enterprise customers can define custom roles with specific permissions
- **Audit Trail**: All authorization decisions are logged for compliance and security analysis

### Authentication Methods

| Method | Use Case | Supported Plans |
|--------|----------|-----------------|
| **Email/Password** | Individual users, small teams | All plans |
| **OAuth 2.0** | Third-party integrations (GitHub, Google) | Team, Enterprise |
| **SAML 2.0** | Enterprise SSO with identity providers | Enterprise only |
| **OIDC** | Modern SSO with providers like Okta, Auth0 | Enterprise only |
| **API Keys** | Service-to-service authentication | All plans |
| **Service Accounts** | Automation and CI/CD pipelines | Team, Enterprise |

---

## Three-Tier Hierarchy Model

The a11i RBAC system is built on a three-tier organizational hierarchy that provides clear separation of concerns and scalable access management.

### Hierarchy Structure

```
Organization (Tenant)
├── Workspace A
│   ├── Project 1
│   │   └── Members & Roles
│   │       ├── Alice (Project Admin)
│   │       ├── Bob (Project Editor)
│   │       └── Charlie (Project Viewer)
│   └── Project 2
│       └── Members & Roles
│           ├── Diana (Project Admin)
│           └── Eve (Project Analyst)
└── Workspace B
    └── Project 3
        └── Members & Roles
            ├── Frank (Project Admin)
            └── Grace (Project Editor)
```

### Hierarchy Levels Explained

#### Organization Level (Tenant)

The top-level entity representing a company or organization. All billing, user management, and compliance settings are configured at this level.

**Characteristics:**
- Single billing account
- Centralized user directory
- Organization-wide audit logs
- Global security policies
- SSO configuration (Enterprise)

**Key Use Cases:**
- Managing company-wide user access
- Controlling billing and subscription
- Enforcing compliance policies
- Configuring SSO and identity providers

#### Workspace Level

Logical grouping of related projects, typically aligned with teams, departments, or business units.

**Characteristics:**
- Isolated data boundaries
- Workspace-specific settings
- Shared dashboards and alerts
- Resource quotas and limits

**Key Use Cases:**
- Engineering team workspaces
- Department-specific observability
- Cross-project analytics
- Team collaboration spaces

#### Project Level

Individual observability projects containing traces, spans, metrics, and dashboards for specific applications or services.

**Characteristics:**
- Project-specific trace data
- Dedicated dashboards and alerts
- Project-level API keys
- Sampling and retention policies

**Key Use Cases:**
- Microservice observability
- Application-specific monitoring
- Feature team ownership
- Development/staging/production separation

### Permission Inheritance

Permissions flow down the hierarchy:
- **Organization Admin** has admin rights to all workspaces and projects
- **Workspace Admin** has admin rights to all projects within the workspace
- **Project roles** are isolated to specific projects

---

## Role Definitions

The a11i platform provides predefined roles at each hierarchy level, each with specific responsibilities and permissions.

### Organization-Level Roles

| Role | Description | Typical Users |
|------|-------------|---------------|
| **Org Admin** | Full control over organization, including billing, users, and all resources | CTO, VP Engineering, IT Admins |
| **Org Billing Admin** | Manages billing, subscriptions, and usage monitoring | Finance team, Procurement |
| **Org Member** | Base access to assigned projects with no org-level permissions | All employees |

**Organization Admin Capabilities:**
- Create and delete workspaces
- Manage organization members and roles
- Configure SSO and authentication settings
- Access billing and subscription management
- View organization-wide audit logs
- Set global security and compliance policies

**Organization Billing Admin Capabilities:**
- View and manage billing information
- Monitor usage and costs across all projects
- Set budget alerts and spending limits
- Download invoices and usage reports
- No access to observability data or project settings

### Workspace-Level Roles

| Role | Description | Typical Users |
|------|-------------|---------------|
| **Workspace Admin** | Full workspace control including project creation and member management | Team Leads, Engineering Managers |
| **Workspace Editor** | Read and write access to all projects in workspace | Senior Engineers, DevOps |
| **Workspace Viewer** | Read-only access to all projects in workspace | Stakeholders, Product Managers |

**Workspace Admin Capabilities:**
- Create and delete projects within workspace
- Manage workspace members and roles
- Configure workspace-level settings
- Create shared dashboards and alerts
- Set workspace retention policies

**Workspace Editor Capabilities:**
- Create and modify traces, spans, and metrics
- Create and edit dashboards and alerts
- Export data from all projects
- No access to workspace settings or member management

**Workspace Viewer Capabilities:**
- View traces, spans, and metrics
- View dashboards and alerts
- No write or export capabilities

### Project-Level Roles

| Role | Description | Typical Users |
|------|-------------|---------------|
| **Project Admin** | Full project control including member management | Project Owners, Tech Leads |
| **Project Editor** | Create and edit all project resources | Software Engineers, SREs |
| **Project Viewer** | Read-only access to project data | QA, Support, Stakeholders |
| **Project Analyst** | View access with export capabilities | Data Analysts, Compliance |

**Project Admin Capabilities:**
- Manage project members and roles
- Configure project settings (retention, sampling)
- Create and manage API keys
- Delete project (if workspace admin)
- All editor capabilities

**Project Editor Capabilities:**
- Send traces and spans to project
- Create and modify dashboards
- Create and manage alerts
- Edit dashboard configurations
- No member management or settings access

**Project Viewer Capabilities:**
- View traces and spans
- View dashboards and alerts
- Search and filter observability data
- No write or configuration access

**Project Analyst Capabilities:**
- All viewer capabilities
- Export traces and metrics
- Generate reports
- Access to data export APIs

---

## Permission Matrix

The permission system is implemented using a granular permission enumeration that maps to specific capabilities within the platform.

### Permission Enumeration

```python
from enum import Enum, auto
from typing import Set

class Permission(Enum):
    """Granular permissions for a11i platform."""

    # Trace permissions
    TRACES_READ = auto()
    TRACES_WRITE = auto()
    TRACES_DELETE = auto()
    TRACES_EXPORT = auto()

    # Dashboard permissions
    DASHBOARDS_READ = auto()
    DASHBOARDS_WRITE = auto()
    DASHBOARDS_DELETE = auto()

    # Alert permissions
    ALERTS_READ = auto()
    ALERTS_WRITE = auto()
    ALERTS_DELETE = auto()

    # Member management
    MEMBERS_READ = auto()
    MEMBERS_INVITE = auto()
    MEMBERS_REMOVE = auto()
    MEMBERS_ROLE_ASSIGN = auto()

    # Project management
    PROJECT_SETTINGS = auto()
    PROJECT_DELETE = auto()
    PROJECT_API_KEYS = auto()

    # Workspace management
    WORKSPACE_CREATE_PROJECT = auto()
    WORKSPACE_SETTINGS = auto()
    WORKSPACE_DELETE = auto()

    # Organization management
    ORG_BILLING = auto()
    ORG_SETTINGS = auto()
    ORG_CREATE_WORKSPACE = auto()
    ORG_SSO_CONFIG = auto()
    ORG_AUDIT_LOGS = auto()
```

### Role-Permission Mappings

```python
ROLE_PERMISSIONS: dict[str, Set[Permission]] = {
    # Organization-level roles
    "org_admin": {p for p in Permission},  # All permissions

    "org_billing_admin": {
        Permission.ORG_BILLING,
        Permission.TRACES_READ,
        Permission.DASHBOARDS_READ,
    },

    "org_member": set(),  # No default permissions, must be assigned to projects

    # Workspace-level roles
    "workspace_admin": {
        Permission.TRACES_READ,
        Permission.TRACES_WRITE,
        Permission.TRACES_DELETE,
        Permission.TRACES_EXPORT,
        Permission.DASHBOARDS_READ,
        Permission.DASHBOARDS_WRITE,
        Permission.DASHBOARDS_DELETE,
        Permission.ALERTS_READ,
        Permission.ALERTS_WRITE,
        Permission.ALERTS_DELETE,
        Permission.MEMBERS_READ,
        Permission.MEMBERS_INVITE,
        Permission.MEMBERS_REMOVE,
        Permission.MEMBERS_ROLE_ASSIGN,
        Permission.PROJECT_SETTINGS,
        Permission.PROJECT_API_KEYS,
        Permission.WORKSPACE_CREATE_PROJECT,
        Permission.WORKSPACE_SETTINGS,
    },

    "workspace_editor": {
        Permission.TRACES_READ,
        Permission.TRACES_WRITE,
        Permission.TRACES_EXPORT,
        Permission.DASHBOARDS_READ,
        Permission.DASHBOARDS_WRITE,
        Permission.ALERTS_READ,
        Permission.ALERTS_WRITE,
        Permission.MEMBERS_READ,
    },

    "workspace_viewer": {
        Permission.TRACES_READ,
        Permission.DASHBOARDS_READ,
        Permission.ALERTS_READ,
        Permission.MEMBERS_READ,
    },

    # Project-level roles
    "project_admin": {
        Permission.TRACES_READ,
        Permission.TRACES_WRITE,
        Permission.TRACES_DELETE,
        Permission.TRACES_EXPORT,
        Permission.DASHBOARDS_READ,
        Permission.DASHBOARDS_WRITE,
        Permission.DASHBOARDS_DELETE,
        Permission.ALERTS_READ,
        Permission.ALERTS_WRITE,
        Permission.ALERTS_DELETE,
        Permission.MEMBERS_READ,
        Permission.MEMBERS_INVITE,
        Permission.MEMBERS_REMOVE,
        Permission.MEMBERS_ROLE_ASSIGN,
        Permission.PROJECT_SETTINGS,
        Permission.PROJECT_API_KEYS,
    },

    "project_editor": {
        Permission.TRACES_READ,
        Permission.TRACES_WRITE,
        Permission.DASHBOARDS_READ,
        Permission.DASHBOARDS_WRITE,
        Permission.ALERTS_READ,
        Permission.ALERTS_WRITE,
        Permission.MEMBERS_READ,
    },

    "project_viewer": {
        Permission.TRACES_READ,
        Permission.DASHBOARDS_READ,
        Permission.ALERTS_READ,
        Permission.MEMBERS_READ,
    },

    "project_analyst": {
        Permission.TRACES_READ,
        Permission.TRACES_EXPORT,
        Permission.DASHBOARDS_READ,
        Permission.ALERTS_READ,
        Permission.MEMBERS_READ,
    },
}
```

### Capability Comparison Table

| Capability | Org Admin | Workspace Admin | Project Admin | Project Editor | Project Viewer | Project Analyst |
|------------|-----------|-----------------|---------------|----------------|----------------|-----------------|
| **Traces** |
| View traces | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Send traces | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Delete traces | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Export traces | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| **Dashboards** |
| View dashboards | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Create dashboards | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Delete dashboards | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| **Alerts** |
| View alerts | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Create alerts | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Delete alerts | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| **Members** |
| View members | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Invite members | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Remove members | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Assign roles | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| **Settings** |
| Project settings | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Workspace settings | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Organization settings | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Billing | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| SSO configuration | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## Authorization Implementation

The a11i platform implements authorization using a centralized service that evaluates user permissions based on their roles and the resource hierarchy.

### Authorization Service

```python
from functools import wraps
from typing import Callable, Set, Optional
from datetime import datetime

class AuthorizationService:
    """RBAC authorization service for a11i platform."""

    def __init__(self, db):
        """Initialize authorization service with database connection."""
        self.db = db
        self.permission_cache = {}  # In-memory cache for performance

    async def get_user_permissions(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
    ) -> Set[Permission]:
        """
        Get all permissions a user has for a specific resource.

        Args:
            user_id: Unique identifier for the user
            resource_type: Type of resource (organization, workspace, project)
            resource_id: Unique identifier for the resource

        Returns:
            Set of Permission enums the user has for this resource

        Example:
            >>> permissions = await auth.get_user_permissions(
            ...     user_id="user-123",
            ...     resource_type="project",
            ...     resource_id="proj-abc"
            ... )
            >>> Permission.TRACES_WRITE in permissions
            True
        """
        # Check cache first
        cache_key = f"{user_id}:{resource_type}:{resource_id}"
        if cache_key in self.permission_cache:
            cached = self.permission_cache[cache_key]
            if (datetime.utcnow() - cached["timestamp"]).seconds < 300:  # 5 min TTL
                return cached["permissions"]

        # Get resource hierarchy for permission inheritance
        hierarchy = await self._get_resource_hierarchy(resource_type, resource_id)

        # Query all roles for user across the hierarchy
        roles = await self.db.query("""
            SELECT role, resource_type, resource_id
            FROM user_roles
            WHERE user_id = {user_id:String}
              AND (
                (resource_type = 'organization' AND resource_id = {org_id:String})
                OR (resource_type = 'workspace' AND resource_id = {workspace_id:String})
                OR (resource_type = 'project' AND resource_id = {project_id:String})
              )
        """,
            user_id=user_id,
            org_id=hierarchy["org_id"],
            workspace_id=hierarchy.get("workspace_id", ""),
            project_id=hierarchy.get("project_id", "")
        )

        # Aggregate permissions from all roles
        permissions = set()
        for role_row in roles:
            role = role_row["role"]
            role_perms = ROLE_PERMISSIONS.get(role, set())
            permissions.update(role_perms)

        # Cache the result
        self.permission_cache[cache_key] = {
            "permissions": permissions,
            "timestamp": datetime.utcnow()
        }

        return permissions

    async def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """
        Check if user has a specific permission for a resource.

        Args:
            user_id: Unique identifier for the user
            permission: Permission to check
            resource_type: Type of resource (organization, workspace, project)
            resource_id: Unique identifier for the resource

        Returns:
            True if user has the permission, False otherwise

        Example:
            >>> has_access = await auth.check_permission(
            ...     user_id="user-123",
            ...     permission=Permission.TRACES_WRITE,
            ...     resource_type="project",
            ...     resource_id="proj-abc"
            ... )
            >>> if has_access:
            ...     await write_traces(project_id)
        """
        permissions = await self.get_user_permissions(
            user_id, resource_type, resource_id
        )

        # Log authorization decision for audit trail
        await self._log_authorization_check(
            user_id=user_id,
            permission=permission,
            resource_type=resource_type,
            resource_id=resource_id,
            granted=permission in permissions
        )

        return permission in permissions

    async def check_any_permission(
        self,
        user_id: str,
        permissions: list[Permission],
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """
        Check if user has any of the specified permissions.

        Useful for endpoints that accept multiple permission levels.
        """
        user_perms = await self.get_user_permissions(
            user_id, resource_type, resource_id
        )
        return any(p in user_perms for p in permissions)

    async def _get_resource_hierarchy(
        self,
        resource_type: str,
        resource_id: str
    ) -> dict:
        """Get the full hierarchy for a resource (org, workspace, project IDs)."""
        if resource_type == "organization":
            return {"org_id": resource_id}

        elif resource_type == "workspace":
            workspace = await self.db.query_one(
                "SELECT org_id FROM workspaces WHERE workspace_id = {id:String}",
                id=resource_id
            )
            return {
                "org_id": workspace["org_id"],
                "workspace_id": resource_id
            }

        elif resource_type == "project":
            project = await self.db.query_one("""
                SELECT p.workspace_id, w.org_id
                FROM projects p
                JOIN workspaces w ON p.workspace_id = w.workspace_id
                WHERE p.project_id = {id:String}
            """, id=resource_id)
            return {
                "org_id": project["org_id"],
                "workspace_id": project["workspace_id"],
                "project_id": resource_id
            }

        raise ValueError(f"Unknown resource type: {resource_type}")

    async def _log_authorization_check(
        self,
        user_id: str,
        permission: Permission,
        resource_type: str,
        resource_id: str,
        granted: bool
    ):
        """Log authorization decision for audit trail."""
        await self.db.insert("authorization_logs", {
            "user_id": user_id,
            "permission": permission.name,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "granted": granted,
            "timestamp": datetime.utcnow()
        })
```

### Permission Decorator

```python
from functools import wraps
from typing import Callable
from fastapi import Request, HTTPException
from starlette.status import HTTP_403_FORBIDDEN

class PermissionDeniedError(Exception):
    """Raised when user lacks required permission."""
    pass

def require_permission(permission: Permission, resource_type: str = None):
    """
    Decorator to require a specific permission for an endpoint.

    Args:
        permission: Permission required for access
        resource_type: Type of resource (if not inferred from request)

    Usage:
        @require_permission(Permission.TRACES_WRITE, resource_type="project")
        async def send_traces(project_id: str, traces: list):
            # User is authorized
            return await trace_repository.insert_traces(project_id, traces)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request and user from function arguments
            request = next((arg for arg in args if isinstance(arg, Request)), None)
            if not request:
                raise ValueError("Request object not found in function arguments")

            # Get current user from request state (set by authentication middleware)
            user_id = request.state.user_id

            # Extract resource information from request path or body
            res_type, res_id = extract_resource_from_request(request, resource_type)

            # Check permission
            auth_service = get_auth_service()
            if not await auth_service.check_permission(
                user_id, permission, res_type, res_id
            ):
                raise HTTPException(
                    status_code=HTTP_403_FORBIDDEN,
                    detail=f"Missing required permission: {permission.name}"
                )

            # Execute the protected function
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_any_permission(*permissions: Permission, resource_type: str = None):
    """
    Decorator to require any of the specified permissions.

    Usage:
        @require_any_permission(
            Permission.TRACES_READ,
            Permission.TRACES_EXPORT,
            resource_type="project"
        )
        async def get_traces(project_id: str):
            # User has either read or export permission
            return await trace_repository.get_traces(project_id)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = next((arg for arg in args if isinstance(arg, Request)), None)
            if not request:
                raise ValueError("Request object not found")

            user_id = request.state.user_id
            res_type, res_id = extract_resource_from_request(request, resource_type)

            auth_service = get_auth_service()
            if not await auth_service.check_any_permission(
                user_id, list(permissions), res_type, res_id
            ):
                raise HTTPException(
                    status_code=HTTP_403_FORBIDDEN,
                    detail=f"Missing required permissions: {[p.name for p in permissions]}"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### Usage Examples

```python
from fastapi import FastAPI, Request
from typing import List

app = FastAPI()

# Example 1: Endpoint requiring trace write permission
@app.post("/api/projects/{project_id}/traces")
@require_permission(Permission.TRACES_WRITE, resource_type="project")
async def send_traces(
    request: Request,
    project_id: str,
    traces: List[dict]
):
    """Send traces to a project. Requires TRACES_WRITE permission."""
    return await trace_repository.insert_traces(project_id, traces)

# Example 2: Endpoint requiring dashboard write permission
@app.post("/api/projects/{project_id}/dashboards")
@require_permission(Permission.DASHBOARDS_WRITE, resource_type="project")
async def create_dashboard(
    request: Request,
    project_id: str,
    dashboard: dict
):
    """Create a dashboard. Requires DASHBOARDS_WRITE permission."""
    return await dashboard_repository.create(project_id, dashboard)

# Example 3: Endpoint accepting multiple permission levels
@app.get("/api/projects/{project_id}/traces")
@require_any_permission(
    Permission.TRACES_READ,
    Permission.TRACES_EXPORT,
    resource_type="project"
)
async def get_traces(
    request: Request,
    project_id: str,
    limit: int = 100
):
    """Get traces. Requires either TRACES_READ or TRACES_EXPORT permission."""
    return await trace_repository.get_traces(project_id, limit=limit)

# Example 4: Workspace-level endpoint
@app.post("/api/workspaces/{workspace_id}/projects")
@require_permission(Permission.WORKSPACE_CREATE_PROJECT, resource_type="workspace")
async def create_project(
    request: Request,
    workspace_id: str,
    project: dict
):
    """Create a project in workspace. Requires WORKSPACE_CREATE_PROJECT permission."""
    return await project_repository.create(workspace_id, project)
```

---

## SSO Integration

The a11i platform supports enterprise-grade Single Sign-On (SSO) integration with industry-standard protocols including SAML 2.0 and OpenID Connect (OIDC).

### SAML 2.0 Integration

SAML 2.0 is supported for enterprise customers who use identity providers like Okta, Azure AD, OneLogin, or on-premise solutions.

#### SAML Configuration

```python
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.settings import OneLogin_Saml2_Settings
from typing import Optional

class SAMLAuthProvider:
    """SAML 2.0 authentication provider for enterprise SSO."""

    def __init__(self, settings: dict):
        """
        Initialize SAML provider with configuration.

        Args:
            settings: SAML configuration including IDP metadata, SP endpoints

        Example settings:
            {
                "sp": {
                    "entityId": "https://a11i.example.com",
                    "assertionConsumerService": {
                        "url": "https://a11i.example.com/auth/saml/acs",
                        "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                    },
                    "singleLogoutService": {
                        "url": "https://a11i.example.com/auth/saml/sls",
                        "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
                    },
                    "x509cert": "<SP_CERTIFICATE>",
                    "privateKey": "<SP_PRIVATE_KEY>"
                },
                "idp": {
                    "entityId": "https://idp.example.com/saml",
                    "singleSignOnService": {
                        "url": "https://idp.example.com/saml/sso",
                        "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
                    },
                    "x509cert": "<IDP_CERTIFICATE>"
                },
                "security": {
                    "authnRequestsSigned": True,
                    "wantAssertionsSigned": True,
                    "wantNameIdEncrypted": False
                }
            }
        """
        self.settings = OneLogin_Saml2_Settings(settings)

    def get_login_url(self, request, relay_state: Optional[str] = None) -> str:
        """
        Generate SAML SSO login URL.

        Args:
            request: HTTP request object
            relay_state: Optional state to preserve across SSO flow

        Returns:
            URL to redirect user to for SAML authentication
        """
        auth = OneLogin_Saml2_Auth(request, self.settings)
        return auth.login(return_to=relay_state)

    def handle_callback(self, request) -> dict:
        """
        Process SAML authentication callback.

        Args:
            request: HTTP request containing SAML response

        Returns:
            Dictionary containing user information:
            {
                "user_id": "user@example.com",
                "email": "user@example.com",
                "name": "John Doe",
                "attributes": {
                    "groups": ["engineering", "admins"],
                    "department": "Engineering"
                }
            }

        Raises:
            AuthenticationError: If SAML response is invalid
        """
        auth = OneLogin_Saml2_Auth(request, self.settings)
        auth.process_response()

        if not auth.is_authenticated():
            errors = auth.get_errors()
            raise AuthenticationError(f"SAML authentication failed: {errors}")

        # Extract user information from SAML assertion
        attributes = auth.get_attributes()
        nameid = auth.get_nameid()

        return {
            "user_id": nameid,
            "email": attributes.get("email", [nameid])[0],
            "name": attributes.get("name", [""])[0],
            "attributes": {
                "groups": attributes.get("groups", []),
                "department": attributes.get("department", [""])[0],
            }
        }

    def get_logout_url(self, request, session_index: str) -> str:
        """
        Generate SAML logout URL.

        Args:
            request: HTTP request object
            session_index: SAML session index to terminate

        Returns:
            URL to redirect user to for SAML logout
        """
        auth = OneLogin_Saml2_Auth(request, self.settings)
        return auth.logout(session_index=session_index)
```

#### SAML Attribute Mapping

```python
class SAMLAttributeMapper:
    """Map SAML attributes to a11i user roles."""

    def __init__(self, org_id: str, mapping_config: dict):
        """
        Initialize attribute mapper with organization-specific configuration.

        Args:
            org_id: Organization ID
            mapping_config: Mapping from SAML groups to a11i roles

        Example mapping_config:
            {
                "group_attribute": "groups",
                "role_mappings": {
                    "engineering-admins": "org_admin",
                    "engineering-team": "workspace_editor",
                    "support-team": "project_viewer"
                }
            }
        """
        self.org_id = org_id
        self.mapping_config = mapping_config

    def map_attributes_to_roles(self, attributes: dict) -> list[dict]:
        """
        Map SAML attributes to a11i roles.

        Args:
            attributes: SAML attributes from authentication

        Returns:
            List of role assignments:
            [
                {"resource_type": "organization", "resource_id": "org-123", "role": "org_admin"},
                {"resource_type": "workspace", "resource_id": "ws-456", "role": "workspace_editor"}
            ]
        """
        group_attr = self.mapping_config.get("group_attribute", "groups")
        groups = attributes.get(group_attr, [])

        role_mappings = self.mapping_config.get("role_mappings", {})

        roles = []
        for group in groups:
            if group in role_mappings:
                role = role_mappings[group]
                # Determine resource type and ID based on role
                if role.startswith("org_"):
                    roles.append({
                        "resource_type": "organization",
                        "resource_id": self.org_id,
                        "role": role
                    })
                # Additional workspace/project mappings can be configured

        return roles
```

### OpenID Connect (OIDC) Integration

OIDC is supported for modern identity providers like Auth0, Okta, Google, and Azure AD.

#### OIDC Configuration

```python
from authlib.integrations.starlette_client import OAuth
from authlib.integrations.httpx_client import AsyncOAuth2Client

class OIDCAuthProvider:
    """OpenID Connect authentication provider."""

    def __init__(self, client_id: str, client_secret: str, issuer_url: str):
        """
        Initialize OIDC provider.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            issuer_url: OIDC issuer URL (e.g., https://your-org.okta.com)
        """
        self.oauth = OAuth()
        self.oauth.register(
            name='oidc',
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=f'{issuer_url}/.well-known/openid-configuration',
            client_kwargs={
                'scope': 'openid email profile groups',
                'token_endpoint_auth_method': 'client_secret_post'
            }
        )

    async def get_authorization_url(self, redirect_uri: str, state: str) -> str:
        """
        Get OAuth authorization URL.

        Args:
            redirect_uri: Callback URL after authentication
            state: State parameter for CSRF protection

        Returns:
            Authorization URL to redirect user to
        """
        return await self.oauth.oidc.create_authorization_url(
            redirect_uri=redirect_uri,
            state=state
        )

    async def handle_callback(self, request) -> dict:
        """
        Handle OAuth callback and exchange code for tokens.

        Args:
            request: HTTP request containing authorization code

        Returns:
            User information dictionary:
            {
                "user_id": "auth0|123456",
                "email": "user@example.com",
                "name": "John Doe",
                "groups": ["engineering", "admins"]
            }
        """
        # Exchange authorization code for tokens
        token = await self.oauth.oidc.authorize_access_token(request)

        # Get user info from token or userinfo endpoint
        user_info = token.get('userinfo')
        if not user_info:
            # Fetch from userinfo endpoint
            async with AsyncOAuth2Client(
                token=token,
                client_id=self.oauth.oidc.client_id
            ) as client:
                resp = await client.get(self.oauth.oidc.server_metadata['userinfo_endpoint'])
                user_info = resp.json()

        return {
            "user_id": user_info.get('sub'),
            "email": user_info.get('email'),
            "name": user_info.get('name'),
            "groups": user_info.get('groups', [])
        }
```

#### OIDC Group Mapping

```python
def map_oidc_groups_to_roles(groups: list[str], org_id: str) -> list[dict]:
    """
    Map OIDC groups to a11i roles.

    Args:
        groups: List of group names from OIDC provider
        org_id: Organization ID

    Returns:
        List of role assignments

    Example:
        >>> groups = ["engineering-team", "platform-admins"]
        >>> roles = map_oidc_groups_to_roles(groups, "org-123")
        >>> roles
        [
            {"resource_type": "organization", "resource_id": "org-123", "role": "org_admin"},
            {"resource_type": "workspace", "resource_id": "ws-eng", "role": "workspace_editor"}
        ]
    """
    # Load organization-specific group mapping configuration
    mapping_config = load_org_group_mapping(org_id)

    roles = []
    for group in groups:
        if group in mapping_config:
            mapping = mapping_config[group]
            roles.append({
                "resource_type": mapping["resource_type"],
                "resource_id": mapping["resource_id"],
                "role": mapping["role"]
            })

    return roles

async def create_session_from_sso(user_info: dict, roles: list[dict]) -> str:
    """
    Create user session after successful SSO authentication.

    Args:
        user_info: User information from SSO provider
        roles: List of role assignments

    Returns:
        Session token
    """
    # Get or create user
    user = await user_repository.get_or_create_user(
        user_id=user_info["user_id"],
        email=user_info["email"],
        name=user_info["name"]
    )

    # Update user roles
    await user_repository.update_user_roles(user["id"], roles)

    # Create session token
    session_token = generate_session_token(user["id"])
    await session_repository.create_session(user["id"], session_token)

    return session_token
```

### SSO Integration Endpoints

```python
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import RedirectResponse

app = FastAPI()

@app.get("/auth/saml/login")
async def saml_login(request: Request, org_id: str):
    """Initiate SAML SSO login flow."""
    # Get organization SAML configuration
    saml_config = await get_org_saml_config(org_id)
    if not saml_config:
        raise HTTPException(status_code=400, detail="SAML not configured for organization")

    # Create SAML provider and get login URL
    saml_provider = SAMLAuthProvider(saml_config)
    login_url = saml_provider.get_login_url(request)

    return RedirectResponse(url=login_url)

@app.post("/auth/saml/acs")
async def saml_callback(request: Request):
    """Handle SAML authentication callback."""
    # Extract org_id from SAML relay state
    org_id = extract_org_from_relay_state(request)

    # Get SAML configuration
    saml_config = await get_org_saml_config(org_id)
    saml_provider = SAMLAuthProvider(saml_config)

    # Process SAML response
    user_info = saml_provider.handle_callback(request)

    # Map SAML attributes to roles
    mapper = SAMLAttributeMapper(org_id, saml_config["attribute_mapping"])
    roles = mapper.map_attributes_to_roles(user_info["attributes"])

    # Create session
    session_token = await create_session_from_sso(user_info, roles)

    # Redirect to application with session
    response = RedirectResponse(url="/dashboard")
    response.set_cookie(key="session", value=session_token, httponly=True, secure=True)
    return response

@app.get("/auth/oidc/login")
async def oidc_login(request: Request, org_id: str):
    """Initiate OIDC login flow."""
    # Get organization OIDC configuration
    oidc_config = await get_org_oidc_config(org_id)
    if not oidc_config:
        raise HTTPException(status_code=400, detail="OIDC not configured for organization")

    # Create OIDC provider and get authorization URL
    oidc_provider = OIDCAuthProvider(
        client_id=oidc_config["client_id"],
        client_secret=oidc_config["client_secret"],
        issuer_url=oidc_config["issuer_url"]
    )

    redirect_uri = str(request.url_for('oidc_callback'))
    state = generate_csrf_token()

    auth_url = await oidc_provider.get_authorization_url(redirect_uri, state)
    return RedirectResponse(url=auth_url)

@app.get("/auth/oidc/callback")
async def oidc_callback(request: Request):
    """Handle OIDC callback."""
    # Verify state for CSRF protection
    if not verify_csrf_token(request.query_params.get('state')):
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    # Extract org_id from state
    org_id = extract_org_from_state(request.query_params.get('state'))

    # Get OIDC configuration
    oidc_config = await get_org_oidc_config(org_id)
    oidc_provider = OIDCAuthProvider(
        client_id=oidc_config["client_id"],
        client_secret=oidc_config["client_secret"],
        issuer_url=oidc_config["issuer_url"]
    )

    # Process callback
    user_info = await oidc_provider.handle_callback(request)

    # Map groups to roles
    roles = map_oidc_groups_to_roles(user_info.get("groups", []), org_id)

    # Create session
    session_token = await create_session_from_sso(user_info, roles)

    # Redirect with session
    response = RedirectResponse(url="/dashboard")
    response.set_cookie(key="session", value=session_token, httponly=True, secure=True)
    return response
```

---

## Custom Roles (Enterprise)

Enterprise customers can define custom roles with specific permission sets tailored to their organizational needs.

### Custom Role Service

```python
from typing import Set, Optional
from datetime import datetime

class CustomRoleService:
    """Manage custom roles for enterprise customers."""

    def __init__(self, db):
        """Initialize custom role service."""
        self.db = db

    async def create_custom_role(
        self,
        org_id: str,
        name: str,
        permissions: Set[Permission],
        description: str = "",
        scope: str = "project"
    ) -> str:
        """
        Create a custom role for an organization.

        Args:
            org_id: Organization ID
            name: Human-readable role name
            permissions: Set of permissions for this role
            description: Role description
            scope: Scope where role can be assigned (organization, workspace, project)

        Returns:
            Custom role ID

        Example:
            >>> role_id = await custom_role_service.create_custom_role(
            ...     org_id="org-123",
            ...     name="Security Auditor",
            ...     permissions={
            ...         Permission.TRACES_READ,
            ...         Permission.TRACES_EXPORT,
            ...         Permission.DASHBOARDS_READ,
            ...         Permission.ORG_AUDIT_LOGS
            ...     },
            ...     description="Can view and export traces for security audits",
            ...     scope="organization"
            ... )
        """
        # Validate permissions are allowed for scope
        self._validate_permissions_for_scope(permissions, scope)

        role_id = generate_role_id()

        await self.db.insert("custom_roles", {
            "role_id": role_id,
            "org_id": org_id,
            "name": name,
            "permissions": [p.name for p in permissions],
            "description": description,
            "scope": scope,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })

        return role_id

    async def update_custom_role(
        self,
        role_id: str,
        org_id: str,
        permissions: Optional[Set[Permission]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Update an existing custom role.

        Args:
            role_id: Custom role ID
            org_id: Organization ID (for verification)
            permissions: New permission set (if updating)
            name: New role name (if updating)
            description: New description (if updating)
        """
        updates = {"updated_at": datetime.utcnow()}

        if permissions is not None:
            # Get current role to check scope
            role = await self.get_custom_role(role_id, org_id)
            self._validate_permissions_for_scope(permissions, role["scope"])
            updates["permissions"] = [p.name for p in permissions]

        if name is not None:
            updates["name"] = name

        if description is not None:
            updates["description"] = description

        await self.db.update(
            "custom_roles",
            updates,
            where="role_id = {role_id:String} AND org_id = {org_id:String}",
            role_id=role_id,
            org_id=org_id
        )

    async def delete_custom_role(self, role_id: str, org_id: str):
        """
        Delete a custom role.

        Args:
            role_id: Custom role ID
            org_id: Organization ID (for verification)

        Note:
            This will also remove all user assignments to this role.
        """
        # Remove user assignments
        await self.db.execute("""
            DELETE FROM user_roles
            WHERE role = {role_id:String}
              AND org_id = {org_id:String}
        """, role_id=role_id, org_id=org_id)

        # Delete the role
        await self.db.execute("""
            DELETE FROM custom_roles
            WHERE role_id = {role_id:String}
              AND org_id = {org_id:String}
        """, role_id=role_id, org_id=org_id)

    async def get_custom_role(self, role_id: str, org_id: str) -> dict:
        """Get custom role details."""
        role = await self.db.query_one("""
            SELECT * FROM custom_roles
            WHERE role_id = {role_id:String}
              AND org_id = {org_id:String}
        """, role_id=role_id, org_id=org_id)

        if not role:
            raise ValueError(f"Custom role not found: {role_id}")

        # Convert permission names back to Permission enums
        role["permissions"] = {
            Permission[p] for p in role["permissions"]
        }

        return role

    async def get_org_roles(self, org_id: str) -> list[dict]:
        """
        Get all roles available to an organization (built-in + custom).

        Args:
            org_id: Organization ID

        Returns:
            List of role definitions including built-in and custom roles
        """
        # Built-in roles
        builtin_roles = [
            {
                "role_id": role_name,
                "name": role_name.replace("_", " ").title(),
                "permissions": list(perms),
                "type": "builtin",
                "scope": self._get_role_scope(role_name)
            }
            for role_name, perms in ROLE_PERMISSIONS.items()
        ]

        # Custom roles
        custom_roles = await self.db.query("""
            SELECT role_id, name, permissions, description, scope
            FROM custom_roles
            WHERE org_id = {org_id:String}
        """, org_id=org_id)

        for role in custom_roles:
            role["type"] = "custom"
            role["permissions"] = {Permission[p] for p in role["permissions"]}

        return builtin_roles + custom_roles

    def _validate_permissions_for_scope(self, permissions: Set[Permission], scope: str):
        """Validate that permissions are appropriate for the role scope."""
        if scope == "project":
            # Project-scoped roles cannot have org or workspace permissions
            forbidden = {
                Permission.ORG_BILLING,
                Permission.ORG_SETTINGS,
                Permission.ORG_CREATE_WORKSPACE,
                Permission.WORKSPACE_SETTINGS,
                Permission.WORKSPACE_DELETE
            }
            if permissions & forbidden:
                raise ValueError(
                    f"Project-scoped roles cannot have org/workspace permissions: {permissions & forbidden}"
                )

        elif scope == "workspace":
            # Workspace-scoped roles cannot have org permissions
            forbidden = {
                Permission.ORG_BILLING,
                Permission.ORG_SETTINGS,
                Permission.ORG_CREATE_WORKSPACE
            }
            if permissions & forbidden:
                raise ValueError(
                    f"Workspace-scoped roles cannot have org permissions: {permissions & forbidden}"
                )

    def _get_role_scope(self, role_name: str) -> str:
        """Determine scope from role name."""
        if role_name.startswith("org_"):
            return "organization"
        elif role_name.startswith("workspace_"):
            return "workspace"
        else:
            return "project"
```

### Custom Role Management API

```python
@app.post("/api/organizations/{org_id}/roles")
@require_permission(Permission.ORG_SETTINGS)
async def create_custom_role(
    request: Request,
    org_id: str,
    role: dict
):
    """
    Create a custom role.

    Request body:
    {
        "name": "Security Auditor",
        "description": "Can view and export traces for security audits",
        "scope": "organization",
        "permissions": [
            "TRACES_READ",
            "TRACES_EXPORT",
            "DASHBOARDS_READ",
            "ORG_AUDIT_LOGS"
        ]
    }
    """
    # Convert permission names to Permission enums
    permissions = {Permission[p] for p in role["permissions"]}

    custom_role_service = get_custom_role_service()
    role_id = await custom_role_service.create_custom_role(
        org_id=org_id,
        name=role["name"],
        permissions=permissions,
        description=role.get("description", ""),
        scope=role.get("scope", "project")
    )

    return {"role_id": role_id}

@app.get("/api/organizations/{org_id}/roles")
@require_permission(Permission.MEMBERS_READ)
async def list_org_roles(request: Request, org_id: str):
    """List all roles available to organization."""
    custom_role_service = get_custom_role_service()
    roles = await custom_role_service.get_org_roles(org_id)

    # Serialize permissions to strings
    for role in roles:
        role["permissions"] = [p.name for p in role["permissions"]]

    return {"roles": roles}

@app.put("/api/organizations/{org_id}/roles/{role_id}")
@require_permission(Permission.ORG_SETTINGS)
async def update_custom_role(
    request: Request,
    org_id: str,
    role_id: str,
    updates: dict
):
    """Update a custom role."""
    custom_role_service = get_custom_role_service()

    permissions = None
    if "permissions" in updates:
        permissions = {Permission[p] for p in updates["permissions"]}

    await custom_role_service.update_custom_role(
        role_id=role_id,
        org_id=org_id,
        permissions=permissions,
        name=updates.get("name"),
        description=updates.get("description")
    )

    return {"status": "updated"}

@app.delete("/api/organizations/{org_id}/roles/{role_id}")
@require_permission(Permission.ORG_SETTINGS)
async def delete_custom_role(request: Request, org_id: str, role_id: str):
    """Delete a custom role."""
    custom_role_service = get_custom_role_service()
    await custom_role_service.delete_custom_role(role_id, org_id)

    return {"status": "deleted"}
```

---

## Security Best Practices

### Principle of Least Privilege

Always assign users the minimum permissions necessary for their role:

- Start with **Viewer** roles and escalate only when needed
- Use **Project-scoped** roles instead of Organization-wide roles when possible
- Regularly audit user permissions and remove unnecessary access
- Implement time-limited elevated access for administrative tasks

### Session Management

```python
class SessionManager:
    """Secure session management."""

    async def create_session(
        self,
        user_id: str,
        ttl_seconds: int = 86400  # 24 hours
    ) -> str:
        """Create a new session with expiration."""
        session_id = generate_secure_token()

        await self.db.insert("sessions", {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl_seconds),
            "ip_address": get_client_ip(),
            "user_agent": get_user_agent()
        })

        return session_id

    async def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return user_id if valid."""
        session = await self.db.query_one("""
            SELECT user_id, expires_at
            FROM sessions
            WHERE session_id = {session_id:String}
        """, session_id=session_id)

        if not session or session["expires_at"] < datetime.utcnow():
            return None

        # Extend session on activity (sliding expiration)
        await self.db.update(
            "sessions",
            {"expires_at": datetime.utcnow() + timedelta(seconds=86400)},
            where="session_id = {session_id:String}",
            session_id=session_id
        )

        return session["user_id"]
```

### API Key Security

```python
class APIKeyManager:
    """Manage API keys with secure practices."""

    async def create_api_key(
        self,
        project_id: str,
        name: str,
        permissions: Set[Permission],
        expires_at: Optional[datetime] = None
    ) -> tuple[str, str]:
        """
        Create an API key.

        Returns:
            Tuple of (key_id, api_key). The api_key is only shown once.
        """
        key_id = generate_key_id()
        api_key = generate_secure_api_key()
        api_key_hash = hash_api_key(api_key)

        await self.db.insert("api_keys", {
            "key_id": key_id,
            "project_id": project_id,
            "name": name,
            "key_hash": api_key_hash,
            "permissions": [p.name for p in permissions],
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used_at": None
        })

        return key_id, api_key

    async def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key and return associated metadata."""
        api_key_hash = hash_api_key(api_key)

        key_info = await self.db.query_one("""
            SELECT key_id, project_id, permissions, expires_at
            FROM api_keys
            WHERE key_hash = {key_hash:String}
        """, key_hash=api_key_hash)

        if not key_info:
            return None

        # Check expiration
        if key_info["expires_at"] and key_info["expires_at"] < datetime.utcnow():
            return None

        # Update last used timestamp
        await self.db.update(
            "api_keys",
            {"last_used_at": datetime.utcnow()},
            where="key_id = {key_id:String}",
            key_id=key_info["key_id"]
        )

        return key_info
```

### Audit Logging

All authentication and authorization events should be logged:

```python
async def log_auth_event(
    event_type: str,
    user_id: Optional[str],
    resource_type: Optional[str],
    resource_id: Optional[str],
    success: bool,
    metadata: dict = None
):
    """Log authentication/authorization event."""
    await db.insert("auth_audit_log", {
        "event_type": event_type,  # login, logout, permission_check, etc.
        "user_id": user_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "success": success,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow(),
        "ip_address": get_client_ip(),
        "user_agent": get_user_agent()
    })
```

---

## Key Takeaways

> **Critical Security Requirements**
>
> 1. **Three-Tier Hierarchy**: Organization → Workspace → Project structure provides clear access boundaries
> 2. **Granular Permissions**: Fine-grained permission system enables precise access control
> 3. **SSO Integration**: Enterprise SAML 2.0 and OIDC support for centralized authentication
> 4. **Custom Roles**: Enterprise customers can define custom roles tailored to their needs
> 5. **Least Privilege**: Always assign minimum necessary permissions
> 6. **Audit Everything**: All auth events must be logged for compliance and security
> 7. **Secure Sessions**: Implement proper session management with expiration and rotation
> 8. **API Key Security**: Hash API keys, implement expiration, and track usage
> 9. **Permission Caching**: Cache permission lookups with short TTL for performance
> 10. **Regular Audits**: Periodically review user permissions and remove unnecessary access

> **Implementation Checklist**
>
> - [ ] Define organization hierarchy (orgs, workspaces, projects)
> - [ ] Implement Permission enumeration and role mappings
> - [ ] Build AuthorizationService with permission checking
> - [ ] Create permission decorators for API endpoints
> - [ ] Integrate SAML 2.0 for enterprise SSO
> - [ ] Integrate OIDC for modern SSO providers
> - [ ] Implement group-to-role mapping for SSO
> - [ ] Build custom role management for enterprise
> - [ ] Implement secure session management
> - [ ] Build API key management with hashing
> - [ ] Set up comprehensive audit logging
> - [ ] Create permission review and audit tools
> - [ ] Document SSO configuration for customers
> - [ ] Test permission inheritance across hierarchy
> - [ ] Validate least privilege enforcement

---

## References

### Related Documentation

- **API Gateway**: [../04-deployment-operations/api-gateway.md](../04-deployment-operations/api-gateway.md) - API authentication and authorization
- **Data Privacy**: [./data-privacy.md](./data-privacy.md) - Data access controls and privacy compliance
- **Audit Logging**: [./audit-logging.md](./audit-logging.md) - Comprehensive audit trail for security events

### External Standards

- **SAML 2.0**: [OASIS SAML 2.0 Specification](https://docs.oasis-open.org/security/saml/v2.0/)
- **OpenID Connect**: [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html)
- **OAuth 2.0**: [RFC 6749 - The OAuth 2.0 Authorization Framework](https://datatracker.ietf.org/doc/html/rfc6749)
- **RBAC Standard**: [NIST RBAC Model](https://csrc.nist.gov/projects/role-based-access-control)

### Implementation Libraries

- **Python SAML**: [python3-saml](https://github.com/onelogin/python3-saml)
- **Authlib**: [Authlib - OAuth/OIDC Client](https://docs.authlib.org/)
- **FastAPI Security**: [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)

### Security Resources

- **OWASP Authentication**: [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- **OWASP Session Management**: [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
- **API Key Security**: [OWASP API Security Project](https://owasp.org/www-project-api-security/)

---

**Document Maintenance**

- **Last Updated**: 2025-11-26
- **Next Review**: 2026-02-26
- **Owner**: Security Team
- **Feedback**: security@a11i.example.com
