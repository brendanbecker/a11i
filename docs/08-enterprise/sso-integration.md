---
title: "SSO Integration Guide"
category: "Enterprise"
tags: ["sso", "saml", "oidc", "okta", "azure-ad", "google-workspace", "auth0", "scim"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "./enterprise-features.md"
  - "./audit-logging.md"
  - "../05-security-compliance/rbac-auth.md"
---

# SSO Integration Guide

## Table of Contents

1. [Overview](#overview)
2. [SAML 2.0 Integration](#saml-20-integration)
3. [OIDC/OAuth Integration](#oidcoauth-integration)
4. [SCIM User Provisioning](#scim-user-provisioning)
5. [Provider-Specific Guides](#provider-specific-guides)
6. [Group-to-Role Mapping](#group-to-role-mapping)
7. [Just-in-Time (JIT) Provisioning](#just-in-time-jit-provisioning)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Key Takeaways](#key-takeaways)

---

## Overview

Single Sign-On (SSO) integration enables enterprise customers to authenticate users through their existing identity provider (IdP), providing centralized user management, enhanced security, and streamlined access control.

### Supported Protocols

| Protocol | Version | Use Case | Supported Providers |
|----------|---------|----------|---------------------|
| **SAML 2.0** | 2.0 | Enterprise SSO with traditional IdPs | Okta, Azure AD, OneLogin, Ping Identity |
| **OIDC** | 1.0 | Modern OAuth-based SSO | Auth0, Okta, Google Workspace, Azure AD |
| **SCIM** | 2.0 | Automated user provisioning | Okta, Azure AD, OneLogin |

### Benefits of SSO Integration

**For Administrators:**
- Centralized user management in single identity system
- Automated user provisioning and deprovisioning
- Group-based access control with automatic role assignment
- Enhanced security with MFA enforcement at IdP level
- Comprehensive audit trail of authentication events

**For End Users:**
- Single set of credentials for all applications
- Faster login experience (already authenticated to IdP)
- No password fatigue or weak password risks
- Seamless access across organization's tools

### Prerequisites

Before configuring SSO, ensure you have:

- [ ] Enterprise a11i subscription
- [ ] Organization Admin role in a11i
- [ ] Admin access to your identity provider
- [ ] Understanding of your organization's group/role structure
- [ ] (For SAML) X.509 certificate from your IdP
- [ ] (For OIDC) OAuth client credentials from your IdP

---

## SAML 2.0 Integration

SAML 2.0 (Security Assertion Markup Language) is the industry-standard protocol for enterprise SSO, enabling federated authentication between identity providers and service providers.

### SAML Architecture Overview

```
┌─────────────────┐                    ┌──────────────────┐
│                 │                    │                  │
│  User Browser   │                    │  a11i (SP)       │
│                 │                    │  Service Provider│
└────────┬────────┘                    └────────┬─────────┘
         │                                      │
         │  1. Access a11i                      │
         │────────────────────────────────────> │
         │                                      │
         │  2. Redirect to IdP with SAML Request│
         │ <────────────────────────────────────│
         │                                      │
┌────────▼────────┐                             │
│                 │                             │
│  Identity       │                             │
│  Provider (IdP) │                             │
│  (Okta, etc.)   │                             │
└────────┬────────┘                             │
         │                                      │
         │  3. User authenticates               │
         │     (username/password + MFA)        │
         │                                      │
         │  4. IdP generates SAML assertion     │
         │                                      │
         │  5. POST SAML response to SP         │
         │──────────────────────────────────────>│
         │                                      │
         │  6. SP validates assertion           │
         │     & creates session                │
         │                                      │
         │  7. User logged into a11i            │
         │ <────────────────────────────────────│
         │                                      │
```

### SAML Configuration Steps

#### Step 1: Obtain a11i Service Provider Metadata

Log into a11i as Organization Admin and navigate to **Settings → Authentication → SSO Configuration**.

**a11i Service Provider (SP) Information:**

```xml
<!-- a11i SAML SP Metadata -->
<EntityDescriptor xmlns="urn:oasis:names:tc:SAML:2.0:metadata"
                  entityID="https://a11i.dev/saml/metadata">

  <SPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">

    <!-- Assertion Consumer Service (ACS) URL -->
    <AssertionConsumerService
      Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
      Location="https://a11i.dev/auth/saml/acs"
      index="0"
      isDefault="true"/>

    <!-- Single Logout Service -->
    <SingleLogoutService
      Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
      Location="https://a11i.dev/auth/saml/sls"/>

    <!-- Name ID Format -->
    <NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</NameIDFormat>

    <!-- Signing Certificate -->
    <KeyDescriptor use="signing">
      <KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#">
        <X509Data>
          <X509Certificate>MIIDXTCCAkWgAwIBAgIJ...</X509Certificate>
        </X509Data>
      </KeyInfo>
    </KeyDescriptor>
  </SPSSODescriptor>
</EntityDescriptor>
```

**Key URLs for Your IdP Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Entity ID** | `https://a11i.dev/saml/metadata` | Unique identifier for a11i |
| **ACS URL** | `https://a11i.dev/auth/saml/acs` | Where IdP sends SAML assertions |
| **Single Logout URL** | `https://a11i.dev/auth/saml/sls` | Optional logout endpoint |
| **Name ID Format** | `emailAddress` | User identifier format |

#### Step 2: Configure Identity Provider

Upload a11i metadata to your IdP or manually configure using the URLs above.

**Required SAML Attributes:**

| SAML Attribute | Description | Required | Example Value |
|----------------|-------------|----------|---------------|
| `email` | User email address | Yes | `user@company.com` |
| `firstName` | User's first name | No | `John` |
| `lastName` | User's last name | No | `Doe` |
| `groups` | User's group memberships | Recommended | `["engineering", "platform-team"]` |
| `department` | User's department | No | `Engineering` |
| `title` | User's job title | No | `Senior Engineer` |

#### Step 3: Configure a11i with IdP Metadata

In a11i SSO configuration, upload your IdP metadata XML or enter manually:

```yaml
# a11i SAML Configuration
saml_config:
  # Identity Provider Settings
  idp:
    entity_id: "http://www.okta.com/exkabcdef123456"
    sso_url: "https://company.okta.com/app/a11i/exkabcdef123456/sso/saml"
    slo_url: "https://company.okta.com/app/a11i/exkabcdef123456/slo/saml"
    x509_cert: |
      -----BEGIN CERTIFICATE-----
      MIIDpDCCAoygAwIBAgIGAXYZ9...
      -----END CERTIFICATE-----

  # Service Provider Settings
  sp:
    entity_id: "https://a11i.dev/saml/metadata"
    acs_url: "https://a11i.dev/auth/saml/acs"
    sls_url: "https://a11i.dev/auth/saml/sls"

  # Security Settings
  security:
    authn_requests_signed: true
    want_assertions_signed: true
    want_assertions_encrypted: false
    want_name_id_encrypted: false
    signature_algorithm: "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"
    digest_algorithm: "http://www.w3.org/2001/04/xmlenc#sha256"

  # Attribute Mapping
  attribute_mapping:
    email: "email"
    first_name: "firstName"
    last_name: "lastName"
    groups: "groups"
    department: "department"
```

#### Step 4: Test SAML Integration

1. Click **Test SSO Configuration** in a11i admin panel
2. You'll be redirected to your IdP login page
3. Authenticate with your corporate credentials
4. If successful, you'll be redirected back to a11i
5. Verify user profile was created with correct attributes

**Test Checklist:**

- [ ] SSO login redirect works correctly
- [ ] User can authenticate at IdP
- [ ] SAML assertion is received and validated
- [ ] User profile is created in a11i
- [ ] Email and name attributes are populated
- [ ] Group memberships are received
- [ ] User is assigned correct roles based on groups
- [ ] SSO logout works correctly (optional)

### SAML Configuration Example

**Complete Python Implementation:**

```python
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.utils import OneLogin_Saml2_Utils
from flask import Flask, request, redirect, session

app = Flask(__name__)

# SAML Settings
SAML_SETTINGS = {
    "strict": True,
    "debug": False,

    "sp": {
        "entityId": "https://a11i.dev/saml/metadata",
        "assertionConsumerService": {
            "url": "https://a11i.dev/auth/saml/acs",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
        },
        "singleLogoutService": {
            "url": "https://a11i.dev/auth/saml/sls",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
        },
        "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
        "x509cert": "",  # SP certificate
        "privateKey": ""  # SP private key
    },

    "idp": {
        "entityId": "http://www.okta.com/exkabcdef123456",
        "singleSignOnService": {
            "url": "https://company.okta.com/app/a11i/exkabcdef123456/sso/saml",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
        },
        "singleLogoutService": {
            "url": "https://company.okta.com/app/a11i/exkabcdef123456/slo/saml",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
        },
        "x509cert": "MIIDpDCCAoygAwIBAgIGAXYZ9..."  # IdP certificate
    },

    "security": {
        "nameIdEncrypted": False,
        "authnRequestsSigned": True,
        "logoutRequestSigned": True,
        "logoutResponseSigned": True,
        "signMetadata": True,
        "wantMessagesSigned": True,
        "wantAssertionsSigned": True,
        "wantAssertionsEncrypted": False,
        "wantNameIdEncrypted": False,
        "signatureAlgorithm": "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256",
        "digestAlgorithm": "http://www.w3.org/2001/04/xmlenc#sha256"
    }
}

@app.route('/auth/saml/login')
def saml_login():
    """Initiate SAML SSO login."""
    auth = OneLogin_Saml2_Auth(request, SAML_SETTINGS)

    # Redirect to IdP for authentication
    return redirect(auth.login())

@app.route('/auth/saml/acs', methods=['POST'])
def saml_acs():
    """Handle SAML assertion from IdP."""
    auth = OneLogin_Saml2_Auth(request, SAML_SETTINGS)

    # Process SAML response
    auth.process_response()

    # Check for errors
    errors = auth.get_errors()
    if errors:
        return f"SAML Authentication Failed: {', '.join(errors)}", 400

    # Verify authentication
    if not auth.is_authenticated():
        return "Authentication failed", 401

    # Extract user attributes
    attributes = auth.get_attributes()
    nameid = auth.get_nameid()

    user_info = {
        "email": nameid,
        "first_name": attributes.get("firstName", [""])[0],
        "last_name": attributes.get("lastName", [""])[0],
        "groups": attributes.get("groups", []),
        "department": attributes.get("department", [""])[0],
    }

    # Create or update user in database
    user = await create_or_update_user(user_info)

    # Map groups to roles
    roles = map_groups_to_roles(user_info["groups"], user.org_id)
    await assign_user_roles(user.id, roles)

    # Create session
    session['user_id'] = user.id
    session['saml_session_index'] = auth.get_session_index()

    # Log authentication event
    await audit_log.log(
        action="saml_login_success",
        user_id=user.id,
        details={"idp": SAML_SETTINGS["idp"]["entityId"]}
    )

    # Redirect to application
    return redirect('/dashboard')

@app.route('/auth/saml/logout')
def saml_logout():
    """Initiate SAML logout."""
    auth = OneLogin_Saml2_Auth(request, SAML_SETTINGS)

    # Get session index for logout
    session_index = session.get('saml_session_index')

    # Clear local session
    session.clear()

    # Redirect to IdP for logout
    return redirect(auth.logout(session_index=session_index))

@app.route('/auth/saml/metadata')
def saml_metadata():
    """Serve SAML SP metadata."""
    auth = OneLogin_Saml2_Auth(request, SAML_SETTINGS)
    settings = auth.get_settings()
    metadata = settings.get_sp_metadata()

    return metadata, 200, {'Content-Type': 'text/xml'}
```

---

## OIDC/OAuth Integration

OpenID Connect (OIDC) is a modern authentication layer built on top of OAuth 2.0, providing a simpler integration path for identity providers that support it.

### OIDC Architecture Overview

```
┌─────────────────┐                    ┌──────────────────┐
│                 │                    │                  │
│  User Browser   │                    │  a11i (Client)   │
│                 │                    │  Relying Party   │
└────────┬────────┘                    └────────┬─────────┘
         │                                      │
         │  1. Click "Login with SSO"           │
         │────────────────────────────────────> │
         │                                      │
         │  2. Redirect to authorization URL    │
         │ <────────────────────────────────────│
         │                                      │
┌────────▼────────┐                             │
│                 │                             │
│  Authorization  │                             │
│  Server         │                             │
│  (Auth0, Okta)  │                             │
└────────┬────────┘                             │
         │                                      │
         │  3. User authenticates & consents    │
         │                                      │
         │  4. Redirect with authorization code │
         │──────────────────────────────────────>│
         │                                      │
         │  5. Exchange code for tokens         │
         │ <────────────────────────────────────│
         │──────────────────────────────────────>│
         │                                      │
         │  6. Return ID token + access token   │
         │ <────────────────────────────────────│
         │                                      │
         │  7. User logged into a11i            │
         │ <────────────────────────────────────│
         │                                      │
```

### OIDC Configuration Steps

#### Step 1: Register a11i as OAuth Client

In your identity provider, create a new OAuth 2.0 / OIDC application:

**Application Settings:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Application Name** | `a11i AI Observability` | Displayed to users during consent |
| **Application Type** | `Web Application` | Server-side OAuth flow |
| **Redirect URIs** | `https://a11i.dev/auth/oidc/callback` | Where IdP sends authorization code |
| **Logout URIs** | `https://a11i.dev/auth/logout` | Optional post-logout redirect |
| **Scopes** | `openid email profile groups` | Requested user information |

**Save the following credentials:**
- Client ID (e.g., `0oa1bc2d3e4f5g6h7i8j`)
- Client Secret (e.g., `abcdef123456...` - keep secure!)

#### Step 2: Configure a11i OIDC Settings

In a11i admin panel, navigate to **Settings → Authentication → SSO Configuration → OIDC**.

```yaml
# a11i OIDC Configuration
oidc_config:
  # Provider Information
  provider_name: "Okta"
  issuer_url: "https://company.okta.com"
  discovery_url: "https://company.okta.com/.well-known/openid-configuration"

  # OAuth Client Credentials
  client_id: "0oa1bc2d3e4f5g6h7i8j"
  client_secret: "abcdef123456789..."  # Encrypted in database

  # OAuth Endpoints (auto-discovered from issuer)
  authorization_endpoint: "https://company.okta.com/oauth2/v1/authorize"
  token_endpoint: "https://company.okta.com/oauth2/v1/token"
  userinfo_endpoint: "https://company.okta.com/oauth2/v1/userinfo"
  jwks_uri: "https://company.okta.com/oauth2/v1/keys"

  # OAuth Scopes
  scopes:
    - "openid"
    - "email"
    - "profile"
    - "groups"

  # Token Validation
  validate_issuer: true
  validate_audience: true
  require_https: true

  # Claims Mapping
  claims_mapping:
    user_id: "sub"
    email: "email"
    name: "name"
    first_name: "given_name"
    last_name: "family_name"
    groups: "groups"
```

#### Step 3: Test OIDC Integration

**Python Implementation:**

```python
from authlib.integrations.flask_client import OAuth
from flask import Flask, redirect, url_for, session

app = Flask(__name__)
oauth = OAuth(app)

# Register OIDC provider
oidc = oauth.register(
    name='oidc',
    client_id='0oa1bc2d3e4f5g6h7i8j',
    client_secret='abcdef123456789...',
    server_metadata_url='https://company.okta.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile groups',
        'token_endpoint_auth_method': 'client_secret_post'
    }
)

@app.route('/auth/oidc/login')
def oidc_login():
    """Initiate OIDC login flow."""
    # Generate redirect URI
    redirect_uri = url_for('oidc_callback', _external=True)

    # Redirect to authorization endpoint
    return oidc.authorize_redirect(redirect_uri)

@app.route('/auth/oidc/callback')
async def oidc_callback():
    """Handle OIDC callback with authorization code."""
    try:
        # Exchange authorization code for tokens
        token = await oidc.authorize_access_token()

        # Parse ID token (JWT)
        user_info = token.get('userinfo')
        if not user_info:
            # Fetch from userinfo endpoint
            user_info = await oidc.userinfo(token=token)

        # Extract user details
        user_data = {
            "user_id": user_info.get('sub'),
            "email": user_info.get('email'),
            "name": user_info.get('name'),
            "first_name": user_info.get('given_name'),
            "last_name": user_info.get('family_name'),
            "groups": user_info.get('groups', [])
        }

        # Create or update user
        user = await create_or_update_user(user_data)

        # Map groups to roles
        roles = map_groups_to_roles(user_data["groups"], user.org_id)
        await assign_user_roles(user.id, roles)

        # Create session
        session['user_id'] = user.id
        session['id_token'] = token['id_token']
        session['access_token'] = token['access_token']

        # Log authentication
        await audit_log.log(
            action="oidc_login_success",
            user_id=user.id,
            details={"provider": "okta"}
        )

        return redirect('/dashboard')

    except Exception as e:
        await audit_log.log(
            action="oidc_login_failed",
            error=str(e)
        )
        return f"Authentication failed: {str(e)}", 401

@app.route('/auth/logout')
def logout():
    """Logout user and optionally redirect to IdP logout."""
    # Get ID token for logout
    id_token = session.get('id_token')

    # Clear session
    session.clear()

    # Optional: Redirect to IdP logout
    if id_token:
        logout_url = f"https://company.okta.com/oauth2/v1/logout?id_token_hint={id_token}&post_logout_redirect_uri={url_for('home', _external=True)}"
        return redirect(logout_url)

    return redirect('/')
```

---

## SCIM User Provisioning

System for Cross-domain Identity Management (SCIM) enables automated user provisioning and deprovisioning between your IdP and a11i.

### SCIM Benefits

- **Automatic User Creation**: New employees automatically get a11i access
- **Automatic Deprovisioning**: Terminated employees immediately lose access
- **Group Synchronization**: IdP groups automatically sync to a11i
- **Profile Updates**: User attribute changes propagate automatically

### SCIM Configuration

#### Step 1: Generate SCIM API Token in a11i

Navigate to **Settings → Authentication → SCIM Configuration** and click **Generate SCIM Token**.

**SCIM Endpoint Information:**

| Parameter | Value |
|-----------|-------|
| **SCIM Base URL** | `https://a11i.dev/scim/v2` |
| **Authentication** | `Bearer <SCIM_TOKEN>` |
| **Supported Resources** | `Users`, `Groups` |
| **SCIM Version** | `2.0` |

#### Step 2: Configure SCIM in Identity Provider

**For Okta:**

1. Navigate to your a11i application in Okta
2. Go to **Provisioning** → **Integration**
3. Click **Configure API Integration**
4. Enable **API integration**
5. Enter SCIM Base URL: `https://a11i.dev/scim/v2`
6. Enter SCIM Token in **API Token** field
7. Click **Test API Credentials**
8. Go to **Provisioning** → **To App**
9. Enable:
   - Create Users
   - Update User Attributes
   - Deactivate Users
10. Save configuration

**For Azure AD:**

1. Navigate to **Enterprise Applications** → Your a11i app
2. Go to **Provisioning** → **Automatic**
3. Enter:
   - Tenant URL: `https://a11i.dev/scim/v2`
   - Secret Token: `<SCIM_TOKEN>`
4. Click **Test Connection**
5. Configure **Mappings**:
   - Azure AD Attribute → a11i Attribute
   - `userPrincipalName` → `userName`
   - `mail` → `emails[type eq "work"].value`
   - `givenName` → `name.givenName`
   - `surname` → `name.familyName`
   - `groups` → `groups`
6. Set **Provisioning Status** to **On**
7. Save configuration

### SCIM API Implementation

```python
from fastapi import FastAPI, Header, HTTPException, status
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

# SCIM User Schema
class SCIMName(BaseModel):
    givenName: Optional[str]
    familyName: Optional[str]
    formatted: Optional[str]

class SCIMEmail(BaseModel):
    value: str
    type: str = "work"
    primary: bool = True

class SCIMUser(BaseModel):
    schemas: List[str] = ["urn:ietf:params:scim:schemas:core:2.0:User"]
    id: Optional[str]
    userName: str
    name: Optional[SCIMName]
    emails: List[SCIMEmail]
    active: bool = True
    groups: List[str] = []
    meta: Optional[dict]

# SCIM Group Schema
class SCIMGroupMember(BaseModel):
    value: str
    display: str

class SCIMGroup(BaseModel):
    schemas: List[str] = ["urn:ietf:params:scim:schemas:core:2.0:Group"]
    id: Optional[str]
    displayName: str
    members: List[SCIMGroupMember] = []
    meta: Optional[dict]

async def verify_scim_token(authorization: str):
    """Verify SCIM bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )

    token = authorization.replace("Bearer ", "")
    is_valid = await validate_scim_token(token)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid SCIM token"
        )

@app.post("/scim/v2/Users", status_code=status.HTTP_201_CREATED)
async def create_user(
    user: SCIMUser,
    authorization: str = Header(None)
):
    """Create a new user via SCIM."""
    await verify_scim_token(authorization)

    # Create user in database
    db_user = await user_repository.create_user(
        email=user.emails[0].value,
        username=user.userName,
        first_name=user.name.givenName if user.name else None,
        last_name=user.name.familyName if user.name else None,
        active=user.active
    )

    # Map groups to roles
    if user.groups:
        roles = map_groups_to_roles(user.groups, db_user.org_id)
        await assign_user_roles(db_user.id, roles)

    # Log provisioning event
    await audit_log.log(
        action="scim_user_created",
        user_id=db_user.id,
        details={"username": user.userName, "groups": user.groups}
    )

    # Return SCIM user resource
    return SCIMUser(
        id=db_user.id,
        userName=user.userName,
        name=user.name,
        emails=user.emails,
        active=user.active,
        groups=user.groups,
        meta={
            "resourceType": "User",
            "created": db_user.created_at.isoformat(),
            "lastModified": db_user.updated_at.isoformat(),
            "location": f"/scim/v2/Users/{db_user.id}"
        }
    )

@app.get("/scim/v2/Users/{user_id}")
async def get_user(
    user_id: str,
    authorization: str = Header(None)
):
    """Get user by ID."""
    await verify_scim_token(authorization)

    user = await user_repository.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get user groups
    groups = await get_user_groups(user.id)

    return SCIMUser(
        id=user.id,
        userName=user.username,
        name=SCIMName(
            givenName=user.first_name,
            familyName=user.last_name,
            formatted=f"{user.first_name} {user.last_name}"
        ),
        emails=[SCIMEmail(value=user.email)],
        active=user.active,
        groups=[g.name for g in groups],
        meta={
            "resourceType": "User",
            "created": user.created_at.isoformat(),
            "lastModified": user.updated_at.isoformat(),
            "location": f"/scim/v2/Users/{user.id}"
        }
    )

@app.patch("/scim/v2/Users/{user_id}")
async def update_user(
    user_id: str,
    operations: dict,
    authorization: str = Header(None)
):
    """Update user attributes."""
    await verify_scim_token(authorization)

    user = await user_repository.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Process SCIM PATCH operations
    for op in operations.get("Operations", []):
        if op["op"] == "replace":
            if op["path"] == "active":
                await user_repository.update_user(user_id, {"active": op["value"]})
            # Handle other attributes...

    await audit_log.log(
        action="scim_user_updated",
        user_id=user_id,
        details={"operations": operations}
    )

    return await get_user(user_id, authorization)

@app.delete("/scim/v2/Users/{user_id}")
async def delete_user(
    user_id: str,
    authorization: str = Header(None)
):
    """Deactivate user (soft delete)."""
    await verify_scim_token(authorization)

    # Soft delete by setting active=false
    await user_repository.update_user(user_id, {"active": False})

    await audit_log.log(
        action="scim_user_deleted",
        user_id=user_id
    )

    return {"status": "deleted"}, 204
```

---

## Provider-Specific Guides

### Okta Configuration

**Step-by-Step Okta Integration:**

1. **Create a11i Application in Okta:**
   - Log into Okta Admin Console
   - Navigate to **Applications** → **Applications**
   - Click **Create App Integration**
   - Select **SAML 2.0** or **OIDC - Web Application**
   - Click **Next**

2. **Configure SAML Settings:**
   ```
   General Settings:
   - App name: a11i AI Observability
   - App logo: (upload a11i logo)

   SAML Settings:
   - Single sign on URL: https://a11i.dev/auth/saml/acs
   - Audience URI (SP Entity ID): https://a11i.dev/saml/metadata
   - Name ID format: EmailAddress
   - Application username: Email

   Attribute Statements:
   - email: user.email
   - firstName: user.firstName
   - lastName: user.lastName
   - groups: appuser.groups (combine values)
   ```

3. **Assign Users and Groups:**
   - Navigate to **Assignments** tab
   - Click **Assign** → **Assign to Groups**
   - Select groups that should have a11i access
   - Click **Done**

4. **Download IdP Metadata:**
   - Navigate to **Sign On** tab
   - Under **SAML Setup**, click **View SAML setup instructions**
   - Download **IDP metadata** XML
   - Upload to a11i SSO configuration

5. **Configure SCIM Provisioning** (Optional):
   - Navigate to **Provisioning** tab
   - Click **Configure API Integration**
   - Enable **API integration**
   - Base URL: `https://a11i.dev/scim/v2`
   - API Token: (from a11i SCIM settings)
   - Click **Test API Credentials**
   - Enable provisioning features:
     - ✓ Create Users
     - ✓ Update User Attributes
     - ✓ Deactivate Users
     - ✓ Import Groups

### Azure AD Configuration

**Step-by-Step Azure AD Integration:**

1. **Register a11i in Azure AD:**
   - Navigate to **Azure Active Directory** → **Enterprise Applications**
   - Click **New application** → **Create your own application**
   - Name: `a11i AI Observability`
   - Select: **Integrate any other application you don't find in the gallery (Non-gallery)**
   - Click **Create**

2. **Configure Single Sign-On:**
   - Navigate to **Single sign-on**
   - Select **SAML**
   - Click **Edit** on **Basic SAML Configuration**:
     ```
     Identifier (Entity ID): https://a11i.dev/saml/metadata
     Reply URL (Assertion Consumer Service URL): https://a11i.dev/auth/saml/acs
     Sign on URL: https://a11i.dev
     Logout URL: https://a11i.dev/auth/saml/sls
     ```

3. **Configure Attributes & Claims:**
   ```
   Required claims:
   - Unique User Identifier: user.mail
   - email: user.mail
   - givenname: user.givenname
   - surname: user.surname
   - groups: user.groups (requires app role assignment)
   ```

4. **Download Metadata:**
   - In **SAML Signing Certificate** section
   - Download **Federation Metadata XML**
   - Upload to a11i SSO configuration

5. **Assign Users:**
   - Navigate to **Users and groups**
   - Click **Add user/group**
   - Select users or groups
   - Click **Assign**

6. **Configure Provisioning** (Optional):
   - Navigate to **Provisioning**
   - Click **Get started**
   - Provisioning Mode: **Automatic**
   - Admin Credentials:
     - Tenant URL: `https://a11i.dev/scim/v2`
     - Secret Token: (from a11i SCIM settings)
   - Click **Test Connection**
   - Configure **Mappings**
   - Set **Provisioning Status** to **On**

### Google Workspace Configuration

**Step-by-Step Google Workspace Integration:**

1. **Create Custom SAML App:**
   - Go to **Google Admin Console** → **Apps** → **Web and mobile apps**
   - Click **Add App** → **Add custom SAML app**
   - App name: `a11i AI Observability`
   - Click **Continue**

2. **Download IdP Metadata:**
   - Download **IDP metadata**
   - Or manually note:
     - SSO URL
     - Entity ID
     - Certificate
   - Click **Continue**

3. **Configure Service Provider:**
   ```
   ACS URL: https://a11i.dev/auth/saml/acs
   Entity ID: https://a11i.dev/saml/metadata
   Start URL: https://a11i.dev
   Name ID format: EMAIL
   Name ID: Basic Information > Primary email
   ```
   - Click **Continue**

4. **Configure Attribute Mapping:**
   ```
   Google Directory attributes → App attributes:
   - Primary email → email
   - First name → firstName
   - Last name → lastName
   ```
   - Click **Finish**

5. **Enable for Users:**
   - Select organizational unit
   - Service status: **ON for everyone** (or specific OU)
   - Click **Save**

6. **Upload Metadata to a11i:**
   - Upload downloaded IdP metadata to a11i
   - Or manually enter SSO URL, Entity ID, Certificate

**Note:** Google Workspace does not support SCIM provisioning natively. Consider using a third-party service like Okta for automated provisioning.

### Auth0 Configuration

**Step-by-Step Auth0 Integration (OIDC):**

1. **Create Application in Auth0:**
   - Navigate to **Applications** → **Applications**
   - Click **Create Application**
   - Name: `a11i AI Observability`
   - Application Type: **Regular Web Application**
   - Click **Create**

2. **Configure Application Settings:**
   ```
   Settings:
   - Allowed Callback URLs: https://a11i.dev/auth/oidc/callback
   - Allowed Logout URLs: https://a11i.dev/auth/logout
   - Allowed Web Origins: https://a11i.dev
   ```
   - Click **Save Changes**

3. **Note Credentials:**
   ```
   Domain: company.auth0.com
   Client ID: AbCd1234EfGh5678IjKl
   Client Secret: (copy and keep secure)
   ```

4. **Configure Groups (Optional):**
   - Install **Authorization Extension** from Marketplace
   - Create groups in Authorization Extension
   - Assign users to groups
   - Add rule to include groups in ID token:

   ```javascript
   function(user, context, callback) {
     const namespace = 'https://a11i.dev/';
     if (context.authorization) {
       const groups = context.authorization.groups || [];
       context.idToken[namespace + 'groups'] = groups;
       context.accessToken[namespace + 'groups'] = groups;
     }
     callback(null, user, context);
   }
   ```

5. **Configure a11i:**
   - Issuer URL: `https://company.auth0.com/`
   - Client ID: (from step 3)
   - Client Secret: (from step 3)
   - Scopes: `openid email profile groups`

---

## Group-to-Role Mapping

Automatically assign a11i roles based on identity provider group memberships.

### Mapping Configuration

```yaml
# Group-to-Role Mapping Configuration
group_mappings:
  organization_id: "acme-corp"

  mappings:
    # Engineering team gets workspace editor access
    - idp_group: "engineering-team"
      a11i_role: "workspace_editor"
      scope: "workspace"
      workspace_id: "ws-engineering"

    # Platform team gets workspace admin
    - idp_group: "platform-team"
      a11i_role: "workspace_admin"
      scope: "workspace"
      workspace_id: "ws-engineering"

    # IT admins get org admin
    - idp_group: "it-admins"
      a11i_role: "org_admin"
      scope: "organization"
      org_id: "acme-corp"

    # Security team gets custom security auditor role
    - idp_group: "security-team"
      a11i_role: "security_auditor"  # Custom enterprise role
      scope: "organization"
      org_id: "acme-corp"

    # QA team gets viewer access
    - idp_group: "qa-team"
      a11i_role: "project_viewer"
      scope: "project"
      project_ids: ["proj-api", "proj-frontend", "proj-backend"]

  # Default role for users with no group mappings
  default_role:
    role: "project_viewer"
    scope: "organization"

  # JIT provisioning settings
  jit_provisioning:
    enabled: true
    create_user_on_login: true
    update_attributes_on_login: true
    deactivate_on_group_removal: true
```

### Mapping Implementation

```python
from typing import List, Dict

class GroupRoleMapper:
    """Map IdP groups to a11i roles."""

    def __init__(self, org_id: str, mapping_config: dict):
        self.org_id = org_id
        self.mapping_config = mapping_config

    async def map_groups_to_roles(
        self,
        groups: List[str]
    ) -> List[Dict]:
        """
        Map IdP groups to a11i role assignments.

        Args:
            groups: List of group names from IdP

        Returns:
            List of role assignments:
            [
                {
                    "role": "workspace_admin",
                    "resource_type": "workspace",
                    "resource_id": "ws-engineering"
                },
                ...
            ]
        """
        role_assignments = []

        for mapping in self.mapping_config.get("mappings", []):
            if mapping["idp_group"] in groups:
                assignment = {
                    "role": mapping["a11i_role"],
                    "resource_type": mapping["scope"],
                }

                # Add resource ID based on scope
                if mapping["scope"] == "organization":
                    assignment["resource_id"] = mapping.get("org_id", self.org_id)
                elif mapping["scope"] == "workspace":
                    assignment["resource_id"] = mapping["workspace_id"]
                elif mapping["scope"] == "project":
                    # Handle multiple projects
                    for project_id in mapping.get("project_ids", []):
                        role_assignments.append({
                            **assignment,
                            "resource_id": project_id
                        })
                    continue

                role_assignments.append(assignment)

        # Apply default role if no mappings found
        if not role_assignments and self.mapping_config.get("default_role"):
            default = self.mapping_config["default_role"]
            role_assignments.append({
                "role": default["role"],
                "resource_type": default["scope"],
                "resource_id": self.org_id
            })

        return role_assignments

    async def sync_user_roles(
        self,
        user_id: str,
        groups: List[str]
    ):
        """
        Synchronize user roles based on current group memberships.

        This updates the user's roles to match their IdP groups,
        removing roles from groups they're no longer in.
        """
        # Get target role assignments from groups
        target_roles = await self.map_groups_to_roles(groups)

        # Get current role assignments
        current_roles = await user_repository.get_user_roles(user_id)

        # Determine additions and removals
        to_add = [r for r in target_roles if r not in current_roles]
        to_remove = [r for r in current_roles if r not in target_roles]

        # Apply changes
        for role in to_add:
            await user_repository.assign_role(user_id, role)
            await audit_log.log(
                action="role_assigned_via_sso",
                user_id=user_id,
                details=role
            )

        for role in to_remove:
            await user_repository.remove_role(user_id, role)
            await audit_log.log(
                action="role_removed_via_sso",
                user_id=user_id,
                details=role
            )
```

---

## Just-in-Time (JIT) Provisioning

JIT provisioning automatically creates user accounts when they first login via SSO, eliminating the need for manual user creation.

### JIT Provisioning Configuration

```yaml
jit_provisioning:
  enabled: true

  # User creation settings
  create_user:
    on_first_login: true
    email_verification: false  # Skip verification for SSO users
    send_welcome_email: true

  # Attribute synchronization
  sync_attributes:
    on_every_login: true
    attributes:
      - email
      - first_name
      - last_name
      - department
      - title
      - groups

  # Role assignment
  role_assignment:
    method: "group_mapping"  # Use group-to-role mappings
    fallback_role: "project_viewer"

  # Account lifecycle
  lifecycle:
    deactivate_on_group_removal: true
    grace_period_days: 7  # Keep account active for 7 days after group removal
    delete_inactive_after_days: 90
```

### JIT Implementation

```python
from datetime import datetime, timedelta

class JITProvisioningService:
    """Just-in-Time user provisioning for SSO."""

    async def provision_user(
        self,
        user_info: dict,
        org_id: str,
        groups: List[str]
    ) -> dict:
        """
        Create or update user account based on SSO login.

        Args:
            user_info: User information from IdP
            org_id: Organization ID
            groups: User's group memberships

        Returns:
            User object with assigned roles
        """
        email = user_info.get("email")

        # Check if user already exists
        user = await user_repository.get_user_by_email(email, org_id)

        if user:
            # Update existing user
            await self._update_user_attributes(user.id, user_info)
            await self._sync_user_groups(user.id, groups, org_id)

            await audit_log.log(
                action="jit_user_updated",
                user_id=user.id,
                details={"groups": groups}
            )
        else:
            # Create new user
            user = await self._create_new_user(user_info, org_id, groups)

            await audit_log.log(
                action="jit_user_created",
                user_id=user.id,
                details={"email": email, "groups": groups}
            )

        # Update last login timestamp
        await user_repository.update_last_login(user.id)

        return user

    async def _create_new_user(
        self,
        user_info: dict,
        org_id: str,
        groups: List[str]
    ):
        """Create new user account."""
        # Create user record
        user = await user_repository.create_user(
            email=user_info["email"],
            first_name=user_info.get("first_name"),
            last_name=user_info.get("last_name"),
            org_id=org_id,
            sso_enabled=True,
            email_verified=True,  # Trust IdP verification
            created_via="sso_jit"
        )

        # Assign roles based on groups
        mapper = GroupRoleMapper(org_id, await get_group_mapping_config(org_id))
        roles = await mapper.map_groups_to_roles(groups)

        for role in roles:
            await user_repository.assign_role(user.id, role)

        # Send welcome email
        if self.config.get("send_welcome_email"):
            await email_service.send_welcome_email(user.email, user.first_name)

        return user

    async def _update_user_attributes(
        self,
        user_id: str,
        user_info: dict
    ):
        """Update user attributes from IdP."""
        updates = {
            "first_name": user_info.get("first_name"),
            "last_name": user_info.get("last_name"),
            "department": user_info.get("department"),
            "title": user_info.get("title"),
        }

        # Remove None values
        updates = {k: v for k, v in updates.items() if v is not None}

        if updates:
            await user_repository.update_user(user_id, updates)

    async def _sync_user_groups(
        self,
        user_id: str,
        groups: List[str],
        org_id: str
    ):
        """Synchronize user roles based on current groups."""
        mapper = GroupRoleMapper(org_id, await get_group_mapping_config(org_id))
        await mapper.sync_user_roles(user_id, groups)
```

---

## Troubleshooting Common Issues

### SAML Issues

**Issue: "SAML Response Signature Validation Failed"**

**Causes:**
- Certificate mismatch between IdP and a11i configuration
- Expired IdP certificate
- Clock skew between systems

**Solutions:**
1. Download latest IdP metadata and re-upload to a11i
2. Verify certificate in a11i matches IdP certificate
3. Check system clocks are synchronized (NTP)
4. Ensure signature algorithm matches (RS256 recommended)

---

**Issue: "SAML Assertion Not Accepted" (Invalid Audience)**

**Cause:** Entity ID mismatch

**Solution:**
1. Verify a11i Entity ID: `https://a11i.dev/saml/metadata`
2. Check IdP configuration has correct audience/entity ID
3. Ensure no trailing slashes or URL differences

---

**Issue: "User Attributes Not Populating"**

**Cause:** Attribute mapping misconfiguration

**Solution:**
1. Check IdP attribute statement configuration
2. Verify attribute names match a11i expectations:
   - `email` → user email
   - `firstName` → first name
   - `lastName` → last name
   - `groups` → group memberships
3. Test with SAML tracer browser extension to inspect assertion

---

### OIDC Issues

**Issue: "Invalid Redirect URI"**

**Cause:** Redirect URI mismatch

**Solution:**
1. Verify redirect URI in IdP matches exactly: `https://a11i.dev/auth/oidc/callback`
2. Check for trailing slashes
3. Ensure HTTPS is used (not HTTP)

---

**Issue: "ID Token Validation Failed"**

**Causes:**
- Issuer mismatch
- Audience mismatch
- Expired token
- Invalid signature

**Solutions:**
1. Verify issuer URL matches IdP configuration
2. Check client ID matches in token audience claim
3. Ensure clock synchronization
4. Validate JWKS endpoint is accessible

---

**Issue: "Groups Not Included in Token"**

**Cause:** Missing scope or custom claim configuration

**Solution:**
1. Ensure `groups` scope is requested
2. For Auth0: Configure rule to add groups to token
3. For Okta: Add groups claim to ID token
4. For Azure AD: Configure group claims in token configuration

---

### SCIM Issues

**Issue: "SCIM API Authentication Failed"**

**Solution:**
1. Verify SCIM token is correct
2. Check authorization header format: `Bearer <token>`
3. Regenerate SCIM token if necessary
4. Verify token hasn't expired

---

**Issue: "Users Not Provisioning Automatically"**

**Solutions:**
1. Check SCIM provisioning is enabled in IdP
2. Verify users are assigned to application in IdP
3. Check SCIM sync logs in IdP for errors
4. Test SCIM API manually with Postman:
   ```bash
   curl -X POST https://a11i.dev/scim/v2/Users \
     -H "Authorization: Bearer <SCIM_TOKEN>" \
     -H "Content-Type: application/scim+json" \
     -d '{
       "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
       "userName": "test@company.com",
       "emails": [{"value": "test@company.com", "primary": true}],
       "name": {"givenName": "Test", "familyName": "User"}
     }'
   ```

---

**Issue: "Group Membership Not Syncing"**

**Solution:**
1. Verify IdP is configured to push groups
2. Check group attribute mapping in SCIM configuration
3. Ensure groups exist in a11i or are created via SCIM
4. Review SCIM sync logs for group-related errors

---

### General SSO Issues

**Issue: "Infinite Redirect Loop"**

**Causes:**
- Session cookie issues
- Incorrect SSO configuration
- Browser privacy settings

**Solutions:**
1. Clear browser cookies and cache
2. Try incognito/private browsing mode
3. Check browser cookie settings allow third-party cookies
4. Verify SSO configuration in a11i is correct

---

**Issue: "Users See 'Access Denied' After SSO Login"**

**Cause:** No role assignments

**Solution:**
1. Verify group-to-role mappings are configured
2. Check user's groups in IdP
3. Ensure default role is configured for JIT provisioning
4. Manually assign user to at least one role in a11i

---

## Key Takeaways

> **SSO Integration Summary**
>
> **Supported Protocols:**
> - SAML 2.0 for traditional enterprise IdPs (Okta, Azure AD, OneLogin, Ping Identity)
> - OIDC/OAuth for modern cloud-based IdPs (Auth0, Okta, Google Workspace, Azure AD)
> - SCIM 2.0 for automated user provisioning and deprovisioning
>
> **Key Features:**
> - Just-in-Time (JIT) provisioning - automatic user account creation on first login
> - Group-to-role mapping - automatic role assignment based on IdP groups
> - Attribute synchronization - keep user profiles updated from IdP
> - Automated deprovisioning - disable access when users leave organization
> - Comprehensive audit logging - track all SSO authentication events
>
> **Configuration Steps:**
> 1. Choose protocol (SAML or OIDC) based on your IdP
> 2. Configure application in your identity provider
> 3. Exchange metadata/credentials between IdP and a11i
> 4. Set up group-to-role mappings
> 5. Enable SCIM provisioning (optional but recommended)
> 6. Test SSO flow end-to-end
> 7. Roll out to users gradually
>
> **Best Practices:**
> - Use SCIM for automated provisioning when available
> - Configure group-based access instead of individual user assignments
> - Set up default roles for new users via JIT provisioning
> - Test thoroughly in staging before production rollout
> - Monitor audit logs for failed authentication attempts
> - Keep IdP certificates up to date
> - Document your group-to-role mappings

**Certified Integrations:**
- ✓ Okta (SAML, OIDC, SCIM)
- ✓ Azure Active Directory (SAML, OIDC, SCIM)
- ✓ Google Workspace (SAML, OIDC)

**Need Help?**
- Enterprise customers: Contact your Technical Account Manager
- Email: enterprise-support@a11i.dev
- Documentation: https://docs.a11i.dev/enterprise/sso

---

**Related Documentation:**
- [Enterprise Features Overview](./enterprise-features.md)
- [RBAC and Authentication](../05-security-compliance/rbac-auth.md)
- [Audit Logging](./audit-logging.md)
- [Compliance Framework](../05-security-compliance/compliance-framework.md)

---

*Document Status: Draft | Last Updated: 2025-11-26 | Maintained by: Enterprise Solutions Team*
