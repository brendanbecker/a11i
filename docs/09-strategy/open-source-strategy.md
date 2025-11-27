---
title: "Open-Source Strategy"
category: "Strategy"
tags: ["open-source", "licensing", "community", "AGPL", "open-core", "governance"]
status: "Draft"
last_updated: "2025-11-26"
related_docs:
  - "go-to-market.md"
  - "pricing-model.md"
  - "../01-overview/competitive-landscape.md"
---

# Open-Source Strategy

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [License Recommendation](#license-recommendation)
3. [Open vs Proprietary Feature Split](#open-vs-proprietary-feature-split)
4. [Community Building Strategies](#community-building-strategies)
5. [Governance Model](#governance-model)
6. [CNCF and Foundation Considerations](#cncf-and-foundation-considerations)
7. [Dual Licensing Discussion](#dual-licensing-discussion)
8. [Key Takeaways](#key-takeaways)

## Executive Summary

a11i's open-source strategy balances **community growth and trust** with **sustainable commercial monetization**. The recommended approach is an **open-core model** with AGPL licensing for the core platform and proprietary enterprise add-ons.

### Strategic Rationale

**Why Open-Source?**
1. **Developer Trust**: Developers prefer open-source for infrastructure tools (no black boxes)
2. **Community Innovation**: Contributions accelerate development and ecosystem growth
3. **Market Validation**: Langfuse (10K stars) and Phoenix (7.8K stars) prove OSS demand
4. **Competitive Differentiation**: vs. proprietary LangSmith and closed Datadog/Dynatrace
5. **Standards Leadership**: Contribute to OpenTelemetry semantic conventions for AI

**Why AGPL Over MIT/Apache?**
1. **SaaS Protection**: Prevents cloud providers from offering a11i as competing SaaS without contribution
2. **Revenue Preservation**: Forces commercial users to either self-host or buy our cloud offering
3. **Community Alignment**: Ensures derivative works remain open-source
4. **Precedent**: Successful with GitLab, Grafana, TimescaleDB

### Model Summary

```
┌─────────────────────────────────────────┐
│        Open-Source Core (AGPL)          │
│  - Core tracing & metrics               │
│  - Basic dashboards                     │
│  - CLI tools                            │
│  - Python/TS SDKs                       │
│  - Single-tenant deployment             │
└─────────────────────────────────────────┘
              │
              │ Used by
              ▼
┌─────────────────────────────────────────┐
│    Commercial Offerings (Proprietary)   │
│  - Multi-tenant SaaS platform           │
│  - Advanced RBAC & SSO                  │
│  - Enterprise compliance features       │
│  - Priority support & SLAs              │
│  - Advanced analytics & ML insights     │
└─────────────────────────────────────────┘
```

**Revenue Model**: Capture ~5-10% of value created through premium features and managed cloud service

## License Recommendation

### Primary License: AGPL v3 (GNU Affero General Public License)

**Chosen License**: **AGPL v3** for core platform

**Rationale**:

| Criterion | AGPL v3 Advantage |
|-----------|------------------|
| **SaaS Protection** | Requires source disclosure even for network-delivered services |
| **Community Copyleft** | Derivative works must remain open-source |
| **Commercial Viability** | Forces commercial SaaS users to buy license or contribute |
| **Developer Trust** | OSI-approved, widely understood |
| **Precedent** | Grafana, GitLab, MongoDB, TimescaleDB successfully use AGPL |

**How AGPL Works**:

```
IF company uses a11i (AGPL) AND
   company offers it as a SaaS service THEN
   company must open-source their modifications
OR
   company must purchase commercial license from a11i
```

**Example Scenarios**:

| Scenario | AGPL Requirement | Commercial Impact |
|----------|-----------------|-------------------|
| **Self-hosted, internal use** | No source disclosure needed | ✅ Free to use |
| **Self-hosted, modifications** | Modifications must stay internal or be open-sourced | ✅ Free to use |
| **Offer as SaaS to others** | Must open-source all modifications OR buy license | 💰 Revenue opportunity |
| **Distribute modified binary** | Must provide source code to recipients | ✅ Community contribution |

---

### Alternative Considered: Apache 2.0 or MIT

**Why Not Apache 2.0/MIT?**

| Factor | Apache 2.0/MIT | AGPL v3 | Winner |
|--------|---------------|---------|--------|
| **Permissiveness** | Maximum (no restrictions) | Copyleft (must share changes) | - |
| **SaaS Protection** | None (AWS can fork and compete) | Strong (SaaS usage triggers sharing) | **AGPL** |
| **Corporate Adoption** | Easier (fewer legal concerns) | Harder (legal review needed) | Apache |
| **Community Growth** | Faster initial growth | Slower but more aligned | Draw |
| **Revenue Protection** | Weak (easy to undercut) | Strong (forces paid license) | **AGPL** |
| **Developer Perception** | "Truly open" | "Defensively open" | Apache |

**Decision**: Revenue protection and SaaS defense outweigh ease of adoption. Mitigate corporate concerns with clear documentation and dual licensing option.

---

### Dual Licensing Option (Future Consideration)

**What**: Offer both AGPL (default) and commercial license (paid)

**When to Offer Commercial License**:
- Enterprise customers who want proprietary modifications without disclosure
- SaaS providers who want to offer a11i to their customers
- Companies with corporate policies against AGPL

**Pricing**: Commercial license = $50K-$200K/year + perpetual source access

**Precedent**: MongoDB (SSPL + Commercial), MySQL (GPL + Commercial), Qt (GPL + Commercial)

---

### Component Licensing Breakdown

| Component | License | Rationale |
|-----------|---------|-----------|
| **Core Platform** (collector, storage, query engine) | **AGPL v3** | SaaS protection for main value |
| **SDKs** (Python, TypeScript, Go clients) | **Apache 2.0** | Maximize adoption, no SaaS risk |
| **CLI Tools** | **AGPL v3** | Part of core experience |
| **Documentation** | **Creative Commons BY-SA 4.0** | Share-alike for community contributions |
| **Example Code/Tutorials** | **MIT** | Zero friction for learning |
| **Enterprise Add-Ons** | **Proprietary** | Revenue-generating features |

**Key Insight**: SDKs under Apache 2.0 removes barrier to instrumentation while core AGPL prevents SaaS competition.

## Open vs Proprietary Feature Split

### Core Principles

1. **Open-Source Core**: Everything needed for production single-tenant deployment
2. **Proprietary Premium**: Multi-tenancy, advanced enterprise features, managed cloud
3. **Value Capture**: Target 5-10% of users upgrading to commercial offering

---

### Open-Source Features (AGPL)

**Instrumentation & Collection**:
- ✅ OpenTelemetry collector configuration
- ✅ Python, TypeScript, Go SDKs
- ✅ Framework integrations (LangChain, CrewAI, AutoGen, LlamaIndex)
- ✅ Auto-instrumentation libraries
- ✅ Proxy/sidecar mode

**Core Metrics & Tracing**:
- ✅ Five core metrics (token usage, cost, context saturation, tool errors, loop velocity)
- ✅ Trace collection and storage (ClickHouse)
- ✅ Span reconstruction and visualization
- ✅ Basic cost attribution (single tenant)

**Dashboards & Visualization**:
- ✅ Agent performance dashboard
- ✅ Cost tracking dashboard
- ✅ Context saturation monitoring
- ✅ Tool reliability dashboard
- ✅ Trace viewer with flamegraphs

**Alerting & Monitoring**:
- ✅ Threshold-based alerts
- ✅ Webhook notifications
- ✅ Email alerts
- ✅ Basic anomaly detection (statistical)

**Deployment & Operations**:
- ✅ Docker Compose deployment
- ✅ Kubernetes Helm charts
- ✅ Single-tenant architecture
- ✅ Data retention policies
- ✅ Backup and restore

**Developer Tools**:
- ✅ CLI for trace querying
- ✅ Local development mode
- ✅ Testing utilities
- ✅ Debug mode

---

### Proprietary Features (Commercial License)

**Multi-Tenancy & Isolation**:
- 💰 Multi-tenant data isolation (row-level security)
- 💰 Per-tenant quotas and rate limiting
- 💰 Cross-tenant cost allocation
- 💰 Tenant-specific retention policies
- 💰 White-label/custom branding

**Enterprise Authentication & Authorization**:
- 💰 SSO integration (Okta, Azure AD, Google Workspace)
- 💰 SAML 2.0 and OIDC support
- 💰 Advanced RBAC (custom roles, fine-grained permissions)
- 💰 SCIM provisioning
- 💰 Audit logging (SOC2/HIPAA compliant)

**Advanced Analytics & Insights**:
- 💰 ML-powered anomaly detection
- 💰 Automatic root cause analysis
- 💰 Cost optimization recommendations (ML-driven)
- 💰 Prompt efficiency analysis
- 💰 A/B testing framework for agent changes
- 💰 Predictive cost forecasting

**Compliance & Security**:
- 💰 HIPAA-ready deployment configurations
- 💰 SOC2 Type II compliance tooling
- 💰 PII detection and redaction (advanced)
- 💰 Data residency controls (geographic)
- 💰 Encryption key management (BYOK)

**Managed Cloud Platform**:
- 💰 Fully managed SaaS offering
- 💰 99.9% uptime SLA
- 💰 Automatic scaling and maintenance
- 💰 Priority support (24/7)
- 💰 Professional services and onboarding

**Enterprise Integrations**:
- 💰 Salesforce integration (usage → CRM)
- 💰 Jira/ServiceNow incident management
- 💰 Custom API integrations
- 💰 Enterprise data export (S3, BigQuery)

---

### Feature Split Decision Matrix

**When to keep a feature open-source**:
- ✅ Core to basic observability functionality
- ✅ Needed for single-tenant production deployment
- ✅ Benefits from community contributions (e.g., framework integrations)
- ✅ Competitive differentiator vs. proprietary competitors
- ✅ Enables ecosystem growth

**When to make a feature proprietary**:
- 💰 Enterprise-specific requirement (SSO, advanced RBAC)
- 💰 Multi-tenancy complexity (SaaS platform infrastructure)
- 💰 Advanced ML/AI capabilities (significant R&D investment)
- 💰 Compliance tooling (niche regulatory requirements)
- 💰 White-glove support and SLAs

---

### Competitive Benchmarking

| Feature | a11i (AGPL) | Langfuse (MIT) | Phoenix (MIT) | LangSmith (Proprietary) |
|---------|-------------|----------------|---------------|------------------------|
| **Core Tracing** | ✅ OSS | ✅ OSS | ✅ OSS | ❌ Proprietary |
| **Dashboards** | ✅ OSS | ✅ OSS | ✅ OSS | ❌ Proprietary |
| **Multi-Tenancy** | 💰 Paid | 💰 Paid (Enterprise) | ❌ None | ❌ Proprietary |
| **SSO/RBAC** | 💰 Paid | 💰 Paid (Enterprise) | ❌ None | ❌ Proprietary |
| **Self-Hosted** | ✅ OSS | ✅ OSS | ✅ OSS | 💰 Enterprise only |
| **Cloud Offering** | 💰 Paid SaaS | 💰 Paid SaaS | Roadmap | 💰 Paid SaaS |

**Insight**: a11i matches or exceeds OSS competitors while maintaining AGPL protection.

## Community Building Strategies

### 1. Documentation Excellence

**Philosophy**: Documentation is marketing for developers.

**Components**:

| Docs Type | Description | Update Frequency |
|-----------|-------------|-----------------|
| **Quick Start** | 5-minute setup to first trace | Every release |
| **Framework Guides** | LangChain, CrewAI, AutoGen integrations | Monthly |
| **API Reference** | Auto-generated from code | Continuous |
| **Tutorials** | Step-by-step walkthroughs | Bi-weekly |
| **Best Practices** | Agent observability patterns | Monthly |
| **Troubleshooting** | Common issues + solutions | As needed |

**Tools**:
- Docusaurus or GitBook for hosting
- Algolia for search
- Vercel for deployment
- GitHub for version control

**Success Metrics**:
- Tutorial completion rate >70%
- Avg. time to first successful trace <5 minutes
- Documentation NPS >60

**Investment**: $120K/year (2 technical writers, tooling)

---

### 2. Discord/Slack Community

**Platform**: Discord (better for real-time, community-owned feeling)

**Channels**:
```
📢 #announcements          - Product updates, releases
💬 #general                - General discussion
🆘 #help                   - Technical support
🛠 #integrations          - Framework-specific help
💡 #feature-requests      - Community suggestions
🐛 #bug-reports           - Issue tracking
👨‍💻 #contributors          - OSS contribution coordination
🎉 #showcase              - User success stories
🗳 #off-topic             - Community building
```

**Engagement Strategy**:
- **Office Hours**: Weekly 30-min Q&A with founders/engineers
- **Response SLA**: <2 hours during business hours, <24h weekends
- **Contributor Recognition**: Monthly spotlight on top contributors
- **Events**: Virtual meetups, hackathons, demo days

**Moderation**:
- Code of Conduct (Contributor Covenant)
- 2+ community moderators
- Automated spam/abuse detection

**Success Metrics**:
- Monthly active users: 1,000+ by Year 1
- Avg. first response time: <30 minutes
- Community health score (Discord native): >85%

**Investment**: $80K/year (community manager, moderator stipends)

---

### 3. Conference Presence

**Strategy**: Establish thought leadership in AI observability space

**Target Conferences**:

| Conference | Tier | Audience | Activity | Investment |
|------------|------|----------|----------|------------|
| **AI Engineer Summit** | Tier 1 | AI engineers | Speaking + Booth | $30K |
| **KubeCon + CloudNativeCon** | Tier 1 | Platform engineers | Speaking + Booth | $40K |
| **PyCon** | Tier 2 | Python developers | Speaking | $8K |
| **OpenTelemetry Community Day** | Tier 1 | OTel community | Sponsorship + Talk | $15K |
| **Re:Invent (AWS)** | Tier 2 | Enterprise | Booth (later stage) | $50K (Year 2+) |

**Speaking Topics**:
- "Agent-Native Observability: Beyond LLM Tracing"
- "Cost Optimization for Production AI Agents"
- "OpenTelemetry for AI: Lessons Learned"
- "From Prototype to Production: Making AI Agents Observable"

**Content Strategy**:
- Submit CFPs 6 months in advance
- Repurpose talks into blog posts and videos
- Live tweet key insights
- Record talks for YouTube

**Success Metrics**:
- 8+ conference talks Year 1, 20+ Year 2
- 500+ booth conversations per event
- 50+ qualified leads per Tier 1 conference

**Investment**: $150K/year (travel, sponsorships, booth)

---

### 4. Contributor Program

**Goal**: Convert users into contributors, create sustainable ecosystem

**Contribution Pathways**:

```
Level 1: First Contribution
├─ Good First Issue (code)
├─ Documentation improvement
└─ Bug report with reproduction

Level 2: Regular Contributor
├─ 5+ merged PRs
├─ Framework integration
└─ Community support in Discord

Level 3: Core Contributor
├─ 25+ merged PRs
├─ Feature development
├─ Code review participation
└─ Community leadership

Level 4: Maintainer
├─ Commit access
├─ Release management
├─ Strategic input
└─ Paid contractor/full-time
```

**Incentives**:

| Level | Benefits |
|-------|----------|
| **Level 1** | GitHub badge, community recognition |
| **Level 2** | a11i swag pack, priority support |
| **Level 3** | Conference travel sponsorship, co-author blog posts |
| **Level 4** | Paid consulting, full-time employment consideration |

**Support Mechanisms**:
- Bi-weekly contributor sync calls
- Dedicated #contributors Discord channel
- Contribution guide with clear setup instructions
- Video walkthroughs of codebase architecture
- 1:1 pairing sessions for complex contributions

**Success Metrics**:
- 50+ contributors Year 1, 150+ Year 2
- 10+ regular contributors (>5 PRs/year)
- 2+ framework integrations from community
- 30% of PRs from external contributors

**Investment**: $60K/year (swag, travel sponsorships, contributor perks)

---

### 5. Content Marketing (SEO-Driven)

**Strategy**: Rank #1 for "AI agent observability" and related keywords

**Content Pillars**:

1. **Technical Tutorials** (SEO + education)
   - "Monitoring LangChain Agents with OpenTelemetry"
   - "Debugging CrewAI Multi-Agent Systems"
   - "Cost Optimization for GPT-4 Agents"

2. **Thought Leadership** (positioning)
   - "The Five Pillars of Agent Observability"
   - "Why Traditional APM Fails for AI Agents"
   - "The Hidden Costs of Context Window Mismanagement"

3. **Case Studies** (social proof)
   - "How [Customer] Reduced AI Costs 40% with a11i"
   - "[Startup] Ships Agents 3x Faster with Agent-Native Tracing"

4. **Comparison Content** (competitive)
   - "a11i vs LangSmith: Which is Right for You?"
   - "Open-Source AI Observability: Langfuse vs Phoenix vs a11i"

**Distribution Channels**:
- Company blog (SEO-optimized)
- Dev.to (developer audience)
- Medium (cross-posting)
- Hacker News (strategic timing)
- Reddit (r/MachineLearning, r/LangChain)
- Twitter/X (founder + company accounts)

**Publishing Cadence**:
- 2x technical tutorials per week
- 1x thought leadership per month
- 1x case study per quarter
- 1x comparison piece per quarter

**Success Metrics**:
- 50K monthly blog visitors Year 1
- 10+ top-10 Google rankings for target keywords
- 500+ newsletter subscribers
- 20% blog-to-signup conversion rate

**Investment**: $100K/year (writers, SEO tools, design)

---

### 6. Open-Source Program Office (OSPO)

**Purpose**: Manage community health, contributions, and ecosystem growth

**Responsibilities**:
- Triage issues and PRs
- Coordinate releases
- Manage security disclosures
- Track community metrics
- Facilitate contributor onboarding

**Team Structure**:
- 1 OSPO Lead (full-time)
- 2 Developer Advocates (full-time)
- 5+ volunteer moderators (community)

**Policies**:
- **Code of Conduct**: Contributor Covenant
- **Security Policy**: Coordinated disclosure, 90-day embargo
- **Contribution License Agreement (CLA)**: Required for code contributions
- **Release Cadence**: Monthly minor releases, quarterly major releases
- **Deprecation Policy**: 6-month notice for breaking changes

**Tools**:
- GitHub Insights for contribution tracking
- CHAOSS metrics for community health
- All Contributors bot for recognition
- Probot for automation (stale issues, PR checks)

**Success Metrics**:
- Issue triage time: <48 hours
- PR review time: <72 hours
- Security response time: <24 hours critical, <7 days others
- Community health score: >80/100 (CHAOSS)

**Investment**: $200K/year (OSPO Lead + Developer Advocates)

## Governance Model

### Phase 1: Benevolent Dictatorship (Year 1-2)

**Model**: Founder-led with community input

**Decision-Making**:
- Founders make final calls on roadmap and architecture
- Community input via RFCs (Request for Comments)
- Transparent decision rationale

**Rationale**: Speed and decisive direction critical in early stage

---

### Phase 2: Tiered Governance (Year 2-3)

**Model**: Contributor tiers with voting rights

**Structure**:

```
┌──────────────────────────┐
│   Steering Committee     │  (Strategic direction)
│   - Founders (2)         │
│   - Maintainers (3)      │
│   - Community Reps (2)   │
└──────────────────────────┘
           │
           ├─────────────────────────┐
           │                         │
┌──────────▼──────────┐   ┌─────────▼──────────┐
│  Technical Committee│   │  Community Council │
│  - Core Maintainers │   │  - Top Contributors│
│  - Security Team    │   │  - User Reps       │
│  - Architecture     │   │  - Moderators      │
└─────────────────────┘   └────────────────────┘
```

**Voting Rights**:
- **Steering Committee**: Major decisions (licensing, foundation donation)
- **Technical Committee**: Architecture, breaking changes, security
- **Community Council**: Community initiatives, code of conduct enforcement

---

### Phase 3: Foundation Governance (Year 3+)

**Model**: Donate to CNCF or form independent foundation

**See**: [CNCF and Foundation Considerations](#cncf-and-foundation-considerations)

---

### RFC Process (Request for Comments)

**Purpose**: Community input on major changes

**When to Use RFC**:
- Breaking API changes
- New architectural components
- Significant feature additions
- Changes to governance or licensing

**Process**:
```
1. Draft RFC (GitHub issue with template)
2. Community comment period (14 days)
3. Technical Committee review
4. Acceptance or rejection (with rationale)
5. Implementation (if accepted)
```

**Example RFCs**:
- "RFC-001: Agent Workflow Semantic Conventions"
- "RFC-012: Multi-Tenant Data Isolation Architecture"
- "RFC-023: Governance Transition to CNCF"

---

### Community Representation

**Mechanisms**:
- Quarterly community surveys
- User advisory board (invite top contributors)
- Public roadmap voting (feature prioritization)
- Office hours for direct feedback

**Success Metrics**:
- Survey response rate: >30%
- Advisory board participation: 8+ members
- Roadmap voting participation: 100+ community members

## CNCF and Foundation Considerations

### Why CNCF (Cloud Native Computing Foundation)?

**Advantages**:

| Benefit | Impact |
|---------|--------|
| **Credibility** | CNCF badge signals maturity and neutrality |
| **Community** | Access to CNCF's 800+ member organizations |
| **Marketing** | Promotion through CNCF channels (blog, events, social) |
| **Legal Support** | CNCF handles trademark, licensing, security disclosures |
| **Sustainability** | Reduces burden on founding company for OSS maintenance |
| **Ecosystem** | Integration with Kubernetes, Prometheus, OpenTelemetry |

**Requirements for CNCF Donation**:

| Level | Criteria | Timeline |
|-------|----------|----------|
| **Sandbox** | - Demonstrable value<br>- Active development<br>- 2+ contributors | Year 1 (possible) |
| **Incubating** | - Production use (3+ orgs)<br>- Healthy community<br>- Security audit | Year 2-3 |
| **Graduated** | - Enterprise adoption<br>- Documented governance<br>- Committer diversity | Year 4+ |

---

### CNCF Donation Timeline

**Year 1 (Month 12)**: Apply for CNCF Sandbox
- Requirements:
  - ✅ 10K+ GitHub stars
  - ✅ 50+ contributors
  - ✅ Production deployments documented
  - ✅ Code of Conduct and governance docs
- Benefits:
  - CNCF Sandbox badge
  - Promotion at KubeCon
  - Access to CNCF community infrastructure

**Year 2 (Month 24)**: Aim for CNCF Incubating
- Requirements:
  - ✅ 3+ production companies (documented)
  - ✅ Security audit (CNCF provides)
  - ✅ 100+ contributors
  - ✅ Formal governance structure
- Benefits:
  - Increased visibility
  - Vendor-neutral perception
  - CNCF-sponsored security audit

**Year 3+**: Target CNCF Graduated (optional)
- Requirements:
  - ✅ Widespread adoption (50+ production companies)
  - ✅ Committer diversity (10+ organizations contributing)
  - ✅ Documented governance and sustainability
- Benefits:
  - Tier 1 CNCF project status
  - Maximum ecosystem leverage
  - Long-term sustainability signal

---

### Alternative: Independent Foundation

**When to Consider**:
- CNCF doesn't accept project (niche focus)
- Want more control over governance
- Building broader ecosystem beyond cloud-native

**Model**: a11i Foundation (independent 501(c)(6))

**Structure**:
```
Board of Directors
├─ a11i Inc. Representative (2 seats)
├─ Platinum Sponsors (2 seats)
├─ Community-Elected (2 seats)
└─ Independent Advisors (1 seat)

Technical Steering Committee
├─ Maintainers (elected)
└─ Special Interest Groups
    ├─ SIG Integrations
    ├─ SIG Security
    └─ SIG Enterprise
```

**Funding**:
- Sponsorships (Platinum: $100K/year, Gold: $50K, Silver: $25K)
- a11i Inc. contribution ($500K/year)
- Conference revenue

**Precedent**: OpenJS Foundation, Eclipse Foundation, Apache Foundation

---

### Decision Framework

| Factor | CNCF | Independent Foundation |
|--------|------|----------------------|
| **Credibility** | ⭐⭐⭐⭐⭐ (highest) | ⭐⭐⭐ (earned over time) |
| **Control** | ⭐⭐ (CNCF governance) | ⭐⭐⭐⭐⭐ (full control) |
| **Marketing** | ⭐⭐⭐⭐⭐ (CNCF reach) | ⭐⭐ (DIY) |
| **Cost** | ⭐⭐⭐⭐ (CNCF covers legal) | ⭐⭐ ($200K+ annually) |
| **Ecosystem** | ⭐⭐⭐⭐⭐ (Kubernetes, etc.) | ⭐⭐⭐ (build from scratch) |
| **Speed** | ⭐⭐⭐ (approval process) | ⭐⭐⭐⭐⭐ (immediate) |

**Recommendation**: **Target CNCF Sandbox (Year 1)** for credibility and ecosystem leverage.

## Dual Licensing Discussion

### What is Dual Licensing?

**Definition**: Offer same software under two licenses:
1. **Open-Source License** (AGPL): Free for compliant users
2. **Commercial License**: Paid for users who need proprietary rights

**When Users Choose Commercial License**:
- Want to make proprietary modifications without disclosing
- Offer a11i as SaaS to their customers (avoids AGPL network clause)
- Corporate policy prohibits AGPL usage
- Need indemnification and warranties

---

### Commercial License Terms

**What It Grants**:
- ✅ Right to modify source code without disclosure
- ✅ Right to offer as SaaS to third parties
- ✅ Indemnification from patent claims
- ✅ No copyleft obligations
- ✅ Priority support and bug fixes

**Pricing Model**:

| Tier | Price | Use Case |
|------|-------|----------|
| **Startup** | $25K/year | <$10M ARR, single product |
| **Business** | $75K/year | $10M-$100M ARR, multiple products |
| **Enterprise** | $150K+/year | $100M+ ARR, custom terms |

**Contract Terms**:
- Perpetual license (not subscription)
- Source access included
- Upgrade rights for 1 year
- Named commercial support contacts

---

### Revenue Projections

**Assumption**: 2% of self-hosted users choose commercial license

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| Self-Hosted Users | 1,000 | 5,000 | 15,000 |
| Commercial Licenses (2%) | 20 | 100 | 300 |
| Avg License Price | $50K | $60K | $75K |
| **Commercial License Revenue** | **$1M** | **$6M** | **$22.5M** |

**Insight**: Dual licensing can become significant revenue stream by Year 2-3.

---

### Legal Considerations

**CLA (Contributor License Agreement) Required**:
- Contributors grant a11i Inc. rights to relicense contributions
- Enables dual licensing model
- Standard practice (MySQL, MongoDB, Qt)

**CLA Template**:
```
By contributing code, you grant a11i Inc.:
1. Perpetual, worldwide, non-exclusive license to your contributions
2. Right to sublicense under different terms (commercial license)
3. Patent license for contributions

You retain copyright and can use your contributions elsewhere.
```

**Enforcement**:
- CLA Assistant bot (GitHub integration)
- Blocks PRs without signed CLA
- One-time signature (via DocuSign or GitHub)

---

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **Community Perception** | Transparent communication: "AGPL protects OSS community, commercial license funds development" |
| **CLA Friction** | Clear explanation, streamlined signing process |
| **Legal Complexity** | Engage OSS-friendly law firm (Cooley, Fenwick, Helix Law) |
| **Dual Maintenance** | Single codebase, different licensing metadata |

## Key Takeaways

> **Open-Source Strategy Summary**
>
> 1. **AGPL v3 Core License**: Protects against SaaS competition while enabling free self-hosted use. SDKs under Apache 2.0 for maximum adoption.
>
> 2. **Open-Core Model**: Core platform (tracing, metrics, dashboards) is open-source. Enterprise features (multi-tenancy, SSO, advanced analytics) are proprietary.
>
> 3. **Community-First Growth**: Invest heavily in documentation, Discord community, conference presence, and contributor programs to build sustainable ecosystem.
>
> 4. **CNCF Pathway**: Target CNCF Sandbox by Month 12 for credibility and ecosystem leverage. Progress to Incubating by Year 2.
>
> 5. **Dual Licensing Option**: Offer commercial licenses for users who need proprietary rights or want to offer a11i as SaaS. Projected $1M+ revenue Year 1.
>
> 6. **Governance Evolution**: Start with benevolent dictatorship (Year 1), transition to tiered governance (Year 2), and consider foundation donation (Year 3+).
>
> 7. **Value Capture**: Target 5-10% of users upgrading to commercial offerings through premium features and managed cloud service.

**Critical Success Factors**:

| Factor | Importance | Status |
|--------|-----------|--------|
| **License Clarity** | Critical | ✅ AGPL + Apache for SDKs |
| **Documentation Quality** | Critical | 🔄 Ongoing investment |
| **Community Engagement** | High | 🔄 Discord, conferences, contributors |
| **CNCF Alignment** | High | 📅 Target Month 12 |
| **Feature Split** | Critical | ✅ Defined (OSS core vs proprietary) |
| **Dual Licensing** | Medium | 📅 Year 2+ consideration |

**Investment Summary**:

| Area | Annual Investment | ROI |
|------|------------------|-----|
| **Documentation** | $120K | Developer trust, reduced support burden |
| **Community Management** | $80K | Organic growth, contributor pipeline |
| **Conferences** | $150K | Brand awareness, lead generation |
| **Contributor Program** | $60K | Ecosystem expansion, feature development |
| **Content Marketing** | $100K | SEO traffic, inbound leads |
| **OSPO** | $200K | Community health, contribution velocity |
| **TOTAL** | **$710K/year** | 10K+ GitHub stars, 1,000+ active deployments |

**Competitive Advantages**:
- ✅ **AGPL Protection**: Prevents AWS/GCP from undercutting without contributing
- ✅ **Open Standards**: OTel-native builds trust vs. proprietary LangSmith
- ✅ **Community Velocity**: Active OSS community accelerates development
- ✅ **Ecosystem Leverage**: CNCF alignment opens enterprise doors

---

**Related Documentation:**
- [Go-to-Market Strategy](./go-to-market.md) - Customer acquisition and growth
- [Pricing Model](./pricing-model.md) - Revenue strategy and tiers
- [Competitive Landscape](../01-overview/competitive-landscape.md) - Market positioning

---

*Document Status: Draft | Last Updated: 2025-11-26 | Owner: Open-Source Strategy Team*
