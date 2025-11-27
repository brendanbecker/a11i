---
title: Provider Support
section: Implementation
weight: 43
description: Comprehensive guide to LLM provider integration, capabilities, and extensibility in a11i
keywords: [providers, openai, anthropic, azure, bedrock, vertex-ai, cohere, self-hosted, vllm, plugin-architecture]
related:
  - 04-implementation/observability.md
  - 04-implementation/token-accounting.md
  - 02-core-concepts/llm-abstraction.md
last_updated: 2025-11-26
---

# Provider Support

**a11i** supports multiple LLM providers through a pluggable architecture that normalizes different APIs, tokenization methods, and pricing models. This document covers supported providers, their capabilities, and how to extend support for new providers.

## Table of Contents

1. [Provider Capabilities Matrix](#provider-capabilities-matrix)
2. [Supported Providers](#supported-providers)
   - [OpenAI](#openai)
   - [Anthropic (Claude)](#anthropic-claude)
   - [Azure OpenAI](#azure-openai)
   - [AWS Bedrock](#aws-bedrock)
   - [Google Vertex AI](#google-vertex-ai)
   - [Cohere](#cohere)
   - [Self-Hosted Models](#self-hosted-models)
3. [Plugin Architecture](#plugin-architecture)
4. [Adding New Providers](#adding-new-providers)
5. [Provider-Specific Features](#provider-specific-features)
6. [Key Takeaways](#key-takeaways)

## Provider Capabilities Matrix

| Provider | Status | Tokenizer | Streaming | Cost Track | Special Features |
|----------|--------|-----------|-----------|------------|------------------|
| OpenAI | ✅ Full | tiktoken | ✅ SSE | ✅ | Function calling, JSON mode |
| Anthropic | ✅ Full | GPT-2 BPE | ✅ Custom | ✅ | Extended thinking |
| Azure OpenAI | ✅ Full | tiktoken | ✅ SSE | ✅ | Content filtering |
| AWS Bedrock | ✅ Full | Varies | ✅ | ✅ | Guardrails, knowledge bases |
| Google Vertex AI | 🔧 Partial | SentencePiece | ✅ | 🔧 | Grounding |
| Cohere | 🔧 Partial | BPE | ✅ | ✅ | RAG |
| Self-hosted (vLLM, TGI) | ✅ Full | Varies | ✅ | N/A | Custom models |

**Legend:**
- ✅ Full: Complete support with all features
- 🔧 Partial: Basic support, some features in development
- N/A: Not applicable for this provider type

## Supported Providers

### OpenAI

**Status:** ✅ Full Support

OpenAI provides GPT-4, GPT-4 Turbo, GPT-4o, and GPT-3.5 Turbo models through a standardized REST API.

#### Implementation

```python
class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    name = "openai"
    base_url = "https://api.openai.com/v1"

    # Model-specific tokenizers
    tokenizers = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
    }

    # Pricing per 1K tokens (USD)
    pricing = {
        "gpt-4o": {"input": 0.000003, "output": 0.000015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def parse_response(self, response: dict) -> LLMResponse:
        """Parse OpenAI API response into standardized format."""
        return LLMResponse(
            content=response["choices"][0]["message"]["content"],
            model=response["model"],
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
            finish_reason=response["choices"][0]["finish_reason"],
        )

    def parse_streaming_chunk(self, chunk: bytes) -> StreamChunk:
        """Parse Server-Sent Events (SSE) streaming chunks."""
        data = json.loads(chunk.decode().removeprefix("data: "))

        if data == "[DONE]":
            return StreamChunk(done=True)

        return StreamChunk(
            content=data["choices"][0]["delta"].get("content", ""),
            finish_reason=data["choices"][0].get("finish_reason"),
        )

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens using tiktoken."""
        import tiktoken
        encoding = tiktoken.get_encoding(self.tokenizers[model])
        return len(encoding.encode(text))
```

#### Key Features

- **Function Calling:** Structured tool/function invocation
- **JSON Mode:** Guaranteed valid JSON output
- **Vision Support:** Image input for GPT-4o and GPT-4 Turbo
- **Reproducible Outputs:** Seed parameter for deterministic generation

#### Configuration Example

```python
from a11i import A11I

client = A11I(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
    telemetry={"service.name": "my-service"}
)
```

### Anthropic (Claude)

**Status:** ✅ Full Support

Anthropic's Claude models offer extended context windows and advanced reasoning capabilities.

#### Implementation

```python
class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation."""

    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"

    # Claude uses GPT-2 style BPE tokenization (approximation)
    tokenizer = "gpt2"

    # Pricing per 1K tokens (USD)
    pricing = {
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def parse_response(self, response: dict) -> LLMResponse:
        """Parse Anthropic API response."""
        return LLMResponse(
            content=response["content"][0]["text"],
            model=response["model"],
            input_tokens=response["usage"]["input_tokens"],
            output_tokens=response["usage"]["output_tokens"],
            stop_reason=response["stop_reason"],
        )

    def parse_streaming_event(self, event: dict) -> StreamChunk:
        """Parse Anthropic's custom streaming event format."""
        if event["type"] == "content_block_delta":
            return StreamChunk(content=event["delta"]["text"])
        elif event["type"] == "message_stop":
            return StreamChunk(done=True)
        return StreamChunk()

    def add_auth_headers(self, headers: dict, api_key: str) -> dict:
        """Add Anthropic-specific authentication."""
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        return headers
```

#### Key Features

- **Extended Thinking:** Claude can show reasoning process
- **Long Context:** Up to 200K tokens for Claude 3 models
- **Vision Support:** Image analysis capabilities
- **System Prompts:** Dedicated system message support

#### Configuration Example

```python
client = A11I(
    provider="anthropic",
    api_key="sk-ant-...",
    model="claude-3-5-sonnet",
    max_tokens=4096  # Required for Anthropic
)
```

### Azure OpenAI

**Status:** ✅ Full Support

Azure OpenAI provides enterprise-grade OpenAI models with Microsoft's security and compliance features.

#### Implementation

```python
class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI Service provider implementation."""

    name = "azure.openai"

    def __init__(self, deployment_name: str, resource_name: str, api_version: str):
        """Initialize with Azure-specific configuration."""
        self.deployment_name = deployment_name
        self.resource_name = resource_name
        self.api_version = api_version
        self.base_url = (
            f"https://{resource_name}.openai.azure.com/openai/"
            f"deployments/{deployment_name}"
        )

    # Uses same tokenizers as OpenAI
    tokenizers = OpenAIProvider.tokenizers

    # Pricing varies by Azure tier (example: Standard tier)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-35-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def add_auth_headers(self, headers: dict, api_key: str) -> dict:
        """Add Azure API key authentication."""
        headers["api-key"] = api_key
        return headers

    def get_otel_attributes(self) -> dict:
        """Return Azure-specific OpenTelemetry attributes."""
        return {
            "gen_ai.system": "azure.openai",
            "azure.ai.deployment.name": self.deployment_name,
            "azure.ai.resource.name": self.resource_name,
        }
```

#### Key Features

- **Content Filtering:** Built-in content safety filters
- **Private Networking:** VNet integration for secure deployments
- **Regional Deployment:** Data residency compliance
- **Enterprise SSO:** Azure AD authentication integration

#### Configuration Example

```python
client = A11I(
    provider="azure.openai",
    deployment_name="gpt-4-deployment",
    resource_name="my-openai-resource",
    api_version="2024-02-01",
    api_key="..."  # Or use Azure AD token
)
```

### AWS Bedrock

**Status:** ✅ Full Support

AWS Bedrock provides access to multiple foundation models through a unified API with AWS security and compliance.

#### Implementation

```python
class BedrockProvider(LLMProvider):
    """AWS Bedrock multi-model provider implementation."""

    name = "aws.bedrock"

    # Bedrock hosts multiple model families
    model_mapping = {
        "anthropic.claude": AnthropicProvider,
        "amazon.titan": TitanProvider,
        "meta.llama": LlamaProvider,
        "cohere.command": CohereProvider,
    }

    # Pricing varies by model family
    pricing = {
        "anthropic.claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "amazon.titan-text": {"input": 0.0003, "output": 0.0004},
        "meta.llama3-70b": {"input": 0.00265, "output": 0.0035},
    }

    def get_provider_for_model(self, model_id: str) -> LLMProvider:
        """Route to appropriate provider based on model ID."""
        prefix = model_id.split(".")[0]
        provider_class = self.model_mapping.get(prefix, GenericProvider)
        return provider_class()

    def parse_response(self, response: dict, model_id: str) -> LLMResponse:
        """Delegate parsing to model-specific provider."""
        provider = self.get_provider_for_model(model_id)
        return provider.parse_response(response)

    def get_otel_attributes(self) -> dict:
        """Return Bedrock-specific telemetry attributes."""
        return {
            "gen_ai.system": "aws.bedrock",
            "aws.bedrock.guardrail.id": getattr(self, "guardrail_id", None),
            "aws.region": self.region,
        }
```

#### Key Features

- **Guardrails:** Content filtering and safety policies
- **Knowledge Bases:** RAG integration with managed vector databases
- **Model Evaluation:** Automated model performance testing
- **Custom Models:** Fine-tuning and continued pre-training

#### Configuration Example

```python
client = A11I(
    provider="aws.bedrock",
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    region="us-east-1",
    guardrail_id="ff6w9c0kqv7q"  # Optional
)
```

### Google Vertex AI

**Status:** 🔧 Partial Support

Google Vertex AI provides access to Gemini models with enterprise features. Cost tracking is in development.

#### Implementation

```python
class VertexAIProvider(LLMProvider):
    """Google Vertex AI provider implementation."""

    name = "google.vertex_ai"

    # Gemini uses SentencePiece tokenization
    tokenizer = "sentencepiece"

    # Pricing per 1K tokens (USD)
    pricing = {
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.00375},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    }

    def parse_response(self, response: dict) -> LLMResponse:
        """Parse Vertex AI response format."""
        return LLMResponse(
            content=response["candidates"][0]["content"]["parts"][0]["text"],
            model=response["model"],
            input_tokens=response["usageMetadata"]["promptTokenCount"],
            output_tokens=response["usageMetadata"]["candidatesTokenCount"],
        )

    def count_tokens(self, text: str, model: str) -> int:
        """Use Vertex AI's token counting API."""
        # Vertex AI provides a dedicated token counting endpoint
        # Implementation uses google-cloud-aiplatform SDK
        from google.cloud import aiplatform

        client = aiplatform.gapic.PredictionServiceClient()
        response = client.count_tokens(
            endpoint=f"projects/{self.project}/locations/{self.location}/models/{model}",
            contents=[{"role": "user", "parts": [{"text": text}]}]
        )
        return response.total_tokens

    def get_otel_attributes(self) -> dict:
        """Return Google Cloud-specific attributes."""
        return {
            "gen_ai.system": "google.vertex_ai",
            "gcp.project.id": self.project,
            "gcp.location": self.location,
        }
```

#### Key Features

- **Grounding:** Search and fact-checking integration
- **Safety Ratings:** Built-in content safety classification
- **Multimodal Input:** Text, image, video, and audio support
- **Model Garden:** Access to various open-source models

#### Configuration Example

```python
client = A11I(
    provider="google.vertex_ai",
    project="my-gcp-project",
    location="us-central1",
    model="gemini-1.5-pro"
)
```

### Cohere

**Status:** 🔧 Partial Support

Cohere provides language models optimized for enterprise applications and RAG workflows.

#### Implementation

```python
class CohereProvider(LLMProvider):
    """Cohere API provider implementation."""

    name = "cohere"
    base_url = "https://api.cohere.ai/v1"

    # Pricing per 1K tokens (USD)
    pricing = {
        "command-r-plus": {"input": 0.003, "output": 0.015},
        "command-r": {"input": 0.0005, "output": 0.0015},
    }

    def parse_response(self, response: dict) -> LLMResponse:
        """Parse Cohere response format."""
        return LLMResponse(
            content=response["text"],
            model=response["model"],
            input_tokens=response["meta"]["tokens"]["input_tokens"],
            output_tokens=response["meta"]["tokens"]["output_tokens"],
            finish_reason=response.get("finish_reason"),
        )

    def parse_streaming_chunk(self, chunk: bytes) -> StreamChunk:
        """Parse Cohere streaming format."""
        event = json.loads(chunk.decode())

        if event["event_type"] == "text-generation":
            return StreamChunk(content=event["text"])
        elif event["event_type"] == "stream-end":
            return StreamChunk(done=True)
        return StreamChunk()
```

#### Key Features

- **RAG Optimization:** Built-in retrieval-augmented generation
- **Reranking:** Document relevance reranking API
- **Embedding Models:** High-quality text embeddings
- **Multilingual Support:** Strong performance across languages

#### Configuration Example

```python
client = A11I(
    provider="cohere",
    api_key="...",
    model="command-r-plus"
)
```

### Self-Hosted Models

**Status:** ✅ Full Support

Support for self-hosted models via vLLM, Text Generation Inference (TGI), Ollama, and other OpenAI-compatible servers.

#### Implementation

```python
class SelfHostedProvider(LLMProvider):
    """Generic provider for self-hosted models."""

    name = "self_hosted"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        tokenizer_name: str = None
    ):
        """Initialize with custom endpoint and model configuration."""
        self.base_url = base_url
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer(tokenizer_name or model_name)

    def _load_tokenizer(self, name: str):
        """Load tokenizer from HuggingFace or local path."""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer {name}: {e}")
            return None

    # No pricing for self-hosted (or use custom pricing config)
    pricing = {}

    def count_tokens(self, text: str) -> int:
        """Count tokens using loaded tokenizer or fallback estimation."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: rough estimate (4 chars ≈ 1 token)
        return len(text) // 4

    def parse_response(self, response: dict) -> LLMResponse:
        """Parse OpenAI-compatible response format."""
        # Most self-hosted servers implement OpenAI API format
        return OpenAIProvider().parse_response(response)

    def get_otel_attributes(self) -> dict:
        """Return self-hosted specific attributes."""
        return {
            "gen_ai.system": "self_hosted",
            "llm.endpoint": self.base_url,
            "llm.model_name": self.model_name,
        }
```

#### Supported Backends

- **vLLM:** High-throughput LLM serving
- **Text Generation Inference (TGI):** HuggingFace's production server
- **Ollama:** Local model deployment
- **FastChat:** Multi-model serving platform
- **LocalAI:** OpenAI-compatible local API

#### Configuration Example

```python
# vLLM deployment
client = A11I(
    provider="self_hosted",
    base_url="http://localhost:8000",
    model_name="meta-llama/Llama-3-70b-instruct",
    tokenizer_name="meta-llama/Llama-3-70b-instruct"
)

# Ollama local deployment
client = A11I(
    provider="self_hosted",
    base_url="http://localhost:11434/v1",
    model_name="llama3:70b"
)
```

## Plugin Architecture

The provider system is built on an extensible plugin architecture that allows adding new providers without modifying core code.

### Base Provider Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None

@dataclass
class StreamChunk:
    """Streaming response chunk."""
    content: str = ""
    finish_reason: Optional[str] = None
    done: bool = False

class LLMProvider(ABC):
    """Abstract base class for LLM provider plugins."""

    # Provider identifier (e.g., "openai", "anthropic")
    name: str

    # API base URL
    base_url: str

    # Pricing information (USD per 1K tokens)
    pricing: Dict[str, Dict[str, float]] = {}

    @abstractmethod
    def parse_response(self, response: Dict[str, Any]) -> LLMResponse:
        """Parse provider-specific response into standardized format.

        Args:
            response: Raw API response dictionary

        Returns:
            Standardized LLMResponse object
        """
        pass

    @abstractmethod
    def parse_streaming_chunk(self, chunk: bytes) -> StreamChunk:
        """Parse streaming chunk into standardized format.

        Args:
            chunk: Raw streaming chunk bytes

        Returns:
            Standardized StreamChunk object
        """
        pass

    def count_tokens(self, text: str, model: str = None) -> int:
        """Count tokens using provider's tokenizer.

        Args:
            text: Input text to tokenize
            model: Model name (for model-specific tokenizers)

        Returns:
            Token count
        """
        # Default implementation: rough estimation
        return len(text) // 4

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> float:
        """Calculate cost based on token usage and pricing.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name for pricing lookup

        Returns:
            Cost in USD
        """
        rates = self.pricing.get(model, {"input": 0, "output": 0})
        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000

    def get_otel_attributes(self) -> Dict[str, Any]:
        """Return provider-specific OpenTelemetry attributes.

        Returns:
            Dictionary of semantic convention attributes
        """
        return {
            "gen_ai.system": self.name,
        }

    def add_auth_headers(self, headers: dict, api_key: str) -> dict:
        """Add authentication headers to request.

        Args:
            headers: Existing headers dictionary
            api_key: API key or token

        Returns:
            Updated headers dictionary
        """
        headers["Authorization"] = f"Bearer {api_key}"
        return headers
```

### Provider Registry

```python
# Global registry for dynamic provider loading
PROVIDER_REGISTRY: Dict[str, type[LLMProvider]] = {}

def register_provider(provider_class: type[LLMProvider]):
    """Register a provider plugin.

    Can be used as a decorator:

    @register_provider
    class MyCustomProvider(LLMProvider):
        name = "my_provider"
        ...
    """
    PROVIDER_REGISTRY[provider_class.name] = provider_class
    return provider_class

def get_provider(name: str, **kwargs) -> LLMProvider:
    """Get provider instance by name.

    Args:
        name: Provider identifier (e.g., "openai")
        **kwargs: Provider-specific initialization arguments

    Returns:
        Provider instance

    Raises:
        ValueError: If provider not found in registry
    """
    if name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider: {name}. "
            f"Available providers: {list(PROVIDER_REGISTRY.keys())}"
        )
    return PROVIDER_REGISTRY[name](**kwargs)

def list_providers() -> list[str]:
    """List all registered providers."""
    return list(PROVIDER_REGISTRY.keys())
```

## Adding New Providers

To add support for a new LLM provider:

### Step 1: Implement Provider Class

```python
from a11i.providers.base import LLMProvider, register_provider, LLMResponse, StreamChunk

@register_provider
class MyCustomProvider(LLMProvider):
    """Custom LLM provider implementation."""

    name = "my_custom_provider"
    base_url = "https://api.mycustomprovider.com/v1"

    # Define pricing
    pricing = {
        "my-model-v1": {"input": 0.001, "output": 0.002},
    }

    def parse_response(self, response: dict) -> LLMResponse:
        """Parse custom API response format."""
        return LLMResponse(
            content=response["output"]["text"],
            model=response["model_id"],
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
            finish_reason=response["stop_reason"],
        )

    def parse_streaming_chunk(self, chunk: bytes) -> StreamChunk:
        """Parse custom streaming format."""
        data = json.loads(chunk.decode())

        if data.get("done"):
            return StreamChunk(done=True)

        return StreamChunk(
            content=data.get("text", ""),
            finish_reason=data.get("finish_reason"),
        )

    def count_tokens(self, text: str, model: str = None) -> int:
        """Use custom tokenization."""
        # Option 1: Use provider's API
        response = requests.post(
            f"{self.base_url}/tokenize",
            json={"text": text, "model": model}
        )
        return response.json()["token_count"]

        # Option 2: Load custom tokenizer
        # from my_tokenizer import CustomTokenizer
        # return len(CustomTokenizer.encode(text))
```

### Step 2: Add Tests

```python
import pytest
from a11i.providers.my_custom import MyCustomProvider

def test_parse_response():
    """Test response parsing."""
    provider = MyCustomProvider()

    raw_response = {
        "output": {"text": "Hello, world!"},
        "model_id": "my-model-v1",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5
        },
        "stop_reason": "end_turn"
    }

    parsed = provider.parse_response(raw_response)

    assert parsed.content == "Hello, world!"
    assert parsed.model == "my-model-v1"
    assert parsed.input_tokens == 10
    assert parsed.output_tokens == 5

def test_cost_calculation():
    """Test cost calculation."""
    provider = MyCustomProvider()

    cost = provider.calculate_cost(
        input_tokens=1000,
        output_tokens=500,
        model="my-model-v1"
    )

    # (1000 * 0.001 + 500 * 0.002) / 1000 = 0.002
    assert cost == 0.002
```

### Step 3: Update Configuration

```python
# Add to provider registry (if not using @register_provider decorator)
from a11i.providers import PROVIDER_REGISTRY
from a11i.providers.my_custom import MyCustomProvider

PROVIDER_REGISTRY["my_custom_provider"] = MyCustomProvider
```

### Step 4: Document Usage

```python
# Example usage documentation
from a11i import A11I

# Initialize with custom provider
client = A11I(
    provider="my_custom_provider",
    api_key="...",
    model="my-model-v1",
    telemetry={"service.name": "my-app"}
)

# Use like any other provider
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    stream=False
)

print(response.choices[0].message.content)
```

## Provider-Specific Features

### Function Calling

Some providers support structured function/tool calling:

```python
# OpenAI function calling
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Boston?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }]
)

# Anthropic tool use
response = client.messages.create(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "What's the weather in Boston?"}],
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }]
)
```

### JSON Mode

Guaranteed JSON output:

```python
# OpenAI JSON mode
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List 3 colors in JSON"}],
    response_format={"type": "json_object"}
)
```

### Vision/Multimodal

Image input support:

```python
# OpenAI vision
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
)

# Anthropic vision
response = client.messages.create(
    model="claude-3-5-sonnet",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "source": {"type": "base64", "data": "..."}}
        ]
    }]
)
```

### Content Filtering

Azure OpenAI content filtering:

```python
# Content filtering configuration
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "..."}],
    content_filter_policy="default"
)

# Check filtering results
if response.prompt_filter_results:
    for result in response.prompt_filter_results:
        print(f"Category: {result.category}, Severity: {result.severity}")
```

### Guardrails

AWS Bedrock guardrails:

```python
# Apply guardrails
client = A11I(
    provider="aws.bedrock",
    model="anthropic.claude-3-sonnet",
    guardrail_id="ff6w9c0kqv7q",
    guardrail_version="1"
)
```

## Key Takeaways

> **Provider Support Summary**
>
> - **Unified Interface:** All providers implement the same `LLMProvider` interface for consistent usage
> - **Automatic Token Counting:** Each provider implements model-specific tokenization
> - **Cost Tracking:** Built-in pricing information for automatic cost calculation
> - **Streaming Support:** All providers support streaming responses with standardized parsing
> - **OpenTelemetry Integration:** Provider-specific semantic conventions for observability
> - **Extensible Architecture:** Add new providers by implementing base class and registering
> - **Feature Detection:** Provider capabilities exposed through metadata
> - **Self-Hosted Friendly:** Full support for custom deployments and local models
>
> **Best Practices:**
> - Choose provider based on model capabilities, latency requirements, and cost constraints
> - Use provider-specific features (function calling, vision, etc.) when available
> - Test token counting accuracy for cost estimation
> - Implement retry logic for provider-specific errors
> - Monitor provider-specific telemetry attributes for debugging

**Related Documentation:**
- [LLM Abstraction](/home/becker/projects/a11i/docs/02-core-concepts/llm-abstraction.md) - Core abstraction layer design
- [Token Accounting](/home/becker/projects/a11i/docs/04-implementation/token-accounting.md) - Token counting and cost tracking
- [Observability](/home/becker/projects/a11i/docs/04-implementation/observability.md) - Provider telemetry and monitoring
