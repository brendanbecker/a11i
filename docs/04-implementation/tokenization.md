---
title: Tokenization and Token Counting
section: Implementation
version: 1.0.0
last_updated: 2025-11-26
status: active
related:
  - cost-tracking.md
  - streaming-responses.md
  - provider-adapters.md
  - caching-strategies.md
---

# Tokenization and Token Counting

## Table of Contents

1. [Overview](#overview)
2. [The Tokenization Challenge](#the-tokenization-challenge)
3. [Tokenizer by Model Family](#tokenizer-by-model-family)
4. [Implementation: tiktoken (OpenAI)](#implementation-tiktoken-openai)
5. [Implementation: SentencePiece (LLaMA/Mistral)](#implementation-sentencepiece-llamamistral)
6. [Implementation: Anthropic (Claude)](#implementation-anthropic-claude)
7. [Unified Token Counter](#unified-token-counter)
8. [Local vs API-Returned Counts](#local-vs-api-returned-counts)
9. [Performance Optimization](#performance-optimization)
10. [Caching Strategies](#caching-strategies)
11. [Async Token Counting](#async-token-counting)
12. [Best Practices](#best-practices)
13. [Key Takeaways](#key-takeaways)

## Overview

Token counting is critical for LLM applications to:

- **Track costs accurately**: Providers bill by token usage
- **Monitor context windows**: Ensure requests fit within model limits
- **Optimize prompt efficiency**: Identify opportunities to reduce token usage
- **Budget management**: Predict and control API spending

The a11i framework provides unified token counting across all supported providers with intelligent fallback strategies and performance optimization.

## The Tokenization Challenge

### Provider Variability

Each LLM provider uses different tokenization algorithms:

- **OpenAI**: Uses tiktoken with different encodings per model generation
- **Anthropic**: Proprietary BPE tokenizer (GPT-2 similar, but not identical)
- **Meta/LLaMA**: SentencePiece unigram language model
- **Mistral**: SentencePiece BPE
- **Google/Gemini**: SentencePiece with custom vocabulary
- **Cohere**: Custom BPE implementation

### Key Challenges

1. **Accuracy vs Speed**: Exact tokenization requires model-specific libraries
2. **Availability**: Not all providers publish their tokenizers
3. **Context Overhead**: Chat format adds structural tokens
4. **Streaming Responses**: Usage data often arrives at completion
5. **Performance**: Tokenization at scale requires caching and optimization

### When Token Counts Matter

```python
# Cost tracking - need exact counts for billing
total_cost = (prompt_tokens * prompt_price) + (completion_tokens * completion_price)

# Context management - must not exceed limits
if total_tokens > model.max_context:
    raise ContextLengthExceeded()

# Budget enforcement - prevent runaway costs
if estimated_tokens * price_per_token > budget_remaining:
    return BudgetExceededError()
```

## Tokenizer by Model Family

| Model Family | Tokenizer Type | Library | Encoding Name | Availability |
|--------------|----------------|---------|---------------|--------------|
| OpenAI GPT-4o | tiktoken | tiktoken | o200k_base | Public |
| OpenAI GPT-4 | tiktoken | tiktoken | cl100k_base | Public |
| OpenAI GPT-3.5 | tiktoken | tiktoken | cl100k_base | Public |
| Anthropic Claude | BPE (proprietary) | tiktoken (approx) | gpt2 (similar) | API only |
| Meta LLaMA 2/3 | SentencePiece | sentencepiece | N/A | Public |
| Mistral | SentencePiece BPE | sentencepiece | N/A | Public |
| Google Gemini | SentencePiece | sentencepiece | N/A | Public |
| Cohere | BPE | tokenizers | N/A | Public |

### Installation Requirements

```bash
# OpenAI models
pip install tiktoken

# LLaMA, Mistral, Gemini
pip install sentencepiece

# Cohere
pip install tokenizers

# All providers
pip install tiktoken sentencepiece tokenizers
```

## Implementation: tiktoken (OpenAI)

### Basic Implementation

```python
import tiktoken
from typing import Dict, List

class TiktokenTokenizer:
    """OpenAI tokenizer using tiktoken.

    Supports all OpenAI models with accurate token counting including
    chat formatting overhead.
    """

    # Model to encoding mapping
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
    }

    def __init__(self):
        """Initialize tokenizer with encoding cache."""
        self._encodings: Dict[str, tiktoken.Encoding] = {}

    def get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get or create encoding for model.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-4o")

        Returns:
            tiktoken.Encoding instance

        Raises:
            ValueError: If model not supported
        """
        # Normalize model name (handle versioned names)
        base_model = self._normalize_model_name(model)
        encoding_name = self.MODEL_ENCODINGS.get(base_model, "cl100k_base")

        # Cache encodings for reuse
        if encoding_name not in self._encodings:
            self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)

        return self._encodings[encoding_name]

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in plain text.

        Args:
            text: Text to tokenize
            model: Model name for correct encoding

        Returns:
            Number of tokens

        Example:
            >>> tokenizer = TiktokenTokenizer()
            >>> tokenizer.count_tokens("Hello, world!", "gpt-4")
            4
        """
        encoding = self.get_encoding(model)
        return len(encoding.encode(text))

    def count_messages(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4"
    ) -> int:
        """Count tokens in chat messages including formatting overhead.

        OpenAI's chat format adds special tokens for message structure:
        - Each message: <|start|>role<|message|>content<|end|>
        - Final assistant prompt: <|start|>assistant<|message|>

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name for correct encoding

        Returns:
            Total token count including formatting

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> tokenizer.count_messages(messages, "gpt-4")
            17
        """
        encoding = self.get_encoding(model)

        # Formatting tokens vary by model
        if "gpt-4" in model or "gpt-3.5" in model:
            tokens_per_message = 3  # <|start|>role<|message|>
            tokens_per_name = 1     # If name field present
        else:
            tokens_per_message = 3
            tokens_per_name = 1

        total_tokens = 0

        for message in messages:
            total_tokens += tokens_per_message

            for key, value in message.items():
                total_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    total_tokens += tokens_per_name

        # Every reply is primed with <|start|>assistant<|message|>
        total_tokens += 3

        return total_tokens

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name to base version.

        Example:
            "gpt-4-0613" -> "gpt-4"
            "gpt-4o-2024-05-13" -> "gpt-4o"
        """
        for base_name in self.MODEL_ENCODINGS.keys():
            if model.startswith(base_name):
                return base_name
        return model
```

### Advanced Features

```python
class AdvancedTiktokenTokenizer(TiktokenTokenizer):
    """Extended tiktoken tokenizer with tool calling and function support."""

    def count_function_call(
        self,
        function_name: str,
        arguments: Dict,
        model: str = "gpt-4"
    ) -> int:
        """Count tokens in function call response.

        Function calls add tokens for:
        - Function name
        - JSON-formatted arguments
        - Structural formatting
        """
        encoding = self.get_encoding(model)

        # Function name tokens
        tokens = len(encoding.encode(function_name))

        # Arguments as JSON string
        import json
        args_json = json.dumps(arguments)
        tokens += len(encoding.encode(args_json))

        # Formatting overhead (~10 tokens)
        tokens += 10

        return tokens

    def count_tool_definitions(
        self,
        tools: List[Dict],
        model: str = "gpt-4"
    ) -> int:
        """Count tokens in tool/function definitions.

        Tool definitions are sent in every request and can be significant.
        Cache these counts as they're typically static.
        """
        encoding = self.get_encoding(model)

        import json
        tools_json = json.dumps(tools)
        return len(encoding.encode(tools_json))
```

## Implementation: SentencePiece (LLaMA/Mistral)

### SentencePiece Tokenizer

```python
import sentencepiece as spm
from pathlib import Path
from typing import Optional

class SentencePieceTokenizer:
    """Tokenizer for LLaMA, Mistral, Gemini, and other SentencePiece models.

    Requires model-specific tokenizer files (.model files).
    Download from model repositories or Hugging Face.
    """

    # Paths to tokenizer model files
    MODEL_PATHS = {
        "llama2": "tokenizers/llama2.model",
        "llama3": "tokenizers/llama3.model",
        "llama3.1": "tokenizers/llama3.1.model",
        "mistral": "tokenizers/mistral.model",
        "mixtral": "tokenizers/mixtral.model",
        "gemini": "tokenizers/gemini.model",
    }

    def __init__(self, tokenizer_dir: Optional[Path] = None):
        """Initialize SentencePiece tokenizer.

        Args:
            tokenizer_dir: Directory containing .model files
        """
        self._processors = {}
        self._tokenizer_dir = tokenizer_dir or Path("tokenizers")

    def get_processor(self, model_family: str) -> spm.SentencePieceProcessor:
        """Load or get cached SentencePiece processor.

        Args:
            model_family: Model family name (e.g., "llama3", "mistral")

        Returns:
            SentencePieceProcessor instance

        Raises:
            ValueError: If tokenizer model file not found
        """
        if model_family not in self._processors:
            model_path = self._get_model_path(model_family)

            if not model_path.exists():
                raise ValueError(
                    f"Tokenizer model not found: {model_path}\n"
                    f"Download from: https://huggingface.co/{self._get_hf_repo(model_family)}"
                )

            sp = spm.SentencePieceProcessor()
            sp.Load(str(model_path))
            self._processors[model_family] = sp

        return self._processors[model_family]

    def count_tokens(self, text: str, model_family: str = "llama3") -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize
            model_family: Model family for correct tokenizer

        Returns:
            Number of tokens

        Example:
            >>> tokenizer = SentencePieceTokenizer()
            >>> tokenizer.count_tokens("Hello, world!", "llama3")
            5
        """
        sp = self.get_processor(model_family)
        return len(sp.EncodeAsIds(text))

    def encode(self, text: str, model_family: str = "llama3") -> list[int]:
        """Encode text to token IDs.

        Useful for debugging or advanced token manipulation.
        """
        sp = self.get_processor(model_family)
        return sp.EncodeAsIds(text)

    def decode(self, token_ids: list[int], model_family: str = "llama3") -> str:
        """Decode token IDs back to text."""
        sp = self.get_processor(model_family)
        return sp.DecodeIds(token_ids)

    def _get_model_path(self, model_family: str) -> Path:
        """Get path to tokenizer model file."""
        relative_path = self.MODEL_PATHS.get(model_family)
        if not relative_path:
            raise ValueError(f"Unknown model family: {model_family}")

        return self._tokenizer_dir / Path(relative_path).name

    def _get_hf_repo(self, model_family: str) -> str:
        """Get Hugging Face repository for tokenizer download."""
        repos = {
            "llama2": "meta-llama/Llama-2-7b-hf",
            "llama3": "meta-llama/Meta-Llama-3-8B",
            "llama3.1": "meta-llama/Meta-Llama-3.1-8B",
            "mistral": "mistralai/Mistral-7B-v0.1",
            "mixtral": "mistralai/Mixtral-8x7B-v0.1",
        }
        return repos.get(model_family, "")
```

## Implementation: Anthropic (Claude)

### Anthropic Token Counter

```python
import tiktoken
from typing import Optional

class AnthropicTokenizer:
    """Tokenizer for Claude models.

    Anthropic uses a proprietary tokenizer similar to GPT-2 BPE.
    For exact counts, use API-returned usage data.
    For estimation, use GPT-2 encoding as approximation.
    """

    def __init__(self):
        """Initialize with GPT-2 encoding for approximation."""
        self._encoding = tiktoken.get_encoding("gpt2")

    def count_tokens(self, text: str) -> int:
        """Approximate token count using GPT-2 encoding.

        Note: This is an approximation. Claude's actual tokenizer
        may produce slightly different counts (+/- 5%).

        Prefer API-returned usage data when available.

        Args:
            text: Text to tokenize

        Returns:
            Approximate token count
        """
        return len(self._encoding.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        """Approximate token count for chat messages.

        Claude's message format:
        - System prompt sent separately
        - User/assistant messages in conversation
        - Additional overhead for message structure
        """
        total_tokens = 0

        for message in messages:
            # Role and content
            role = message.get("role", "")
            content = message.get("content", "")

            total_tokens += self.count_tokens(role)
            total_tokens += self.count_tokens(content)

            # Message formatting overhead (~5 tokens per message)
            total_tokens += 5

        return total_tokens

    def count_tokens_accurate(
        self,
        text: str,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022"
    ) -> int:
        """Get exact token count via Anthropic's API.

        Uses the beta token counting endpoint for 100% accurate counts.

        Args:
            text: Text to tokenize
            api_key: Anthropic API key
            model: Model name for tokenization

        Returns:
            Exact token count

        Example:
            >>> tokenizer = AnthropicTokenizer()
            >>> count = tokenizer.count_tokens_accurate(
            ...     "Hello, Claude!",
            ...     api_key=os.getenv("ANTHROPIC_API_KEY")
            ... )
        """
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        # Use beta token counting endpoint
        response = client.beta.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}]
        )

        return response.input_tokens
```

## Unified Token Counter

### Multi-Provider Token Counter

```python
from typing import Optional, Union
from enum import Enum

class ProviderType(Enum):
    """Supported provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GEMINI = "gemini"
    COHERE = "cohere"

class TokenCounter:
    """Unified token counter with intelligent fallback strategy.

    Strategy:
    1. If API returned usage data, use it (100% accurate for billing)
    2. Otherwise, use provider-specific local tokenizer
    3. If no tokenizer available, estimate from character count
    """

    def __init__(self):
        """Initialize with all tokenizer implementations."""
        self.tiktoken = TiktokenTokenizer()
        self.sentencepiece = SentencePieceTokenizer()
        self.anthropic = AnthropicTokenizer()

    def count(
        self,
        text: str,
        model: str,
        api_returned: Optional[int] = None,
        prefer_api: bool = True,
    ) -> int:
        """Count tokens with intelligent fallback.

        Args:
            text: Text to tokenize
            model: Model name
            api_returned: Token count from API response (if available)
            prefer_api: Whether to prefer API count over local

        Returns:
            Token count

        Example:
            >>> counter = TokenCounter()
            >>>
            >>> # With API count (preferred)
            >>> count = counter.count(
            ...     "Hello, world!",
            ...     "gpt-4",
            ...     api_returned=4
            ... )
            >>>
            >>> # Without API count (local tokenizer)
            >>> count = counter.count("Hello, world!", "gpt-4")
        """
        # Prefer API-returned count if available
        if api_returned is not None and prefer_api:
            return api_returned

        # Route to appropriate tokenizer
        provider = self._detect_provider(model)

        if provider == ProviderType.OPENAI:
            return self.tiktoken.count_tokens(text, model)

        elif provider == ProviderType.ANTHROPIC:
            return self.anthropic.count_tokens(text)

        elif provider == ProviderType.LLAMA:
            model_family = self._get_llama_family(model)
            return self.sentencepiece.count_tokens(text, model_family)

        elif provider == ProviderType.MISTRAL:
            return self.sentencepiece.count_tokens(text, "mistral")

        elif provider == ProviderType.GEMINI:
            return self.sentencepiece.count_tokens(text, "gemini")

        else:
            # Fallback: character-based estimation
            return self._estimate_tokens(text)

    def count_messages(
        self,
        messages: list[dict],
        model: str,
        api_returned: Optional[int] = None,
    ) -> int:
        """Count tokens in chat messages."""
        if api_returned is not None:
            return api_returned

        provider = self._detect_provider(model)

        if provider == ProviderType.OPENAI:
            return self.tiktoken.count_messages(messages, model)

        elif provider == ProviderType.ANTHROPIC:
            return self.anthropic.count_messages(messages)

        else:
            # Fallback: sum individual messages
            total = 0
            for msg in messages:
                content = msg.get("content", "")
                total += self.count(content, model)
                total += 5  # Message overhead
            return total

    def _detect_provider(self, model: str) -> ProviderType:
        """Detect provider from model name."""
        model_lower = model.lower()

        if any(x in model_lower for x in ["gpt", "davinci", "curie", "babbage"]):
            return ProviderType.OPENAI

        elif "claude" in model_lower:
            return ProviderType.ANTHROPIC

        elif "llama" in model_lower:
            return ProviderType.LLAMA

        elif "mistral" in model_lower or "mixtral" in model_lower:
            return ProviderType.MISTRAL

        elif "gemini" in model_lower:
            return ProviderType.GEMINI

        elif "cohere" in model_lower or "command" in model_lower:
            return ProviderType.COHERE

        else:
            return ProviderType.OPENAI  # Default fallback

    def _get_llama_family(self, model: str) -> str:
        """Determine LLaMA model family."""
        if "llama-3.1" in model.lower() or "llama3.1" in model.lower():
            return "llama3.1"
        elif "llama-3" in model.lower() or "llama3" in model.lower():
            return "llama3"
        else:
            return "llama2"

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 characters per token for English.

        This is a fallback for unsupported models.
        Actual ratio varies: 3-5 chars/token depending on language.
        """
        return max(1, len(text) // 4)
```

## Local vs API-Returned Counts

### When to Use Each

| Scenario | Recommended Approach | Accuracy |
|----------|---------------------|----------|
| Pre-request estimation | Local tokenizer | 95-99% |
| Cost tracking | API-returned count | 100% |
| Budget enforcement | Local tokenizer | 95-99% |
| Context window check | Local tokenizer | 95-99% |
| Billing reconciliation | API-returned count | 100% |
| Streaming responses | Local + API final | 100% at end |

### Handling API Response Usage

```python
from dataclasses import dataclass

@dataclass
class TokenUsage:
    """Token usage data from API response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_openai_response(cls, response) -> "TokenUsage":
        """Extract usage from OpenAI response."""
        usage = response.usage
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    @classmethod
    def from_anthropic_response(cls, response) -> "TokenUsage":
        """Extract usage from Anthropic response."""
        return cls(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

class UsageTracker:
    """Track token usage with API validation."""

    def __init__(self, counter: TokenCounter):
        self.counter = counter
        self._estimated_vs_actual = []

    def track_request(
        self,
        prompt: str,
        model: str,
        response_usage: TokenUsage,
    ):
        """Track request and validate estimation accuracy.

        Useful for monitoring tokenizer accuracy and identifying
        cases where estimation significantly differs from actual.
        """
        # Get local estimate
        estimated = self.counter.count(prompt, model)
        actual = response_usage.prompt_tokens

        # Calculate error
        error_pct = abs(estimated - actual) / actual * 100

        self._estimated_vs_actual.append({
            "model": model,
            "estimated": estimated,
            "actual": actual,
            "error_pct": error_pct,
        })

        # Alert if estimation is significantly off
        if error_pct > 10:
            print(f"Warning: Token estimation off by {error_pct:.1f}% for {model}")

    def get_accuracy_stats(self) -> dict:
        """Get tokenizer accuracy statistics."""
        if not self._estimated_vs_actual:
            return {}

        errors = [x["error_pct"] for x in self._estimated_vs_actual]

        return {
            "samples": len(errors),
            "mean_error_pct": sum(errors) / len(errors),
            "max_error_pct": max(errors),
            "accuracy_within_5pct": sum(1 for e in errors if e <= 5) / len(errors),
        }
```

## Performance Optimization

### Token Count Caching

```python
import functools
import hashlib
from collections import OrderedDict
from typing import Tuple

class CachedTokenCounter:
    """Token counter with LRU caching for repeated content.

    Common use cases:
    - System prompts (reused across requests)
    - Few-shot examples (static templates)
    - Tool definitions (rarely change)
    """

    def __init__(self, counter: TokenCounter, cache_size: int = 10000):
        """Initialize with cache.

        Args:
            counter: Underlying token counter
            cache_size: Maximum number of cached entries
        """
        self.counter = counter
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._cache_size = cache_size
        self._hits = 0
        self._misses = 0

    def count(self, text: str, model: str) -> int:
        """Count tokens with caching.

        Cache key is based on text hash and model name.
        """
        key = self._make_key(text, model)

        # Check cache
        if key in self._cache:
            self._hits += 1
            # Move to end (LRU)
            self._cache.move_to_end(key)
            return self._cache[key]

        # Cache miss - compute
        self._misses += 1
        count = self.counter.count(text, model)

        # Add to cache
        self._cache[key] = count

        # Evict oldest if over limit
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return count

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text hash and model.

        Uses MD5 hash of text (fast, collision-resistant enough for cache).
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{model}:{text_hash}"

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "efficiency": f"{hit_rate * 100:.1f}%",
        }

    def clear_cache(self):
        """Clear cache and reset stats."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
```

## Caching Strategies

### Prompt Component Cache

```python
class PromptTokenCache:
    """Specialized cache for prompt components.

    Optimized for common LLM patterns:
    - System prompts: High reuse, cache indefinitely
    - Tool definitions: Static, pre-compute
    - Few-shot examples: Template-based, cache per template
    """

    def __init__(self, counter: TokenCounter):
        self.counter = counter
        self._system_prompts = {}
        self._tool_definitions = {}
        self._few_shot_examples = {}

    def cache_system_prompt(
        self,
        name: str,
        prompt: str,
        model: str
    ) -> int:
        """Pre-cache a system prompt.

        Args:
            name: Identifier for this system prompt
            prompt: System prompt text
            model: Model to use for tokenization

        Returns:
            Token count

        Example:
            >>> cache = PromptTokenCache(counter)
            >>> cache.cache_system_prompt(
            ...     "helpful_assistant",
            ...     "You are a helpful assistant.",
            ...     "gpt-4"
            ... )
            7
        """
        count = self.counter.count(prompt, model)
        self._system_prompts[(name, model)] = count
        return count

    def get_cached_system_prompt_tokens(
        self,
        name: str,
        model: str
    ) -> int:
        """Get cached token count for system prompt.

        Returns 0 if not cached.
        """
        return self._system_prompts.get((name, model), 0)

    def cache_tool_definitions(
        self,
        name: str,
        tools: list[dict],
        model: str
    ) -> int:
        """Pre-cache tool/function definitions.

        Tool definitions are sent with every request and can be
        substantial (100-1000+ tokens). Pre-compute and cache.
        """
        import json
        tools_json = json.dumps(tools)
        count = self.counter.count(tools_json, model)
        self._tool_definitions[(name, model)] = count
        return count

    def cache_few_shot_examples(
        self,
        name: str,
        examples: list[dict],
        model: str
    ) -> int:
        """Cache few-shot example tokens."""
        total = 0
        for example in examples:
            # Count each message in example
            for msg in example.get("messages", []):
                content = msg.get("content", "")
                total += self.counter.count(content, model)
                total += 5  # Message overhead

        self._few_shot_examples[(name, model)] = total
        return total

    def total_fixed_tokens(self, model: str) -> int:
        """Calculate total tokens from all cached components.

        Useful for budget calculations:
        fixed_tokens + estimated_user_prompt + max_completion
        """
        total = 0

        # Sum all system prompts for this model
        total += sum(
            v for (_, m), v in self._system_prompts.items()
            if m == model
        )

        # Sum all tool definitions
        total += sum(
            v for (_, m), v in self._tool_definitions.items()
            if m == model
        )

        # Sum all few-shot examples
        total += sum(
            v for (_, m), v in self._few_shot_examples.items()
            if m == model
        )

        return total

    def estimate_request_tokens(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        tool_set: Optional[str] = None,
        examples_set: Optional[str] = None,
    ) -> int:
        """Estimate total tokens for a request.

        Combines cached components with dynamic content.
        """
        total = 0

        # System prompt
        total += self.get_cached_system_prompt_tokens(system_prompt, model)

        # Tool definitions (if using)
        if tool_set:
            total += self._tool_definitions.get((tool_set, model), 0)

        # Few-shot examples (if using)
        if examples_set:
            total += self._few_shot_examples.get((examples_set, model), 0)

        # User message (not cached)
        total += self.counter.count(user_message, model)

        # Message formatting overhead
        total += 10

        return total
```

## Async Token Counting

### Non-Blocking Implementation

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

class AsyncTokenCounter:
    """Non-blocking token counter for async applications.

    Tokenization can be CPU-intensive for large texts.
    This runs tokenization in thread pool to avoid blocking
    the event loop.
    """

    def __init__(
        self,
        counter: TokenCounter,
        max_workers: int = 4
    ):
        """Initialize async counter.

        Args:
            counter: Underlying token counter
            max_workers: Thread pool size
        """
        self.counter = counter
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def count_async(
        self,
        text: str,
        model: str,
        api_returned: Optional[int] = None,
    ) -> int:
        """Count tokens without blocking event loop.

        Example:
            >>> counter = AsyncTokenCounter(TokenCounter())
            >>> count = await counter.count_async("Hello!", "gpt-4")
        """
        # If API count available, return immediately
        if api_returned is not None:
            return api_returned

        # Otherwise, run tokenization in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.counter.count,
            text,
            model,
        )

    async def count_batch_async(
        self,
        texts: List[str],
        model: str,
    ) -> List[int]:
        """Count tokens for multiple texts concurrently.

        Efficient for batch processing.

        Example:
            >>> texts = ["Hello", "World", "How are you?"]
            >>> counts = await counter.count_batch_async(texts, "gpt-4")
            >>> # [2, 1, 4]
        """
        tasks = [
            self.count_async(text, model)
            for text in texts
        ]
        return await asyncio.gather(*tasks)

    async def count_messages_async(
        self,
        messages: List[dict],
        model: str,
    ) -> int:
        """Count message tokens asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.counter.count_messages,
            messages,
            model,
        )

    def shutdown(self):
        """Shutdown thread pool."""
        self._executor.shutdown(wait=True)
```

## Best Practices

### 1. Prefer API-Returned Counts for Billing

```python
# GOOD: Use API count for cost tracking
response = client.chat.completions.create(...)
usage = response.usage
cost = (usage.prompt_tokens * PROMPT_PRICE +
        usage.completion_tokens * COMPLETION_PRICE)

# AVOID: Local estimate for billing
estimated = counter.count(prompt, model)
cost = estimated * PRICE  # May be inaccurate
```

### 2. Cache Static Components

```python
# GOOD: Pre-compute tool definitions once
cache.cache_tool_definitions("my_tools", tools, "gpt-4")
total_tokens = cache.total_fixed_tokens("gpt-4") + user_tokens

# AVOID: Re-tokenize on every request
for request in requests:
    tools_tokens = counter.count(json.dumps(tools), "gpt-4")  # Wasteful
```

### 3. Use Local Counts for Pre-Request Checks

```python
# GOOD: Check context limit before API call
estimated = counter.count_messages(messages, "gpt-4")
if estimated > 128000:
    # Truncate or summarize
    messages = truncate_messages(messages, 120000)

# Then make API call
response = client.chat.completions.create(...)
```

### 4. Monitor Estimation Accuracy

```python
# Track and alert on estimation errors
tracker = UsageTracker(counter)

for request, response in request_response_pairs:
    tracker.track_request(
        request.prompt,
        request.model,
        TokenUsage.from_openai_response(response)
    )

stats = tracker.get_accuracy_stats()
if stats["mean_error_pct"] > 5:
    print("Warning: Token estimation accuracy degraded")
```

### 5. Handle Streaming Responses

```python
# For streaming, estimate upfront, validate at end
estimated_prompt = counter.count_messages(messages, "gpt-4")
estimated_total = estimated_prompt + max_tokens

# Stream response
completion_tokens = 0
for chunk in stream:
    # Count as you go (approximate)
    if chunk.choices[0].delta.content:
        completion_tokens += 1

# Validate at end
if hasattr(stream, 'usage'):
    actual_tokens = stream.usage.total_tokens
    # Update billing with actual
```

## Key Takeaways

### Critical Points

1. **API Counts are Ground Truth**: Always prefer provider-returned usage for billing and cost tracking
2. **Local Counts for Pre-Flight**: Use local tokenizers for context checks and budget enforcement before API calls
3. **Different Tokenizers**: Each provider uses different algorithms; use correct implementation
4. **Cache Aggressively**: System prompts, tool definitions, and examples should be pre-computed
5. **Performance Matters**: Use async and caching for large-scale applications

### Tokenizer Selection Matrix

| Provider | Best Method | Accuracy | Speed | Availability |
|----------|-------------|----------|-------|--------------|
| OpenAI | tiktoken | 99%+ | Fast | Public library |
| Anthropic | API count | 100% | N/A | API only (approximation: GPT-2) |
| LLaMA | SentencePiece | 99%+ | Fast | Requires .model file |
| Mistral | SentencePiece | 99%+ | Fast | Requires .model file |
| Gemini | SentencePiece | 95%+ | Fast | Requires .model file |

### Implementation Checklist

- [ ] Install correct tokenizer libraries (tiktoken, sentencepiece)
- [ ] Download model-specific tokenizer files where needed
- [ ] Implement caching for static prompt components
- [ ] Use API-returned counts for billing
- [ ] Monitor estimation vs actual accuracy
- [ ] Handle streaming response token counting
- [ ] Consider async tokenization for high-throughput applications
- [ ] Set up alerts for estimation errors > 10%

### Cross-References

- **Cost Tracking**: See `/home/becker/projects/a11i/docs/04-implementation/cost-tracking.md` for usage monitoring
- **Streaming**: See `/home/becker/projects/a11i/docs/04-implementation/streaming-responses.md` for handling streaming token counts
- **Provider Adapters**: See `/home/becker/projects/a11i/docs/04-implementation/provider-adapters.md` for integration
- **Caching**: See `/home/becker/projects/a11i/docs/04-implementation/caching-strategies.md` for caching patterns

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-26
**Status**: Active
