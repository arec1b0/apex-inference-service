"""Production-grade cost tracking for LLM inference requests.

Provides token counting, cost calculation, and budget enforcement
with Prometheus metrics and graceful degradation.

Example:
    >>> from src.app.monitoring.cost_tracker import cost_tracker
    >>> 
    >>> @router.post("/predict")
    >>> async def predict(request: Request, body: PredictionRequest):
    >>>     async with cost_tracker.track_tokens(
    >>>         tenant_id=request.state.tenant_id,
    >>>         model="gpt-4",
    >>>     ):
    >>>         response = await llm_service.complete(prompt)
    >>>     return response
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict
import json

from prometheus_client import Counter, Gauge
from loguru import logger


@dataclass
class TokenUsage:
    """Token usage for a single request."""
    input_tokens: int
    output_tokens: int
    model: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PricingConfig:
    """Pricing configuration per model provider.
    
    Prices are per 1K tokens in USD.
    """
    openai: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    })
    
    anthropic: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    })
    
    local: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "llama-2-7b": {"input": 0.0001, "output": 0.0001},
        "llama-2-70b": {"input": 0.0005, "output": 0.0005},
    })
    
    def get_price(self, model: str, direction: str) -> float:
        """Get price for model and token direction.
        
        Args:
            model: Model name (e.g., "gpt-4")
            direction: "input" or "output"
            
        Returns:
            Price per 1K tokens in USD
            
        Raises:
            ValueError: If model not found in pricing config
        """
        for provider_models in [self.openai, self.anthropic, self.local]:
            if model in provider_models:
                return provider_models[model][direction]
        
        raise ValueError(f"Model {model} not found in pricing config")


# Prometheus Metrics
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total number of LLM tokens processed",
    ["model", "direction", "tenant_id"]
)

LLM_COST_USD_TOTAL = Counter(
    "llm_cost_usd_total",
    "Total cost in USD for LLM usage",
    ["model", "tenant_id"]
)

BUDGET_REMAINING_USD = Gauge(
    "budget_remaining_usd",
    "Remaining budget for current period in USD",
    ["tenant_id"]
)

BUDGET_DAILY_RESET_TIME = Gauge(
    "budget_daily_reset_time",
    "Unix timestamp of next daily budget reset",
    ["tenant_id"]
)


class TokenCounter:
    """Tracks token usage per model with daily reset capability.
    
    Thread-safe implementation using asyncio locks.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._daily_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )
        self._last_reset_date = datetime.now(timezone.utc).date()
    
    async def track(self, model: str, input_tokens: int, output_tokens: int, tenant_id: Optional[str] = None) -> None:
        """Track token usage for a model.
        
        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            tenant_id: Optional tenant identifier for metrics
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        
        async with self._lock:
            await self._check_daily_reset()
            
            self._daily_usage[model]["input"] += input_tokens
            self._daily_usage[model]["output"] += output_tokens
            
            # Update Prometheus metrics
            LLM_TOKENS_TOTAL.labels(
                model=model,
                direction="input",
                tenant_id=tenant_id or "unknown"
            ).inc(input_tokens)
            
            LLM_TOKENS_TOTAL.labels(
                model=model,
                direction="output",
                tenant_id=tenant_id or "unknown"
            ).inc(output_tokens)
            
            logger.debug(
                f"Tracked tokens: {model} - {input_tokens} input, {output_tokens} output"
            )
    
    async def get_total(self, model: Optional[str] = None) -> Dict[str, int]:
        """Get total tokens for a model or all models.
        
        Args:
            model: Specific model to query, None for all models
            
        Returns:
            Dictionary with token counts
        """
        async with self._lock:
            await self._check_daily_reset()
            
            if model:
                return dict(self._daily_usage.get(model, {"input": 0, "output": 0}))
            return {k: dict(v) for k, v in self._daily_usage.items()}
    
    async def reset_daily(self) -> None:
        """Reset daily token counts."""
        async with self._lock:
            self._daily_usage.clear()
            self._last_reset_date = datetime.now(timezone.utc).date()
            logger.info("Daily token counts reset")
    
    async def _check_daily_reset(self) -> None:
        """Check if we need to reset for new day."""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self._last_reset_date:
            await self.reset_daily()


class CostCalculator:
    """Calculates costs based on token usage and pricing.
    
    Maintains running total of costs with per-model breakdown.
    """
    
    def __init__(self, pricing_config: Optional[PricingConfig] = None):
        self.pricing = pricing_config or PricingConfig()
        self._lock = asyncio.Lock()
        self._total_costs: Dict[str, float] = defaultdict(float)
        self._daily_costs: Dict[str, float] = defaultdict(float)
        self._last_reset_date = datetime.now(timezone.utc).date()
    
    async def calculate(self, model: str, input_tokens: int, output_tokens: int, tenant_id: Optional[str] = None) -> float:
        """Calculate cost for token usage.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            tenant_id: Optional tenant for metrics
            
        Returns:
            Cost in USD
        """
        try:
            input_price = self.pricing.get_price(model, "input")
            output_price = self.pricing.get_price(model, "output")
            
            # Cost per token = price_per_1k / 1000
            input_cost = (input_tokens / 1000) * input_price
            output_cost = (output_tokens / 1000) * output_price
            total_cost = input_cost + output_cost
            
            async with self._lock:
                await self._check_daily_reset()
                
                self._total_costs[model] += total_cost
                self._daily_costs[model] += total_cost
                
                # Update Prometheus
                LLM_COST_USD_TOTAL.labels(
                    model=model,
                    tenant_id=tenant_id or "unknown"
                ).inc(total_cost)
            
            logger.debug(
                f"Cost calculated: {model} - ${total_cost:.6f} "
                f"({input_tokens} input, {output_tokens} output)"
            )
            
            return total_cost
            
        except ValueError as e:
            logger.error(f"Cost calculation failed: {e}")
            raise
    
    async def get_cost_so_far(self, model: Optional[str] = None, daily: bool = False) -> float:
        """Get total cost incurred.
        
        Args:
            model: Specific model to query
            daily: If True, return daily cost instead of total
            
        Returns:
            Cost in USD
        """
        async with self._lock:
            await self._check_daily_reset()
            
            costs = self._daily_costs if daily else self._total_costs
            
            if model:
                return costs.get(model, 0.0)
            return sum(costs.values())
    
    async def _check_daily_reset(self) -> None:
        """Reset daily costs if date changed."""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self._last_reset_date:
            self._daily_costs.clear()
            self._last_reset_date = current_date
            logger.info("Daily cost tracking reset")


class BudgetEnforcer:
    """Enforces daily budgets with automatic fallback handling.
    
    Provides budget checking and graceful degradation when limits
    are exceeded.
    """
    
    def __init__(self, fallback_model: str = "gpt-3.5-turbo"):
        self.fallback_model = fallback_model
        self._lock = asyncio.Lock()
        self._daily_budgets: Dict[str, float] = {}
        self._daily_spent: Dict[str, float] = defaultdict(float)
        self._last_reset_date = datetime.now(timezone.utc).date()
    
    async def set_daily_budget(self, tenant_id: str, budget_usd: float) -> None:
        """Set daily budget for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            budget_usd: Daily budget in USD
        """
        if budget_usd < 0:
            raise ValueError("Budget cannot be negative")
        
        async with self._lock:
            self._daily_budgets[tenant_id] = budget_usd
            
            # Update Prometheus
            remaining = budget_usd - self._daily_spent.get(tenant_id, 0.0)
            BUDGET_REMAINING_USD.labels(tenant_id=tenant_id).set(remaining)
            
            # Set next reset time (start of next day UTC)
            import time
            tomorrow = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            BUDGET_DAILY_RESET_TIME.labels(tenant_id=tenant_id).set(
                tomorrow.timestamp()
            )
            
            logger.info(f"Set daily budget for {tenant_id}: ${budget_usd:.2f}")
    
    async def check_budget(self, tenant_id: str, additional_cost: float = 0.0) -> Tuple[bool, float]:
        """Check if tenant is within budget.
        
        Args:
            tenant_id: Tenant identifier
            additional_cost: Cost of upcoming request
            
        Returns:
            Tuple of (is_within_budget, remaining_budget)
        """
        async with self._lock:
            await self._check_daily_reset()
            
            budget = self._daily_budgets.get(tenant_id, float('inf'))
            spent = self._daily_spent.get(tenant_id, 0.0)
            remaining = budget - spent - additional_cost
            
            is_within = remaining >= 0
            
            # Update Prometheus
            BUDGET_REMAINING_USD.labels(tenant_id=tenant_id).set(
                max(0, remaining)
            )
            
            logger.debug(
                f"Budget check for {tenant_id}: ${remaining:.2f} remaining "
                f"({'OK' if is_within else 'EXCEEDED'})"
            )
            
            return is_within, remaining
    
    async def record_spent(self, tenant_id: str, amount: float) -> None:
        """Record spent amount against budget.
        
        Args:
            tenant_id: Tenant identifier
            amount: Amount spent in USD
        """
        async with self._lock:
            await self._check_daily_reset()
            self._daily_spent[tenant_id] += amount
            
            # Update Prometheus
            remaining = self._daily_budgets.get(tenant_id, float('inf')) - self._daily_spent[tenant_id]
            BUDGET_REMAINING_USD.labels(tenant_id=tenant_id).set(
                max(0, remaining)
            )
    
    async def _check_daily_reset(self) -> None:
        """Reset daily spending if date changed."""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self._last_reset_date:
            self._daily_spent.clear()
            self._last_reset_date = current_date
            logger.info("Daily budget tracking reset")


class CostTracker:
    """Main interface for cost tracking functionality.
    
    Combines token counting, cost calculation, and budget enforcement
    into a single convenient interface.
    """
    
    def __init__(self, pricing_config: Optional[PricingConfig] = None):
        self.token_counter = TokenCounter()
        self.cost_calculator = CostCalculator(pricing_config)
        self.budget_enforcer = BudgetEnforcer()
    
    @asynccontextmanager
    async def track_tokens(self, tenant_id: str, model: str, expected_input_tokens: Optional[int] = None):
        """Context manager for tracking LLM token usage.
        
        Automatically tracks tokens and costs, enforcing budget limits.
        
        Args:
            tenant_id: Tenant identifier for budget tracking
            model: Model name being used
            expected_input_tokens: Optional estimate for pre-check
            
        Yields:
            Dictionary to record actual token usage
            
        Example:
            >>> async with cost_tracker.track_tokens("tenant123", "gpt-4") as usage:
            >>>     response = await llm.complete(prompt)
            >>>     usage["input_tokens"] = response.usage.prompt_tokens
            >>>     usage["output_tokens"] = response.usage.completion_tokens
        """
        usage_data = {"input_tokens": 0, "output_tokens": 0}
        
        try:
            # Pre-check budget if we have an estimate
            if expected_input_tokens:
                estimated_cost = await self.cost_calculator.calculate(
                    model, expected_input_tokens, 0, tenant_id
                )
                within_budget, remaining = await self.budget_enforcer.check_budget(
                    tenant_id, estimated_cost
                )
                
                if not within_budget:
                    logger.warning(
                        f"Budget exceeded for {tenant_id}, switching to {self.budget_enforcer.fallback_model}"
                    )
                    model = self.budget_enforcer.fallback_model
                    # Recalculate with fallback model
                    estimated_cost = await self.cost_calculator.calculate(
                        model, expected_input_tokens, 0, tenant_id
                    )
            
            yield usage_data
            
            # Record actual usage
            input_tokens = usage_data.get("input_tokens", 0)
            output_tokens = usage_data.get("output_tokens", 0)
            
            if input_tokens > 0 or output_tokens > 0:
                # Track tokens
                await self.token_counter.track(model, input_tokens, output_tokens, tenant_id)
                
                # Calculate and record cost
                cost = await self.cost_calculator.calculate(
                    model, input_tokens, output_tokens, tenant_id
                )
                
                # Update budget
                await self.budget_enforcer.record_spent(tenant_id, cost)
                
                logger.info(
                    f"Tracked usage for {tenant_id}: {model} - "
                    f"{input_tokens} input, {output_tokens} output tokens, ${cost:.6f}"
                )
            
        except Exception as e:
            logger.error(f"Error in token tracking: {e}")
            raise
    
    async def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive stats for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Dictionary with usage, cost, and budget information
        """
        total_tokens = await self.token_counter.get_total()
        daily_cost = await self.cost_calculator.get_cost_so_far(daily=True)
        _, remaining_budget = await self.budget_enforcer.check_budget(tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "total_tokens_by_model": total_tokens,
            "daily_cost_usd": daily_cost,
            "remaining_budget_usd": max(0, remaining_budget),
            "fallback_model": self.budget_enforcer.fallback_model,
        }


# Global instance for application use
cost_tracker = CostTracker()
