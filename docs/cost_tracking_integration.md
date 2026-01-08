# Cost Tracking Integration Guide

## Overview
The cost tracking system provides production-grade monitoring of LLM inference costs with automatic budget enforcement and fallback handling.

## Integration Example

### 1. FastAPI Endpoint Integration

```python
from fastapi import Request, APIRouter
from src.app.monitoring.cost_tracker import cost_tracker
from src.app.services.llm_service import LLMService

router = APIRouter()
llm_service = LLMService()

@router.post("/api/v1/chat/completions")
async def chat_completion(request: Request, body: ChatCompletionRequest):
    """
    OpenAI-compatible chat endpoint with cost tracking.
    
    Automatically tracks tokens, calculates costs, and enforces budgets.
    Falls back to cheaper model if budget exceeded.
    """
    # Extract tenant from request (e.g., from JWT token)
    tenant_id = getattr(request.state, "tenant_id", "default")
    
    # Estimate input tokens for pre-budget check
    estimated_input = len(body.messages) * 100  # Rough estimate
    
    async with cost_tracker.track_tokens(
        tenant_id=tenant_id,
        model=body.model,
        expected_input_tokens=estimated_input
    ) as usage:
        # Make the actual LLM call
        response = await llm_service.complete(
            messages=body.messages,
            model=body.model,
            temperature=body.temperature
        )
        
        # Record actual token usage
        usage["input_tokens"] = response.usage.prompt_tokens
        usage["output_tokens"] = response.usage.completion_tokens
        
        # Add cost info to response headers
        cost = await cost_tracker.cost_calculator.calculate(
            model=response.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            tenant_id=tenant_id
        )
        
        return response
```

### 2. Tenant Budget Setup

```python
# On tenant registration or update
async def set_tenant_budget(tenant_id: str, monthly_budget_usd: float):
    """Set daily budget from monthly allocation."""
    daily_budget = monthly_budget_usd / 30.44  # Average month length
    await cost_tracker.budget_enforcer.set_daily_budget(
        tenant_id, 
        daily_budget
    )
    
    # Log the change
    logger.info(f"Set budget for {tenant_id}: ${daily_budget:.2f}/day")

# Example usage
await set_tenant_budget("acme-corp", 1000.0)  # $1000/month → ~$32.86/day
```

### 3. Monitoring Dashboard Integration

```python
@router.get("/api/v1/usage/stats")
async def get_usage_stats(request: Request):
    """Get comprehensive usage statistics for tenant."""
    tenant_id = getattr(request.state, "tenant_id", "default")
    
    stats = await cost_tracker.get_tenant_stats(tenant_id)
    
    return {
        "current_usage": {
            "tokens_by_model": stats["total_tokens_by_model"],
            "cost_today_usd": stats["daily_cost_usd"],
            "budget_remaining_usd": stats["remaining_budget_usd"],
            "budget_utilization_percent": (
                (1 - stats["remaining_budget_usd"] / 32.86) * 100
                if stats["remaining_budget_usd"] != float('inf') else 0
            )
        },
        "model_info": {
            "primary_model": "gpt-4",
            "fallback_model": stats["fallback_model"]
        }
    }
```

### 4. Custom Pricing Configuration

```python
from src.app.monitoring.cost_tracker import PricingConfig, CostTracker

# Custom pricing for your models
custom_pricing = PricingConfig()
custom_pricing.openai["custom-model"] = {"input": 0.02, "output": 0.04}
custom_pricing.local["custom-local"] = {"input": 0.0002, "output": 0.0002}

# Create tracker with custom pricing
custom_tracker = CostTracker(custom_pricing)
```

## Integration Checklist

### Pre-Integration
- [ ] Review current LLM service integration points
- [ ] Identify tenant identification mechanism (JWT, API key, etc.)
- [ ] Determine budget allocation strategy per tenant
- [ ] Set up Prometheus metrics endpoint if not already exists
- [ ] Configure alerting thresholds for budget overruns

### Code Integration
- [ ] Wrap all LLM calls with `cost_tracker.track_tokens()` context manager
- [ ] Add tenant ID extraction middleware if not present
- [ ] Implement budget setup flow for new tenants
- [ ] Add cost headers to API responses (optional but recommended)
- [ ] Create usage statistics endpoint for customer dashboard
- [ ] Add logging for budget exceeded events

### Configuration
- [ ] Set appropriate pricing for your model mix
- [ ] Configure fallback model (usually cheapest option)
- [ ] Set up daily budget reset schedule
- [ ] Configure Prometheus metric labels and retention
- [ ] Set up alerting on `budget_remaining_usd` metric

### Testing
- [ ] Unit tests pass: `pytest tests/monitoring/test_cost_tracker.py`
- [ ] Integration tests with actual LLM provider
- [ ] Budget enforcement test with limit scenarios
- [ ] Fallback model verification
- [ ] Daily reset functionality verification
- [ ] Load testing with concurrent requests

### Monitoring & Alerting
- [ ] Prometheus metrics exporting correctly
- [ ] Grafana dashboard configured with:
  - Token usage by model
  - Cost trends per tenant
  - Budget utilization gauges
  - Fallback model usage alerts
- [ ] Alerts configured for:
  - Budget at 80% utilization
  - Budget exceeded
  - Unusual cost spikes
  - High fallback model usage

### Production Rollout
- [ ] Feature flag for gradual rollout
- [ ] Database migration for tenant budgets (if persistent storage needed)
- [ ] Documentation for API changes
- [ ] Customer communication about cost tracking
- [ ] Run book for budget exceeded incidents

## Prometheus Metrics Reference

### Counters
- `llm_tokens_total{model, direction, tenant_id}` - Total tokens processed
- `llm_cost_usd_total{model, tenant_id}` - Total cost incurred

### Gauges
- `budget_remaining_usd{tenant_id}` - Remaining budget for period
- `budget_daily_reset_time{tenant_id}` - Unix timestamp of next reset

## Common Patterns

### 1. Batch Processing
```python
async def process_batch(tenant_id: str, items: List[Item]):
    """Process multiple items with cost tracking."""
    total_tokens = 0
    
    async with cost_tracker.track_tokens(tenant_id, "gpt-4") as usage:
        for item in items:
            result = await process_item(item)
            total_tokens += result.tokens
    
    usage["input_tokens"] = total_tokens
    usage["output_tokens"] = 0  # No output for batch processing
```

### 2. Streaming Responses
```python
async def stream_response(tenant_id: str, model: str):
    """Handle streaming with cost tracking."""
    async with cost_tracker.track_tokens(tenant_id, model) as usage:
        input_tokens = count_input_tokens()
        
        async for chunk in llm_stream():
            yield chunk
            # Accumulate output tokens as they arrive
            if chunk.usage:
                usage["output_tokens"] += chunk.usage.completion_tokens
        
        usage["input_tokens"] = input_tokens
```

### 3. Multi-Model Routing
```python
async def smart_route(tenant_id: str, complexity: str):
    """Route to appropriate model based on complexity and budget."""
    model_map = {
        "simple": "gpt-3.5-turbo",
        "complex": "gpt-4",
        "critical": "gpt-4-turbo"
    }
    
    preferred_model = model_map.get(complexity, "gpt-3.5-turbo")
    
    async with cost_tracker.track_tokens(tenant_id, preferred_model) as usage:
        # Cost tracker will auto-fallback if needed
        response = await llm_service.complete(...)
        usage["input_tokens"] = response.usage.prompt_tokens
        usage["output_tokens"] = response.usage.completion_tokens
        return response
```

## Troubleshooting

### Issue: Budget not enforced
- Check tenant ID is properly set in request state
- Verify budget has been set for the tenant
- Check logs for budget enforcement messages

### Issue: Missing metrics
- Ensure Prometheus metrics endpoint is configured
- Check that metrics are being exported on `/metrics`
- Verify metric labels match your queries

### Issue: Fallback not triggered
- Check budget_enforcer.fallback_model is set correctly
- Verify budget is actually exceeded
- Check logs for fallback messages

### Issue: Daily reset not working
- Verify system timezone is UTC
- Check that `datetime.now(timezone.utc)` is used
- Review last reset date in logs
