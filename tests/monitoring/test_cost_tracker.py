"""Unit tests for cost tracking functionality.

Tests cover token counting, cost calculation, budget enforcement,
and integration scenarios.
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from src.app.monitoring.cost_tracker import (
    TokenCounter,
    CostCalculator,
    BudgetEnforcer,
    CostTracker,
    PricingConfig,
    cost_tracker
)


class TestTokenCounter:
    """Test token counting functionality."""
    
    @pytest.fixture
    def counter(self):
        """Create fresh token counter for each test."""
        return TokenCounter()
    
    @pytest.mark.asyncio
    async def test_token_counting_normal(self, counter):
        """Test normal token counting behavior."""
        await counter.track("gpt-4", 100, 50, "tenant1")
        await counter.track("gpt-4", 200, 100, "tenant1")
        
        totals = await counter.get_total("gpt-4")
        assert totals["input"] == 300
        assert totals["output"] == 150
    
    @pytest.mark.asyncio
    async def test_token_counting_multiple_models(self, counter):
        """Test tracking multiple models separately."""
        await counter.track("gpt-4", 100, 50, "tenant1")
        await counter.track("gpt-3.5-turbo", 200, 100, "tenant1")
        
        all_totals = await counter.get_total()
        assert all_totals["gpt-4"]["input"] == 100
        assert all_totals["gpt-3.5-turbo"]["input"] == 200
    
    @pytest.mark.asyncio
    async def test_negative_tokens_rejected(self, counter):
        """Test that negative token counts raise error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            await counter.track("gpt-4", -10, 50, "tenant1")
    
    @pytest.mark.asyncio
    async def test_daily_reset(self, counter):
        """Test daily reset functionality."""
        # Track some tokens
        await counter.track("gpt-4", 100, 50, "tenant1")
        
        # Verify tokens tracked
        totals = await counter.get_total()
        assert totals["gpt-4"]["input"] == 100
        
        # Manually reset
        await counter.reset_daily()
        
        # Verify cleared
        totals = await counter.get_total()
        assert totals == {}
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_model(self, counter):
        """Test getting totals for model that hasn't been used."""
        totals = await counter.get_total("nonexistent-model")
        assert totals == {"input": 0, "output": 0}


class TestCostCalculator:
    """Test cost calculation functionality."""
    
    @pytest.fixture
    def calculator(self):
        """Create cost calculator with test pricing."""
        pricing = PricingConfig()
        return CostCalculator(pricing)
    
    @pytest.mark.asyncio
    async def test_cost_calculation_openai(self, calculator):
        """Test cost calculation for OpenAI models."""
        # GPT-4: $0.03/1K input, $0.06/1K output
        cost = await calculator.calculate("gpt-4", 1000, 500, "tenant1")
        
        # Expected: (1000/1000)*0.03 + (500/1000)*0.06 = 0.03 + 0.03 = 0.06
        assert cost == 0.06
    
    @pytest.mark.asyncio
    async def test_cost_calculation_anthropic(self, calculator):
        """Test cost calculation for Anthropic models."""
        # Claude-3 Sonnet: $0.003/1K input, $0.015/1K output
        cost = await calculator.calculate("claude-3-sonnet", 2000, 1000, "tenant1")
        
        # Expected: (2000/1000)*0.003 + (1000/1000)*0.015 = 0.006 + 0.015 = 0.021
        assert cost == 0.021
    
    @pytest.mark.asyncio
    async def test_cost_calculation_local(self, calculator):
        """Test cost calculation for local models."""
        # Llama-2-7b: $0.0001/1K tokens
        cost = await calculator.calculate("llama-2-7b", 5000, 5000, "tenant1")
        
        # Expected: (5000/1000)*0.0001 + (5000/1000)*0.0001 = 0.0005 + 0.0005 = 0.001
        assert cost == 0.001
    
    @pytest.mark.asyncio
    async def test_cost_calculation_unknown_model(self, calculator):
        """Test error for unknown model."""
        with pytest.raises(ValueError, match="not found in pricing config"):
            await calculator.calculate("unknown-model", 100, 50, "tenant1")
    
    @pytest.mark.asyncio
    async def test_get_cost_so_far(self, calculator):
        """Test accumulating costs over time."""
        await calculator.calculate("gpt-4", 1000, 500, "tenant1")
        await calculator.calculate("gpt-4", 500, 250, "tenant1")
        await calculator.calculate("gpt-3.5-turbo", 1000, 1000, "tenant1")
        
        total_cost = await calculator.get_cost_so_far()
        assert total_cost == 0.06 + 0.03 + 0.0035  # 0.0935
        
        gpt4_cost = await calculator.get_cost_so_far(model="gpt-4")
        assert gpt4_cost == 0.09
    
    @pytest.mark.asyncio
    async def test_daily_cost_tracking(self, calculator):
        """Test daily cost accumulation."""
        await calculator.calculate("gpt-4", 1000, 500, "tenant1")
        
        daily_cost = await calculator.get_cost_so_far(daily=True)
        assert daily_cost == 0.06
        
        total_cost = await calculator.get_cost_so_far(daily=False)
        assert total_cost == 0.06  # Same on first day


class TestBudgetEnforcer:
    """Test budget enforcement functionality."""
    
    @pytest.fixture
    def enforcer(self):
        """Create budget enforcer for testing."""
        return BudgetEnforcer(fallback_model="gpt-3.5-turbo")
    
    @pytest.mark.asyncio
    async def test_set_and_check_budget(self, enforcer):
        """Test setting and checking budget limits."""
        await enforcer.set_daily_budget("tenant1", 10.0)
        
        # Should be within budget initially
        within, remaining = await enforcer.check_budget("tenant1")
        assert within is True
        assert remaining == 10.0
        
        # After spending $5
        await enforcer.record_spent("tenant1", 5.0)
        within, remaining = await enforcer.check_budget("tenant1")
        assert within is True
        assert remaining == 5.0
    
    @pytest.mark.asyncio
    async def test_budget_exceeded(self, enforcer):
        """Test budget exceeded detection."""
        await enforcer.set_daily_budget("tenant1", 1.0)
        await enforcer.record_spent("tenant1", 1.5)
        
        within, remaining = await enforcer.check_budget("tenant1")
        assert within is False
        assert remaining == -0.5
    
    @pytest.mark.asyncio
    async def test_check_budget_with_future_cost(self, enforcer):
        """Test budget checking with anticipated future cost."""
        await enforcer.set_daily_budget("tenant1", 1.0)
        await enforcer.record_spent("tenant1", 0.8)
        
        # Check if we can afford $0.3 more
        within, remaining = await enforcer.check_budget("tenant1", 0.3)
        assert within is False
        assert remaining == -0.1  # 1.0 - 0.8 - 0.3
        
        # Check if we can afford $0.1 more
        within, remaining = await enforcer.check_budget("tenant1", 0.1)
        assert within is True
        assert remaining == 0.1
    
    @pytest.mark.asyncio
    async def test_multiple_tenants(self, enforcer):
        """Test budget tracking for multiple tenants."""
        await enforcer.set_daily_budget("tenant1", 10.0)
        await enforcer.set_daily_budget("tenant2", 20.0)
        
        await enforcer.record_spent("tenant1", 5.0)
        await enforcer.record_spent("tenant2", 15.0)
        
        _, remaining1 = await enforcer.check_budget("tenant1")
        _, remaining2 = await enforcer.check_budget("tenant2")
        
        assert remaining1 == 5.0
        assert remaining2 == 5.0
    
    @pytest.mark.asyncio
    async def test_negative_budget_rejected(self, enforcer):
        """Test that negative budgets are rejected."""
        with pytest.raises(ValueError, match="cannot be negative"):
            await enforcer.set_daily_budget("tenant1", -10.0)
    
    @pytest.mark.asyncio
    async def test_unlimited_budget_default(self, enforcer):
        """Test that tenants have unlimited budget by default."""
        within, remaining = await enforcer.check_budget("tenant1", 1000000.0)
        assert within is True
        assert remaining == float('inf')


class TestCostTracker:
    """Test integrated cost tracking functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create cost tracker for testing."""
        pricing = PricingConfig()
        return CostTracker(pricing)
    
    @pytest.mark.asyncio
    async def test_track_tokens_context_manager(self, tracker):
        """Test the track_tokens context manager."""
        await tracker.budget_enforcer.set_daily_budget("tenant1", 10.0)
        
        async with tracker.track_tokens("tenant1", "gpt-4") as usage:
            usage["input_tokens"] = 100
            usage["output_tokens"] = 50
        
        # Verify tokens tracked
        totals = await tracker.token_counter.get_total("gpt-4")
        assert totals["input"] == 100
        assert totals["output"] == 50
        
        # Verify cost calculated
        cost = await tracker.cost_calculator.get_cost_so_far()
        assert cost == 0.006  # (100/1000)*0.03 + (50/1000)*0.06
        
        # Verify budget updated
        _, remaining = await tracker.budget_enforcer.check_budget("tenant1")
        assert remaining == 9.994  # 10.0 - 0.006
    
    @pytest.mark.asyncio
    async def test_fallback_on_budget_exceeded(self, tracker):
        """Test automatic fallback model when budget exceeded."""
        await tracker.budget_enforcer.set_daily_budget("tenant1", 0.001)
        
        async with tracker.track_tokens("tenant1", "gpt-4", expected_input_tokens=1000) as usage:
            # Should automatically switch to fallback model
            usage["input_tokens"] = 1000
            usage["output_tokens"] = 500
        
        # Verify fallback model was used
        totals = await tracker.token_counter.get_total("gpt-3.5-turbo")
        assert totals["input"] == 1000
        
        # GPT-4 should have no usage
        gpt4_totals = await tracker.token_counter.get_total("gpt-4")
        assert gpt4_totals["input"] == 0
    
    @pytest.mark.asyncio
    async def test_get_tenant_stats(self, tracker):
        """Test getting comprehensive tenant statistics."""
        await tracker.budget_enforcer.set_daily_budget("tenant1", 10.0)
        
        async with tracker.track_tokens("tenant1", "gpt-4") as usage:
            usage["input_tokens"] = 1000
            usage["output_tokens"] = 500
        
        stats = await tracker.get_tenant_stats("tenant1")
        
        assert stats["tenant_id"] == "tenant1"
        assert stats["daily_cost_usd"] == 0.09
        assert stats["remaining_budget_usd"] == 9.91
        assert stats["fallback_model"] == "gpt-3.5-turbo"
        assert "gpt-4" in stats["total_tokens_by_model"]
    
    @pytest.mark.asyncio
    async def test_error_handling_in_context(self, tracker):
        """Test that errors in context are properly handled."""
        with pytest.raises(ValueError, match="test error"):
            async with tracker.track_tokens("tenant1", "gpt-4") as usage:
                raise ValueError("test error")
        
        # Verify no tokens were tracked
        totals = await tracker.token_counter.get_total()
        assert totals == {}


class TestPricingConfig:
    """Test pricing configuration."""
    
    def test_get_price_openai(self):
        """Test getting OpenAI model prices."""
        config = PricingConfig()
        
        assert config.get_price("gpt-4", "input") == 0.03
        assert config.get_price("gpt-4", "output") == 0.06
        assert config.get_price("gpt-3.5-turbo", "input") == 0.0015
    
    def test_get_price_anthropic(self):
        """Test getting Anthropic model prices."""
        config = PricingConfig()
        
        assert config.get_price("claude-3-opus", "input") == 0.015
        assert config.get_price("claude-3-sonnet", "output") == 0.015
    
    def test_get_price_local(self):
        """Test getting local model prices."""
        config = PricingConfig()
        
        assert config.get_price("llama-2-7b", "input") == 0.0001
        assert config.get_price("llama-2-70b", "output") == 0.0005
    
    def test_get_price_unknown_model(self):
        """Test error for unknown model."""
        config = PricingConfig()
        
        with pytest.raises(ValueError):
            config.get_price("unknown-model", "input")


class TestGlobalInstance:
    """Test the global cost_tracker instance."""
    
    @pytest.mark.asyncio
    async def test_global_instance_available(self):
        """Test that global instance is properly initialized."""
        assert cost_tracker is not None
        assert isinstance(cost_tracker, CostTracker)
        assert cost_tracker.token_counter is not None
        assert cost_tracker.cost_calculator is not None
        assert cost_tracker.budget_enforcer is not None


@pytest.mark.asyncio
async def test_budget_reset_daily():
    """Test that budget tracking resets daily."""
    import time
    from unittest.mock import patch
    
    enforcer = BudgetEnforcer()
    await enforcer.set_daily_budget("tenant1", 10.0)
    await enforcer.record_spent("tenant1", 5.0)
    
    # Mock time to be tomorrow
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    with patch('src.app.monitoring.cost_tracker.datetime') as mock_datetime:
        mock_datetime.now.return_value = tomorrow
        mock_datetime.now.side_effect = lambda tz=timezone.utc: tomorrow
        
        # Check should show fresh budget
        within, remaining = await enforcer.check_budget("tenant1")
        assert within is True
        assert remaining == 10.0
