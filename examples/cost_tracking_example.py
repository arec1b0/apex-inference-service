"""
Example demonstrating cost tracking integration.

This script shows how to integrate the cost tracker into an inference service.
Run with: python examples/cost_tracking_example.py
"""
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any

# Mock FastAPI request for demonstration
class MockRequest:
    def __init__(self, tenant_id: str):
        self.state = MockState(tenant_id)

class MockState:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

# Mock LLM service
class MockLLMService:
    async def complete(self, prompt: str, model: str) -> Dict[str, Any]:
        """Mock LLM completion with realistic token counts."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock response with token usage
        input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        output_tokens = len(prompt.split()) * 0.8  # Rough estimate
        
        return {
            "text": f"Response to: {prompt[:50]}...",
            "model": model,
            "usage": {
                "prompt_tokens": int(input_tokens),
                "completion_tokens": int(output_tokens)
            }
        }

async def demonstrate_cost_tracking():
    """Demonstrate cost tracking functionality."""
    from src.app.monitoring.cost_tracker import cost_tracker
    
    # Initialize
    llm_service = MockLLMService()
    
    # Set up tenant budgets
    print("🔧 Setting up tenant budgets...")
    await cost_tracker.budget_enforcer.set_daily_budget("tenant-abc", 1.00)  # $1/day
    await cost_tracker.budget_enforcer.set_daily_budget("tenant-xyz", 0.50)  # $0.50/day
    
    # Example 1: Normal usage within budget
    print("\n📊 Example 1: Normal usage within budget")
    request = MockRequest("tenant-abc")
    
    async with cost_tracker.track_tokens(
        tenant_id=request.state.tenant_id,
        model="gpt-4",
        expected_input_tokens=100
    ) as usage:
        response = await llm_service.complete(
            "Explain quantum computing in simple terms",
            "gpt-4"
        )
        
        usage["input_tokens"] = response["usage"]["prompt_tokens"]
        usage["output_tokens"] = response["usage"]["completion_tokens"]
        
        print(f"  Model used: {response['model']}")
        print(f"  Tokens: {usage['input_tokens']} input, {usage['output_tokens']} output")
    
    # Check remaining budget
    _, remaining = await cost_tracker.budget_enforcer.check_budget("tenant-abc")
    print(f"  Remaining budget: ${remaining:.4f}")
    
    # Example 2: Budget exceeded with fallback
    print("\n⚠️  Example 2: Budget exceeded - automatic fallback")
    request = MockRequest("tenant-xyz")
    
    async with cost_tracker.track_tokens(
        tenant_id=request.state.tenant_id,
        model="gpt-4",  # Expensive model
        expected_input_tokens=1000  # Large request
    ) as usage:
        response = await llm_service.complete(
            "Write a detailed business plan for a startup", 
            "gpt-4"  # Will be switched to fallback
        )
        
        usage["input_tokens"] = response["usage"]["prompt_tokens"]
        usage["output_tokens"] = response["usage"]["completion_tokens"]
        
        print(f"  Model used: {response['model']} (fallback automatically applied)")
        print(f"  Tokens: {usage['input_tokens']} input, {usage['output_tokens']} output")
    
    # Example 3: Get tenant statistics
    print("\n📈 Example 3: Tenant statistics")
    stats = await cost_tracker.get_tenant_stats("tenant-abc")
    
    print(f"  Tenant: {stats['tenant_id']}")
    print(f"  Daily cost: ${stats['daily_cost_usd']:.6f}")
    print(f"  Remaining budget: ${stats['remaining_budget_usd']:.4f}")
    print(f"  Token usage by model: {stats['total_tokens_by_model']}")
    
    # Example 4: Multiple requests tracking
    print("\n🔄 Example 4: Multiple requests batch processing")
    
    prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "How does backpropagation work?",
        "What are transformers?",
        "Describe attention mechanism"
    ]
    
    total_cost = 0
    for i, prompt in enumerate(prompts, 1):
        async with cost_tracker.track_tokens("tenant-abc", "gpt-3.5-turbo") as usage:
            response = await llm_service.complete(prompt, "gpt-3.5-turbo")
            usage["input_tokens"] = response["usage"]["prompt_tokens"]
            usage["output_tokens"] = response["usage"]["completion_tokens"]
            
            cost = await cost_tracker.cost_calculator.calculate(
                response["model"],
                usage["input_tokens"],
                usage["output_tokens"],
                "tenant-abc"
            )
            total_cost += cost
            
            print(f"  Request {i}: ${cost:.6f}")
    
    print(f"\n  Total batch cost: ${total_cost:.6f}")
    
    # Final statistics
    print("\n📊 Final Statistics:")
    final_stats = await cost_tracker.get_tenant_stats("tenant-abc")
    print(f"  Total tokens used: {sum(t['input'] + t['output'] for t in final_stats['total_tokens_by_model'].values())}")
    print(f"  Total cost: ${final_stats['daily_cost_usd']:.6f}")
    print(f"  Budget remaining: ${final_stats['remaining_budget_usd']:.4f}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Cost Tracking Integration Example")
    print("=" * 60)
    
    asyncio.run(demonstrate_cost_tracking())
    
    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("1. Check Prometheus metrics at http://localhost:8000/metrics")
    print("2. Set up Grafana dashboard for visualization")
    print("3. Configure alerts for budget thresholds")
    print("4. Review integration guide at docs/cost_tracking_integration.md")
