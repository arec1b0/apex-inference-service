"""Circuit breaker для защиты от каскадных падений."""
import pybreaker
from prometheus_client import Gauge
from loguru import logger

CIRCUIT_STATE = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state: 0=closed, 1=open, 2=half_open",
    ["service"],
)

def on_circuit_open(cb, exc):
    """Called when circuit opens."""
    logger.error(f"🔴 Circuit OPEN for {cb.name}: {exc}")
    CIRCUIT_STATE.labels(service=cb.name).set(1)

def on_circuit_close(cb):
    """Called when circuit closes."""
    logger.info(f"🟢 Circuit CLOSED for {cb.name}")
    CIRCUIT_STATE.labels(service=cb.name).set(0)

def on_circuit_half_open(cb):
    """Called when circuit enters half-open state."""
    logger.warning(f"🟡 Circuit HALF-OPEN for {cb.name}")
    CIRCUIT_STATE.labels(service=cb.name).set(2)

# Model inference circuit breaker
model_breaker = pybreaker.CircuitBreaker(
    fail_max=5,              # Open after 5 consecutive failures
    reset_timeout=60,        # Try again after 60 seconds
    name="model_inference",
    listeners=[
        pybreaker.CircuitOpenedListener(on_circuit_open),
        pybreaker.CircuitClosedListener(on_circuit_close),
        pybreaker.CircuitHalfOpenListener(on_circuit_half_open),
    ],
)
