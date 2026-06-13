"""Rate limiting middleware for API request throttling."""

import structlog  # Structured logging
from fastapi import FastAPI, Request  # Web framework types
from slowapi import Limiter  # Rate limiting library for FastAPI
from slowapi.errors import RateLimitExceeded  # Rate limit exception
from slowapi.middleware import SlowAPIMiddleware  # Middleware integration
from slowapi.util import get_remote_address  # Client IP extraction
from starlette.responses import JSONResponse  # HTTP response type

from memory_graph.config.settings import RateLimitSettings  # Rate limit configuration

# Module logger for rate limiting events
logger = structlog.get_logger(__name__)  # Named logger for this module


def create_rate_limiter(settings: RateLimitSettings) -> Limiter:
    """Create a configured rate limiter instance.

    Uses client IP address as the rate limit key.
    """
    # Build the rate limit string in slowapi format
    rate_limit_string = f"{settings.requests_per_minute}/minute"  # Format: count/period

    limiter = Limiter(  # Create the limiter instance
        key_func=get_remote_address,  # Use client IP as the key
        default_limits=[rate_limit_string],  # Apply default limit globally
    )

    logger.info(  # Log rate limiter configuration
        "rate_limiter_created",
        limit=rate_limit_string,
        burst_size=settings.burst_size,
    )
    return limiter  # Return the configured limiter


def register_rate_limiter(app: FastAPI, limiter: Limiter) -> None:
    """Register the rate limiter middleware and error handler on the FastAPI app."""
    # Store limiter on app state for access in route decorators
    app.state.limiter = limiter  # Attach limiter to app state

    # Add the SlowAPI middleware for request interception
    app.add_middleware(SlowAPIMiddleware)  # Register the middleware

    # Register custom error handler for rate limit exceeded responses
    @app.exception_handler(RateLimitExceeded)  # Handle rate limit errors
    async def rate_limit_exceeded_handler(
        request: Request,  # The incoming request that was limited
        exc: RateLimitExceeded,  # The rate limit exception
    ) -> JSONResponse:
        """Return a 429 response when rate limit is exceeded."""
        client_ip = get_remote_address(request)  # Get the client IP
        logger.warning(  # Log the rate limit event
            "rate_limit_exceeded",
            client_ip=client_ip,
            path=request.url.path,
        )
        return JSONResponse(  # Return structured error response
            status_code=429,  # HTTP 429 Too Many Requests
            content={
                "error": "rate_limit_exceeded",  # Error type identifier
                "message": "Too many requests. Please retry later.",  # Human message
            },
        )

    logger.info("rate_limiter_registered")  # Log successful registration
