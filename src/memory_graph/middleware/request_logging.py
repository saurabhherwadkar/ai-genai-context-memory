"""Request/response logging middleware for HTTP traffic observability."""

import time  # Timing for request duration measurement

import structlog  # Structured logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint  # Middleware base
from starlette.requests import Request  # HTTP request type
from starlette.responses import Response  # HTTP response type

# Module logger for request logging
logger = structlog.get_logger(__name__)  # Named logger for this module


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming requests and outgoing responses with timing information.

    Captures method, path, status code, and duration for each HTTP exchange.
    Supports configurable exclusion of health check endpoints from logging.
    """

    def __init__(self, app: "ASGIApp", exclude_paths: list[str] | None = None) -> None:
        """Initialize the middleware with optional path exclusions."""
        super().__init__(app)  # Initialize base middleware
        self._exclude_paths = set(exclude_paths or ["/health", "/readiness"])  # Paths to skip

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process each request through the logging middleware.

        Records start time, delegates to the next handler, and logs
        the complete request/response cycle with duration.
        """
        # Skip logging for excluded paths (health checks, etc.)
        if request.url.path in self._exclude_paths:  # Path is excluded
            return await call_next(request)  # Pass through without logging

        # Record the request start time
        start_time = time.perf_counter()  # High-resolution timer

        # Extract request metadata for logging
        method = request.method  # HTTP method (GET, POST, etc.)
        path = request.url.path  # URL path
        client_ip = request.client.host if request.client else "unknown"  # Client IP address

        # Log the incoming request
        logger.info(  # Log at info level
            "request_started",
            method=method,
            path=path,
            client_ip=client_ip,
        )

        # Process the request through the application
        try:  # Wrap call_next to capture errors
            response = await call_next(request)  # Delegate to next handler
        except Exception as exc:  # Unhandled error during processing
            # Calculate duration even on error
            duration_ms = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
            logger.error(  # Log the failed request
                "request_failed",
                method=method,
                path=path,
                duration_ms=round(duration_ms, 2),
                error=str(exc),
            )
            raise  # Re-raise for error handler middleware

        # Calculate request duration
        duration_ms = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds

        # Log the completed request with response info
        log_level = "info" if response.status_code < 400 else "warning"  # Choose level by status
        log_method = getattr(logger, log_level)  # Get the appropriate log method
        log_method(  # Log the completed request
            "request_completed",
            method=method,
            path=path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client_ip=client_ip,
        )

        return response  # Return the response to the client
