"""Global exception handler middleware for consistent error responses."""

import traceback  # Stack trace formatting for error logging

import structlog  # Structured logging
from fastapi import FastAPI, Request  # Web framework types
from fastapi.responses import JSONResponse  # JSON response builder

# Module logger for error handling
logger = structlog.get_logger(__name__)  # Named logger for this module


class AppError(Exception):
    """Base application error with structured error information."""

    def __init__(
        self,
        message: str,  # Human-readable error message
        error_code: str = "internal_error",  # Machine-readable error identifier
        status_code: int = 500,  # HTTP status code
        details: dict | None = None,  # Optional additional context
    ) -> None:
        """Initialize the application error with all error metadata."""
        super().__init__(message)  # Initialize base exception
        self.message = message  # Store the error message
        self.error_code = error_code  # Store the error code
        self.status_code = status_code  # Store the HTTP status
        self.details = details  # Store optional details


class NotFoundError(AppError):
    """Error raised when a requested resource does not exist."""

    def __init__(self, resource: str, resource_id: str) -> None:
        """Initialize with the resource type and ID that was not found."""
        message = f"{resource} with id '{resource_id}' not found"  # Build descriptive message
        super().__init__(  # Delegate to base class
            message=message,  # The error message
            error_code="not_found",  # Standard not-found code
            status_code=404,  # HTTP 404
            details={"resource": resource, "id": resource_id},  # Extra context
        )


class ValidationError(AppError):
    """Error raised when input validation fails."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Initialize with validation error details."""
        super().__init__(  # Delegate to base class
            message=message,  # The validation error message
            error_code="validation_error",  # Standard validation code
            status_code=422,  # HTTP 422 Unprocessable Entity
            details=details,  # Validation detail context
        )


class CapacityError(AppError):
    """Error raised when a resource limit is reached."""

    def __init__(self, message: str) -> None:
        """Initialize with capacity limit message."""
        super().__init__(  # Delegate to base class
            message=message,  # The capacity error message
            error_code="capacity_exceeded",  # Capacity error code
            status_code=507,  # HTTP 507 Insufficient Storage
        )


def register_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI application."""

    @app.exception_handler(AppError)  # Handle all application errors
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Handle known application errors with structured responses."""
        logger.error(  # Log the application error
            "application_error",
            error_code=exc.error_code,
            message=exc.message,
            status_code=exc.status_code,
            path=request.url.path,
        )
        return JSONResponse(  # Return structured error response
            status_code=exc.status_code,  # Use the error's status code
            content={
                "error": exc.error_code,  # Machine-readable error type
                "message": exc.message,  # Human-readable message
                "details": exc.details,  # Optional extra context
            },
        )

    @app.exception_handler(ValueError)  # Handle validation value errors
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError exceptions as validation errors."""
        logger.warning(  # Log at warning level for input errors
            "validation_error",
            message=str(exc),
            path=request.url.path,
        )
        return JSONResponse(  # Return 422 for validation failures
            status_code=422,  # Unprocessable Entity
            content={
                "error": "validation_error",  # Error type
                "message": str(exc),  # Error description
                "details": None,  # No additional details
            },
        )

    @app.exception_handler(Exception)  # Catch-all for unhandled exceptions
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions with a safe generic response."""
        # Log the full stack trace for debugging
        error_traceback = traceback.format_exc()  # Format the full traceback
        logger.error(  # Log at error level with full details
            "unhandled_exception",
            exception_type=type(exc).__name__,
            message=str(exc),
            path=request.url.path,
            traceback=error_traceback,
        )
        return JSONResponse(  # Return generic 500 without leaking internals
            status_code=500,  # Internal Server Error
            content={
                "error": "internal_error",  # Generic error type
                "message": "An unexpected error occurred",  # Safe message
                "details": None,  # Never expose internal details
            },
        )

    logger.info("error_handlers_registered")  # Log successful registration
