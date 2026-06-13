"""Health check and readiness probe endpoints."""

import structlog  # Structured logging
from fastapi import APIRouter  # Router for grouping endpoints

from memory_graph import __version__  # Package version
from memory_graph.graph.graph_manager import GraphManager  # Graph for node/edge counts
from memory_graph.models.api_responses import HealthResponse  # Response schema

# Module logger for health check operations
logger = structlog.get_logger(__name__)  # Named logger for this module

# Create the router for health endpoints
router = APIRouter(tags=["health"])  # Group under "health" tag in docs


def create_health_router(graph_manager: GraphManager) -> APIRouter:
    """Create and configure the health check router with dependencies."""

    @router.get("/health", response_model=HealthResponse)  # Health check endpoint
    async def health_check() -> HealthResponse:
        """Return the current health status of the service.

        Reports service version and graph statistics for monitoring.
        """
        logger.debug("health_check_requested")  # Log health check access

        return HealthResponse(  # Build the health response
            status="healthy",  # Service is operational
            version=__version__,  # Current package version
            node_count=graph_manager.node_count,  # Total nodes in graph
            edge_count=graph_manager.edge_count,  # Total edges in graph
        )

    @router.get("/readiness")  # Readiness probe endpoint
    async def readiness_check() -> dict:
        """Check if the service is ready to accept traffic.

        Verifies the graph is loaded and the service is fully initialized.
        """
        is_ready = graph_manager.graph is not None  # Graph must be initialized

        if not is_ready:  # Service not yet ready
            logger.warning("readiness_check_failed")  # Log unready state
            return {"status": "not_ready"}  # Indicate not ready

        return {"status": "ready"}  # Service is ready

    return router  # Return the configured router
