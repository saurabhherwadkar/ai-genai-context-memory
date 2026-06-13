"""Async task queue for background memory extraction processing."""

import asyncio  # Async queue and task management

import structlog  # Structured logging

from memory_graph.config.settings import ExtractionSettings  # Extraction configuration
from memory_graph.models.proxy_models import ConversationRecord  # Queue item type

# Module logger for queue operations
logger = structlog.get_logger(__name__)  # Named logger for this module


class ExtractionQueue:
    """Manages an async queue for background memory extraction tasks.

    Accepts conversation records and processes them via worker coroutines
    with configurable concurrency and retry behavior.
    """

    def __init__(self, settings: ExtractionSettings) -> None:
        """Initialize the extraction queue with configuration settings."""
        self._enabled = settings.enabled  # Whether extraction is active
        self._concurrency = settings.concurrency  # Max concurrent workers
        self._queue: asyncio.Queue[ConversationRecord] = asyncio.Queue()  # Async FIFO queue
        self._workers: list[asyncio.Task] = []  # Running worker tasks
        self._processor_callback = None  # Callback for processing items
        self._is_running = False  # Queue processing state
        logger.info(  # Log initialization
            "extraction_queue_initialized",
            enabled=self._enabled,
            concurrency=self._concurrency,
        )

    async def enqueue(self, record: ConversationRecord) -> None:
        """Add a conversation record to the extraction queue.

        Does nothing if extraction is disabled.
        """
        if not self._enabled:  # Extraction disabled
            logger.debug("extraction_disabled_skipping_enqueue")  # Log skip
            return  # Do nothing

        await self._queue.put(record)  # Add to the async queue
        logger.debug(  # Log enqueue
            "conversation_enqueued",
            conversation_id=record.conversation_id,
            queue_size=self._queue.qsize(),
        )

    def set_processor(self, callback) -> None:
        """Set the processing callback function for queue items.

        The callback receives a ConversationRecord and processes it.
        """
        self._processor_callback = callback  # Store the processor function

    async def start_workers(self) -> None:
        """Start the background worker coroutines for processing the queue."""
        if not self._enabled:  # Extraction disabled
            logger.info("extraction_disabled_workers_not_started")  # Log
            return  # Don't start workers

        if self._is_running:  # Already running
            return  # Prevent duplicate starts

        self._is_running = True  # Mark as running
        logger.info("starting_extraction_workers", count=self._concurrency)  # Log start

        # Create worker tasks up to configured concurrency
        for worker_id in range(self._concurrency):  # Create each worker
            task = asyncio.create_task(  # Create async task
                self._worker_loop(worker_id),  # Worker coroutine
                name=f"extraction_worker_{worker_id}",  # Task name for debugging
            )
            self._workers.append(task)  # Track the worker task

    async def stop_workers(self) -> None:
        """Stop all background worker coroutines gracefully."""
        self._is_running = False  # Signal workers to stop
        logger.info("stopping_extraction_workers")  # Log stop

        # Cancel all worker tasks
        for task in self._workers:  # Iterate worker tasks
            task.cancel()  # Request cancellation

        # Wait for all workers to finish
        if self._workers:  # Only gather if workers exist
            await asyncio.gather(*self._workers, return_exceptions=True)  # Wait for completion

        self._workers.clear()  # Clear the worker list
        logger.info("extraction_workers_stopped")  # Log completion

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker coroutine that continuously processes queue items."""
        logger.debug("extraction_worker_started", worker_id=worker_id)  # Log worker start

        while self._is_running:  # Continue while queue is active
            try:  # Handle errors in queue processing
                # Wait for an item with timeout (allows checking _is_running)
                record = await asyncio.wait_for(  # Wait with timeout
                    self._queue.get(),  # Get next item from queue
                    timeout=1.0,  # Check running state every second
                )

                if self._processor_callback:  # Processor is set
                    logger.debug(  # Log processing start
                        "processing_conversation",
                        worker_id=worker_id,
                        conversation_id=record.conversation_id,
                    )
                    await self._processor_callback(record)  # Process the item

                self._queue.task_done()  # Mark task as complete

            except asyncio.TimeoutError:  # No item available within timeout
                continue  # Loop back to check _is_running

            except asyncio.CancelledError:  # Worker cancelled
                break  # Exit the loop

            except Exception as exc:  # Unexpected processing error
                logger.error(  # Log the error
                    "extraction_worker_error",
                    worker_id=worker_id,
                    error=str(exc),
                )
                # Continue processing - don't let one failure stop the worker

        logger.debug("extraction_worker_stopped", worker_id=worker_id)  # Log worker stop

    @property
    def queue_size(self) -> int:
        """Get the current number of items waiting in the queue."""
        return self._queue.qsize()  # Return queue depth

    @property
    def is_running(self) -> bool:
        """Check whether the queue workers are currently active."""
        return self._is_running  # Return running state
