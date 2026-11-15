"""One persistent model worker, bounded admission and request-local futures."""
import asyncio
from contextvars import copy_context
from functools import partial
import logging
import time
from uuid import uuid4

from .errors import BackendError
from .observability import event, failure, request_id_context

logger = logging.getLogger(__name__)


class InferenceQueue:
    def __init__(self, executor, capacity=16):
        self.executor = executor
        self.queue = asyncio.Queue(maxsize=capacity)
        self.closing = False

    async def __aenter__(self):
        self.worker = asyncio.create_task(self._work())
        event(logger, 'inference_queue_started', capacity=self.queue.maxsize, workers=1)
        return self

    async def run(self, function, *args):
        if self.closing:
            event(logger, 'inference_rejected', level=logging.WARNING, reason='closing')
            raise BackendError(503, 'Классификатор останавливается.')
        future = asyncio.get_running_loop().create_future()
        job_id = uuid4().hex[:12]
        try:
            self.queue.put_nowait((future, function, args, copy_context(), time.monotonic(), job_id))
        except asyncio.QueueFull:
            event(logger, 'inference_rejected', level=logging.WARNING, reason='queue_full', capacity=self.queue.maxsize)
            raise BackendError(503, 'Очередь классификации заполнена. Повторите запрос позже.') from None
        event(logger, 'inference_enqueued', job_id=job_id, queue_depth=self.queue.qsize())
        try:
            return await future
        except asyncio.CancelledError:
            event(logger, 'inference_cancel_requested', job_id=job_id)
            raise

    async def _work(self):
        loop = asyncio.get_running_loop()
        while True:
            job = await self.queue.get()
            token = None
            try:
                if job is None:
                    return
                future, function, args, context, queued_at, job_id = job
                token = request_id_context.set(context.get(request_id_context, '-'))
                if future.cancelled():
                    event(logger, 'inference_skipped', job_id=job_id, reason='cancelled')
                    continue
                started = time.monotonic()
                event(logger, 'inference_started', job_id=job_id, queue_depth=self.queue.qsize(), wait_ms=int((started-queued_at)*1000))
                try:
                    result = await loop.run_in_executor(self.executor, partial(context.run, function, *args))
                except Exception as exc:
                    failure(logger, 'inference_failed', exc, job_id=job_id, elapsed_ms=int((time.monotonic()-started)*1000))
                    if not future.done():
                        future.set_exception(exc)
                else:
                    event(logger, 'inference_complete', job_id=job_id, elapsed_ms=int((time.monotonic()-started)*1000))
                    if not future.done():
                        future.set_result(result)
            finally:
                if token is not None:
                    request_id_context.reset(token)
                self.queue.task_done()

    async def __aexit__(self, *_):
        self.closing = True
        event(logger, 'inference_queue_stopping', queued=self.queue.qsize())
        while not self.queue.empty():
            future, *_ = self.queue.get_nowait()
            if not future.done():
                future.set_exception(BackendError(503, 'Классификатор останавливается.'))
            self.queue.task_done()
        self.queue.put_nowait(None)
        await self.worker
        event(logger, 'inference_queue_stopped')
