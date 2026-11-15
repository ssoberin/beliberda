"""One persistent model worker, bounded admission and request-local futures."""
import asyncio
from .errors import BackendError

class InferenceQueue:
    def __init__(self, executor, capacity=16):
        self.executor = executor
        self.queue = asyncio.Queue(maxsize=capacity)
        self.closing = False

    async def __aenter__(self):
        self.worker = asyncio.create_task(self._work())
        return self

    async def run(self, function, *args):
        if self.closing:
            raise BackendError(503, 'Классификатор останавливается.')
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((future, function, args))
        except asyncio.QueueFull:
            raise BackendError(503, 'Очередь классификации заполнена. Повторите запрос позже.') from None
        return await future

    async def _work(self):
        loop = asyncio.get_running_loop()
        while True:
            job = await self.queue.get()
            try:
                if job is None:
                    return
                future, function, args = job
                if future.cancelled():
                    continue
                try:
                    result = await loop.run_in_executor(self.executor, function, *args)
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                else:
                    if not future.done():
                        future.set_result(result)
            finally:
                self.queue.task_done()

    async def __aexit__(self, *_):
        self.closing = True
        while not self.queue.empty():
            future, _, _ = self.queue.get_nowait()
            if not future.done():
                future.set_exception(BackendError(503, 'Классификатор останавливается.'))
            self.queue.task_done()
        self.queue.put_nowait(None)
        # Let the active native model call finish before releasing shared weights.
        await self.worker
