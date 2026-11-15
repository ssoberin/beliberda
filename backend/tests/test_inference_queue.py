import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from backend.app.errors import BackendError
from backend.app.inference import InferenceQueue

def test_serial_worker_preserves_request_identity_without_blocking_event_loop():
    async def check():
        threads=[];active=0;peak=0
        def predict(text, source):
            nonlocal active,peak
            active+=1;peak=max(peak,active);threads.append(threading.get_ident())
            time.sleep(.01);active-=1
            return text,source
        with ThreadPoolExecutor(max_workers=1) as executor:
            async with InferenceQueue(executor) as worker:
                tasks=[asyncio.create_task(worker.run(predict,str(i),101+i)) for i in range(6)]
                await asyncio.sleep(.002)
                assert not all(t.done() for t in tasks)
                assert await asyncio.gather(*tasks)==[(str(i),101+i) for i in range(6)]
        assert peak==1 and len(set(threads))==1
    asyncio.run(check())

def test_bounded_queue_rejects_overload_and_skips_cancelled_job():
    async def check():
        entered=threading.Event();release=threading.Event();calls=[]
        def predict(value):
            calls.append(value)
            if value==1:
                entered.set();assert release.wait(3)
            return value
        with ThreadPoolExecutor(max_workers=1) as executor:
            async with InferenceQueue(executor,capacity=1) as worker:
                first=asyncio.create_task(worker.run(predict,1))
                while not entered.is_set():await asyncio.sleep(.001)
                queued=asyncio.create_task(worker.run(predict,2));await asyncio.sleep(.001)
                with pytest.raises(BackendError):await worker.run(predict,3)
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):await queued
                release.set();assert await first==1
                await worker.queue.join()
                assert await worker.run(predict,4)==4
        assert calls==[1,4]
    asyncio.run(check())
