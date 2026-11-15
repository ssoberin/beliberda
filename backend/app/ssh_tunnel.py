"""OpenSSH local forwarding with bounded reconnects and lifespan cleanup."""
import asyncio
import contextlib
import logging
from pathlib import Path
import socket
import time

from .config import Settings

LOG = logging.getLogger('uvicorn.error')


def ssh_command(settings: Settings):
    command = [
        'ssh', '-N', '-T',
        '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
        '-L', f'127.0.0.1:{settings.ssh_local_port}:{settings.ssh_remote_host}:{settings.ssh_remote_port}',
    ]
    if settings.ssh_port is not None:
        command += ['-p', str(settings.ssh_port)]
    if settings.ssh_identity_file:
        command += ['-i', str(Path(settings.ssh_identity_file).expanduser())]
    return command + [settings.ssh_target]


class SSHTunnel:
    RETRY_DELAY = 1.0

    def __init__(self, settings: Settings):
        self.settings = settings
        self.process = None
        self.last_error = None
        self._connected = False
        self._closing = False
        self._monitor = None
        self._stderr_reader = None

    @property
    def alive(self):
        return self._connected and self.process is not None and self.process.returncode is None

    async def _read_errors(self, process):
        # Report known failure classes, never dump arbitrary remote text,
        # command arguments, credentials or contents of keys into logs/API.
        reasons = {
            'connection timed out': 'Сервер не отвечает на SSH-порту (таймаут).',
            'connection refused': 'SSH-порт сервера отклонил соединение.',
            'network is unreachable': 'Сетевой маршрут к серверу недоступен.',
            'no route to host': 'Нет сетевого маршрута к серверу.',
            'permission denied': 'SSH-ключ не принят. Проверьте ключ и ssh-agent.',
            'host key verification failed': 'Не удалось проверить SSH-ключ сервера.',
            'address already in use': 'Локальный порт SSH-туннеля занят.',
            'could not resolve hostname': 'Не удалось определить адрес SSH-сервера.',
            'broken pipe': 'SSH-соединение оборвалось.',
            'connection reset': 'Сервер сбросил SSH-соединение.',
        }
        while line := await process.stderr.readline():
            text = line.decode(errors='replace').lower()
            for phrase, message in reasons.items():
                if phrase in text:
                    self.last_error = message
                    break

    async def _connect(self):
        self._connected = False
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', self.settings.ssh_local_port))
            except OSError as exc:
                raise RuntimeError('Локальный порт SSH-туннеля занят или недоступен.') from exc
        self.process = await asyncio.create_subprocess_exec(
            *ssh_command(self.settings), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_reader = asyncio.create_task(self._read_errors(self.process))
        deadline = time.monotonic() + self.settings.ssh_startup_timeout
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                raise RuntimeError(self.last_error or f'SSH завершился с кодом {self.process.returncode}.')
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection('127.0.0.1', self.settings.ssh_local_port), timeout=.5,
                )
                writer.close()
                await writer.wait_closed()
                if self.process.returncode is not None:
                    raise RuntimeError(self.last_error or 'SSH завершился при открытии туннеля.')
                self._connected = True
                self.last_error = None
                LOG.info('SSH tunnel connected on local port %s', self.settings.ssh_local_port)
                return
            except (OSError, TimeoutError):
                await asyncio.sleep(.1)
        raise RuntimeError(self.last_error or 'Истекло время запуска SSH-туннеля.')

    async def _close_process(self):
        self._connected = False
        process = self.process
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError): process.kill()
                await process.wait()
        if self._stderr_reader:
            self._stderr_reader.cancel()
            with contextlib.suppress(asyncio.CancelledError): await self._stderr_reader
            self._stderr_reader = None
        self.process = None

    async def __aenter__(self):
        try:
            await self._connect()
        except FileNotFoundError:
            await self.stop()
            raise
        except (OSError, RuntimeError) as exc:
            self.last_error = str(exc)
            await self._close_process()
            LOG.warning('SSH initial connection failed: %s; reconnect scheduled', self.last_error)
        except BaseException:
            await self.stop()
            raise
        self._monitor = asyncio.create_task(self._watch())
        return self

    async def _watch(self):
        delay = self.RETRY_DELAY
        while not self._closing:
            if self.process is not None and self._connected:
                code = await self.process.wait()
                if self._closing: return
                self.last_error = self.last_error or f'SSH-соединение завершилось (код {code}).'
                LOG.warning('SSH disconnected: %s', self.last_error)
                await self._close_process()
            await asyncio.sleep(delay)
            if self._closing: return
            try:
                await self._connect()
                delay = self.RETRY_DELAY
            except (OSError, RuntimeError) as exc:
                self.last_error = str(exc)
                await self._close_process()
                LOG.warning('SSH reconnect failed: %s; next attempt in %.0fs', self.last_error, min(delay * 2, 30))
                delay = min(delay * 2, 30)

    async def stop(self):
        self._closing = True
        if self._monitor:
            self._monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError): await self._monitor
            self._monitor = None
        await self._close_process()

    async def __aexit__(self, *_):
        await self.stop()
