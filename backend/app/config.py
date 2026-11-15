"""Environment-based configuration. No credentials are committed or logged."""
from dataclasses import dataclass, field
import os
import re
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    classifier_model_path: str = ''
    qwen_base_url: str = 'http://127.0.0.1:8000/v1'
    qwen_model: str = 'avito-qwen'
    qwen_api_key: str = field(default='', repr=False)
    timeout_seconds: float = 90
    max_tokens: int = 4096
    max_concurrent_requests: int = 2
    response_format: str = 'json_schema'
    tokenize_before_generate: bool = True
    ssh_target: str = ''
    ssh_port: int | None = None
    ssh_local_port: int = 18000
    ssh_remote_host: str = '127.0.0.1'
    ssh_remote_port: int = 8000
    ssh_identity_file: str = ''
    ssh_startup_timeout: float = 15
    cors_origins: tuple[str, ...] = ('http://127.0.0.1:5173', 'http://localhost:5173')

    def __post_init__(self):
        url = urlsplit(self.qwen_base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('QWEN_BASE_URL должен быть HTTP(S) URL без логина, пароля, query и fragment.')
        if not self.qwen_model.strip():
            raise ValueError('QWEN_MODEL не может быть пустым.')
        if self.response_format not in ('json_object', 'json_schema', 'none'):
            raise ValueError('QWEN_RESPONSE_FORMAT: json_object, json_schema или none.')
        if self.timeout_seconds <= 0 or self.ssh_startup_timeout <= 0 or self.max_tokens < 128 or self.max_concurrent_requests < 1:
            raise ValueError('Таймауты, max_tokens и concurrency должны быть положительными.')
        for port in (self.ssh_local_port, self.ssh_remote_port) + ((self.ssh_port,) if self.ssh_port is not None else ()):
            if not 1 <= port <= 65535: raise ValueError('Порт должен быть от 1 до 65535.')
        if self.ssh_target and not re.fullmatch(r'(?:[\w][\w.-]*@)?[\w][\w.-]*', self.ssh_target):
            raise ValueError('SSH_TARGET: user@host или алиас SSH config; без параметров командной строки.')
        if not re.fullmatch(r'[\w][\w.-]*', self.ssh_remote_host):
            raise ValueError('SSH_REMOTE_HOST: IPv4 или имя узла.')
        if self.ssh_target and (url.hostname != '127.0.0.1' or url.port != self.ssh_local_port):
            raise ValueError('При SSH_TARGET адрес Qwen должен указывать на 127.0.0.1:SSH_LOCAL_PORT.')

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
        target = os.getenv('SSH_TARGET', '').strip()
        local_port = int(os.getenv('SSH_LOCAL_PORT', '18000'))
        base = f'http://127.0.0.1:{local_port if target else 8000}/v1'
        return cls(
            classifier_model_path=os.getenv('CLASSIFIER_MODEL_PATH', ''),
            qwen_base_url=os.getenv('QWEN_BASE_URL', base).rstrip('/'),
            qwen_model=os.getenv('QWEN_MODEL', 'avito-qwen'),
            qwen_api_key=os.getenv('QWEN_API_KEY', ''),
            timeout_seconds=float(os.getenv('QWEN_TIMEOUT_SECONDS', '90')),
            max_tokens=int(os.getenv('QWEN_MAX_TOKENS', '4096')),
            max_concurrent_requests=int(os.getenv('QWEN_MAX_CONCURRENT_REQUESTS', '2')),
            response_format=os.getenv('QWEN_RESPONSE_FORMAT', 'json_schema'),
            tokenize_before_generate=os.getenv('QWEN_TOKENIZE_BEFORE_GENERATE', '1').lower() not in ('0', 'false', 'no', 'off'),
            ssh_target=target, ssh_port=int(os.environ['SSH_PORT']) if os.getenv('SSH_PORT') else None,
            ssh_local_port=local_port, ssh_remote_host=os.getenv('SSH_REMOTE_HOST', '127.0.0.1'),
            ssh_remote_port=int(os.getenv('SSH_REMOTE_PORT', '8000')),
            ssh_identity_file=os.getenv('SSH_IDENTITY_FILE', ''),
            ssh_startup_timeout=float(os.getenv('SSH_STARTUP_TIMEOUT_SECONDS', '15')),
            cors_origins=tuple(x.strip() for x in os.getenv('CORS_ORIGINS', 'http://127.0.0.1:5173,http://localhost:5173').split(',') if x.strip()),
        )
