class BackendError(Exception):
    def __init__(self, status_code: int, message: str, code: str | None = None):
        self.status_code = status_code
        self.message = message
        self.code = code or {400: 'bad_request', 413: 'payload_too_large', 415: 'unsupported_format',
                             422: 'validation_error', 502: 'upstream_error', 503: 'service_unavailable',
                             504: 'upstream_timeout'}.get(status_code, 'internal_error')
        super().__init__(message)
