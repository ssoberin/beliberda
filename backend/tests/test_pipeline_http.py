"""Optional full HTTP contract test: Node proxy -> FastAPI -> frozen TF-IDF / fake Qwen.

Run with PIPELINE_MODEL_PATH pointing to the trusted baseline and NODE_BIN set to
Node 22+. This test never calls an external model or publishes an advert.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from backend.app.classifier import load_classifier

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTION = 'Укладываем плитку и керамогранит, меняем смесители и розетки.'


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class FakeQwen(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        assert self.path == '/v1/models'
        self.send_json({'data': [{'id': 'test-qwen'}]})

    def do_POST(self):
        assert self.path == '/v1/chat/completions'
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert payload['model'] == 'test-qwen'
        assert payload['stream'] is False
        chosen = json.loads(payload['messages'][-1]['content'])['requested_categories']
        assert [c['id'] for c in chosen] == [103]
        answer = {'drafts': [{'categoryId': 103,
                              'title': 'Укладка плитки', 'text': 'Укладываем плитку.'}]}
        self.send_json({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]})

    def send_json(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def wait_for(url, process):
    for _ in range(100):
        if process.poll() is not None:
            raise AssertionError(f'Server exited with code {process.returncode}: {url}')
        try:
            if httpx.get(url, timeout=.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(.05)
    raise AssertionError(f'Server did not start: {url}')


def test_form_http_through_classifier_and_generator(tmp_path):
    model_path = os.getenv('PIPELINE_MODEL_PATH')
    node = os.getenv('NODE_BIN') or shutil.which('node')
    if not model_path or not node:
        pytest.skip('Set PIPELINE_MODEL_PATH and NODE_BIN for the full HTTP test')
    load_classifier(model_path)  # Reject an unknown pickle before starting services.
    qwen = ThreadingHTTPServer(('127.0.0.1', 0), FakeQwen)
    worker = threading.Thread(target=qwen.serve_forever, daemon=True)
    worker.start()
    backend_port, frontend_port = free_port(), free_port()
    while frontend_port == backend_port:
        frontend_port = free_port()
    backend_url = f'http://127.0.0.1:{backend_port}'
    frontend_url = f'http://127.0.0.1:{frontend_port}'
    processes = []
    try:
        backend_env = {**os.environ, 'CLASSIFIER_MODEL_PATH': model_path,
                       'QWEN_BASE_URL': f'http://127.0.0.1:{qwen.server_port}/v1',
                       'QWEN_MODEL': 'test-qwen', 'QWEN_TOKENIZE_BEFORE_GENERATE': '0',
                       'SSH_TARGET': ''}
        backend_log = (tmp_path / 'backend.log').open('w')
        frontend_log = (tmp_path / 'frontend.log').open('w')
        try:
            processes.append(subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'backend.app.main:app',
                 '--host', '127.0.0.1', '--port', str(backend_port)],
                cwd=ROOT, env=backend_env, stdout=backend_log, stderr=subprocess.STDOUT))
            wait_for(backend_url + '/health', processes[-1])
            processes.append(subprocess.Popen(
                [node, 'scripts/server.mjs'], cwd=ROOT / 'frontend',
                env={**os.environ, 'API_TARGET': backend_url, 'PORT': str(frontend_port), 'HOST': '127.0.0.1'},
                stdout=frontend_log, stderr=subprocess.STDOUT))
            wait_for(frontend_url + '/', processes[-1])
            with httpx.Client(base_url=frontend_url, timeout=10) as browser:
                status = browser.get('/api/status')
                assert status.status_code == 200, status.text
                assert status.json()['categoryGeneration'] is True
                assert status.json()['ready'] is True
                payload = {'title': 'Плиточные работы', 'description': DESCRIPTION, 'mcId': 101}
                result = browser.post('/api/categories', json=payload)
                assert result.status_code == 200, result.text
                assert 103 in [c['id'] for c in result.json()['categories']]
                # Send one selected model suggestion, just as the form does.
                category = next(c for c in result.json()['categories'] if c['id'] == 103)
                draft = browser.post('/api/generate', json={**payload, 'categories': [category]})
                assert draft.status_code == 200, draft.text
                assert draft.json() == {'drafts': [{'categoryId': 103, 'title': 'Укладка плитки',
                                                   'description': 'Укладываем плитку.'}], 'warnings': []}
        finally:
            backend_log.close()
            frontend_log.close()
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        qwen.shutdown()
        qwen.server_close()
        worker.join(timeout=5)
