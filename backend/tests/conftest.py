"""Generation-only unit tests don't initialize a neural classification encoder."""
import pytest
from avito_compare.c_syntax import CSyntaxClassifier

@pytest.fixture(autouse=True)
def isolate_generator_unit_tests(request, monkeypatch):
    if request.node.path.name == 'test_backend.py':
        monkeypatch.setattr(CSyntaxClassifier, 'warmup', lambda self: None)
