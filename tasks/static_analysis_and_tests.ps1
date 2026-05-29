.\scripts\format_and_lint.ps1 ; 

uv run pytest tests/ --cov=src --cov-report=html --cov-report=xml --cov-config=.coveragerc -vv -m "not slow" ; 

uv run deepeval test run tests/eval/test_rag_accuracy.py ; 
# uv run pytest tests/eval/ --cov=src -vv -m slow ; 

# uv run pytest tests/ -m slow --cov=src -vv ; 
# uv run pytest tests/ -m "not slow" --cov=src -vv ; 
Start-Process .\htmlcov\index.html ; 
