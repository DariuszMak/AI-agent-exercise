.\scripts\format_and_lint.ps1 ; 

uv run pytest tests/ --cov=src --cov-report=html --cov-report=xml --cov-config=.coveragerc -vv -m "not slow" ; 
uv run pytest tests/ --cov=src --cov-report=html --cov-report=xml --cov-config=.coveragerc -vv -m "slow" ; 

uv run pytest tests/eval/test_rag_accuracy.py ;  

Start-Process .\htmlcov\index.html ; 
