.\scripts\format_and_lint.ps1 ; 

uv run pytest tests/ --cov=src --cov-report=html --cov-report=xml --cov-config=.coveragerc -vv -m "slow" ; 

uv run deepeval test run tests/eval/test_rag_accuracy.py ; 

Start-Process .\htmlcov\index.html ; 
