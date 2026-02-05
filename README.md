# Python AI Agent Excercise

## Requirements

- [UV](https://github.com/astral-sh/uv) package manager

## Local development (Windows PowerShell):

You can also use VSCode `settings.json` and `launch.json` files to run the project (choose interpreter created by UV).

### Fast native Windows development:

```commandline
deactivate ; 
clear ; 

Copy-Item .env.example .env -Force

uv self update ; 
uv cache clean ; 

git reset --hard HEAD ; 
git clean -x -d -f ; 

uv python install 3.11 ; 
uv python pin 3.11 ; 
uv sync --dev --no-cache ; 
uv lock ; 

##### STATIC ANALYSIS & TESTS

.venv\Scripts\Activate.ps1 ; 
$env:PYTHONPATH="." ; 

.\scripts\format_and_lint.ps1 ; 

uv run pytest test/ --cov=src -vv ; 
# uv run pytest test/ -m slow --cov=src -vv ; 
# uv run pytest test/ -m "not slow" --cov=src -vv ; 

##### LOCAL RUN:

Start-Process uv -ArgumentList "run", "python", "src\main.py" ; 
```

### INVOKE API REQUESTS:

```commandline

Invoke-RestMethod -Uri http://127.0.0.1:5000/index -Method POST ; 

#####

# Prepare JSON body
$body = @{ query = "What is KSeF?" } | ConvertTo-Json

# Call the Flask query endpoint
$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/query `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response

#####

$body = @{ query = "What is KSeF?" } | ConvertTo-Json

# Call the Flask query endpoint
$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/ask `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response
```

### Code linting:

```commandline
.venv\Scripts\Activate.ps1 ; 
$env:PYTHONPATH="." ; 

clear ; 

uv run pip-audit ; 
uv run ruff check test src ; 
uv run ruff format --check test src ; 

uv run mypy --strict test src ; 

# uv run mypy --explicit-package-bases test src ; 
# uv run mypy --explicit-package-bases --check-untyped-defs test src ; 
# uv run mypy --strict test src ; 
```

### Code autoformat:

```commandline
.venv\Scripts\Activate.ps1 ; 
$env:PYTHONPATH="." ; 

clear ; 

uv run ruff format test src ; 

uv run ruff check --fix test src ; 
uv run ruff check --fix --unsafe-fixes test src ; 
uv run ruff check --fix --select I test src ; 
```