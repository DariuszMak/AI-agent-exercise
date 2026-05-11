# Python AI Agent Excercise

## Requirements

- [UV](https://github.com/astral-sh/uv) package manager

### Project structure diagrams

##### Modular perspective

<p align="center">
  <img src="images/structure_module.svg" alt="Modular perspective" width="600">
</p>

##### Library dependencies perspective

<p align="center">
  <img src="images/structure_module_clustered.svg" alt="Library dependencies perspective" width="600">
</p>

## Local development (Windows PowerShell):

You can also use VSCode `settings.json` and `launch.json` files to run the project (choose interpreter created by UV).

### Fast native Windows development:

```commandline
deactivate ; 
clear ; 

$ports = 5433, 9000

foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conns) {
        $conns | Select-Object -ExpandProperty OwningProcess -Unique |
            Where-Object { $_ -gt 0 } |
            ForEach-Object {
                Write-Host "Port $port is used by PID $_. Killing..."
                Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
            }
    } else {
        Write-Host "No process is using port $port."
    }
}

uv self update ; 
uv cache clean ; 

git reset --hard HEAD ; 
git clean -x -d -f ; 

#####

uv python install 3.14 ; 
uv python pin 3.14 ; 
uv sync --dev --no-cache ; 
uv lock ; 

##### STATIC ANALYSIS & TESTS

.venv\Scripts\Activate.ps1 ; 
$env:UV_ENV_FILE = ".env.example" ; 

.\scripts\format_and_lint.ps1 ; 

uv run pytest tests/ --cov=src --cov-report=html --cov-report=xml --cov-config=.coveragerc -vv -m "not slow" ; 

uv run deepeval test run tests/eval/test_rag_accuracy.py ; 
# uv run pytest tests/eval/ --cov=src -vv -m slow ; 

# uv run pytest tests/ -m slow --cov=src -vv ; 
# uv run pytest tests/ -m "not slow" --cov=src -vv ; 
Start-Process .\htmlcov\index.html ; 

##### GENERATE PROJECT STRUCTURE DIAGRAM:

# uv run pydeps src\main.py --noshow -T svg -o images\structure_runner_clustered.svg --max-bacon 100 --max-module-depth 100 --rankdir LR --cluster ; 
# uv run pydeps src\main.py --noshow -T svg -o images\structure_runner.svg --max-bacon 2 --max-module-depth 100 --rankdir LR ; 
# uv run pydeps src\main.py --noshow -T svg -o images\structure_runner_pylib.svg --max-bacon 2 --max-module-depth 100 --rankdir LR --pylib ; 

# uv run pydeps src --noshow -T svg -o images\structure_module_clustered.svg --max-bacon 100 --max-module-depth 100 --rankdir LR --cluster ; 
# uv run pydeps src --noshow -T svg -o images\structure_module.svg --max-bacon 2 --max-module-depth 100 --rankdir LR ; 
# uv run pydeps src --noshow -T svg -o images\structure_module_pylib.svg --max-bacon 2 --max-module-depth 100 --rankdir LR --pylib ; 

##### LOCAL RUN:

Start-Process uv -ArgumentList "run", "python", "src\main.py" ; 
Start-Sleep -Seconds 20 ; 

##### INVOKE API REQUESTS:

Invoke-RestMethod -Uri http://127.0.0.1:5000/index -Method POST ; 

##########

# Prepare JSON body
$body = @{ query = "What is KSeF?" } | ConvertTo-Json

# Call the Flask query endpoint
$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/query `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response

##########

$body = @{ query = "What is Camunda?" } | ConvertTo-Json

# Call the Flask query endpoint
$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/ask `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response

#####

$body = @{ query = "What is Devapo?" } | ConvertTo-Json

# Call the Flask query endpoint
$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/ask `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response

#####

$body = @{ query = "What is Ksef?" } | ConvertTo-Json

# Call the Flask query endpoint
$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/ask `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response
```


Check installed models:

```commandline
ollama list ; 
```