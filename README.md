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

## Fast native Windows development

```commandline

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

## Full static analysis

Login in SonarQube as `admin` with password `Admin1@Admin1@`.

```commandline
```

Check installed models:

```commandline
ollama list ; 
```