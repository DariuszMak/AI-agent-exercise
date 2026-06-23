$body = @{ query = "What is Empire State Building?" } | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/query `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response


$body = @{ query = "What is Jeddah Tower?" } | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/query `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response


$body = @{ query = "What is Empire State Building?" } | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/ask `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response


$body = @{ query = "What is Jeddah Tower?" } | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/ask `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response
