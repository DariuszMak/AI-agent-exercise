Invoke-RestMethod -Uri http://127.0.0.1:5000/index -Method POST


$body = @{ query = "What is KSeF?" } | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/query `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response


$body = @{ query = "What is Camunda?" } | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/ask `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response


$body = @{ query = "What is Devapo?" } | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/ask `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response


$body = @{ query = "What is Ksef?" } | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://127.0.0.1:5000/ask `
    -Method POST `
    -Body $body `
    -ContentType "application/json"

$response
