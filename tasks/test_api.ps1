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
