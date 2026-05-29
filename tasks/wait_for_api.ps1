do {
    Start-Sleep -Seconds 3

    try {
        $api = Invoke-RestMethod `
            -Uri "http://127.0.0.1:5000/index" `
            -Method Post
    }
    catch {
        $api = $null
    }

} until ($api)
