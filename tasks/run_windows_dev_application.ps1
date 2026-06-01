Start-Process uv -ArgumentList "run", "python", "-m", "src.mcp_client.mcp_server" ; 
Start-Process uv -ArgumentList "run", "python", "src\agent_runner.py" ; 
Start-Process uv -ArgumentList "run", "python", "src\server.py" ; 
