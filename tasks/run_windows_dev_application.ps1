# Start-Process uv -ArgumentList "run", "python", "-m", "src.mcp_client.server" ; 
Start-Process uv -ArgumentList "run", "python", "src\agent_main.py" ; 
Start-Process uv -ArgumentList "run", "python", "src\main.py" ; 
