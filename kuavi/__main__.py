"""Allow running KUAVi MCP server as: python -m kuavi"""

from kuavi.mcp_server import mcp

mcp.run(transport="stdio")
