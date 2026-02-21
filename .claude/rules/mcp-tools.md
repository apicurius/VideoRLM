---
paths:
  - "kuavi/mcp_server.py"
  - "kuavi/search.py"
---

# MCP Tool Conventions

## Return Values
- Tools must return dicts, not raw strings
- Always include an `"error"` key with a descriptive message on failure
- Include relevant metadata (timestamps, counts) in successful responses

## State Management
- All shared state lives in the global `_state` dict — do not use module-level mutable variables elsewhere
- Tools must be stateless aside from reading/writing `_state`
- Check `_state` for an active index before performing search operations

## Dependencies
- Lazy-import heavy dependencies (`torch`, `transformers`, `cv2`) inside tool functions, not at module level
- This keeps MCP server startup fast and avoids import errors when optional deps are missing

## Error Handling
- Catch exceptions within each tool and return structured error dicts
- Never let unhandled exceptions propagate to the MCP client
