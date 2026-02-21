# CLI reference

> Complete reference for Claude Code command-line interface, including commands and flags.

## CLI commands

| Command                         | Description                                            | Example                                           |
| :------------------------------ | :----------------------------------------------------- | :------------------------------------------------ |
| `claude`                        | Start interactive REPL                                 | `claude`                                          |
| `claude "query"`                | Start REPL with initial prompt                         | `claude "explain this project"`                   |
| `claude -p "query"`             | Query via SDK, then exit                               | `claude -p "explain this function"`               |
| `cat file \| claude -p "query"` | Process piped content                                  | `cat logs.txt \| claude -p "explain"`             |
| `claude -c`                     | Continue most recent conversation in current directory | `claude -c`                                       |
| `claude -c -p "query"`          | Continue via SDK                                       | `claude -c -p "Check for type errors"`            |
| `claude -r "<session>" "query"` | Resume session by ID or name                           | `claude -r "auth-refactor" "Finish this PR"`      |
| `claude update`                 | Update to latest version                               | `claude update`                                   |
| `claude mcp`                    | Configure Model Context Protocol (MCP) servers         | See the Claude Code MCP documentation.            |

## CLI flags

| Flag                                   | Description                                                                                                                                                                                               | Example                                                                                            |
| :------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| `--add-dir`                            | Add additional working directories for Claude to access                                                                                                                                                   | `claude --add-dir ../apps ../lib`                                                                  |
| `--agent`                              | Specify an agent for the current session                                                                                                                                                                  | `claude --agent my-custom-agent`                                                                   |
| `--agents`                             | Define custom subagents dynamically via JSON                                                                                                                                                              | `claude --agents '{"reviewer":{"description":"Reviews code","prompt":"You are a code reviewer"}}'` |
| `--allow-dangerously-skip-permissions` | Enable permission bypassing as an option without immediately activating it                                                                                                                                | `claude --permission-mode plan --allow-dangerously-skip-permissions`                               |
| `--allowedTools`                       | Tools that execute without prompting for permission                                                                                                                                                       | `"Bash(git log *)" "Bash(git diff *)" "Read"`                                                      |
| `--append-system-prompt`               | Append custom text to the end of the default system prompt                                                                                                                                                | `claude --append-system-prompt "Always use TypeScript"`                                            |
| `--append-system-prompt-file`          | Load additional system prompt text from a file and append to the default prompt (print mode only)                                                                                                         | `claude -p --append-system-prompt-file ./extra-rules.txt "query"`                                  |
| `--betas`                              | Beta headers to include in API requests (API key users only)                                                                                                                                              | `claude --betas interleaved-thinking`                                                              |
| `--chrome`                             | Enable Chrome browser integration for web automation and testing                                                                                                                                          | `claude --chrome`                                                                                  |
| `--continue`, `-c`                     | Load the most recent conversation in the current directory                                                                                                                                                | `claude --continue`                                                                                |
| `--dangerously-skip-permissions`       | Skip all permission prompts (use with caution)                                                                                                                                                            | `claude --dangerously-skip-permissions`                                                            |
| `--debug`                              | Enable debug mode with optional category filtering                                                                                                                                                        | `claude --debug "api,mcp"`                                                                         |
| `--disable-slash-commands`             | Disable all skills and slash commands for this session                                                                                                                                                    | `claude --disable-slash-commands`                                                                  |
| `--disallowedTools`                    | Tools that are removed from the model's context and cannot be used                                                                                                                                        | `"Bash(git log *)" "Bash(git diff *)" "Edit"`                                                      |
| `--fallback-model`                     | Enable automatic fallback to specified model when default model is overloaded (print mode only)                                                                                                           | `claude -p --fallback-model sonnet "query"`                                                        |
| `--fork-session`                       | When resuming, create a new session ID instead of reusing the original                                                                                                                                    | `claude --resume abc123 --fork-session`                                                            |
| `--from-pr`                            | Resume sessions linked to a specific GitHub PR                                                                                                                                                            | `claude --from-pr 123`                                                                             |
| `--ide`                                | Automatically connect to IDE on startup                                                                                                                                                                   | `claude --ide`                                                                                     |
| `--init`                               | Run initialization hooks and start interactive mode                                                                                                                                                       | `claude --init`                                                                                    |
| `--init-only`                          | Run initialization hooks and exit (no interactive session)                                                                                                                                                | `claude --init-only`                                                                               |
| `--include-partial-messages`           | Include partial streaming events in output                                                                                                                                                                | `claude -p --output-format stream-json --include-partial-messages "query"`                         |
| `--input-format`                       | Specify input format for print mode (options: `text`, `stream-json`)                                                                                                                                      | `claude -p --output-format json --input-format stream-json`                                        |
| `--json-schema`                        | Get validated JSON output matching a JSON Schema (print mode only)                                                                                                                                        | `claude -p --json-schema '{"type":"object","properties":{...}}' "query"`                           |
| `--maintenance`                        | Run maintenance hooks and exit                                                                                                                                                                            | `claude --maintenance`                                                                             |
| `--max-budget-usd`                     | Maximum dollar amount to spend on API calls before stopping (print mode only)                                                                                                                             | `claude -p --max-budget-usd 5.00 "query"`                                                          |
| `--max-turns`                          | Limit the number of agentic turns (print mode only)                                                                                                                                                       | `claude -p --max-turns 3 "query"`                                                                  |
| `--mcp-config`                         | Load MCP servers from JSON files or strings (space-separated)                                                                                                                                             | `claude --mcp-config ./mcp.json`                                                                   |
| `--model`                              | Sets the model for the current session                                                                                                                                                                    | `claude --model claude-sonnet-4-6`                                                                 |
| `--no-chrome`                          | Disable Chrome browser integration for this session                                                                                                                                                       | `claude --no-chrome`                                                                               |
| `--no-session-persistence`             | Disable session persistence (print mode only)                                                                                                                                                             | `claude -p --no-session-persistence "query"`                                                       |
| `--output-format`                      | Specify output format for print mode (options: `text`, `json`, `stream-json`)                                                                                                                             | `claude -p "query" --output-format json`                                                           |
| `--permission-mode`                    | Begin in a specified permission mode                                                                                                                                                                      | `claude --permission-mode plan`                                                                    |
| `--permission-prompt-tool`             | Specify an MCP tool to handle permission prompts in non-interactive mode                                                                                                                                  | `claude -p --permission-prompt-tool mcp_auth_tool "query"`                                         |
| `--plugin-dir`                         | Load plugins from directories for this session only (repeatable)                                                                                                                                          | `claude --plugin-dir ./my-plugins`                                                                 |
| `--print`, `-p`                        | Print response without interactive mode                                                                                                                                                                   | `claude -p "query"`                                                                                |
| `--remote`                             | Create a new web session on claude.ai with the provided task description                                                                                                                                  | `claude --remote "Fix the login bug"`                                                              |
| `--resume`, `-r`                       | Resume a specific session by ID or name, or show an interactive picker                                                                                                                                    | `claude --resume auth-refactor`                                                                    |
| `--session-id`                         | Use a specific session ID for the conversation (must be a valid UUID)                                                                                                                                     | `claude --session-id "550e8400-e29b-41d4-a716-446655440000"`                                       |
| `--setting-sources`                    | Comma-separated list of setting sources to load                                                                                                                                                           | `claude --setting-sources user,project`                                                            |
| `--settings`                           | Path to a settings JSON file or a JSON string to load additional settings from                                                                                                                            | `claude --settings ./settings.json`                                                                |
| `--strict-mcp-config`                  | Only use MCP servers from `--mcp-config`, ignoring all other MCP configurations                                                                                                                           | `claude --strict-mcp-config --mcp-config ./mcp.json`                                               |
| `--system-prompt`                      | Replace the entire system prompt with custom text                                                                                                                                                         | `claude --system-prompt "You are a Python expert"`                                                 |
| `--system-prompt-file`                 | Load system prompt from a file, replacing the default prompt (print mode only)                                                                                                                            | `claude -p --system-prompt-file ./custom-prompt.txt "query"`                                       |
| `--teleport`                           | Resume a web session in your local terminal                                                                                                                                                               | `claude --teleport`                                                                                |
| `--teammate-mode`                      | Set how agent team teammates display: `auto`, `in-process`, or `tmux`                                                                                                                                     | `claude --teammate-mode in-process`                                                                |
| `--tools`                              | Restrict which built-in tools Claude can use                                                                                                                                                              | `claude --tools "Bash,Edit,Read"`                                                                  |
| `--verbose`                            | Enable verbose logging                                                                                                                                                                                    | `claude --verbose`                                                                                 |
| `--version`, `-v`                      | Output the version number                                                                                                                                                                                 | `claude -v`                                                                                        |

### Agents flag format

The `--agents` flag accepts a JSON object that defines one or more custom subagents. Each subagent requires a unique name (as the key) and a definition object with the following fields:

| Field             | Required | Description                                                                                                                                                                                                        |
| :---------------- | :------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `description`     | Yes      | Natural language description of when the subagent should be invoked                                                                                                                                                |
| `prompt`          | Yes      | The system prompt that guides the subagent's behavior                                                                                                                                                              |
| `tools`           | No       | Array of specific tools the subagent can use. If omitted, inherits all tools                                                                                                                                       |
| `disallowedTools` | No       | Array of tool names to explicitly deny for this subagent                                                                                                                                                           |
| `model`           | No       | Model alias to use: `sonnet`, `opus`, `haiku`, or `inherit`. Defaults to `inherit`                                                                                                                                  |
| `skills`          | No       | Array of skill names to preload into the subagent's context                                                                                                                                                        |
| `mcpServers`      | No       | Array of MCP servers for this subagent                                                                                                                                                                             |
| `maxTurns`        | No       | Maximum number of agentic turns before the subagent stops                                                                                                                                                          |

Example:

```bash
claude --agents '{
  "code-reviewer": {
    "description": "Expert code reviewer. Use proactively after code changes.",
    "prompt": "You are a senior code reviewer. Focus on code quality, security, and best practices.",
    "tools": ["Read", "Grep", "Glob", "Bash"],
    "model": "sonnet"
  },
  "debugger": {
    "description": "Debugging specialist for errors and test failures.",
    "prompt": "You are an expert debugger. Analyze errors, identify root causes, and provide fixes."
  }
}'
```

### System prompt flags

Claude Code provides four flags for customizing the system prompt:

| Flag                          | Behavior                                    | Modes               | Use Case                                                             |
| :---------------------------- | :------------------------------------------ | :------------------ | :------------------------------------------------------------------- |
| `--system-prompt`             | **Replaces** entire default prompt          | Interactive + Print | Complete control over Claude's behavior and instructions             |
| `--system-prompt-file`        | **Replaces** with file contents             | Print only          | Load prompts from files for reproducibility and version control      |
| `--append-system-prompt`      | **Appends** to default prompt               | Interactive + Print | Add specific instructions while keeping default Claude Code behavior |
| `--append-system-prompt-file` | **Appends** file contents to default prompt | Print only          | Load additional instructions from files while keeping defaults       |

* **`--system-prompt`**: Use when you need complete control over Claude's system prompt. This removes all default Claude Code instructions.
* **`--system-prompt-file`**: Use when you want to load a custom prompt from a file.
* **`--append-system-prompt`**: Use when you want to add specific instructions while keeping Claude Code's default capabilities intact. This is the safest option for most use cases.
* **`--append-system-prompt-file`**: Use when you want to append instructions from a file while keeping Claude Code's defaults.

`--system-prompt` and `--system-prompt-file` are mutually exclusive. The append flags can be used together with either replacement flag.

## See also

* Chrome extension - Browser automation and web testing
* Interactive mode - Shortcuts, input modes, and interactive features
* Quickstart guide - Getting started with Claude Code
* Common workflows - Advanced workflows and patterns
* Settings - Configuration options
* Agent SDK documentation - Programmatic usage and integrations
