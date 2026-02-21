# Plugins reference

> Complete technical reference for Claude Code plugin system, including schemas, CLI commands, and component specifications.

This reference provides complete technical specifications for the Claude Code plugin system, including component schemas, CLI commands, and development tools.

A **plugin** is a self-contained directory of components that extends Claude Code with custom functionality. Plugin components include skills, agents, hooks, MCP servers, and LSP servers.

## Plugin components reference

### Skills

Plugins add skills to Claude Code, creating `/name` shortcuts that you or Claude can invoke.

**Location**: `skills/` or `commands/` directory in plugin root

**File format**: Skills are directories with `SKILL.md`; commands are simple markdown files

**Skill structure**:

```
skills/
├── pdf-processor/
│   ├── SKILL.md
│   ├── reference.md (optional)
│   └── scripts/ (optional)
└── code-reviewer/
    └── SKILL.md
```

**Integration behavior**:

* Skills and commands are automatically discovered when the plugin is installed
* Claude can invoke them automatically based on task context
* Skills can include supporting files alongside SKILL.md

### Agents

Plugins can provide specialized subagents for specific tasks that Claude can invoke automatically when appropriate.

**Location**: `agents/` directory in plugin root

**File format**: Markdown files describing agent capabilities

**Agent structure**:

```markdown
---
name: agent-name
description: What this agent specializes in and when Claude should invoke it
---

Detailed system prompt for the agent describing its role, expertise, and behavior.
```

### Hooks

Plugins can provide event handlers that respond to Claude Code events automatically.

**Location**: `hooks/hooks.json` in plugin root, or inline in plugin.json

**Format**: JSON configuration with event matchers and actions

**Hook configuration**:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/format-code.sh"
          }
        ]
      }
    ]
  }
}
```

**Available events**:

* `PreToolUse`: Before Claude uses any tool
* `PostToolUse`: After Claude successfully uses any tool
* `PostToolUseFailure`: After Claude tool execution fails
* `PermissionRequest`: When a permission dialog is shown
* `UserPromptSubmit`: When user submits a prompt
* `Notification`: When Claude Code sends notifications
* `Stop`: When Claude attempts to stop
* `SubagentStart`: When a subagent is started
* `SubagentStop`: When a subagent attempts to stop
* `SessionStart`: At the beginning of sessions
* `SessionEnd`: At the end of sessions
* `TeammateIdle`: When an agent team teammate is about to go idle
* `TaskCompleted`: When a task is being marked as completed
* `PreCompact`: Before conversation history is compacted

**Hook types**:

* `command`: Execute shell commands or scripts
* `prompt`: Evaluate a prompt with an LLM (uses `$ARGUMENTS` placeholder for context)
* `agent`: Run an agentic verifier with tools for complex verification tasks

### MCP servers

Plugins can bundle Model Context Protocol (MCP) servers to connect Claude Code with external tools and services.

**Location**: `.mcp.json` in plugin root, or inline in plugin.json

**Format**: Standard MCP server configuration

```json
{
  "mcpServers": {
    "plugin-database": {
      "command": "${CLAUDE_PLUGIN_ROOT}/servers/db-server",
      "args": ["--config", "${CLAUDE_PLUGIN_ROOT}/config.json"],
      "env": {
        "DB_PATH": "${CLAUDE_PLUGIN_ROOT}/data"
      }
    }
  }
}
```

### LSP servers

Plugins can provide Language Server Protocol (LSP) servers to give Claude real-time code intelligence while working on your codebase.

LSP integration provides:

* **Instant diagnostics**: Claude sees errors and warnings immediately after each edit
* **Code navigation**: go to definition, find references, and hover information
* **Language awareness**: type information and documentation for code symbols

**Location**: `.lsp.json` in plugin root, or inline in `plugin.json`

**`.lsp.json` file format**:

```json
{
  "go": {
    "command": "gopls",
    "args": ["serve"],
    "extensionToLanguage": {
      ".go": "go"
    }
  }
}
```

**Required fields:**

| Field                 | Description                                  |
| :-------------------- | :------------------------------------------- |
| `command`             | The LSP binary to execute (must be in PATH)  |
| `extensionToLanguage` | Maps file extensions to language identifiers |

**Optional fields:**

| Field                   | Description                                               |
| :---------------------- | :-------------------------------------------------------- |
| `args`                  | Command-line arguments for the LSP server                 |
| `transport`             | Communication transport: `stdio` (default) or `socket`    |
| `env`                   | Environment variables to set when starting the server     |
| `initializationOptions` | Options passed to the server during initialization        |
| `settings`              | Settings passed via `workspace/didChangeConfiguration`    |
| `workspaceFolder`       | Workspace folder path for the server                      |
| `startupTimeout`        | Max time to wait for server startup (milliseconds)        |
| `shutdownTimeout`       | Max time to wait for graceful shutdown (milliseconds)     |
| `restartOnCrash`        | Whether to automatically restart the server if it crashes |
| `maxRestarts`           | Maximum number of restart attempts before giving up       |

**Available LSP plugins:**

| Plugin           | Language server            | Install command                                            |
| :--------------- | :------------------------- | :--------------------------------------------------------- |
| `pyright-lsp`    | Pyright (Python)           | `pip install pyright` or `npm install -g pyright`          |
| `typescript-lsp` | TypeScript Language Server | `npm install -g typescript-language-server typescript`     |
| `rust-lsp`       | rust-analyzer              | See rust-analyzer installation docs                        |

## Plugin installation scopes

| Scope     | Settings file                 | Use case                                                 |
| :-------- | :---------------------------- | :------------------------------------------------------- |
| `user`    | `~/.claude/settings.json`     | Personal plugins available across all projects (default) |
| `project` | `.claude/settings.json`       | Team plugins shared via version control                  |
| `local`   | `.claude/settings.local.json` | Project-specific plugins, gitignored                     |
| `managed` | `managed-settings.json`       | Managed plugins (read-only, update only)                 |

## Plugin manifest schema

The `.claude-plugin/plugin.json` file defines your plugin's metadata and configuration. The manifest is optional. If omitted, Claude Code auto-discovers components in default locations and derives the plugin name from the directory name.

### Complete schema

```json
{
  "name": "plugin-name",
  "version": "1.2.0",
  "description": "Brief plugin description",
  "author": {
    "name": "Author Name",
    "email": "author@example.com",
    "url": "https://github.com/author"
  },
  "homepage": "https://docs.example.com/plugin",
  "repository": "https://github.com/author/plugin",
  "license": "MIT",
  "keywords": ["keyword1", "keyword2"],
  "commands": ["./custom/commands/special.md"],
  "agents": "./custom/agents/",
  "skills": "./custom/skills/",
  "hooks": "./config/hooks.json",
  "mcpServers": "./mcp-config.json",
  "outputStyles": "./styles/",
  "lspServers": "./.lsp.json"
}
```

### Required fields

If you include a manifest, `name` is the only required field.

| Field  | Type   | Description                               | Example              |
| :----- | :----- | :---------------------------------------- | :------------------- |
| `name` | string | Unique identifier (kebab-case, no spaces) | `"deployment-tools"` |

### Metadata fields

| Field         | Type   | Description                       | Example                                            |
| :------------ | :----- | :-------------------------------- | :------------------------------------------------- |
| `version`     | string | Semantic version                  | `"2.1.0"`                                          |
| `description` | string | Brief explanation of purpose      | `"Deployment automation tools"`                    |
| `author`      | object | Author information                | `{"name": "Dev Team", "email": "dev@company.com"}` |
| `homepage`    | string | Documentation URL                 | `"https://docs.example.com"`                       |
| `repository`  | string | Source code URL                   | `"https://github.com/user/plugin"`                 |
| `license`     | string | License identifier                | `"MIT"`, `"Apache-2.0"`                            |
| `keywords`    | array  | Discovery tags                    | `["deployment", "ci-cd"]`                          |

### Component path fields

| Field          | Type                  | Description                       | Example                                |
| :------------- | :-------------------- | :-------------------------------- | :------------------------------------- |
| `commands`     | string or array       | Additional command files/dirs     | `"./custom/cmd.md"` or `["./cmd1.md"]` |
| `agents`       | string or array       | Additional agent files            | `"./custom/agents/reviewer.md"`        |
| `skills`       | string or array       | Additional skill directories      | `"./custom/skills/"`                   |
| `hooks`        | string/array/object   | Hook config paths or inline       | `"./my-extra-hooks.json"`              |
| `mcpServers`   | string/array/object   | MCP config paths or inline        | `"./my-extra-mcp-config.json"`         |
| `outputStyles` | string or array       | Output style files/dirs           | `"./styles/"`                          |
| `lspServers`   | string/array/object   | LSP configs                       | `"./.lsp.json"`                        |

### Environment variables

**`${CLAUDE_PLUGIN_ROOT}`**: Contains the absolute path to your plugin directory. Use this in hooks, MCP servers, and scripts to ensure correct paths regardless of installation location.

## Plugin caching and file resolution

Plugins are copied to the user's local **plugin cache** (`~/.claude/plugins/cache`) rather than being used in-place. Installed plugins cannot reference files outside their directory. If your plugin needs to access files outside its directory, create symbolic links within your plugin directory.

## Plugin directory structure

### Standard plugin layout

```
enterprise-plugin/
├── .claude-plugin/           # Metadata directory (optional)
│   └── plugin.json             # plugin manifest
├── commands/                 # Default command location
├── agents/                   # Default agent location
├── skills/                   # Agent Skills
├── hooks/                    # Hook configurations
│   └── hooks.json
├── .mcp.json                # MCP server definitions
├── .lsp.json                # LSP server configurations
├── scripts/                 # Hook and utility scripts
├── LICENSE
└── CHANGELOG.md
```

### File locations reference

| Component       | Default Location             | Purpose                                    |
| :-------------- | :--------------------------- | :----------------------------------------- |
| **Manifest**    | `.claude-plugin/plugin.json` | Plugin metadata and configuration          |
| **Commands**    | `commands/`                  | Skill Markdown files (legacy)              |
| **Agents**      | `agents/`                    | Subagent Markdown files                    |
| **Skills**      | `skills/`                    | Skills with `<name>/SKILL.md` structure    |
| **Hooks**       | `hooks/hooks.json`           | Hook configuration                         |
| **MCP servers** | `.mcp.json`                  | MCP server definitions                     |
| **LSP servers** | `.lsp.json`                  | Language server configurations             |

## CLI commands reference

### plugin install

```bash
claude plugin install <plugin> [options]
```

| Option                | Description                                       | Default |
| :-------------------- | :------------------------------------------------ | :------ |
| `-s, --scope <scope>` | Installation scope: `user`, `project`, or `local` | `user`  |

### plugin uninstall

```bash
claude plugin uninstall <plugin> [options]
```

Aliases: `remove`, `rm`

### plugin enable

```bash
claude plugin enable <plugin> [options]
```

### plugin disable

```bash
claude plugin disable <plugin> [options]
```

### plugin update

```bash
claude plugin update <plugin> [options]
```

## Debugging and development tools

Use `claude --debug` (or `/debug` within the TUI) to see plugin loading details.

### Common issues

| Issue                               | Cause                           | Solution                                                     |
| :---------------------------------- | :------------------------------ | :----------------------------------------------------------- |
| Plugin not loading                  | Invalid `plugin.json`           | Validate JSON syntax with `claude plugin validate`           |
| Commands not appearing              | Wrong directory structure       | Ensure `commands/` at root, not in `.claude-plugin/`         |
| Hooks not firing                    | Script not executable           | Run `chmod +x script.sh`                                     |
| MCP server fails                    | Missing `${CLAUDE_PLUGIN_ROOT}` | Use variable for all plugin paths                            |
| Path errors                         | Absolute paths used             | All paths must be relative and start with `./`               |
| LSP `Executable not found in $PATH` | Language server not installed   | Install the binary                                           |

## Version management

Follow semantic versioning for plugin releases:

```json
{
  "name": "my-plugin",
  "version": "2.1.0"
}
```

**Version format**: `MAJOR.MINOR.PATCH`

* **MAJOR**: Breaking changes
* **MINOR**: New features (backward-compatible)
* **PATCH**: Bug fixes (backward-compatible)

> **Warning:** Claude Code uses the version to determine whether to update your plugin. If you change your plugin's code but don't bump the version, existing users won't see your changes due to caching.

## See also

* [Plugins](/en/plugins) - Tutorials and practical usage
* [Plugin marketplaces](/en/plugin-marketplaces) - Creating and managing marketplaces
* [Skills](/en/skills) - Skill development details
* [Subagents](/en/sub-agents) - Agent configuration and capabilities
* [Hooks](/en/hooks) - Event handling and automation
* [MCP](/en/mcp) - External tool integration
* [Settings](/en/settings) - Configuration options for plugins
