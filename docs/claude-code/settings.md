# Claude Code settings

> Configure Claude Code with global and project-level settings, and environment variables.

Claude Code offers a variety of settings to configure its behavior to meet your needs. You can configure Claude Code by running the `/config` command when using the interactive REPL, which opens a tabbed Settings interface where you can view status information and modify configuration options.

## Configuration scopes

Claude Code uses a **scope system** to determine where configurations apply and who they're shared with. Understanding scopes helps you decide how to configure Claude Code for personal use, team collaboration, or enterprise deployment.

### Available scopes

| Scope       | Location                             | Who it affects                       | Shared with team?      |
| :---------- | :----------------------------------- | :----------------------------------- | :--------------------- |
| **Managed** | System-level `managed-settings.json` | All users on the machine             | Yes (deployed by IT)   |
| **User**    | `~/.claude/` directory               | You, across all projects             | No                     |
| **Project** | `.claude/` in repository             | All collaborators on this repository | Yes (committed to git) |
| **Local**   | `.claude/*.local.*` files            | You, in this repository only         | No (gitignored)        |

### When to use each scope

**Managed scope** is for:

* Security policies that must be enforced organization-wide
* Compliance requirements that can't be overridden
* Standardized configurations deployed by IT/DevOps

**User scope** is best for:

* Personal preferences you want everywhere (themes, editor settings)
* Tools and plugins you use across all projects
* API keys and authentication (stored securely)

**Project scope** is best for:

* Team-shared settings (permissions, hooks, MCP servers)
* Plugins the whole team should have
* Standardizing tooling across collaborators

**Local scope** is best for:

* Personal overrides for a specific project
* Testing configurations before sharing with the team
* Machine-specific settings that won't work for others

### How scopes interact

When the same setting is configured in multiple scopes, more specific scopes take precedence:

1. **Managed** (highest) - can't be overridden by anything
2. **Command line arguments** - temporary session overrides
3. **Local** - overrides project and user settings
4. **Project** - overrides user settings
5. **User** (lowest) - applies when nothing else specifies the setting

## Settings files

The `settings.json` file is the official mechanism for configuring Claude Code through hierarchical settings:

* **User settings** are defined in `~/.claude/settings.json` and apply to all projects.
* **Project settings** are saved in your project directory:
  * `.claude/settings.json` for settings that are checked into source control and shared with your team
  * `.claude/settings.local.json` for settings that are not checked in
* **Managed settings**: For organizations that need centralized control, Claude Code supports `managed-settings.json` and `managed-mcp.json` files that can be deployed to system directories:
  * macOS: `/Library/Application Support/ClaudeCode/`
  * Linux and WSL: `/etc/claude-code/`
  * Windows: `C:\Program Files\ClaudeCode\`

### Example settings.json

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": [
      "Bash(npm run lint)",
      "Bash(npm run test *)",
      "Read(~/.zshrc)"
    ],
    "deny": [
      "Bash(curl *)",
      "Read(./.env)",
      "Read(./.env.*)",
      "Read(./secrets/**)"
    ]
  },
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "otlp"
  },
  "companyAnnouncements": [
    "Welcome to Acme Corp! Review our code guidelines at docs.acme.com",
    "Reminder: Code reviews required for all PRs",
    "New security policy in effect"
  ]
}
```

### Available settings

`settings.json` supports a number of options:

| Key                               | Description                                                                                                                                                                                          | Example                                                                 |
| :-------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------- |
| `apiKeyHelper`                    | Custom script to generate an auth value                                                                                                                                                              | `/bin/generate_temp_api_key.sh`                                         |
| `cleanupPeriodDays`               | Sessions inactive for longer than this period are deleted at startup (default: 30)                                                                                                                   | `20`                                                                    |
| `companyAnnouncements`            | Announcement to display to users at startup                                                                                                                                                          | `["Welcome to Acme Corp!"]`                                             |
| `env`                             | Environment variables applied to every session                                                                                                                                                       | `{"FOO": "bar"}`                                                        |
| `attribution`                     | Customize attribution for git commits and pull requests                                                                                                                                              | `{"commit": "Generated with AI", "pr": ""}`                             |
| `permissions`                     | Permission rules (allow, ask, deny arrays)                                                                                                                                                           |                                                                         |
| `hooks`                           | Custom commands to run at lifecycle events                                                                                                                                                           | See hooks documentation                                                 |
| `disableAllHooks`                 | Disable all hooks and custom status line                                                                                                                                                             | `true`                                                                  |
| `model`                           | Override the default model                                                                                                                                                                           | `"claude-sonnet-4-6"`                                                   |
| `availableModels`                 | Restrict which models users can select                                                                                                                                                               | `["sonnet", "haiku"]`                                                   |
| `outputStyle`                     | Configure an output style                                                                                                                                                                            | `"Explanatory"`                                                         |
| `forceLoginMethod`                | Restrict login to specific account type                                                                                                                                                              | `"claudeai"` or `"console"`                                             |
| `enableAllProjectMcpServers`      | Automatically approve all MCP servers in project `.mcp.json`                                                                                                                                         | `true`                                                                  |
| `alwaysThinkingEnabled`           | Enable extended thinking by default                                                                                                                                                                  | `true`                                                                  |
| `plansDirectory`                  | Customize where plan files are stored                                                                                                                                                                | `"./plans"`                                                             |
| `showTurnDuration`                | Show turn duration messages after responses                                                                                                                                                          | `true`                                                                  |
| `language`                        | Configure Claude's preferred response language                                                                                                                                                       | `"japanese"`                                                            |
| `autoUpdatesChannel`              | Release channel for updates (`"stable"` or `"latest"`)                                                                                                                                               | `"stable"`                                                              |
| `teammateMode`                    | How agent team teammates display                                                                                                                                                                     | `"in-process"`                                                          |

### Permission settings

| Keys                           | Description                                                                  | Example                                                                |
| :----------------------------- | :--------------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| `allow`                        | Array of permission rules to allow tool use                                  | `[ "Bash(git diff *)" ]`                                               |
| `ask`                          | Array of permission rules to ask for confirmation                            | `[ "Bash(git push *)" ]`                                               |
| `deny`                         | Array of permission rules to deny tool use                                   | `[ "WebFetch", "Bash(curl *)", "Read(./.env)" ]`                       |
| `additionalDirectories`        | Additional working directories                                               | `[ "../docs/" ]`                                                       |
| `defaultMode`                  | Default permission mode                                                      | `"acceptEdits"`                                                        |
| `disableBypassPermissionsMode` | Prevent bypass permissions mode (managed settings only)                      | `"disable"`                                                            |

### Sandbox settings

| Keys                          | Description                                                        | Example                         |
| :---------------------------- | :----------------------------------------------------------------- | :------------------------------ |
| `enabled`                     | Enable bash sandboxing (default: false)                            | `true`                          |
| `autoAllowBashIfSandboxed`    | Auto-approve bash commands when sandboxed (default: true)          | `true`                          |
| `excludedCommands`            | Commands that should run outside of the sandbox                    | `["git", "docker"]`             |
| `allowUnsandboxedCommands`    | Allow the escape hatch (default: true)                             | `false`                         |
| `network.allowUnixSockets`    | Unix socket paths accessible in sandbox                            | `["~/.ssh/agent-socket"]`       |
| `network.allowAllUnixSockets` | Allow all Unix socket connections (default: false)                 | `true`                          |
| `network.allowLocalBinding`   | Allow binding to localhost ports (macOS only, default: false)      | `true`                          |
| `network.allowedDomains`      | Domains to allow for outbound traffic (supports wildcards)         | `["github.com", "*.npmjs.org"]` |
| `network.httpProxyPort`       | HTTP proxy port for custom proxy                                   | `8080`                          |
| `network.socksProxyPort`      | SOCKS5 proxy port for custom proxy                                 | `8081`                          |

**Configuration example:**

```json
{
  "sandbox": {
    "enabled": true,
    "autoAllowBashIfSandboxed": true,
    "excludedCommands": ["docker"],
    "network": {
      "allowedDomains": ["github.com", "*.npmjs.org", "registry.yarnpkg.com"],
      "allowUnixSockets": ["/var/run/docker.sock"],
      "allowLocalBinding": true
    }
  },
  "permissions": {
    "deny": [
      "Read(.envrc)",
      "Read(~/.aws/**)"
    ]
  }
}
```

### Attribution settings

| Keys     | Description                                                                                |
| :------- | :----------------------------------------------------------------------------------------- |
| `commit` | Attribution for git commits, including any trailers. Empty string hides commit attribution |
| `pr`     | Attribution for pull request descriptions. Empty string hides PR attribution               |

### Settings precedence

Settings apply in order of precedence. From highest to lowest:

1. **Managed settings** (`managed-settings.json` or server-managed settings)
2. **Command line arguments**
3. **Local project settings** (`.claude/settings.local.json`)
4. **Shared project settings** (`.claude/settings.json`)
5. **User settings** (`~/.claude/settings.json`)

## Plugin configuration

### Plugin settings

```json
{
  "enabledPlugins": {
    "formatter@acme-tools": true,
    "deployer@acme-tools": true,
    "analyzer@security-plugins": false
  },
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": {
        "source": "github",
        "repo": "acme-corp/claude-plugins"
      }
    }
  }
}
```

#### `enabledPlugins`

Controls which plugins are enabled. Format: `"plugin-name@marketplace-name": true/false`

#### `extraKnownMarketplaces`

Defines additional marketplaces that should be made available for the repository.

#### `strictKnownMarketplaces`

**Managed settings only**: Controls which plugin marketplaces users are allowed to add.

* `undefined` (default): No restrictions
* Empty array `[]`: Complete lockdown
* List of sources: Users can only add matching marketplaces

## Environment variables

Claude Code supports many environment variables. Key ones include:

| Variable                                   | Purpose                                                              |
| :----------------------------------------- | :------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`                        | API key for Claude SDK                                               |
| `ANTHROPIC_MODEL`                          | Model setting to use                                                 |
| `CLAUDE_CODE_EFFORT_LEVEL`                 | Effort level: `low`, `medium`, `high` (default)                      |
| `CLAUDE_CODE_ENABLE_TELEMETRY`             | Enable OpenTelemetry data collection                                 |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS`            | Max output tokens (default: 32,000, max: 64,000)                     |
| `CLAUDE_CODE_USE_BEDROCK`                  | Use Amazon Bedrock                                                   |
| `CLAUDE_CODE_USE_VERTEX`                   | Use Google Vertex AI                                                 |
| `CLAUDE_CODE_USE_FOUNDRY`                  | Use Microsoft Foundry                                                |
| `CLAUDE_CODE_DISABLE_AUTO_MEMORY`          | Disable auto memory                                                  |
| `CLAUDE_CODE_SHELL`                        | Override shell detection                                             |
| `DISABLE_AUTOUPDATER`                      | Disable automatic updates                                            |
| `DISABLE_TELEMETRY`                        | Opt out of Statsig telemetry                                         |
| `HTTP_PROXY` / `HTTPS_PROXY`              | Proxy server configuration                                          |
| `MAX_THINKING_TOKENS`                      | Override extended thinking token budget                               |
| `BASH_DEFAULT_TIMEOUT_MS`                  | Default timeout for bash commands                                    |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`          | Auto-compaction trigger percentage (1-100)                           |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | Disable autoupdater, bug command, error reporting, and telemetry     |

All environment variables can also be configured in `settings.json` via the `env` key.

## Tools available to Claude

| Tool                | Description                                                          | Permission Required |
| :------------------ | :------------------------------------------------------------------- | :------------------ |
| **AskUserQuestion** | Asks multiple-choice questions to gather requirements                | No                  |
| **Bash**            | Executes shell commands in your environment                          | Yes                 |
| **TaskOutput**      | Retrieves output from a background task                              | No                  |
| **Edit**            | Makes targeted edits to specific files                               | Yes                 |
| **ExitPlanMode**    | Prompts the user to exit plan mode                                   | No                  |
| **Glob**            | Pattern-based file finding                                           | No                  |
| **Grep**            | Content search across files                                          | No                  |
| **Read**            | Reads file contents                                                  | No                  |
| **Write**           | Creates or overwrites files                                          | Yes                 |
| **WebFetch**        | Fetches content from URLs                                            | Yes                 |
| **WebSearch**       | Performs web searches                                                 | Yes                 |
| **NotebookEdit**    | Edits Jupyter notebook cells                                         | Yes                 |
| **Task**            | Launches subagent to handle complex tasks                            | No                  |
| **TodoWrite**       | Manages todo/task lists                                              | No                  |

## See also

* [Permissions](/en/permissions): fine-grained access control
* [Sandboxing](/en/sandboxing): filesystem and network isolation
* [Hooks](/en/hooks): event-driven automation
* [MCP](/en/mcp): external tool integration
* [Server-managed settings](/en/server-managed-settings): centralized configuration
* [CLI reference](/en/cli-reference): command-line options
