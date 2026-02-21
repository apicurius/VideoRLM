# Hooks reference

> Reference for Claude Code hook events, configuration schema, JSON input/output formats, exit codes, async hooks, prompt hooks, and MCP tool hooks.

> For a quickstart guide with examples, see [Automate workflows with hooks](/en/hooks-guide).

Hooks are user-defined shell commands or LLM prompts that execute automatically at specific points in Claude Code's lifecycle. Use this reference to look up event schemas, configuration options, JSON input/output formats, and advanced features like async hooks and MCP tool hooks. If you're setting up hooks for the first time, start with the [guide](/en/hooks-guide) instead.

## Hook lifecycle

Hooks fire at specific points during a Claude Code session. When an event fires and a matcher matches, Claude Code passes JSON context about the event to your hook handler. For command hooks, this arrives on stdin. Your handler can then inspect the input, take action, and optionally return a decision. Some events fire once per session, while others fire repeatedly inside the agentic loop.

| Event | When it fires |
| :--- | :--- |
| `SessionStart` | When a session begins or resumes |
| `UserPromptSubmit` | When you submit a prompt, before Claude processes it |
| `PreToolUse` | Before a tool call executes. Can block it |
| `PermissionRequest` | When a permission dialog appears |
| `PostToolUse` | After a tool call succeeds |
| `PostToolUseFailure` | After a tool call fails |
| `Notification` | When Claude Code sends a notification |
| `SubagentStart` | When a subagent is spawned |
| `SubagentStop` | When a subagent finishes |
| `Stop` | When Claude finishes responding |
| `TeammateIdle` | When an [agent team](/en/agent-teams) teammate is about to go idle |
| `TaskCompleted` | When a task is being marked as completed |
| `PreCompact` | Before context compaction |
| `SessionEnd` | When a session terminates |

### How a hook resolves

To see how these pieces fit together, consider this `PreToolUse` hook that blocks destructive shell commands. The hook runs `block-rm.sh` before every Bash tool call:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/block-rm.sh"
          }
        ]
      }
    ]
  }
}
```

The script reads the JSON input from stdin, extracts the command, and returns a `permissionDecision` of `"deny"` if it contains `rm -rf`:

```bash
#!/bin/bash
# .claude/hooks/block-rm.sh
COMMAND=$(jq -r '.tool_input.command')

if echo "$COMMAND" | grep -q 'rm -rf'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "Destructive command blocked by hook"
    }
  }'
else
  exit 0  # allow the command
fi
```

Now suppose Claude Code decides to run `Bash "rm -rf /tmp/build"`. Here's what happens:

1. **Event fires**: The `PreToolUse` event fires. Claude Code sends the tool input as JSON on stdin to the hook.
2. **Matcher checks**: The matcher `"Bash"` matches the tool name, so `block-rm.sh` runs.
3. **Hook handler runs**: The script extracts `"rm -rf /tmp/build"` from the input and finds `rm -rf`, so it prints a decision to stdout.
4. **Claude Code acts on the result**: Claude Code reads the JSON decision, blocks the tool call, and shows Claude the reason.

## Configuration

Hooks are defined in JSON settings files. The configuration has three levels of nesting:

1. Choose a [hook event](#hook-events) to respond to, like `PreToolUse` or `Stop`
2. Add a [matcher group](#matcher-patterns) to filter when it fires, like "only for the Bash tool"
3. Define one or more [hook handlers](#hook-handler-fields) to run when matched

### Hook locations

Where you define a hook determines its scope:

| Location | Scope | Shareable |
| :--- | :--- | :--- |
| `~/.claude/settings.json` | All your projects | No, local to your machine |
| `.claude/settings.json` | Single project | Yes, can be committed to the repo |
| `.claude/settings.local.json` | Single project | No, gitignored |
| Managed policy settings | Organization-wide | Yes, admin-controlled |
| [Plugin](/en/plugins) `hooks/hooks.json` | When plugin is enabled | Yes, bundled with the plugin |
| [Skill](/en/skills) or [agent](/en/sub-agents) frontmatter | While the component is active | Yes, defined in the component file |

For details on settings file resolution, see [settings](/en/settings). Enterprise administrators can use `allowManagedHooksOnly` to block user, project, and plugin hooks. See [Hook configuration](/en/settings#hook-configuration).

### Matcher patterns

The `matcher` field is a regex string that filters when hooks fire. Use `"*"`, `""`, or omit `matcher` entirely to match all occurrences. Each event type matches on a different field:

| Event | What the matcher filters | Example matcher values |
| :--- | :--- | :--- |
| `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest` | tool name | `Bash`, `Edit\|Write`, `mcp__.*` |
| `SessionStart` | how the session started | `startup`, `resume`, `clear`, `compact` |
| `SessionEnd` | why the session ended | `clear`, `logout`, `prompt_input_exit`, `bypass_permissions_disabled`, `other` |
| `Notification` | notification type | `permission_prompt`, `idle_prompt`, `auth_success`, `elicitation_dialog` |
| `SubagentStart` | agent type | `Bash`, `Explore`, `Plan`, or custom agent names |
| `PreCompact` | what triggered compaction | `manual`, `auto` |
| `SubagentStop` | agent type | same values as `SubagentStart` |
| `UserPromptSubmit`, `Stop`, `TeammateIdle`, `TaskCompleted` | no matcher support | always fires on every occurrence |

The matcher is a regex, so `Edit|Write` matches either tool and `Notebook.*` matches any tool starting with Notebook.

#### Match MCP tools

[MCP](/en/mcp) server tools appear as regular tools in tool events (`PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest`), so you can match them the same way you match any other tool name.

MCP tools follow the naming pattern `mcp__<server>__<tool>`, for example:

* `mcp__memory__create_entities`: Memory server's create entities tool
* `mcp__filesystem__read_file`: Filesystem server's read file tool
* `mcp__github__search_repositories`: GitHub server's search tool

Use regex patterns to target specific MCP tools or groups of tools:

* `mcp__memory__.*` matches all tools from the `memory` server
* `mcp__.*__write.*` matches any tool containing "write" from any server

### Hook handler fields

Each object in the inner `hooks` array is a hook handler. There are three types:

* **Command hooks** (`type: "command"`): run a shell command
* **Prompt hooks** (`type: "prompt"`): send a prompt to a Claude model for single-turn evaluation
* **Agent hooks** (`type: "agent"`): spawn a subagent that can use tools to verify conditions

#### Common fields

| Field | Required | Description |
| :--- | :--- | :--- |
| `type` | yes | `"command"`, `"prompt"`, or `"agent"` |
| `timeout` | no | Seconds before canceling. Defaults: 600 for command, 30 for prompt, 60 for agent |
| `statusMessage` | no | Custom spinner message displayed while the hook runs |
| `once` | no | If `true`, runs only once per session then is removed. Skills only, not agents |

#### Command hook fields

| Field | Required | Description |
| :--- | :--- | :--- |
| `command` | yes | Shell command to execute |
| `async` | no | If `true`, runs in the background without blocking |

#### Prompt and agent hook fields

| Field | Required | Description |
| :--- | :--- | :--- |
| `prompt` | yes | Prompt text to send to the model. Use `$ARGUMENTS` as a placeholder for the hook input JSON |
| `model` | no | Model to use for evaluation. Defaults to a fast model |

### Reference scripts by path

Use environment variables to reference hook scripts relative to the project or plugin root:

* `$CLAUDE_PROJECT_DIR`: the project root
* `${CLAUDE_PLUGIN_ROOT}`: the plugin's root directory

### Hooks in skills and agents

Hooks can be defined directly in [skills](/en/skills) and [subagents](/en/sub-agents) using frontmatter. These hooks are scoped to the component's lifecycle and only run when that component is active.

```yaml
---
name: secure-operations
description: Perform operations with security checks
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/security-check.sh"
---
```

### The `/hooks` menu

Type `/hooks` in Claude Code to open the interactive hooks manager, where you can view, add, and delete hooks without editing settings files directly.

### Disable or remove hooks

To temporarily disable all hooks without removing them, set `"disableAllHooks": true` in your settings file or use the toggle in the `/hooks` menu.

## Hook input and output

Hooks receive JSON data via stdin and communicate results through exit codes, stdout, and stderr.

### Common input fields

All hook events receive these fields via stdin as JSON:

| Field | Description |
| :--- | :--- |
| `session_id` | Current session identifier |
| `transcript_path` | Path to conversation JSON |
| `cwd` | Current working directory when the hook is invoked |
| `permission_mode` | Current permission mode |
| `hook_event_name` | Name of the event that fired |

### Exit code output

**Exit 0**: success. Claude Code parses stdout for JSON output fields.

**Exit 2**: blocking error. stderr text is fed back to Claude as an error message.

**Any other exit code**: non-blocking error. stderr is shown in verbose mode.

#### Exit code 2 behavior per event

| Hook event | Can block? | What happens on exit 2 |
| :--- | :--- | :--- |
| `PreToolUse` | Yes | Blocks the tool call |
| `PermissionRequest` | Yes | Denies the permission |
| `UserPromptSubmit` | Yes | Blocks prompt processing and erases the prompt |
| `Stop` | Yes | Prevents Claude from stopping, continues the conversation |
| `SubagentStop` | Yes | Prevents the subagent from stopping |
| `TeammateIdle` | Yes | Prevents the teammate from going idle |
| `TaskCompleted` | Yes | Prevents the task from being marked as completed |
| `PostToolUse` | No | Shows stderr to Claude (tool already ran) |
| `PostToolUseFailure` | No | Shows stderr to Claude (tool already failed) |
| `Notification` | No | Shows stderr to user only |
| `SubagentStart` | No | Shows stderr to user only |
| `SessionStart` | No | Shows stderr to user only |
| `SessionEnd` | No | Shows stderr to user only |
| `PreCompact` | No | Shows stderr to user only |

### JSON output

Exit codes let you allow or block, but JSON output gives you finer-grained control. Exit 0 and print a JSON object to stdout.

| Field | Default | Description |
| :--- | :--- | :--- |
| `continue` | `true` | If `false`, Claude stops processing entirely after the hook runs |
| `stopReason` | none | Message shown to the user when `continue` is `false` |
| `suppressOutput` | `false` | If `true`, hides stdout from verbose mode output |
| `systemMessage` | none | Warning message shown to the user |

#### Decision control

| Events | Decision pattern | Key fields |
| :--- | :--- | :--- |
| UserPromptSubmit, PostToolUse, PostToolUseFailure, Stop, SubagentStop | Top-level `decision` | `decision: "block"`, `reason` |
| TeammateIdle, TaskCompleted | Exit code only | Exit code 2 blocks the action |
| PreToolUse | `hookSpecificOutput` | `permissionDecision` (allow/deny/ask), `permissionDecisionReason` |
| PermissionRequest | `hookSpecificOutput` | `decision.behavior` (allow/deny) |

## Hook events

### SessionStart

Runs when Claude Code starts a new session or resumes an existing session.

Matcher values: `startup`, `resume`, `clear`, `compact`

Additional input fields: `source`, `model`, optionally `agent_type`.

Can return `additionalContext` in `hookSpecificOutput`. Supports `CLAUDE_ENV_FILE` for persisting environment variables.

### UserPromptSubmit

Runs when the user submits a prompt, before Claude processes it.

Additional input: `prompt` field.

Can block with `decision: "block"` and `reason`. Can add context via stdout text or `additionalContext`.

### PreToolUse

Runs before a tool call executes. Matches on tool name.

Additional input: `tool_name`, `tool_input`, `tool_use_id`.

Decision control via `hookSpecificOutput`: `permissionDecision` (allow/deny/ask), `permissionDecisionReason`, `updatedInput`, `additionalContext`.

### PermissionRequest

Runs when a permission dialog appears. Matches on tool name.

Additional input: `tool_name`, `tool_input`, `permission_suggestions`.

Decision control via `hookSpecificOutput.decision`: `behavior` (allow/deny), `updatedInput`, `updatedPermissions`, `message`, `interrupt`.

### PostToolUse

Runs after a tool call succeeds. Matches on tool name.

Additional input: `tool_input`, `tool_response`, `tool_use_id`.

Can return `decision: "block"` with `reason`, or `additionalContext`. For MCP tools: `updatedMCPToolOutput`.

### PostToolUseFailure

Runs when a tool execution fails. Matches on tool name.

Additional input: `tool_input`, `tool_use_id`, `error`, `is_interrupt`.

Can return `additionalContext`.

### Notification

Runs when Claude Code sends notifications. Matches on notification type: `permission_prompt`, `idle_prompt`, `auth_success`, `elicitation_dialog`.

Additional input: `message`, `title`, `notification_type`.

### SubagentStart

Runs when a subagent is spawned. Matches on agent type.

Additional input: `agent_id`, `agent_type`.

Can return `additionalContext`.

### SubagentStop

Runs when a subagent finishes. Matches on agent type.

Additional input: `stop_hook_active`, `agent_id`, `agent_type`, `agent_transcript_path`, `last_assistant_message`.

Uses same decision control as Stop hooks.

### Stop

Runs when the main Claude Code agent has finished responding.

Additional input: `stop_hook_active`, `last_assistant_message`.

Can return `decision: "block"` with `reason` to prevent Claude from stopping.

### TeammateIdle

Runs when an agent team teammate is about to go idle.

Additional input: `teammate_name`, `team_name`.

Uses exit codes only (exit 2 to block).

### TaskCompleted

Runs when a task is being marked as completed.

Additional input: `task_id`, `task_subject`, `task_description`, `teammate_name`, `team_name`.

Uses exit codes only (exit 2 to block).

### PreCompact

Runs before context compaction. Matches on trigger: `manual`, `auto`.

Additional input: `trigger`, `custom_instructions`.

### SessionEnd

Runs when a session terminates.

Additional input: `reason` (clear, logout, prompt_input_exit, bypass_permissions_disabled, other).

Cannot block session termination.

## Prompt-based hooks

Prompt-based hooks (`type: "prompt"`) use an LLM to evaluate whether to allow or block an action. The LLM responds with `{ "ok": true }` or `{ "ok": false, "reason": "..." }`.

Supported events: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest`, `UserPromptSubmit`, `Stop`, `SubagentStop`, `TaskCompleted`.

## Agent-based hooks

Agent-based hooks (`type: "agent"`) spawn a subagent that can use tools like Read, Grep, and Glob to verify conditions before returning a decision. Same response schema as prompt hooks.

## Run hooks in the background

Set `"async": true` on command hooks to run them in the background. Async hooks cannot block or return decisions. Only `type: "command"` hooks support async.

## Security considerations

> **Warning**: Hooks execute shell commands with your full user permissions. They can modify, delete, or access any files your user account can access. Review and test all hook commands before adding them to your configuration.

### Security best practices

* **Validate and sanitize inputs**: never trust input data blindly
* **Always quote shell variables**: use `"$VAR"` not `$VAR`
* **Block path traversal**: check for `..` in file paths
* **Use absolute paths**: specify full paths for scripts
* **Skip sensitive files**: avoid `.env`, `.git/`, keys, etc.

## Debug hooks

Run `claude --debug` to see hook execution details. Toggle verbose mode with `Ctrl+O` to see hook progress in the transcript.

For troubleshooting common issues, see [Limitations and troubleshooting](/en/hooks-guide#limitations-and-troubleshooting) in the guide.
