# Create custom subagents

> Create and use specialized AI subagents in Claude Code for task-specific workflows and improved context management.

Subagents are specialized AI assistants that handle specific types of tasks. Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions. When Claude encounters a task that matches a subagent's description, it delegates to that subagent, which works independently and returns results.

> If you need multiple agents working in parallel and communicating with each other, see agent teams instead. Subagents work within a single session; agent teams coordinate across separate sessions.

Subagents help you:

* **Preserve context** by keeping exploration and implementation out of your main conversation
* **Enforce constraints** by limiting which tools a subagent can use
* **Reuse configurations** across projects with user-level subagents
* **Specialize behavior** with focused system prompts for specific domains
* **Control costs** by routing tasks to faster, cheaper models like Haiku

## Built-in subagents

Claude Code includes built-in subagents that Claude automatically uses when appropriate.

### Explore

A fast, read-only agent optimized for searching and analyzing codebases.

* **Model**: Haiku (fast, low-latency)
* **Tools**: Read-only tools (denied access to Write and Edit tools)
* **Purpose**: File discovery, code search, codebase exploration

### Plan

A research agent used during plan mode to gather context before presenting a plan.

* **Model**: Inherits from main conversation
* **Tools**: Read-only tools (denied access to Write and Edit tools)
* **Purpose**: Codebase research for planning

### General-purpose

A capable agent for complex, multi-step tasks that require both exploration and action.

* **Model**: Inherits from main conversation
* **Tools**: All tools
* **Purpose**: Complex research, multi-step operations, code modifications

### Other

| Agent             | Model    | When Claude uses it                                      |
| :---------------- | :------- | :------------------------------------------------------- |
| Bash              | Inherits | Running terminal commands in a separate context          |
| statusline-setup  | Sonnet   | When you run `/statusline` to configure your status line |
| Claude Code Guide | Haiku    | When you ask questions about Claude Code features        |

## Quickstart: create your first subagent

Subagents are defined in Markdown files with YAML frontmatter. You can create them manually or use the `/agents` command.

1. In Claude Code, run `/agents`
2. Select **Create new agent**, then choose **User-level**
3. Select **Generate with Claude** and describe the subagent
4. Select tools (e.g., Read-only tools for a reviewer)
5. Select model (e.g., Sonnet for balanced capability and speed)
6. Choose a background color
7. Save and try it out

## Configure subagents

### Use the /agents command

The `/agents` command provides an interactive interface for managing subagents. Run `/agents` to:

* View all available subagents (built-in, user, project, and plugin)
* Create new subagents with guided setup or Claude generation
* Edit existing subagent configuration and tool access
* Delete custom subagents

### Choose the subagent scope

| Location                     | Scope                   | Priority    | How to create                         |
| :--------------------------- | :---------------------- | :---------- | :------------------------------------ |
| `--agents` CLI flag          | Current session         | 1 (highest) | Pass JSON when launching Claude Code  |
| `.claude/agents/`            | Current project         | 2           | Interactive or manual                 |
| `~/.claude/agents/`          | All your projects       | 3           | Interactive or manual                 |
| Plugin's `agents/` directory | Where plugin is enabled | 4 (lowest)  | Installed with plugins                |

**CLI-defined subagents** are passed as JSON when launching Claude Code:

```bash
claude --agents '{
  "code-reviewer": {
    "description": "Expert code reviewer. Use proactively after code changes.",
    "prompt": "You are a senior code reviewer. Focus on code quality, security, and best practices.",
    "tools": ["Read", "Grep", "Glob", "Bash"],
    "model": "sonnet"
  }
}'
```

### Write subagent files

Subagent files use YAML frontmatter for configuration, followed by the system prompt in Markdown:

```markdown
---
name: code-reviewer
description: Reviews code for quality and best practices
tools: Read, Glob, Grep
model: sonnet
---

You are a code reviewer. When invoked, analyze the code and provide
specific, actionable feedback on quality, security, and best practices.
```

#### Supported frontmatter fields

| Field             | Required | Description                                                                                                                                                                                                                                                                 |
| :---------------- | :------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`            | Yes      | Unique identifier using lowercase letters and hyphens                                                                                                                                                                                                                       |
| `description`     | Yes      | When Claude should delegate to this subagent                                                                                                                                                                                                                                |
| `tools`           | No       | Tools the subagent can use. Inherits all tools if omitted                                                                                                                                                                                                                   |
| `disallowedTools` | No       | Tools to deny, removed from inherited or specified list                                                                                                                                                                                                                     |
| `model`           | No       | Model to use: `sonnet`, `opus`, `haiku`, or `inherit`. Defaults to `inherit`                                                                                                                                                                                                |
| `permissionMode`  | No       | Permission mode: `default`, `acceptEdits`, `dontAsk`, `bypassPermissions`, or `plan`                                                                                                                                                                                        |
| `maxTurns`        | No       | Maximum number of agentic turns before the subagent stops                                                                                                                                                                                                                   |
| `skills`          | No       | Skills to load into the subagent's context at startup                                                                                                                                                                                                                       |
| `mcpServers`      | No       | MCP servers available to this subagent                                                                                                                                                                                                                                      |
| `hooks`           | No       | Lifecycle hooks scoped to this subagent                                                                                                                                                                                                                                     |
| `memory`          | No       | Persistent memory scope: `user`, `project`, or `local`                                                                                                                                                                                                                      |

### Choose a model

* **Model alias**: Use one of the available aliases: `sonnet`, `opus`, or `haiku`
* **inherit**: Use the same model as the main conversation
* **Omitted**: Defaults to `inherit`

### Control subagent capabilities

#### Available tools

Subagents can use any of Claude Code's internal tools. By default, subagents inherit all tools from the main conversation, including MCP tools.

```yaml
---
name: safe-researcher
description: Research agent with restricted capabilities
tools: Read, Grep, Glob, Bash
disallowedTools: Write, Edit
---
```

#### Restrict which subagents can be spawned

Use `Task(agent_type)` syntax in the `tools` field:

```yaml
---
name: coordinator
description: Coordinates work across specialized agents
tools: Task(worker, researcher), Read, Bash
---
```

#### Permission modes

| Mode                | Behavior                                                           |
| :------------------ | :----------------------------------------------------------------- |
| `default`           | Standard permission checking with prompts                          |
| `acceptEdits`       | Auto-accept file edits                                             |
| `dontAsk`           | Auto-deny permission prompts (explicitly allowed tools still work) |
| `bypassPermissions` | Skip all permission checks                                         |
| `plan`              | Plan mode (read-only exploration)                                  |

#### Preload skills into subagents

```yaml
---
name: api-developer
description: Implement API endpoints following team conventions
skills:
  - api-conventions
  - error-handling-patterns
---

Implement API endpoints. Follow the conventions and patterns from the preloaded skills.
```

#### Enable persistent memory

The `memory` field gives the subagent a persistent directory that survives across conversations.

```yaml
---
name: code-reviewer
description: Reviews code for quality and best practices
memory: user
---

You are a code reviewer. As you review code, update your agent memory with
patterns, conventions, and recurring issues you discover.
```

| Scope     | Location                                      | Use when                                                                                    |
| :-------- | :-------------------------------------------- | :------------------------------------------------------------------------------------------ |
| `user`    | `~/.claude/agent-memory/<name-of-agent>/`     | the subagent should remember learnings across all projects                                  |
| `project` | `.claude/agent-memory/<name-of-agent>/`       | the subagent's knowledge is project-specific and shareable via version control              |
| `local`   | `.claude/agent-memory-local/<name-of-agent>/` | the subagent's knowledge is project-specific but should not be checked into version control |

#### Disable specific subagents

```json
{
  "permissions": {
    "deny": ["Task(Explore)", "Task(my-custom-agent)"]
  }
}
```

### Define hooks for subagents

#### Hooks in subagent frontmatter

All hook events are supported. The most common events for subagents:

| Event         | Matcher input | When it fires                                                       |
| :------------ | :------------ | :------------------------------------------------------------------ |
| `PreToolUse`  | Tool name     | Before the subagent uses a tool                                     |
| `PostToolUse` | Tool name     | After the subagent uses a tool                                      |
| `Stop`        | (none)        | When the subagent finishes (converted to `SubagentStop` at runtime) |

```yaml
---
name: code-reviewer
description: Review code changes with automatic linting
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-command.sh $TOOL_INPUT"
  PostToolUse:
    - matcher: "Edit|Write"
      hooks:
        - type: command
          command: "./scripts/run-linter.sh"
---
```

#### Project-level hooks for subagent events

| Event           | Matcher input   | When it fires                    |
| :-------------- | :-------------- | :------------------------------- |
| `SubagentStart` | Agent type name | When a subagent begins execution |
| `SubagentStop`  | Agent type name | When a subagent completes        |

## Work with subagents

### Understand automatic delegation

Claude automatically delegates tasks based on the task description, the `description` field in subagent configurations, and current context.

You can also request a specific subagent explicitly:

```
Use the test-runner subagent to fix failing tests
Have the code-reviewer subagent look at my recent changes
```

### Run subagents in foreground or background

* **Foreground subagents** block the main conversation until complete.
* **Background subagents** run concurrently while you continue working.

Press **Ctrl+B** to background a running task.

### Common patterns

#### Isolate high-volume operations

```
Use a subagent to run the test suite and report only the failing tests with their error messages
```

#### Run parallel research

```
Research the authentication, database, and API modules in parallel using separate subagents
```

#### Chain subagents

```
Use the code-reviewer subagent to find performance issues, then use the optimizer subagent to fix them
```

### Resume subagents

Each subagent invocation creates a new instance with fresh context. To continue an existing subagent's work, ask Claude to resume it:

```
Continue that code review and now analyze the authorization logic
```

## Example subagents

### Code reviewer

```markdown
---
name: code-reviewer
description: Expert code review specialist. Proactively reviews code for quality, security, and maintainability.
tools: Read, Grep, Glob, Bash
model: inherit
---

You are a senior code reviewer ensuring high standards of code quality and security.

When invoked:
1. Run git diff to see recent changes
2. Focus on modified files
3. Begin review immediately

Review checklist:
- Code is clear and readable
- Functions and variables are well-named
- No duplicated code
- Proper error handling
- No exposed secrets or API keys
- Input validation implemented
- Good test coverage
- Performance considerations addressed

Provide feedback organized by priority:
- Critical issues (must fix)
- Warnings (should fix)
- Suggestions (consider improving)
```

### Debugger

```markdown
---
name: debugger
description: Debugging specialist for errors, test failures, and unexpected behavior.
tools: Read, Edit, Bash, Grep, Glob
---

You are an expert debugger specializing in root cause analysis.

When invoked:
1. Capture error message and stack trace
2. Identify reproduction steps
3. Isolate the failure location
4. Implement minimal fix
5. Verify solution works

For each issue, provide:
- Root cause explanation
- Evidence supporting the diagnosis
- Specific code fix
- Testing approach
- Prevention recommendations
```

### Data scientist

```markdown
---
name: data-scientist
description: Data analysis expert for SQL queries, BigQuery operations, and data insights.
tools: Bash, Read, Write
model: sonnet
---

You are a data scientist specializing in SQL and BigQuery analysis.

When invoked:
1. Understand the data analysis requirement
2. Write efficient SQL queries
3. Use BigQuery command line tools (bq) when appropriate
4. Analyze and summarize results
5. Present findings clearly
```

## Next steps

* Distribute subagents with plugins to share subagents across teams or projects
* Run Claude Code programmatically with the Agent SDK for CI/CD and automation
* Use MCP servers to give subagents access to external tools and data
