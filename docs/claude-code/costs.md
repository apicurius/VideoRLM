# Manage costs effectively

> Track token usage, set team spend limits, and reduce Claude Code costs with context management, model selection, extended thinking settings, and preprocessing hooks.

Claude Code consumes tokens for each interaction. Costs vary based on codebase size, query complexity, and conversation length. The average cost is $6 per developer per day, with daily costs remaining below $12 for 90% of users.

For team usage, Claude Code charges by API token consumption. On average, Claude Code costs ~$100-200/developer per month with Sonnet 4.6 though there is large variance depending on how many instances users are running and whether they're using it in automation.

## Track your costs

### Using the `/cost` command

> **Note:** The `/cost` command shows API token usage and is intended for API users. Claude Max and Pro subscribers have usage included in their subscription, so `/cost` data isn't relevant for billing purposes. Subscribers can use `/stats` to view usage patterns.

The `/cost` command provides detailed token usage statistics for your current session:

```
Total cost:            $0.55
Total duration (API):  6m 19.7s
Total duration (wall): 6h 33m 10.2s
Total code changes:    0 lines added, 0 lines removed
```

## Managing costs for teams

When using Claude API, you can set workspace spend limits on the total Claude Code workspace spend. Admins can view cost and usage reporting in the Console.

> **Note:** When you first authenticate Claude Code with your Claude Console account, a workspace called "Claude Code" is automatically created for you.

On Bedrock, Vertex, and Foundry, Claude Code does not send metrics from your cloud. To get cost metrics, several large enterprises reported using LiteLLM, which is an open-source tool that helps companies track spend by key.

### Rate limit recommendations

| Team size     | TPM per user | RPM per user |
| ------------- | ------------ | ------------ |
| 1-5 users     | 200k-300k    | 5-7          |
| 5-20 users    | 100k-150k    | 2.5-3.5      |
| 20-50 users   | 50k-75k      | 1.25-1.75    |
| 50-100 users  | 25k-35k      | 0.62-0.87    |
| 100-500 users | 15k-20k      | 0.37-0.47    |
| 500+ users    | 10k-15k      | 0.25-0.35    |

The TPM per user decreases as team size grows because fewer users tend to use Claude Code concurrently in larger organizations.

### Agent team token costs

Agent teams spawn multiple Claude Code instances, each with its own context window. Token usage scales with the number of active teammates and how long each one runs.

To keep agent team costs manageable:

* Use Sonnet for teammates
* Keep teams small
* Keep spawn prompts focused
* Clean up teams when work is done
* Agent teams are disabled by default. Set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` to enable them.

## Reduce token usage

Token costs scale with context size: the more context Claude processes, the more tokens you use.

### Manage context proactively

* **Clear between tasks**: Use `/clear` to start fresh when switching to unrelated work
* **Add custom compaction instructions**: `/compact Focus on code samples and API usage`

You can also customize compaction behavior in your CLAUDE.md:

```markdown
# Compact instructions

When you are using compact, please focus on test output and code changes
```

### Choose the right model

Sonnet handles most coding tasks well and costs less than Opus. Reserve Opus for complex architectural decisions or multi-step reasoning. Use `/model` to switch models mid-session. For simple subagent tasks, specify `model: haiku` in your subagent configuration.

### Reduce MCP server overhead

Each MCP server adds tool definitions to your context, even when idle.

* **Prefer CLI tools when available**: Tools like `gh`, `aws`, `gcloud`, and `sentry-cli` are more context-efficient
* **Disable unused servers**: Run `/mcp` to see configured servers and disable any you're not actively using
* **Tool search is automatic**: Set a lower threshold with `ENABLE_TOOL_SEARCH=auto:<N>`

### Install code intelligence plugins for typed languages

Code intelligence plugins give Claude precise symbol navigation instead of text-based search, reducing unnecessary file reads.

### Offload processing to hooks and skills

Custom hooks can preprocess data before Claude sees it. A skill can give Claude domain knowledge so it doesn't have to explore.

For example, this PreToolUse hook filters test output to show only failures:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/filter-test-output.sh"
          }
        ]
      }
    ]
  }
}
```

```bash
#!/bin/bash
input=$(cat)
cmd=$(echo "$input" | jq -r '.tool_input.command')

# If running tests, filter to show only failures
if [[ "$cmd" =~ ^(npm test|pytest|go test) ]]; then
  filtered_cmd="$cmd 2>&1 | grep -A 5 -E '(FAIL|ERROR|error:)' | head -100"
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"allow\",\"updatedInput\":{\"command\":\"$filtered_cmd\"}}}"
else
  echo "{}"
fi
```

### Move instructions from CLAUDE.md to skills

Your CLAUDE.md file is loaded into context at session start. Skills load on-demand only when invoked, so moving specialized instructions into skills keeps your base context smaller. Aim to keep CLAUDE.md under ~500 lines.

### Adjust extended thinking

Extended thinking is enabled by default with a budget of 31,999 tokens. For simpler tasks, reduce costs by lowering the effort level in `/model` for Opus 4.6, disabling thinking in `/config`, or lowering the budget (e.g., `MAX_THINKING_TOKENS=8000`).

### Delegate verbose operations to subagents

Running tests, fetching documentation, or processing log files can consume significant context. Delegate these to subagents so the verbose output stays in the subagent's context.

### Manage agent team costs

Agent teams use approximately 7x more tokens than standard sessions when teammates run in plan mode.

### Write specific prompts

Vague requests like "improve this codebase" trigger broad scanning. Specific requests like "add input validation to the login function in auth.ts" let Claude work efficiently.

### Work efficiently on complex tasks

* **Use plan mode for complex tasks**: Press Shift+Tab to enter plan mode before implementation
* **Course-correct early**: Press Escape to stop immediately. Use `/rewind` to restore to a previous checkpoint
* **Give verification targets**: Include test cases, paste screenshots, or define expected output
* **Test incrementally**: Write one file, test it, then continue

## Background token usage

Claude Code uses tokens for some background functionality even when idle:

* **Conversation summarization**: Background jobs that summarize previous conversations for the `claude --resume` feature
* **Command processing**: Some commands like `/cost` may generate requests to check status

These background processes consume a small amount of tokens (typically under $0.04 per session).

## Understanding changes in Claude Code behavior

Claude Code regularly receives updates that may change how features work. Run `claude --version` to check your current version. For specific billing questions, contact Anthropic support through your Console account.
