# Customize your status line

> Configure a custom status bar to monitor context window usage, costs, and git status in Claude Code

The status line is a customizable bar at the bottom of Claude Code that runs any shell script you configure. It receives JSON session data on stdin and displays whatever your script prints, giving you a persistent, at-a-glance view of context usage, costs, git status, or anything else you want to track.

Status lines are useful when you:

* Want to monitor context window usage as you work
* Need to track session costs
* Work across multiple sessions and need to distinguish them
* Want git branch and status always visible

## Set up a status line

Use the `/statusline` command to have Claude Code generate a script for you, or manually create a script and add it to your settings.

### Use the /statusline command

The `/statusline` command accepts natural language instructions describing what you want displayed. Claude Code generates a script file in `~/.claude/` and updates your settings automatically:

```
/statusline show model name and context percentage with a progress bar
```

### Manually configure a status line

Add a `statusLine` field to your user settings (`~/.claude/settings.json`, where `~` is your home directory) or project settings. Set `type` to `"command"` and point `command` to a script path or an inline shell command.

```json
{
  "statusLine": {
    "type": "command",
    "command": "~/.claude/statusline.sh",
    "padding": 2
  }
}
```

The `command` field runs in a shell, so you can also use inline commands instead of a script file. This example uses `jq` to parse the JSON input and display the model name and context percentage:

```json
{
  "statusLine": {
    "type": "command",
    "command": "jq -r '\"[\\(.model.display_name)] \\(.context_window.used_percentage // 0)% context\"'"
  }
}
```

The optional `padding` field adds extra horizontal spacing (in characters) to the status line content. Defaults to `0`.

### Disable the status line

Run `/statusline` and ask it to remove or clear your status line (e.g., `/statusline delete`, `/statusline clear`, `/statusline remove it`). You can also manually delete the `statusLine` field from your settings.json.

## Build a status line step by step

This walkthrough shows what's happening under the hood by manually creating a status line that displays the current model, working directory, and context window usage percentage.

> Running `/statusline` with a description of what you want configures all of this for you automatically.

**Step 1: Create a script that reads JSON and prints output**

Claude Code sends JSON data to your script via stdin. This script uses `jq`, a command-line JSON parser you may need to install, to extract the model name, directory, and context percentage, then prints a formatted line.

Save this to `~/.claude/statusline.sh`:

```bash
#!/bin/bash
# Read JSON data that Claude Code sends to stdin
input=$(cat)

# Extract fields using jq
MODEL=$(echo "$input" | jq -r '.model.display_name')
DIR=$(echo "$input" | jq -r '.workspace.current_dir')
# The "// 0" provides a fallback if the field is null
PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)

# Output the status line - ${DIR##*/} extracts just the folder name
echo "[$MODEL] ${DIR##*/} | ${PCT}% context"
```

**Step 2: Make it executable**

```bash
chmod +x ~/.claude/statusline.sh
```

**Step 3: Add to settings**

Add this configuration to `~/.claude/settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "~/.claude/statusline.sh"
  }
}
```

Your status line appears at the bottom of the interface. Settings reload automatically, but changes won't appear until your next interaction with Claude Code.

## How status lines work

Claude Code runs your script and pipes JSON session data to it via stdin. Your script reads the JSON, extracts what it needs, and prints text to stdout. Claude Code displays whatever your script prints.

**When it updates**

Your script runs after each new assistant message, when the permission mode changes, or when vim mode toggles. Updates are debounced at 300ms. If a new update triggers while your script is still running, the in-flight execution is cancelled. If you edit your script, the changes won't appear until your next interaction with Claude Code triggers an update.

**What your script can output**

* **Multiple lines**: each `echo` or `print` statement displays as a separate row.
* **Colors**: use ANSI escape codes like `\033[32m` for green (terminal must support them).
* **Links**: use OSC 8 escape sequences to make text clickable (Cmd+click on macOS, Ctrl+click on Windows/Linux). Requires a terminal that supports hyperlinks like iTerm2, Kitty, or WezTerm.

> The status line runs locally and does not consume API tokens. It temporarily hides during certain UI interactions, including autocomplete suggestions, the help menu, and permission prompts.

## Available data

Claude Code sends the following JSON fields to your script via stdin:

| Field                                                                     | Description                                                                                                                                                                                  |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.id`, `model.display_name`                                          | Current model identifier and display name                                                                                                                                                    |
| `cwd`, `workspace.current_dir`                                            | Current working directory. Both fields contain the same value; `workspace.current_dir` is preferred.                                                                                         |
| `workspace.project_dir`                                                   | Directory where Claude Code was launched                                                                                                                                                     |
| `cost.total_cost_usd`                                                     | Total session cost in USD                                                                                                                                                                    |
| `cost.total_duration_ms`                                                  | Total wall-clock time since the session started, in milliseconds                                                                                                                             |
| `cost.total_api_duration_ms`                                              | Total time spent waiting for API responses in milliseconds                                                                                                                                   |
| `cost.total_lines_added`, `cost.total_lines_removed`                      | Lines of code changed                                                                                                                                                                        |
| `context_window.total_input_tokens`, `context_window.total_output_tokens` | Cumulative token counts across the session                                                                                                                                                   |
| `context_window.context_window_size`                                      | Maximum context window size in tokens. 200000 by default, or 1000000 for models with extended context.                                                                                       |
| `context_window.used_percentage`                                          | Pre-calculated percentage of context window used                                                                                                                                             |
| `context_window.remaining_percentage`                                     | Pre-calculated percentage of context window remaining                                                                                                                                        |
| `context_window.current_usage`                                            | Token counts from the last API call                                                                                                                                                          |
| `exceeds_200k_tokens`                                                     | Whether the total token count from the most recent API response exceeds 200k                                                                                                                 |
| `session_id`                                                              | Unique session identifier                                                                                                                                                                    |
| `transcript_path`                                                         | Path to conversation transcript file                                                                                                                                                         |
| `version`                                                                 | Claude Code version                                                                                                                                                                          |
| `output_style.name`                                                       | Name of the current output style                                                                                                                                                             |
| `vim.mode`                                                                | Current vim mode (`NORMAL` or `INSERT`) when vim mode is enabled                                                                                                                             |
| `agent.name`                                                              | Agent name when running with the `--agent` flag or agent settings configured                                                                                                                 |

### Context window fields

The `context_window` object provides two ways to track context usage:

* **Cumulative totals** (`total_input_tokens`, `total_output_tokens`): sum of all tokens across the entire session
* **Current usage** (`current_usage`): token counts from the most recent API call

The `current_usage` object contains:

* `input_tokens`: input tokens in current context
* `output_tokens`: output tokens generated
* `cache_creation_input_tokens`: tokens written to cache
* `cache_read_input_tokens`: tokens read from cache

The `used_percentage` field is calculated from input tokens only: `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. It does not include `output_tokens`.

The `current_usage` object is `null` before the first API call in a session.

## Examples

These examples show common status line patterns. To use any example:

1. Save the script to a file like `~/.claude/statusline.sh` (or `.py`/`.js`)
2. Make it executable: `chmod +x ~/.claude/statusline.sh`
3. Add the path to your settings

### Context window usage

Display the current model and context window usage with a visual progress bar.

```bash
#!/bin/bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)

BAR_WIDTH=10
FILLED=$((PCT * BAR_WIDTH / 100))
EMPTY=$((BAR_WIDTH - FILLED))
BAR=""
[ "$FILLED" -gt 0 ] && BAR=$(printf "%${FILLED}s" | tr ' ' '~')
[ "$EMPTY" -gt 0 ] && BAR="${BAR}$(printf "%${EMPTY}s" | tr ' ' ' ')"

echo "[$MODEL] $BAR $PCT%"
```

### Git status with colors

Show git branch with color-coded indicators for staged and modified files.

```bash
#!/bin/bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
DIR=$(echo "$input" | jq -r '.workspace.current_dir')

GREEN='\033[32m'
YELLOW='\033[33m'
RESET='\033[0m'

if git rev-parse --git-dir > /dev/null 2>&1; then
    BRANCH=$(git branch --show-current 2>/dev/null)
    STAGED=$(git diff --cached --numstat 2>/dev/null | wc -l | tr -d ' ')
    MODIFIED=$(git diff --numstat 2>/dev/null | wc -l | tr -d ' ')

    GIT_STATUS=""
    [ "$STAGED" -gt 0 ] && GIT_STATUS="${GREEN}+${STAGED}${RESET}"
    [ "$MODIFIED" -gt 0 ] && GIT_STATUS="${GIT_STATUS}${YELLOW}~${MODIFIED}${RESET}"

    echo -e "[$MODEL] ${DIR##*/} | $BRANCH $GIT_STATUS"
else
    echo "[$MODEL] ${DIR##*/}"
fi
```

### Cost and duration tracking

Track your session's API costs and elapsed time.

```bash
#!/bin/bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
COST=$(echo "$input" | jq -r '.cost.total_cost_usd // 0')
DURATION_MS=$(echo "$input" | jq -r '.cost.total_duration_ms // 0')

COST_FMT=$(printf '$%.2f' "$COST")
DURATION_SEC=$((DURATION_MS / 1000))
MINS=$((DURATION_SEC / 60))
SECS=$((DURATION_SEC % 60))

echo "[$MODEL] $COST_FMT | ${MINS}m ${SECS}s"
```

### Cache expensive operations

Your status line script runs frequently during active sessions. Commands like `git status` can be slow in large repositories. Use a temp file cache that refreshes every 5 seconds:

```bash
#!/bin/bash
input=$(cat)

MODEL=$(echo "$input" | jq -r '.model.display_name')
DIR=$(echo "$input" | jq -r '.workspace.current_dir')

CACHE_FILE="/tmp/statusline-git-cache"
CACHE_MAX_AGE=5  # seconds

cache_is_stale() {
    [ ! -f "$CACHE_FILE" ] || \
    [ $(($(date +%s) - $(stat -f %m "$CACHE_FILE" 2>/dev/null || stat -c %Y "$CACHE_FILE" 2>/dev/null || echo 0))) -gt $CACHE_MAX_AGE ]
}

if cache_is_stale; then
    if git rev-parse --git-dir > /dev/null 2>&1; then
        BRANCH=$(git branch --show-current 2>/dev/null)
        STAGED=$(git diff --cached --numstat 2>/dev/null | wc -l | tr -d ' ')
        MODIFIED=$(git diff --numstat 2>/dev/null | wc -l | tr -d ' ')
        echo "$BRANCH|$STAGED|$MODIFIED" > "$CACHE_FILE"
    else
        echo "||" > "$CACHE_FILE"
    fi
fi

IFS='|' read -r BRANCH STAGED MODIFIED < "$CACHE_FILE"

if [ -n "$BRANCH" ]; then
    echo "[$MODEL] ${DIR##*/} | $BRANCH +$STAGED ~$MODIFIED"
else
    echo "[$MODEL] ${DIR##*/}"
fi
```

## Tips

* **Test with mock input**: `echo '{"model":{"display_name":"Opus"},"context_window":{"used_percentage":25}}' | ./statusline.sh`
* **Keep output short**: the status bar has limited width
* **Cache slow operations**: your script runs frequently during active sessions

Community projects like ccstatusline and starship-claude provide pre-built configurations with themes and additional features.

## Troubleshooting

**Status line not appearing**

* Verify your script is executable: `chmod +x ~/.claude/statusline.sh`
* Check that your script outputs to stdout, not stderr
* Run your script manually to verify it produces output
* If `disableAllHooks` is set to `true` in your settings, the status line is also disabled

**Status line shows `--` or empty values**

* Fields may be `null` before the first API response completes
* Handle null values in your script with fallbacks such as `// 0` in jq
* Restart Claude Code if values remain empty after multiple messages

**Context percentage shows unexpected values**

* Use `used_percentage` for accurate context state rather than cumulative totals
* The `total_input_tokens` and `total_output_tokens` are cumulative across the session
* Context percentage may differ from `/context` output due to when each is calculated

**OSC 8 links not clickable**

* Verify your terminal supports OSC 8 hyperlinks (iTerm2, Kitty, WezTerm)
* Terminal.app does not support clickable links
* SSH and tmux sessions may strip OSC sequences

**Script errors or hangs**

* Scripts that exit with non-zero codes or produce no output cause the status line to go blank
* Slow scripts block the status line from updating
* If a new update triggers while a slow script is running, the in-flight script is cancelled
* Test your script independently with mock input before configuring it
