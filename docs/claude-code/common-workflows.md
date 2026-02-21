# Common workflows

> Step-by-step guides for exploring codebases, fixing bugs, refactoring, testing, and other everyday tasks with Claude Code.

This page covers practical workflows for everyday development: exploring unfamiliar code, debugging, refactoring, writing tests, creating PRs, and managing sessions. Each section includes example prompts you can adapt to your own projects.

## Understand new codebases

### Get a quick codebase overview

1. Navigate to the project root directory
2. Start Claude Code: `claude`
3. Ask for a high-level overview: `give me an overview of this codebase`
4. Dive deeper into specific components:
   - `explain the main architecture patterns used here`
   - `what are the key data models?`
   - `how is authentication handled?`

**Tips:**
* Start with broad questions, then narrow down to specific areas
* Ask about coding conventions and patterns used in the project
* Request a glossary of project-specific terms

### Find relevant code

1. Ask Claude to find relevant files: `find the files that handle user authentication`
2. Get context on how components interact: `how do these authentication files work together?`
3. Understand the execution flow: `trace the login process from front-end to database`

**Tips:**
* Be specific about what you're looking for
* Use domain language from the project
* Install a code intelligence plugin for your language to give Claude precise navigation

---

## Fix bugs efficiently

1. Share the error with Claude: `I'm seeing an error when I run npm test`
2. Ask for fix recommendations: `suggest a few ways to fix the @ts-ignore in user.ts`
3. Apply the fix: `update user.ts to add the null check you suggested`

**Tips:**
* Tell Claude the command to reproduce the issue and get a stack trace
* Mention any steps to reproduce the error
* Let Claude know if the error is intermittent or consistent

---

## Refactor code

1. Identify legacy code: `find deprecated API usage in our codebase`
2. Get recommendations: `suggest how to refactor utils.js to use modern JavaScript features`
3. Apply changes safely: `refactor utils.js to use ES2024 features while maintaining the same behavior`
4. Verify: `run tests for the refactored code`

**Tips:**
* Ask Claude to explain the benefits of the modern approach
* Request that changes maintain backward compatibility when needed
* Do refactoring in small, testable increments

---

## Use specialized subagents

1. View available subagents: `/agents`
2. Use subagents automatically:
   - `review my recent code changes for security issues`
   - `run all tests and fix any failures`
3. Explicitly request specific subagents:
   - `use the code-reviewer subagent to check the auth module`
4. Create custom subagents: `/agents` then select "Create New subagent"

**Tips:**
* Create project-specific subagents in `.claude/agents/` for team sharing
* Use descriptive `description` fields to enable automatic delegation
* Limit tool access to what each subagent actually needs

---

## Use Plan Mode for safe code analysis

Plan Mode instructs Claude to create a plan by analyzing the codebase with read-only operations.

### When to use Plan Mode

* **Multi-step implementation**: When your feature requires making edits to many files
* **Code exploration**: When you want to research the codebase thoroughly before changing anything
* **Interactive development**: When you want to iterate on the direction with Claude

### How to use Plan Mode

**Turn on Plan Mode during a session**: Use **Shift+Tab** to cycle through permission modes.

**Start a new session in Plan Mode**:

```bash
claude --permission-mode plan
```

**Run "headless" queries in Plan Mode**:

```bash
claude --permission-mode plan -p "Analyze the authentication system and suggest improvements"
```

### Configure Plan Mode as default

```json
// .claude/settings.json
{
  "permissions": {
    "defaultMode": "plan"
  }
}
```

---

## Work with tests

1. Identify untested code: `find functions in NotificationsService.swift that are not covered by tests`
2. Generate test scaffolding: `add tests for the notification service`
3. Add meaningful test cases: `add test cases for edge conditions in the notification service`
4. Run and verify: `run the new tests and fix any failures`

---

## Create pull requests

You can create pull requests by asking Claude directly ("create a pr for my changes") or by using the `/commit-push-pr` skill.

```
> /commit-push-pr
```

For more control, guide Claude step-by-step:

1. Summarize your changes: `summarize the changes I've made to the authentication module`
2. Generate a PR: `create a pr`
3. Review and refine: `enhance the PR description with more context about the security improvements`

When you create a PR using `gh pr create`, the session is automatically linked to that PR. You can resume it later with `claude --from-pr <number>`.

---

## Handle documentation

1. Identify undocumented code: `find functions without proper JSDoc comments in the auth module`
2. Generate documentation: `add JSDoc comments to the undocumented functions in auth.js`
3. Review and enhance: `improve the generated documentation with more context and examples`
4. Verify: `check if the documentation follows our project standards`

---

## Work with images

1. Add an image to the conversation:
   - Drag and drop an image into the Claude Code window
   - Copy an image and paste it with ctrl+v
   - Provide an image path
2. Ask Claude to analyze: `What does this image show?`
3. Use images for context: `Here's a screenshot of the error. What's causing it?`
4. Get code suggestions: `Generate CSS to match this design mockup`

---

## Reference files and directories

* Reference a single file: `Explain the logic in @src/utils/auth.js`
* Reference a directory: `What's the structure of @src/components?`
* Reference MCP resources: `Show me the data from @github:repos/owner/repo/issues`

**Tips:**
* File paths can be relative or absolute
* Directory references show file listings, not contents
* You can reference multiple files in a single message

---

## Use extended thinking (thinking mode)

Extended thinking is enabled by default, giving Claude space to reason through complex problems step-by-step.

### Configure thinking mode

| Scope                  | How to configure                                                           |
| ---------------------- | -------------------------------------------------------------------------- |
| **Effort level**       | Adjust in `/model` or set `CLAUDE_CODE_EFFORT_LEVEL`                       |
| **Toggle shortcut**    | Press `Option+T` (macOS) or `Alt+T` (Windows/Linux)                        |
| **Global default**     | Use `/config` to toggle thinking mode                                      |
| **Limit token budget** | Set `MAX_THINKING_TOKENS` environment variable                             |

---

## Resume previous conversations

* `claude --continue` continues the most recent conversation in the current directory
* `claude --resume` opens a conversation picker or resumes by name
* `claude --from-pr 123` resumes sessions linked to a specific pull request

### Name your sessions

Use `/rename` during a session to give it a memorable name:

```
> /rename auth-refactor
```

Resume by name later:

```bash
claude --resume auth-refactor
```

### Session picker keyboard shortcuts

| Shortcut  | Action                                            |
| :-------- | :------------------------------------------------ |
| `Up/Down` | Navigate between sessions                         |
| `Right/Left` | Expand or collapse grouped sessions            |
| `Enter`   | Select and resume the highlighted session         |
| `P`       | Preview the session content                       |
| `R`       | Rename the highlighted session                    |
| `/`       | Search to filter sessions                         |
| `A`       | Toggle between current directory and all projects |
| `B`       | Filter to sessions from your current git branch   |
| `Esc`     | Exit the picker or search mode                    |

---

## Run parallel Claude Code sessions with Git worktrees

1. Create a new worktree:
   ```bash
   git worktree add ../project-feature-a -b feature-a
   ```
2. Run Claude Code in each worktree:
   ```bash
   cd ../project-feature-a
   claude
   ```
3. Manage your worktrees:
   ```bash
   git worktree list
   git worktree remove ../project-feature-a
   ```

---

## Use Claude as a unix-style utility

### Add Claude to your verification process

```json
// package.json
{
    "scripts": {
        "lint:claude": "claude -p 'you are a linter. please look at the changes vs. main and report any issues related to typos.'"
    }
}
```

### Pipe in, pipe out

```bash
cat build-error.txt | claude -p 'concisely explain the root cause of this build error' > output.txt
```

### Control output format

* `--output-format text`: plain text response (default)
* `--output-format json`: JSON array of messages with metadata
* `--output-format stream-json`: streaming JSON objects in real-time

---

## Ask Claude about its capabilities

Claude has built-in access to its documentation and can answer questions about its own features:

```
> can Claude Code create pull requests?
> how does Claude Code handle permissions?
> what skills are available?
> how do I use MCP with Claude Code?
> how do I configure Claude Code for Amazon Bedrock?
```
