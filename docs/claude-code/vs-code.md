# Use Claude Code in VS Code

> Install and configure the Claude Code extension for VS Code. Get AI coding assistance with inline diffs, @-mentions, plan review, and keyboard shortcuts.

The VS Code extension provides a native graphical interface for Claude Code, integrated directly into your IDE. This is the recommended way to use Claude Code in VS Code.

With the extension, you can review and edit Claude's plans before accepting them, auto-accept edits as they're made, @-mention files with specific line ranges from your selection, access conversation history, and open multiple conversations in separate tabs or windows.

## Prerequisites

* VS Code 1.98.0 or higher
* An Anthropic account (you'll sign in when you first open the extension). If you're using a third-party provider like Amazon Bedrock or Google Vertex AI, see Use third-party providers instead.

> The extension includes the CLI (command-line interface), which you can access from VS Code's integrated terminal for advanced features.

## Install the extension

In VS Code, press `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X` (Windows/Linux) to open the Extensions view, search for "Claude Code", and click **Install**.

> If the extension doesn't appear after installation, restart VS Code or run "Developer: Reload Window" from the Command Palette.

## Get started

### Step 1: Open the Claude Code panel

The quickest way to open Claude is to click the Spark icon in the **Editor Toolbar** (top-right corner of the editor). The icon only appears when you have a file open.

Other ways to open Claude Code:

* **Command Palette**: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux), type "Claude Code", and select an option like "Open in New Tab"
* **Status Bar**: Click **Claude Code** in the bottom-right corner of the window. This works even when no file is open.

When you first open the panel, a **Learn Claude Code** checklist appears. Work through each item by clicking **Show me**, or dismiss it with the X.

### Step 2: Send a prompt

Ask Claude to help with your code or files, whether that's explaining how something works, debugging an issue, or making changes.

> Claude automatically sees your selected text. Press `Option+K` (Mac) / `Alt+K` (Windows/Linux) to also insert an @-mention reference (like `@file.ts#5-10`) into your prompt.

### Step 3: Review changes

When Claude wants to edit a file, it shows a side-by-side comparison of the original and proposed changes, then asks for permission. You can accept, reject, or tell Claude what to do instead.

For more ideas on what you can do with Claude Code, see Common workflows.

## Use the prompt box

The prompt box supports several features:

* **Permission modes**: Click the mode indicator at the bottom of the prompt box to switch modes. In normal mode, Claude asks permission before each action. In Plan mode, Claude describes what it will do and waits for approval. In auto-accept mode, Claude makes edits without asking.
* **Command menu**: Click `/` or type `/` to open the command menu.
* **Context indicator**: The prompt box shows how much of Claude's context window you're using.
* **Extended thinking**: Lets Claude spend more time reasoning through complex problems. Toggle it on via the command menu (`/`).
* **Multi-line input**: Press `Shift+Enter` to add a new line without sending.

### Reference files and folders

Use @-mentions to give Claude context about specific files or folders. When you type `@` followed by a file or folder name, Claude reads that content and can answer questions about it or make changes to it. Claude Code supports fuzzy matching:

```
> Explain the logic in @auth (fuzzy matches auth.js, AuthService.ts, etc.)
> What's in @src/components/ (include a trailing slash for folders)
```

For large PDFs, you can ask Claude to read specific pages instead of the whole file.

When you select text in the editor, Claude can see your highlighted code automatically. Press `Option+K` (Mac) / `Alt+K` (Windows/Linux) to insert an @-mention with the file path and line numbers.

You can also hold `Shift` while dragging files into the prompt box to add them as attachments.

### Resume past conversations

Click the dropdown at the top of the Claude Code panel to access your conversation history. You can search by keyword or browse by time.

### Resume remote sessions from Claude.ai

If you use Claude Code on the web, you can resume those remote sessions directly in VS Code. This requires signing in with **Claude.ai Subscription**, not Anthropic Console.

1. Click the **Past Conversations** dropdown
2. Click **Remote** to see sessions from claude.ai
3. Click any session to download it and continue locally

> Only web sessions started with a GitHub repository appear in the Remote tab.

## Customize your workflow

### Choose where Claude lives

You can drag the Claude panel to reposition it anywhere in VS Code:

* **Secondary sidebar**: The right side of the window
* **Primary sidebar**: The left sidebar with icons for Explorer, Search, etc.
* **Editor area**: Opens Claude as a tab alongside your files

### Run multiple conversations

Use **Open in New Tab** or **Open in New Window** from the Command Palette to start additional conversations.

### Switch to terminal mode

If you prefer the CLI-style interface, open the Use Terminal setting and check the box. You can also open VS Code settings, go to Extensions -> Claude Code, and check **Use Terminal**.

## Manage plugins

Type `/plugins` in the prompt box to open the **Manage plugins** interface.

### Install plugins

The plugin dialog shows two tabs: **Plugins** and **Marketplaces**.

When you install a plugin, choose the installation scope:

* **Install for you**: Available in all your projects (user scope)
* **Install for this project**: Shared with project collaborators (project scope)
* **Install locally**: Only for you, only in this repository (local scope)

### Manage marketplaces

Switch to the **Marketplaces** tab to add or remove plugin sources.

## Automate browser tasks with Chrome

Connect Claude to your Chrome browser to test web apps, debug with console logs, and automate browser workflows. This requires the Claude in Chrome extension version 1.0.36 or higher.

Type `@browser` in the prompt box followed by what you want Claude to do:

```text
@browser go to localhost:3000 and check the console for errors
```

## VS Code commands and shortcuts

Open the Command Palette (`Cmd+Shift+P` on Mac or `Ctrl+Shift+P` on Windows/Linux) and type "Claude Code" to see all available commands.

| Command                    | Shortcut                                                 | Description                                                                          |
| -------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Focus Input                | `Cmd+Esc` (Mac) / `Ctrl+Esc` (Windows/Linux)             | Toggle focus between editor and Claude                                               |
| Open in Side Bar           | -                                                        | Open Claude in the left sidebar                                                      |
| Open in Terminal           | -                                                        | Open Claude in terminal mode                                                         |
| Open in New Tab            | `Cmd+Shift+Esc` (Mac) / `Ctrl+Shift+Esc` (Windows/Linux) | Open a new conversation as an editor tab                                             |
| Open in New Window         | -                                                        | Open a new conversation in a separate window                                         |
| New Conversation           | `Cmd+N` (Mac) / `Ctrl+N` (Windows/Linux)                 | Start a new conversation (requires Claude to be focused)                             |
| Insert @-Mention Reference | `Option+K` (Mac) / `Alt+K` (Windows/Linux)               | Insert a reference to the current file and selection (requires editor to be focused) |
| Show Logs                  | -                                                        | View extension debug logs                                                            |
| Logout                     | -                                                        | Sign out of your Anthropic account                                                   |

## Configure settings

The extension has two types of settings:

* **Extension settings** in VS Code: Control the extension's behavior within VS Code. Open with `Cmd+,` (Mac) or `Ctrl+,` (Windows/Linux), then go to Extensions -> Claude Code.
* **Claude Code settings** in `~/.claude/settings.json`: Shared between the extension and CLI.

> Add `"$schema": "https://json.schemastore.org/claude-code-settings.json"` to your `settings.json` to get autocomplete and inline validation.

### Extension settings

| Setting                           | Default   | Description                                                                                           |
| --------------------------------- | --------- | ----------------------------------------------------------------------------------------------------- |
| `selectedModel`                   | `default` | Model for new conversations. Change per-session with `/model`.                                        |
| `useTerminal`                     | `false`   | Launch Claude in terminal mode instead of graphical panel                                             |
| `initialPermissionMode`           | `default` | Controls approval prompts: `default`, `plan`, `acceptEdits`, or `bypassPermissions`                    |
| `preferredLocation`               | `panel`   | Where Claude opens: `sidebar` or `panel`                                                               |
| `autosave`                        | `true`    | Auto-save files before Claude reads or writes them                                                    |
| `useCtrlEnterToSend`              | `false`   | Use Ctrl/Cmd+Enter instead of Enter to send prompts                                                   |
| `enableNewConversationShortcut`   | `true`    | Enable Cmd/Ctrl+N to start a new conversation                                                         |
| `hideOnboarding`                  | `false`   | Hide the onboarding checklist                                                                         |
| `respectGitIgnore`                | `true`    | Exclude .gitignore patterns from file searches                                                        |
| `environmentVariables`            | `[]`      | Set environment variables for the Claude process                                                      |
| `disableLoginPrompt`              | `false`   | Skip authentication prompts (for third-party provider setups)                                         |
| `allowDangerouslySkipPermissions` | `false`   | Bypass all permission prompts. **Use with extreme caution.**                                          |
| `claudeProcessWrapper`            | -         | Executable path used to launch the Claude process                                                     |

## VS Code extension vs. Claude Code CLI

| Feature             | CLI                                           | VS Code Extension                        |
| ------------------- | --------------------------------------------- | ---------------------------------------- |
| Commands and skills | All                                           | Subset (type `/` to see available)       |
| MCP server config   | Yes                                           | No (configure via CLI, use in extension) |
| Checkpoints         | Yes                                           | Yes                                      |
| `!` bash shortcut   | Yes                                           | No                                       |
| Tab completion      | Yes                                           | No                                       |

### Rewind with checkpoints

The VS Code extension supports checkpoints. Hover over any message to reveal the rewind button, then choose from three options:

* **Fork conversation from here**: start a new conversation branch from this message
* **Rewind code to here**: revert file changes back to this point
* **Fork conversation and rewind code**: start a new branch and revert file changes

### Run CLI in VS Code

Open the integrated terminal (`` Ctrl+` `` on Windows/Linux or `` Cmd+` `` on Mac) and run `claude`.

If using an external terminal, run `/ide` inside Claude Code to connect it to VS Code.

### Switch between extension and CLI

The extension and CLI share the same conversation history. To continue an extension conversation in the CLI, run `claude --resume` in the terminal.

### Include terminal output in prompts

Reference terminal output using `@terminal:name` where `name` is the terminal's title.

### Connect to external tools with MCP

MCP servers give Claude access to external tools, databases, and APIs. Configure them via CLI:

```bash
claude mcp add --transport http github https://api.githubcopilot.com/mcp/
```

## Work with git

Claude Code integrates with git to help with version control workflows directly in VS Code.

### Create commits and pull requests

```
> commit my changes with a descriptive message
> create a pr for this feature
> summarize the changes I've made to the auth module
```

### Use git worktrees for parallel tasks

```bash
# Create a worktree for a new feature
git worktree add ../project-feature-a -b feature-a

# Run Claude Code in each worktree
cd ../project-feature-a && claude
```

## Use third-party providers

1. Open the Disable Login Prompt setting and check the box.
2. Follow the setup guide for your provider:
   * Claude Code on Amazon Bedrock
   * Claude Code on Google Vertex AI
   * Claude Code on Microsoft Foundry

## Security and privacy

Your code stays private. Claude Code processes your code to provide assistance but does not use it to train models.

With auto-edit permissions enabled, Claude Code can modify VS Code configuration files that VS Code may execute automatically. To reduce risk:

* Enable VS Code Restricted Mode for untrusted workspaces
* Use manual approval mode instead of auto-accept for edits
* Review changes carefully before accepting them

## Fix common issues

### Extension won't install

* Ensure you have a compatible version of VS Code (1.98.0 or later)
* Check that VS Code has permission to install extensions

### Spark icon not visible

1. **Open a file**: The icon requires a file to be open
2. **Check VS Code version**: Requires 1.98.0 or higher
3. **Restart VS Code**: Run "Developer: Reload Window"
4. **Disable conflicting extensions**: Temporarily disable other AI extensions
5. **Check workspace trust**: The extension doesn't work in Restricted Mode

Alternatively, click "Claude Code" in the **Status Bar** or use the **Command Palette**.

### Claude Code never responds

1. Check your internet connection
2. Start a new conversation
3. Try the CLI: Run `claude` from the terminal for more detailed error messages

## Uninstall the extension

1. Open the Extensions view (`Cmd+Shift+X` on Mac or `Ctrl+Shift+X` on Windows/Linux)
2. Search for "Claude Code"
3. Click **Uninstall**

To also remove extension data:

```bash
rm -rf ~/.vscode/globalStorage/anthropic.claude-code
```

## Next steps

* Explore common workflows
* Set up MCP servers to extend Claude's capabilities
* Configure Claude Code settings
