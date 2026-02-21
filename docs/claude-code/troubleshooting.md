# Troubleshooting

> Discover solutions to common issues with Claude Code installation and usage.

## Common installation issues

### Windows installation issues: errors in WSL

You might encounter the following issues in WSL:

**OS/platform detection issues**: If you receive an error during installation, WSL may be using Windows `npm`. Try:

* Run `npm config set os linux` before installation
* Install with `npm install -g @anthropic-ai/claude-code --force --no-os-check` (Do NOT use `sudo`)

**Node not found errors**: If you see `exec: node: not found` when running `claude`, your WSL environment may be using a Windows installation of Node.js. You can confirm this with `which npm` and `which node`, which should point to Linux paths starting with `/usr/` rather than `/mnt/c/`. To fix this, try installing Node via your Linux distribution's package manager or via `nvm`.

**nvm version conflicts**: If you have nvm installed in both WSL and Windows, you may experience version conflicts when switching Node versions in WSL. This happens because WSL imports the Windows PATH by default, causing Windows nvm/npm to take priority over the WSL installation.

**Primary solution: Ensure nvm is properly loaded in your shell**

Add the following to your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# Load nvm if it exists
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
```

**Alternative: Adjust PATH order**

If nvm is properly loaded but Windows paths still take priority:

```bash
export PATH="$HOME/.nvm/versions/node/$(node -v)/bin:$PATH"
```

> Avoid disabling Windows PATH importing (`appendWindowsPath = false`) as this breaks the ability to call Windows executables from WSL.

### WSL2 sandbox setup

Sandboxing is supported on WSL2 but requires installing additional packages. If you see an error like "Sandbox requires socat and bubblewrap":

**Ubuntu/Debian:**

```bash
sudo apt-get install bubblewrap socat
```

**Fedora:**

```bash
sudo dnf install bubblewrap socat
```

WSL1 does not support sandboxing.

### Linux and Mac installation issues: permission or command not found errors

#### Recommended solution: Native Claude Code installation

```bash
# Install stable version (default)
curl -fsSL https://claude.ai/install.sh | bash

# Install latest version
curl -fsSL https://claude.ai/install.sh | bash -s latest

# Install specific version number
curl -fsSL https://claude.ai/install.sh | bash -s 1.0.58
```

**Windows PowerShell:**

```powershell
# Install stable version (default)
irm https://claude.ai/install.ps1 | iex

# Install latest version
& ([scriptblock]::Create((irm https://claude.ai/install.ps1))) latest

# Install specific version number
& ([scriptblock]::Create((irm https://claude.ai/install.ps1))) 1.0.58
```

This installs the appropriate build of Claude Code for your operating system and architecture and adds a symlink to the installation at `~/.local/bin/claude` (or `%USERPROFILE%\.local\bin\claude.exe` on Windows).

> Make sure that you have the installation directory in your system PATH.

### Windows: "Claude Code on Windows requires git-bash"

Claude Code on native Windows requires Git for Windows which includes Git Bash. If Git is installed but not detected:

1. Set the path explicitly in PowerShell:
   ```powershell
   $env:CLAUDE_CODE_GIT_BASH_PATH="C:\Program Files\Git\bin\bash.exe"
   ```

2. Or add it to your system environment variables permanently through System Properties -> Environment Variables.

### Windows: "installMethod is native, but claude command not found"

If you see this error after installation, the `claude` command isn't in your PATH. Add it manually:

1. Press `Win + R`, type `sysdm.cpl`, and press Enter. Click **Advanced** -> **Environment Variables**.
2. Under "User variables", select **Path** and click **Edit**. Click **New** and add:
   ```
   %USERPROFILE%\.local\bin
   ```
3. Close and reopen PowerShell or CMD for changes to take effect.

Verify installation:

```bash
claude doctor # Check installation health
```

## Permissions and authentication

### Repeated permission prompts

If you find yourself repeatedly approving the same commands, you can allow specific tools to run without approval using the `/permissions` command.

### Authentication issues

If you're experiencing authentication problems:

1. Run `/logout` to sign out completely
2. Close Claude Code
3. Restart with `claude` and complete the authentication process again

If the browser doesn't open automatically during login, press `c` to copy the OAuth URL to your clipboard.

If problems persist, try:

```bash
rm -rf ~/.config/claude-code/auth.json
claude
```

## Configuration file locations

| File                          | Purpose                                                  |
| :---------------------------- | :------------------------------------------------------- |
| `~/.claude/settings.json`     | User settings (permissions, hooks, model overrides)      |
| `.claude/settings.json`       | Project settings (checked into source control)           |
| `.claude/settings.local.json` | Local project settings (not committed)                   |
| `~/.claude.json`              | Global state (theme, OAuth, MCP servers)                 |
| `.mcp.json`                   | Project MCP servers (checked into source control)        |
| `managed-settings.json`       | Managed settings                                         |
| `managed-mcp.json`            | Managed MCP servers                                      |

On Windows, `~` refers to your user home directory, such as `C:\Users\YourName`.

**Managed file locations:**

* macOS: `/Library/Application Support/ClaudeCode/`
* Linux/WSL: `/etc/claude-code/`
* Windows: `C:\Program Files\ClaudeCode\`

### Resetting configuration

```bash
# Reset all user settings and state
rm ~/.claude.json
rm -rf ~/.claude/

# Reset project-specific settings
rm -rf .claude/
rm .mcp.json
```

> This will remove all your settings, MCP server configurations, and session history.

## Performance and stability

### High CPU or memory usage

If you're experiencing performance issues:

1. Use `/compact` regularly to reduce context size
2. Close and restart Claude Code between major tasks
3. Consider adding large build directories to your `.gitignore` file

### Command hangs or freezes

If Claude Code seems unresponsive:

1. Press Ctrl+C to attempt to cancel the current operation
2. If unresponsive, you may need to close the terminal and restart

### Search and discovery issues

If Search tool, `@file` mentions, custom agents, and custom skills aren't working, install system `ripgrep`:

```bash
# macOS (Homebrew)
brew install ripgrep

# Windows (winget)
winget install BurntSushi.ripgrep.MSVC

# Ubuntu/Debian
sudo apt install ripgrep

# Alpine Linux
apk add ripgrep

# Arch Linux
pacman -S ripgrep
```

Then set `USE_BUILTIN_RIPGREP=0` in your environment.

### Slow or incomplete search results on WSL

Disk read performance penalties when working across file systems on WSL may result in fewer-than-expected matches.

> `/doctor` will show Search as OK in this case.

**Solutions:**

1. **Submit more specific searches**: Reduce the number of files searched by specifying directories or file types.
2. **Move project to Linux filesystem**: Ensure your project is located on the Linux filesystem (`/home/`) rather than the Windows filesystem (`/mnt/c/`).
3. **Use native Windows instead**: Consider running Claude Code natively on Windows instead of through WSL.

## IDE integration issues

### JetBrains IDE not detected on WSL2

If you're using Claude Code on WSL2 with JetBrains IDEs and getting "No available IDEs detected" errors, this is likely due to WSL2's networking configuration or Windows Firewall blocking the connection.

**Option 1: Configure Windows Firewall** (recommended)

1. Find your WSL2 IP address:
   ```bash
   wsl hostname -I
   ```

2. Open PowerShell as Administrator and create a firewall rule:
   ```powershell
   New-NetFirewallRule -DisplayName "Allow WSL2 Internal Traffic" -Direction Inbound -Protocol TCP -Action Allow -RemoteAddress 172.21.0.0/16 -LocalAddress 172.21.0.0/16
   ```

3. Restart both your IDE and Claude Code

**Option 2: Switch to mirrored networking**

Add to `.wslconfig` in your Windows user directory:

```ini
[wsl2]
networkingMode=mirrored
```

Then restart WSL with `wsl --shutdown` from PowerShell.

### Escape key not working in JetBrains terminals

If the `Esc` key doesn't interrupt the agent as expected:

1. Go to Settings -> Tools -> Terminal
2. Either:
   * Uncheck "Move focus to the editor with Escape", or
   * Click "Configure terminal keybindings" and delete the "Switch focus to Editor" shortcut
3. Apply the changes

## Markdown formatting issues

### Missing language tags in code blocks

If you notice code blocks without language tags, ask Claude to add them or use post-processing hooks to detect and add missing language tags.

### Inconsistent spacing and formatting

* Request formatting corrections
* Use formatting tools like `prettier`
* Include formatting requirements in your prompts or CLAUDE.md files

### Best practices for markdown generation

* Be explicit in requests: Ask for "properly formatted markdown with language-tagged code blocks"
* Use project conventions: Document your preferred markdown style in CLAUDE.md
* Set up validation hooks for common formatting issues

## Getting more help

If you're experiencing issues not covered here:

1. Use the `/bug` command within Claude Code to report problems directly to Anthropic
2. Check the GitHub repository for known issues
3. Run `/doctor` to diagnose issues. It checks:
   * Installation type, version, and search functionality
   * Auto-update status and available versions
   * Invalid settings files (malformed JSON, incorrect types)
   * MCP server configuration errors
   * Keybinding configuration problems
   * Context usage warnings (large CLAUDE.md files, high MCP token usage, unreachable permission rules)
   * Plugin and agent loading errors
4. Ask Claude directly about its capabilities and features
