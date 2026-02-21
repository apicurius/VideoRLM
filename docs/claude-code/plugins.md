# Create plugins

> Create custom plugins to extend Claude Code with skills, agents, hooks, and MCP servers.

Plugins let you extend Claude Code with custom functionality that can be shared across projects and teams. This guide covers creating your own plugins with skills, agents, hooks, and MCP servers.

Looking to install existing plugins? See [Discover and install plugins](/en/discover-plugins). For complete technical specifications, see [Plugins reference](/en/plugins-reference).

## When to use plugins vs standalone configuration

Claude Code supports two ways to add custom skills, agents, and hooks:

| Approach                                                    | Skill names          | Best for                                                                                        |
| :---------------------------------------------------------- | :------------------- | :---------------------------------------------------------------------------------------------- |
| **Standalone** (`.claude/` directory)                       | `/hello`             | Personal workflows, project-specific customizations, quick experiments                          |
| **Plugins** (directories with `.claude-plugin/plugin.json`) | `/plugin-name:hello` | Sharing with teammates, distributing to community, versioned releases, reusable across projects |

**Use standalone configuration when**:

* You're customizing Claude Code for a single project
* The configuration is personal and doesn't need to be shared
* You're experimenting with skills or hooks before packaging them
* You want short skill names like `/hello` or `/review`

**Use plugins when**:

* You want to share functionality with your team or community
* You need the same skills/agents across multiple projects
* You want version control and easy updates for your extensions
* You're distributing through a marketplace
* You're okay with namespaced skills like `/my-plugin:hello`

> **Tip:** Start with standalone configuration in `.claude/` for quick iteration, then convert to a plugin when you're ready to share.

## Quickstart

This quickstart walks you through creating a plugin with a custom skill. You'll create a manifest, add a skill, and test it locally using the `--plugin-dir` flag.

### Prerequisites

* Claude Code installed and authenticated
* Claude Code version 1.0.33 or later (run `claude --version` to check)

### Create your first plugin

**Step 1: Create the plugin directory**

```bash
mkdir my-first-plugin
```

**Step 2: Create the plugin manifest**

Create the `.claude-plugin` directory inside your plugin folder:

```bash
mkdir my-first-plugin/.claude-plugin
```

Then create `my-first-plugin/.claude-plugin/plugin.json`:

```json
{
  "name": "my-first-plugin",
  "description": "A greeting plugin to learn the basics",
  "version": "1.0.0",
  "author": {
    "name": "Your Name"
  }
}
```

| Field         | Purpose                                                                                                |
| :------------ | :----------------------------------------------------------------------------------------------------- |
| `name`        | Unique identifier and skill namespace. Skills are prefixed with this (e.g., `/my-first-plugin:hello`). |
| `description` | Shown in the plugin manager when browsing or installing plugins.                                       |
| `version`     | Track releases using semantic versioning.                                                              |
| `author`      | Optional. Helpful for attribution.                                                                     |

**Step 3: Add a skill**

```bash
mkdir -p my-first-plugin/skills/hello
```

Create `my-first-plugin/skills/hello/SKILL.md`:

```markdown
---
description: Greet the user with a friendly message
disable-model-invocation: true
---

Greet the user warmly and ask how you can help them today.
```

**Step 4: Test your plugin**

```bash
claude --plugin-dir ./my-first-plugin
```

Then try your new skill:

```shell
/my-first-plugin:hello
```

**Step 5: Add skill arguments**

Update your `SKILL.md` file:

```markdown
---
description: Greet the user with a personalized message
---

# Hello Skill

Greet the user named "$ARGUMENTS" warmly and ask how you can help them today. Make the greeting personal and encouraging.
```

Restart Claude Code, then try:

```shell
/my-first-plugin:hello Alex
```

## Plugin structure overview

| Directory         | Location    | Purpose                                                                        |
| :---------------- | :---------- | :----------------------------------------------------------------------------- |
| `.claude-plugin/` | Plugin root | Contains `plugin.json` manifest (optional if components use default locations) |
| `commands/`       | Plugin root | Skills as Markdown files                                                       |
| `agents/`         | Plugin root | Custom agent definitions                                                       |
| `skills/`         | Plugin root | Agent Skills with `SKILL.md` files                                             |
| `hooks/`          | Plugin root | Event handlers in `hooks.json`                                                 |
| `.mcp.json`       | Plugin root | MCP server configurations                                                      |
| `.lsp.json`       | Plugin root | LSP server configurations for code intelligence                                |

> **Warning:** Don't put `commands/`, `agents/`, `skills/`, or `hooks/` inside the `.claude-plugin/` directory. Only `plugin.json` goes inside `.claude-plugin/`. All other directories must be at the plugin root level.

## Develop more complex plugins

### Add Skills to your plugin

Add a `skills/` directory at your plugin root with Skill folders containing `SKILL.md` files:

```
my-plugin/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    └── code-review/
        └── SKILL.md
```

Each `SKILL.md` needs frontmatter with `name` and `description` fields:

```yaml
---
name: code-review
description: Reviews code for best practices and potential issues. Use when reviewing code, checking PRs, or analyzing code quality.
---

When reviewing code, check for:
1. Code organization and structure
2. Error handling
3. Security concerns
4. Test coverage
```

### Add LSP servers to your plugin

LSP plugins give Claude real-time code intelligence. Add an `.lsp.json` file to your plugin:

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

Users installing your plugin must have the language server binary installed on their machine.

### Test your plugins locally

Use the `--plugin-dir` flag to test plugins during development:

```bash
claude --plugin-dir ./my-plugin
```

You can load multiple plugins at once:

```bash
claude --plugin-dir ./plugin-one --plugin-dir ./plugin-two
```

### Share your plugins

When your plugin is ready to share:

1. **Add documentation**: Include a `README.md` with installation and usage instructions
2. **Version your plugin**: Use semantic versioning in your `plugin.json`
3. **Create or use a marketplace**: Distribute through plugin marketplaces for installation
4. **Test with others**: Have team members test the plugin before wider distribution

## Convert existing configurations to plugins

### Migration steps

**Step 1: Create the plugin structure**

```bash
mkdir -p my-plugin/.claude-plugin
```

Create the manifest file at `my-plugin/.claude-plugin/plugin.json`:

```json
{
  "name": "my-plugin",
  "description": "Migrated from standalone configuration",
  "version": "1.0.0"
}
```

**Step 2: Copy your existing files**

```bash
# Copy commands
cp -r .claude/commands my-plugin/

# Copy agents (if any)
cp -r .claude/agents my-plugin/

# Copy skills (if any)
cp -r .claude/skills my-plugin/
```

**Step 3: Migrate hooks**

Create `my-plugin/hooks/hooks.json` with your hooks configuration:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [{ "type": "command", "command": "jq -r '.tool_input.file_path' | xargs npm run lint:fix" }]
      }
    ]
  }
}
```

**Step 4: Test your migrated plugin**

```bash
claude --plugin-dir ./my-plugin
```

### What changes when migrating

| Standalone (`.claude/`)       | Plugin                           |
| :---------------------------- | :------------------------------- |
| Only available in one project | Can be shared via marketplaces   |
| Files in `.claude/commands/`  | Files in `plugin-name/commands/` |
| Hooks in `settings.json`      | Hooks in `hooks/hooks.json`      |
| Must manually copy to share   | Install with `/plugin install`   |

## Next steps

### For plugin users

* [Discover and install plugins](/en/discover-plugins): browse marketplaces and install plugins
* [Configure team marketplaces](/en/discover-plugins#configure-team-marketplaces): set up repository-level plugins for your team

### For plugin developers

* [Create and distribute a marketplace](/en/plugin-marketplaces): package and share your plugins
* [Plugins reference](/en/plugins-reference): complete technical specifications
* [Skills](/en/skills): skill development details
* [Subagents](/en/sub-agents): agent configuration and capabilities
* [Hooks](/en/hooks): event handling and automation
* [MCP](/en/mcp): external tool integration
