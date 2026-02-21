# Create and distribute a plugin marketplace

> Build and host plugin marketplaces to distribute Claude Code extensions across teams and communities.

A **plugin marketplace** is a catalog that lets you distribute plugins to others. Marketplaces provide centralized discovery, version tracking, automatic updates, and support for multiple source types (git repositories, local paths, and more). This guide shows you how to create your own marketplace to share plugins with your team or community.

Looking to install plugins from an existing marketplace? See [Discover and install prebuilt plugins](/en/discover-plugins).

## Overview

Creating and distributing a marketplace involves:

1. **Creating plugins**: build one or more plugins with commands, agents, hooks, MCP servers, or LSP servers. This guide assumes you already have plugins to distribute; see [Create plugins](/en/plugins) for details on how to create them.
2. **Creating a marketplace file**: define a `marketplace.json` that lists your plugins and where to find them (see [Create the marketplace file](#create-the-marketplace-file)).
3. **Host the marketplace**: push to GitHub, GitLab, or another git host (see [Host and distribute marketplaces](#host-and-distribute-marketplaces)).
4. **Share with users**: users add your marketplace with `/plugin marketplace add` and install individual plugins (see [Discover and install plugins](/en/discover-plugins)).

Once your marketplace is live, you can update it by pushing changes to your repository. Users refresh their local copy with `/plugin marketplace update`.

## Walkthrough: create a local marketplace

This example creates a marketplace with one plugin: a `/review` skill for code reviews. You'll create the directory structure, add a skill, create the plugin manifest and marketplace catalog, then install and test it.

### Step 1: Create the directory structure

```bash
mkdir -p my-marketplace/.claude-plugin
mkdir -p my-marketplace/plugins/review-plugin/.claude-plugin
mkdir -p my-marketplace/plugins/review-plugin/skills/review
```

### Step 2: Create the skill

Create a `SKILL.md` file that defines what the `/review` skill does.

```markdown
---
description: Review code for bugs, security, and performance
disable-model-invocation: true
---

Review the code I've selected or the recent changes for:
- Potential bugs or edge cases
- Security concerns
- Performance issues
- Readability improvements

Be concise and actionable.
```

### Step 3: Create the plugin manifest

Create a `plugin.json` file that describes the plugin. The manifest goes in the `.claude-plugin/` directory.

```json
{
  "name": "review-plugin",
  "description": "Adds a /review skill for quick code reviews",
  "version": "1.0.0"
}
```

### Step 4: Create the marketplace file

Create the marketplace catalog that lists your plugin.

```json
{
  "name": "my-plugins",
  "owner": {
    "name": "Your Name"
  },
  "plugins": [
    {
      "name": "review-plugin",
      "source": "./plugins/review-plugin",
      "description": "Adds a /review skill for quick code reviews"
    }
  ]
}
```

### Step 5: Add and install

Add the marketplace and install the plugin.

```shell
/plugin marketplace add ./my-marketplace
/plugin install review-plugin@my-plugins
```

### Step 6: Try it out

Select some code in your editor and run your new command.

```shell
/review
```

To learn more about what plugins can do, including hooks, agents, MCP servers, and LSP servers, see [Plugins](/en/plugins).

> **Note:** **How plugins are installed**: When users install a plugin, Claude Code copies the plugin directory to a cache location. This means plugins can't reference files outside their directory using paths like `../shared-utils`, because those files won't be copied.
>
> If you need to share files across plugins, use symlinks (which are followed during copying). See [Plugin caching and file resolution](/en/plugins-reference#plugin-caching-and-file-resolution) for details.

## Create the marketplace file

Create `.claude-plugin/marketplace.json` in your repository root. This file defines your marketplace's name, owner information, and a list of plugins with their sources.

Each plugin entry needs at minimum a `name` and `source` (where to fetch it from). See the [full schema](#marketplace-schema) below for all available fields.

```json
{
  "name": "company-tools",
  "owner": {
    "name": "DevTools Team",
    "email": "devtools@example.com"
  },
  "plugins": [
    {
      "name": "code-formatter",
      "source": "./plugins/formatter",
      "description": "Automatic code formatting on save",
      "version": "2.1.0",
      "author": {
        "name": "DevTools Team"
      }
    },
    {
      "name": "deployment-tools",
      "source": {
        "source": "github",
        "repo": "company/deploy-plugin"
      },
      "description": "Deployment automation tools"
    }
  ]
}
```

## Marketplace schema

### Required fields

| Field     | Type   | Description                                                                                                                                                            | Example        |
| :-------- | :----- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------- |
| `name`    | string | Marketplace identifier (kebab-case, no spaces). This is public-facing: users see it when installing plugins (for example, `/plugin install my-tool@your-marketplace`). | `"acme-tools"` |
| `owner`   | object | Marketplace maintainer information                                                                                                                                     |                |
| `plugins` | array  | List of available plugins                                                                                                                                              | See below      |

> **Note:** **Reserved names**: The following marketplace names are reserved for official Anthropic use and cannot be used by third-party marketplaces: `claude-code-marketplace`, `claude-code-plugins`, `claude-plugins-official`, `anthropic-marketplace`, `anthropic-plugins`, `agent-skills`, `life-sciences`. Names that impersonate official marketplaces (like `official-claude-plugins` or `anthropic-tools-v2`) are also blocked.

### Owner fields

| Field   | Type   | Required | Description                      |
| :------ | :----- | :------- | :------------------------------- |
| `name`  | string | Yes      | Name of the maintainer or team   |
| `email` | string | No       | Contact email for the maintainer |

### Optional metadata

| Field                  | Type   | Description                                                                                                                                                               |
| :--------------------- | :----- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `metadata.description` | string | Brief marketplace description                                                                                                                                             |
| `metadata.version`     | string | Marketplace version                                                                                                                                                       |
| `metadata.pluginRoot`  | string | Base directory prepended to relative plugin source paths (for example, `"./plugins"` lets you write `"source": "formatter"` instead of `"source": "./plugins/formatter"`) |

## Plugin entries

Each plugin entry in the `plugins` array describes a plugin and where to find it. You can include any field from the plugin manifest schema (like `description`, `version`, `author`, `commands`, `hooks`, etc.), plus these marketplace-specific fields: `source`, `category`, `tags`, and `strict`.

### Required fields

| Field    | Type             | Description                                                                                                                                            |
| :------- | :--------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`   | string           | Plugin identifier (kebab-case, no spaces). This is public-facing: users see it when installing (for example, `/plugin install my-plugin@marketplace`). |
| `source` | string or object | Where to fetch the plugin from (see [Plugin sources](#plugin-sources) below)                                                                           |

### Optional plugin fields

**Standard metadata fields:**

| Field         | Type    | Description                                                                                                                       |
| :------------ | :------ | :-------------------------------------------------------------------------------------------------------------------------------- |
| `description` | string  | Brief plugin description                                                                                                          |
| `version`     | string  | Plugin version                                                                                                                    |
| `author`      | object  | Plugin author information (`name` required, `email` optional)                                                                     |
| `homepage`    | string  | Plugin homepage or documentation URL                                                                                              |
| `repository`  | string  | Source code repository URL                                                                                                        |
| `license`     | string  | SPDX license identifier (for example, MIT, Apache-2.0)                                                                            |
| `keywords`    | array   | Tags for plugin discovery and categorization                                                                                      |
| `category`    | string  | Plugin category for organization                                                                                                  |
| `tags`        | array   | Tags for searchability                                                                                                            |
| `strict`      | boolean | Controls whether `plugin.json` is the authority for component definitions (default: true). See [Strict mode](#strict-mode) below. |

**Component configuration fields:**

| Field        | Type             | Description                                      |
| :----------- | :--------------- | :----------------------------------------------- |
| `commands`   | string or array  | Custom paths to command files or directories     |
| `agents`     | string or array  | Custom paths to agent files                      |
| `hooks`      | string or object | Custom hooks configuration or path to hooks file |
| `mcpServers` | string or object | MCP server configurations or path to MCP config  |
| `lspServers` | string or object | LSP server configurations or path to LSP config  |

## Plugin sources

Plugin sources tell Claude Code where to fetch each individual plugin listed in your marketplace. These are set in the `source` field of each plugin entry in `marketplace.json`.

Once a plugin is cloned or copied into the local machine, it is copied into the local versioned plugin cache at `~/.claude/plugins/cache`.

| Source        | Type                             | Fields                                | Notes                                                             |
| ------------- | -------------------------------- | ------------------------------------- | ----------------------------------------------------------------- |
| Relative path | `string` (e.g. `"./my-plugin"`) | --                                    | Local directory within the marketplace repo. Must start with `./` |
| `github`      | object                           | `repo`, `ref?`, `sha?`                |                                                                   |
| `url`         | object                           | `url` (must end .git), `ref?`, `sha?` | Git URL source                                                    |
| `npm`         | object                           | `package`, `version?`, `registry?`    | Installed via `npm install`                                       |
| `pip`         | object                           | `package`, `version?`, `registry?`    | Installed via pip                                                 |

> **Note:** **Marketplace sources vs plugin sources**: These are different concepts that control different things.
>
> * **Marketplace source** -- where to fetch the `marketplace.json` catalog itself. Set when users run `/plugin marketplace add` or in `extraKnownMarketplaces` settings. Supports `ref` (branch/tag) but not `sha`.
> * **Plugin source** -- where to fetch an individual plugin listed in the marketplace. Set in the `source` field of each plugin entry inside `marketplace.json`. Supports both `ref` (branch/tag) and `sha` (exact commit).

### Relative paths

For plugins in the same repository:

```json
{
  "name": "my-plugin",
  "source": "./plugins/my-plugin"
}
```

> **Note:** Relative paths only work when users add your marketplace via Git (GitHub, GitLab, or git URL). If users add your marketplace via a direct URL to the `marketplace.json` file, relative paths will not resolve correctly. For URL-based distribution, use GitHub, npm, or git URL sources instead.

### GitHub repositories

```json
{
  "name": "github-plugin",
  "source": {
    "source": "github",
    "repo": "owner/plugin-repo"
  }
}
```

You can pin to a specific branch, tag, or commit:

```json
{
  "name": "github-plugin",
  "source": {
    "source": "github",
    "repo": "owner/plugin-repo",
    "ref": "v2.0.0",
    "sha": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
  }
}
```

| Field  | Type   | Description                                                           |
| :----- | :----- | :-------------------------------------------------------------------- |
| `repo` | string | Required. GitHub repository in `owner/repo` format                    |
| `ref`  | string | Optional. Git branch or tag (defaults to repository default branch)   |
| `sha`  | string | Optional. Full 40-character git commit SHA to pin to an exact version |

### Git repositories

```json
{
  "name": "git-plugin",
  "source": {
    "source": "url",
    "url": "https://gitlab.com/team/plugin.git"
  }
}
```

You can pin to a specific branch, tag, or commit:

```json
{
  "name": "git-plugin",
  "source": {
    "source": "url",
    "url": "https://gitlab.com/team/plugin.git",
    "ref": "main",
    "sha": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
  }
}
```

| Field | Type   | Description                                                           |
| :---- | :----- | :-------------------------------------------------------------------- |
| `url` | string | Required. Full git repository URL (must end with `.git`)              |
| `ref` | string | Optional. Git branch or tag (defaults to repository default branch)   |
| `sha` | string | Optional. Full 40-character git commit SHA to pin to an exact version |

### Strict mode

The `strict` field controls whether `plugin.json` is the authority for component definitions (commands, agents, hooks, skills, MCP servers, output styles).

| Value            | Behavior                                                                                                                                                         |
| :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `true` (default) | `plugin.json` is the authority. The marketplace entry can supplement it with additional components, and both sources are merged.                                 |
| `false`          | The marketplace entry is the entire definition. If the plugin also has a `plugin.json` that declares components, that's a conflict and the plugin fails to load. |

**When to use each mode:**

* **`strict: true`**: the plugin has its own `plugin.json` and manages its own components. The marketplace entry can add extra commands or hooks on top. This is the default and works for most plugins.
* **`strict: false`**: the marketplace operator wants full control. The plugin repo provides raw files, and the marketplace entry defines which of those files are exposed as commands, agents, hooks, etc. Useful when the marketplace restructures or curates a plugin's components differently than the plugin author intended.

## Host and distribute marketplaces

### Host on GitHub (recommended)

GitHub provides the easiest distribution method:

1. **Create a repository**: Set up a new repository for your marketplace
2. **Add marketplace file**: Create `.claude-plugin/marketplace.json` with your plugin definitions
3. **Share with teams**: Users add your marketplace with `/plugin marketplace add owner/repo`

### Host on other git services

Any git hosting service works, such as GitLab, Bitbucket, and self-hosted servers. Users add with the full repository URL:

```shell
/plugin marketplace add https://gitlab.com/company/plugins.git
```

### Private repositories

Claude Code supports installing plugins from private repositories. For manual installation and updates, Claude Code uses your existing git credential helpers. If `git clone` works for a private repository in your terminal, it works in Claude Code too.

Background auto-updates run at startup without credential helpers. To enable auto-updates for private marketplaces, set the appropriate authentication token in your environment:

| Provider  | Environment variables        | Notes                                     |
| :-------- | :--------------------------- | :---------------------------------------- |
| GitHub    | `GITHUB_TOKEN` or `GH_TOKEN` | Personal access token or GitHub App token |
| GitLab    | `GITLAB_TOKEN` or `GL_TOKEN` | Personal access token or project token    |
| Bitbucket | `BITBUCKET_TOKEN`            | App password or repository access token   |

### Version resolution and release channels

Plugin versions determine cache paths and update detection. You can specify the version in the plugin manifest (`plugin.json`) or in the marketplace entry (`marketplace.json`).

> **Warning:** When possible, avoid setting the version in both places. The plugin manifest always wins silently, which can cause the marketplace version to be ignored. For relative-path plugins, set the version in the marketplace entry. For all other plugin sources, set it in the plugin manifest.

### Managed marketplace restrictions

For organizations requiring strict control over plugin sources, administrators can restrict which plugin marketplaces users are allowed to add using the `strictKnownMarketplaces` setting in managed settings.

When `strictKnownMarketplaces` is configured in managed settings, the restriction behavior depends on the value:

| Value               | Behavior                                                         |
| ------------------- | ---------------------------------------------------------------- |
| Undefined (default) | No restrictions. Users can add any marketplace                   |
| Empty array `[]`    | Complete lockdown. Users cannot add any new marketplaces         |
| List of sources     | Users can only add marketplaces that match the allowlist exactly |

## Validation and testing

Test your marketplace before sharing.

Validate your marketplace JSON syntax:

```bash
claude plugin validate .
```

Or from within Claude Code:

```shell
/plugin validate .
```

## Troubleshooting

### Marketplace not loading

**Symptoms**: Can't add marketplace or see plugins from it

**Solutions**:

* Verify the marketplace URL is accessible
* Check that `.claude-plugin/marketplace.json` exists at the specified path
* Ensure JSON syntax is valid using `claude plugin validate` or `/plugin validate`
* For private repositories, confirm you have access permissions

### Plugin installation failures

**Symptoms**: Marketplace appears but plugin installation fails

**Solutions**:

* Verify plugin source URLs are accessible
* Check that plugin directories contain required files
* For GitHub sources, ensure repositories are public or you have access
* Test plugin sources manually by cloning/downloading

### Files not found after installation

**Symptoms**: Plugin installs but references to files fail, especially files outside the plugin directory

**Cause**: Plugins are copied to a cache directory rather than used in-place. Paths that reference files outside the plugin's directory (such as `../shared-utils`) won't work because those files aren't copied.

**Solutions**: See [Plugin caching and file resolution](/en/plugins-reference#plugin-caching-and-file-resolution) for workarounds including symlinks and directory restructuring.

## See also

* [Discover and install prebuilt plugins](/en/discover-plugins) - Installing plugins from existing marketplaces
* [Plugins](/en/plugins) - Creating your own plugins
* [Plugins reference](/en/plugins-reference) - Complete technical specifications and schemas
* [Plugin settings](/en/settings#plugin-settings) - Plugin configuration options
* [strictKnownMarketplaces reference](/en/settings#strictknownmarketplaces) - Managed marketplace restrictions
