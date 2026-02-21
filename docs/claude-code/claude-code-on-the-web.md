# Claude Code on the web

> Run Claude Code tasks asynchronously on secure cloud infrastructure

> **Note:** Claude Code on the web is currently in research preview.

## What is Claude Code on the web?

Claude Code on the web lets developers kick off Claude Code from the Claude app. This is perfect for:

* **Answering questions**: Ask about code architecture and how features are implemented
* **Bug fixes and routine tasks**: Well-defined tasks that don't require frequent steering
* **Parallel work**: Tackle multiple bug fixes in parallel
* **Repositories not on your local machine**: Work on code you don't have checked out locally
* **Backend changes**: Where Claude Code can write tests and then write code to pass those tests

Claude Code is also available on the Claude iOS app for kicking off tasks on the go and monitoring work in progress.

You can move between local and remote development: send tasks from your terminal to run on the web with the `&` prefix, or teleport web sessions back to your terminal to continue locally.

## Who can use Claude Code on the web?

Claude Code on the web is available in research preview to:

* **Pro users**
* **Max users**
* **Team users**
* **Enterprise users** with premium seats or Chat + Claude Code seats

## Getting started

1. Visit [claude.ai/code](https://claude.ai/code)
2. Connect your GitHub account
3. Install the Claude GitHub app in your repositories
4. Select your default environment
5. Submit your coding task
6. Review changes in diff view, iterate with comments, then create a pull request

## How it works

When you start a task on Claude Code on the web:

1. **Repository cloning**: Your repository is cloned to an Anthropic-managed virtual machine
2. **Environment setup**: Claude prepares a secure cloud environment with your code
3. **Network configuration**: Internet access is configured based on your settings
4. **Task execution**: Claude analyzes code, makes changes, runs tests, and checks its work
5. **Completion**: You're notified when finished and can create a PR with the changes
6. **Results**: Changes are pushed to a branch, ready for pull request creation

## Review changes with diff view

Diff view lets you see exactly what Claude changed before creating a pull request. Instead of clicking "Create PR" to review changes in GitHub, view the diff directly in the app and iterate with Claude until the changes are ready.

When Claude makes changes to files, a diff stats indicator appears showing the number of lines added and removed (for example, `+12 -1`). Select this indicator to open the diff viewer, which displays a file list on the left and the changes for each file on the right.

From the diff view, you can:

* Review changes file by file
* Comment on specific changes to request modifications
* Continue iterating with Claude based on what you see

## Moving tasks between web and terminal

You can start tasks on the web and continue them in your terminal, or send tasks from your terminal to run on the web. Web sessions persist even if you close your laptop, and you can monitor them from anywhere including the Claude iOS app.

> **Note:** Session handoff is one-way: you can pull web sessions into your terminal, but you can't push an existing terminal session to the web. The `&` prefix creates a *new* web session with your current conversation context.

### From terminal to web

Start a message with `&` inside Claude Code to send a task to run on the web:

```
& Fix the authentication bug in src/auth/login.ts
```

This creates a new web session on claude.ai with your current conversation context. The task runs in the cloud while you continue working locally. Use `/tasks` to check progress, or open the session on claude.ai or the Claude iOS app to interact directly.

You can also start a web session directly from the command line:

```bash
claude --remote "Fix the authentication bug in src/auth/login.ts"
```

#### Tips for background tasks

**Plan locally, execute remotely**: For complex tasks, start Claude in plan mode to collaborate on the approach before sending work to the web:

```bash
claude --permission-mode plan
```

In plan mode, Claude can only read files and explore the codebase. Once you're satisfied with the plan, send it to the web for autonomous execution:

```
& Execute the migration plan we discussed
```

**Run tasks in parallel**: Each `&` command creates its own web session that runs independently:

```
& Fix the flaky test in auth.spec.ts
& Update the API documentation
& Refactor the logger to use structured output
```

Monitor all sessions with `/tasks`.

### From web to terminal

There are several ways to pull a web session into your terminal:

* **Using `/teleport`**: From within Claude Code, run `/teleport` (or `/tp`) to see an interactive picker of your web sessions.
* **Using `--teleport`**: From the command line, run `claude --teleport` for an interactive session picker, or `claude --teleport <session-id>` to resume a specific session directly.
* **From `/tasks`**: Run `/tasks` to see your background sessions, then press `t` to teleport into one
* **From the web interface**: Click "Open in CLI" to copy a command you can paste into your terminal

When you teleport a session, Claude verifies you're in the correct repository, fetches and checks out the branch from the remote session, and loads the full conversation history into your terminal.

#### Requirements for teleporting

| Requirement        | Details                                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Clean git state    | Your working directory must have no uncommitted changes. Teleport prompts you to stash changes if needed.              |
| Correct repository | You must run `--teleport` from a checkout of the same repository, not a fork.                                          |
| Branch available   | The branch from the web session must have been pushed to the remote. Teleport automatically fetches and checks it out. |
| Same account       | You must be authenticated to the same Claude.ai account used in the web session.                                       |

### Sharing sessions

To share a session, toggle its visibility according to the account types below. After that, share the session link as-is.

#### Sharing from an Enterprise or Teams account

For Enterprise and Teams accounts, the two visibility options are **Private** and **Team**. Team visibility makes the session visible to other members of your Claude.ai organization. Repository access verification is enabled by default, based on the GitHub account connected to the recipient's account.

#### Sharing from a Max or Pro account

For Max and Pro accounts, the two visibility options are **Private** and **Public**. Public visibility makes the session visible to any user logged into claude.ai.

Check your session for sensitive content before sharing. Sessions may contain code and credentials from private GitHub repositories. Enable repository access verification and/or withhold your name from your shared sessions by going to Settings > Claude Code > Sharing settings.

## Cloud environment

### Default image

We build and maintain a universal image with common toolchains and language ecosystems pre-installed. This image includes:

* Popular programming languages and runtimes
* Common build tools and package managers
* Testing frameworks and linters

#### Checking available tools

To see what's pre-installed in your environment, ask Claude Code to run:

```bash
check-tools
```

#### Language-specific setups

The universal image includes pre-configured environments for:

* **Python**: Python 3.x with pip, poetry, and common scientific libraries
* **Node.js**: Latest LTS versions with npm, yarn, pnpm, and bun
* **Ruby**: Versions 3.1.6, 3.2.6, 3.3.6 (default: 3.3.6) with gem, bundler, and rbenv
* **PHP**: Version 8.4.14
* **Java**: OpenJDK with Maven and Gradle
* **Go**: Latest stable version with module support
* **Rust**: Rust toolchain with cargo
* **C++**: GCC and Clang compilers

#### Databases

The universal image includes:

* **PostgreSQL**: Version 16
* **Redis**: Version 7.0

### Environment configuration

When you start a session in Claude Code on the web:

1. **Environment preparation**: We clone your repository and run any configured Claude hooks for initialization.
2. **Network configuration**: We configure internet access for the agent. Internet access is limited by default.
3. **Claude Code execution**: Claude Code runs to complete your task, writing code, running tests, and checking its work.
4. **Outcome**: When Claude completes its work, it will push the branch to remote. You will be able to create a PR for the branch.

**To add a new environment:** Select the current environment to open the environment selector, and then select "Add environment".

**To update an existing environment:** Select the current environment, to the right of the environment name, and select the settings button.

**To select your default environment from the terminal:** Run `/remote-env` to choose which one to use when starting web sessions.

### Dependency management

Custom environment images and snapshots are not yet supported. As a workaround, you can use SessionStart hooks to install packages when a session starts.

To configure automatic dependency installation, add a SessionStart hook to your repository's `.claude/settings.json` file:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/scripts/install_pkgs.sh"
          }
        ]
      }
    ]
  }
}
```

Create the corresponding script at `scripts/install_pkgs.sh`:

```bash
#!/bin/bash

# Only run in remote environments
if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

npm install
pip install -r requirements.txt
exit 0
```

Make it executable: `chmod +x scripts/install_pkgs.sh`

#### Persist environment variables

SessionStart hooks can persist environment variables for subsequent Bash commands by writing to the file specified in the `CLAUDE_ENV_FILE` environment variable.

#### Dependency management limitations

* **Hooks fire for all sessions**: Check the `CLAUDE_CODE_REMOTE` environment variable in your script.
* **Requires network access**: Install commands need network access to reach package registries.
* **Proxy compatibility**: All outbound traffic passes through a security proxy. Some package managers do not work correctly with this proxy. Bun is a known example.
* **Runs on every session start**: Hooks run each time a session starts or resumes, adding startup latency.

## Network access and security

### GitHub proxy

All GitHub operations go through a dedicated proxy service that manages GitHub authentication securely.

### Security proxy

Environments run behind an HTTP/HTTPS network proxy for security and abuse prevention purposes.

### Access levels

By default, network access is limited to allowlisted domains. You can configure custom network access, including disabling network access.

### Default allowed domains

When using "Limited" network access, many domains are allowed by default including:

* **Anthropic Services**: api.anthropic.com, claude.ai, etc.
* **Version Control**: github.com, gitlab.com, bitbucket.org, etc.
* **Container Registries**: Docker, GCR, GHCR, etc.
* **Cloud Platforms**: AWS, GCP, Azure, etc.
* **Package Managers**: npm, PyPI, RubyGems, crates.io, Go modules, Maven, NuGet, etc.
* **Linux Distributions**: Ubuntu repositories
* **Development Tools**: Kubernetes, HashiCorp, Anaconda, Node.js, etc.

## Security and isolation

Claude Code on the web provides strong security guarantees:

* **Isolated virtual machines**: Each session runs in an isolated, Anthropic-managed VM
* **Network access controls**: Network access is limited by default, and can be disabled
* **Credential protection**: Sensitive credentials are never inside the sandbox with Claude Code. Authentication is handled through a secure proxy using scoped credentials
* **Secure analysis**: Code is analyzed and modified within isolated VMs before creating PRs

## Pricing and rate limits

Claude Code on the web shares rate limits with all other Claude and Claude Code usage within your account. Running multiple tasks in parallel will consume more rate limits proportionately.

## Limitations

* **Repository authentication**: You can only move sessions from web to local when you are authenticated to the same account
* **Platform restrictions**: Claude Code on the web only works with code hosted in GitHub

## Best practices

1. **Use Claude Code hooks**: Configure SessionStart hooks to automate environment setup and dependency installation.
2. **Document requirements**: Clearly specify dependencies and commands in your `CLAUDE.md` file.

## Related resources

* Hooks configuration
* Settings reference
* Security
* Data usage
