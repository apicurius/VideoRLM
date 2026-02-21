# Claude Code Changelog

This is the changelog for Claude Code maintained by Anthropic. The document covers releases from version 0.2.21 through 2.1.47.

## Latest Release: 2.1.47

### Key Fixes in 2.1.47:
- **FileWriteTool**: Fixed line counting to preserve intentional trailing blank lines
- **Windows Terminal**: Fixed rendering bugs caused by `os.EOL` (`\r\n`)
- **VS Code Plan Preview**: Auto-updates as Claude iterates, enables commenting only when ready
- **Unicode Support**: Fixed bold/colored text shifting on Windows due to line endings
- **PDF Handling**: Fixed compaction failing with many PDF documents by stripping document blocks
- **Performance**: Improved memory usage in long-running sessions, improved startup performance (~500ms reduction)
- **Bash Tool**: Fixed output silently discarded on Windows with MSYS2/Cygwin
- **File Mentions**: Improved `@` file completion performance with pre-warming and session-based caching
- **LSP Operations**: Fixed gitignored files appearing in `findReferences` results
- **Bash Commands**: Fixed commands with backslash-newline continuation lines producing spurious empty arguments
- **CJK Text**: Fixed wide characters causing misaligned timestamps
- **Custom Agents**: Fixed model field being ignored when spawning team teammates
- **Heredoc Support**: Fixed zsh heredoc failing with "read-only file system" error
- **Session Management**: Fixed session names being lost after compaction and context truncation
- **Image Pasting**: Fixed on WSL2 systems handling BMP format
- **Background Agents**: Fixed returning raw transcript data instead of final answer
- **Unicode Quotes**: Fixed Edit tool corrupting Unicode curly quotes

## Version 2.1.45
- Added support for Claude Sonnet 4.6
- Added `spinnerTipsOverride` setting for custom spinner tips
- Added `SDKRateLimitInfo` and `SDKRateLimitEvent` types to SDK
- Fixed Agent Teams teammates on Bedrock, Vertex, and Foundry
- Improved memory usage for shell commands with large output

## Version 2.1.32
- **Claude Opus 4.6** is now available
- Added research preview **Agent Teams** feature (set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`)
- Claude now automatically records and recalls memories
- Added "Summarize from here" to message selector
- Skills in `.claude/skills/` loaded automatically from additional directories
- Fixed heredoc handling with JavaScript template literals

## Version 2.1.30
- Added `pages` parameter to Read tool for PDF page ranges (e.g., `pages: "1-5"`)
- Large PDFs (>10 pages) return lightweight reference when mentioned
- Added `/debug` command for troubleshooting
- Added token count, tool uses, and duration metrics to Task tool results
- Added reduced motion mode config option

## Configuration & Installation

The repository contains configuration files and documentation for:
- **Agent definitions** in `.claude/agents/`
- **Skills** in `.claude/skills/`
- **Project configuration** via `.claude.json` and `CLAUDE.md`
- **Settings** in `settings.json` with extensive customization options

## Supported Providers

Claude Code works with:
- **Anthropic API** (direct)
- **AWS Bedrock**
- **Google Vertex AI**
- **Anthropic Foundry**

## Notable Historical Features

The changelog tracks the evolution from version 0.2.21 through 2.1.47, including:
- Extended thinking support
- Plan mode for step-by-step problem solving
- File system operations (Read, Write, Edit, Glob, Grep)
- Bash command execution with sandboxing
- Web search capabilities
- MCP (Model Context Protocol) integration
- Plugin system
- Session management and resumption
- Permission management and bash classifiers

## Recent Performance & Stability Improvements

- Memory usage optimization for long-running sessions
- Startup performance improvements
- Better handling of large conversations
- Improved terminal rendering
- Fixed various crashes and hangs
- Better error messages and diagnostics
