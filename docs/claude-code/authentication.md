# Authentication

> Learn how to configure user authentication and credential management for Claude Code in your organization.

## Authentication methods

Setting up Claude Code requires access to Anthropic models. For teams, you can set up Claude Code access in one of these ways:

* Claude for Teams or Enterprise (recommended)
* Claude Console
* Amazon Bedrock
* Google Vertex AI
* Microsoft Foundry

### Claude for Teams or Enterprise

Claude for Teams and Claude for Enterprise provide the best experience for organizations using Claude Code. Team members get access to both Claude Code and Claude on the web with centralized billing and team management.

* **Claude for Teams**: self-service plan with collaboration features, admin tools, and billing management. Best for smaller teams.
* **Claude for Enterprise**: adds SSO, domain capture, role-based permissions, compliance API, and managed policy settings for organization-wide Claude Code configurations. Best for larger organizations with security and compliance requirements.

**Steps:**

1. Subscribe to Claude for Teams or contact sales for Claude for Enterprise.
2. Invite team members from the admin dashboard.
3. Team members install Claude Code and log in with their Claude.ai accounts.

### Claude Console authentication

For organizations that prefer API-based billing, you can set up access through the Claude Console.

**Steps:**

1. Create or use a Console account: Use your existing Claude Console account or create a new one.
2. Add users: You can add users through either method:
   * Bulk invite users from within the Console (Console -> Settings -> Members -> Invite)
   * Set up SSO
3. Assign roles: When inviting users, assign one of:
   * **Claude Code** role: users can only create Claude Code API keys
   * **Developer** role: users can create any kind of API key
4. Users complete setup: Each invited user needs to:
   * Accept the Console invite
   * Check system requirements
   * Install Claude Code
   * Log in with Console account credentials

### Cloud provider authentication

For teams using Amazon Bedrock, Google Vertex AI, or Microsoft Azure:

**Steps:**

1. Follow provider setup: Follow the Bedrock docs, Vertex docs, or Microsoft Foundry docs.
2. Distribute configuration: Distribute the environment variables and instructions for generating cloud credentials to your users.
3. Install Claude Code: Users can install Claude Code.

## Credential management

Claude Code securely manages your authentication credentials:

* **Storage location**: on macOS, API keys, OAuth tokens, and other credentials are stored in the encrypted macOS Keychain.
* **Supported authentication types**: Claude.ai credentials, Claude API credentials, Azure Auth, Bedrock Auth, and Vertex Auth.
* **Custom credential scripts**: the `apiKeyHelper` setting can be configured to run a shell script that returns an API key.
* **Refresh intervals**: by default, `apiKeyHelper` is called after 5 minutes or on HTTP 401 response. Set `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` environment variable for custom refresh intervals.

## See also

* Permissions: configure what Claude Code can access and do
* Settings: complete configuration reference
* Security: security safeguards and best practices
