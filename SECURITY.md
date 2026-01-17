# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report through GitHub's private vulnerability reporting feature.

When reporting, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- Any suggested fixes (if available)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours.
- **Assessment**: We will assess the vulnerability and determine its severity.
- **Updates**: We will keep you informed of our progress.
- **Resolution**: Once fixed, we will notify you and credit you (unless you prefer anonymity).

### Disclosure Policy

- We follow coordinated disclosure practices
- We aim to release patches within 90 days of confirmed vulnerabilities
- We will publicly disclose vulnerabilities after a fix is available

## Security Best Practices for Users

When using mcpbr:

1. **API Keys**: Never commit API keys to version control. Use environment variables.
2. **Docker**: Ensure Docker is properly secured on your system.
3. **MCP Servers**: Only use trusted MCP servers from verified sources.
4. **Network**: Be aware that MCP servers may have network access within their containers.

## Dependencies

We regularly update dependencies to address known vulnerabilities. You can check for outdated packages with:

```bash
pip list --outdated
```

Thank you for helping keep mcpbr secure!
