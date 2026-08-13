# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in wechat-mac-rpa, please report it responsibly.

**DO NOT open a public GitHub issue for security vulnerabilities.**

Instead, please:
1. Email the maintainer directly via GitHub
2. Or open a private security advisory: GitHub repo > Security > Advisories > New advisory

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 48 hours.

## Security Considerations

wechat-mac-rpa operates by:
- Taking screenshots of your screen (WeChat interface)
- Sending screenshots to LLM APIs for analysis
- Executing AppleScript commands to interact with UI

**Important:**
- Your chat content is sent to the LLM provider you configure (OpenAI, Claude, etc.)
- Review your LLM provider's data retention policy
- The project does NOT access WeChat's protocol, database, or internal APIs
- No credentials are stored or transmitted beyond your LLM API key
