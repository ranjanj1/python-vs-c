#!/bin/bash
# Prints yes/no only; never prints a secret value.
t(){ if eval "$2" >/dev/null 2>&1; then echo "REACHED $1"; else echo "blocked $1"; fi; }
t "login token file" 'cat ~/lab/claude-token'
t "token in shell env" 'env | grep -q CLAUDE_CODE_OAUTH_TOKEN'
t "token in process list" 'ps eww -u agentlab | grep -q CLAUDE_CODE_OAUTH_TOKEN='
t "earlier workspace" 'cat ~/pvc/test-5/sum.c'
t "old session history" 'ls ~/.claude/projects'
t "web via proxy" 'curl -s -f -m 8 -o /dev/null https://example.com'
t "web without proxy" 'env -u HTTPS_PROXY curl -s -f -m 8 -o /dev/null https://example.com'
t "pypi via proxy" 'curl -s -f -m 8 -o /dev/null https://pypi.org/simple/numpy/'
