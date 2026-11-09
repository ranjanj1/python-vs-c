#!/bin/bash
# Run inside the sandbox. Every line after "allowed" should say what the session must not reach.
t(){ if eval "$2" >/dev/null 2>&1; then echo "REACHED  $1"; else echo "blocked  $1"; fi; }
echo "== must work"; t "write own workspace" 'echo x > ~/pvc/probe/a.txt && cat ~/pvc/probe/a.txt'
t "write own config" 'echo x > ~/pvcfg/probe/a'; t "run clang" 'cd ~/pvc/probe && printf "int main(){return 0;}" > a.c && clang a.c -o a && ./a'
t "run claude binary" '~/.local/bin/claude --version'
echo "== must be blocked"; t "login token file" 'cat ~/lab/claude-token'; t "earlier workspace" 'cat ~/pvc/test-5/sum.c'
t "old session history" 'ls ~/.claude/projects'; t "race folder" 'ls ~/race'; t "Shared folder" 'ls /Users/Shared'
t "owner .env" 'cat /Users/OWNER/.env'; t "write to home" 'echo x > ~/x.txt'
t "token in process env list" "ps eww -u agentlab | grep -q PROBE_SECRET_VALUE"
