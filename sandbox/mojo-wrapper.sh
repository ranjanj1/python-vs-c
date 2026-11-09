#!/bin/bash
# Installed as bin/mojo in the mojo kit. Mojo writes under home, which the session sandbox closes, and Claude Code's shell drops XDG settings.
d="/tmp/pvc-mojo/$(id -un)"
export XDG_CACHE_HOME="$d/cache" XDG_DATA_HOME="$d/data" XDG_CONFIG_HOME="$d/config" MODULAR_TELEMETRY_ENABLED=false
exec /Users/Shared/pvc-kit/mojo/bin/mojo-real "$@"
