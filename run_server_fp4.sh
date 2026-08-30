#!/bin/bash
# Deprecated name -- kept as a compatibility shim so other machines/scripts
# using the old name don't break. The script never actually loaded FP4
# weights (there's only one DiT checkpoint); it just disabled FP8, which is
# what run_server_bf16.sh is named for. Use run_server_bf16.sh going forward.
echo "run_server_fp4.sh has been renamed to run_server_bf16.sh (same behavior -- FP8 disabled, bf16 attention/MLP). Update your scripts/habits; this shim will be removed eventually." >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_server_bf16.sh" "$@"
