#!/usr/bin/env bash
# Launch the Wan MemDual benchmark matrix from a shared VeOmni checkout.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/profile/run_memdual.sh CONFIG.yaml [launcher options] [-- benchmark/config overrides]

  --master-addr HOST    Reachable address of node 0 (or MASTER_ADDR).
  --peer USER@HOST     Start node 1 over passwordless SSH from node 0.
  --node-rank N        Node 0 or 1 when launching separately (or NODE_RANK).
  --nodes N            One or two nodes. Default: 2.
  --gpus-per-node N    Default: 8. Two-node cross-Ulysses requires an even count.
  --master-port N      First rendezvous port. Default: 29500 (or MASTER_PORT).
  --modes LIST         Comma-separated modes. Default: baseline,kvf,anchor,combined.
                       Also accepts baseline-sync and anchor-sync.
  --out-dir DIR        Default: results/memdual, relative to the checkout.
  --python PATH        Python executable in the training environment on both nodes.
  -h, --help           Show this help.

Example on node 0:
  bash scripts/profile/run_memdual.sh wan.yaml --master-addr 10.0.0.1 \
    --peer user@10.0.0.2 -- --benchmark.steps 20
EOF
}

fail() {
    printf '%s\n' "$*" >&2
    exit 2
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
config=$1
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
script_path="$script_dir/run_memdual.sh"
cd -- "$script_dir/../.."

nodes=2
gpus=8
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-}
master_port=${MASTER_PORT:-29500}
modes=baseline,kvf,anchor,combined
out_dir=results/memdual
python_bin=$(command -v python || true)
peer=
while (($#)); do
    case "$1" in
        --master-addr|--peer|--node-rank|--nodes|--gpus-per-node|--master-port|--modes|--out-dir|--python)
            [[ $# -ge 2 && -n $2 ]] || fail "Missing value for $1."
            case "$1" in
                --master-addr) master_addr=$2 ;;
                --peer) peer=$2 ;;
                --node-rank) node_rank=$2 ;;
                --nodes) nodes=$2 ;;
                --gpus-per-node) gpus=$2 ;;
                --master-port) master_port=$2 ;;
                --modes) modes=$2 ;;
                --out-dir) out_dir=$2 ;;
                --python) python_bin=$2 ;;
            esac
            shift 2
            ;;
        --) shift; break ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown launcher option $1. Put training overrides after --." ;;
    esac
done
overrides=("$@")

[[ $nodes =~ ^[12]$ && $gpus =~ ^[1-9][0-9]*$ && $node_rank =~ ^[01]$ ]] ||
    fail "Invalid nodes, GPUs per node, or node rank."
((node_rank < nodes)) || fail "Node rank must be smaller than the number of nodes."
[[ $master_port =~ ^[1-9][0-9]{0,4}$ ]] || fail "Invalid rendezvous port."
[[ -f $config ]] || fail "Missing training config: $config"
[[ -n $python_bin && -x $python_bin ]] || fail "Activate the training environment or pass --python /path/to/python."
topology=()
if ((nodes == 2)); then
    [[ -n $master_addr ]] || fail "Set --master-addr to the reachable address of node 0."
    ((gpus % 2 == 0)) || fail "Cross-node Ulysses requires an even number of GPUs per node."
    topology=(--model.accelerator.ulysses_size "$gpus" --model.accelerator.cp_size 1
              --model.accelerator.dp_replicate_size 2 --model.accelerator.dp_shard_size 1)
else
    [[ -z $peer ]] || fail "--peer requires two nodes."
    master_addr=${master_addr:-127.0.0.1}
fi
IFS=, read -r -a cases <<< "$modes"
(("${#cases[@]}" > 0 && master_port + ${#cases[@]} - 1 <= 65535)) || fail "Invalid modes or port range."
for mode in "${cases[@]}"; do
    case "$mode" in
        baseline|kvf|anchor|combined|baseline-sync|anchor-sync) ;;
        *) fail "Unknown benchmark mode: $mode" ;;
    esac
done

if [[ -n $peer ]]; then
    ((node_rank == 0)) || fail "Use --peer only on node 0."
    # Bash-quote each argument for the SSH login shell.
    printf -v remote_command '%q ' bash "$script_path" "$config" \
        --master-addr "$master_addr" --master-port "$master_port" --node-rank 1 \
        --nodes "$nodes" --gpus-per-node "$gpus" --modes "$modes" --out-dir "$out_dir" \
        --python "$python_bin" -- "${overrides[@]}"
    ssh -o BatchMode=yes -- "$peer" "$remote_command" &
    peer_pid=$!
    trap 'kill "$peer_pid" 2>/dev/null || true' EXIT
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MEMDUAL_NODES="$nodes"
for index in "${!cases[@]}"; do
    label=${cases[index]}
    mode=${label%-sync}
    reference=async_ulysses
    if [[ $label == *-sync ]]; then
        reference=ulysses
    fi
    printf 'Node %s starting %s\n' "$node_rank" "$label"
    # Apply rank reordering in workers, after torchrun establishes the static rendezvous.
    "$python_bin" -m torch.distributed.run --nnodes="$nodes" --nproc-per-node="$gpus" \
        --rdzv-backend=static --node-rank="$node_rank" --master-addr="$master_addr" \
        --master-port="$((master_port + index))" --rdzv-conf=timeout=90 \
        --no-python /bin/bash -c '
            if ((MEMDUAL_NODES == 2)); then
                half=$((LOCAL_WORLD_SIZE / 2))
                export RANK=$(( (LOCAL_RANK / half) * LOCAL_WORLD_SIZE + GROUP_RANK * half + LOCAL_RANK % half ))
            fi
            exec "$@"
        ' _ "$python_bin" scripts/profile/benchmark_memdual.py "$config" \
        --benchmark.mode "$mode" --benchmark.reference_attention "$reference" \
        --benchmark.output "$out_dir/$label.json" --train.checkpoint.output_dir "$out_dir/runargs/$label" \
        "${topology[@]}" "${overrides[@]}"
done

if [[ -n $peer ]]; then
    wait "$peer_pid"
    trap - EXIT
fi
