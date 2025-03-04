#!/usr/bin/env bash
#
# Usage:
#   ./run_script_remotely.sh [path_to_hostfile] [remote_script]
#
# Example:
#   ./run_script_remotely.sh hosts.txt my_script.sh
#
# Description:
#   Reads each hostname from the specified file, then SSH into each host
#   and run the remote script in the background on the remote machine using nohup.
#

# Exit immediately if any command fails:
# set -e

WORKSPACE_DIR=/workspace
HOSTFILE=${1:-"${WORKSPACE_DIR}/hostfile-2x8xA"}       # Default to "hosts.txt" if not provided
REMOTEPATH=${2:-"${WORKSPACE_DIR}/Megatron-LM/scripts/debug_core_moe_dev-ref-120b-token-cosine.sh"}   # Default to "script.sh" if not provided

# print the hostfile and remote script path
echo "HOSTFILE: $HOSTFILE"
cat $HOSTFILE
echo "REMOTEPATH: $REMOTEPATH"



if [[ ! -f "$HOSTFILE" ]]; then
  echo "Error: host file '$HOSTFILE' not found."
  exit 1
fi

TIMESTAMP=$(date '+%Y%m%d-%H%M%S')

# We'll keep an index that increments for each valid host
index=0

while IFS= read -r HOST; do
  # Skip empty lines or lines starting with '#' (comment)
  [[ -z "$HOST" || "$HOST" =~ ^# ]] && continue

  echo "Starting remote script on host '$HOST' with index=$index ..."

  # Run the remote script in the background, streaming its output locally.
  # - The remote script is assumed to already exist at $REMOTEPATH on the remote machine.
  # - We pass two arguments: (1) index, (2) the hostfile path.
  # - We pipe stderr and stdout together (2>&1) so we can see all output in one stream.
  # - 'sed "s/^/[$HOST - $index] /"' prefixes each line with the host and index.
  #
  # Putting '&' at the end runs the ssh command in the background locally.
  # Use -n with SSH -n tells SSH to redirect its standard input from /dev/null, so it cannot eat your local script’s input
  ssh -n -p 30255 -o StrictHostKeyChecking=no "$HOST" "$REMOTEPATH $index '$HOSTFILE' '$TIMESTAMP'" 2>&1 | sed "s/^/[$HOST - $index] /" &
#   echo "ssh return code: $?"
#   echo "Submitted remote script on host '$HOST' with index=$index."

  ((index++))

done < "$HOSTFILE"

# Wait for all background SSH jobs to finish
wait

echo "All remote scripts have finished."
