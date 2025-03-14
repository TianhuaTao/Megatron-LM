
# kill
HOSTFILE=/workspace/hostfile-8x8xA
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "kill -9 \$(ps aux | grep pretrain_gpt | grep -v grep | awk '{print \$2}')" &
done < $HOSTFILE

HOSTFILE=/workspace/hostfile-2x8xB
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "kill -9 \$(ps aux | grep pretrain_gpt | grep -v grep | awk '{print \$2}')" &
done < $HOSTFILE

######

HOSTFILE=/workspace/hostfile
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "kill -9 \$(ps aux | grep pretrain_gpt | grep -v grep | awk '{print \$2}')" &
done < $HOSTFILE

# test
HOSTFILE=/workspace/hostfile
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "uname -a" &
done < $HOSTFILE

###
#custom cmd
HOSTFILE=/workspace/hostfile-8x8xA
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "bash /workspace/Megatron-LM/scripts/remote_cmd.sh" &
done < $HOSTFILE

