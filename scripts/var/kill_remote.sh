
# kill
HOSTFILE=/workspace/hostfile
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "kill -9 \$(ps aux | grep pretrain_gpt | grep -v grep | awk '{print \$2}')" &
done < $HOSTFILE


kill -9 $(ps aux | grep pretrain_gpt | grep -v grep | awk '{print $2}')

