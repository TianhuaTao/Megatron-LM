
# kill
HOSTFILE=/workspace/hostfile-4x8xA
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "kill -9 \$(ps aux | grep pretrain_gpt | grep -v grep | awk '{print \$2}')" &
done < $HOSTFILE

HOSTFILE=/workspace/hostfile-2x8xB
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "kill -9 \$(ps aux | grep pretrain_gpt | grep -v grep | awk '{print \$2}')" &
done < $HOSTFILE

######


HOSTFILE=/workspace/hostfile-8x8xB
while read -r host; do
    ssh -p30255 -o "StrictHostKeyChecking no" "$host" "kill -9 \$(ps aux | grep pretrain_gpt | grep -v grep | awk '{print \$2}')" &
done < $HOSTFILE

HOSTFILE=/home/tianhua_tao_genbio_ai/bio/protein-moe/hostfile-8x8xB
while read -r host; do
    ssh -o "StrictHostKeyChecking no" "$host" "sudo kill -9 \$(ps aux | grep "runner/train.py" | grep -v grep | awk '{print \$2}')" &
done < "$HOSTFILE"
