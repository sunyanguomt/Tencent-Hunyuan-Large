#!/bin/bash

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

WORK_HOME="$PWD"
LOG_DIR=./output/$CURRENT_TIME
SCRIPT_FILE=./train.sh
HOSTFILE="./hostfile"

# 验证必要文件
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "ERROR: Training script $SCRIPT_FILE not found!"
    exit 1
fi

if [ ! -f "$HOSTFILE" ]; then
    echo "ERROR: Hostfile $HOSTFILE not found!"
    exit 1
fi

# 确保训练脚本有执行权限
chmod +x $SCRIPT_FILE

# 获取主机列表 (过滤注释行和空行)
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}')
if [ -z "$hostlist" ]; then
    echo "ERROR: No valid hosts found in $HOSTFILE"
    exit 1
fi

echo "Hosts to launch training on:"
echo "$hostlist"
echo "----------------------------------------"

COUNT=0

for host in $hostlist; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching on host: $host"

    # 正确构建远程命令
    REMOTE_CMD="
        cd $WORK_HOME && \
        bash $SCRIPT_FILE \"$WORK_HOME\" \"$HOSTFILE\" \"$CURRENT_TIME\" \"$COUNT\" \"$host\" \
        > $LOG_DIR/train.log.$COUNT.$host 2>&1
    "

    # 调试输出
    echo "Executing on $host: $REMOTE_CMD"

    # 执行远程SSH命令 (后台执行)
    ssh -n $host "bash -c '$(echo "$REMOTE_CMD" | sed "s/'/'\\\\''/g")'" &

    sleep 1  # 避免SSH连接风暴
    ((COUNT++))
done

echo "========================================"
echo "Launched training on $COUNT nodes"
echo "Logs are being written to: $LOG_DIR/"
echo "========================================"
