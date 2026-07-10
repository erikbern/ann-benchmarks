#!/bin/bash

# PostgreSQL 日志过滤脚本
# 功能：过滤掉包含 STATEMENT 的行及其后的 SQL 内容
# 用法：filter_postgres_log.sh [log_file] [log_dir]
#
# 参数：
#   log_file: 可选，指定要处理的日志文件路径
#   log_dir: 可选，指定日志目录（默认: /var/lib/postgresql/16/main/log）
#
# 如果未指定 log_file，会自动查找最新的日志文件

LOG_DIR="${2:-/var/lib/postgresql/16/main/log}"

# 如果提供了日志文件路径，直接使用；否则查找最新的日志文件
if [ -n "$1" ] && [ -f "$1" ]; then
    LOG_FILE="$1"
elif [ -n "$1" ] && [ ! -f "$1" ]; then
    echo "错误: 指定的日志文件不存在: $1"
    exit 1
else
    LOG_FILE=$(ls -t ${LOG_DIR}/postgresql-*.log 2>/dev/null | head -1)
fi

if [ -z "$LOG_FILE" ]; then
    echo "警告: 未找到日志文件（目录: ${LOG_DIR}）"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "错误: 日志文件不存在: $LOG_FILE"
    exit 1
fi

# 过滤日志
FILTERED_LOG="${LOG_FILE}.filtered"

# 使用 awk 过滤：跳过包含 STATEMENT 的行及其后所有不包含日期时间戳的行
# PostgreSQL 日志格式：2025-12-05 21:00:19.989 CST [280809] ...
# STATEMENT 行本身也带日期时间戳，需要先检查 STATEMENT，再检查日期时间戳
# 注意：awk 的正则表达式不支持 {4} 这种语法，需要使用 [0-9][0-9][0-9][0-9]
awk '
BEGIN { skip_mode = 0 }
# 先检查是否是 STATEMENT 行（即使它也有日期时间戳）
/STATEMENT/ {
    skip_mode = 1  # 进入跳过模式
    next           # 跳过包含 STATEMENT 的行
}
# 匹配日期时间戳格式：以 YYYY-MM-DD HH:MM:SS 开头的行
/^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]/ {
    if (skip_mode == 1) {
        skip_mode = 0  # 遇到新的日期时间戳（非 STATEMENT），停止跳过模式
    }
    print
    next
}
skip_mode == 1 {
    next           # 跳过模式下，跳过所有不包含日期时间戳的行
}
{ print }
' "$LOG_FILE" > "$FILTERED_LOG"

echo "过滤后的日志保存到: $FILTERED_LOG"
echo "原始日志: $LOG_FILE"
echo "过滤后的日志行数: $(wc -l < "$FILTERED_LOG")"
echo "原始日志行数: $(wc -l < "$LOG_FILE")"

