set -e
trap 'kill -TERM $CASS_PID' EXIT

docker-entrypoint.sh cassandra -R &
CASS_PID=$!

echo "⏳ Waiting for Cassandra to come up …"
until cqlsh -u cassandra -p cassandra 127.0.0.1 9042 -e "DESCRIBE CLUSTER" >/dev/null 2>&1
do
  sleep 3
done
echo "✅ Cassandra is up!"

python3 -u /home/app/run.py "$@"

wait $CASS_PID