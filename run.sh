#!/usr/bin/env bash
# Usage: ./run.sh [api|worker|ui|upload <file>]
set -euo pipefail

source .venv/bin/activate

API_URL="${API_URL:-http://localhost:8000}"

case "${1:-}" in
  api)
    uvicorn backend.api.app:app --host localhost --port 8000
    ;;

  worker)
    python -m backend.worker.sqs_listener
    ;;

  ui)
    cd frontend && npm run dev
    ;;

  upload)
    FILE="${2:?usage: ./run.sh upload <path-to-file>}"
    FILENAME=$(basename "$FILE")

    RESPONSE=$(curl -sf "$API_URL/presigned-url?filename=$FILENAME")
    URL=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['url'])")
    KEY=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['key'])")
    echo "key: $KEY"

    HTTP=$(curl -sf -o /dev/null -w "%{http_code}" -X PUT "$URL" --upload-file "$FILE")
    echo "s3 upload: $HTTP"

    NOTIFY=$(curl -sf -X POST "$API_URL/notify" \
      -H "Content-Type: application/json" \
      -d "{\"key\": \"$KEY\"}")
    echo "notify: $NOTIFY"
    ;;

  *)
    echo "Usage: $0 [api|worker|ui|upload <file>]"
    exit 1
    ;;
esac
