#!/bin/bash

MODE=""
case "${1:-}" in
  mock|MOCK|--mock|-m)
    MODE="mock"
    ;;
esac

echo "Stopping React App..."
kill $(cat react.pid 2>/dev/null) 2>/dev/null && echo "✓ React stopped" || echo "⚠️ React not running"
rm -f react.pid

echo "Stopping FastAPI App..."
kill $(cat fastapi.pid 2>/dev/null) 2>/dev/null && echo "✓ FastAPI stopped" || echo "⚠️ FastAPI not running"
rm -f fastapi.pid

# If mock mode, also stop mock sensors
if [ "$MODE" = "mock" ]; then
  echo ""
  echo "Stopping mock sensors..."
  cd mock-sensors && ./stop.sh && cd ..
fi