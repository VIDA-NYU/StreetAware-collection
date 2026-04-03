#!/bin/bash

MODE="real"
case "${1:-}" in
  mock|MOCK|--mock|-m)
    MODE="mock"
    ;;
esac

# Stop existing processes if running
./stop.sh "$MODE" 2>/dev/null

# If mock mode, start mock sensors first
if [ "$MODE" = "mock" ]; then
  echo "Starting mock sensors..."
  cd mock-sensors && ./build_and_run.sh && cd ..
  echo ""
fi

echo "Starting React App (mode: $MODE)..."
SENSOR_MODE="$MODE" nohup bash -c "cd street-aware-app && npm run start" > react.log 2>&1 &
sleep 3
lsof -ti :4000 > react.pid

echo "Starting FastAPI App (mode: $MODE)..."
SENSOR_MODE="$MODE" nohup bash -c "cd street-aware-service && source myenv/bin/activate && python app.py" > fastapi.log 2>&1 &
sleep 1
lsof -ti :8080 > fastapi.pid

echo ""
echo "✅ Services started in $MODE mode. Use ./stop.sh to terminate them."
