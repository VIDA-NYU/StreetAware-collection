# #!/bin/bash

# rm -f react.pid fastapi.pid fastapi.log react.log

# echo "Starting React App..."
# nohup bash -c "cd street-aware-app && npm start" > react.log 2>&1 & echo $! > react.pid

# echo "Starting FastAPI Service..."
# nohup bash -c "cd street-aware-service && source myenv/bin/activate && python app.py" > fastapi.log 2>&1 & echo $! > fastapi.pid

# echo "✅ Services started in background. Use ./stop.sh to terminate them."

#!/bin/bash

# Stop existing processes if running
./stop.sh 2>/dev/null

echo "Starting React App..."
nohup bash -c "cd street-aware-app && npm run start" > react.log 2>&1 &
sleep 3
lsof -ti :4000 > react.pid

echo "Starting FastAPI App..."
nohup bash -c "cd street-aware-service && source myenv/bin/activate && python app.py" > fastapi.log 2>&1 &
sleep 1
lsof -ti :8080 > fastapi.pid

echo "✅ Services started. Use ./stop.sh to terminate them."