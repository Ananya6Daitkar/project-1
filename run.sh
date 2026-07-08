#!/bin/bash
# run.sh — always uses the crater conda env, no matter what 'python' points to
PYTHON=/opt/anaconda3/envs/crater/bin/python
STREAMLIT=/opt/anaconda3/envs/crater/bin/streamlit
DIR="$(cd "$(dirname "$0")" && pwd)"

case "$1" in
  data)
    $PYTHON "$DIR/src/data_pipeline.py" --synthetic --n "${2:-100}"
    ;;
  train)
    $PYTHON "$DIR/src/train.py" --all --epochs "${2:-30}"
    ;;
  evaluate)
    $PYTHON "$DIR/src/evaluate.py" --all
    ;;
  visualize)
    $PYTHON "$DIR/src/visualize.py" --model yolov8n --demo
    ;;
  iot)
    $PYTHON "$DIR/src/iot_simulator.py" --burst "${2:-10}" --target Moon
    ;;
  app)
    $STREAMLIT run "$DIR/app/app.py" --server.port 8501 --server.headless true
    ;;
  *)
    echo "Usage: bash run.sh [data|train|evaluate|visualize|iot|app] [optional arg]"
    echo ""
    echo "  bash run.sh data 100    # generate 100 synthetic tiles"
    echo "  bash run.sh train 30    # train yolov8n + yolov8s for 30 epochs"
    echo "  bash run.sh evaluate    # evaluate both models"
    echo "  bash run.sh visualize   # generate sample visualisations"
    echo "  bash run.sh iot 10      # emit 10 IoT telemetry events"
    echo "  bash run.sh app         # launch Streamlit app at :8501"
    ;;
esac
