# 强行杀死所有
pkill -TERM -f 'vllm\.entrypoints\.openai\.api_server|EngineCore|multiprocessing\.spawn.*vllm' || true
sleep 3
pkill -KILL -f 'vllm\.entrypoints\.openai\.api_server|EngineCore|multiprocessing\.spawn.*vllm' || true
