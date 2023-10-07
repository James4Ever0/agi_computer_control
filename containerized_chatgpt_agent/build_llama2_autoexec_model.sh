docker cp Modelfile ollama:/root/.ollama/Modelfile
# docker exec -it ollama ollama create autoexec -f Modelfile
docker exec -it ollama ollama create autoexec -f "root/.ollama/Modelfile"