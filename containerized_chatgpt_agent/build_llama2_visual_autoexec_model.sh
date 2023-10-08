docker cp Modelfile_visual ollama:/root/.ollama/Modelfile_visual
# docker exec -it ollama ollama create autoexec -f Modelfile
docker exec -it ollama ollama create autoexec_visual -f "root/.ollama/Modelfile_visual"