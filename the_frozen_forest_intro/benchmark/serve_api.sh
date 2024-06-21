
# to resolve openmp related issues:

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macos
    export MKL_THREADING_LAYER=TBB
else
    # linux
    export MKL_THREADING_LAYER=GNU
fi

conda run -n sentence_similarity --no-capture-output python -m uvicorn similarity_server:app --host 0.0.0.0 --port 9000