AGG_BINARY="/home/jamesbrown/Downloads/agg-x86_64-unknown-linux-musl"
INPUT_CAST="./vim_rectangle_selection.cast"

echo "Using agg: $AGG_BINARY"
echo "Input asciicast: $INPUT_CAST"

echo "Rendering using default settings (resvg)"

$AGG_BINARY --renderer resvg $INPUT_CAST vim_rectangle_selection_resvg.gif

echo "Rendering using fontdue"

$AGG_BINARY --renderer fontdue $INPUT_CAST vim_rectangle_selection_fontdue.gif

# conclusion:
# fontdue is faster, while the results are basically the same,
# prefer fontdue