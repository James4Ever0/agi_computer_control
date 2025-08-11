SLEEP_INTERVAL=100

echo "Using Python: $(which python)"

echo "Confirm? (y/n)"
read -r confirm  # Read user input

# Check user confirmation
if [[ "$confirm" != "y" ]]; then
    echo "Exiting without running."
    exit 0
fi

while true; do
    python enumerate_and_scrape_pages.py
    exitcode=$?  # Corrected: no spaces around assignment
    
    # Exit loop if exitcode is 0 (success)
    if [[ $exitcode -eq 0 ]]; then
        break
    fi
    
    sleep $SLEEP_INTERVAL
done
echo "Loop complete"