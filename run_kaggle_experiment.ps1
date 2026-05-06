# Set UTF8 encoding to avoid Windows charmap encoding errors with Kaggle CLI
$env:PYTHONUTF8="1"

# Define variables
$KERNEL_ID = "kausikvaibhavpatra/psm-final-cli"
$PUSH_DIR = "tmp_kaggle_pull"
$OUTPUT_DIR = "kaggle_run_output_latest"

Write-Host "========================================="
Write-Host "Kaggle Experiment Runner"
Write-Host "========================================="

Write-Host "[1/3] Pushing kernel to Kaggle to start the run..."
kaggle kernels push -p $PUSH_DIR
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to push kernel. Exiting." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[2/3] Waiting for the kernel to finish running..."
$status = "running"
while ($status -match "running" -or $status -match "queued") {
    Start-Sleep -Seconds 30
    $statusOutput = kaggle kernels status $KERNEL_ID
    
    if ($statusOutput -match 'status "([^"]+)"') {
        $status = $matches[1]
    } else {
        # Fallback
        if ($statusOutput -match "running") { $status = "running" }
        elseif ($statusOutput -match "queued") { $status = "queued" }
        elseif ($statusOutput -match "complete") { $status = "complete" }
        elseif ($statusOutput -match "error") { $status = "error" }
        else { $status = "unknown" }
    }
    Write-Host "Current status: $status"
}

if ($status -match "error") {
    Write-Host "Kernel run ended with an error." -ForegroundColor Red
    exit 1
}

Write-Host "[3/3] Downloading output files..."
if (!(Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
}
kaggle kernels output $KERNEL_ID -p $OUTPUT_DIR

Write-Host "========================================="
Write-Host "Experiment run completed successfully!" -ForegroundColor Green
Write-Host "Output saved to: $OUTPUT_DIR"
Write-Host "========================================="
