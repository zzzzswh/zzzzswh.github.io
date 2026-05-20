# Export-PptSlides.ps1
# Export PowerPoint slides to PNG images
# Usage: .\scripts\Export-PptSlides.ps1 -PptPath "path\to\slides.pptx" -OutDir "public\assets\fno-slides" -Start 2 -End 30

param(
    [Parameter(Mandatory = $true)]
    [string]$PptPath,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [int]$Start = 1,
    [int]$End,
    [int]$Scale = 2
)

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = $false
} catch {
    Write-Error "Failed to start PowerPoint. Is it installed?"
    exit 1
}

try {
    Write-Host "Opening: $PptPath"
    $presentation = $ppt.Presentations.Open($PptPath, $true, $false, $false)

    $totalSlides = $presentation.Slides.Count
    if (-not $End -or $End -gt $totalSlides) {
        $End = $totalSlides
    }

    Write-Host "Total slides: $totalSlides, exporting $Start ~ $End"

    for ($i = $Start; $i -le $End; $i++) {
        $num = $i.ToString("00")
        $outPath = Join-Path $OutDir "slide-$num.png"
        Write-Host "[$num/$End] Exporting..."
        $presentation.Slides[$i].Export($outPath, "PNG", 1920 * $Scale)
        Write-Host "  -> $outPath"
    }

    Write-Host "`nDone. Exported $($End - $Start + 1) images to: $OutDir"
} finally {
    $presentation.Close()
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
