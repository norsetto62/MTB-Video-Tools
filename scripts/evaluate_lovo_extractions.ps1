$ErrorActionPreference = "Stop"

# ==============================================================================
# LOVO EXTRACTOR EVALUATION
#
# Reads the final clip CSVs produced by the production extractor and evaluates
# them against the original 0..3 annotations.
#
# Evaluation:
#   - 2 FPS and 10 FPS extractions
#   - 4s ML window
#   - 2s prediction stride
#   - 120s target
#
# Input:
#   output\extractions\<video>_2fps.csv
#   output\extractions\<video>_10fps.csv
#
# Metrics:
#   SelectedSeconds
#   ClipCount
#   MTB3Seconds
#   MTB3TotalSeconds
#   MTB3Recall
#   MTB3Precision
#   MTB23Seconds
#   MTB23TotalSeconds
#   MTB23Precision
#   MTB0Seconds
#   MTB0Contamination
#
# No project files are modified except the output CSV.
# ==============================================================================

$RepoRoot = Split-Path -Parent $PSScriptRoot

$ExtractionDir = Join-Path $RepoRoot "output\extractions"
$AnnotationDir = Join-Path $RepoRoot "data\annotations"

$Videos = @(
    "Ascoli",
    "Capranica",
    "G2",
    "G2_steep",
    "Iano",
    "Mentorella",
    "Orvinio",
    "Tivoli"
)

$FPSValues = @(
    "2fps",
    "10fps"
)

# ------------------------------------------------------------------------------
# Time parser
# ------------------------------------------------------------------------------

function Convert-ToSeconds {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Value
    )

    $Value = $Value.Trim()

    if ($Value -match "^\d+(\.\d+)?$") {
        return [double]$Value
    }

    $parts = $Value.Split(":")

    if ($parts.Count -eq 2) {
        return (
            [double]$parts[0] * 60.0 +
            [double]$parts[1]
        )
    }

    if ($parts.Count -eq 3) {
        return (
            [double]$parts[0] * 3600.0 +
            [double]$parts[1] * 60.0 +
            [double]$parts[2]
        )
    }

    throw "Invalid time value: '$Value'"
}

# ------------------------------------------------------------------------------
# Read annotations
# ------------------------------------------------------------------------------

function Read-Annotations {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Path
    )

    $annotations = @()

    foreach ($line in Get-Content -LiteralPath $Path) {

        $text = $line.Trim()

        if ([string]::IsNullOrWhiteSpace($text)) {
            continue
        }

        if ($text.StartsWith("#")) {
            continue
        }

        if ($text -match "^\s*Start\s+End\s+MTB") {
            continue
        }

        # Expected:
        # Start End MTB Remarks

        $parts = $text -split "\s+", 4

        if ($parts.Count -lt 3) {
            continue
        }

        try {
            $start = Convert-ToSeconds $parts[0]
            $end   = Convert-ToSeconds $parts[1]
            $score = [int]$parts[2]
        }
        catch {
            continue
        }

        if ($end -le $start) {
            throw "Invalid annotation in $Path`: $text"
        }

        $annotations += [PSCustomObject]@{
            Start = $start
            End   = $end
            Score = $score
        }
    }

    return $annotations
}

# ------------------------------------------------------------------------------
# Read clip CSV
# ------------------------------------------------------------------------------

function Read-ClipCsv {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Path
    )

    $clips = @()

    foreach ($row in Import-Csv -LiteralPath $Path) {

        try {
            $start = [double]$row.start
            $end   = [double]$row.end
        }
        catch {
            throw "Invalid clip CSV row in $Path"
        }

        if ($end -le $start) {
            throw (
                "Invalid clip interval in $Path`: " +
                "start=$start end=$end"
            )
        }

        $clips += [PSCustomObject]@{
            Start = $start
            End   = $end
        }
    }

    return $clips
}

# ------------------------------------------------------------------------------
# Interval overlap
# ------------------------------------------------------------------------------

function Get-Overlap {
    param(
        [double]$Start1,
        [double]$End1,
        [double]$Start2,
        [double]$End2
    )

    $start = [Math]::Max($Start1, $Start2)
    $end   = [Math]::Min($End1, $End2)

    if ($end -gt $start) {
        return ($end - $start)
    }

    return 0.0
}

# ------------------------------------------------------------------------------
# Calculate metrics
# ------------------------------------------------------------------------------

function Evaluate-Selection {
    param(
        [array]$Clips,
        [array]$Annotations
    )

    $selectedSeconds = 0.0

    foreach ($clip in $Clips) {
        $selectedSeconds += ($clip.End - $clip.Start)
    }

    $mtb3Seconds  = 0.0
    $mtb23Seconds = 0.0
    $mtb0Seconds  = 0.0

    foreach ($clip in $Clips) {

        foreach ($annotation in $Annotations) {

            $overlap = Get-Overlap `
                $clip.Start `
                $clip.End `
                $annotation.Start `
                $annotation.End

            if ($overlap -le 0) {
                continue
            }

            if ($annotation.Score -eq 3) {
                $mtb3Seconds += $overlap
            }

            if ($annotation.Score -ge 2) {
                $mtb23Seconds += $overlap
            }

            if ($annotation.Score -eq 0) {
                $mtb0Seconds += $overlap
            }
        }
    }

    $totalMTB3Seconds  = 0.0
    $totalMTB23Seconds = 0.0
    $totalMTB0Seconds  = 0.0

    foreach ($annotation in $Annotations) {

        $duration = $annotation.End - $annotation.Start

        if ($annotation.Score -eq 3) {
            $totalMTB3Seconds += $duration
        }

        if ($annotation.Score -ge 2) {
            $totalMTB23Seconds += $duration
        }

        if ($annotation.Score -eq 0) {
            $totalMTB0Seconds += $duration
        }
    }

    $mtb3Recall = 0.0

    if ($totalMTB3Seconds -gt 0) {
        $mtb3Recall = $mtb3Seconds / $totalMTB3Seconds
    }

    $mtb3Precision = 0.0

    if ($selectedSeconds -gt 0) {
        $mtb3Precision = $mtb3Seconds / $selectedSeconds
    }

    $mtb23Precision = 0.0

    if ($selectedSeconds -gt 0) {
        $mtb23Precision = $mtb23Seconds / $selectedSeconds
    }

    $mtb0Contamination = 0.0

    if ($selectedSeconds -gt 0) {
        $mtb0Contamination = $mtb0Seconds / $selectedSeconds
    }

    return [PSCustomObject]@{
        SelectedSeconds          = $selectedSeconds
        ClipCount                = $Clips.Count

        MTB3Seconds              = $mtb3Seconds
        MTB3TotalSeconds         = $totalMTB3Seconds
        MTB3Recall               = $mtb3Recall
        MTB3Precision            = $mtb3Precision

        MTB23Seconds             = $mtb23Seconds
        MTB23TotalSeconds        = $totalMTB23Seconds
        MTB23Precision           = $mtb23Precision

        MTB0Seconds              = $mtb0Seconds
        MTB0Contamination        = $mtb0Contamination
    }
}

# ------------------------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------------------------

$results = @()

Write-Host ""
Write-Host ("=" * 80)
Write-Host "LOVO EXTRACTOR EVALUATION — 2 FPS vs 10 FPS"
Write-Host ("=" * 80)
Write-Host ""

foreach ($video in $Videos) {

    $annotationsPath = Join-Path `
        $AnnotationDir `
        ("{0}.txt" -f $video)

    if (-not (Test-Path $annotationsPath)) {
        Write-Warning "Annotations not found: $annotationsPath"
        continue
    }

    $annotationData = Read-Annotations $annotationsPath

    if ($annotationData.Count -eq 0) {
        throw "No annotations found for $video"
    }

    foreach ($fps in $FPSValues) {

        Write-Host ""
        Write-Host ("-" * 80)
        Write-Host "EVALUATING: $video — $fps"
        Write-Host ("-" * 80)

        $clipCsv = Join-Path `
            $ExtractionDir `
            ("{0}_{1}.csv" -f $video.ToLower(), $fps)

        if (-not (Test-Path $clipCsv)) {
            Write-Warning "Clip CSV not found: $clipCsv"
            continue
        }

        Write-Host "Clips:       $clipCsv"
        Write-Host "Annotations: $annotationsPath"
        Write-Host ""

        $clips = Read-ClipCsv $clipCsv

        if ($clips.Count -eq 0) {
            throw "No clips found in $clipCsv"
        }

        $metrics = Evaluate-Selection `
            -Clips $clips `
            -Annotations $annotationData

        $result = [PSCustomObject]@{
            Video = $video
            FPS   = $fps

            SelectedSeconds = [Math]::Round(
                $metrics.SelectedSeconds,
                2
            )

            ClipCount = $metrics.ClipCount

            MTB3Seconds = [Math]::Round(
                $metrics.MTB3Seconds,
                2
            )

            MTB3TotalSeconds = [Math]::Round(
                $metrics.MTB3TotalSeconds,
                2
            )

            MTB3Recall = [Math]::Round(
                $metrics.MTB3Recall * 100.0,
                2
            )

            MTB3Precision = [Math]::Round(
                $metrics.MTB3Precision * 100.0,
                2
            )

            MTB23Seconds = [Math]::Round(
                $metrics.MTB23Seconds,
                2
            )

            MTB23TotalSeconds = [Math]::Round(
                $metrics.MTB23TotalSeconds,
                2
            )

            MTB23Precision = [Math]::Round(
                $metrics.MTB23Precision * 100.0,
                2
            )

            MTB0Seconds = [Math]::Round(
                $metrics.MTB0Seconds,
                2
            )

            MTB0Contamination = [Math]::Round(
                $metrics.MTB0Contamination * 100.0,
                2
            )
        }

        $results += $result

        Write-Host "RESULT:"
        Write-Host ("  Selected:        {0:F1}s" -f `
            $metrics.SelectedSeconds)
        Write-Host ("  Clips:           {0}" -f `
            $metrics.ClipCount)
        Write-Host ("  MTB3 captured:   {0:F1}s / {1:F1}s" -f `
            $metrics.MTB3Seconds,
            $metrics.MTB3TotalSeconds)
        Write-Host ("  MTB3 recall:     {0:F1}%" -f `
            ($metrics.MTB3Recall * 100.0))
        Write-Host ("  MTB3 precision:  {0:F1}%" -f `
            ($metrics.MTB3Precision * 100.0))
        Write-Host ("  MTB2+3 captured: {0:F1}s / {1:F1}s" -f `
            $metrics.MTB23Seconds,
            $metrics.MTB23TotalSeconds)
        Write-Host ("  MTB2+3 precision:{0:F1}%" -f `
            ($metrics.MTB23Precision * 100.0))
        Write-Host ("  MTB0 contam.:    {0:F1}%" -f `
            ($metrics.MTB0Contamination * 100.0))
    }
}

# ------------------------------------------------------------------------------
# Detailed summary
# ------------------------------------------------------------------------------

Write-Host ""
Write-Host ""
Write-Host ("=" * 80)
Write-Host "LOVO EXTRACTOR SUMMARY"
Write-Host ("=" * 80)
Write-Host ""

$results |
    Sort-Object Video, FPS |
    Format-Table `
        Video,
        FPS,
        SelectedSeconds,
        ClipCount,
        MTB3Seconds,
        MTB3Recall,
        MTB3Precision,
        MTB23Precision,
        MTB0Contamination `
        -AutoSize

# ------------------------------------------------------------------------------
# Aggregate statistics
# ------------------------------------------------------------------------------

if ($results.Count -gt 0) {

    foreach ($fps in $FPSValues) {

        $fpsResults = @(
            $results | Where-Object {
                $_.FPS -eq $fps
            }
        )

        if ($fpsResults.Count -eq 0) {
            continue
        }

        $avgSelected = (
            $fpsResults |
                Measure-Object SelectedSeconds -Average
        ).Average

        $avgMTB3Recall = (
            $fpsResults |
                Measure-Object MTB3Recall -Average
        ).Average

        $avgMTB3Precision = (
            $fpsResults |
                Measure-Object MTB3Precision -Average
        ).Average

        $avgMTB23Precision = (
            $fpsResults |
                Measure-Object MTB23Precision -Average
        ).Average

        $avgMTB0 = (
            $fpsResults |
                Measure-Object MTB0Contamination -Average
        ).Average

        Write-Host ""
        Write-Host "MEAN — $fps"
        Write-Host "---------------------------"
        Write-Host ("Mean selected duration:   {0:F1}s" -f `
            $avgSelected)
        Write-Host ("Mean MTB3 recall:         {0:F1}%" -f `
            $avgMTB3Recall)
        Write-Host ("Mean MTB3 precision:      {0:F1}%" -f `
            $avgMTB3Precision)
        Write-Host ("Mean MTB2+3 precision:    {0:F1}%" -f `
            $avgMTB23Precision)
        Write-Host ("Mean MTB0 contamination:  {0:F1}%" -f `
            $avgMTB0)
    }
}

# ------------------------------------------------------------------------------
# CSV output
# ------------------------------------------------------------------------------

$outputCsv = Join-Path `
    $RepoRoot `
    "output\lovo_extractor_evaluation.csv"

$results |
    Export-Csv `
        -LiteralPath $outputCsv `
        -NoTypeInformation `
        -Encoding UTF8

Write-Host ""
Write-Host "Results CSV:"
Write-Host "  $outputCsv"
Write-Host ""
Write-Host "EVALUATION COMPLETE"