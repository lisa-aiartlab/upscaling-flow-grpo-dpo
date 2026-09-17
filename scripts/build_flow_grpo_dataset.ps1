param(
    [string]$DataDirectory = (Join-Path $PSScriptRoot "..\data"),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\flow_grpo_dataset")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Drawing

$dataRoot = [IO.Path]::GetFullPath($DataDirectory)
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)

if (-not (Test-Path -LiteralPath $dataRoot -PathType Container)) {
    throw "Data directory does not exist: $dataRoot"
}
if (Test-Path -LiteralPath $outputRoot) {
    throw "Output directory already exists: $outputRoot"
}

$preferredDirectory = Join-Path $outputRoot "preferred"
$lrDirectory = Join-Path $outputRoot "lr"
New-Item -ItemType Directory -Path $preferredDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $lrDirectory -Force | Out-Null

$prompts = @{
    "15" = "An eighteenth-century decorative painted bouquet with freely arranged flowers, simplified painterly forms, visible brushstrokes and ample background"
    "23" = "An eighteenth-century decorative painted panel of a stylized bird perched on a rose branch, simplified forms and visible painterly brushstrokes"
    "26" = "An eighteenth-century decorative landscape panel with a river, rocks, castle ruins, a distant windmill and a muted green mountain, simplified painterly forms"
    "27" = "An eighteenth-century decorative painted bouquet of flowers with butterflies, simplified forms, visible brushstrokes and ample background"
    "33" = "An eighteenth-century decorative painted panel of a young woman holding a pitcher, flat stylized forms, terracotta tones and broad brushstrokes"
    "35" = "An eighteenth-century decorative hunting scene with deer and running dogs, simplified painterly forms, broad brushstrokes and ample background"
    "36" = "An eighteenth-century decorative hunting scene with a deer, spotted hunting dogs, a tree canopy and sky, simplified painterly forms"
    "44" = "An eighteenth-century decorative painted panel of a bullfinch perched on a branch beside a rose, simplified forms and visible brushstrokes"
    "52" = "An eighteenth-century decorative pastoral landscape with cattle, a shepherd, trees and a ruined stone tower, simplified forms and broad painterly brushstrokes"
    "53" = "An eighteenth-century decorative landscape with a waterfall, trees, a small distant house and a classical pavilion, warm yellow tones and layered scenery"
}

function Get-ArtworkPrompt([string]$ArtworkId) {
    if (-not $prompts.ContainsKey($ArtworkId)) {
        throw "No curated prompt for artwork $ArtworkId"
    }
    return $prompts[$ArtworkId]
}

function Get-ZipEntry([IO.Compression.ZipArchive]$Archive, [string]$EntryName) {
    $entry = $Archive.GetEntry($EntryName)
    if ($null -eq $entry) {
        throw "Archive entry not found: $EntryName"
    }
    return $entry
}

function Read-ZipEntryText([IO.Compression.ZipArchiveEntry]$Entry) {
    $stream = $Entry.Open()
    try {
        $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $true)
        try {
            return $reader.ReadToEnd()
        }
        finally {
            $reader.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
}

function Get-CornerBackground([Drawing.Bitmap]$Bitmap) {
    $points = @(
        @(0, 0),
        @([int]($Bitmap.Width - 1), 0),
        @(0, [int]($Bitmap.Height - 1)),
        @([int]($Bitmap.Width - 1), [int]($Bitmap.Height - 1))
    )
    $red = 0
    $green = 0
    $blue = 0
    foreach ($point in $points) {
        $color = $Bitmap.GetPixel($point[0], $point[1])
        $red += $color.R
        $green += $color.G
        $blue += $color.B
    }
    return [Drawing.Color]::FromArgb(
        [int]($red / $points.Count),
        [int]($green / $points.Count),
        [int]($blue / $points.Count)
    )
}

function Set-HighQualityGraphics([Drawing.Graphics]$Graphics) {
    $Graphics.CompositingMode = [Drawing.Drawing2D.CompositingMode]::SourceCopy
    $Graphics.CompositingQuality = [Drawing.Drawing2D.CompositingQuality]::HighQuality
    $Graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $Graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::HighQuality
    $Graphics.PixelOffsetMode = [Drawing.Drawing2D.PixelOffsetMode]::HighQuality
}

function Save-NormalizedPair(
    [IO.Compression.ZipArchiveEntry]$Entry,
    [string]$PreferredPath,
    [string]$LowResolutionPath
) {
    $stream = $Entry.Open()
    try {
        $sourceImage = [Drawing.Image]::FromStream($stream, $true, $true)
        try {
            $sourceBitmap = [Drawing.Bitmap]::new($sourceImage)
            try {
                $background = Get-CornerBackground $sourceBitmap
                $preferred = [Drawing.Bitmap]::new(
                    480,
                    480,
                    [Drawing.Imaging.PixelFormat]::Format24bppRgb
                )
                try {
                    $graphics = [Drawing.Graphics]::FromImage($preferred)
                    try {
                        Set-HighQualityGraphics $graphics
                        $graphics.Clear($background)
                        # Match ImageOps.fit in flow_grpo.py: fill a square and crop
                        # equally from the two long sides instead of learning bars.
                        $scale = [Math]::Max(
                            480.0 / $sourceBitmap.Width,
                            480.0 / $sourceBitmap.Height
                        )
                        $width = [Math]::Max(1, [int][Math]::Round($sourceBitmap.Width * $scale))
                        $height = [Math]::Max(1, [int][Math]::Round($sourceBitmap.Height * $scale))
                        $x = [int][Math]::Floor((480 - $width) / 2.0)
                        $y = [int][Math]::Floor((480 - $height) / 2.0)
                        $graphics.DrawImage($sourceBitmap, $x, $y, $width, $height)
                    }
                    finally {
                        $graphics.Dispose()
                    }
                    $preferred.Save($PreferredPath, [Drawing.Imaging.ImageFormat]::Png)

                    $lowResolution = [Drawing.Bitmap]::new(
                        120,
                        120,
                        [Drawing.Imaging.PixelFormat]::Format24bppRgb
                    )
                    try {
                        $lrGraphics = [Drawing.Graphics]::FromImage($lowResolution)
                        try {
                            Set-HighQualityGraphics $lrGraphics
                            $lrGraphics.DrawImage($preferred, 0, 0, 120, 120)
                        }
                        finally {
                            $lrGraphics.Dispose()
                        }
                        $lowResolution.Save(
                            $LowResolutionPath,
                            [Drawing.Imaging.ImageFormat]::Png
                        )
                    }
                    finally {
                        $lowResolution.Dispose()
                    }
                }
                finally {
                    $preferred.Dispose()
                }

                return [ordered]@{
                    source_width = $sourceBitmap.Width
                    source_height = $sourceBitmap.Height
                }
            }
            finally {
                $sourceBitmap.Dispose()
            }
        }
        finally {
            $sourceImage.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

$sourceRecords = [Collections.Generic.List[object]]::new()
$reactionRecords = [Collections.Generic.List[object]]::new()
$reactionArchivePath = Join-Path $dataRoot "like_dislike_meetings.zip"
$reactionArchive = [IO.Compression.ZipFile]::OpenRead($reactionArchivePath)
try {
    foreach ($csvEntry in $reactionArchive.Entries | Where-Object { $_.FullName.EndsWith("/reactions.csv") }) {
        $csvRows = Read-ZipEntryText $csvEntry | ConvertFrom-Csv
        $parent = $csvEntry.FullName.Substring(0, $csvEntry.FullName.LastIndexOf("/"))
        $segments = $parent.Split("/")
        $meetingDate = $segments[1]
        $artworkId = $segments[2]

        foreach ($row in $csvRows) {
            $entryName = "$parent/$($row.image_path)"
            $reactionRecords.Add([ordered]@{
                meeting_date = $meetingDate
                artwork_id = $artworkId
                reaction = $row.reaction
                comment = $row.comment
                original_filename = $row.original_filename
                source_archive = [IO.Path]::GetFileName($reactionArchivePath)
                source_entry = $entryName
            })
            if ($row.reaction -ne "like") {
                continue
            }
            $sourceRecords.Add([ordered]@{
                archive_path = $reactionArchivePath
                archive_name = [IO.Path]::GetFileName($reactionArchivePath)
                entry_name = $entryName
                meeting_date = $meetingDate
                artwork_id = $artworkId
                original_filename = $row.original_filename
                selection = "like"
                selection_reason = "Positive reaction in reactions.csv"
                annotation_comment = $row.comment
            })
        }
    }
}
finally {
    $reactionArchive.Dispose()
}

# These variants predate the reaction CSV export but are explicitly accepted in
# the 14 August expert protocol. Ambiguous recommendations are intentionally not used.
$august14ArchivePath = Join-Path $dataRoot "Раб. сессия 14 августа.zip"
$august14Archive = [IO.Compression.ZipFile]::OpenRead($august14ArchivePath)
try {
    $accepted14 = @(
        @{ suffix = "/26(6).png"; artwork_id = "26"; reason = "Expert protocol: variant 6 recommended and accepted" },
        @{ suffix = "/35_1.jpg"; artwork_id = "35"; reason = "Expert protocol: presented solution evaluated positively" },
        @{ suffix = "/36_1.jpg"; artwork_id = "36"; reason = "Expert protocol: variant 1 named most convincing" },
        @{ suffix = "/44 (1).png"; artwork_id = "44"; reason = "Expert protocol: variant 1 selected for further work" }
    )
    foreach ($accepted in $accepted14) {
        $matches = @($august14Archive.Entries | Where-Object { $_.FullName.EndsWith($accepted.suffix) })
        if ($matches.Count -ne 1) {
            throw "Expected one entry ending in $($accepted.suffix), found $($matches.Count)"
        }
        $sourceRecords.Add([ordered]@{
            archive_path = $august14ArchivePath
            archive_name = [IO.Path]::GetFileName($august14ArchivePath)
            entry_name = $matches[0].FullName
            meeting_date = "2026-08-14"
            artwork_id = $accepted.artwork_id
            original_filename = [IO.Path]::GetFileName($matches[0].FullName)
            selection = "expert_accepted"
            selection_reason = $accepted.reason
            annotation_comment = ""
        })
    }
}
finally {
    $august14Archive.Dispose()
}

$manifest = [Collections.Generic.List[object]]::new()
$metadata = [Collections.Generic.List[object]]::new()
$openArchives = @{}

try {
    $index = 0
    foreach ($sourceRecord in $sourceRecords) {
        $index += 1
        $archivePath = $sourceRecord.archive_path
        if (-not $openArchives.ContainsKey($archivePath)) {
            $openArchives[$archivePath] = [IO.Compression.ZipFile]::OpenRead($archivePath)
        }
        $entry = Get-ZipEntry $openArchives[$archivePath] $sourceRecord.entry_name
        $stem = "{0:D4}_artwork_{1}_{2}" -f $index, $sourceRecord.artwork_id, $sourceRecord.selection
        $preferredName = "$stem.png"
        $lrName = "$stem.png"
        $preferredPath = Join-Path $preferredDirectory $preferredName
        $lrPath = Join-Path $lrDirectory $lrName
        $dimensions = Save-NormalizedPair $entry $preferredPath $lrPath
        $prompt = Get-ArtworkPrompt $sourceRecord.artwork_id

        $manifest.Add([ordered]@{
            id = $index
            lr_path = "lr/$lrName"
            pref_generated_path = "preferred/$preferredName"
            prompt = $prompt
        })
        $metadata.Add([ordered]@{
            id = $index
            artwork_id = $sourceRecord.artwork_id
            meeting_date = $sourceRecord.meeting_date
            selection = $sourceRecord.selection
            selection_reason = $sourceRecord.selection_reason
            annotation_comment = $sourceRecord.annotation_comment
            prompt = $prompt
            source_archive = $sourceRecord.archive_name
            source_entry = $sourceRecord.entry_name
            original_filename = $sourceRecord.original_filename
            source_width = $dimensions.source_width
            source_height = $dimensions.source_height
            normalization = "aspect-ratio-preserving center crop matching ImageOps.fit"
            preferred_path = "preferred/$preferredName"
            lr_path = "lr/$lrName"
        })
    }
}
finally {
    foreach ($archive in $openArchives.Values) {
        $archive.Dispose()
    }
}

Write-Utf8NoBom (Join-Path $outputRoot "manifest.json") ($manifest | ConvertTo-Json -Depth 8)
Write-Utf8NoBom (Join-Path $outputRoot "metadata.json") ($metadata | ConvertTo-Json -Depth 8)

$metadataLines = foreach ($record in $metadata) {
    $record | ConvertTo-Json -Depth 8 -Compress
}
Write-Utf8NoBom (Join-Path $outputRoot "metadata.jsonl") (($metadataLines -join [Environment]::NewLine) + [Environment]::NewLine)

$reactionLines = foreach ($record in $reactionRecords) {
    $record | ConvertTo-Json -Depth 8 -Compress
}
Write-Utf8NoBom (Join-Path $outputRoot "reaction_annotations.jsonl") (($reactionLines -join [Environment]::NewLine) + [Environment]::NewLine)

$likeCount = @($reactionRecords | Where-Object { $_.reaction -eq "like" }).Count
$dislikeCount = @($reactionRecords | Where-Object { $_.reaction -eq "dislike" }).Count

$contactColumns = 4
$contactCellWidth = 250
$contactCellHeight = 250
$contactRows = [int][Math]::Ceiling($metadata.Count / $contactColumns)
$contactSheet = [Drawing.Bitmap]::new(
    $contactColumns * $contactCellWidth,
    $contactRows * $contactCellHeight,
    [Drawing.Imaging.PixelFormat]::Format24bppRgb
)
try {
    $contactGraphics = [Drawing.Graphics]::FromImage($contactSheet)
    try {
        $contactGraphics.Clear([Drawing.Color]::White)
        $contactGraphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $contactFont = [Drawing.Font]::new("Arial", 10)
        try {
            for ($contactIndex = 0; $contactIndex -lt $metadata.Count; $contactIndex++) {
                $record = $metadata[$contactIndex]
                $x = ($contactIndex % $contactColumns) * $contactCellWidth
                $y = [int][Math]::Floor($contactIndex / $contactColumns) * $contactCellHeight
                $imagePath = Join-Path $outputRoot $record.preferred_path
                $image = [Drawing.Image]::FromFile($imagePath)
                try {
                    $contactGraphics.DrawImage($image, $x + 5, $y + 5, 220, 220)
                }
                finally {
                    $image.Dispose()
                }
                $label = "{0:D2} | art {1} | {2}" -f [int]$record.id, $record.artwork_id, $record.selection
                $contactGraphics.DrawString(
                    $label,
                    $contactFont,
                    [Drawing.Brushes]::Black,
                    $x + 5,
                    $y + 228
                )
            }
        }
        finally {
            $contactFont.Dispose()
        }
    }
    finally {
        $contactGraphics.Dispose()
    }
    $contactSheet.Save(
        (Join-Path $outputRoot "contact_sheet.jpg"),
        [Drawing.Imaging.ImageFormat]::Jpeg
    )
}
finally {
    $contactSheet.Dispose()
}

$readme = @"
# Flow-GRPO training dataset

This dataset was built from the meeting archives in `data/`.

- `manifest.json`: directly accepted by `scripts/training_scripts/flow_grpo.py`
- `lr/`: 120x120 conditioning images
- `preferred/`: aligned 480x480 positively rated or expert-accepted references
- `metadata.json` and `metadata.jsonl`: provenance and selection rationale
- `reaction_annotations.jsonl`: all source likes and dislikes, including excluded images
- `contact_sheet.jpg`: visual quality-control overview of selected references

Only images marked `like` in reaction CSVs and four unambiguous selections from the
14 August expert protocol are included. Dislikes and ambiguous recommendations were
excluded because the current Flow-GRPO loader has no negative-image field.

Non-square source artwork is center-cropped with the same fill-and-crop behavior used
by `ImageOps.fit` in the trainer. Each LR image is downsampled from its exact preferred
reference, so every pair is spatially aligned and contains no synthetic letterbox bars.

Records: $($manifest.Count)
Reaction annotations: $likeCount likes and $dislikeCount dislikes
"@
Write-Utf8NoBom (Join-Path $outputRoot "README.md") $readme

Write-Output "Created $($manifest.Count) records in $outputRoot"
