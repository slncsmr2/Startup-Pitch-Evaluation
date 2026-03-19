$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$targetRoot = Join-Path $root "notebooks"
$outputPath = Join-Path $targetRoot "startup_pitch_evaluation_all.ipynb"

if (Test-Path $targetRoot) {
    Remove-Item -Path $targetRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $targetRoot | Out-Null

function Get-LanguageForExtension {
    param ([string]$extension)

    $ext = $extension.ToLowerInvariant()
    switch ($ext) {
        ".py" { return "python" }
        ".ps1" { return "powershell" }
        ".json" { return "json" }
        ".yaml" { return "yaml" }
        ".yml" { return "yaml" }
        ".md" { return "markdown" }
        ".txt" { return "markdown" }
        ".toml" { return "toml" }
        ".ini" { return "ini" }
        ".cfg" { return "ini" }
        ".gitignore" { return "text" }
        ".gitattributes" { return "text" }
        ".csv" { return "csv" }
        ".tsv" { return "text" }
        ".log" { return "text" }
        default { return "text" }
    }
}

function Is-IncludedTextFile {
    param ([System.IO.FileInfo]$file)

    $name = $file.Name.ToLowerInvariant()
    $ext = $file.Extension.ToLowerInvariant()
    $allowByName = @("license", "readme", "plan.md", "missing.md")
    $allowByExtension = @(
        ".py", ".ps1", ".json", ".yaml", ".yml", ".md", ".txt", ".toml", ".ini", ".cfg", ".csv", ".tsv", ".log"
    )

    if ($allowByName -contains $name) {
        return $true
    }

    return $allowByExtension -contains $ext
}

function Get-NotebookSourceLines {
    param ([string]$content)

    $normalized = $content -replace "`r`n", "`n"
    if ([string]::IsNullOrEmpty($normalized)) {
        return @("`n")
    }

    $lines = $normalized -split "`n"
    $sourceLines = @()
    foreach ($line in $lines) {
        $sourceLines += ($line + "`n")
    }
    return $sourceLines
}

$excludedDirNames = @(".git", ".venv", "__pycache__", "notebooks", ".pytest_cache")

$candidateFiles = Get-ChildItem -Path $root -Recurse -File | Where-Object {
    $relative = $_.FullName.Substring($root.Length).TrimStart("\\")
    $segments = $relative.Split("\\")
    foreach ($segment in $segments) {
        if ($excludedDirNames -contains $segment.ToLowerInvariant()) {
            return $false
        }
    }
    return (Is-IncludedTextFile -file $_)
} | Sort-Object FullName

$cells = @()

$cells += [ordered]@{
    cell_type = "markdown"
    metadata = [ordered]@{ language = "markdown" }
    source = @(
        "# Startup Pitch Evaluation - Single Notebook Project`n",
        "This notebook contains a full project snapshot merged into one document.`n",
        "Each section starts with the source file path and then the file content.`n"
    )
}

foreach ($file in $candidateFiles) {
    $relativePath = $file.FullName.Substring($root.Length).TrimStart("\\") -replace "\\", "/"
    $extension = $file.Extension
    $language = Get-LanguageForExtension -extension $extension
    $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    $sourceLines = Get-NotebookSourceLines -content $content

    $cells += [ordered]@{
        cell_type = "markdown"
        metadata = [ordered]@{ language = "markdown" }
        source = @(
            "## File: $relativePath`n"
        )
    }

    if ($language -eq "markdown") {
        $cells += [ordered]@{
            cell_type = "markdown"
            metadata = [ordered]@{ language = "markdown" }
            source = $sourceLines
        }
    }
    else {
        $cells += [ordered]@{
            cell_type = "code"
            execution_count = $null
            metadata = [ordered]@{ language = $language }
            outputs = @()
            source = $sourceLines
        }
    }
}

$notebook = [ordered]@{
    cells = $cells
    metadata = [ordered]@{
        kernelspec = [ordered]@{
            display_name = "Python 3"
            language = "python"
            name = "python3"
        }
        language_info = [ordered]@{
            name = "python"
            version = "3.11"
        }
    }
    nbformat = 4
    nbformat_minor = 5
}

$json = $notebook | ConvertTo-Json -Depth 100
Set-Content -Path $outputPath -Value $json -Encoding UTF8

Write-Output ("Converted {0} files into single notebook: {1}" -f $candidateFiles.Count, $outputPath)
