# detect_amd_gpu.ps1 - prints one whitelisted TheRock target or nothing.
# Uses actual Win32 display adapter names. No CUDA/HIP installation required.
# Prefers a recognized discrete Radeon RX over any integrated AMD graphics.
param([string[]]$TestNames)
$ErrorActionPreference = 'Stop'
if ($TestNames -and $TestNames.Count -gt 0) {
    $names = @($TestNames)
} else {
    try {
        $names = @(Get-CimInstance Win32_VideoController | ForEach-Object { [string]$_.Name })
    } catch {
        exit 2
    }
}

# Explicit and conservative mapping, based on AMD's TheRock device packages.
# Supported architectures may still fail to install or run; installer tests HIP.
$rules = @(
    @{ Pattern = '\bRX\s*9070(?:\s*XT)?\b';                    Gfx = 'gfx1201'; Score = 9070 },
    @{ Pattern = '\bRX\s*9060(?:\s*XT)?\b';                    Gfx = 'gfx1200'; Score = 9060 },
    @{ Pattern = '\bRX\s*7900\s*(?:XTX|XT|GRE)\b';             Gfx = 'gfx1100'; Score = 7900 },
    @{ Pattern = '\bRX\s*(?:7800|7700)\s*XT\b';                Gfx = 'gfx1101'; Score = 7800 },
    @{ Pattern = '\bRX\s*7600(?:\s*XT)?\b';                    Gfx = 'gfx1102'; Score = 7600 },
    @{ Pattern = '\bRX\s*(?:6950|6900|6800)(?:\s*XT)?\b';     Gfx = 'gfx1030'; Score = 6900 },
    @{ Pattern = '\bRX\s*(?:6750|6700)(?:\s*XT)?\b';           Gfx = 'gfx1031'; Score = 6700 },
    @{ Pattern = '\bRX\s*(?:6650|6600)(?:\s*XT)?\b';           Gfx = 'gfx1032'; Score = 6600 },
    @{ Pattern = '\bRX\s*6500\s*XT\b';                         Gfx = 'gfx1034'; Score = 6500 },
    @{ Pattern = '\bRX\s*5700(?:\s*XT)?\b';                    Gfx = 'gfx1010'; Score = 5700 }
)

$candidates = @()
foreach ($name in $names) {
    if ($name -notmatch 'AMD|Radeon') { continue }
    foreach ($rule in $rules) {
        if ($name -match $rule.Pattern) {
            $candidates += [pscustomobject]@{ Name = $name; Gfx = $rule.Gfx; Score = $rule.Score }
            break
        }
    }
}
if ($candidates.Count -eq 0) { exit 1 }
$best = $candidates | Sort-Object Score -Descending | Select-Object -First 1
[Console]::Out.WriteLine([string]$best.Gfx)
exit 0
