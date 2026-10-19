param(
    [string]$Binary = "$PSScriptRoot/../build-cpu/cairn.exe",
    [string]$Baseline = "",
    [int]$Repeats = 3,
    [string]$Output = "$PSScriptRoot/../build-cpu/benchmark.csv"
)
$ErrorActionPreference = 'Stop'
if ($Repeats -lt 1) { throw 'Repeats must be positive' }
$taskCases = @(
    @{Tokens=192; Dim=32; Backend='auto'; Weight='f32'},
    @{Tokens=64; Dim=128; Backend='auto'; Weight='f32'},
    @{Tokens=64; Dim=128; Backend='scalar'; Weight='f32'},
    @{Tokens=64; Dim=128; Backend='auto'; Weight='fp8'},
    @{Tokens=64; Dim=128; Backend='scalar'; Weight='fp8'},
    @{Tokens=64; Dim=128; Backend='auto'; Weight='fp4'},
    @{Tokens=64; Dim=128; Backend='scalar'; Weight='fp4'}
)
$taskRows = foreach ($taskRound in 1..$Repeats) {
    foreach ($taskCase in $taskCases) {
        $taskPrograms = @(@{Path=$Binary; Name='runtime'})
        if ($Baseline -and $taskCase.Backend -eq 'auto' -and $taskCase.Weight -eq 'f32') {
            $taskPrograms += @{Path=$Baseline; Name='before'}
        }
        # Alternate order to reduce systematic warm-up/thermal bias.
        if ($taskRound % 2 -eq 0) { [array]::Reverse($taskPrograms) }
        foreach ($taskProgram in $taskPrograms) {
            $taskArguments = @('bench', '1', [string]$taskCase.Tokens, [string]$taskCase.Dim)
            if ($taskProgram.Name -eq 'runtime') {
                $taskArguments += @($taskCase.Backend, $taskCase.Weight)
            }
            $taskText = (& $taskProgram.Path @taskArguments) -join "`n"
            if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $taskText" }
            $taskRecord = [ordered]@{
                round=$taskRound; program=$taskProgram.Name; dim=$taskCase.Dim;
                tokens=$taskCase.Tokens; backend=$taskCase.Backend; weights=$taskCase.Weight
            }
            foreach ($taskMetric in @('prefill', 'decode', 'train', 'ced_exact_prefill', 'ced_bounded_prefill')) {
                if ($taskText -notmatch "(?m)(?:^| )$taskMetric=([0-9.]+)") { throw "Missing $taskMetric" }
                $taskRecord[$taskMetric] = [double]::Parse($Matches[1], [cultureinfo]::InvariantCulture)
            }
            [pscustomobject]$taskRecord
        }
    }
}
$taskRows | Export-Csv -LiteralPath $Output -NoTypeInformation -Encoding utf8
$taskRows | Format-Table -AutoSize
Write-Output "Saved $Output"
