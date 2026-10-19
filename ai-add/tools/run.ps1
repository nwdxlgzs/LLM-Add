param(
    [ValidateSet('all','build','data','train','qat','validate')][string]$Stage='all',
    [string]$Run='reproduce',
    [int]$Pretrain=200,
    [int]$Sft=3000,
    [int]$Rl=400,
    [int]$Batch=4,
    [int]$QatSteps=2000,
    [int]$QatBatch=8
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskBuild=Join-Path $taskRoot 'build'
$taskRun=Join-Path $taskRoot $Run
$taskExe=Join-Path $taskBuild 'ai_add.exe'
if ([IO.Path]::IsPathRooted($Run) -or $Run.Contains('..') -or $Run.Contains('/') -or $Run.Contains('\')) { throw 'Run must be a direct child directory name.' }
function Invoke-Checked([string]$Program,[string[]]$Parameters) {
    & $Program @Parameters
    if ($LASTEXITCODE -ne 0) { throw "$Program failed: exit code $LASTEXITCODE" }
}
if ($Stage -in @('all','build')) {
    Invoke-Checked 'cmake' @('-S',$taskRoot,'-B',$taskBuild,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release')
    Invoke-Checked 'cmake' @('--build',$taskBuild,'-j','4')
    Invoke-Checked 'ctest' @('--test-dir',$taskBuild,'--output-on-failure')
}
if ($Stage -in @('all','data')) { Invoke-Checked $taskExe @('prepare',$taskRun) }
if ($Stage -in @('all','train')) { Invoke-Checked $taskExe @('train',$taskRun,"$Pretrain","$Sft","$Rl","$Batch") }
if ($Stage -in @('all','qat')) {
    Invoke-Checked $taskExe @('qat',(Join-Path $taskRun 'final.cn'),$taskRun,"$QatSteps","$QatBatch")
}
if ($Stage -in @('all','validate')) {
    $taskModel=Join-Path $taskRun 'final.cn'
    Invoke-Checked $taskExe @('certify',$taskModel,(Join-Path $taskRun 'certificate-fp32.json'),'keep','all')
    Invoke-Checked $taskExe @('evaluate',$taskModel,(Join-Path $taskRun 'evaluation.json'),'10000')
    Invoke-Checked $taskExe @('add',$taskModel,'9999999999','9999999999')
    $taskFp8=Join-Path $taskRun 'final-fp8.cn'
    $taskEngineExe=Join-Path $taskBuild 'cairn/cairn.exe'
    Invoke-Checked $taskEngineExe @('quantize',$taskModel,$taskFp8,'fp8')
    Invoke-Checked $taskExe @('certify',$taskFp8,(Join-Path $taskRun 'certificate-fp8-file.json'),'keep','all')
    $taskFp4=Join-Path $taskRun 'final-fp4.cn'
    Invoke-Checked $taskExe @('certify',$taskFp4,(Join-Path $taskRun 'certificate-fp4-file.json'),'keep','all')
    Invoke-Checked $taskExe @('evaluate',$taskFp4,(Join-Path $taskRun 'evaluation-fp4.json'),'10000')
    Invoke-Checked $taskExe @('add',$taskFp4,'9999999999','9999999999')
    $taskFiles=@($taskExe,(Join-Path $taskRoot 'CMakeLists.txt'),$PSCommandPath)
    $taskFiles+=@(Get-ChildItem -LiteralPath $taskRun -Filter '*.cn' -File | Select-Object -ExpandProperty FullName)
    $taskFiles+=@(Get-ChildItem -LiteralPath (Join-Path $taskRoot 'src'),(Join-Path $taskRoot 'include') -File | Select-Object -ExpandProperty FullName)
    $taskEngine=Join-Path (Split-Path -Parent $taskRoot) 'cairn'
    $taskFiles+=@(Get-ChildItem -LiteralPath (Join-Path $taskEngine 'src'),(Join-Path $taskEngine 'include') -File | Select-Object -ExpandProperty FullName)
    Get-FileHash -Algorithm SHA256 -LiteralPath $taskFiles | Select-Object Path,Hash | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $taskRun 'sha256.json')
}
