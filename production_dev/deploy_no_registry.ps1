param(
    [Parameter(Mandatory = $true)]
    [string]$Server,

    [string]$Tag = "",
    [string]$TargetDir = "/home/user/GROM/ostrofun",
    [string]$SshKey = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Tag)) {
    $Tag = Get-Date -Format "yyyy.MM.dd-HHmm"
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$backendImage = "ostrofun-backend:$Tag"
$frontendImage = "ostrofun-frontend:$Tag"
$archiveName = "ostrofun-images-$Tag.tar"

Write-Host "[1/4] Build backend image: $backendImage"
docker build -f production_dev/Dockerfile -t $backendImage .

Write-Host "[2/4] Build frontend image: $frontendImage"
docker build -f production_dev/frontend/Dockerfile -t $frontendImage .

Write-Host "[3/4] Save images to archive: $archiveName"
docker save -o $archiveName $backendImage $frontendImage

$scpArgs = @("-r")
if (-not [string]::IsNullOrWhiteSpace($SshKey)) {
    $scpArgs += @("-i", $SshKey)
}

Write-Host "[4/4] Copy release files to ${Server}:$TargetDir"
$sources = @(
    "production_dev/docker-compose.deploy.yml",
    "production_dev/.env.deploy.example",
    "production_dev/server_apply_release.sh",
    "production_dev/security",
    $archiveName
)
& scp @scpArgs @sources "$Server`:$TargetDir/"

Write-Host ""
Write-Host "Done."
Write-Host "Run on server:"
Write-Host "ssh $Server"
Write-Host "cd $TargetDir"
Write-Host "chmod +x ./server_apply_release.sh"
Write-Host "./server_apply_release.sh $Tag"
