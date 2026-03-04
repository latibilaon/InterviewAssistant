# Packaging Notes

- macOS build app only: `../scripts/build_macos_app.sh`
- macOS package dmg only: `../scripts/package_macos_dmg.sh`
- macOS full flow: `../scripts/build_macos_dmg.sh`
- Windows: `../scripts/build_windows_exe.ps1`
- CI workflow: `../.github/workflows/build-release.yml`

Build outputs are written to `../release/`.
