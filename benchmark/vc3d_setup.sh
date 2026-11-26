#!/bin/bash
# VC3D (for GrowPatch: vc_grow_seg_from_seed) on Linux x86_64 without FUSE or Docker: the release AppImage of
# ScrollPrize/villa the benchmark used (VC3D-e5707e4-2026-11-23, kept under its archive tag; VC3D_URL=<AppImage URL>
# for another, or VC3D_URL=newest), extracted, and the system libraries its README lists (glibc >= 2.31).
#   bash vc3d_setup.sh DIR       -> DIR/squashfs-root/AppRun
set -euo pipefail
D=${1:-vc3d}; mkdir -p "$D"; cd "$D"
if [ ! -x squashfs-root/AppRun ]; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=sudo
  timeout 300 $SUDO apt-get update -qq > /dev/null 2>&1 || echo "apt-get update failed or timed out"
  DEBIAN_FRONTEND=noninteractive timeout 300 $SUDO apt-get install -y -qq --no-install-recommends libglvnd0 libegl1 \
    libglx0 libopengl0 libgl1 libx11-6 libx11-xcb1 libxcb1 libice6 libsm6 libfontconfig1 libfreetype6 libharfbuzz0b \
    > /dev/null 2>&1 || echo "apt-get install: some libraries may be missing"
  URL=${VC3D_URL:-https://github.com/ScrollPrize/villa/releases/download/vc3d-archive-latest-454931750/VC3D-e5707e4-2026-11-23-linux-x86_64.AppImage}
  TAG=pinned
  [ "$URL" = newest ] || curl -fsIL -o /dev/null "$URL" || { echo "$URL is no longer published: using the newest release"; URL=newest; }
  [ "$URL" != newest ] || read -r TAG URL < <(curl -fsSL "https://api.github.com/repos/ScrollPrize/villa/releases?per_page=30" | python3 -c '
import json, sys
for r in json.load(sys.stdin):
    for a in r.get("assets", []):
        if a["name"].endswith(".AppImage"):
            print(r["tag_name"], a["browser_download_url"]); sys.exit()
sys.exit("no AppImage among the releases")')
  echo "VC3D release $TAG: $URL" | tee release.txt
  curl -fsSL -o VC3D.AppImage "$URL"; chmod +x VC3D.AppImage
  ./VC3D.AppImage --appimage-extract > /dev/null
fi
./squashfs-root/AppRun vc_grow_seg_from_seed --help 2>&1 | head -3 || true
echo "VC3D_READY $(pwd)/squashfs-root/AppRun"
