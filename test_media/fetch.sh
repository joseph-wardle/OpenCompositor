#!/bin/sh
set -eu
cd "$(dirname "$0")"

usage() {
    cat >&2 <<'END'
usage: fetch.sh SET...

  solemates_comp    Sole Mates, the finished comp: 61 frames, 2048x1024 (431 MB)
  solemates_frame   Sole Mates, one frame of every render layer and the Nuke script (1.7 GB)
  tears_of_steel    Tears of Steel, shot 04_5f: 150 frames, 4096x2160 (8 GB)

Downloading a set accepts its licence, which README.md names.
END
    exit 2
}

archive() {
    into=$1 unpacked=$2 url=$3 sum=$4
    if [ -d "$into/$unpacked" ]; then
        echo "$into/$unpacked is already here"
        return
    fi
    mkdir -p "$into"
    rm -rf "$into/unpacking"
    curl -fL --retry 3 -o "$into/download.zip" "$url"
    echo "$sum  $into/download.zip" | sha256sum -c -
    unzip -q "$into/download.zip" -d "$into/unpacking"
    mv "$into/unpacking/$unpacked" "$into/"
    rmdir "$into/unpacking"
    rm "$into/download.zip"
}

tears_of_steel() {
    shot=tears_of_steel/04_5f
    url=https://media.xiph.org/tearsofsteel/linear-exr/04_5f
    mkdir -p "$shot"
    frame=0
    while [ "$frame" -lt 150 ]; do
        printf '04_5f_%05d.exr\n' "$frame"
        frame=$((frame + 1))
    done | xargs -P 4 -I{} sh -c '
        [ -f "$0/{}" ] && exit
        curl -fsS --retry 3 -o "$0/{}.part" "$1/{}" && mv "$0/{}.part" "$0/{}"
    ' "$shot" "$url"
    sha256sum -c --quiet tears_of_steel.sha256
}

[ $# -gt 0 ] || usage
for set in "$@"; do
    case $set in
    solemates_comp)
        archive solemates exr_sequences-v1.0.0 \
            https://dpel-assets.aswf.io/solemates/exr_sequences-v1.0.0.zip \
            3a6faaa863d3f527fc0ff5f47bfa843aa2954e9674c22de32287fa1e93f0b78d
        ;;
    solemates_frame)
        archive solemates SoleMates-1.0.0 \
            https://github.com/DigitalProductionExampleLibrary/SoleMates/archive/refs/tags/v1.0.0.zip \
            2953b8fd6eb5183463c4ccd77870968fe80c0b44a1544530059ff5f8bfc48236
        ;;
    tears_of_steel) tears_of_steel ;;
    *) usage ;;
    esac
    echo "$set: fetched"
done
