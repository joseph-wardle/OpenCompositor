# Test media

Image sequences from real productions, for the compositor's `--drag-check` and `--playback-check` and for trying the app on real work. None of it is in the repository: `fetch.sh` downloads a set into this directory, where Git ignores it.

```sh
./test_media/fetch.sh                  # lists the sets
./test_media/fetch.sh tears_of_steel   # downloads one and checks it against its SHA-256
```

Downloading a set accepts its licence. The licences allow development and benchmarking; read them before using the media for anything else.

## Sets

| Set | Contents | Size | Path |
|---|---|---|---|
| `solemates_comp` | The finished comp of one shot: 61 frames (994 to 1054), 2048×1024, RGB half float, ZIP, ACES2065-1 | 431 MB | `solemates/exr_sequences-v1.0.0/unh0400_0010_lighting.####.exr` |
| `solemates_frame` | Frame 1037 of that shot's 42 render layers, with up to 40 passes and Cryptomatte, in ACEScg, and the Nuke script that composites them | 1.7 GB | `solemates/SoleMates-1.0.0/` |
| `tears_of_steel` | A live-action plate, shot `04_5f`: 150 frames (0 to 149), 4096×2160, RGB half float, uncompressed | 8 GB | `tears_of_steel/04_5f/04_5f_#####.exr` |

The two sequences differ on purpose. Sole Mates is clean, compressed computer graphics, where decoding dominates. The Tears of Steel plate is uncompressed camera footage with grain, 53 MB a frame, where storage dominates.

## Sources and licences

- **Sole Mates** is Netflix Animation Studios' HDR production example, built on the ALab scene and published in the Academy Software Foundation's [Digital Production Example Library](https://dpel.aswf.io/solemates). Copyright 2025 Netflix, Inc., under the [ASWF Digital Assets License v1.1](https://dpel.aswf.io/solemates/solemates-license). Its other downloads, up to every render layer as a sequence (105 GB), are listed there.
- **Tears of Steel** plates are from the Blender Foundation's open movie, mirrored at [media.xiph.org](https://media.xiph.org/tearsofsteel/), under [Creative Commons Attribution 3.0](http://creativecommons.org/licenses/by/3.0/). Credit: (CC) Blender Foundation | mango.blender.org.

## Adding a set

Add a case to `fetch.sh` with the URL and the archive's SHA-256, a row to the table above, and its source and licence. A set with no archive, like the Tears of Steel frames, lists each file's checksum in a `.sha256` file beside the script, which `.gitignore` must then allow.
