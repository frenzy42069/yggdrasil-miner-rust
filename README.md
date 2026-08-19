# Yggdrasil Miner

Address miner for the Yggdrasil network

Supports mining for high and/or short addresses

## Binary builds

You can download binary builds for Linux in the [releases](https://codeberg.org/Acemany/yggdrasil-miner/releases).

## Basic usage (just search high addresses)

```shell
# build and run from source
cargo run --release -- -H
# or download and run
./yggdrasil-miner -H
```

## Address matching

To change which address will be searched ou will need to rewrite the matching part of the code! `src/handler.rs:22`

## Benchmarks

- AMD Radeon RX 550: 0.52 MH/s
- AMD Radeon RX 5500 XT: 5.9 MH/s
- Nvidia GeForce RTX 4060: 18.13 MH/s
- AMD Radeon RX 6800 XT: 25 MH/s (likely outdated)
- Intel Iris Xe TGL GT2 (integrated in i5-1135G7): 0.67 MH/s (likely outdated)

Feel free to add your results and report hardware-specific issues!

## Credits
- [nano-vanity](https://github.com/PlasmaPower/nano-vanity) for ed25519 OpenCL kernel
- [opencl_brute](https://github.com/bkerler/opencl_brute) for sha512 OpenCL kernel
