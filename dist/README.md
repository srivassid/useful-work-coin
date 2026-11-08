# uwc-miner package

One download for testnet miners, run like SRBMiner: unpack, `./uwc-miner`. Nothing is installed
on the system; the miner's data lives in `~/.uwc-miner/testnet`. What users read is
[README.txt](README.txt), which ships in the package.

## What runs

`uwc-miner` (`uwc_miner/__main__.py`) starts these child processes, restarts any that stop and
prints one status line every 30 s:

| process | what | log |
|---|---|---|
| `pearld` | the node: chain, peers (seeds from `testnet.json`), RPC on 127.0.0.1 only | `logs/node.log` |
| gateway | Pearl's gateway: block templates in, zk certificates and blocks out | `logs/gateway.log` |
| miner per GPU | `uwc_ampere.miner_loop`, the noise miner | `logs/miner-gpuN.log`, `noise-gpuN.jsonl` |
| jobs (`--jobs`) | `agent.roles provider --mine` on one GPU | `logs/jobs.log` |

There is no pool, so every miner runs its own node: the gateway pays each block to one address
(`PEARLD_MINING_ADDRESS`), so a shared gateway would have to be a pool with payouts.

On the jobs GPU, the provider creates `pause-gpuN` for each GPU pass (`--pause-file` on both
sides). The noise miner idles while the file exists, and the job's own GEMMs mine instead.

`--jobs` downloads Qwen3-1.7B at the pinned revision, checks the safetensors sha256s, builds the
weights store (`agent.weights_store`) and checks it against the registered weights root. It also
keeps `--reserve` coins (20 by default) in the jobs balance, depositing mined coins when the
balance falls below half of that.

The data directory has the layout the job agents expect for a network directory (`env`,
`bin/`, `pearlgw.sock`, `agent/`, `models/`, `pool/`, `weights/`). The RPC password is random,
stored in `rpc.json` and `pearld.conf` (mode 0600).

## Build

On a Linux box with uv and cargo (box 71):

    dist/build.sh <go-bin-dir> <testnet-dir> <out-dir>

- `<go-bin-dir>` holds `pearld`, `uwcctl` and `prlctl`, built with `go build -tags xmss,zkpow`.
- `<testnet-dir>` supplies `models/qwen3-1.7b{,.reg}.json` and `pool/*.json`.
- `PEARL` points at the Pearl fork; the default is `<repo>/ext/pearl`.

The script copies uv's standalone CPython 3.12 into the package, then installs
`requirements.txt` with `--torch-backend cu130`, Pearl's gateway packages and py-pearl-mining
(built from source). It copies our code into `app/`, adds a `.pth` file for `app/` and
`app/runtime`, and checks that the copied `detrt` gives the registered runtime hash. Finally it
writes `uwc-miner-<version>-linux-x64.tar.gz` and its `.sha256`.

Size: about 5.0 GB unpacked and a 2.9 GB tarball, mostly CUDA libraries (2.7 GB) and torch.

Requirements on the user's side:
- **NVIDIA driver 580 or newer**: torch and CuPy use CUDA 13; `check` and `run` test the version.
- **glibc 2.33 or newer**: `pearld` and `uwcctl` use cgo for the zk verifier and XMSS. That
  means Ubuntu 22.04 or newer.
- **No CUDA toolkit**: NVRTC comes from the bundled wheels, and kernels compile on first use into
  `cache/cupy`.

## Not yet

- The seed and data-service addresses in `testnet.json` are LAN addresses until there is a
  public seed node.
- There is one job agent per wallet: two providers with the same key would share a state file.
- There is no Windows build yet; WSL2 runs the Linux package.
