uwc-miner: Useful-Work Coin testnet miner
=========================================

Needs: Linux x86-64 (Ubuntu 22.04 or newer, or WSL2 on Windows), an NVIDIA GPU from the
RTX 30, 40 or 50 series, and NVIDIA driver 580 or newer. Nothing is installed on the system:
everything is in this folder, and the miner's data is in ~/.uwc-miner/testnet.

Start
-----
    tar xzf uwc-miner-*-linux-x64.tar.gz
    cd uwc-miner-*
    ./uwc-miner check        # finds your GPUs and compiles the CUDA kernels once
    ./uwc-miner              # mines on every GPU

The first run makes a wallet, ~/.uwc-miner/testnet/wallet.json. Back it up: it is the only way
to spend what you mine. Stop the miner with Ctrl-C.

AI jobs
-------
    ./uwc-miner --jobs

This also runs paid AI jobs (Qwen3-1.7B embeddings) on the first GPU. The first time, it downloads
the model (about 4 GB) and checks that it is the exact checkpoint the chain registered. While a
job runs, mining on that GPU pauses, and the job itself mines. Jobs need a bond, so the miner
keeps about 20 mined coins in your jobs balance (--reserve 0 turns that off).

Options
-------
    --gpus 0,2           only these GPUs (numbering as in nvidia-smi)
    --jobs-gpu 1         run jobs on GPU 1 instead of the first one
    --address <addr>     pay mined blocks to another address
    --peer host:port     an extra peer to connect to
    --nolisten           accept no incoming connections (port 18690 otherwise)
    --datadir <dir>      keep the data somewhere else

Wallet (while the miner runs, from a second terminal)
------------------------------------------------------
    ./uwc-miner wallet            address and balances
    ./uwc-miner send 5 <address>  pay 5 coins
    ./uwc-miner deposit 10        wallet -> jobs balance
    ./uwc-miner withdraw 10       jobs balance -> wallet

Logs are in ~/.uwc-miner/testnet/logs (node, gateway, miner-gpuN, jobs). This is a testnet:
its coins have no value, and the chain may be reset.
