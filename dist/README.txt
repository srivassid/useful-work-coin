uwc-miner: Useful-Work Coin (AICoin) miner
==========================================

Needs: Linux x86-64 (Ubuntu 22.04 or newer, or WSL2 on Windows), an NVIDIA GPU from the
RTX 30, 40 or 50 series, and NVIDIA driver 580 or newer. Nothing is installed on the system:
everything is in this folder, and the miner's data is in ~/.uwc-miner/<network> (mainnet for
this package; the VERSION file says which).

Start
-----
    tar xzf uwc-miner-*-linux-x64.tar.gz
    cd uwc-miner-*
    ./uwc-miner check        # finds your GPUs and compiles the CUDA kernels once
    ./uwc-miner              # mines on every GPU

The first run makes a wallet, ~/.uwc-miner/mainnet/wallet.json, and prints its address. Back it
up: it is the only way to spend what you mine. Stop the miner with Ctrl-C. Mined coins become
spendable after 100 blocks (about 100 minutes).

AI jobs
-------
    ./uwc-miner --jobs

This also runs paid AI jobs (Qwen3-1.7B text generation) on the first GPU. The first time, it
downloads the model (about 4 GB) and checks that it is the exact checkpoint the chain registered.
While a job runs, mining on that GPU pauses, and the job itself mines. Jobs need a bond, so the
miner keeps about 100 mined coins in your jobs balance (--reserve 0 turns that off).

Options
-------
    --gpus 0,2           only these GPUs (numbering as in nvidia-smi)
    --jobs-gpu 1         run jobs on GPU 1 instead of the first one
    --address <addr>     pay mined blocks to another address
    --peer host:port     an extra peer to connect to
    --nolisten           accept no incoming connections (port 8690 otherwise)
    --datadir <dir>      keep the data somewhere else

Wallet (while the miner runs, from a second terminal)
------------------------------------------------------
    ./uwc-miner wallet            address and balances
    ./uwc-miner send 5 <address>  pay 5 coins
    ./uwc-miner deposit 10        wallet -> jobs balance
    ./uwc-miner withdraw 10       jobs balance -> wallet

To use the AI API with your coins, create a key at https://165-232-182-224.sslip.io/ and send
coins to its address with ./uwc-miner send.

Logs are in ~/.uwc-miner/mainnet/logs (node, gateway, miner-gpuN, jobs). The node connects to
the seed 165.232.182.224:8690, and other nodes can see your IP address. AI job inputs are public.
