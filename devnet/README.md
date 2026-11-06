# UWC devnet (build step 7, private)

Two nodes on the LAN with real cuPOW mining: node A on box 71 (3x RTX 3080), node B on the
5090 box (wired 192.168.29.50). Nothing listens on a public interface.

Per box, `~/Documents/useful-work-coin/devnet/` holds `bin/` (pearld, prlctl, uwcctl), `env`
(not in git) and the data and log directories. `env` sets:

    NODE_IP=192.168.29.71        # this box's LAN address (P2P listens here)
    PEER_IP=192.168.29.50        # the other node
    RPC_USER=... RPC_PASS=...    # RPC listens on 127.0.0.1:18671 only
    MINING_ADDR=rprl1p...        # coinbase address (uwcctl info <key>)
    VENV=~/Documents/useful-work-coin/port/.venv

Scripts: `node.sh` (pearld), `gateway.sh` (Pearl's gateway with the UWC coinbase outputs),
`miner.sh <gpu>` (Ampere noise miner), `ctl.sh <cmd>` (prlctl against the local node).
