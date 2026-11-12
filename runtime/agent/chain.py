"""The chain, through uwcctl (devnet/uwc.sh environment). Every call returns parsed JSON."""
from __future__ import annotations

import json
import os
import subprocess


class ChainError(RuntimeError):
    pass


class Chain:
    def __init__(self, devnet_dir: str):
        self.dir = os.path.abspath(devnet_dir)
        env = dict(os.environ)
        with open(os.path.join(self.dir, "env")) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k] = os.path.expandvars(v)
        env.update(UWC_RPC=f"127.0.0.1:{env.get('RPC_PORT', '18671')}", UWC_RPC_USER=env["RPC_USER"],
                   UWC_RPC_PASS=env["RPC_PASS"])
        net = {"uwctestnet": "testnet", "uwcmainnet": "mainnet"}.get(env.get("NET", ""))
        if net:
            env["UWC_NET"] = net
        self.env = env
        self.bin = os.path.join(self.dir, "bin", "uwcctl")

    def run(self, *args: str, stdin: str | None = None):
        p = subprocess.run([self.bin, *args], input=stdin, capture_output=True, text=True, env=self.env, timeout=120)
        if p.returncode != 0:
            raise ChainError(p.stderr.strip() or p.stdout.strip())
        return json.loads(p.stdout) if p.stdout.strip().startswith(("{", "[")) else p.stdout

    def state(self) -> dict:
        return self.run("state")

    def height(self) -> int:
        return self.state()["height"]

    def job(self, jid: str) -> dict | None:
        try:
            return self.run("job", jid)
        except ChainError as e:
            if "unknown job" in str(e):
                return None
            raise

    def jobs(self, status: str | None = None) -> dict:
        return self.run("jobs", *([status] if status else [])) or {}

    def account(self, acct: str) -> dict:
        return self.run("account", acct)

    def category(self, cid: str) -> dict:
        return self.run("category", cid)

    def mempool(self) -> set[str]:
        """Txids in the local node's mempool."""
        p = subprocess.run([os.path.join(self.dir, "bin", "prlctl"), "--configfile=/dev/null",
                            f"--rpcserver=127.0.0.1:{self.env.get('RPC_PORT', '18671')}", f"--rpcuser={self.env['RPC_USER']}",
                            f"--rpcpass={self.env['RPC_PASS']}", "--notls", "getrawmempool"],
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            raise ChainError(p.stderr.strip() or p.stdout.strip())
        return set(json.loads(p.stdout or "[]"))

    def ops(self, key: str, ops: list[dict]) -> dict:
        return self.run("ops", key, "-", stdin=json.dumps(ops))

    def key_info(self, key: str) -> dict:
        out = self.run("info", key)
        return dict(line.split(None, 1) for line in out.strip().splitlines())
