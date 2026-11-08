"""uwc-miner: everything a Useful-Work Coin testnet miner runs, started from one command.

  ./uwc-miner                      mine on every NVIDIA GPU (makes a wallet on first run)
  ./uwc-miner --jobs               also run AI jobs on the first GPU (downloads the model once)
  ./uwc-miner --gpus 0,2           only these GPUs (nvidia-smi numbering)
  ./uwc-miner wallet               address and balances (while uwc-miner runs)
  ./uwc-miner deposit 10           move 10 coins into the jobs balance (bonds come from it)
  ./uwc-miner withdraw 10 | send 10 <address>
  ./uwc-miner fetch-model          download and check the job model ahead of time
  ./uwc-miner check                GPUs, driver and CUDA kernels

Inside, it runs the node (pearld: the chain, peers), the gateway (block templates in, proofs to
blocks out), one noise miner per GPU and, with --jobs, a job agent (provider) that mines on its
own job GEMMs and pauses the noise miner on its GPU while a job pass runs. All state is under
--datadir (default ~/.uwc-miner/testnet); the wallet key is wallet.json there.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # <pkg>/app
PKG = os.path.dirname(APP)
SHARE = os.path.join(PKG, "share")
BIN = os.path.join(PKG, "bin")
PY = [sys.executable, "-E", "-s", "-u"]


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def die(msg: str):
    print(f"uwc-miner: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def coins(units: int, cfg: dict) -> str:
    return f"{units / cfg['coin']:.4f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------ home directory


class Home:
    """The data directory. It is laid out as the job agents expect a network directory: env,
    bin/uwcctl, bin/prlctl, pearlgw.sock, agent/, models/, pool/, weights/."""

    def __init__(self, path: str, cfg: dict, rpc_port: int | None = None):
        self.d = os.path.abspath(os.path.expanduser(path))
        self.cfg = cfg
        self.want_port = rpc_port
        self.key = os.path.join(self.d, "wallet.json")
        self.sock = os.path.join(self.d, "pearlgw.sock")
        self.logs = os.path.join(self.d, "logs")

    def p(self, *a) -> str:
        return os.path.join(self.d, *a)

    def setup(self):
        for sub in ("logs", "node", "models", "weights", "agent", "cache"):
            os.makedirs(self.p(sub), exist_ok=True)
        os.chmod(self.d, 0o700)
        for name, target in (("bin", BIN), ("pool", os.path.join(SHARE, "pool"))):
            link = self.p(name)
            if os.path.islink(link) or os.path.exists(link):
                if os.path.islink(link) and os.readlink(link) == target:
                    continue
                os.remove(link) if os.path.islink(link) else shutil.rmtree(link)
            os.symlink(target, link)
        rpc = self.p("rpc.json")
        r = json.load(open(rpc)) if os.path.exists(rpc) else {"user": "uwc", "pass": secrets.token_hex(16)}
        r["port"] = self.want_port or r.get("port") or self.cfg["rpc_port"]  # wallet commands find it here
        with open(os.open(rpc, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(r, f)
        self.rpc = r
        self.rpc_port = r["port"]
        c = self.cfg
        env = (f"NET={c['net']}\nRPC_USER={self.rpc['user']}\nRPC_PASS={self.rpc['pass']}\n"
               f"RPC_PORT={self.rpc_port}\nP2P_PORT={c['p2p_port']}\nUWC_HOME={self.d}\n")
        with open(os.open(self.p("env"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            f.write(env)
        with open(os.open(self.p("pearld.conf"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            f.write(f"[Application Options]\nrpcuser={self.rpc['user']}\nrpcpass={self.rpc['pass']}\n")

    def lock(self):
        self._lock = open(self.p("lock"), "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            die(f"another uwc-miner is already running on {self.d}")

    # -------------------------------------------------------------- chain access

    def ctl_env(self) -> dict:
        env = dict(os.environ, UWC_RPC=f"127.0.0.1:{self.rpc_port}", UWC_RPC_USER=self.rpc["user"],
                   UWC_RPC_PASS=self.rpc["pass"])
        if self.cfg.get("uwcctl_net"):
            env["UWC_NET"] = self.cfg["uwcctl_net"]
        return env

    def ctl(self, *args: str, timeout: float = 60) -> str:
        r = subprocess.run([os.path.join(BIN, "uwcctl"), *args], capture_output=True, text=True,
                           env=self.ctl_env(), timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip())
        return r.stdout

    def rpc_call(self, method: str, params: list | None = None, timeout: float = 10):
        body = json.dumps({"jsonrpc": "1.0", "id": 1, "method": method, "params": params or []}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.rpc_port}", data=body)
        tok = base64.b64encode(f"{self.rpc['user']}:{self.rpc['pass']}".encode()).decode()
        req.add_header("Authorization", f"Basic {tok}")
        req.add_header("Content-Type", "application/json")
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        if r.get("error"):
            raise RuntimeError(r["error"])
        return r["result"]

    def ensure_key(self) -> dict:
        if not os.path.exists(self.key):
            self.ctl("keygen", self.key)
            os.chmod(self.key, 0o600)
            info = self.key_info()
            say(f"new wallet: {self.key}")
            say(f"  address {info['address']}")
            say("  back up wallet.json: it is the only way to spend what this miner earns")
        return self.key_info()

    def key_info(self) -> dict:
        out = self.ctl("info", self.key)
        return dict(line.split(None, 1) for line in out.strip().splitlines())

    def maturing(self, script: str, won: dict | None = None) -> int:
        """Mined coins not yet spendable: this wallet's coinbase outputs in the last
        coinbase_maturity blocks (uwcctl coins skips them). Blocks paying the wallet go into won
        (height -> hash), so a block later orphaned drops out."""
        tip = self.rpc_call("getblockcount")
        total = 0
        for hgt in range(max(1, tip - self.cfg["coinbase_maturity"] + 1), tip + 1):
            b = self.rpc_call("getblock", [self.rpc_call("getblockhash", [hgt]), 2])
            mine = 0
            for o in (b.get("rawtx") or b.get("tx"))[0]["vout"]:
                if o.get("scriptPubKey", {}).get("hex") == script:
                    mine += round(o["value"] * self.cfg["coin"])
            total += mine
            if won is not None:
                if mine:
                    won[hgt] = b["hash"]
                else:
                    won.pop(hgt, None)
        return total

    def balances(self, account: str) -> tuple[int, int]:
        """(spendable coins of the wallet, jobs balance of its module account)."""
        spend = json.loads(self.ctl("coins", self.key, timeout=120))["total"]
        acct = json.loads(self.ctl("account", account))
        return spend, int(acct.get("balance", 0) or 0)


# ------------------------------------------------------------------ GPUs


def gpus() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    res = []
    for line in out.strip().splitlines():
        i, name, mem, drv = [x.strip() for x in line.split(",")]
        res.append({"index": int(i), "name": name, "mem_mb": int(float(mem)), "driver": drv})
    return res


def pick_gpus(spec: str, cfg: dict) -> list[dict]:
    have = gpus()
    if not have:
        say("no NVIDIA GPU found (nvidia-smi): running the node only")
        return []
    drv = int(have[0]["driver"].split(".")[0])
    if drv < cfg["min_driver"]:
        die(f"NVIDIA driver {have[0]['driver']} is too old: this build uses CUDA 13 and needs driver "
            f"{cfg['min_driver']} or newer")
    if spec in ("all", ""):
        return have
    if spec == "none":
        return []
    want = [int(x) for x in spec.split(",")]
    by = {g["index"]: g for g in have}
    missing = [w for w in want if w not in by]
    if missing:
        die(f"no GPU {missing} (nvidia-smi lists {sorted(by)})")
    return [by[w] for w in want]


# ------------------------------------------------------------------ the job model


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def ensure_model(home: Home) -> str:
    """Download the job model at its pinned revision, check its files, build the weights store
    (the rows the chain's weights root commits to) and check that root. Returns the model json."""
    m = home.cfg["model"]
    hf_dir = home.p("models", m["hf_repo"].split("/")[-1])
    ok_mark = os.path.join(hf_dir, ".uwc-verified")
    if not os.path.exists(ok_mark):
        from huggingface_hub import snapshot_download
        say(f"downloading {m['hf_repo']} ({m['revision'][:10]}), about 4 GB")
        snapshot_download(m["hf_repo"], revision=m["revision"], local_dir=hf_dir,
                          allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE"])
        for name, want in m["sha256"].items():
            got = sha256_file(os.path.join(hf_dir, name))
            if got != want:
                die(f"{name}: sha256 {got[:16]}... is not the registered checkpoint's {want[:16]}...")
        open(ok_mark, "w").write(m["revision"])
        say("model files check out")
    info = json.load(open(os.path.join(SHARE, "models", f"{m['name']}.json")))
    info["model_dir"] = hf_dir
    mj = home.p("models", f"{m['name']}.json")
    json.dump(info, open(mj, "w"), indent=1)
    shutil.copy(os.path.join(SHARE, "models", f"{m['name']}.reg.json"), home.p("models", f"{m['name']}.reg.json"))
    store = home.p("weights", m["name"])
    meta = os.path.join(store, "meta.json")
    if not (os.path.exists(meta) and json.load(open(meta)).get("weights_root") == info["weights_root"]):
        say("building the weights store (the rows a dispute proof cites); takes a few minutes, once")
        r = subprocess.run([*PY, "-m", "agent.weights_store", "--model", mj, "--out", store], cwd=home.d,
                           env=child_env(home))
        if r.returncode != 0 or json.load(open(meta)).get("weights_root") != info["weights_root"]:
            die("the weights store does not match the registered model's weights root")
        say("weights store matches the registered weights root")
    return mj


# ------------------------------------------------------------------ child processes


def child_env(home: Home) -> dict:
    cache = home.p("cache")
    return dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUPY_CACHE_DIR=os.path.join(cache, "cupy"),
                TRITON_CACHE_DIR=os.path.join(cache, "triton"), TORCHINDUCTOR_CACHE_DIR=os.path.join(cache, "inductor"),
                HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                UWC_HOME=home.d)


def _die_with_parent():
    os.setsid()
    with contextlib.suppress(Exception):  # PR_SET_PDEATHSIG: stop if uwc-miner is killed
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)


class Proc:
    def __init__(self, name: str, argv: list[str], home: Home, env: dict | None = None, essential: bool = False):
        self.name, self.argv, self.home, self.essential = name, argv, home, essential
        self.env = env or child_env(home)
        self.log = os.path.join(home.logs, f"{name}.log")
        self.p: subprocess.Popen | None = None
        self.starts: list[float] = []

    def start(self):
        out = open(self.log, "ab")
        out.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} start\n".encode())
        out.flush()
        self.p = subprocess.Popen(self.argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  env=self.env, cwd=self.home.d, preexec_fn=_die_with_parent)
        out.close()
        self.starts = [t for t in self.starts if time.time() - t < 600] + [time.time()]

    def alive(self) -> bool:
        return self.p is not None and self.p.poll() is None

    def tail(self, n: int = 12) -> str:
        try:
            with open(self.log, "rb") as f:
                f.seek(max(0, os.path.getsize(self.log) - 8000))
                return "\n".join(f.read().decode(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def stop(self, wait: float = 30):
        if not self.alive():
            return
        self.p.send_signal(signal.SIGTERM)
        try:
            self.p.wait(wait)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait(5)


# ------------------------------------------------------------------ run


def run(a: argparse.Namespace, cfg: dict):
    home = Home(a.datadir, cfg, a.rpc_port)
    home.setup()
    home.lock()
    sel = pick_gpus(a.gpus, cfg)
    info = home.ensure_key()
    address = a.address or info["address"]  # the wallet key still signs job operations
    model_json = None
    job_gpu = None
    if a.jobs:
        if not sel:
            die("--jobs needs a GPU")
        job_gpu = a.jobs_gpu if a.jobs_gpu is not None else sel[0]["index"]
        if job_gpu not in [g["index"] for g in sel]:
            die(f"--jobs-gpu {job_gpu} is not one of the mining GPUs {[g['index'] for g in sel]}")
        model_json = ensure_model(home)

    seeds = [] if a.no_seeds else list(cfg["seeds"])
    peers = seeds + (a.peer or [])
    node_argv = [os.path.join(BIN, "pearld"), f"--{cfg['net']}", f"--configfile={home.p('pearld.conf')}",
                 f"--datadir={home.p('node')}", f"--logdir={home.p('logs', 'node')}",
                 f"--rpclisten=127.0.0.1:{home.rpc_port}", "--notls", "--txindex", "--addrindex",
                 f"--miningaddr={address}", "--debuglevel=info"]
    node_argv += ["--nolisten"] if a.nolisten else [f"--listen={a.listen or '0.0.0.0'}:{a.port or cfg['p2p_port']}"]
    node_argv += [f"--addpeer={p}" for p in peers]
    gw_env = dict(child_env(home), PEARLD_RPC_URL=f"http://127.0.0.1:{home.rpc_port}", PEARLD_RPC_USER=home.rpc["user"],
                  PEARLD_RPC_PASSWORD=home.rpc["pass"], PEARLD_MINING_ADDRESS=address,
                  MINER_RPC_TRANSPORT="uds", MINER_RPC_SOCKET_PATH=home.sock)
    node = Proc("node", node_argv, home, essential=True)
    gateway = Proc("gateway", [*PY, "-m", "pearl_gateway.cli", "start"], home, env=gw_env, essential=True)
    miners = []
    for g in sel:
        i = g["index"]
        miners.append(Proc(f"miner-gpu{i}", [*PY, "-m", "uwc_ampere.miner_loop", "--gateway", home.sock, "--device", str(i),
                                             "--stats", home.p("logs", f"noise-gpu{i}.jsonl"),
                                             "--pause-file", home.p(f"pause-gpu{i}")], home))
    agent = None
    if a.jobs:
        units = ",".join(home.p("pool", u) for u in cfg["units"])
        argv = [*PY, "-m", "agent.roles", "provider", "--key", home.key, "--devnet", home.d, "--data", a.data or cfg["data_url"],
                "--model", model_json, "--weights", home.p("weights", cfg["model"]["name"]), "--units", units,
                "--device", f"cuda:{job_gpu}", "--mine", "--pause-file", home.p(f"pause-gpu{job_gpu}")]
        if a.max_jobs:
            argv += ["--max-jobs", str(a.max_jobs)]
        agent = Proc("jobs", argv, home)
    for f in os.listdir(home.d):  # stale pause files from a previous run
        if f.startswith("pause-gpu"):
            os.remove(home.p(f))

    stopping = []

    def on_signal(sig, _):
        stopping.append(sig)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    say(f"uwc-miner {open(os.path.join(PKG, 'VERSION')).read().strip()} on {cfg['net']}, data in {home.d}")
    say(f"mining to {address}")
    for g in sel:
        say(f"  GPU {g['index']}: {g['name']}, {g['mem_mb'] // 1024} GB" + ("  + AI jobs" if g["index"] == job_gpu else ""))
    say(f"peers: {', '.join(peers) or 'none'}; logs in {home.logs}")
    node.start()
    t0 = time.time()
    while not stopping:  # wait for the node's RPC
        try:
            home.rpc_call("getblockcount", timeout=3)
            break
        except Exception:
            if not node.alive():
                die("the node stopped:\n" + node.tail())
            if time.time() - t0 > 120:
                die("the node's RPC did not come up in 2 minutes:\n" + node.tail())
            time.sleep(1)
    procs = [gateway, *miners] + ([agent] if agent else [])
    if not stopping:
        with contextlib.suppress(FileNotFoundError):
            os.remove(home.sock)  # a stale socket would let the miners start before the gateway listens
        gateway.start()
        for _ in range(60):
            if os.path.exists(home.sock) or stopping:
                break
            time.sleep(1)
        for p in procs[1:]:
            p.start()

    status = Status(home, cfg, sel, info["account"], info["script"] if not a.address else "", job_gpu, a)
    next_status = time.time() + 20
    while not stopping:
        for p in [node, *procs]:
            if p.p is not None and not p.alive():
                code = p.p.returncode
                if len(p.starts) >= 5:
                    say(f"{p.name} keeps stopping (exit {code}); last log lines:\n{p.tail()}")
                    if p.essential:
                        stopping.append("fail")
                        break
                    p.p = None  # give up on it; the rest keeps running
                    continue
                say(f"{p.name} stopped (exit {code}), restarting; see {p.log}")
                time.sleep(2)
                p.start()
        if time.time() >= next_status:
            next_status = time.time() + a.status_every
            with contextlib.suppress(Exception):
                status.show()
        time.sleep(1)

    say("stopping")
    for p in reversed(procs):
        p.stop(wait=20)
    node.stop(wait=60)
    for f in os.listdir(home.d):
        if f.startswith("pause-gpu"):
            os.remove(home.p(f))
    say("stopped")
    if "fail" in stopping:
        sys.exit(1)


class Status:
    """One line every --status-every seconds: chain, mining per GPU, wallet, jobs."""

    def __init__(self, home: Home, cfg: dict, sel: list[dict], account: str, script: str, job_gpu: int | None, a):
        self.home, self.cfg, self.sel, self.account, self.job_gpu, self.a = home, cfg, sel, account, job_gpu, a
        self.script = script
        self.won: dict[int, str] = {}  # blocks paying this wallet since start
        self.t0 = time.time()
        self.bal: tuple[int, int] | None = None
        self.bal_t = 0.0
        self.deposit_t = 0.0

    def stats(self, i: int) -> tuple[float | None, int]:
        """(latest noise rate in MAC/s, tiles submitted since start) of GPU i."""
        path = self.home.p("logs", f"noise-gpu{i}.jsonl")
        rate, sub = None, 0
        with contextlib.suppress(OSError):
            with open(path, "rb") as f:
                skip = os.path.getsize(path) > 200_000
                f.seek(max(0, os.path.getsize(path) - 200_000))
                for line in f.read().decode(errors="replace").splitlines()[int(skip):]:
                    with contextlib.suppress(Exception):
                        r = json.loads(line)
                        if r["t"] >= self.t0:
                            rate = r["macs_per_s"]
                            sub += r.get("submitted", 0)
        return rate, sub

    def jobs_done(self) -> int:
        path = self.home.p("agent", f"provider-{self.account[:8]}.json")
        with contextlib.suppress(Exception):
            return sum(1 for j in json.load(open(path))["jobs"].values() if j.get("roots"))
        return 0

    def show(self):
        h = self.home
        try:
            height = h.rpc_call("getblockcount")
            peers = h.rpc_call("getpeerinfo")
            best = max([p.get("startingheight", 0) for p in peers] + [height])
            chain = f"height {height}, {len(peers)} peer{'s' * (len(peers) != 1)}" + (
                f", syncing to {best}" if best > height + 2 else "") + (
                " (mining starts when a peer connects)" if not peers else "")
        except Exception:
            chain = "node starting"
        parts = [chain]
        subs = 0
        for g in self.sel:
            i = g["index"]
            rate, sub = self.stats(i)
            subs += sub
            if os.path.exists(h.p(f"pause-gpu{i}")):
                s = "AI job"
            elif rate is None:
                s = "starting"
            else:
                s = f"{rate / 1e12:.1f} TMAC/s"
            parts.append(f"GPU{i} {s}")
        if self.sel and self.script:
            n = len(self.won)
            parts.append(f"{n} block{'s' * (n != 1)} won" + (f" ({subs} solutions sent)" if subs > n else ""))
        if time.time() - self.bal_t > 60:
            self.bal_t = time.time()
            with contextlib.suppress(Exception):
                self.bal = h.balances(self.account)
            with contextlib.suppress(Exception):
                self.mat = h.maturing(self.script, self.won)
        if self.bal:
            spend, jb = self.bal
            mat = getattr(self, "mat", 0)
            parts.append(f"wallet {coins(spend, self.cfg)}" + (f" (+{coins(mat, self.cfg)} maturing)" if mat else "")
                         + (f" + jobs {coins(jb, self.cfg)}" if self.job_gpu is not None or jb else ""))
            self.auto_deposit(spend, jb)
        if self.job_gpu is not None:
            parts.append(f"{self.jobs_done()} jobs done")
        say(" | ".join(parts))

    def auto_deposit(self, spend: int, jb: int):
        """With --jobs, keep the jobs balance near --reserve coins from mined coins (job bonds and
        pool claims are paid from it)."""
        if self.job_gpu is None or self.a.reserve <= 0 or time.time() - self.deposit_t < 180:
            return
        c = self.cfg["coin"]
        want = int(self.a.reserve * c)
        if jb >= want // 2:
            return
        amt = min(want - jb, spend - c // 100)  # leave a little for transaction fees
        if amt < c:
            return
        self.deposit_t = time.time()
        try:
            self.home.ctl("deposit", self.home.key, str(amt))
            say(f"moved {coins(amt, self.cfg)} coins into the jobs balance (keeps --reserve {self.a.reserve:g})")
            self.bal_t = 0
        except Exception as e:
            say(f"deposit failed: {e}")


# ------------------------------------------------------------------ other commands


def running_home(a, cfg) -> Home:
    home = Home(a.datadir, cfg)
    if not os.path.exists(home.p("rpc.json")):
        die(f"no uwc-miner data in {home.d}; run ./uwc-miner first")
    home.setup()
    try:
        home.rpc_call("getblockcount", timeout=3)
    except Exception:
        die("uwc-miner is not running (its node answers wallet commands); start it first")
    return home


def cmd_wallet(a, cfg):
    home = running_home(a, cfg)
    info = home.key_info()
    spend, jb = home.balances(info["account"])
    mat = home.maturing(info["script"])
    print(f"address  {info['address']}\naccount  {info['account']}\nwallet   {coins(spend, cfg)} coins (spendable)\n"
          f"maturing {coins(mat, cfg)} coins (mined in the last {cfg['coinbase_maturity']} blocks)\n"
          f"jobs     {coins(jb, cfg)} coins (bonds and job earnings; withdraw moves them to the wallet)\nkey      {home.key}")


def cmd_money(a, cfg):
    home = running_home(a, cfg)
    amt = str(int(round(float(a.amount) * cfg["coin"])))
    args = {"deposit": ["deposit", home.key, amt], "withdraw": ["withdraw", home.key, amt],
            "send": ["send", home.key, amt, getattr(a, "to", "")]}[a.cmd]
    print(home.ctl(*args).strip())


def cmd_fetch_model(a, cfg):
    home = Home(a.datadir, cfg)
    home.setup()
    ensure_model(home)


def cmd_check(a, cfg):
    home = Home(a.datadir, cfg)
    home.setup()
    have = gpus()
    if not have:
        die("no NVIDIA GPU or driver found (nvidia-smi)")
    for g in have:
        print(f"GPU {g['index']}: {g['name']}, {g['mem_mb']} MB, driver {g['driver']}")
    drv = int(have[0]["driver"].split(".")[0])
    need = cfg["min_driver"]
    print("driver: ok" if drv >= need else f"driver: too old, needs {need} or newer")
    code = ("import torch, uwc_ampere.ops as o\n"
            "for i in range(torch.cuda.device_count()):\n"
            "    torch.cuda.set_device(i); o._module(); print(f'cuda:{i} kernels ok:', torch.cuda.get_device_name(i))\n")
    r = subprocess.run([*PY, "-c", code], env=child_env(home))
    sys.exit(r.returncode)


def main():
    cfg = json.load(open(os.path.join(SHARE, "testnet.json")))
    ap = argparse.ArgumentParser(prog="uwc-miner", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="~/.uwc-miner/testnet")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="mine (the default)")
    for p in (ap, r):
        p.add_argument("--gpus", default="all", help="all, none, or nvidia-smi indexes like 0,2")
        p.add_argument("--jobs", action="store_true", help="also run AI jobs (on --jobs-gpu, the first GPU by default)")
        p.add_argument("--jobs-gpu", type=int, default=None)
        p.add_argument("--max-jobs", type=int, default=None, help="jobs computed together in one GPU pass")
        p.add_argument("--reserve", type=float, default=100,
                       help="coins kept in the jobs balance from mining, for job bonds (a pool unit bonds 10; 0: off)")
        p.add_argument("--address", default=None, help="pay mined blocks here instead of the wallet")
        p.add_argument("--peer", action="append", help="extra peer host:port (repeatable)")
        p.add_argument("--no-seeds", action="store_true", help="do not connect to the built-in seed nodes")
        p.add_argument("--nolisten", action="store_true", help="accept no incoming peer connections")
        p.add_argument("--port", type=int, default=None, help="P2P listen port (18690)")
        p.add_argument("--listen", default=None, help="P2P listen address (all interfaces)")
        p.add_argument("--rpc-port", type=int, default=None, help="the node's local RPC port (18691); remembered in --datadir")
        p.add_argument("--data", default=None, help="job data service URL")
        p.add_argument("--status-every", type=float, default=30)
    sub.add_parser("wallet", help="address and balances")
    for name in ("deposit", "withdraw"):
        sub.add_parser(name).add_argument("amount")
    s = sub.add_parser("send")
    s.add_argument("amount")
    s.add_argument("to")
    sub.add_parser("fetch-model", help="download and check the job model")
    sub.add_parser("check", help="GPUs, driver, CUDA kernels")
    a = ap.parse_args()
    if a.cmd in (None, "run"):
        run(a, cfg)
    elif a.cmd == "wallet":
        cmd_wallet(a, cfg)
    elif a.cmd in ("deposit", "withdraw", "send"):
        cmd_money(a, cfg)
    elif a.cmd == "fetch-model":
        cmd_fetch_model(a, cfg)
    elif a.cmd == "check":
        cmd_check(a, cfg)


if __name__ == "__main__":
    main()
