"""Coin market: miners sell UWC for bitcoin or ether, paid straight to the seller.

The gateway never holds bitcoin or ether. It holds the coins on sale, in escrow:

  1. A seller (any API key) opens an offer: how many coins, a price in US dollars per coin, and
     their own BTC and/or ETH address. The gateway makes a fresh escrow key for the offer and the
     seller sends the coins to its address (plus one coin for transaction fees).
  2. A buyer (any API key) orders some of an offer's coins in BTC or ETH. The gateway prices the
     order at the live exchange rate and adds a few satoshi or gwei so that the amount is unique
     to the order, then shows the seller's address and the exact amount to pay.
  3. The gateway watches the seller's address on a public block explorer (mempool.space for
     bitcoin, Blockscout for ether). Once the payment has --market-btc-confs / --market-eth-confs
     confirmations, it deposits the coins from escrow into the buyer's jobs balance.
  4. A seller can close an offer with no orders waiting: the coins left in escrow go back to the
     offer's refund address.

An order must be paid within --market-order-minutes; a payment that arrives late is still
honoured for a day when the offer has the coins, and is reported as "late_unfilled" otherwise
(the seller was paid, so the seller should refund the buyer). Only the exact amount, or more,
counts; a payment from an exchange that takes its fee out of the amount does not.

Endpoints (Authorization: Bearer <api key>, except where noted):
  GET  /v1/market               open offers, exchange rates and the market's rules (no auth)
  POST /v1/market/offers        {"coins", "price_usd", "btc_address"?, "eth_address"?, "refund_address"?}
  POST /v1/market/offers/<id>/close
  POST /v1/market/orders        {"offer", "coins", "currency": "BTC" | "ETH"}
  GET  /v1/market/orders/<id>   an order (its buyer or the offer's seller)
  GET  /v1/market/mine          the key's offers and orders, as buyer and as seller
"""
from __future__ import annotations

import json
import math
import os
import re
import secrets
import threading
import time
import traceback
import urllib.request
from decimal import Decimal

from .chain import ChainError

COIN = 100_000_000
FEE_BUFFER = COIN          # a seller sends this much more than they sell, for transaction fees
FEE_RESERVE = COIN // 100  # never sold out of an escrow: its transaction fees (about 0.00001 a transaction)
SEND_MARGIN = 100_000      # left behind when an escrow refunds its balance (the send's own fee)
LATE_WINDOW = 86400        # seconds an expired order is still watched for its payment
OPEN = ("awaiting_payment", "seen", "confirmed", "delivering")
UA = {"User-Agent": "uwc-gateway/1", "Accept": "application/json"}

RE_BTC = re.compile(r"^(bc1[qpzry9x8gf2tvdw0s3jn54khce6mua7l]{25,87}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})$")
RE_ETH = re.compile(r"^0x[0-9a-fA-F]{40}$")
RE_UWC = re.compile(r"^uwc1p[qpzry9x8gf2tvdw0s3jn54khce6mua7l]{50,70}$")
UNIT = {"BTC": 10 ** 8, "ETH": 10 ** 18}          # satoshi, wei
TAG_STEP = {"BTC": 1, "ETH": 10 ** 9}              # 1 satoshi, 1 gwei
TAG_MAX = {"BTC": 300, "ETH": 9999}                # at most ~$0.30 (BTC) / ~$0.04 (ETH) added

SCHEMA = """
CREATE TABLE IF NOT EXISTS offers (id TEXT PRIMARY KEY, key_id INTEGER, created REAL, status TEXT,
  coins INTEGER, sold INTEGER DEFAULT 0, price_usd REAL, btc_address TEXT, eth_address TEXT,
  refund_address TEXT, escrow_keyfile TEXT, escrow_address TEXT, balance INTEGER DEFAULT 0,
  refund_tx TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, offer TEXT, key_id INTEGER, created REAL,
  expires REAL, status TEXT, coins INTEGER, currency TEXT, pay_to TEXT, amount TEXT, usd REAL,
  rate REAL, txid TEXT, confirmations INTEGER DEFAULT 0, delivered_tx TEXT, done REAL, error TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS orders_payment ON orders (currency, txid);
"""


class MarketError(Exception):
    def __init__(self, status: int, message: str, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


def fmt_units(units: int, currency: str) -> str:
    """Satoshi or wei as a decimal BTC or ETH amount, without trailing zeros."""
    s = format(Decimal(units) / Decimal(UNIT[currency]), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def http_json(url: str, timeout: float = 20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ------------------------------------------------------------------ prices and explorers


class Rates:
    """USD prices of BTC and ETH, from Coinbase with CoinGecko as a fallback; cached a minute."""

    def __init__(self, fetch=http_json, ttl: float = 60):
        self.fetch, self.ttl, self.cache, self.t = fetch, ttl, {}, 0.0
        self.lock = threading.Lock()

    def get(self) -> dict:
        with self.lock:
            if self.cache and time.time() - self.t < self.ttl:
                return self.cache
            try:
                got = {c: float(self.fetch(f"https://api.coinbase.com/v2/prices/{c}-USD/spot")["data"]["amount"])
                       for c in ("BTC", "ETH")}
            except Exception:
                try:
                    g = self.fetch("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd")
                    got = {"BTC": float(g["bitcoin"]["usd"]), "ETH": float(g["ethereum"]["usd"])}
                except Exception:
                    if self.cache and time.time() - self.t < 15 * 60:
                        return self.cache  # a stale price beats none for a few minutes
                    raise MarketError(503, "exchange rates are unavailable right now; try again shortly",
                                      "service_unavailable")
            if not all(v > 0 for v in got.values()):
                raise MarketError(503, "exchange rates are unavailable right now", "service_unavailable")
            self.cache, self.t = got, time.time()
            return got


def btc_payments(addr: str, fetch=http_json, base: str = "https://mempool.space/api") -> list[dict]:
    """Payments to a bitcoin address: [{"txid", "units" (satoshi to addr), "confirmations", "time"}]."""
    txs = fetch(f"{base}/address/{addr}/txs")
    tip = None
    out = []
    for tx in txs:
        units = sum(int(o.get("value", 0)) for o in tx.get("vout", []) if o.get("scriptpubkey_address") == addr)
        if units <= 0:
            continue
        st = tx.get("status") or {}
        conf, t = 0, None
        if st.get("confirmed"):
            if tip is None:
                tip = int(fetch(f"{base}/blocks/tip/height"))
            conf = tip - int(st["block_height"]) + 1
            t = st.get("block_time")
        out.append({"txid": tx["txid"], "units": units, "confirmations": conf, "time": t})
    return out


def eth_payments(addr: str, fetch=http_json, base: str = "https://eth.blockscout.com/api/v2") -> list[dict]:
    """Plain ether transfers to an address: [{"txid", "units" (wei), "confirmations", "time"}]."""
    d = fetch(f"{base}/addresses/{addr}/transactions?filter=to")
    out = []
    for tx in d.get("items", []):
        to = ((tx.get("to") or {}).get("hash") or "").lower()
        if to != addr.lower() or tx.get("status") not in ("ok", None) or tx.get("result") in ("error", "failed"):
            continue
        units = int(tx.get("value") or 0)
        if units <= 0:
            continue
        conf = int(tx.get("confirmations") or 0) if tx.get("status") == "ok" else 0
        t = None
        if tx.get("timestamp"):
            try:
                from datetime import datetime
                t = datetime.fromisoformat(tx["timestamp"].replace("Z", "+00:00")).timestamp()
            except ValueError:
                t = None
        out.append({"txid": tx["hash"].lower(), "units": units, "confirmations": conf, "time": t})
    return out


def match(payments: list[dict], orders: list[dict], used: set[str]) -> dict[str, dict]:
    """Assign payments to orders that all pay the same address in the same currency. A payment
    pays an order when it is at least the order's amount and was not mined well before the order;
    each payment goes to the open order with the largest amount it covers, and pays one order."""
    out: dict[str, dict] = {}
    left = sorted(orders, key=lambda o: -int(o["amount"]))
    for p in sorted(payments, key=lambda p: -p["units"]):
        if p["txid"] in used:
            continue
        for o in left:
            if o["id"] in out or int(o["amount"]) > p["units"]:
                continue
            if p["time"] is not None and p["time"] < o["created"] - 3600:
                continue  # mined more than an hour before the order existed (block times drift)
            out[o["id"]] = p
            break
    return out


# ------------------------------------------------------------------ the market


class Market:
    def __init__(self, gw, a, rates: Rates | None = None, fetch=http_json):
        self.gw, self.a, self.store, self.chain = gw, a, gw.store, gw.chain
        self.rates = rates or Rates(fetch)
        self.fetch = fetch
        self.lock = threading.Lock()  # order creation against an offer's free coins; deliveries
        self.dir = os.path.join(a.home, "escrow")
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        with self.store.lock:
            self.store.db.executescript(SCHEMA)
        # an order caught between "delivering" and "paid" by a restart may have been delivered:
        # a person checks it on chain rather than the gateway paying twice
        for o in self.store.q("SELECT id FROM orders WHERE status = 'delivering'"):
            self.store.q("UPDATE orders SET status = 'check', error = ? WHERE id = ?",
                         "the gateway restarted while delivering: check the escrow on chain", o["id"])

    # -------------------------------------------------------------- views

    def free(self, offer) -> int:
        """Coins of an offer that orders can still take."""
        locked = self.store.one(f"SELECT COALESCE(SUM(coins), 0) n FROM orders WHERE offer = ? AND status IN "
                                f"({','.join('?' * len(OPEN))})", offer["id"], *OPEN)["n"]
        return max(0, min(offer["coins"] - offer["sold"], offer["balance"] - FEE_RESERVE) - locked)

    def offer_view(self, o, mine: bool = False) -> dict:
        seller = self.store.one("SELECT name FROM keys WHERE id = ?", o["key_id"])
        sales = self.store.one("SELECT COUNT(*) n FROM orders o JOIN offers f ON o.offer = f.id "
                               "WHERE f.key_id = ? AND o.status = 'paid'", o["key_id"])["n"]
        v = {"id": o["id"], "status": o["status"], "price_usd": o["price_usd"], "coins": o["coins"] / COIN,
             "sold_coins": o["sold"] / COIN, "available_coins": self.free(o) / COIN,
             "currencies": [c for c, f in (("BTC", "btc_address"), ("ETH", "eth_address")) if o[f]],
             "seller": (seller["name"] if seller else "")[:40], "seller_sales": sales, "created": o["created"]}
        if mine:
            v.update({"escrow_address": o["escrow_address"], "escrow_balance_coins": o["balance"] / COIN,
                      "send_coins": (0 if o["balance"] >= o["coins"] - o["sold"] + FEE_RESERVE
                                     else o["coins"] - o["sold"] + FEE_BUFFER - o["balance"]) / COIN, "btc_address": o["btc_address"],
                      "eth_address": o["eth_address"], "refund_address": o["refund_address"],
                      "refund_tx": o["refund_tx"], "note": o["note"]})
        return v

    def order_view(self, o) -> dict:
        cur = o["currency"]
        amt = fmt_units(int(o["amount"]), cur)
        uri = f"bitcoin:{o['pay_to']}?amount={amt}" if cur == "BTC" else f"ethereum:{o['pay_to']}?value={o['amount']}"
        return {"id": o["id"], "offer": o["offer"], "status": o["status"], "coins": o["coins"] / COIN,
                "currency": cur, "pay_to": o["pay_to"], "amount": amt, "amount_units": o["amount"], "uri": uri,
                "usd": round(o["usd"], 2), "rate_usd": o["rate"], "created": o["created"], "expires": o["expires"],
                "txid": o["txid"], "confirmations": o["confirmations"],
                "confirmations_needed": self.a.market_btc_confs if cur == "BTC" else self.a.market_eth_confs,
                "delivered_tx": o["delivered_tx"], "error": o["error"]}

    def overview(self) -> dict:
        try:
            rates = self.rates.get()
        except MarketError:
            rates = None
        offers = [self.offer_view(o) for o in self.store.q("SELECT * FROM offers WHERE status = 'active' ORDER BY price_usd, created")]
        return {"offers": [o for o in offers if o["available_coins"] > 0], "rates_usd": rates,
                "min_order_usd": self.a.market_min_usd, "order_minutes": self.a.market_order_minutes,
                "confirmations": {"BTC": self.a.market_btc_confs, "ETH": self.a.market_eth_confs},
                "escrow_fee_coins": FEE_BUFFER / COIN}

    def mine(self, key) -> dict:
        offers = self.store.q("SELECT * FROM offers WHERE key_id = ? ORDER BY created DESC LIMIT 50", key["id"])
        bought = self.store.q("SELECT * FROM orders WHERE key_id = ? ORDER BY created DESC LIMIT 50", key["id"])
        sold = self.store.q("SELECT o.* FROM orders o JOIN offers f ON o.offer = f.id WHERE f.key_id = ? "
                            "ORDER BY o.created DESC LIMIT 50", key["id"])
        return {"offers": [self.offer_view(o, True) for o in offers], "orders": [self.order_view(o) for o in bought],
                "sales": [self.order_view(o) for o in sold]}

    def order(self, key, oid: str) -> dict:
        o = self.store.one("SELECT * FROM orders WHERE id = ?", oid)
        f = o and self.store.one("SELECT key_id FROM offers WHERE id = ?", o["offer"])
        if o is None or key["id"] not in (o["key_id"], f["key_id"]):
            raise MarketError(404, "no such order", "not_found")
        return self.order_view(o)

    # -------------------------------------------------------------- offers

    def open_offer(self, key, b: dict) -> dict:
        try:
            coins = int(round(float(b.get("coins")) * COIN))
            price = float(b.get("price_usd"))
        except (TypeError, ValueError):
            raise MarketError(400, "coins and price_usd must be numbers")
        if coins < COIN or not (0 < price < 1e6) or not math.isfinite(price):
            raise MarketError(400, "an offer sells at least 1 coin at a positive price_usd")
        btc, eth = (b.get("btc_address") or "").strip(), (b.get("eth_address") or "").strip()
        if not btc and not eth:
            raise MarketError(400, "give btc_address, eth_address or both: buyers pay you there")
        if btc and not RE_BTC.match(btc):
            raise MarketError(400, "btc_address is not a bitcoin mainnet address (bc1…, 1… or 3…)")
        if eth and not RE_ETH.match(eth):
            raise MarketError(400, "eth_address is not an Ethereum address (0x and 40 hex digits)")
        refund = (b.get("refund_address") or key["address"]).strip()
        if not RE_UWC.match(refund):
            raise MarketError(400, "refund_address is not a uwc1p… address")
        n = self.store.one("SELECT COUNT(*) n FROM offers WHERE key_id = ? AND status IN ('funding', 'active', 'closing')",
                           key["id"])["n"]
        if n >= self.a.market_max_offers:
            raise MarketError(429, f"at most {self.a.market_max_offers} open offers per key", "rate_limit_exceeded")
        oid = secrets.token_hex(8)
        keyfile = os.path.join(self.dir, f"{oid}.json")
        self.chain.run("keygen", keyfile)
        os.chmod(keyfile, 0o600)
        info = self.chain.key_info(keyfile)
        self.store.q("INSERT INTO offers (id, key_id, created, status, coins, price_usd, btc_address, eth_address, "
                     "refund_address, escrow_keyfile, escrow_address) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     oid, key["id"], time.time(), "funding", coins, price, btc or None, eth or None, refund,
                     keyfile, info["address"])
        self.gw_log(f"offer {oid}: {coins / COIN} coins at ${price} by key {key['account'][:12]}")
        return self.offer_view(self.store.one("SELECT * FROM offers WHERE id = ?", oid), True)

    def close_offer(self, key, oid: str) -> dict:
        o = self.store.one("SELECT * FROM offers WHERE id = ? AND key_id = ?", oid, key["id"])
        if o is None:
            raise MarketError(404, "no such offer", "not_found")
        if o["status"] in ("closing", "closed"):
            return self.offer_view(o, True)
        self.store.q("UPDATE offers SET status = 'closing' WHERE id = ?", oid)
        self.gw_log(f"offer {oid}: closing")
        return self.offer_view(self.store.one("SELECT * FROM offers WHERE id = ?", oid), True)

    # -------------------------------------------------------------- orders

    def new_order(self, key, b: dict) -> dict:
        cur = str(b.get("currency") or "").upper()
        if cur not in UNIT:
            raise MarketError(400, "currency must be BTC or ETH")
        try:
            coins = int(round(float(b.get("coins")) * COIN))
        except (TypeError, ValueError):
            raise MarketError(400, "coins must be a number")
        n = self.store.one("SELECT COUNT(*) n FROM orders WHERE key_id = ? AND status = 'awaiting_payment'", key["id"])["n"]
        if n >= self.a.market_max_orders:
            raise MarketError(429, f"at most {self.a.market_max_orders} unpaid orders per key", "rate_limit_exceeded")
        rate = self.rates.get()[cur]
        with self.lock:
            o = self.store.one("SELECT * FROM offers WHERE id = ?", str(b.get("offer") or ""))
            if o is None or o["status"] != "active":
                raise MarketError(404, "no such open offer", "not_found")
            if o["key_id"] == key["id"]:
                raise MarketError(400, "that is your own offer")
            pay_to = o["btc_address"] if cur == "BTC" else o["eth_address"]
            if not pay_to:
                raise MarketError(400, f"this offer does not take {cur}")
            if coins <= 0 or coins > self.free(o):
                raise MarketError(400, f"this offer has {self.free(o) / COIN:g} coins available")
            usd = coins / COIN * o["price_usd"]
            if usd < self.a.market_min_usd:
                raise MarketError(400, f"an order is at least ${self.a.market_min_usd:g} "
                                       f"({math.ceil(self.a.market_min_usd / o['price_usd']):,} coins here)")
            step = TAG_STEP[cur]
            base = math.ceil(Decimal(usd) / Decimal(rate) * UNIT[cur] / step) * step
            taken = {int(r["amount"]) for r in self.store.q(
                f"SELECT amount FROM orders WHERE pay_to = ? AND currency = ? AND (status IN ({','.join('?' * len(OPEN))}) "
                f"OR (status = 'expired' AND expires > ?))", pay_to, cur, *OPEN, time.time() - LATE_WINDOW)}
            free_tags = [t for t in range(1, TAG_MAX[cur] + 1) if base + t * step not in taken]
            if not free_tags:
                raise MarketError(429, "too many orders are waiting on this offer; try again later", "rate_limit_exceeded")
            amount = base + secrets.choice(free_tags[:20]) * step  # the smallest free tags: a few satoshi or gwei
            oid = secrets.token_hex(8)
            now = time.time()
            self.store.q("INSERT INTO orders (id, offer, key_id, created, expires, status, coins, currency, pay_to, "
                         "amount, usd, rate) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", oid, o["id"], key["id"], now,
                         now + self.a.market_order_minutes * 60, "awaiting_payment", coins, cur, pay_to, str(amount),
                         usd, rate)
        self.gw_log(f"order {oid}: {coins / COIN} coins of offer {o['id']} for {fmt_units(amount, cur)} {cur}")
        return self.order_view(self.store.one("SELECT * FROM orders WHERE id = ?", oid))

    # -------------------------------------------------------------- the watcher

    def run(self):
        while True:
            try:
                self.tick()
            except Exception:
                traceback.print_exc()
            time.sleep(self.a.market_poll)

    def tick(self):
        now = time.time()
        for o in self.store.q("SELECT * FROM offers WHERE status IN ('funding', 'active', 'closing')"):
            self.refresh_offer(o)
        self.store.q("UPDATE orders SET status = 'expired' WHERE status = 'awaiting_payment' AND expires < ?", now)
        watch = self.store.q("SELECT * FROM orders WHERE status IN ('awaiting_payment', 'seen', 'confirmed') "
                             "OR (status = 'expired' AND expires > ?)", now - LATE_WINDOW)
        groups: dict[tuple, list] = {}
        for o in watch:
            groups.setdefault((o["currency"], o["pay_to"]), []).append(dict(o))
        for (cur, addr), orders in groups.items():
            try:
                pays = btc_payments(addr, self.fetch) if cur == "BTC" else eth_payments(addr, self.fetch)
            except Exception as e:
                self.gw_log(f"market: {cur} explorer for {addr[:10]}…: {e}")
                continue
            self.apply(cur, orders, pays)
        for o in self.store.q("SELECT * FROM orders WHERE status = 'confirmed'"):
            self.deliver(o)
        for o in self.store.q("SELECT * FROM offers WHERE status = 'closing'"):
            self.refund(o)

    def refresh_offer(self, o):
        try:
            bal = int(self.chain.run("coins", o["escrow_keyfile"])["total"])
        except ChainError as e:
            self.gw_log(f"offer {o['id']}: {e}")
            return
        status = o["status"]
        if status in ("funding", "active"):  # a sold-out offer closes, refunding what is left of its fee coin
            status = "closing" if o["sold"] >= o["coins"] else ("active" if bal > FEE_RESERVE else "funding")
        self.store.q("UPDATE offers SET balance = ?, status = ? WHERE id = ?", bal, status, o["id"])

    def apply(self, cur: str, orders: list[dict], pays: list[dict]):
        used = {r["txid"] for r in self.store.q("SELECT txid FROM orders WHERE currency = ? AND txid IS NOT NULL", cur)}
        for o in orders:  # orders already holding a payment: update its confirmations
            if o["txid"]:
                p = next((p for p in pays if p["txid"] == o["txid"]), None)
                if p:
                    self.confirm(o, cur, p)
        waiting = [o for o in orders if not o["txid"]]
        for oid, p in match(pays, waiting, used).items():
            o = next(x for x in waiting if x["id"] == oid)
            try:
                self.store.q("UPDATE orders SET txid = ?, confirmations = ? WHERE id = ? AND txid IS NULL", p["txid"],
                             p["confirmations"], oid)
            except Exception:  # the payment was claimed by another order meanwhile (unique index)
                continue
            late = o["status"] == "expired"
            self.gw_log(f"order {oid}: payment {p['txid'][:12]} seen{' (late)' if late else ''}, {p['confirmations']} confirmations")
            if late:
                with self.lock:
                    f = self.store.one("SELECT * FROM offers WHERE id = ?", o["offer"])
                    if f is None or self.free(f) < o["coins"]:
                        self.store.q("UPDATE orders SET status = 'late_unfilled', error = ? WHERE id = ?",
                                     "paid after the order expired, and the offer no longer has the coins: "
                                     "ask the seller for a refund", oid)
                        continue
                    self.store.q("UPDATE orders SET status = 'seen' WHERE id = ?", oid)
            else:
                self.store.q("UPDATE orders SET status = 'seen' WHERE id = ?", oid)
            o.update(status="seen", txid=p["txid"])
            self.confirm(o, cur, p)

    def confirm(self, o: dict, cur: str, p: dict):
        need = self.a.market_btc_confs if cur == "BTC" else self.a.market_eth_confs
        status = "confirmed" if p["confirmations"] >= need and o["status"] in ("seen", "confirmed") else o["status"]
        self.store.q("UPDATE orders SET confirmations = ?, status = ? WHERE id = ? AND status IN ('seen', 'confirmed')",
                     p["confirmations"], status, o["id"])

    def deliver(self, o):
        f = self.store.one("SELECT * FROM offers WHERE id = ?", o["offer"])
        buyer = self.store.one("SELECT * FROM keys WHERE id = ?", o["key_id"])
        with self.lock:
            self.store.q("UPDATE orders SET status = 'delivering' WHERE id = ? AND status = 'confirmed'", o["id"])
            try:
                r = self.chain.run("deposit", f["escrow_keyfile"], str(o["coins"]), buyer["account"])
            except ChainError as e:  # e.g. the escrow's change is still unconfirmed: next round
                self.store.q("UPDATE orders SET status = 'confirmed', error = ? WHERE id = ?", f"delivery: {e}"[:300], o["id"])
                self.gw_log(f"order {o['id']}: delivery failed, retrying: {e}")
                return
            txid = r.get("txid") if isinstance(r, dict) else str(r).strip()[:80]
            self.store.q("UPDATE orders SET status = 'paid', delivered_tx = ?, done = ?, error = NULL WHERE id = ?",
                         txid, time.time(), o["id"])
            self.store.q("UPDATE offers SET sold = sold + ?, balance = balance - ? WHERE id = ?", o["coins"], o["coins"], f["id"])
        self.gw_log(f"order {o['id']}: {o['coins'] / COIN} coins delivered to {buyer['account'][:12]} (tx {txid})")

    def refund(self, f):
        n = self.store.one(f"SELECT COUNT(*) n FROM orders WHERE offer = ? AND status IN ({','.join('?' * len(OPEN))})",
                           f["id"], *OPEN)["n"]
        if n:
            return  # orders still waiting on this offer: refund once they finish or expire
        try:
            bal = int(self.chain.run("coins", f["escrow_keyfile"])["total"])
            if bal > SEND_MARGIN:
                r = self.chain.run("send", f["escrow_keyfile"], str(bal - SEND_MARGIN), f["refund_address"])
                txid = r.get("txid") if isinstance(r, dict) else str(r).strip()[:80]
            else:
                txid = None
        except ChainError as e:
            self.store.q("UPDATE offers SET note = ? WHERE id = ?", f"refund: {e}"[:300], f["id"])
            return
        self.store.q("UPDATE offers SET status = 'closed', refund_tx = ?, balance = 0, note = NULL WHERE id = ?", txid, f["id"])
        self.gw_log(f"offer {f['id']}: closed, {max(0, bal - SEND_MARGIN) / COIN} coins refunded (tx {txid})")

    def gw_log(self, msg: str):
        print(time.strftime("%H:%M:%S"), msg, flush=True)
