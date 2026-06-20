import os
import asyncio
import json
import threading
import traceback
from typing import List

import websockets

from .stock_monitor import load_watchlist, _market_prev_volume, _market_prev_turnover, _market_prev_price, _market_delta_history


FINNHUB_WS = "wss://ws.finnhub.io"


def _watchlist_symbols_for_provider(codes: List[str]) -> List[str]:
    """Convert local akshare-style codes (sh600519) to provider symbols.

    For Finnhub the symbol format is provider-specific (e.g. 'AAPL' or 'SSE:600519').
    Here we make a minimal best-effort mapping for common A-share codes: sh600519 -> SSE:600519.
    Users can supply provider-native symbols in the watchlist if needed.
    """
    out = []
    for c in codes:
        s = str(c)
        if s.lower().startswith("sh"):
            out.append(f"SSE:{s[2:]}")
        elif s.lower().startswith("sz"):
            out.append(f"SZSE:{s[2:]}")
        elif s.lower().startswith("bj"):
            out.append(f"BJ:{s[2:]}")
        else:
            # assume bare code like 600519 -> SSE
            if s.startswith("6"):
                out.append(f"SSE:{s}")
            else:
                out.append(s)
    return out


async def _finnhub_ws_loop(token: str):
    uri = FINNHUB_WS + f"?token={token}"
    try:
        async with websockets.connect(uri) as ws:
            # subscribe to current watchlist
            codes = load_watchlist()
            subs = _watchlist_symbols_for_provider(codes)
            for sym in subs:
                msg = json.dumps({"type": "subscribe", "symbol": sym})
                await ws.send(msg)

            async for message in ws:
                try:
                    data = json.loads(message)
                except Exception:
                    continue
                # Finnhub emits trades with symbol & price/volume for many markets.
                # We'll try to update our cached prev values so downstream logic can use them.
                if isinstance(data, dict) and data.get("type") == "trade":
                    for item in data.get("data", []):
                        sym = item.get("s") or item.get("symbol")
                        price = item.get("p") or item.get("price")
                        vol = item.get("v") or item.get("volume")
                        if not sym:
                            continue
                        # Try to normalize back to akshare full_code keys (sh600519 / sz000001)
                        # Very best-effort: if symbol like SSE:600519 -> sh600519
                        key = None
                        if ":" in sym:
                            parts = sym.split(":", 1)
                            if parts[0] in ("SSE", "SH"):
                                key = f"sh{parts[1]}"
                            elif parts[0] in ("SZSE", "SZ"):
                                key = f"sz{parts[1]}"
                            else:
                                key = sym
                        else:
                            key = sym

                        try:
                            if price is not None:
                                _market_prev_price[key] = float(price)
                            if vol is not None:
                                # some providers send trade size, convert to cumulative-like by adding
                                prev = _market_prev_volume.get(key, 0) or 0
                                _market_prev_volume[key] = prev + float(vol)
                        except Exception:
                            continue

    except Exception:
        traceback.print_exc()


def _start_finnhub_ws(token: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_finnhub_ws_loop(token))
    finally:
        loop.close()


def start_realtime_watchlist():
    """Start an optional realtime watchlist connector.

    Behavior:
    - If environment variable `FINNHUB_API_KEY` is set, tries to connect to Finnhub WebSocket and subscribe to watchlist symbols.
    - Otherwise the function returns without starting a connection (safe no-op).

    Note: Finnhub is used here as an example. For production you'd replace/adapt provider mapping and subscription logic.
    """
    token = os.environ.get("FINNHUB_API_KEY")
    if not token:
        print("Realtime: FINNHUB_API_KEY not set, skipping WebSocket realtime connector.")
        return

    print("Realtime: starting Finnhub WebSocket connector")
    t = threading.Thread(target=_start_finnhub_ws, args=(token,), daemon=True)
    t.start()


if __name__ == "__main__":
    start_realtime_watchlist()
