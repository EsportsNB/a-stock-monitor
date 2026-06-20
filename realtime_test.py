import os
import asyncio
import json

try:
    import websockets
except Exception as e:
    print('MISSING_WEBSOCKETS', e)
    raise

KEY = os.environ.get('FINNHUB_API_KEY')
if not KEY:
    print('NO_KEY')
    raise SystemExit(1)

URI = f"wss://ws.finnhub.io?token={KEY}"

async def main():
    try:
        async with websockets.connect(URI) as ws:
            print('WS_CONNECTED')
            await ws.send(json.dumps({'type':'subscribe','symbol':'AAPL'}))
            try:
                m = await asyncio.wait_for(ws.recv(), timeout=5)
                print('RECV', m)
            except asyncio.TimeoutError:
                print('RECV_TIMEOUT')
    except Exception as e:
        print('WS_ERR', type(e).__name__, str(e))

if __name__ == '__main__':
    asyncio.run(main())
