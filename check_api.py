import urllib.request

url = 'http://127.0.0.1:8000/api/stocks/list?page=1&page_size=20'
print('URL:', url)
try:
    with urllib.request.urlopen(url, timeout=20) as resp:
        data = resp.read().decode('utf-8', errors='replace')
        print('STATUS', resp.status)
        print('LENGTH', len(data))
        print('HEAD', data[:1200])
except Exception as e:
    import traceback
    print('ERROR', type(e).__name__)
    print(str(e))
    traceback.print_exc()
