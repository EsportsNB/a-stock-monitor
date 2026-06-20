from app.stock_monitor import get_index_snapshots

try:
    data = get_index_snapshots()
    print('OK')
    print(type(data))
    print(data)
except Exception as e:
    import traceback
    print('ERROR', type(e).__name__)
    print(str(e))
    traceback.print_exc()
