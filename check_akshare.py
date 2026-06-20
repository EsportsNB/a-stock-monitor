import akshare as ak
import inspect

print('akshare version:', getattr(ak, '__version__', 'unknown'))
for name in sorted(dir(ak)):
    lower = name.lower()
    if 'index' in lower or 'spot' in lower or 'zh' in lower:
        try:
            obj = getattr(ak, name)
            if inspect.isfunction(obj):
                print(name, inspect.signature(obj))
            else:
                print(name, type(obj).__name__)
        except Exception as e:
            print(name, 'ERROR', e)
