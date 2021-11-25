"""This can be imported from any page and used for caching slow functions in a fully threadsafe/process safe way

from dash_data_viewer.cache import cache

@cache.memoize()
def func(args)
    return val

Note: Need to run cache.init_app(app.server) at some point if not running from multipage
"""
from flask_caching import Cache


cache = Cache(config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
