from flask_caching import Cache


cache = Cache(config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
# Note: Need to run cache.init_app(app.server) at some point
