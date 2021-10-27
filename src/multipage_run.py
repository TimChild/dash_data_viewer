from src.multipage_util import make_multipage_app, run_app
from pages import dash_labs_templates


if __name__ == '__main__':
    multipage_app = make_multipage_app(
        module_pages=[dash_labs_templates],
        app_name='Dat Viewer',
    )
    run_app(multipage_app, debug=True, debug_port=8051, real_port=50001, threaded=True)
