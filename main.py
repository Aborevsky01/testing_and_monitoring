'''
Uvicorn entrypoint.

Keep this file small so the service logic lives in the `ml_service` package.
'''

from ml_service.app import app  # noqa: F401

