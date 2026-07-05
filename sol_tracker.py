"""Sol glasses tracking - thin compatibility shim.

The implementation now lives in ntuh.sol.* (split during the reorg). This module is kept so
existing ``from sol_tracker import ...`` imports continue to work unchanged.
"""
from ntuh.sol.connector import SolConnector, SDK_AVAILABLE, Camera, AsyncClient
from ntuh.sol.projector import (
    ScreenProjector3D, create_calibration_assets,
    BORDER_PIXEL_WIDTH, SCREEN_MARGIN_PIXELS, SCREEN_MARGIN_BOTTOM_PIXELS,
)
