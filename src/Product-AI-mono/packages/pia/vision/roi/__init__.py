def __getattr__(name):
    """Lazy import for BoundingBoxManager.

    BoundingBoxManager depends on filterpy (Kalman filter) via the SORT tracker.
    Modules that only need ROIManagerBase or batch_crop_region (e.g. perception_encoder)
    should not be forced to install filterpy.
    """
    if name == "BoundingBoxManager":
        from .bbox_manager import BoundingBoxManager

        return BoundingBoxManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BoundingBoxManager"]
