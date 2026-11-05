"""Decorator for showing total elapsed time at the end of operation."""

from rq_geo_toolkit.rich_utils import with_progress_display as show_total_elapsed_time_decorator

__all__ = ["show_total_elapsed_time_decorator"]
