"""Decorator for showing total elapsed time at the end of operation."""

from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def show_total_elapsed_time_decorator(func: F) -> F:
    """Decorator for showing total elapsed time at the end of operation."""
    import time
    from functools import wraps

    from overturemaestro._rich_progress import show_total_elapsed_time

    @wraps(func)
    def timeit_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()

        result = func(*args, **kwargs)

        verbosity_mode = kwargs.get("verbosity_mode", "silent")
        if not is_inside_decorated_calling_function() and not verbosity_mode == "silent":
            end_time = time.time()
            elapsed_seconds = end_time - start_time
            show_total_elapsed_time(elapsed_seconds)

        return result

    func.__is_timed__ = True  # type: ignore[attr-defined]

    return cast(F, timeit_wrapper)


def is_decorated(func: F) -> bool:
    """Get information if functions is already timed."""
    return hasattr(func, "__is_timed__")


# https://stackoverflow.com/a/39079070
def is_inside_decorated_calling_function() -> bool:
    """Finds if any of the calling functions is decorated."""
    import inspect

    for stack_frame in inspect.stack()[1:]:
        fr = stack_frame[0]
        co = fr.f_code
        for get in (
            lambda fr=fr, co=co: fr.f_globals[co.co_name],
            lambda fr=fr, co=co: getattr(fr.f_locals["self"], co.co_name),
            lambda fr=fr, co=co: getattr(fr.f_locals["cls"], co.co_name),
            lambda fr=fr, co=co: fr.f_back.f_locals[co.co_name],  # nested
            lambda fr=fr: fr.f_back.f_locals["func"],  # decorators
            lambda fr=fr: fr.f_back.f_locals["meth"],
            lambda fr=fr: fr.f_back.f_locals["f"],
        ):
            try:
                func = get()  # type: ignore[no-untyped-call]
            except (KeyError, AttributeError):
                pass
            else:
                if func.__code__ == co and is_decorated(func):
                    return True

    return False
