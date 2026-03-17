"""
Runtime context for analysis scripts.

Usage:
    from runtime import ctx

    ctx.status("Loading...")
    ctx.output(df.head())
    ctx.output(fig)

    # Access parameters passed from JavaScript
    activity_id = ctx.params  # if params is a string
    sport = ctx.params.sport  # if params is an object
"""


class Context:
    """Execution context populated by the worker before each script run."""

    def __init__(self):
        self.output = None   # Display values (text, DataFrame, figure)
        self.status = None   # Show transient message
        self.api_key = None
        self.params = None   # Parameters passed from JavaScript


ctx = Context()
