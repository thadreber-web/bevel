"""TUI widgets: transcript, tool cards, diff viewer.

Nothing else lives here. A todo panel and a token meter are *not* built: the
task list is written by the ``todo`` tool (m09b) and read today with
``bevel todos``, and the token count is a line the app writes itself rather
than a widget. Neither has a milestone of its own yet, so neither is
advertised -- see :data:`bevel.core.agent.UNBUILT_SEAMS` for what the loop
itself reports as missing.
"""
