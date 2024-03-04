# In order to make classes available as e.g.
#   ap.pad
# instead of
#   ap.pad.pad

__all__ = ['pad', 'canvas', 'overlay', 'tools', 'style']

from pad     import pad
from canvas  import canvas
from overlay import overlay
from . import tools
from . import style
from style import colours
from style import colours_light
