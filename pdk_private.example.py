"""PDK-specific values withheld under the foundry NDA.

Copy this file to ``pdk_private.py`` and fill in every value from your own
PDK documentation before running simulations.  The analysis scripts do not
need this file.
"""

# Spiral-inductor PDK-cell fixed geometry (um / layer index)
IND_W = None        # trace width
IND_SPACING = None  # turn spacing
IND_LAY = None      # top metal layer index
IND_GDIS = None     # guard-ring distance

# Minimum inner diameter permitted by the design rules (um)
D_IN_MIN_UM = None

# MOS-cap p-cell ring counts
CAP_BR = None
CAP_GR = None

# Simulation environment
PDK_INCLUDE = "/path/to/models/spectre/toplevel.scs"
BALUN_VA = "/path/to/rfLib/balun_ideal/veriloga/veriloga.va"
SPECTRE_BIN = "/path/to/bin/spectre"
REMOTE_HOST = "spectre-host"     # ssh alias of the machine running Spectre
REMOTE_DIR = "/tmp/bo_runs"
