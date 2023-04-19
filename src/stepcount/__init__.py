name = "stepcount"
__author__ = "Shing Chan, Scott Small, Gert Mertes, Aiden Doherty"
__email__ = "shing.chan@ndph.ox.ac.uk, scott.small@ndph.ox.ac.uk, gert.mertes@ndph.ox.ac.uk, aiden.doherty@ndph.ox.ac.uk"
__maintainer__ = "Shing Chan"
__maintainer_email__ = "shing.chan@ndph.ox.ac.uk"
__license__ = "See LICENSE file."

__model_version__ = {
    "rf": "20230210",
    "ssl": "ssl-20230208"
}
__model_md5__ = {
    "rf": "8d803d175b792fff12f93ca37fbc9271",
    "ssl": "eea6179f079b554d5e2c8c98ccea8423"
}

from . import _version
__version__ = _version.get_versions()['version']
