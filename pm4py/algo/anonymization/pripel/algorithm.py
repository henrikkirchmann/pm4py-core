'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''

from pm4py.algo.anonymization.pripel.variants import pripel
from pm4py.util import exec_utils
from enum import Enum
import pandas as pd
from typing import Optional, Dict, Any, Union, Tuple
from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.objects.conversion.log import converter as log_converter


class Variants(Enum):
    PRIPEL = pripel


DEFAULT_VARIANT = Variants.PRIPEL

VERSIONS = {Variants.PRIPEL}


def apply(log: Union[EventLog, pd.DataFrame], epsilon: float, n: int, k: int, blacklist: set = None, variant=DEFAULT_VARIANT):
    """
    PRIPEL (Privacy-preserving event log publishing with contextual information) is a framework to publish event logs that fulfill differential privacy.

    Parameters
    -------------
    log
        Event log
    epsilon
        Strength of the differential privacy guarantee
    n
        Maximum prefix of considered traces for the trace-variant-query
    k
        Pruning parameter of the trace-variant-query. At least k traces must appear in a noisy variant count to be part of the result of the query
    variant
        Variant of the algorithm to use:
        - Variants.PRIPEL

    Returns
    ------------
    anonymised_log
        Anonymised event log
    """
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    return exec_utils.get_variant(variant).apply(log, epsilon, n, k, blacklist = None)