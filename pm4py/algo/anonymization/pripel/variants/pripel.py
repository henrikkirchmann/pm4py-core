import datetime

from pm4py.algo.anonymization.pripel.util.AttributeAnonymizer import AttributeAnonymizer
from pm4py.algo.anonymization.pripel.util.TraceMatcher import TraceMatcher
from pm4py.algo.anonymization.pripel.util import trace_variant_query
from pm4py.objects.log.obj import EventLog


def apply_pripel(log, epsilon, n, k, blacklist):
    def freq(lst):
        d = {}
        for i in lst:
            if d.get(i):
                d[i] += 1
            else:
                d[i] = 1
        return d

    starttime = datetime.datetime.now()
    starttime_tv_query = datetime.datetime.now()

    tv_query_log = trace_variant_query.privatize_tracevariants(log, epsilon, k, n)

    if (len(tv_query_log) == 0):
        raise ValueError(
            "Pruning parameter k is too high. The result of the trace variant query is empty. At least k traces must appear "
            "in a noisy variant count to be part of the result of the query.")

    endtime_tv_query = datetime.datetime.now()
    print("Time of TV Query: " + str((endtime_tv_query - starttime_tv_query)))
    starttime_trace_matcher = datetime.datetime.now()

    traceMatcher = TraceMatcher(tv_query_log, log, blacklist)
    matchedLog = traceMatcher.matchQueryToLog()
    endtime_trace_matcher = datetime.datetime.now()
    print("Time of TraceMatcher: " + str((endtime_trace_matcher - starttime_trace_matcher)))

    distributionOfAttributes = traceMatcher.getAttributeDistribution()
    occurredTimestamps, occurredTimestampDifferences = traceMatcher.getTimeStampData()

    starttime_attribute_anonymizer = datetime.datetime.now()
    attributeAnonymizer = AttributeAnonymizer(blacklist)
    anonymizedLog, attributeDistribution = attributeAnonymizer.anonymize(matchedLog, distributionOfAttributes, epsilon,
                                                                         occurredTimestampDifferences,
                                                                         occurredTimestamps)
    endtime_attribute_anonymizer = datetime.datetime.now()
    print("Time of attribute anonymizer: " + str(endtime_attribute_anonymizer - starttime_attribute_anonymizer))
    endtime = datetime.datetime.now()
    print("Complete Time: " + str((endtime - starttime)))
    print("Time of TV Query: " + str((endtime_tv_query - starttime_tv_query)))
    print("Time of TraceMatcher: " + str((endtime_trace_matcher - starttime_trace_matcher)))
    print("Time of attribute anonymizer: " + str(endtime_attribute_anonymizer - starttime_attribute_anonymizer))
    print(freq(attributeDistribution))
    return anonymizedLog


def apply(log: EventLog, epsilon: float, n: int, k: int, blacklist: set):
    """
    PRIPEL (Privacy-preserving event log publishing with contextual information) is a framework to publish event logs
    that fulfill differential privacy.

    Parameters
    -------------
    log
        Event log
    epsilon
        Strength of the differential privacy guarantee
    n
        Maximum prefix of considered traces for the trace-variant-query
    k
        Pruning parameter of the trace-variant-query. At least k traces must appear in a noisy variant count to be part
        of the result of the query
    blacklist
        Some event logs contain attributes that are equivalent to a case id. For privacy reasons such attributes must be
        deleted from the anonymised log. We handle such attributes with this set
    Returns
    ------------
    anonymised_log
        Anonymised event log
    """

    return apply_pripel(log, epsilon, n, k, blacklist)
