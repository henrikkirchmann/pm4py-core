import datetime
import sys
import diffprivlib.mechanisms as privacyMechanisms
import random
from collections import deque
from datetime import timedelta

from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_import_factory
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log import log as event_log
import numpy as np
from dateutil.tz import tzutc
from scipy.optimize import linear_sum_assignment




TRACE_START = "TRACE_START"
TRACE_END = "TRACE_END"
EVENT_DELIMETER = ">>>"


def privatize_tracevariants(log, epsilon, P, N):
    # transform log into event view and get prefix frequencies
    print("Retrieving true prefix frequencies", end='')
    event_int_mapping = create_event_int_mapping(log)
    known_prefix_frequencies = get_prefix_frequencies_from_log(log)
    events = list(event_int_mapping.keys())
    events.remove(TRACE_START)
    print("Done")

    final_frequencies = {}
    trace_frequencies = {"": 0}
    for n in range(1, N + 1):
        # get prefix_frequencies, using either known frequency, or frequency of parent, or 0
        trace_frequencies = get_prefix_frequencies_length_n(trace_frequencies, events, n, known_prefix_frequencies)
        # laplace_mechanism
        trace_frequencies = apply_laplace_noise_tf(trace_frequencies, epsilon)

        # prune
        trace_frequencies = prune_trace_frequencies(trace_frequencies, P, known_prefix_frequencies)
        # print(trace_frequencies)
        # add finished traces to output, remove from list, sanity checks
        new_frequencies = {}
        for entry in trace_frequencies.items():
            if TRACE_END in entry[0]:
                final_frequencies[entry[0]] = entry[1]
            else:
                new_frequencies[entry[0]] = entry[1]
        trace_frequencies = new_frequencies
        # print(trace_frequencies)
        print(n)
    return generate_pm4py_log(final_frequencies)


def create_event_int_mapping(log):
    event_name_list = []
    for trace in log:
        for event in trace:
            event_name = event["concept:name"]
            if not str(event_name) in event_name_list:
                event_name_list.append(event_name)
    event_int_mapping = {}
    event_int_mapping[TRACE_START] = 0
    current_int = 1
    for event_name in event_name_list:
        event_int_mapping[event_name] = current_int
        current_int = current_int + 1
    event_int_mapping[TRACE_END] = current_int
    return event_int_mapping


def get_prefix_frequencies_from_log(log):
    prefix_frequencies = {}
    for trace in log:
        current_prefix = ""
        for event in trace:
            current_prefix = current_prefix + event["concept:name"] + EVENT_DELIMETER
            if current_prefix in prefix_frequencies:
                frequency = prefix_frequencies[current_prefix]
                prefix_frequencies[current_prefix] += 1
            else:
                prefix_frequencies[current_prefix] = 1
        current_prefix = current_prefix + TRACE_END
        if current_prefix in prefix_frequencies:
            frequency = prefix_frequencies[current_prefix]
            prefix_frequencies[current_prefix] += 1
        else:
            prefix_frequencies[current_prefix] = 1
    return prefix_frequencies


def get_prefix_frequencies_length_n(trace_frequencies, events, n, known_prefix_frequencies):
    prefixes_length_n = {}
    for prefix, frequency in trace_frequencies.items():
        for new_prefix in pref(prefix, events, n):
            if new_prefix in known_prefix_frequencies:
                new_frequency = known_prefix_frequencies[new_prefix]
                prefixes_length_n[new_prefix] = new_frequency
            else:
                prefixes_length_n[new_prefix] = 0
    return prefixes_length_n


def prune_trace_frequencies(trace_frequencies, P, known_prefix_frequencies):  # was macht known_prefix_frequencies?
    pruned_frequencies = {}
    for entry in trace_frequencies.items():
        if entry[1] >= P:
            pruned_frequencies[entry[0]] = entry[1]
    return pruned_frequencies


def pref(prefix, events, n):
    prefixes_length_n = []
    if not TRACE_END in prefix:
        for event in events:
            current_prefix = ""
            if event == TRACE_END:
                current_prefix = prefix + event
            else:
                current_prefix = prefix + event + EVENT_DELIMETER
            prefixes_length_n.append(current_prefix)
    return prefixes_length_n


def apply_laplace_noise_tf(trace_frequencies, epsilon):
    lambd = 1 / epsilon
    for trace_frequency in trace_frequencies:
        noise = int(np.random.laplace(0, lambd))
        trace_frequencies[trace_frequency] = trace_frequencies[trace_frequency] + noise
        if trace_frequencies[trace_frequency] < 0:
            trace_frequencies[trace_frequency] = 0
    return trace_frequencies


def generate_pm4py_log(trace_frequencies):
    log = event_log.EventLog()
    trace_count = 0
    for variant in trace_frequencies.items():
        frequency = variant[1]
        activities = variant[0].split(EVENT_DELIMETER)
        for i in range(0, frequency):
            trace = event_log.Trace()
            trace.attributes["concept:name"] = trace_count
            trace_count = trace_count + 1
            for activity in activities:
                if not TRACE_END in activity:
                    event = event_log.Event()
                    event["concept:name"] = str(activity)
                    event["time:timestamp"] = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=tzutc())
                    trace.append(event)
            log.append(trace)
    return log

def apply_pripel(log, epsilon, N, k, blackList=None):

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

    tv_query_log = privatize_tracevariants(log, epsilon, k, N)

    if (len(tv_query_log) == 0):
        raise ValueError(
            "Pruning parameter k is too high. The result of the trace variant query is empty. At least k traces must appear "
            "in a noisy variant count to be part of the result of the query.")

    endtime_tv_query = datetime.datetime.now()
    print("Time of TV Query: " + str((endtime_tv_query - starttime_tv_query)))
    starttime_trace_matcher = datetime.datetime.now()

    traceMatcher = TraceMatcher(tv_query_log, log)
    matchedLog = traceMatcher.matchQueryToLog()

    print(len(matchedLog))
    endtime_trace_matcher = datetime.datetime.now()
    print("Time of TraceMatcher: " + str((endtime_trace_matcher - starttime_trace_matcher)))

    distributionOfAttributes = traceMatcher.getAttributeDistribution()
    occurredTimestamps, occurredTimestampDifferences = traceMatcher.getTimeStampData()
    print(min(occurredTimestamps))
    starttime_attribute_anonymizer = datetime.datetime.now()
    attributeAnonymizer = AttributeAnonymizer()
    anonymizedLog, attributeDistribution = attributeAnonymizer.anonymize(matchedLog, distributionOfAttributes, epsilon,
                                                                         occurredTimestampDifferences, occurredTimestamps)
    endtime_attribute_anonymizer = datetime.datetime.now()
    print("Time of attribute anonymizer: " + str(endtime_attribute_anonymizer - starttime_attribute_anonymizer))
    endtime = datetime.datetime.now()
    print("Complete Time: " + str((endtime - starttime)))
    print("Time of TV Query: " + str((endtime_tv_query - starttime_tv_query)))
    print("Time of TraceMatcher: " + str((endtime_trace_matcher - starttime_trace_matcher)))
    print("Time of attribute anonymizer: " + str(endtime_attribute_anonymizer - starttime_attribute_anonymizer))
    print(freq(attributeDistribution))
    return anonymizedLog


def apply(log: EventLog, epsilon: float, n: int, k: int, blackList: set = None):
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

    return apply_pripel(log, epsilon, n, k, blackList=None)

class TraceMatcher:
    def __init__(self, tv_query_log, log):
        self.__timestamp = "time:timestamp"
        self.__allTimestamps = list()
        self.__allTimeStampDifferences = list()
        self.__distanceMatrix = dict()
        self.__trace_variants_query = self.__addTraceToAttribute(tv_query_log)
        self.__trace_variants_log = self.__addTraceToAttribute(log)
        attributeBlacklist = self.__getBlacklistOfAttributes()
        self.__distributionOfAttributes, self.__eventStructure = self.__getDistributionOfAttributesAndEventStructure(
            log, attributeBlacklist)
        self.__query_log = tv_query_log
        self.__log = log

    def __addTraceToAttribute(self, log):
        trace_variants = dict()
        for trace in log:
            variant = ""
            for event in trace:
                variant = variant + "@" + event["concept:name"]
            trace.attributes["variant"] = variant
            traceSet = trace_variants.get(variant, set())
            traceSet.add(trace)
            trace_variants[variant] = traceSet
        return trace_variants

    def __getBlacklistOfAttributes(self):
        blacklist = set()
        blacklist.add("concept:name")
        blacklist.add(self.__timestamp)
        blacklist.add("variant")
        blacklist.add("EventID")
        blacklist.add("OfferID")
        blacklist.add("matricola")
        return blacklist

    def __handleVariantsWithSameCount(self, variants, traceMatching):
        for variant in variants:
            for trace in self.__trace_variants_query[variant]:
                traceMatching[trace.attributes["concept:name"]] = self.__trace_variants_log[variant].pop()
            del self.__trace_variants_log[variant]
            del self.__trace_variants_query[variant]

    def __handleVariantsUnderrepresentedInQuery(self, variants, traceMatching):
        for variant in variants:
            if variant in self.__trace_variants_query:
                for trace in self.__trace_variants_query.get(variant, list()):
                    traceMatching[trace.attributes["concept:name"]] = self.__trace_variants_log[variant].pop()
                del self.__trace_variants_query[variant]

    def __handleVariantsOverrepresentedInQuery(self, variants, traceMatching):
        for variant in variants:
            for trace in self.__trace_variants_log[variant]:
                traceFromQuery = self.__trace_variants_query[variant].pop()
                traceMatching[traceFromQuery.attributes["concept:name"]] = trace
            del self.__trace_variants_log[variant]

    def __getDistanceVariants(self, variant1, variant2):
        if variant1 not in self.__distanceMatrix:
            self.__distanceMatrix[variant1] = dict()
        if variant2 not in self.__distanceMatrix[variant1]:
            distance = levenshtein(variant1, variant2)
            self.__distanceMatrix[variant1][variant2] = distance
        else:
            distance = self.__distanceMatrix[variant1][variant2]
        return distance

    def __findCLosestVariantInLog(self, variant, log):
        closestVariant = None
        closestDistance = sys.maxsize
        for comparisonVariant in log.keys():
            distance = self.__getDistanceVariants(variant, comparisonVariant)
            if distance < closestDistance:
                closestVariant = comparisonVariant
                closestDistance = distance
        return closestVariant

    def __findOptimalMatches(self):
        rows = list()
        for traceQuery in self.__query_log:
            row = list()
            for traceLog in self.__log:
                row.append(self.__getDistanceVariants(traceQuery.attributes["variant"], traceLog.attributes["variant"]))
            rows.append(row)
        distanceMatrix = np.array(rows)
        row_ind, col_ind = linear_sum_assignment(distanceMatrix)
        traceMatching = dict()
        for (traceQueryPos, traceLogPos) in zip(row_ind, col_ind):
            traceMatching[self.__query_log[traceQueryPos].attributes["concept:name"]] = self.__log[traceLogPos]
        return traceMatching

    def __matchTraces(self, traceMatching):
        for variant in self.__trace_variants_query.keys():
            closestVariant = self.__findCLosestVariantInLog(variant, self.__trace_variants_log)
            for trace in self.__trace_variants_query[variant]:
                traceMatching[trace.attributes["concept:name"]] = self.__trace_variants_log[closestVariant].pop()
                if not self.__trace_variants_log[closestVariant]:
                    del self.__trace_variants_log[closestVariant]
                    if self.__trace_variants_log:
                        closestVariant = self.__findCLosestVariantInLog(variant, self.__trace_variants_log)
                    else:
                        return

    def __getTraceMatching(self):
        traceMatching = dict()
        variantsWithSameCount = set()
        variantsUnderepresentedInQuery = set()
        variantsOverepresentedInQuery = set()
        for variant in self.__trace_variants_log.keys():
            if len(self.__trace_variants_log[variant]) == len(self.__trace_variants_query.get(variant, set())):
                variantsWithSameCount.add(variant)
            elif len(self.__trace_variants_log[variant]) > len(self.__trace_variants_query.get(variant, set())) and len(
                    self.__trace_variants_query.get(variant, set())) != set():
                variantsUnderepresentedInQuery.add(variant)
            elif len(self.__trace_variants_log[variant]) < len(self.__trace_variants_query.get(variant, 0)):
                variantsOverepresentedInQuery.add(variant)
        self.__handleVariantsWithSameCount(variantsWithSameCount, traceMatching)
        self.__handleVariantsUnderrepresentedInQuery(variantsUnderepresentedInQuery, traceMatching)
        self.__handleVariantsOverrepresentedInQuery(variantsOverepresentedInQuery, traceMatching)
        self.__matchTraces(traceMatching)
        return traceMatching

    def __resolveTrace(self, traceInQuery, correspondingTrace, distributionOfAttributes):
        eventStacks = self.__transformTraceInEventStack(correspondingTrace)
        previousEvent = None
        for eventNr in range(0, len(traceInQuery)):
            currentEvent = traceInQuery[eventNr]
            activity = currentEvent["concept:name"]
            latestTimeStamp = self.__getLastTimestampTraceResolving(traceInQuery, eventNr)
            if activity in eventStacks:
                currentEvent = self.__getEventAndUpdateFromEventStacks(activity, eventStacks)
                if currentEvent[self.__timestamp] < latestTimeStamp:
                    currentEvent[self.__timestamp] = self.__getNewTimeStamp(previousEvent, currentEvent, eventNr,
                                                                            distributionOfAttributes)
            else:
                currentEvent = self.__createRandomNewEvent(currentEvent, activity, distributionOfAttributes,
                                                           previousEvent, eventNr)
            traceInQuery[eventNr] = currentEvent
            previousEvent = currentEvent
            self.__debugCheckTimeStamp(traceInQuery, eventNr)
        return traceInQuery

    def __getEventAndUpdateFromEventStacks(self, activity, eventStacks):
        event = eventStacks[activity].popleft()
        if not eventStacks[activity]:
            del eventStacks[activity]
        return event

    def __debugTraceTimestamps(self, trace):
        for eventNr in range(0):
            self.__debugCheckTimeStamp(trace, eventNr)

    def __debugCheckTimeStamp(self, trace, eventNr):
        if eventNr > 0:
            if trace[eventNr - 1][self.__timestamp] > trace[eventNr][self.__timestamp]:
                print("Fuck")

    def __getLastTimestampTraceResolving(self, trace, eventNr):
        if eventNr == 0:
            latestTimeStamp = trace[eventNr][self.__timestamp]
        else:
            latestTimeStamp = trace[eventNr - 1][self.__timestamp]
        return latestTimeStamp

    def __transformTraceInEventStack(self, trace):
        eventStacks = dict()
        for event in trace:
            stack = eventStacks.get(event["concept:name"], deque())
            stack.append(event)
            eventStacks[event["concept:name"]] = stack
        return eventStacks

    def __createRandomNewEvent(self, event, activity, distributionOfAttributes, previousEvent, eventNr):
        for attribute in self.__eventStructure[activity]:
            if attribute in distributionOfAttributes and attribute not in event and attribute != self.__timestamp:
                event[attribute] = random.choice(distributionOfAttributes[attribute])
            elif attribute == self.__timestamp:
                event[self.__timestamp] = self.__getNewTimeStamp(previousEvent, event, eventNr,
                                                                 distributionOfAttributes)
        return event

    def __getNewTimeStamp(self, previousEvent, currentEvent, eventNr, distributionOfAttributes):
        if eventNr == 0:
            timestamp = random.choice(self.__allTimestamps)
        else:
            if previousEvent["concept:name"] in distributionOfAttributes[self.__timestamp]:
                timestamp = previousEvent[self.__timestamp] + random.choice(
                    distributionOfAttributes[self.__timestamp][previousEvent["concept:name"]].get(
                        currentEvent["concept:name"], self.__allTimeStampDifferences))
            else:
                timestamp = previousEvent[self.__timestamp] + random.choice(self.__allTimeStampDifferences)
        return timestamp

    def __resolveTraceMatching(self, traceMatching, distributionOfAttributes, fillUp):
        log = event_log.EventLog()
        for trace in self.__query_log:
            traceID = trace.attributes["concept:name"]
            if fillUp or traceID in traceMatching:
                matchedTrace = self.__resolveTrace(trace, traceMatching.get(traceID, list()), distributionOfAttributes)
                self.__debugTraceTimestamps(matchedTrace)
                log.append(matchedTrace)
        return log

    def __handleAttributesOfDict(self, dictOfAttributes, distributionOfAttributes, attributeBlacklist,
                                 previousEvent=None):
        for attribute in dictOfAttributes.keys():
            if attribute not in attributeBlacklist:
                distribution = distributionOfAttributes.get(attribute, list())
                distribution.append(dictOfAttributes[attribute])
                distributionOfAttributes[attribute] = distribution
            elif attribute == self.__timestamp and previousEvent is not None:
                self.__handleTimeStamp(distributionOfAttributes, previousEvent, dictOfAttributes)

    def __handleTimeStamp(self, distributionOfAttributes, previousEvent, currentEvent):
        timeStampsDicts = distributionOfAttributes.get(self.__timestamp, dict())
        activityDict = timeStampsDicts.get(previousEvent["concept:name"], dict())
        timeStampsDicts[previousEvent["concept:name"]] = activityDict
        distribution = activityDict.get(currentEvent["concept:name"], list())
        timeStampDifference = currentEvent[self.__timestamp] - previousEvent[self.__timestamp]
        distribution.append(timeStampDifference)
        activityDict[currentEvent["concept:name"]] = distribution
        distributionOfAttributes[self.__timestamp] = timeStampsDicts
        self.__allTimestamps.append(currentEvent[self.__timestamp])
        self.__allTimeStampDifferences.append(timeStampDifference)

    def __getDistributionOfAttributesAndEventStructure(self, log, attributeBlacklist):
        distributionOfAttributes = dict()
        eventStructure = dict()
        for trace in log:
            self.__handleAttributesOfDict(trace.attributes, distributionOfAttributes, attributeBlacklist)
            previousEvent = None
            currentEvent = None
            for eventNr in range(0, len(trace)):
                if currentEvent is not None:
                    previousEvent = currentEvent
                currentEvent = trace[eventNr]
                self.__handleAttributesOfDict(currentEvent, distributionOfAttributes, attributeBlacklist, previousEvent)
                if not currentEvent["concept:name"] in eventStructure:
                    attributesOfEvent = set(currentEvent.keys())
                    attributesOfEvent.remove("concept:name")
                    eventStructure[currentEvent["concept:name"]] = attributesOfEvent
        return distributionOfAttributes, eventStructure

    def matchQueryToLog(self, fillUp=True, greedy=False):
        if greedy:
            traceMatching = self.__getTraceMatching()
        else:
            traceMatching = self.__findOptimalMatches()
        matched_log = self.__resolveTraceMatching(traceMatching, self.__distributionOfAttributes, fillUp)
        return matched_log

    def getAttributeDistribution(self):
        return self.__distributionOfAttributes

    def getTimeStampData(self):
        return self.__allTimestamps, self.__allTimeStampDifferences

# This algorithm was copied at 16/Nov/2018 from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python and applied to activity sequences
delimter = "@"


def length(s):
    return s.count(delimter) + 1


def enumerateSequence(s):
    list = s.split(delimter)
    return enumerate(list, 0)


def levenshtein(s1, s2):
    if length(s1) < length(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if length(s2) == 0:
        return length(s1)

    previous_row = range(length(s2) + 1)
    for i, c1 in enumerateSequence(s1):
        current_row = [i + 1]
        for j, c2 in enumerateSequence(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # then s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class AttributeAnonymizer:

    def __init__(self):
        self.__timestamp = "time:timestamp"
        self.__blacklist = self.__getBlacklistOfAttributes()
        self.__sensitivity = "sensitivity"
        self.__max = "max"
        self.__min = "min"
        self.__infectionSuspected = list()

    def __getBlacklistOfAttributes(self):
        blacklist = set()
        blacklist.add("concept:name")
        blacklist.add(self.__timestamp)
        blacklist.add("variant")
        blacklist.add("EventID")
        blacklist.add("OfferID")
        blacklist.add("matricola")
        return blacklist

    def __retrieveAttributeDomains(self, distributionOfAttributes, dataTypesOfAttributes):
        domains = dict()
        for attribute in dataTypesOfAttributes.keys():
            if dataTypesOfAttributes[attribute] in (int, float):
                domain = dict()
                domain[self.__max] = max(distributionOfAttributes[attribute])
                domain[self.__min] = min(distributionOfAttributes[attribute])
                domain[self.__sensitivity] = abs(domain[self.__max] - domain[self.__min])
                domains[attribute] = domain
        return domains

    def __determineDataType(self, distributionOfAttributes):
        dataTypesOfAttributes = dict()
        for attribute in distributionOfAttributes.keys():
            if attribute not in self.__blacklist:
                dataTypesOfAttributes[attribute] = type(distributionOfAttributes[attribute][0])
        return dataTypesOfAttributes

    def __getPotentialValues(self, distributionOfAttributes, dataTypesOfAttributes):
        potentialValues = dict()
        for attribute in dataTypesOfAttributes:
            if dataTypesOfAttributes[attribute] is str:
                distribution = distributionOfAttributes[attribute]
                values = set(distribution)
                potentialValues[attribute] = values
        return potentialValues

    def __setupBooleanMechanism(self, epsilon):
        binaryMechanism = privacyMechanisms.Binary(epsilon=epsilon, value0=str(True), value1=str(False))
        return binaryMechanism

    def __anonymizeAttribute(self, value, mechanism):
        isBoolean = False
        isInt = False
        if mechanism is not None:
            if type(value) is bool:
                isBoolean = True
                value = str(value)
            if type(value) is int:
                isInt = True
            value = mechanism.randomise(value)
            if isBoolean:
                value = eval(value)
            if isInt:
                value = int(round(value))
        return value

    def __addBooleanMechanisms(self, epsilon, mechanisms, dataTypesOfAttributes):
        binaryMechanism = self.__setupBooleanMechanism(epsilon)
        for attribute in dataTypesOfAttributes.keys():
            if dataTypesOfAttributes[attribute] is bool:
                mechanisms[attribute] = binaryMechanism
        return mechanisms

    def __addNumericMechanisms(self, epsilon, mechanisms, domains):
        for attribute in domains.keys():
            sensitivity = domains[attribute][self.__sensitivity]
            lowerDomainBound = domains[attribute][self.__min]
            upperDomainBound = domains[attribute][self.__max]
            laplaceMechanism = privacyMechanisms.LaplaceBoundedDomain(epsilon=epsilon, sensitivity=sensitivity,
                                                                      lower=lowerDomainBound, upper=upperDomainBound)
            mechanisms[attribute] = laplaceMechanism
        return mechanisms

    def __setupUniformUtilityList(self, potentialValues):
        utilityList = [[x, y, 1] for x in potentialValues for y in potentialValues]
        return utilityList

    def __addCategoricalMechanisms(self, epsilon, mechanisms, dataTypesOfAttributes, potentialValues):
        for attribute in dataTypesOfAttributes.keys():
            if dataTypesOfAttributes[attribute] is str:
                utilityList = self.__setupUniformUtilityList(potentialValues[attribute])
                if len(utilityList) > 0:
                    exponentialMechanism = privacyMechanisms.ExponentialCategorical(epsilon=epsilon,
                                                                                    utility_list=utilityList)
                    mechanisms[attribute] = exponentialMechanism
        return mechanisms

    def __getTimestamp(self, trace, eventNr, allTimestamps):
        if eventNr <= 0:
            return min(allTimestamps)
        elif eventNr >= len(trace):
            return max(allTimestamps)
        else:
            return trace[eventNr][self.__timestamp]

    def __anonymizeTimeStamps(self, timestamp, previousTimestamp, nextTimestamp, sensitivity, minTimestampDifference,
                              mechanism):
        upperPotentialDifference = (nextTimestamp - previousTimestamp).total_seconds()
        currentDifference = (timestamp - previousTimestamp).total_seconds()
        if upperPotentialDifference < 0:
            upperPotentialDifference = currentDifference
        mechanism.sensitivity = sensitivity
        mechanism.lower = minTimestampDifference
        mechanism.upper = upperPotentialDifference
        timestamp = previousTimestamp + timedelta(seconds=currentDifference)
        return timestamp

    def __setupMechanisms(self, epsilon, distributionOfAttributes, lower, upper, sensitivity):
        mechanisms = dict()
        dataTypesOfAttributes = self.__determineDataType(distributionOfAttributes)
        mechanisms = self.__addBooleanMechanisms(epsilon, mechanisms, dataTypesOfAttributes)
        domains = self.__retrieveAttributeDomains(distributionOfAttributes, dataTypesOfAttributes)
        mechanisms = self.__addNumericMechanisms(epsilon, mechanisms, domains)
        potentialValues = self.__getPotentialValues(distributionOfAttributes, dataTypesOfAttributes)
        mechanisms = self.__addCategoricalMechanisms(epsilon, mechanisms, dataTypesOfAttributes, potentialValues)
        mechanisms[self.__timestamp] = privacyMechanisms.LaplaceBoundedDomain(epsilon=epsilon, lower=lower, upper=upper,
                                                                              sensitivity=sensitivity)
        return mechanisms

    def __getTimestampDomain(self, trace, eventNr, distributionOfTimestamps, allTimestampDifferences):
        timestampDomain = self.__domainTimestampData.get(trace[eventNr - 1]["concept:name"], None)
        if timestampDomain is not None:
            timestampDomain = timestampDomain.get(trace[eventNr]["concept:name"], None)
        if timestampDomain is None:
            timestampDistribution = None
            if eventNr != 0:
                dictTimestampDifference = distributionOfTimestamps.get(trace[eventNr - 1]["concept:name"], None)
                if dictTimestampDifference is not None:
                    timestampDistribution = dictTimestampDifference.get(trace[eventNr]["concept:name"], None)
            if timestampDistribution is None:
                maxTimestampDifference = self.__maxAllTimestampDifferences
                minTimestampDifference = self.__minAllTimestampDifferences
            else:
                maxTimestampDifference = max(timestampDistribution)
                minTimestampDifference = min(timestampDistribution)
            sensitivity = abs(maxTimestampDifference - minTimestampDifference).total_seconds()
            sensitivity = max(sensitivity, 1.0)
            timestampDomain = dict()
            timestampDomain["sensitivity"] = sensitivity
            timestampDomain["minTimeStampInLog"] = min(allTimestampDifferences).total_seconds()
            if self.__domainTimestampData.get(trace[eventNr - 1]["concept:name"], None) is None:
                self.__domainTimestampData[trace[eventNr - 1]["concept:name"]] = dict()
            self.__domainTimestampData[trace[eventNr - 1]["concept:name"]][
                trace[eventNr]["concept:name"]] = timestampDomain
        return timestampDomain["sensitivity"], timestampDomain["minTimeStampInLog"]

    def __performTimestampShift(self, trace, mechanism):
        beginOfTrace = trace[0][self.__timestamp]
        deltaBeginOfLogToTrace = (self.__minAllTimestamp - beginOfTrace).total_seconds()
        endOfTrace = trace[-1][self.__timestamp]
        traceDuration = (endOfTrace - beginOfTrace).total_seconds()
        deltaEndOfLogToTrace = (self.__maxAllTimestamp - beginOfTrace).total_seconds()
        upperBound = deltaEndOfLogToTrace - traceDuration
        if deltaBeginOfLogToTrace >= upperBound:
            upperBound = abs((self.__maxAllTimestamp - beginOfTrace).total_seconds())
        mechanism.lower = deltaBeginOfLogToTrace
        mechanism.upper = upperBound
        timestampShift = timedelta(seconds=mechanism.randomise(0.0))
        for event in trace:
            event[self.__timestamp] = event[self.__timestamp] + timestampShift
            if event[self.__timestamp] < self.__minAllTimestamp:
                print("That should not happen")

    def anonymize(self, log, distributionOfAttributes, epsilon, allTimestampDifferences, allTimestamps):
        print("Setting up the mechanisms")
        starttime = datetime.datetime.now()
        self.__maxAllTimestampDifferences = max(allTimestampDifferences)
        self.__minAllTimestampDifferences = min(allTimestampDifferences)
        self.__maxAllTimestamp = max(allTimestamps)
        self.__minAllTimestamp = min(allTimestamps)
        sensitivity = (self.__maxAllTimestamp - self.__minAllTimestamp).total_seconds()
        # lower and upper values are just for initialisation, they get later overwritten in __anonymizeTimeStamps
        # and __performTimestampShift
        lower = 0
        upper = 1
        timeShiftMechanism = privacyMechanisms.LaplaceBoundedDomain(epsilon=epsilon, sensitivity=sensitivity,
                                                                    lower=lower, upper=upper)
        mechanisms = self.__setupMechanisms(epsilon, distributionOfAttributes, lower, upper, sensitivity)
        self.__domainTimestampData = dict()
        endtime = datetime.datetime.now()
        time = endtime - starttime
        print("Done with setting up mechanisms after " + str(time))
        i = 0
        for trace in log:
            for eventNr in range(0, len(trace)):
                event = trace[eventNr]
                for attribute in event.keys():
                    if attribute != self.__timestamp:
                        event[attribute] = self.__anonymizeAttribute(event[attribute], mechanisms.get(attribute, None))
                        if attribute == "InfectionSuspected" and eventNr == 0:
                            self.__infectionSuspected.append(event[attribute])
                    elif eventNr > 0:
                        previousTimestamp = self.__getTimestamp(trace, eventNr - 1, allTimestamps)
                        nextTimestamp = self.__getTimestamp(trace, eventNr + 1, allTimestamps)
                        sensitivity, minTimestampDifference = self.__getTimestampDomain(trace, eventNr,
                                                                                        distributionOfAttributes[
                                                                                            self.__timestamp],
                                                                                        allTimestampDifferences)
                        event[attribute] = self.__anonymizeTimeStamps(event[attribute], previousTimestamp,
                                                                      nextTimestamp, sensitivity,
                                                                      minTimestampDifference,
                                                                      mechanisms[self.__timestamp])
                    elif eventNr == 0:
                        self.__performTimestampShift(trace, timeShiftMechanism)
            i = i + 1
            if (i % 100) == 0:
                print("Iteration " + str((i)))
        return log, self.__infectionSuspected
