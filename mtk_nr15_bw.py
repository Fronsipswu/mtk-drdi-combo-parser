"""Firmware NR15 bandwidth-pair grammar and explicit policy materialization."""

# Indexed BW-pair records, including unreachable enum-13 terminators. These
# are table grammar signatures, not device offsets or oracle-derived combos.
PAIR_TABLES = {
    1: ((10,5),(9,7),(7,9),(5,10)),
    3: ((10,3),(9,6),(8,7),(7,8),(6,9),(3,10)),
    2: ((10,5),(9,6),(9,7),(7,9),(6,9),(5,10),(13,13),(13,13)),
    4: ((10,3),(9,5),(9,6),(8,7),(7,8),(6,9),(5,9),(3,10)),
    6: ((9,3),(8,5),(8,6),(7,7),(6,8),(5,8),(3,9),(13,13)),
}


def table_signature():
    return bytes(value for mode in (1,3,2,4,6) for pair in PAIR_TABLES[mode] for value in pair)


def bandwidth_pairs(limits, mode, bandwidths, pair_tables=PAIR_TABLES):
    """Match the firmware bandwidth function, before per-CC SCS expansion.

    Mode 0 is the no-policy-reduction branch. Modes 1/3/5 are intra-band
    130/120/100-MHz policies; 2/4/6 are the corresponding inter-band policies.
    Callers must select policy explicitly and apply later SCS/hardware gates.
    """
    if len(limits) not in (1,2) or any(not 0 <= c <= 10 for c in limits):
        raise ValueError("NR15 requires one or two validated <=100-MHz band limits")
    intra = len(limits) == 1
    if mode == 0:
        return [(limits[0],limits[0])] if intra else [tuple(limits)]
    if mode not in ((1,3,5) if intra else (2,4,6)):
        raise ValueError("NR15 bandwidth mode does not match intra/inter-band shape")
    total = bandwidths[limits[0]]*2 if intra else sum(bandwidths[c] for c in limits)
    budget = {1:130,2:130,3:120,4:120,5:100,6:100}[mode]
    if total <= budget:
        return [(limits[0],limits[0])] if intra else [tuple(limits)]
    if mode == 5:
        code = 7 if limits[0] > 6 else 10
        return [(code,code)]
    if intra:
        result = [pair for pair in pair_tables[mode] if pair[0] <= limits[0]]
    else:
        reverse = limits[1] < limits[0]
        low,high = sorted(limits)
        result = []
        for a,b in pair_tables[mode]:
            if a > low or b > high:
                continue
            # Firmware exclusions for the 80+40/30 and 60+30 pair boundaries.
            threshold,excluded = (8,(5,8)) if mode == 6 else (9,(6,9)) if mode == 2 else (9,(5,9))
            if low >= threshold:
                reject = high > excluded[0] and (a,b) in (excluded,excluded[::-1])
            else:
                reject = low > excluded[0] and high >= threshold and a == excluded[0] and b == excluded[1]
            if reject:
                continue
            result.append((b,a) if reverse else (a,b))
    # Mode 6 returns zero variants when no pair closes; other modes fall back.
    return result if result or mode == 6 else [(10,10)]
