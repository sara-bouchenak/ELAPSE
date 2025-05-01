from scipy.stats import ttest_rel
import numpy as np
def t_test(original, selection):
    """Comparing method"""
    def two_tailed_t_test(original, selection):
        n_d = len(selection)
        n_c = len(original)
        n = min(n_d, n_c)
        t, p = ttest_rel(original[:n], selection[:n])
        if np.isnan(t):
            t, p = 0, 1
        return {"t-stats":t, "p-value":p}

    def one_tailed_t_test(original, selection, direction):
        two_tail = two_tailed_t_test(original, selection)
        t, p_two = two_tail['t-stats'], two_tail['p-value']
        if direction == 'positive':
            if t > 0 :
                p = p_two * 0.5
            else:
                p = 1 - p_two * 0.5
        else:
            if t < 0:
                p = p_two * 0.5
            else:
                p = 1 - p_two * 0.5
        return {"t-stats":t, "p-value":p}

    result = {}
    result['two_tail'] = two_tailed_t_test(original, selection)
    result['one_tail_pos'] = one_tailed_t_test(original, selection, 'positive')
    result['one_tail_neg'] = one_tailed_t_test(original, selection, 'negative')
    return result


def evaluate_score(original, selection):
    alpha =  0.05
    results = t_test(original, selection)
    difference = 'insignificant'

    if results['two_tail']['p-value'] < alpha:
        if results['one_tail_neg']['p-value'] < alpha:
            difference = 'positive'
        if results['one_tail_pos']['p-value'] < alpha:
            difference = 'negative'

    return difference