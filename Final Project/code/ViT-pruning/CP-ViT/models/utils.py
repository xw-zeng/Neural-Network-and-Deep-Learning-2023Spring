pruned_percent_num = 8 # range(0, 12)
mlp_percent_num = 0

def compute_cascade(ratio):
    global pruned_percent_num
    if pruned_percent_num > 8:
        return ratio - (pruned_percent_num - 2) * 0.03
    elif (pruned_percent_num > 2) & (pruned_percent_num < 9):
        return ratio - (pruned_percent_num - 2) * 0.04
    else:
        return ratio


def compute_mlp_ratio(ratio):
    global mlp_percent_num
    mlp_percent_num += 1
    if mlp_percent_num == 12:
        mlp_percent_num = 0
        return ratio
    else:
        return ratio + mlp_percent_num * 5e-3
