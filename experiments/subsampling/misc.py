import subset2evaluate

def apply_src_len(data):
    for data in data.lp2src_data_list.values():
        for line in data:
            for sys in line["scores"].keys():
                line["scores"][sys]["src_len"] = -len(line["src"])


def apply_subset2evaluate_cache(data, method):
    load_model = None
    for data in data.lp2src_data_list.values():
        scores, load_model = subset2evaluate.methods.METHODS[method](data, return_model=True, load_model=load_model)
        for line in data:
            score = scores.pop(0)
            for sys in line["scores"].keys():
                line["scores"][sys][method] = -score


def apply_subset2evaluate(data, method):
    for data in data.lp2src_data_list.values():
        scores = subset2evaluate.methods.METHODS[method](data)
        for line in data:
            score = scores.pop(0)
            for sys in line["scores"].keys():
                line["scores"][sys][method] = -score


def apply_artificial_crowd_metrics(data, model, metric):
    for data in data.lp2src_data_list.values():
        for line in data:
            score = line["scores"][model][metric]
            for sys in line["scores"].keys():
                line["scores"][sys]["artcrowd|" + model + "|" + metric] = score