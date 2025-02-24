from difficulty_sampling.data import Data


def word_frequency_score(
    data: Data,
    scorer_name: str,
) -> Data:
    """
    Assign to each source a score that depends on the word frequency of its words

    Args:
        data (Data): Data to score.
        scorer_name (str): Name to use to identify the scorer used in {'word_frequency', 'word_zipf_frequency'}.

    Returns:
        scored_data (Data): Input data with "scorer_name" as additional available score for each MT system.
        Higher scores indicate higher frequency, therefore lower rarity.
    """
    from wordfreq import word_frequency, zipf_frequency, tokenize

    print("Counting frequencies for source language: ", data.lp.split("-")[0])
    sources = [{"src": sample["src"]} for sample in data.src_data_list]

    scores = []
    for source_elem in sources:
        src = source_elem["src"]
        tokens = tokenize(src, data.lp.split("-")[0])
        if scorer_name == "word_frequency":
            avg_freq = sum(
                word_frequency(token, data.lp.split("-")[0]) for token in tokens
            ) / len(tokens)
        elif scorer_name == "word_zipf_frequency":
            avg_freq = sum(
                zipf_frequency(token, data.lp.split("-")[0]) for token in tokens
            ) / len(tokens)
        else:
            raise ValueError(f"Scorer name '{scorer_name}' not recognized.")
        scores.append(avg_freq)

    assert len(scores) == len(data.src_data_list)

    for idx, sample in enumerate(data.src_data_list):
        for system in sample["scores"]:
            sample["scores"][system][scorer_name] = scores[idx]

    return data


def subsample_with_word_frequency(args) -> Data:
    """
    Command to subsample WMT data using the frequency of words in the source sentences
    """

    scored_data = word_frequency_score(
        Data.load(
            dataset_name=args.dataset_name,
            lp=args.lp,
            protocol=args.protocol,
            domains=args.domains,
        ),
        scorer_name=args.scorer_name,
    )

    return scored_data
