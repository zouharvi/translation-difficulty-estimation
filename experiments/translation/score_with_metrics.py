import json
import logging
import pickle
import subprocess
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict, Tuple, Union, Literal

logging.basicConfig(level=logging.INFO)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to score translations with a QE MT metric specified in input."
    )

    parser.add_argument(
        "--translations-path",
        type=Path,
        help="Path to the directory containing the translations to be scored. The directory must contain a folder for "
        "each MT system and, in it, a jsonl file for each target language, together with a sources.jsonl file "
        "containing the source texts.",
    )

    parser.add_argument(
        "--sys2translations-path",
        type=Path,
        help="Path to a pickle file containing a dictionary of MT system translations along with any precomputed scores"
        " from QE MT metrics. If this argument is provided, the --translations-path argument will be ignored.",
    )

    parser.add_argument(
        "--source-texts-path",
        type=Path,
        help="Path to a jsonl file containing the source texts translated by the various MT systems. This argument is "
        "ignored if --translations-path is passed.",
    )

    parser.add_argument(
        "--score-distinct-protocols-and-sources",
        action="store_true",
        help="Whether the structure of the input --translations-path directory is based on 'MT_sys/protocol/lp/'. If "
        "passed, not all language pairs (and protocols) share the same source texts, and --source-texts-path points to "
        "a pickle file with distinct source texts per protocol and language pair. ",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["metricx24-hybrid-qe", "xcomet-qe", "cometkiwi-xxl"],
        default="metricx24-hybrid-qe",
        help="Which QE MT metric to use for scoring. Allowed values: 'metricx24-hybrid-qe', 'xcomet-qe', and "
        "'cometkiwi-xxl'. Default: 'metricx24-hybrid-qe'.",
    )

    parser.add_argument(
        "--metricx24-predict-script-path",
        type=Path,
        help="Path to the metricx24 predict.py script to use for running inference with metricx. Required if the metric"
        " specified is `metricx24-hybrid-qe`.",
    )

    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size to use when running inference with the input metric. Default: 32.",
    )

    parser.add_argument(
        "--scored-translations-path",
        type=Path,
        required=True,
        help="[REQUIRED] Path to the output pickle file where the scored translations will be saved.",
    )

    return parser


def get_translations(
    translations_path: Path,
    score_distinct_protocols_and_sources: bool,
) -> Tuple[
    Union[List[str], Dict[str, Dict[str, List[str]]]],
    Dict[str, Dict[str, Union[List[str], Dict[str, List[str]]]]],
]:
    """
    Load translations from files in the specified directory.

    Args:
        translations_path: Path to the directory containing the translations.
        score_distinct_protocols_and_sources: Whether there are distinct source texts for each protocol and lang pair.

    Returns:
        scored_data: Sources and translations dict.
    """
    sys2translations, source_texts = dict(), None

    # Use one MT system to read the `sources.jsonl` file (if they are all the same).
    for sys_dir in translations_path.iterdir():
        if sys_dir.is_dir():
            if score_distinct_protocols_and_sources:
                # protocol → language pair → [source_text_0, source_text_1, ...]
                source_texts = dict()
                for protocol in ["esa", "mqm"]:
                    source_texts[protocol] = dict()
                    protocol_path = sys_dir / protocol
                    for lp in protocol_path.iterdir():
                        if lp.is_dir():
                            for f in lp.iterdir():
                                if f.is_file() and "sources" in f.name:
                                    with f.open(
                                        "r", encoding="utf-8"
                                    ) as source_texts_file:
                                        source_texts[protocol][lp.name] = [
                                            json.loads(line)
                                            for line in source_texts_file
                                        ]

                all_source_texts_path = sys_dir / "sources.pickle"
                with all_source_texts_path.open("wb") as f:
                    pickle.dump(source_texts, f, protocol=pickle.HIGHEST_PROTOCOL)
                break  # Only need to read source texts once.
            else:
                sources_path = sys_dir / "sources.jsonl"
                if sources_path.exists():
                    with sources_path.open("r", encoding="utf-8") as f:
                        source_texts = [json.loads(line) for line in f]
                    break  # Only need to read source texts once.

    # Load all translations
    for sys_dir in translations_path.iterdir():
        if not sys_dir.is_dir():
            continue

        sys_name = sys_dir.name

        if score_distinct_protocols_and_sources:
            # protocol → language pair → [translation_0, translation_1, ...]
            sys2translations[sys_name] = dict()
            for protocol in ["esa", "mqm"]:
                sys2translations[sys_name][protocol] = dict()
                protocol_path = sys_dir / protocol
                for lp in protocol_path.iterdir():
                    if lp.is_dir():
                        for f in lp.iterdir():
                            if f.is_file() and "translation" in f.name:
                                with f.open("r", encoding="utf-8") as translations_file:
                                    sys2translations[sys_name][protocol][lp.name] = [
                                        json.loads(line) for line in translations_file
                                    ]
                                    if len(
                                        sys2translations[sys_name][protocol][lp.name]
                                    ) != len(source_texts[protocol][lp.name]):
                                        raise ValueError(
                                            f"Number of translations for {sys_name} in {lp.name} for {protocol} does "
                                            "not match the number of source texts! Number of translations: "
                                            f"{len(sys2translations[sys_name][protocol][lp.name])}, number of source "
                                            f"texts: {len(source_texts[protocol][lp.name])}."
                                        )
        else:
            sys2translations[sys_name] = dict()

            for file_path in sys_dir.glob("*.jsonl"):
                if file_path.name == "sources.jsonl":
                    continue

                tgt_lang = file_path.stem  # e.g., "japanese" from "japanese.jsonl".
                with file_path.open("r", encoding="utf-8") as f:
                    sys2translations[sys_name][tgt_lang] = [
                        json.loads(line) for line in f
                    ]
                    if len(sys2translations[sys_name][tgt_lang]) != len(source_texts):
                        raise ValueError(
                            f"Number of translations for {sys_name} in {tgt_lang} does not match the number "
                            f"of source texts! Number of translations: {len(sys2translations[sys_name][tgt_lang])}, "
                            f"number of source texts: {len(source_texts)}."
                        )

    return source_texts, sys2translations


def score_with_metricx(
    metricx24_predict_script_path: Path,
    source_texts: Union[List[str], Dict[str, Dict[str, List[str]]]],
    sys2translations: Dict[
        str,
        Dict[
            str,
            Union[
                List[Union[str, Dict[str, Union[str, float]]]],
                Dict[str, List[Union[str, Dict[str, Union[str, float]]]]],
            ],
        ],
    ],
    batch_size: int,
) -> None:
    """
    Score translations with MetricX-24-Hybrid-QE and update the input `sys2translations` dictionary in place.

    Args:
        metricx24_predict_script_path: Path to the MetricX-24-Hybrid-QE predict.py script.
        source_texts: Source texts (either a list or a dict)
        sys2translations: Nested dictionary [sys][tgt_lang] (or [sys][prot][lp]) -> list of translations.
        batch_size: Inference batch size.
    """
    logging.info("Scoring translations with MetricX-24-Hybrid-QE...")

    # MetricX-24-Hybrid predict.py script input arguments
    tokenizer = "google/mt5-xxl"
    model = "google/metricx-24-hybrid-xxl-v2p6"
    max_input_length = 1536

    # Temporary file used as input/output (overwritten each time)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp_path = Path(tmp.name)

    def launch_script(
        _source_texts: List[str], _translations: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Launch the metricx24 scoring script.

        Args:
            _source_texts: Source texts list.
            _translations: Candidate translations list.

        Returns:
            updated_translations: Input translations list of dicts updated with metricx24 scores.
        """
        # Compose input jsonl content
        input_lines = []
        for src, hyp in zip(_source_texts, _translations):
            if isinstance(hyp, str):
                hyp_text = hyp
            elif isinstance(hyp, dict):
                hyp_text = hyp["hypothesis"]
            else:
                raise ValueError(f"Unexpected translation format: {hyp}.")
            input_lines.append(
                json.dumps(
                    {
                        "source": src,
                        "hypothesis": hyp_text,
                        "reference": "",  # Always empty, since we are using a QE metric
                    },
                    ensure_ascii=False,
                )
            )

        # Write to temporary input/output file
        with tmp_path.open("w", encoding="utf-8") as f:
            for line in input_lines:
                f.write(line + "\n")

        # Call metricx24 script
        command = [
            "python",
            str(metricx24_predict_script_path),
            "--tokenizer",
            tokenizer,
            "--model_name_or_path",
            model,
            "--max_input_length",
            str(max_input_length),
            "--batch_size",
            str(batch_size),
            "--input_file",
            str(tmp_path),
            "--output_file",
            str(tmp_path),
            "--qe",
        ]

        subprocess.run(command, check=True)

        # Read output and update translations
        with tmp_path.open("r", encoding="utf-8") as f:
            predictions = [
                -json.loads(line)["prediction"] for line in f
            ]  # MetricX returns error scores

        updated_translations = []
        assert len(_translations) == len(predictions)
        for original, score in zip(_translations, predictions):
            if isinstance(original, str):
                updated = {
                    "hypothesis": original,
                    "metricx24-hybrid-qe_score": score,
                }
            elif isinstance(original, dict):
                updated = dict(original)
                updated["metricx24-hybrid-qe_score"] = score
            else:
                raise ValueError(f"Unexpected translation format: {original}.")

            updated_translations.append(updated)

        return updated_translations

    try:
        for sys, translations_dict in sys2translations.items():
            for key, value in translations_dict.items():
                if isinstance(value, list):
                    if not isinstance(source_texts, list):
                        raise ValueError(
                            "If the sys2translations dict has a [sys][tgt_lang] structure, source_texts must be a "
                            f"list! Found {type(source_texts)}."
                        )

                    tgt_lang, translations = key, value
                    logging.info(f"Scoring {sys} - {tgt_lang}...")

                    if len(source_texts) != len(translations):
                        raise ValueError(
                            f"Number of source texts ({len(source_texts)}) does not match the number of translations "
                            f"({len(translations)}) for {sys} - {tgt_lang}!"
                        )

                    # In-place update
                    sys2translations[sys][tgt_lang] = launch_script(
                        source_texts, translations
                    )

                else:
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"Each value in sys2translations must be either a list or a dict! Found {type(value)}."
                        )
                    if not isinstance(source_texts, dict):
                        raise ValueError(
                            "If the sys2translations dict has a [sys][prot][lp] structure, source_texts must be a "
                            f"dict! Found {type(source_texts)}."
                        )

                    protocol, lp2translations = key, value
                    for lp, translations in lp2translations.items():
                        logging.info(f"Scoring {sys} - {protocol} - {lp}...")

                        if len(source_texts[protocol][lp]) != len(translations):
                            raise ValueError(
                                f"Number of source texts ({len(source_texts[protocol][lp])}) does not match the number "
                                f"of translations ({len(translations)}) for {sys} - {protocol} - {lp}!"
                            )

                        # In-place update
                        sys2translations[sys][protocol][lp] = launch_script(
                            source_texts[protocol][lp], translations
                        )

    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()

    logging.info("Finished scoring with MetricX-24-Hybrid-QE.")


def score_with_comet(
    metric: Literal["xcomet-qe", "cometkiwi-xxl"],
    source_texts: Union[List[str], Dict[str, Dict[str, List[str]]]],
    sys2translations: Dict[
        str,
        Dict[
            str,
            Union[
                List[Union[str, Dict[str, Union[str, float]]]],
                Dict[str, List[Union[str, Dict[str, Union[str, float]]]]],
            ],
        ],
    ],
    batch_size: int,
) -> None:
    """
    Score translations with a metric from the COMET family and update the input `sys2translations` dictionary in place.

    Args:
        metric: Name of the COMET metric to use for scoring. Allowed values: `"xcomet-qe"`, `"cometkiwi-xxl"`.
        source_texts: Source texts (either a list or a dict).
        sys2translations: Nested dictionary [sys][tgt_lang] (or [sys][prot][lp]) -> list of translations.
        batch_size: Inference batch size.
    """
    if metric != "xcomet-qe" and metric != "cometkiwi-xxl":
        raise ValueError(
            f"Invalid COMET QE metric specified: {metric}. Allowed values: 'xcomet-qe', 'cometkiwi-xxl'."
        )

    import comet

    logging.info(f"Scoring translations with {metric}...")

    def create_input_data_for_comet_qe_metric_model(
        src: List[str], cand: List[str]
    ) -> List[Dict[str, str]]:
        """Create the input data for the COMET QE metric model.

        Args:
            src (List[str]): Source texts.
            cand (List[str]): Candidate translations.

        Returns:
            List[Dict[str, str]]: Input data for the COMET QE metric model.
        """
        if len(src) != len(cand):
            raise ValueError(
                f"The number of source texts ({len(src)}) and candidate translations ({len(cand)}) must be the same!"
            )

        return [{"src": s, "mt": c} for s, c in zip(src, cand)]

    comet_qe_metric_model_path = comet.download_model(
        "Unbabel/XCOMET-XXL"
        if metric == "xcomet-qe"
        else "Unbabel/wmt23-cometkiwi-da-xxl"
    )
    comet_qe_metric_model = comet.load_from_checkpoint(comet_qe_metric_model_path)

    def score(
        _source_texts: List[str], _translations: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Score with a COMET metric.

        Args:
            _source_texts: Source texts list.
            _translations: Candidate translations list.

        Returns:
            updated_translations: Input translations list of dicts updated with a COMET metric scores.
        """
        metric_model_output = comet_qe_metric_model.predict(
            create_input_data_for_comet_qe_metric_model(
                _source_texts,
                [t["hypothesis"] for t in _translations]
                if isinstance(_translations[0], dict)
                else _translations,
            ),
            batch_size=batch_size,
            gpus=1,
        )

        updated_translations = []
        assert len(_translations) == len(metric_model_output.scores)
        for original, metric_score in zip(_translations, metric_model_output.scores):
            if isinstance(original, str):
                updated = {
                    "hypothesis": original,
                    f"{metric}_score": metric_score,
                }
            elif isinstance(original, dict):
                updated = dict(original)
                updated[f"{metric}_score"] = metric_score
            else:
                raise ValueError(f"Unexpected translation format: {original}.")

            updated_translations.append(updated)

        return updated_translations

    for sys, translations_dict in sys2translations.items():
        for key, value in translations_dict.items():
            if isinstance(value, list):
                if not isinstance(source_texts, list):
                    raise ValueError(
                        "If the sys2translations dict has a [sys][tgt_lang] structure, source_texts must be a "
                        f"list! Found {type(source_texts)}."
                    )

                tgt_lang, translations = key, value
                logging.info(f"Scoring {sys} - {tgt_lang}...")

                if len(source_texts) != len(translations):
                    raise ValueError(
                        f"Number of source texts ({len(source_texts)}) does not match the number of translations "
                        f"({len(translations)}) for {sys} - {tgt_lang}!"
                    )

                sys2translations[sys][tgt_lang] = score(source_texts, translations)

            else:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Each value in sys2translations must be either a list or a dict! Found {type(value)}."
                    )
                if not isinstance(source_texts, dict):
                    raise ValueError(
                        "If the sys2translations dict has a [sys][prot][lp] structure, source_texts must be a "
                        f"dict! Found {type(source_texts)}."
                    )

                protocol, lp2translations = key, value
                for lp, translations in lp2translations.items():
                    logging.info(f"Scoring {sys} - {protocol} - {lp}...")

                    if len(source_texts[protocol][lp]) != len(translations):
                        raise ValueError(
                            f"Number of source texts ({len(source_texts[protocol][lp])}) does not match the number "
                            f"of translations ({len(translations)}) for {sys} - {protocol} - {lp}!"
                        )

                    # In-place update
                    sys2translations[sys][protocol][lp] = score(
                        source_texts[protocol][lp], translations
                    )

    logging.info(f"Finished scoring with {metric}.")


def score_with_metrics_command() -> None:
    """
    Command to score translations with a QE MT metric specified in input.
    """
    args: Namespace = read_arguments().parse_args()

    if args.sys2translations_path is not None:
        if args.source_texts_path is None:
            raise ValueError(
                "If --sys2translations-path is specified, --source-texts-path must also be provided!"
            )

        if args.score_distinct_protocols_and_sources:
            with args.source_texts_path.open("rb") as handle:
                source_texts = pickle.load(handle)
        else:
            with args.source_texts_path.open("r", encoding="utf-8") as f:
                source_texts = [json.loads(line) for line in f]

        with args.sys2translations_path.open("rb") as handle:
            sys2translations = pickle.load(handle)

    else:
        source_texts, sys2translations = get_translations(
            args.translations_path, args.score_distinct_protocols_and_sources
        )

    if args.metric == "metricx24-hybrid-qe":
        if args.metricx24_predict_script_path is None:
            raise ValueError(
                "If `metricx24-hybrid-qe` is specified as a metric to use, the `--metricx24-predict-script-path` "
                "argument must be provided!"
            )

        score_with_metricx(
            args.metricx24_predict_script_path,
            source_texts,
            sys2translations,
            args.batch_size,
        )
    else:
        score_with_comet(args.metric, source_texts, sys2translations, args.batch_size)

    with args.scored_translations_path.open("wb") as handle:
        pickle.dump(sys2translations, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    score_with_metrics_command()
