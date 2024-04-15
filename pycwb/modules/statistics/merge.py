from functools import reduce

import pandas as pd


def read_data_file(file_path: str, i: int) -> pd.DataFrame:
    """
    Read the data from cwb eff_*.txt files and return a DataFrame

    Parameters
    ----------
    file_path : str
        Path to the file to read
    i : int
        Index of the chunk

    Returns
    -------
    pd.DataFrame
        DataFrame with the data read from the file
    """
    data = pd.read_csv(file_path, sep='\s+', header=None, names=['hrss', f'evt_{i}', f'inj_{i}', f'ratio_{i}'])

    return data


def get_evt_vs_inj(chunks: list[str], wf_selections: list[str]) -> dict[str, pd.DataFrame]:
    """
    Merge the data from the chunks into a single DataFrame for each waveform selected

    Parameters
    ----------
    chunks : list[str]
        List of paths to the chunks to merge
    wf_selections : list[str]
        List of waveform selections to merge

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary with the waveform name as the key and the merged DataFrame as the value

    Examples
    --------
    >>> chunks = ['chunk1', 'chunk2']
    >>> wf_selections = ['waveform1', 'waveform2']
    >>> get_evt_vs_inj(chunks, wf_selections)
    {'waveform1': pd.DataFrame, 'waveform2': pd.DataFrame}
    """
    wf_data = {}
    for wf in wf_selections:
        data = []
        for i, chunk in enumerate(chunks):
            data.append(read_data_file(f"{chunk}/eff_{wf}.txt", i + 1))

        merged_df = reduce(lambda left, right: pd.merge(left, right, on='hrss'), data)

        merged_df['evt_total'] = 0
        merged_df['inj_total'] = 0
        for i in range(len(chunks)):
            merged_df['evt_total'] += merged_df[f'evt_{i + 1}']
            merged_df['inj_total'] += merged_df[f'inj_{i + 1}']
        wf_data[wf] = merged_df

    return wf_data
