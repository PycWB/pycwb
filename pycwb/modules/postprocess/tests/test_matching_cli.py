"""Regression coverage for matching cli."""
import argparse
import pyarrow as pa
import pyarrow.parquet as pq


def test_matching_preserves_large_strings_and_cli_ranking(tmp_path):
    from pycwb.cli import match_simulations as cli
    triggers=pa.table({'id':pa.array(['a','b'], type=pa.large_string()),
        'trial_idx':[0,0], 'gps_time':[100.,100.], 'rho':[30.,10.], 'rho_alt':[5.,20.]})
    sims=pa.table({'sim_idx':[0,1], 'trial_idx':[0,1], 'gps_time':[100.,200.],
        'real_start':[99.,199.], 'real_end':[101.,201.],
        'name':pa.array(['SG235Q9','missed'], type=pa.large_string())})
    pq.write_table(triggers,tmp_path/'triggers.parquet');pq.write_table(sims,tmp_path/'sims.parquet')
    parser=argparse.ArgumentParser();cli.init_parser(parser)
    args=parser.parse_args([str(tmp_path/'triggers.parquet'),str(tmp_path/'sims.parquet'),
        '--how','outer','--ranking-par','rho_alt','-o',str(tmp_path/'matched.parquet')])
    cli.command(args)
    rows=pq.read_table(tmp_path/'matched.parquet').to_pandas()
    assert rows.loc[rows.sim_sim_idx==0,'id'].tolist()==['b']
    assert rows.loc[rows.sim_sim_idx==1,'id'].isna().all()
    assert len(rows)==3  # selected, missed injection, and unmatched trigger
