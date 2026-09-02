#!/nas/hhan/anaconda3/envs/SPMM_env/bin/python
"""Wrapper so the analysis can be launched from the repo root as

    ./run_bivstri.py --num_samples 100

Equivalent to `python -m repr_analysis.run_bi_vs_tri`. The shebang pins the
SPMM_env interpreter, because the login shell's `python` is admet_env.
"""
from repr_analysis.run_bi_vs_tri import main

if __name__ == '__main__':
    main()
