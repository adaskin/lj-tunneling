# data/studies/dt_scan/ -- the snapshot spacing at the production window

`N{3,4}_k12_int_c{1.460,2.260}_T200_dt{0.5,1,2,4}.npz`: the correlated start of N = 3 and N = 4 at k = 12, the same state and window (T = 200), snapshots every 0.5, 1, 2 or 4 time units,
every bipartition measured.  Protocol `custom` except dt = 1 (= the production file).  Written by `run_sampling_study()`, read by `run_protocol_tables()` (`derived/protocol_tables.npz`, "spacing").
Result: spacing 1.0 is within 0.02 bit of the finest spacing for the one-particle entropy (N = 3: 1.382 / 1.367 / 1.367 / 1.367) and the central bond at N = 4 is 2.3346, 2.3346, 2.3346, 2.3265.
