# previous-data/derived/ -- the tables of the earlier setting

`ledger.npz`, `graph.npz`, `protocol_tables.npz`, `larger_N.npz`, `quantum.npz`, `cosine_check.npz`, `parameter_choice.npz`, `parameter_scan_N{3,4}.npz`, `fullgrid_N2.npz`
and `tn_tables.npz` + `tn_tables/*.csv`, as they were when `theta_barrier = 0` (what each file is: `data/derived/README.md`).  Two things to know: `larger_N.npz` and
`ledger.npz` were rebuilt after the N = 5 runs landed, `tn_tables.npz` was not (its `wall_sensitivity` rows come from an earlier `larger_N.npz` that still held the legacy N = 5 point); and `parameter_choice.npz`
is parsed from `../legacy/tier1/tier1_parameter_map.txt`, the Appendix-A ledger of the earlier convention (the new one is `data/studies/parameter_map/`).
