# OpenHAB multi-rule migration

This directory is the EviRepair migration used on OpenHAB multi-rule scenarios. It is separate from the Home Assistant package in `src/smarthome_mdf`.

`run_multirule.py` loads `data/openhab/benchmarks`, keeps scenarios with more than one enabled rule, and skips single-rule scenarios. Repair uses the `ss_trhr_v1` operators, then Strict-v2. The parent step uses `set_remove_v1` only.

The recorded result is `results/openhab_multirule_evirepair.json`. On the 371 multi-rule scenarios in `live_split`, semantic success after repair is 0.8814 and the complete repair rate is 0.8112.

From the repository root:

```bash
python -u openhab_multirule/run_multirule.py
```

The script adds the repository root, `src`, and `scripts` to `sys.path`. It rewrites `results/openhab_multirule_evirepair.json` and does not write a single-rule result.
