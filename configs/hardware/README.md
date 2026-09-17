# Hardware configs

Ship only the template. Local edit stays untracked:

```bash
cp so101_dual_manifest.example.json so101_dual_manifest.json
ls /dev/serial/by-id/   # fill follower/leader ports + camera indices
```

Park pose for online go-home: record on **your** robot with `rlt-so101-dual-record-pose` and pass `--go-home-positions` (do not commit machine-specific ticks).

Calibrate / teleop with upstream LeRobot (`lerobot-calibrate`, `lerobot-teleoperate`) using the same ports/cameras as the manifest. Shape contract: `src/rlt_so101_dual/core/shape_contract.py`.
