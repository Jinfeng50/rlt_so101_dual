# Configs (SO101 dual)

1. Copy the hardware template and fill **your** serial `by-id` paths + camera indices:

```bash
cp configs/hardware/so101_dual_manifest.example.json \
   configs/hardware/so101_dual_manifest.json
ls /dev/serial/by-id/
```

| File | Role |
| --- | --- |
| `hardware/so101_dual_manifest.example.json` | Public template (committed) |
| `hardware/so101_dual_manifest.json` | Local edit (gitignored) |
| `rename_maps/so101_dual.json` | Dataset image keys → π0.5 slots |
| `rlt/so101_dual_rlt.yaml` | RLT shapes / Stage-B defaults |

Camera naming: see `src/rlt_so101_dual/core/shape_contract.py`.
