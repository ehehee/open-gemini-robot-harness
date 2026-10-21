# Extrinsics

The solved geometry of the rig the case study ran on. Every number Gemini sees
or sends is in the **right arm's base frame**; these files place everything
else in it.

| file | transform | solved from |
|---|---|---|
| `T_rightbase_overhead.yaml` | overhead ZED-M → right arm base | eye-to-hand, 30 captures |
| `T_leftbase_overhead.yaml` | overhead ZED-M → left arm base | eye-to-hand, 29 captures |
| `T_leftbase_rightbase.yaml` | left arm base → right arm base | co-observation chain |
| `T_right_wrist_gripper_cam.yaml` | right wrist D405 → right gripper body | eye-in-hand, 2026-10-14 |

The first three were solved with `DANIILIDIS`, board held by the gripper,
`reproj_max_px: 2.0`. Each file carries its solve's metadata; the overhead
files also carry the overhead camera's intrinsics, which `load_overhead_intrinsics`
reads, so keep the `metadata:` block.

These describe one physical setup. Replace them with your own after anything
that moves a camera or an arm base. Re-zeroing a motor does not invalidate
them (that is `fk.joint_offsets` in `../rig.yaml`).
