# Hard-encoded MuJoCo demonstrations

`generate_mujoco_insertion_demos.py` creates scripted, successful insertion
episodes using the known MuJoCo fixture pose. Each episode starts from a
deterministic random offset around the standard post-grasp pose, aligns the peg,
approaches the socket, and inserts it. Failed attempts are discarded.

Run from the repository root:

```bash
python hardEncodedScripts/generate_mujoco_insertion_demos.py --episodes 100 --output-root outputs/mujoco/hardEncodedDemosXYZGenerated
```

The generated LeRobot dataset is written to `outputs/mujoco/hardEncodedDemosXYZGenerated/insert/`.
The faster state-only projection of the original demonstrations is stored at
`outputs/mujoco/hardEncodedDemosXYZ/insert/`.
`trajectory_manifest.json` records every random start, planner parameter,
trajectory length, and final insertion metric. Re-running the command resumes an
incomplete dataset instead of overwriting it.

Train from this dataset with:

```bash
share-learner --env.type=mujoco_ur5e_insertion --env.episode_steps=300 --env.state_only_policy=true --dataset.repo_id=local/mujoco-hard-encoded-insertion-xyz --dataset.root=outputs/mujoco/hardEncodedDemosXYZ --policy.type=sac_dagger_bc --policy.device=cuda --policy.storage_device=cpu --policy.online_steps=5000 --policy.offline_buffer_capacity=10000 --policy.bc_lr=0.0003 --policy.bc_loss_type=mse --output_dir=outputs/mujoco/hardEncodedRunXYZ --job_name=mujoco-hard-encoded-insertion-xyz --batch_size=128 --num_workers=0 --save_freq=1000 --log_freq=500 --wandb.enable=false
```

Evaluate the final checkpoint from unseen randomized starts:

```bash
python hardEncodedScripts/evaluate_mujoco_insertion_policy.py --episodes 20 --seed 20260825
```

## Force-guided search demonstrations

`generate_mujoco_force_search_demos.py` creates contact-rich demonstrations
inspired by ConnTact's insertion search. The fixture estimate is deliberately
offset. Each episode approaches until wrist-force contact, retracts 3 mm, moves
through discrete points on an expanding spiral, probes again, and finally uses
a bounded admittance-style insertion command. This discrete probe variant avoids
dragging the friction-grasped square peg continuously across the fixture.
The spiral direction is fixed counter-clockwise so a deterministic feed-forward
policy is not trained on contradictory clockwise/counter-clockwise labels.
Candidates that remain physically unsuccessful after all retries are skipped;
the manifest candidate index prevents them from blocking dataset resume.
Signed wrist moments bias each next spiral target by 10%. The estimator removes
the transverse-force moment caused by the known sensor-to-tip axial lever before
solving for the lateral contact point. Moment direction therefore guides the
search without replacing the complete spiral fallback.

```bash
MUJOCO_GL=egl python hardEncodedScripts/generate_mujoco_force_search_demos.py --episodes=100 --seed=20260825 --output-root=outputs/mujoco/hardEncodedMomentGuidedVisual64Demos --max-attempts=3 --episode-steps=1300
```

The two `64x64` camera streams, robot state, six-axis wrist wrench, XYZ velocity
action, reward, and done flag use the same LeRobot schema as the standard
insertion pipeline. `force_search_manifest.json` records stage counts, contact
peaks, search offset, recovery count, and final insertion metrics. Re-running
the command resumes an incomplete dataset.

## Online oracle intervention

`auto_control.py` provides intermittent exact-geometry keyboard interventions to an
already running MuJoCo Actor. See [AUTO_CONTROL.md](AUTO_CONTROL.md) for its launch
command, parameters, and the two intentionally narrow core hooks.
