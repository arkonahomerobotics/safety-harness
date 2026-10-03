#!/bin/bash
# Port smoke suite (secondary scripts). Run from /workspace/isaaclab. Writes /tmp/frp_suite_*.log
cd /workspace/isaaclab
D=/workspace/isaaclab/snapshots
L=logs/rsl_rl/franka_stack_rl_expert
C=$PWD/$(ls -d $L/*frp_smoke_snapsparse_256 | tail -1)/model_19.pt
echo "ckpt $C"
# 1. checkpoint tool: gripper std 0.3 + fresh optimizer, then resume 2 iterations from it via the unified train.py
mkdir -p $L/frp_seed_gripstd03
./isaaclab.sh -p franka_rl/reset_gripper_std.py $C $PWD/$L/frp_seed_gripstd03/model_19.pt 0.3 --reset_optimizer --lr 5e-5 > /tmp/frp_suite_ckpt.log 2>&1; echo exit=$? >> /tmp/frp_suite_ckpt.log
timeout 1200 ./isaaclab.sh -p scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Stack-Cube-Franka-IK-Rel-RL-Robosuite-Snap-Sparse-v0 --num_envs 256 --max_iterations 2 --seed 43 --run_name frp_smoke_resume --checkpoint $PWD/$L/frp_seed_gripstd03/model_19.pt > /tmp/frp_suite_resume.log 2>&1; echo exit=$? >> /tmp/frp_suite_resume.log
# 2. Stage2Skill training, 2 iterations, snapshot path overridden through Hydra (the Brev K6/K7 syntax)
timeout 1200 ./isaaclab.sh -p scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Isaac-Stack-Cube-Franka-IK-Rel-RL-Stage2Skill-Contact-v0 --num_envs 256 --max_iterations 2 --seed 44 --run_name frp_smoke_stage2skill_contact env.events.reset_from_snapshot.params.snapshot_path=$D/smoke_stage2_allphases_neargoal.pt > /tmp/frp_suite_s2.log 2>&1; echo exit=$? >> /tmp/frp_suite_s2.log
# 3. stage-2 snapshot reset check
timeout 900 ./isaaclab.sh -p franka_rl/test_snapshot_reset.py --stage2 --task Isaac-Stack-Cube-Franka-IK-Rel-RL-Stage2Skill-v0 --snapshot_path $D/smoke_green_onto_red_256.pt --num_envs 64 > /tmp/frp_suite_snap2.log 2>&1; echo exit=$? >> /tmp/frp_suite_snap2.log
# 4. chained evaluator (smoke ckpt as both stages)
timeout 1200 ./isaaclab.sh -p franka_rl/eval_chain.py --stage1 $C --stage2 $C --num_envs 256 > /tmp/frp_suite_chain.log 2>&1; echo exit=$? >> /tmp/frp_suite_chain.log
# 5. handoff capture (1 round)
timeout 900 ./isaaclab.sh -p franka_rl/capture_handoff.py --stage1 $C --out $D/smoke_handoff.pt --num_envs 256 --rounds 1 > /tmp/frp_suite_handoff.log 2>&1; echo exit=$? >> /tmp/frp_suite_handoff.log
# 6. chained video (1 env, 1 short episode)
timeout 1200 ./isaaclab.sh -p franka_rl/record_chain.py --stage1 $C --stage2 $C --out $PWD/videos/frp_smoke_chain --episodes 1 --episode_s 4 > /tmp/frp_suite_video.log 2>&1; echo exit=$? >> /tmp/frp_suite_video.log
echo SUITE_DONE > /tmp/frp_suite_done
