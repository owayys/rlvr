"""Minimal GRPO entrypoint for countdown task with verl 0.7."""

import argparse
import os
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf

from rlvr.rewards.countdown import compute_score


class RewardManager:
    """Simple reward manager that dispatches to countdown scoring function."""

    def __init__(self, tokenizer, num_examine: int = 0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self._examined_count = 0

    def __call__(self, data):
        """Compute rewards for a batch of data."""
        # return rm_scores if already provided
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids[:0]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length] if valid_response_length > 0 else response_ids[:0]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            score = compute_score(solution_str=sequences_str, ground_truth=ground_truth)

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = score

            # debug output
            if self._examined_count < self.num_examine:
                self._examined_count += 1
                print(f"[Reward Debug {i}] Score: {score}")
                print(sequences_str)
                print("---")

        return reward_tensor


def _import_verl_symbols():
    """Import required verl symbols."""
    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        return RayPPOTrainer, ResourcePoolManager, Role
    except ImportError as e:
        raise RuntimeError(f"Failed to import verl symbols: {e}")


def main():
    parser = argparse.ArgumentParser(description="RLVR GRPO training for countdown task")
    parser.add_argument("--config", required=True, help="Path to OmegaConf YAML config")
    args = parser.parse_args()

    # config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)

    print("[Config]")
    print(OmegaConf.to_yaml(config))

    # ray init
    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
        )

    # tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer

    local_model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_model_path)

    # verl classes
    RayPPOTrainer, ResourcePoolManager, Role = _import_verl_symbols()

    # worker config
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
        from verl.single_controller.ray import RayWorkerGroup

        actor_rollout_cls = ActorRolloutRefWorker
        actor_role = Role.ActorRolloutRef
        critic_worker_cls = TrainingWorker
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")

    role_worker_mapping = {
        actor_role: ray.remote(actor_rollout_cls),
    }

    # add critic if enabled
    if config.critic.get("enable", False):
        role_worker_mapping[Role.Critic] = ray.remote(critic_worker_cls)

    # resource pool
    nnodes = int(config.trainer.nnodes)
    n_gpus_per_node = int(config.trainer.n_gpus_per_node)
    pool_id = "global_pool"
    resource_pool_spec = {pool_id: [n_gpus_per_node] * nnodes}
    mapping = {actor_role: pool_id}
    if config.critic.get("enable", False):
        mapping[Role.Critic] = pool_id

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # reward managers
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    # trainer
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )

    # train
    try:
        trainer.init_workers()
        trainer.fit()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
