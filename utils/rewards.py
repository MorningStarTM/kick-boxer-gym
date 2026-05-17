import numpy as np


class KickBoxingReward:
    def __init__(self, config: dict):
        rc = config.get("reward", {})
        self.strike_w = rc.get("strike_weight", 5.0)
        self.balance_w = rc.get("balance_weight", 2.0)
        self.energy_w = rc.get("energy_weight", -0.01)
        self.posture_w = rc.get("posture_weight", 1.0)
        self.alive_bonus = rc.get("alive_bonus", 0.5)
        self.fall_penalty = rc.get("fall_penalty", -10.0)
        self.target_zones = rc.get("target_zones", {"head": 3.0, "torso": 2.0, "legs": 1.0})

    def compute(
        self,
        agent_pos: np.ndarray,
        agent_vel: np.ndarray,
        opponent_pos: np.ndarray,
        contacts: list,
        joint_torques: np.ndarray,
        com_height: float,
        upright_proj: float,
        is_fallen: bool,
    ) -> tuple[float, dict]:
        info = {}

        strike_reward = self._strike_reward(contacts, opponent_pos)
        info["strike"] = strike_reward

        balance_reward = self._balance_reward(com_height, upright_proj)
        info["balance"] = balance_reward

        energy_penalty = self.energy_w * np.sum(joint_torques ** 2)
        info["energy"] = energy_penalty

        posture_reward = self.posture_w * max(0, upright_proj)
        info["posture"] = posture_reward

        if is_fallen:
            info["alive"] = self.fall_penalty
            total = self.fall_penalty
        else:
            info["alive"] = self.alive_bonus
            total = (
                self.strike_w * strike_reward
                + self.balance_w * balance_reward
                + energy_penalty
                + posture_reward
                + self.alive_bonus
            )

        info["total"] = total
        return total, info

    def _strike_reward(self, contacts: list, opponent_pos: np.ndarray) -> float:
        reward = 0.0
        strike_links = {"left_hand", "right_hand", "left_foot", "right_foot"}
        target_links = {
            "head": {"head"},
            "torso": {"spine1", "spine2", "spine3", "pelvis"},
            "legs": {"left_upper_leg", "right_upper_leg", "left_lower_leg", "right_lower_leg"},
        }

        for contact in contacts:
            link_a, link_b = contact.get("link_a", ""), contact.get("link_b", "")
            force = contact.get("normal_force", 0.0)

            attacker = link_a if link_a in strike_links else (link_b if link_b in strike_links else None)
            if attacker is None:
                continue

            defender = link_b if link_a == attacker else link_a
            for zone, zone_links in target_links.items():
                if defender in zone_links:
                    reward += self.target_zones.get(zone, 1.0) * min(force / 100.0, 1.0)
                    break

        return reward

    def _balance_reward(self, com_height: float, upright_proj: float) -> float:
        height_reward = np.exp(-5.0 * max(0, 0.85 - com_height) ** 2)
        upright_reward = max(0, upright_proj)
        return 0.5 * height_reward + 0.5 * upright_reward
