from highway_env.envs import IntersectionEnv
from config import *


class TrafficLightIntersectionEnv(IntersectionEnv):

    traffic_light_cycle = 50  # steps for one phase (green or red)
    traffic_light_state = "green"
    steps = 0
    passed_vehicle_count = 0

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            {
                "screen_width": 800,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "scaling": 5.5,
                "simulation_frequency": CYCLE_FREQUENCY,
                "spawn_probability": FLOW_RATE / CYCLE_FREQUENCY,
            }
        )
        return config

    def step(self, action):
        self.steps += 1
        obs, reward, terminated, truncated, info = super().step(action)

        # Update traffic light state
        if self.steps % (2 * self.traffic_light_cycle) < self.traffic_light_cycle:
            self.traffic_light_state = "green"
        else:
            self.traffic_light_state = "red"

        # Control vehicles based on traffic light
        for vehicle in self.road.vehicles:
            if vehicle.crashed:
                terminated = True
            if self.traffic_light_state == "red" and self.is_near_intersection(vehicle):
                vehicle.target_speed = 0

        # Throughput counting
        for vehicle in self.road.vehicles:
            if not hasattr(vehicle, "spawned"):
                vehicle.spawned = True
                vehicle.spawned_step = self.steps
            if (
                np.abs(vehicle.position[0] > 5.0) or np.abs(vehicle.position[1]) > 5.0
            ) and not hasattr(vehicle, "counted"):
                self.passed_vehicle_count += 1
                vehicle.counted = True
            if np.abs(vehicle.position[0]) > 90.0 or np.abs(vehicle.position[1]) > 90.0:
                if not hasattr(vehicle, "time_to_goal"):
                    self.time_to_goal = []
                self.time_to_goal.append(self.steps - vehicle.spawned_step)

        average_time_to_goal = (
            (sum(self.time_to_goal) / len(self.time_to_goal))
            if hasattr(self, "time_to_goal") and self.time_to_goal
            else 0.0
        )

        # Return throughput in info
        info["throughput"] = self.passed_vehicle_count / (self.steps + 1)
        info["traffic_light"] = self.traffic_light_state
        info["ttg"] = average_time_to_goal
        return obs, reward, terminated, truncated, info

    def is_near_intersection(self, vehicle):
        return abs(vehicle.position[0]) < 5.0 and abs(vehicle.position[1]) < 5.0
