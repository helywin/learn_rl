import itertools

import numpy as np
import gymnasium

env = gymnasium.make("CartPole-v1", render_mode="human")
observation, info = env.reset()


def no_control():
    global observation, info
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, terminated, truncated, info)
        if terminated or truncated:  # 检查是否需要重置环境
            observation, info = env.reset()

    env.close()


def simple_control():
    global observation, info
    for _ in range(1000):
        env.render()
        # action = env.action_space.sample()
        theta = observation[2]  # 获取角度
        action = 0 if theta < 0 else 1  # 根据角度选择动作
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, terminated, truncated, info)
        if terminated or truncated:  # 检查是否需要重置环境
            observation, info = env.reset()

    env.close()

def pid_control():
    global observation, info
    # PID 参数
    Kp, Ki, Kd = 100.0, 1.0, 20.0
    integral = 0.0
    prev_error = 0.0
    dt = 0.02

    for _ in range(1000):
        env.render()
        # 以杆子偏角为误差
        theta = observation[2]
        error = theta
        integral += error * dt
        derivative = (error - prev_error) / dt
        prev_error = error

        # 连续控制量 u，再映射到离散动作
        u = Kp * error + Ki * integral + Kd * derivative
        action = 1 if u > 0 else 0

        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, terminated, truncated, info)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

def mpc_control():
    global observation, info
    # MPC 参数
    N = 3  # 预测步长
    Q = np.diag([1.0, 1.0, 10.0, 1.0])
    R = 0.1
    # 线性化模型参数（近似离散化）
    dt = 0.02
    M, m, l, g = 1.0, 0.1, 0.5, 9.8
    A = np.array([
        [1, dt, 0, 0],
        [0, 1, -(m * g * dt) / M, 0],
        [0, 0, 1, dt],
        [0, 0, ((M + m) * g * dt) / (l * M), 1]
    ])
    B = np.array([
        [0],
        [dt / M],
        [0],
        [-dt / (l * M)]
    ])

    for _ in range(1000):
        env.render()
        x0 = observation.copy()
        best_cost = float('inf')
        best_u0 = 0
        # 枚举所有 2^N 种动作序列
        for seq in itertools.product([0, 1], repeat=N):
            cost = 0.0
            x = x0.copy()
            for a in seq:
                # 离散动作映射到连续力：0→-10, 1→+10
                u = (2 * a - 1) * 10.0
                x = A.dot(x) + B.flatten() * u
                cost += x.T.dot(Q).dot(x) + R * (u ** 2)
            if cost < best_cost:
                best_cost, best_u0 = cost, seq[0]

        # 执行最优序列的第一个动作
        action = best_u0
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, terminated, truncated, info)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    # no_control()
    # simple_control()
    pid_control()
    # mpc_control()
