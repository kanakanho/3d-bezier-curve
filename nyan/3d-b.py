import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import scipy.special
import json
import subprocess
from mpl_toolkits.mplot3d import Axes3D


def cubic_bezier_point(p0, p1, p2, p3, t):
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )

# 3次ベジェ曲線をプロットする
def plot_3d_bezieres(cubic_bezierses, original_points=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    t_values = np.linspace(0, 1, 300)

    if original_points is not None:
        original_x = [p[0] for p in original_points]
        original_y = [p[1] for p in original_points]
        original_z = [p[2] for p in original_points]
        ax.scatter(
            original_x,
            original_y,
            original_z,
            color="blue",
            marker=".",
            s=10,
            alpha=0.1,
            label="元の点",
        )

    n = -1
    for cubic_beziers in cubic_bezierses:
        n += 1
        for i, cubic in enumerate(cubic_beziers):
            curve = [cubic_bezier_point(*cubic, t) for t in t_values]
            ctrl_x = [p[0] for p in cubic]
            ctrl_y = [p[1] for p in cubic]
            ctrl_z = [p[2] for p in cubic]

            ax.plot(
                *zip(*curve),
                color="orange",
                ls="--",
                alpha=0.8,
                label="3次ベジェ_"+ str(n) if i == 0 else None
            )
            # ax.scatter(
            #     ctrl_x,
            #     ctrl_y,
            #     ctrl_z,
            #     color="cyan",
            #     marker="o",
            #     s=50,
            #     alpha=0.5,
            #     label="3次制御点_"+ str(n) if i == 0 else None,
            # )

    ax.set_title("3次元ベジェ曲線と制御点")
    ax.set_xlabel("X軸")
    ax.set_ylabel("Y軸")
    ax.set_zlabel("Z軸")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.show()

# 3次ベジェ曲線をプロットする
def plot_3d_bezier(cubic_beziers, original_points=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    t_values = np.linspace(0, 1, 300)

    if original_points is not None:
        original_x = [p[0] for p in original_points]
        original_y = [p[1] for p in original_points]
        original_z = [p[2] for p in original_points]
        ax.scatter(
            original_x,
            original_y,
            original_z,
            color="blue",
            marker=".",
            s=10,
            alpha=0.1,
            label="元の点",
        )

    for i, cubic in enumerate(cubic_beziers):
        curve = [cubic_bezier_point(*cubic, t) for t in t_values]
        ctrl_x = [p[0] for p in cubic]
        ctrl_y = [p[1] for p in cubic]
        ctrl_z = [p[2] for p in cubic]

        ax.plot(
            *zip(*curve),
            color="orange",
            ls="--",
            alpha=0.8,
            label="3次ベジェ" if i == 0 else None
        )
        ax.scatter(
            ctrl_x,
            ctrl_y,
            ctrl_z,
            color="cyan",
            marker="o",
            s=50,
            alpha=0.5,
            label="3次制御点" if i == 0 else None,
        )

    ax.set_title("3次元ベジェ曲線と制御点")
    ax.set_xlabel("X軸")
    ax.set_ylabel("Y軸")
    ax.set_zlabel("Z軸")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.show()


result = subprocess.run(["swift", "3d.swift"])
# result = subprocess.run(["swift", "run", "3d-b.swift"])

cubic_bezierses = []
# with open("cubic_segments_3d.json", "r", encoding="utf-8") as f:
for i in range(27):
    cubic_beziers = []
    with open(f"cubic_segments_3d_curvature_{i}.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
        for segment in json_data:
            p0 = segment["p0"]
            p1 = segment["p1"]
            p2 = segment["p2"]
            p3 = segment["p3"]
            cubic_beziers.append(
                (
                    (p0["x"], p0["y"], p0["z"]),
                    (p1["x"], p1["y"], p1["z"]),
                    (p2["x"], p2["y"], p2["z"]),
                    (p3["x"], p3["y"], p3["z"]),
                )
            )
    cubic_bezierses.append(cubic_beziers)

original_points = []
with open("original_points_3d.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    """
    [
    {
        "y" : 2.301418259191532,
        "x" : -5,
        "z" : -125
    },
    {
        "y" : 2.30888188037134,
        "z" : -124.25149900000001,
        "x" : -4.99
    },
    ]
    """
    for point in json_data:
        original_points.append((point["x"], point["y"], point["z"]))


# plot_3d_bezier(cubic_beziers, original_points=original_points)
plot_3d_bezieres(cubic_bezierses, original_points=original_points)
