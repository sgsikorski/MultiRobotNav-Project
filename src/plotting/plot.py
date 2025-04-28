import matplotlib.pyplot as plt


def plot(ylabel, units, light, auction, llm):
    fig, ax = plt.subplots(1, 1)

    flow_rates = [30, 60, 90, 120]

    ax.set_xlabel("Vehicle flow rate (veh/min)")
    ax.set_ylabel(f"{ylabel} {units}")

    ax.set_ylim(
        0.8 * min(min(light), min(auction), min(llm)),
        1.2 * max(max(light), max(auction), max(llm)),
    )

    ax.plot(
        flow_rates,
        auction,
        label="Auction",
        marker="o",
        color="red",
    )

    ax.plot(flow_rates, light, label="Traffic Light", marker="o", color="black")

    ax.plot(flow_rates, llm, label="LLM-based", marker="o", color="blue")

    fig.legend()
    fig.show()
    fig.savefig(f"./plotting/{ylabel}_comparison.png")


def plot_throughput():
    light = [0.31, 0.38, 0.54, 0.71]
    auction = [0.0075 * 60, 0.0152 * 60, 0.022 * 60, 0.0268 * 60]
    llm = [0.0083 * 60, 0.0159 * 60, 0.0208 * 60, 0.0261 * 60]

    plot("Throughput", "(Veh/min)", light, auction, llm)


def plot_ttg():
    # Traffic light data
    light = [24, 27, 29, 31]
    auction = [20.0075, 20, 24, 24]
    llm = [19.94, 19.98, 19.97, 19.99]

    plot("Average Time to Goal", "(s)", light, auction, llm)


def plot_llm_avg_time():
    flow_rates = [30, 60, 90, 120]
    llm_times = [4.886, 5.423, 4.893, 10.1237]
    llm_priority = [0.5262, 0.5277, 0.5492, 0.5281]

    fig, ax = plt.subplots(1, 1)

    ax.set_xlabel("Vehicle flow rate (veh/min)")
    ax.set_ylabel(f"Time (s)")

    ax.set_ylim(0, 1.3 * max(max(llm_times), max(llm_priority)))

    ax.plot(flow_rates, llm_times, label="V2V", marker="o", color="blue")
    ax.plot(flow_rates, llm_priority, label="Priority", marker="o", color="red")

    fig.legend()
    fig.show()
    fig.savefig(f"./plotting/llm_time.png")


def main():
    plot_throughput()
    plot_ttg()
    plot_llm_avg_time()


if __name__ == "__main__":
    main()
