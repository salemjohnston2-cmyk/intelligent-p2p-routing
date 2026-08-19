# ============================================================
# INTELLIGENT P2P ROUTING FRAMEWORK
# FINAL YEAR PROJECT — EXPERIMENTAL IMPLEMENTATION
#
# Topic:
# An Intelligent Peer-to-Peer Framework for Efficient Data
# Dissemination Using Deep Learning-Based Routing
#
# IMPORTANT:
# This experiment uses synthetic network conditions.
# Results are experimental simulation results, not real-world
# network measurements.
# ============================================================


# ============================================================
# STEP 1 — SETUP
# ============================================================

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import random
import time
import os
import json
import csv
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

print("=" * 75)
print("INTELLIGENT P2P ROUTING FRAMEWORK")
print("Deep Learning-Based Data Dissemination")
print("=" * 75)

print("\n[STEP 1] Environment ready")
print(f"NumPy: {np.__version__}")
print(f"NetworkX: {nx.__version__}")


# ============================================================
# STEP 2 — EXPERIMENT CONFIGURATION
# ============================================================

print("\n" + "=" * 75)
print("[STEP 2] Experiment Configuration")
print("=" * 75)

N_NODES = 12

# Training and testing are separated by NETWORK SCENARIO.
TRAIN_SCENARIOS = 50
TEST_SCENARIOS = 25

PAIRS_PER_SCENARIO = 12

MAX_PATH_LENGTH = 5
MAX_CANDIDATE_PATHS = 12
DETOUR_ALLOWANCE = 1  # candidate paths may be at most this many hops longer than the true shortest path

PAYLOAD_SIZE_KB = 1024

print(f"  → Network nodes: {N_NODES}")
print(f"  → Training scenarios: {TRAIN_SCENARIOS}")
print(f"  → Testing scenarios: {TEST_SCENARIOS}")
print(f"  → Pairs per scenario: {PAIRS_PER_SCENARIO}")
print(f"  → Maximum candidate path length: {MAX_PATH_LENGTH}")
print(f"  → Maximum candidate routes evaluated: {MAX_CANDIDATE_PATHS}")
print(f"  → Payload size: {PAYLOAD_SIZE_KB} KB")


# ============================================================
# STEP 3 — CREATE P2P TOPOLOGY
# ============================================================

print("\n" + "=" * 75)
print("[STEP 3] Creating P2P Network Topology")
print("=" * 75)


def create_connected_topology(n_nodes=12):
    """
    Create a connected random geometric P2P topology.
    """

    for attempt in range(100):

        G = nx.random_geometric_graph(
            n_nodes,
            radius=0.42,
            seed=SEED + attempt
        )

        if nx.is_connected(G):
            return G

    # Deterministic fallback
    return nx.watts_strogatz_graph(
        n_nodes,
        k=4,
        p=0.3,
        seed=SEED
    )


BASE_GRAPH = create_connected_topology(N_NODES)

print(f"  → Nodes: {BASE_GRAPH.number_of_nodes()}")
print(f"  → Links: {BASE_GRAPH.number_of_edges()}")
print(f"  → Connected: {nx.is_connected(BASE_GRAPH)}")


# ============================================================
# STEP 4 — GENERATE DYNAMIC NETWORK CONDITIONS
# ============================================================

print("\n" + "=" * 75)
print("[STEP 4] Dynamic Network Conditions")
print("=" * 75)


def generate_network_state(base_graph, scenario_id):
    """
    Create a new network state while preserving topology.

    Each link receives:
        latency       : milliseconds
        bandwidth     : Mbps
        congestion    : 0-1
        packet_loss   : probability 0-0.15
    """

    G = base_graph.copy()

    rng = np.random.default_rng(SEED + 1000 + scenario_id)

    for u, v in G.edges():

        G[u][v]["latency"] = rng.uniform(5, 50)

        G[u][v]["bandwidth"] = rng.uniform(10, 100)

        G[u][v]["congestion"] = rng.uniform(0.0, 0.9)

        G[u][v]["packet_loss"] = rng.uniform(0.0, 0.15)

    return G


sample_graph = generate_network_state(BASE_GRAPH, 0)

sample_edge = list(sample_graph.edges())[0]

u, v = sample_edge

print(f"  → Dynamic conditions generated")
print(f"  → Sample edge: {sample_edge}")
print(f"     Latency:     {sample_graph[u][v]['latency']:.2f} ms")
print(f"     Bandwidth:   {sample_graph[u][v]['bandwidth']:.2f} Mbps")
print(f"     Congestion:  {sample_graph[u][v]['congestion']:.2f}")
print(f"     Packet loss: {sample_graph[u][v]['packet_loss']:.3f}")


# ============================================================
# STEP 5 — ROUTE FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 75)
print("[STEP 5] Route Feature Extraction")
print("=" * 75)


FEATURE_NAMES = [
    "total_latency",
    "min_bandwidth",
    "avg_congestion",
    "hop_count",
    "avg_packet_loss"
]


def extract_route_features(G, path):

    if path is None or len(path) < 2:
        return None

    latencies = []
    bandwidths = []
    congestions = []
    losses = []

    for i in range(len(path) - 1):

        u = path[i]
        v = path[i + 1]

        latencies.append(G[u][v]["latency"])
        bandwidths.append(G[u][v]["bandwidth"])
        congestions.append(G[u][v]["congestion"])
        losses.append(G[u][v]["packet_loss"])

    return {
        "total_latency": float(np.sum(latencies)),
        "min_bandwidth": float(np.min(bandwidths)),
        "avg_congestion": float(np.mean(congestions)),
        "hop_count": len(path) - 1,
        "avg_packet_loss": float(np.mean(losses)),
        "path": path
    }


def feature_vector(features):

    return np.array([
        features["total_latency"] / 200.0,
        features["min_bandwidth"] / 100.0,
        features["avg_congestion"],
        features["hop_count"] / MAX_PATH_LENGTH,
        features["avg_packet_loss"]
    ], dtype=float)


print("  → Features:")
for f in FEATURE_NAMES:
    print(f"     • {f}")


# ============================================================
# STEP 6 — ROUTING QUALITY OBJECTIVE
# ============================================================

print("\n" + "=" * 75)
print("[STEP 6] Routing Quality Objective")
print("=" * 75)


def route_quality_score(features):
    """
    Composite route-quality objective.

    Higher score = better route.

    Components:

        30% latency
        25% bandwidth
        20% congestion
        15% packet loss
        10% hop count

    This is the experimental routing objective that the
    neural network learns to approximate.
    """

    latency_score = max(
        0.0,
        1.0 - features["total_latency"] / 200.0
    )

    bandwidth_score = min(
        1.0,
        features["min_bandwidth"] / 100.0
    )

    congestion_score = 1.0 - features["avg_congestion"]

    loss_score = 1.0 - features["avg_packet_loss"] / 0.15

    hop_score = max(
        0.0,
        1.0 - features["hop_count"] / MAX_PATH_LENGTH
    )

    score = (
        0.30 * latency_score +
        0.25 * bandwidth_score +
        0.20 * congestion_score +
        0.15 * loss_score +
        0.10 * hop_score
    )

    return float(np.clip(score, 0.0, 1.0))


print("  → Routing objective defined")
print("  → Latency contribution: 30%")
print("  → Bandwidth contribution: 25%")
print("  → Congestion contribution: 20%")
print("  → Packet-loss contribution: 15%")
print("  → Hop-count contribution: 10%")


# ============================================================
# STEP 7 — CANDIDATE ROUTE GENERATION
# ============================================================

print("\n" + "=" * 75)
print("[STEP 7] Candidate Route Generation")
print("=" * 75)


def get_candidate_routes(G, src, dst):
    """
    Candidate routes are bounded relative to the true shortest
    path length for this pair (shortest + DETOUR_ALLOWANCE hops),
    not a fixed global cutoff. This keeps the candidate set
    realistic: a routing protocol would not seriously consider
    a path several times longer than the shortest available one.
    """

    try:

        shortest_len = nx.shortest_path_length(G, src, dst)

        bounded_cutoff = min(
            MAX_PATH_LENGTH,
            shortest_len + DETOUR_ALLOWANCE
        )

        paths = nx.all_simple_paths(
            G,
            src,
            dst,
            cutoff=bounded_cutoff
        )

        paths = list(paths)[:MAX_CANDIDATE_PATHS]

        return paths

    except nx.NetworkXNoPath:

        return []

    except Exception:

        return []


# Test candidate generation
src_test, dst_test = 0, 5

candidate_test = get_candidate_routes(
    sample_graph,
    src_test,
    dst_test
)

print(
    f"  → Example {src_test} → {dst_test}: "
    f"{len(candidate_test)} candidate routes"
)


# ============================================================
# STEP 8 — BUILD TRAINING DATASET
# ============================================================

print("\n" + "=" * 75)
print("[STEP 8] Building Training Dataset")
print("=" * 75)


def build_training_dataset():

    X = []
    y = []

    scenario_route_counts = []

    rng = random.Random(SEED)

    for scenario_id in range(TRAIN_SCENARIOS):

        G = generate_network_state(
            BASE_GRAPH,
            scenario_id
        )

        scenario_examples = 0

        for _ in range(PAIRS_PER_SCENARIO):

            src, dst = rng.sample(
                list(G.nodes()),
                2
            )

            paths = get_candidate_routes(
                G,
                src,
                dst
            )

            if len(paths) < 2:
                continue

            for path in paths:

                features = extract_route_features(
                    G,
                    path
                )

                if features is None:
                    continue

                X.append(
                    feature_vector(features)
                )

                y.append(
                    route_quality_score(features)
                )

                scenario_examples += 1

        scenario_route_counts.append(
            scenario_examples
        )

        if (scenario_id + 1) % 10 == 0:

            print(
                f"  → Training scenarios processed: "
                f"{scenario_id + 1}/{TRAIN_SCENARIOS}"
            )

    return (
        np.array(X),
        np.array(y),
        scenario_route_counts
    )


X_train_raw, y_train, train_counts = build_training_dataset()

print("\n  → Training dataset complete")
print(f"  → Routing examples: {len(X_train_raw)}")
print(f"  → Input features: {X_train_raw.shape[1]}")
print(f"  → Target: continuous route-quality score")
print(
    f"  → Target range: "
    f"{y_train.min():.3f} - {y_train.max():.3f}"
)


# ============================================================
# STEP 9 — FEATURE SCALING
# ============================================================

print("\n" + "=" * 75)
print("[STEP 9] Feature Scaling")
print("=" * 75)


scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train_raw
)

print("  → StandardScaler fitted using TRAINING data only")
print("  → No test information used during preprocessing")


# ============================================================
# STEP 10 — TRAIN DEEP LEARNING MODEL
# ============================================================

print("\n" + "=" * 75)
print("[STEP 10] Training Deep Learning Model")
print("=" * 75)


model = MLPRegressor(

    hidden_layer_sizes=(32, 16),

    activation="relu",

    solver="adam",

    alpha=0.001,

    learning_rate_init=0.001,

    max_iter=800,

    early_stopping=True,

    validation_fraction=0.10,

    n_iter_no_change=25,

    random_state=SEED
)


print("  → Model: Multi-Layer Perceptron")
print("  → Architecture: 5 → 32 → 16 → 1")
print("  → Activation: ReLU")
print("  → Optimizer: Adam")
print("  → Early stopping: Enabled")

start_time = time.time()

model.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print(f"\n  → Training complete")
print(f"  → Training time: {training_time:.3f} seconds")
print(f"  → Iterations: {model.n_iter_}")


# ============================================================
# STEP 11 — TEST DATASET
# ============================================================

print("\n" + "=" * 75)
print("[STEP 11] Building Unseen Test Scenarios")
print("=" * 75)


def build_test_scenarios():

    scenarios = []

    rng = random.Random(SEED + 999)

    for i in range(TEST_SCENARIOS):

        # Offset ensures these states were never used during
        # training.
        scenario_id = TRAIN_SCENARIOS + i

        G = generate_network_state(
            BASE_GRAPH,
            scenario_id
        )

        pairs = []

        for _ in range(PAIRS_PER_SCENARIO):

            src, dst = rng.sample(
                list(G.nodes()),
                2
            )

            pairs.append(
                (src, dst)
            )

        scenarios.append(
            {
                "scenario_id": scenario_id,
                "graph": G,
                "pairs": pairs
            }
        )

        if (i + 1) % 5 == 0:

            print(
                f"  → Test scenarios prepared: "
                f"{i + 1}/{TEST_SCENARIOS}"
            )

    return scenarios


test_scenarios = build_test_scenarios()

print(
    f"\n  → {len(test_scenarios)} completely unseen "
    f"network scenarios prepared"
)


# ============================================================
# STEP 12 — MODEL GENERALIZATION EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("[STEP 12] Evaluating Model on Unseen Scenarios")
print("=" * 75)


X_test_raw = []
y_test = []

for scenario in test_scenarios:

    G = scenario["graph"]

    for src, dst in scenario["pairs"]:

        paths = get_candidate_routes(
            G,
            src,
            dst
        )

        for path in paths:

            features = extract_route_features(
                G,
                path
            )

            if features is None:
                continue

            X_test_raw.append(
                feature_vector(features)
            )

            y_test.append(
                route_quality_score(features)
            )


X_test_raw = np.array(X_test_raw)
y_test = np.array(y_test)

X_test = scaler.transform(
    X_test_raw
)

y_pred = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n  MODEL GENERALIZATION RESULTS")
print("  " + "-" * 45)
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")


# ============================================================
# STEP 13 — ROUTING METHODS
# ============================================================

print("\n" + "=" * 75)
print("[STEP 13] Routing Strategies")
print("=" * 75)


def conventional_routing(G, src, dst):
    """
    Conventional baseline:
    minimum-hop shortest path.
    """

    try:

        return nx.shortest_path(
            G,
            src,
            dst
        )

    except:

        return None


def intelligent_routing(
    G,
    src,
    dst,
    model,
    scaler
):
    """
    Deep-learning routing.

    The trained MLP predicts the quality score of every
    candidate route and selects the highest predicted score.
    """

    paths = get_candidate_routes(
        G,
        src,
        dst
    )

    if not paths:
        return None

    features_list = []

    valid_paths = []

    for path in paths:

        features = extract_route_features(
            G,
            path
        )

        if features is None:
            continue

        features_list.append(
            feature_vector(features)
        )

        valid_paths.append(
            path
        )

    if not valid_paths:
        return None

    X_routes = scaler.transform(
        np.array(features_list)
    )

    predicted_scores = model.predict(
        X_routes
    )

    best_index = int(
        np.argmax(predicted_scores)
    )

    return valid_paths[best_index]


def oracle_routing(G, src, dst):
    """
    Oracle reference.

    Selects the route with the highest TRUE routing-quality
    score.

    This is NOT a deployable routing method.

    It represents the theoretical upper reference for the
    experimental routing objective.
    """

    paths = get_candidate_routes(
        G,
        src,
        dst
    )

    if not paths:
        return None

    best_path = None
    best_score = -np.inf

    for path in paths:

        features = extract_route_features(
            G,
            path
        )

        score = route_quality_score(
            features
        )

        if score > best_score:

            best_score = score
            best_path = path

    return best_path


print("  → Baseline: shortest-hop routing")
print("  → Proposed: MLP intelligent routing")
print("  → Reference: oracle routing")
print("  → Oracle is used only as an experimental upper reference")


# ============================================================
# STEP 14 — DISSEMINATION MODEL
# ============================================================

print("\n" + "=" * 75)
print("[STEP 14] Data Dissemination Model")
print("=" * 75)


def simulate_dissemination(
    G,
    path,
    payload_size_kb=1024
):
    """
    Deterministic expected-performance dissemination model.

    Metrics:

        latency       = sum of link latency
        bottleneck    = minimum link bandwidth
        PDR            = product of per-hop success probability
        throughput    = bottleneck rate adjusted by losses/PDR
        transmission  = payload / throughput
    """

    if path is None or len(path) < 2:
        return None

    latencies = []
    bandwidths = []
    losses = []

    for i in range(len(path) - 1):

        u = path[i]
        v = path[i + 1]

        latencies.append(
            G[u][v]["latency"]
        )

        bandwidths.append(
            G[u][v]["bandwidth"]
        )

        losses.append(
            G[u][v]["packet_loss"]
        )

    total_latency_ms = float(
        np.sum(latencies)
    )

    bottleneck_mbps = float(
        np.min(bandwidths)
    )

    avg_loss = float(
        np.mean(losses)
    )

    # Probability that data survives every hop
    pdr = float(
        np.prod(
            [1.0 - loss for loss in losses]
        )
    )

    # Convert Mbps → KB/s
    bottleneck_kbs = (
        bottleneck_mbps * 125.0
    )

    # Expected effective throughput
    throughput_kbs = (
        bottleneck_kbs *
        (1.0 - avg_loss) *
        pdr
    )

    if throughput_kbs > 0:

        transmission_time_ms = (
            payload_size_kb /
            throughput_kbs
        ) * 1000.0

    else:

        transmission_time_ms = np.inf

    # End-to-end dissemination delay
    end_to_end_delay_ms = (
        total_latency_ms +
        transmission_time_ms
    )

    return {

        "latency": total_latency_ms,

        "throughput": throughput_kbs,

        "pdr": pdr,

        "hop_count": len(path) - 1,

        "transmission_time": transmission_time_ms,

        "end_to_end_delay": end_to_end_delay_ms,

        "path": path,

        "bottleneck_bandwidth": bottleneck_mbps,

        "avg_loss": avg_loss
    }


print("  → Dissemination model ready")
print("  → Payload size incorporated into transmission time")
print("  → Expected PDR used instead of a single random delivery event")


# ============================================================
# STEP 15 — RUN CONTROLLED COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("[STEP 15] Controlled Experimental Comparison")
print("=" * 75)


def run_experiment():

    results = {
        "conventional": [],
        "intelligent": [],
        "oracle": []
    }

    total_pairs = 0

    for scenario_index, scenario in enumerate(
        test_scenarios
    ):

        G = scenario["graph"]

        for src, dst in scenario["pairs"]:

            conv_path = conventional_routing(
                G,
                src,
                dst
            )

            int_path = intelligent_routing(
                G,
                src,
                dst,
                model,
                scaler
            )

            oracle_path = oracle_routing(
                G,
                src,
                dst
            )

            conv_result = simulate_dissemination(
                G,
                conv_path,
                PAYLOAD_SIZE_KB
            )

            int_result = simulate_dissemination(
                G,
                int_path,
                PAYLOAD_SIZE_KB
            )

            oracle_result = simulate_dissemination(
                G,
                oracle_path,
                PAYLOAD_SIZE_KB
            )

            if conv_result is not None:
                results["conventional"].append(
                    conv_result
                )

            if int_result is not None:
                results["intelligent"].append(
                    int_result
                )

            if oracle_result is not None:
                results["oracle"].append(
                    oracle_result
                )

            total_pairs += 1

        if (scenario_index + 1) % 5 == 0:

            print(
                f"  → Completed "
                f"{scenario_index + 1}/{TEST_SCENARIOS} "
                f"test scenarios"
            )

    return results, total_pairs


experiment_results, total_pairs = run_experiment()

print("\n  → Experiment complete")
print(f"  → Test pairs: {total_pairs}")

for method, values in experiment_results.items():

    print(
        f"  → {method.capitalize()} results: "
        f"{len(values)}"
    )


# ============================================================
# STEP 16 — PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 75)
print("[STEP 16] Performance Metrics")
print("=" * 75)


def calculate_performance(results):

    if not results:
        return None

    latency = np.mean(
        [r["latency"] for r in results]
    )

    throughput = np.mean(
        [r["throughput"] for r in results]
    )

    pdr = np.mean(
        [r["pdr"] for r in results]
    )

    hops = np.mean(
        [r["hop_count"] for r in results]
    )

    transmission = np.mean(
        [r["transmission_time"] for r in results]
    )

    end_to_end = np.mean(
        [r["end_to_end_delay"] for r in results]
    )

    efficiency = (
        (pdr * throughput) /
        end_to_end
    )

    return {

        "latency": latency,

        "throughput": throughput,

        "pdr": pdr,

        "hops": hops,

        "transmission_time": transmission,

        "end_to_end_delay": end_to_end,

        "efficiency": efficiency
    }


metrics = {}

for method, results in experiment_results.items():

    metrics[method] = calculate_performance(
        results
    )


print("\n")
print(
    f"{'METRIC':<25}"
    f"{'CONVENTIONAL':>16}"
    f"{'INTELLIGENT':>16}"
    f"{'ORACLE':>16}"
)

print("-" * 75)

print(
    f"{'Latency (ms)':<25}"
    f"{metrics['conventional']['latency']:>16.2f}"
    f"{metrics['intelligent']['latency']:>16.2f}"
    f"{metrics['oracle']['latency']:>16.2f}"
)

print(
    f"{'Throughput (KB/s)':<25}"
    f"{metrics['conventional']['throughput']:>16.2f}"
    f"{metrics['intelligent']['throughput']:>16.2f}"
    f"{metrics['oracle']['throughput']:>16.2f}"
)

print(
    f"{'PDR':<25}"
    f"{metrics['conventional']['pdr']:>16.4f}"
    f"{metrics['intelligent']['pdr']:>16.4f}"
    f"{metrics['oracle']['pdr']:>16.4f}"
)

print(
    f"{'Average Hops':<25}"
    f"{metrics['conventional']['hops']:>16.2f}"
    f"{metrics['intelligent']['hops']:>16.2f}"
    f"{metrics['oracle']['hops']:>16.2f}"
)

print(
    f"{'Transmission Time (ms)':<25}"
    f"{metrics['conventional']['transmission_time']:>16.2f}"
    f"{metrics['intelligent']['transmission_time']:>16.2f}"
    f"{metrics['oracle']['transmission_time']:>16.2f}"
)

print(
    f"{'End-to-End Delay (ms)':<25}"
    f"{metrics['conventional']['end_to_end_delay']:>16.2f}"
    f"{metrics['intelligent']['end_to_end_delay']:>16.2f}"
    f"{metrics['oracle']['end_to_end_delay']:>16.2f}"
)

print(
    f"{'Efficiency (defined)':<25}"
    f"{metrics['conventional']['efficiency']:>16.4f}"
    f"{metrics['intelligent']['efficiency']:>16.4f}"
    f"{metrics['oracle']['efficiency']:>16.4f}"
)


# ============================================================
# STEP 17 — PERCENTAGE IMPROVEMENTS
# ============================================================

print("\n" + "=" * 75)
print("[STEP 17] Intelligent Routing Improvement")
print("=" * 75)


def percentage_change(
    conventional,
    intelligent
):

    if conventional == 0:
        return 0.0

    return (
        (intelligent - conventional) /
        abs(conventional)
    ) * 100.0


conv = metrics["conventional"]
intel = metrics["intelligent"]


latency_change = percentage_change(
    conv["latency"],
    intel["latency"]
)

throughput_change = percentage_change(
    conv["throughput"],
    intel["throughput"]
)

pdr_change = percentage_change(
    conv["pdr"],
    intel["pdr"]
)

hop_change = percentage_change(
    conv["hops"],
    intel["hops"]
)

efficiency_change = percentage_change(
    conv["efficiency"],
    intel["efficiency"]
)

print(
    f"  Latency change:       {latency_change:+.2f}%"
)

print(
    f"  Throughput change:    {throughput_change:+.2f}%"
)

print(
    f"  PDR change:           {pdr_change:+.2f}%"
)

print(
    f"  Hop-count change:     {hop_change:+.2f}%"
)

print(
    f"  Efficiency change:    {efficiency_change:+.2f}%"
)


# ============================================================
# STEP 18 — ROUTE SELECTION ACCURACY
# ============================================================

print("\n" + "=" * 75)
print("[STEP 18] Route Selection Analysis")
print("=" * 75)


def route_selection_analysis():

    intelligent_matches = 0
    total = 0

    conventional_matches = 0

    for scenario in test_scenarios:

        G = scenario["graph"]

        for src, dst in scenario["pairs"]:

            paths = get_candidate_routes(
                G,
                src,
                dst
            )

            if len(paths) < 2:
                continue

            scored = []

            for path in paths:

                features = extract_route_features(
                    G,
                    path
                )

                score = route_quality_score(
                    features
                )

                scored.append(
                    (path, score)
                )

            oracle_path = max(
                scored,
                key=lambda x: x[1]
            )[0]

            intelligent_path = intelligent_routing(
                G,
                src,
                dst,
                model,
                scaler
            )

            conventional_path = conventional_routing(
                G,
                src,
                dst
            )

            if intelligent_path == oracle_path:
                intelligent_matches += 1

            if conventional_path == oracle_path:
                conventional_matches += 1

            total += 1

    return (
        intelligent_matches,
        conventional_matches,
        total
    )


(
    intelligent_matches,
    conventional_matches,
    selection_total
) = route_selection_analysis()


if selection_total > 0:

    intelligent_selection_accuracy = (
        intelligent_matches /
        selection_total
    )

    conventional_selection_accuracy = (
        conventional_matches /
        selection_total
    )

else:

    intelligent_selection_accuracy = 0
    conventional_selection_accuracy = 0


print(
    f"  → Intelligent route matches oracle: "
    f"{intelligent_matches}/{selection_total}"
)

print(
    f"  → Intelligent selection accuracy: "
    f"{intelligent_selection_accuracy:.4f}"
)

print(
    f"  → Conventional route matches oracle: "
    f"{conventional_matches}/{selection_total}"
)

print(
    f"  → Conventional selection accuracy: "
    f"{conventional_selection_accuracy:.4f}"
)


# ============================================================
# STEP 19 — VISUALIZATION DIRECTORY
# ============================================================

print("\n" + "=" * 75)
print("[STEP 19] Generating Figures")
print("=" * 75)


os.makedirs(
    "figures",
    exist_ok=True
)


# ------------------------------------------------------------
# Figure 1 — Network topology
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

pos = nx.spring_layout(
    BASE_GRAPH,
    seed=SEED
)

nx.draw_networkx_nodes(
    BASE_GRAPH,
    pos,
    node_size=600
)

nx.draw_networkx_edges(
    BASE_GRAPH,
    pos,
    width=1.5
)

nx.draw_networkx_labels(
    BASE_GRAPH,
    pos,
    font_weight="bold"
)

plt.title(
    "Simulated P2P Network Topology"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "figures/figure1_topology.png",
    dpi=250
)

plt.close()


# ------------------------------------------------------------
# Figure 2 — Model prediction
# ------------------------------------------------------------

sample_count = min(
    300,
    len(y_test)
)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test[:sample_count],
    y_pred[:sample_count],
    alpha=0.5
)

plt.xlabel(
    "True Route Quality Score"
)

plt.ylabel(
    "Predicted Route Quality Score"
)

plt.title(
    "Deep Learning Route Quality Prediction"
)

plt.tight_layout()

plt.savefig(
    "figures/figure2_model_prediction.png",
    dpi=250
)

plt.close()


# ------------------------------------------------------------
# Figure 3 — Latency
# ------------------------------------------------------------

methods = [
    "Conventional",
    "Intelligent",
    "Oracle"
]

latency_values = [
    metrics["conventional"]["latency"],
    metrics["intelligent"]["latency"],
    metrics["oracle"]["latency"]
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    latency_values
)

plt.ylabel(
    "Average Latency (ms)"
)

plt.title(
    "Average End-to-End Link Latency"
)

plt.tight_layout()

plt.savefig(
    "figures/figure3_latency.png",
    dpi=250
)

plt.close()


# ------------------------------------------------------------
# Figure 4 — Throughput
# ------------------------------------------------------------

throughput_values = [
    metrics["conventional"]["throughput"],
    metrics["intelligent"]["throughput"],
    metrics["oracle"]["throughput"]
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    throughput_values
)

plt.ylabel(
    "Throughput (KB/s)"
)

plt.title(
    "Average Data Throughput"
)

plt.tight_layout()

plt.savefig(
    "figures/figure4_throughput.png",
    dpi=250
)

plt.close()


# ------------------------------------------------------------
# Figure 5 — PDR
# ------------------------------------------------------------

pdr_values = [
    metrics["conventional"]["pdr"],
    metrics["intelligent"]["pdr"],
    metrics["oracle"]["pdr"]
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    pdr_values
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Packet Delivery Ratio"
)

plt.title(
    "Average Packet Delivery Ratio"
)

plt.tight_layout()

plt.savefig(
    "figures/figure5_pdr.png",
    dpi=250
)

plt.close()


# ------------------------------------------------------------
# Figure 6 — Hop count
# ------------------------------------------------------------

hop_values = [
    metrics["conventional"]["hops"],
    metrics["intelligent"]["hops"],
    metrics["oracle"]["hops"]
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    hop_values
)

plt.ylabel(
    "Average Hop Count"
)

plt.title(
    "Average Route Hop Count"
)

plt.tight_layout()

plt.savefig(
    "figures/figure6_hops.png",
    dpi=250
)

plt.close()


# ------------------------------------------------------------
# Figure 7 — Efficiency
# ------------------------------------------------------------

efficiency_values = [
    metrics["conventional"]["efficiency"],
    metrics["intelligent"]["efficiency"],
    metrics["oracle"]["efficiency"]
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    efficiency_values
)

plt.ylabel(
    "Defined Routing Efficiency"
)

plt.title(
    "Routing Efficiency Comparison"
)

plt.tight_layout()

plt.savefig(
    "figures/figure7_efficiency.png",
    dpi=250
)

plt.close()


print("  → Figure 1: topology")
print("  → Figure 2: model prediction")
print("  → Figure 3: latency")
print("  → Figure 4: throughput")
print("  → Figure 5: PDR")
print("  → Figure 6: hop count")
print("  → Figure 7: efficiency")


# ============================================================
# STEP 20 — SAVE NUMERICAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("[STEP 20] Saving Experimental Results")
print("=" * 75)


results_summary = {

    "project": (
        "Intelligent P2P Routing Framework"
    ),

    "experiment": {
        "nodes": N_NODES,
        "training_scenarios": TRAIN_SCENARIOS,
        "testing_scenarios": TEST_SCENARIOS,
        "pairs_per_scenario": PAIRS_PER_SCENARIO,
        "payload_kb": PAYLOAD_SIZE_KB
    },

    "training": {
        "examples": int(len(X_train_raw)),
        "features": FEATURE_NAMES,
        "architecture": "5-32-16-1",
        "training_time_seconds": training_time,
        "iterations": int(model.n_iter_)
    },

    "model_generalization": {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    },

    "route_selection": {
        "intelligent_selection_accuracy":
            float(intelligent_selection_accuracy),

        "conventional_selection_accuracy":
            float(conventional_selection_accuracy)
    },

    "performance": metrics,

    "percentage_change": {
        "latency": float(latency_change),
        "throughput": float(throughput_change),
        "pdr": float(pdr_change),
        "hop_count": float(hop_change),
        "efficiency": float(efficiency_change)
    }
}


with open(
    "experiment_results.json",
    "w"
) as f:

    json.dump(
        results_summary,
        f,
        indent=4
    )


# CSV summary

with open(
    "performance_comparison.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Metric",
        "Conventional",
        "Intelligent",
        "Oracle"
    ])

    writer.writerow([
        "Latency_ms",
        metrics["conventional"]["latency"],
        metrics["intelligent"]["latency"],
        metrics["oracle"]["latency"]
    ])

    writer.writerow([
        "Throughput_KB_s",
        metrics["conventional"]["throughput"],
        metrics["intelligent"]["throughput"],
        metrics["oracle"]["throughput"]
    ])

    writer.writerow([
        "PDR",
        metrics["conventional"]["pdr"],
        metrics["intelligent"]["pdr"],
        metrics["oracle"]["pdr"]
    ])

    writer.writerow([
        "Average_Hops",
        metrics["conventional"]["hops"],
        metrics["intelligent"]["hops"],
        metrics["oracle"]["hops"]
    ])

    writer.writerow([
        "End_to_End_Delay_ms",
        metrics["conventional"]["end_to_end_delay"],
        metrics["intelligent"]["end_to_end_delay"],
        metrics["oracle"]["end_to_end_delay"]
    ])

    writer.writerow([
        "Efficiency",
        metrics["conventional"]["efficiency"],
        metrics["intelligent"]["efficiency"],
        metrics["oracle"]["efficiency"]
    ])

print("\nFiles created:")
print("  → experiment_results.json")
print("  → performance_comparison.csv")
print("  → figures/")

print("\n" + "=" * 75)
print("EXPERIMENT COMPLETE")
print("=" * 75)
