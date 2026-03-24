"""
Streamlit dashboard — monitor training and visualize agents playing.

Usage:
    cd src
    streamlit run app.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

import pygame
pygame.init()

import utils.config as config

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Synergy Bots",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 Synergy Bots — Dashboard")

# ---------------------------------------------------------------------------
# Helper: render game grid as matplotlib figure
# ---------------------------------------------------------------------------

_ROBOT_COLORS_FILL = ["#E74C3C", "#27AE60", "#2980B9"]
_ROBOT_COLORS_CLEAN = ["#FADBD8", "#D5F5E3", "#D6EAF8"]


def render_grid(grid_state: list, robot_positions: list, cleaned_cells_per_robot: list) -> plt.Figure:
    """
    Renders the 8x6 game grid as a matplotlib figure.
    - White  = uncleaned cell
    - Dark   = obstacle
    - Coloured = cleaned by robot i
    - Circle  = robot position
    """
    grid_h = config.GRID_HEIGHT
    grid_w = config.GRID_WIDTH

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, grid_w)
    ax.set_ylim(0, grid_h)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Background grid
    for row in range(grid_h):
        for col in range(grid_w):
            ax.add_patch(patches.Rectangle(
                (col, row), 1, 1,
                linewidth=0.5, edgecolor="#BDC3C7", facecolor="#FDFEFE",
            ))

    # Obstacles (grid_state == 1 AND not a cleaned cell)
    g = np.array(grid_state)
    for row in range(grid_h):
        for col in range(grid_w):
            if g[row, col] == 1:
                # Check if it's an obstacle (not a cleaned cell)
                cleaned = any(
                    [col, row] in cells or [col, row] in cells
                    for cells in cleaned_cells_per_robot
                )
                if not cleaned:
                    ax.add_patch(patches.Rectangle(
                        (col, row), 1, 1,
                        linewidth=0.5, edgecolor="#7F8C8D", facecolor="#566573",
                    ))

    # Cleaned cells per robot
    for i, cells in enumerate(cleaned_cells_per_robot):
        color = _ROBOT_COLORS_CLEAN[i % len(_ROBOT_COLORS_CLEAN)]
        for cx, cy in cells:
            ax.add_patch(patches.Rectangle(
                (cx, cy), 1, 1,
                linewidth=0.5, edgecolor="#BDC3C7", facecolor=color, alpha=0.85,
            ))

    # Robots
    for i, (rx, ry) in enumerate(robot_positions):
        color = _ROBOT_COLORS_FILL[i % len(_ROBOT_COLORS_FILL)]
        ax.add_patch(plt.Circle((rx + 0.5, ry + 0.5), 0.38, color=color, zorder=5))
        ax.text(rx + 0.5, ry + 0.5, str(i + 1), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", zorder=6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helper: load metrics CSV
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3)
def load_metrics():
    if not os.path.exists(config.METRICS_CSV):
        return None
    try:
        df = pd.read_csv(config.METRICS_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper: load live state snapshot
# ---------------------------------------------------------------------------

def load_snapshot():
    if not os.path.exists(config.STATE_SNAPSHOT_FILE):
        return None
    try:
        with open(config.STATE_SNAPSHOT_FILE) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Run inference episode (for visual play tab)
# ---------------------------------------------------------------------------

def run_visual_episode(placeholder_grid, placeholder_info):
    from game import Game
    from models.mappo_nn import MAPPOAgent

    if not os.path.exists(config.SAVE_PATH):
        st.error(f"Modelo não encontrado em `{config.SAVE_PATH}`. Treine primeiro com `python train.py`.")
        return

    game = Game()
    agent = MAPPOAgent(dim_obs_robot=config.NN_INPUT_SIZE, num_actions=5)

    game.reset()
    states = [r.get_state(game.all_robots_list) for r in game.all_robots_list]
    done = False
    step = 0

    while not done and step < config.MAX_STEPS:
        actions_np, _, _, _, _ = agent.select_actions(states, evaluate=True)
        states, _, dones, _ = game.step(actions_np)
        done = any(dones)
        step += 1

        robot_positions = [
            [r.rect.x // config.CELL_SIZE, r.rect.y // config.CELL_SIZE]
            for r in game.all_robots_list
        ]
        cleaned = [
            [[px // config.CELL_SIZE, py // config.CELL_SIZE] for px, py in r.clear_cells]
            for r in game.all_robots_list
        ]

        fig = render_grid(game.get_grid_state().tolist(), robot_positions, cleaned)
        placeholder_grid.pyplot(fig)
        plt.close(fig)

        uncleaned = game.count_uncleaned_cells()
        status = "✅ Concluído!" if done else f"🔄 Passo {step}/{config.MAX_STEPS}"
        placeholder_info.markdown(
            f"**{status}** &nbsp;|&nbsp; Células sujas: **{uncleaned}** &nbsp;|&nbsp; Passos: **{step}**"
        )
        time.sleep(0.12)

    game.close()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_metrics, tab_live, tab_play = st.tabs([
    "📊 Métricas de Treinamento",
    "🔴 Monitor ao Vivo",
    "🎮 Visualizar Agentes",
])

# ── Tab 1: Training metrics ──────────────────────────────────────────────────

with tab_metrics:
    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    with col_info:
        st.caption(f"Lendo: `{config.METRICS_CSV}`")

    df = load_metrics()

    if df is None or df.empty:
        st.info("Nenhum dado de treinamento encontrado. Execute `python train.py` para iniciar.")
    else:
        total_ep = int(df["episode"].max())
        completed = int(df["completed"].sum())
        best_steps = int(df[df["completed"] == 1]["steps"].min()) if completed > 0 else "—"
        last_reward = round(float(df["total_reward"].iloc[-1]), 2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Episódios", f"{total_ep} / {config.MAX_EPISODES}")
        m2.metric("Episódios concluídos", f"{completed} ({100*completed//max(1,total_ep)}%)")
        m3.metric("Menor nº de passos (concluído)", best_steps)
        m4.metric("Reward médio (último ep.)", last_reward)

        window = min(50, len(df))
        df["reward_smooth"] = df["total_reward"].rolling(window, min_periods=1).mean()
        df["steps_smooth"] = df["steps"].rolling(window, min_periods=1).mean()
        df["entropy_smooth"] = df["entropy"].rolling(window, min_periods=1).mean()
        df["completion"] = df["completed"].rolling(window, min_periods=1).mean() * 100

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Reward médio por episódio")
            st.line_chart(df.set_index("episode")[["total_reward", "reward_smooth"]])

            st.subheader("Passos por episódio")
            st.line_chart(df.set_index("episode")[["steps", "steps_smooth"]])

        with col_b:
            st.subheader("Taxa de conclusão (%) — janela 50 ep.")
            st.line_chart(df.set_index("episode")[["completion"]])

            st.subheader("Losses (Actor / Critic)")
            st.line_chart(df.set_index("episode")[["actor_loss", "critic_loss"]])

        st.subheader("Entropy ao longo do treinamento")
        st.line_chart(df.set_index("episode")[["entropy", "entropy_smooth"]])

        with st.expander("Ver dados brutos"):
            st.dataframe(df.tail(200), use_container_width=True)

    if auto_refresh:
        time.sleep(5)
        st.cache_data.clear()
        st.rerun()

# ── Tab 2: Live monitor ───────────────────────────────────────────────────────

with tab_live:
    st.subheader("Estado atual do treinamento")
    st.caption("Atualiza a cada 3 segundos enquanto o treinamento estiver rodando.")

    live_refresh = st.button("🔄 Atualizar agora")
    snapshot = load_snapshot()

    if snapshot is None:
        st.info("Nenhum snapshot encontrado. Inicie o treinamento com `python train.py`.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Episódio", snapshot["episode"])
        c2.metric("Passos no ep.", snapshot["steps"])
        c3.metric("Células sujas restantes", snapshot["uncleaned_cells"])
        st.caption(f"Atualizado em: {snapshot['timestamp']}")

        fig = render_grid(
            snapshot["grid_state"],
            snapshot["robot_positions"],
            snapshot["cleaned_cells_per_robot"],
        )
        st.pyplot(fig)
        plt.close(fig)

    if live_refresh:
        st.rerun()

# ── Tab 3: Visual play ────────────────────────────────────────────────────────

with tab_play:
    st.subheader("Assistir os agentes jogando")
    st.caption("Carrega o modelo salvo e executa um episódio completo passo a passo.")

    if not os.path.exists(config.SAVE_PATH):
        st.warning(f"Modelo não encontrado em `{config.SAVE_PATH}`. Treine primeiro.")
    else:
        if st.button("▶️  Iniciar episódio"):
            ph_grid = st.empty()
            ph_info = st.empty()
            run_visual_episode(ph_grid, ph_info)
