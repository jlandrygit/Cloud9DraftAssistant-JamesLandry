"""Streamlit demo UI for the LoL AI drafting assistant."""

from __future__ import annotations

import random
from typing import Any

import streamlit as st

from core.config import (
    CHAMPION_SEARCH_LIMIT,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    DEFAULT_LEAGUE,
    DEFAULT_PATCH_VERSION,
)
from core.draft_logic import ActionType, DraftState, Side
from model.inference import recommend_bans, recommend_picks
from draft.draft_order import DRAFT_SEQUENCE
from draft.draft_state import DraftState as ScoringDraftState
from inference.roster_map import get_team_roster
from inference.scoring_engine import DraftScoringEngine

CHAMPIONS = [
    "Aatrox",
    "Ahri",
    "Akali",
    "Akshan",
    "Alistar",
    "Ambessa",
    "Amumu",
    "Anivia",
    "Annie",
    "Aphelios",
    "Ashe",
    "Aurelion Sol",
    "Aurora",
    "Azir",
    "Bard",
    "Bel'Veth",
    "Blitzcrank",
    "Brand",
    "Braum",
    "Briar",
    "Caitlyn",
    "Camille",
    "Cassiopeia",
    "Cho'Gath",
    "Corki",
    "Darius",
    "Diana",
    "Dr. Mundo",
    "Draven",
    "Ekko",
    "Elise",
    "Evelynn",
    "Ezreal",
    "Fiddlesticks",
    "Fiora",
    "Fizz",
    "Galio",
    "Gangplank",
    "Garen",
    "Gnar",
    "Gragas",
    "Graves",
    "Gwen",
    "Hecarim",
    "Heimerdinger",
    "Hwei",
    "Illaoi",
    "Irelia",
    "Ivern",
    "Janna",
    "Jarvan IV",
    "Jax",
    "Jayce",
    "Jhin",
    "Jinx",
    "Kai'Sa",       
    "Kalista",
    "Karma",
    "Karthus",
    "Kassadin",
    "Katarina",
    "Kayle",
    "Kayn",
    "Kennen",
    "Kha'Zix",
    "Kindred",
    "Kled",
    "Kog'Maw",
    "K'Sante",
    "LeBlanc",
    "Lee Sin",
    "Leona",
    "Lillia",
    "Lissandra",
    "Lucian",
    "Lulu",
    "Lux",
    "Malphite",
    "Malzahar",
    "Maokai",
    "Master Yi",
    "Mel",  
    "Milio",
    "Miss Fortune",
    "Mordekaiser",
    "Morgana",
    "Naafiri",
    "Nami",
    "Nasus",
    "Nautilus",
    "Neeko",
    "Nidalee",
    "Nilah",
    "Nocturne",
    "Nunu & Willump",
    "Olaf",
    "Orianna",
    "Ornn",
    "Pantheon",
    "Poppy",
    "Pyke",
    "Qiyana",
    "Quinn",
    "Rakan",
    "Rammus",
    "Rek'Sai",
    "Rell",
    "Renata Glasc",
    "Renekton",
    "Rengar",
    "Riven",
    "Rumble",
    "Ryze",
    "Samira",
    "Sejuani",
    "Senna",
    "Seraphine",
    "Sett",
    "Shaco",
    "Shen",
    "Shyvana",
    "Singed",   
    "Sion",
    "Sivir",
    "Skarner",
    "Smolder",
    "Sona",
    "Soraka",   
    "Swain",
    "Sylas",
    "Syndra",
    "Tahm Kench",
    "Taliyah",  
    "Talon",
    "Taric",
    "Teemo",
    "Thresh",
    "Tristana",
    "Trundle",
    "Tryndamere",   
    "Twisted Fate",
    "Twitch",
    "Udyr",
    "Urgot",
    "Varus",
    "Vayne",
    "Veigar",
    "Vel'Koz",
    "Vex",
    "Vi",
    "Viego",
    "Viktor",
    "Vladimir",
    "Volibear",
    "Warwick",
    "Wukong",
    "Xayah",
    "Xerath",
    "Xin Zhao",
    "Yasuo",
    "Yone",
    "Yorick",
    "Yunara",
    "Yuumi",
    "Zaahen",
    "Zac",
    "Zed",
    "Zeri",
    "Ziggs",
    "Zilean",
    "Zoe",
    "Zyra",
]

OPPONENTS = ["TLAW", "LYON", "SR", "C9", "SEN", "DSG", "DIG", "FLY", 
"BFX", "BRO", "GEN", "T1", "HLE", "KT", "DK", "DNS", "DRX", "NS", 
"JDG", "WBG", "TES",  "IG", "AL", "BLG", "EDG", "NIP", "WE", "TT", "LNG", "LGD", "OMG", "UP",
"KC", "KCB", "FNC", "G2", "GX", "MKOI", "NAVI", "VIT", "LR", "SHFT", "SK", "TH"]


@st.cache_data(show_spinner=False)
def _fetch_recommendations(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Fetch recommendations from the ML inference layer."""
    return {
        "picks": recommend_picks(payload),
        "bans": recommend_bans(payload),
    }


def _get_champion_score_breakdown(state: DraftState, champion: str) -> dict[str, Any] | None:
    """Get the score breakdown for a specific champion."""
    if not champion:
        return None
    
    if champion not in state.available_champions:
        return None
    
    try:
        payload = _state_to_payload(state)
        scoring_state = _payload_to_scoring_state(payload)
        roster = _payload_to_roster(payload)
        include_tier2 = payload.get("include_tier2", False)
        
        acting_side = scoring_state.acting_side()
        if acting_side == "BLUE":
            our_team_name = str(payload.get("blue_team") or payload.get("our_team") or "")
            opponent_team_name = str(payload.get("red_team") or payload.get("opponent_team") or "")
        else:
            our_team_name = str(payload.get("red_team") or payload.get("opponent_team") or "")
            opponent_team_name = str(payload.get("blue_team") or payload.get("our_team") or "")
        
        scoring_engine = DraftScoringEngine()
        recommendations = scoring_engine.score(
            scoring_state, 
            opponent_roster=roster, 
            opponent_team_name=opponent_team_name if opponent_team_name else None,
            our_team_name=our_team_name if our_team_name else None,
            include_tier2=include_tier2
        )
        
        for rec in recommendations:
            if rec.champion == champion:
                return {
                    "champion": rec.champion,
                    "score": float(rec.total_score),
                    "explanation": rec.explanation_text,
                    "policy_score": float(rec.policy_score),
                    "comfort_score": float(rec.comfort_score),
                    "counter_score": float(rec.counter_score),
                    "synergy_score": float(rec.synergy_score),
                    "meta_score": float(rec.meta_score),
                }
        return None
    except Exception:
        return None


def _payload_to_scoring_state(payload: dict[str, Any]) -> ScoringDraftState:
    """Convert UI/API payloads into the ScoringDraftState model."""
    blue_picks = set(payload.get("blue_picks") or [])
    red_picks = set(payload.get("red_picks") or [])
    blue_bans = set(payload.get("blue_bans") or [])
    red_bans = set(payload.get("red_bans") or [])
    fearless_bans = set(payload.get("fearless_bans") or [])
    bans = blue_bans | red_bans | fearless_bans
    step_index = len(blue_picks) + len(red_picks) + len(blue_bans) + len(red_bans)
    if DRAFT_SEQUENCE:
        step_index = min(step_index, len(DRAFT_SEQUENCE) - 1)
    return ScoringDraftState(
        patch=str(payload.get("patch", payload.get("patch_version", DEFAULT_PATCH_VERSION))),
        league=str(payload.get("league", DEFAULT_LEAGUE)),
        blue_team=str(payload.get("blue_team", "BLUE")),
        red_team=str(payload.get("red_team", "RED")),
        blue_picks=blue_picks,
        red_picks=red_picks,
        bans=bans,
        step_index=step_index,
    )


def _payload_to_roster(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build a minimal roster map for comfort scoring."""
    blue_team = str(payload.get("blue_team") or payload.get("our_team") or "BLUE")
    red_team = str(payload.get("red_team") or payload.get("opponent_team") or "RED")
    blue_roster = get_team_roster(blue_team)
    red_roster = get_team_roster(red_team)
    return {
        "BLUE": blue_roster or {"UNKNOWN": blue_team},
        "RED": red_roster or {"UNKNOWN": red_team},
    }


def _state_to_payload(state: DraftState) -> dict[str, Any]:
    """Convert DraftState to a JSON-serializable payload for inference."""
    our_team = st.session_state.get("our_team", "C9")
    opponent = st.session_state.get("opponent_team", "Opponent")
    fearless_bans = st.session_state.get("fearless_bans", [])
    return {
        "patch_version": st.session_state.get("patch_version", DEFAULT_PATCH_VERSION),
        "league": DEFAULT_LEAGUE,
        "blue_team": our_team,
        "red_team": opponent,
        "our_team": our_team,
        "opponent_team": opponent,
        "inference_version": "final_version_v2",
        "include_tier2": st.session_state.get("include_tier2", False),
        "current_phase": state.current_phase,
        "current_turn": state.current_turn.value,
        "blue_bans": list(state.blue_bans),
        "red_bans": list(state.red_bans),
        "blue_picks": list(state.blue_picks),
        "red_picks": list(state.red_picks),
        "available_champions": list(state.available_champions),
        "fearless_bans": fearless_bans,
        "use_opponent_modeling": True,
    }


def _auto_play_sides(state: DraftState, auto_play_sides: set[Side]) -> None:
    """Apply model recommendations for auto-played sides with weighted random selection.
    
    Selects from top 5 recommendations with probabilities:
    - 40% for option 1 (top recommendation)
    - 25% for option 2
    - 20% for option 3
    - 10% for option 4
    - 5% for option 5
    
    When both sides are auto-playing, continues until draft is complete.
    When only one side is auto-playing, processes that side's turn.
    """
    if state.is_complete or not auto_play_sides:
        return

    # Weighted probabilities for top 5 recommendations
    weights = [0.40, 0.25, 0.20, 0.10, 0.05]

    while not state.is_complete and state.current_turn in auto_play_sides:
        payload = _state_to_payload(state)
        results = _fetch_recommendations(payload)
        if state.current_action is ActionType.BAN:
            recs = results.get("bans", [])
        else:
            recs = results.get("picks", [])
        if not recs:
            return
        
        # Take top 5 recommendations (or fewer if less available)
        top_recs = recs[:5]
        if not top_recs:
            return
        
        # Adjust weights if we have fewer than 5 recommendations
        num_recs = len(top_recs)
        if num_recs < 5:
            # Normalize weights for available options
            available_weights = weights[:num_recs]
            total = sum(available_weights)
            normalized_weights = [w / total for w in available_weights]
        else:
            normalized_weights = weights
        
        # Weighted random selection
        selected_idx = random.choices(range(num_recs), weights=normalized_weights)[0]
        champion = top_recs[selected_idx]["champion"]
        
        auto_side = state.current_turn
        state.apply_action(auto_side, state.current_action, champion)


def _confidence_label(score: float, max_score: float, min_score: float) -> str:
    """Map a score into a human-friendly confidence label based on absolute score.
    
    Args:
        score: The champion's overall score
        max_score: Maximum score in the current recommendation list (unused, kept for compatibility)
        min_score: Minimum score in the current recommendation list (unused, kept for compatibility)
    
    Returns:
        "High", "Medium", or "Low" confidence label based on absolute thresholds
    """
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        return "High"
    elif score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def _rescale_scores(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rescale scores to [0, 1] within the current recommendation list."""
    if not recs:
        return recs
    values = [float(rec.get("score", 0.0)) for rec in recs]
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return recs
    rescaled = []
    for rec in recs:
        score = float(rec.get("score", 0.0))
        display_score = (score - min_score) / (max_score - min_score)
        rec = dict(rec)
        rec["display_score"] = display_score
        rescaled.append(rec)
    return rescaled


def _factor_presence(contributing_factors: list[str]) -> dict[str, bool]:
    """Detect which high-level factors are mentioned by the model."""
    joined = " ".join(contributing_factors).lower()
    return {
        "Champion synergy": "synergy" in joined,
        "Meta win rate": "win rate" in joined or "winrate" in joined,
        "Patch strength": "patch" in joined or "strength" in joined,
        "Role flexibility": "flex" in joined or "role" in joined,
    }


def _resolve_champion_input(query: str, pool: list[str]) -> str:
    """Resolve a user search query into an exact champion name."""
    if not query:
        return ""
    lowered = query.strip().lower()
    for champion in pool:
        if champion.lower() == lowered:
            return champion
    return ""


def _find_matches(query: str, pool: list[str]) -> list[str]:
    """Return a list of partial matches for a champion search query."""
    lowered = query.strip().lower()
    if not lowered:
        return []
    matches = [champion for champion in pool if lowered in champion.lower()]
    return matches[:CHAMPION_SEARCH_LIMIT]


def _format_indexed(items: list[str]) -> list[str]:
    """Return a 1-based indexed list for UI display."""
    return [f"{idx + 1}. {item}" for idx, item in enumerate(items)]


def _init_state() -> None:
    """Initialize Streamlit session state for the draft."""
    if "fearless_bans" not in st.session_state:
        st.session_state.fearless_bans = []
    
    if "draft_state" not in st.session_state:
        fearless_bans = set(st.session_state.get("fearless_bans", []))
        champion_pool = [c for c in CHAMPIONS if c not in fearless_bans]
        st.session_state.draft_state = DraftState(champion_pool=champion_pool)


def _reset_state() -> None:
    """Reset the draft state in session."""
    fearless_bans = set(st.session_state.get("fearless_bans", []))
    champion_pool = [c for c in CHAMPIONS if c not in fearless_bans]
    st.session_state.draft_state = DraftState(champion_pool=champion_pool)


def _save_draft_and_reset() -> None:
    """Save the completed draft's picks to fearless_bans and reset the draft."""
    state: DraftState = st.session_state.draft_state
    if not state.is_complete:
        return
    
    all_picks = list(state.blue_picks) + list(state.red_picks)
    
    fearless_bans = st.session_state.get("fearless_bans", [])
    for pick in all_picks:
        if pick and pick not in fearless_bans:
            fearless_bans.append(pick)
    st.session_state.fearless_bans = fearless_bans
    
    _reset_state()


def main() -> None:
    """Render the Streamlit demo UI."""
    st.set_page_config(page_title="LoL Draft Assistant", layout="wide")
    _init_state()

    state: DraftState = st.session_state.draft_state
    auto_play_blue = st.session_state.get("auto_play_blue_side", False)
    auto_play_red = st.session_state.get("auto_play_red_side", False)
    auto_play_sides = {
        side
        for side, enabled in ((Side.BLUE, auto_play_blue), (Side.RED, auto_play_red))
        if enabled
    }
    previous_step = state.step_index
    if state.current_turn in auto_play_sides:
        _auto_play_sides(state, auto_play_sides)
        if state.step_index != previous_step or state.is_complete:
            st.session_state.draft_state = state
            if state.is_complete:
                st.session_state.auto_play_blue_side = False
                st.session_state.auto_play_red_side = False
            st.rerun()
    if state.current_turn not in auto_play_sides:
        st.session_state["draft_side"] = state.current_turn.value
    
    if state.is_complete:
        st.session_state.auto_play_blue_side = False
        st.session_state.auto_play_red_side = False

    with st.sidebar:
        st.header("Draft Configuration")
        st.selectbox("Blue Side Team", OPPONENTS, index=OPPONENTS.index("C9"), key="our_team")
        st.selectbox("Red Side Team", OPPONENTS, key="opponent_team")
        st.caption(f"Current Patch: {DEFAULT_PATCH_VERSION}")
        st.checkbox("Include Tier 2", key="include_tier2", help="Use model trained on Tier 1 + Tier 2 data")
        st.checkbox("Auto-play Blue Side", key="auto_play_blue_side")
        st.checkbox("Auto-play Red Side", key="auto_play_red_side")
        
        col_reset1, col_reset2 = st.columns(2)
        with col_reset1:
            if st.button("Reset Draft"):
                _reset_state()
                st.rerun()
        with col_reset2:
            if st.button("Reset Series"):
                st.session_state.fearless_bans = []
                _reset_state()
                st.rerun()
        
        fearless_bans = st.session_state.get("fearless_bans", [])
        if fearless_bans:
            st.markdown("**Fearless Bans (Series):**")
            st.write(", ".join(fearless_bans) if fearless_bans else "None")
        
        state: DraftState = st.session_state.draft_state
        if st.button("Save Draft", disabled=not state.is_complete):
            _save_draft_and_reset()
            st.rerun()

    st.title("C9 Draft Assistant")
    st.subheader("Real-time pick/ban guidance for League of Legends drafts")
    st.info("Select opponent → Walk through draft → See AI recommendations in real time")
    mode_label = "Opponent-specific mode (recommendations adapt to selected teams)"
    st.caption(f"Active mode: {mode_label}")
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.subheader("Draft Phase")
        st.write(f"Current phase: **{state.current_phase}**")
        st.write(f"Current turn: **{state.current_turn.value}**")

        bans_col, picks_col = st.columns(2)
        with bans_col:
            st.markdown("**Blue Bans**")
            st.write(_format_indexed(state.blue_bans) or ["—"])
            st.markdown("**Red Bans**")
            st.write(_format_indexed(state.red_bans) or ["—"])
        with picks_col:
            st.markdown("**Blue Picks**")
            st.write(_format_indexed(state.blue_picks) or ["—"])
            st.markdown("**Red Picks**")
            st.write(_format_indexed(state.red_picks) or ["—"])

        st.subheader("Champion Selection")
        champion_pool = state.available_champions or CHAMPIONS
        if st.session_state.get("pending_champion"):
            st.session_state["selected_champion_search"] = st.session_state.pop(
                "pending_champion"
            )
        
        search_col, score_col = st.columns([2, 1], gap="medium")
        
        with search_col:
            query = st.text_input("Search champion", key="selected_champion_search")
            selected = _resolve_champion_input(query, champion_pool)
            if query and not selected:
                matches = _find_matches(query, champion_pool)
                if matches:
                    selected = st.selectbox("Matching champions", matches, key="selected_match")
                else:
                    st.caption("No matching champions found.")
        
        with score_col:
            if selected:
                breakdown = _get_champion_score_breakdown(state, selected)
                if breakdown:
                    st.markdown(f"**{selected}**")
                    st.write(f"**Overall Score:** {breakdown['score']:.4f}")
                    st.progress(min(max(breakdown['score'], 0.0), 1.0))
                    with st.expander("Score Breakdown"):
                        st.write(f"**Policy Score:** {breakdown['policy_score']:.4f}")
                        st.write(f"**Comfort Score:** {breakdown['comfort_score']:.4f}")
                        st.write(f"**Counter Score:** {breakdown['counter_score']:.4f}")
                        st.write(f"**Synergy Score:** {breakdown['synergy_score']:.4f}")
                        st.write(f"**Meta Score:** {breakdown['meta_score']:.4f}")
        
        required_action = state.current_action
        st.caption(f"Required action: {required_action.value}")

        acting_side = Side(st.session_state.draft_side)
        is_valid, reason = state.can_take_action(acting_side, required_action, selected)
        if st.button(f"Confirm {required_action.value}", disabled=not is_valid):
            try:
                state.apply_action(acting_side, required_action, selected)
                _auto_play_sides(state, auto_play_sides)
                st.session_state.draft_state = state
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if not is_valid:
            st.error(reason)

    with col_left:
        st.divider()

    with col_right:
        try:
            payload = _state_to_payload(state)
            results = _fetch_recommendations(payload)
            pick_recs = results.get("picks", [])
            ban_recs = results.get("bans", [])
        except Exception as exc:
            st.error(f"Model inference failed: {exc}")
            pick_recs = []
            ban_recs = []

        current_side = state.current_turn.value
        if state.current_action is ActionType.BAN:
            st.subheader(f"AI Recommended Bans — {current_side} side")
            active_recs = ban_recs
        else:
            st.subheader(f"AI Recommended Picks — {current_side} side")
            active_recs = pick_recs

        if not active_recs:
            st.caption("No recommendations available for the current phase.")

        display_recs = active_recs[:5]
        if display_recs:
            scores = [float(rec.get("score", 0.0)) for rec in display_recs]
            max_score = max(scores)
            min_score = min(scores)
        else:
            max_score = 1.0
            min_score = 0.0
        
        action_label = " "
        for idx, rec in enumerate(display_recs, start=1):
            champion = rec["champion"]
            score = float(rec["score"])
            confidence = _confidence_label(score, max_score, min_score)
            row_cols = st.columns([1, 8], vertical_alignment="center")
            with row_cols[0]:
                if st.button(
                    action_label,
                    key=f"take_suggestion_{state.step_index}_{champion}",
                    use_container_width=True,
                ):
                    st.session_state["pending_champion"] = champion
                    st.rerun()
            with row_cols[1]:
                expander_label = f"{champion} — {confidence} confidence"
                with st.expander(expander_label):
                    st.caption(f"Mode: {mode_label}")
                    st.write(rec.get("explanation", ""))
                    st.write(f"Overall score: {score:.4f}")
                    st.progress(min(max(score, 0.0), 1.0))
                    st.markdown("**Score Inputs**")
                    st.write(
                        f"- Policy score: {float(rec.get('policy_score', 0.0)):.4f}"
                    )
                    st.write(
                        f"- Comfort score: {float(rec.get('comfort_score', 0.0)):.4f}"
                    )
                    st.write(
                        f"- Counter score: {float(rec.get('counter_score', 0.0)):.4f}"
                    )
                    st.write(
                        f"- Synergy score: {float(rec.get('synergy_score', 0.0)):.4f}"
                    )
                    st.write(f"- Meta score: {float(rec.get('meta_score', 0.0)):.4f}")
                    st.markdown("**Final Calculation**")
                    st.write(
                        "Overall Score = 60% Pro Draft Model, 13% Synergy, "
                        "12% Meta, 5% Comfort, 5% Counter"
                    )


if __name__ == "__main__":
    main()
