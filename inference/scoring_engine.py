"""Draft scoring engine that combines multiple signals."""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from os import getenv
import json
from pathlib import Path

from core.config import (
    ARCHETYPE_BONUS_BASE,
    ARCHETYPE_BONUS_INCREMENT,
    ARCHETYPE_BONUS_MAX,
    BAN_RATE_COMFORT_THRESHOLD,
    COMFORT_GAMES_THRESHOLD,
    COMFORT_WEIGHT,
    COUNTER_WEIGHT,
    FACTOR_SCORE_THRESHOLD,
    MATCH_CONFIDENCE_THRESHOLD,
    META_WEIGHT,
    POLICY_HIGH_THRESHOLD,
    POLICY_WEIGHT,
    ROLE_DENIAL_BONUS_BASE,
    ROLE_DENIAL_BONUS_INCREMENT,
    ROLE_DENIAL_BONUS_MAX,
    ROLE_DENIAL_BONUS_ONE_GONE,
    ROLE_DENIAL_BONUS_TWO_GONE,
    SYNERGY_WEIGHT,
    WINRATE_BASELINE,
    WINRATE_SCALING_FACTOR,
)
from draft.draft_state import DraftState
from data.meta.load_u_gg_roles import load_roles
from inference.comfort_adapter import get_comfort_scores
from inference.counter_adapter import get_counter_scores
from inference.draft_recommender import get_policy_scores
from inference.explanations import RecommendationExplanation
from inference.meta_adapter import get_meta_scores
from inference.role_filter import filter_by_filled_roles, ALL_ROLES
from inference.synergy_adapter import get_synergy_scores


@dataclass
class DraftScoringEngine:
    """Combine policy, comfort, counter, synergy, and meta scores.

    The expert draft policy model is the primary decision driver.
    """

    policy_weight: float = POLICY_WEIGHT
    comfort_weight: float = COMFORT_WEIGHT
    counter_weight: float = COUNTER_WEIGHT
    synergy_weight: float = SYNERGY_WEIGHT
    meta_weight: float = META_WEIGHT



    def score(
        self, 
        state: DraftState, 
        opponent_roster: dict | None = None, 
        opponent_team_name: str | None = None,
        our_team_name: str | None = None,
        include_tier2: bool = False
    ) -> list[RecommendationExplanation]:
        """Score available champions and return ranked explanations.
        
        Args:
            state: Current draft state
            opponent_roster: Roster mapping for comfort scoring
            opponent_team_name: Name of the opponent team (for ban-based comfort during ban phase)
            our_team_name: Name of our team (for ban-based comfort during pick phase)
            include_tier2: Whether to use Tier 2 model
        """
        policy = get_policy_scores(state, include_tier2=include_tier2)
        if not policy:
            raise ValueError("Policy scores are required but missing.")
        comfort = get_comfort_scores(
            state, 
            opponent_roster=opponent_roster or {},
            opponent_team_name=opponent_team_name,
            our_team_name=our_team_name
        )
        counter = get_counter_scores(state)
        synergy = get_synergy_scores(state)
        meta = get_meta_scores(state)

        candidate_set = set(policy.keys())
        filtered = filter_by_filled_roles(state, candidate_set)
        champions = list(filtered)
        scored: dict[str, RecommendationExplanation] = {}
        for champion in champions:
            policy_score = policy.get(champion)
            if policy_score is None:
                raise ValueError(f"Missing policy score for champion: {champion}")
            if not 0.0 <= policy_score <= 1.0:
                raise ValueError(
                    f"Policy score out of bounds for {champion}: {policy_score}"
                )
            comfort_score = comfort.get(champion, 0.0)
            synergy_score = synergy.get(champion, 0.0)
            counter_score = counter.get(champion, 0.0)
            meta_score = meta.get(champion, 0.0)
            
            archetype_synergy_bonus, archetype_counter_bonus = _get_archetype_bonuses(state, champion)
            synergy_score += archetype_synergy_bonus
            counter_score += archetype_counter_bonus
            
            # Clamp counter and synergy scores to [0, 1] after adding bonuses
            counter_score = max(0.0, min(1.0, counter_score))
            synergy_score = max(0.0, min(1.0, synergy_score))

            total = (
                self.policy_weight * policy_score
                + self.comfort_weight * comfort_score
                + self.counter_weight * counter_score
                + self.synergy_weight * synergy_score
                + self.meta_weight * meta_score
            )
            
            role_denial_bonus = _get_role_denial_bonus(state, champion)
            if role_denial_bonus > 0:
                total += role_denial_bonus
            
            policy_pct, modifier_pct = _contribution_percentages(policy_score, total)

            explanation_text = _build_explanation(
                champion,
                state,
                policy_score,
                comfort_score,
                counter_score,
                synergy_score,
                meta_score,
                self.policy_weight,
                self.comfort_weight,
                self.counter_weight,
                self.synergy_weight,
                self.meta_weight,
                opponent_roster=opponent_roster or {},
                opponent_team_name=opponent_team_name,
                our_team_name=our_team_name,
            )

            scored[champion] = RecommendationExplanation(
                champion=champion,
                total_score=total,
                policy_score=policy_score,
                comfort_score=comfort_score,
                counter_score=counter_score,
                synergy_score=synergy_score,
                meta_score=meta_score,
                policy_contribution_pct=policy_pct,
                modifier_contribution_pct=modifier_pct,
                explanation_text=explanation_text,
            )

        return sorted(
            scored.values(),
            key=lambda item: (-item.total_score, -item.policy_score, item.champion),
        )


def _build_explanation(
    champion: str,
    state: DraftState,
    policy_score: float,
    comfort_score: float,
    counter_score: float,
    synergy_score: float,
    meta_score: float,
    policy_weight: float,
    comfort_weight: float,
    counter_weight: float,
    synergy_weight: float,
    meta_weight: float,
    opponent_roster: dict | None = None,
    opponent_team_name: str | None = None,
    our_team_name: str | None = None,
) -> str:
    """Generate a contextual explanation based on top contributing factors.
    
    Calculates weighted contributions (score * weight) to find the top 2 non-policy factors,
    then generates contextual messages based on those factors.
    
    Uses policy message when policy_score >= POLICY_HIGH_THRESHOLD.
    Only includes other factors when their raw score >= FACTOR_SCORE_THRESHOLD.
    """
    if policy_score >= POLICY_HIGH_THRESHOLD:
        factors_with_scores = {
            "counter": counter_score,
            "synergy": synergy_score,
            "meta": meta_score,
            "comfort": comfort_score,
        }
        top_factor = max(factors_with_scores.items(), key=lambda x: x[1], default=None)
        return _generate_policy_message(
            champion, 
            state, 
            top_factor,
            opponent_roster=opponent_roster or {},
            opponent_team_name=opponent_team_name,
            our_team_name=our_team_name
        )
    
    factors_with_scores = {
        "counter": counter_score,
        "synergy": synergy_score,
        "meta": meta_score,
        "comfort": comfort_score,
    }
    
    valid_factors = {
        factor: score 
        for factor, score in factors_with_scores.items() 
        if score >= FACTOR_SCORE_THRESHOLD
    }
    
    if not valid_factors:
        return _generate_policy_message(
            champion, 
            state, 
            None,
            opponent_roster=opponent_roster or {},
            opponent_team_name=opponent_team_name,
            our_team_name=our_team_name
        )
    
    contributions = {
        factor: score * {
            "counter": counter_weight,
            "synergy": synergy_weight,
            "meta": meta_weight,
            "comfort": comfort_weight,
        }[factor]
        for factor, score in valid_factors.items()
    }
    
    top_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:2]
    
    if not top_factors:
        return _generate_policy_message(
            champion, 
            state, 
            None,
            opponent_roster=opponent_roster or {},
            opponent_team_name=opponent_team_name,
            our_team_name=our_team_name
        )
    
    messages = []
    for factor, _ in top_factors:
        if factor == "counter":
            msg = _generate_counter_message(champion, state)
        elif factor == "synergy":
            msg = _generate_synergy_message(champion, state)
        elif factor == "meta":
            msg = _generate_meta_message(champion, state)
        elif factor == "comfort":
            msg = _generate_comfort_message(
                champion, 
                state,
                opponent_roster=opponent_roster or {},
                opponent_team_name=opponent_team_name,
                our_team_name=our_team_name
            )
        else:
            continue
        
        if msg:
            messages.append(msg)
    
    if messages:
        return " ".join(messages)
    
    return _generate_policy_message(
        champion, 
        state, 
        None,
        opponent_roster=opponent_roster or {},
        opponent_team_name=opponent_team_name,
        our_team_name=our_team_name
    )


def _generate_counter_message(champion: str, state: DraftState) -> str:
    """Generate a counter message for picks or bans."""
    from inference.counter_map import load_counter_map
    from inference.role_filter import get_open_roles
    
    counter_map = load_counter_map()
    roles_map = _load_roles_map()
    if not counter_map or not roles_map:
        return ""
    
    if state.is_pick_phase():
        enemy_picks = state.red_picks if state.acting_side() == "BLUE" else state.blue_picks
        if not enemy_picks:
            return ""
        
        best_target = None
        best_score = 0.0
        candidate_roles = roles_map.get(champion, set())
        open_roles = get_open_roles(state, roles_map, state.acting_side())
        if open_roles:
            candidate_roles = candidate_roles & open_roles
        
        for enemy_champ in enemy_picks:
            enemy_roles = roles_map.get(enemy_champ, set())
            for role in candidate_roles & enemy_roles:
                counter_data = counter_map.get(enemy_champ, {}).get(role, {}).get(champion)
                if counter_data:
                    winrate, matches = counter_data
                    score = (winrate - WINRATE_BASELINE) / WINRATE_SCALING_FACTOR * min(1.0, matches / MATCH_CONFIDENCE_THRESHOLD)
                    if score > best_score:
                        best_score = score
                        best_target = enemy_champ
        
        if best_target:
            return f"{champion} is a strong counter to {best_target}."
    else:
        our_picks = state.blue_picks if state.acting_side() == "BLUE" else state.red_picks
        if our_picks:
            for our_champ in our_picks:
                our_roles = roles_map.get(our_champ, set())
                champ_roles = roles_map.get(champion, set())
                for role in our_roles & champ_roles:
                    counter_data = counter_map.get(our_champ, {}).get(role, {}).get(champion)
                    if counter_data:
                        return f"{champion} counters {our_champ}, denying a strong counter pick."
    return ""


def _generate_synergy_message(champion: str, state: DraftState) -> str:
    """Generate a synergy message for picks or bans."""
    from inference.duos_map import load_duos_map
    from inference.role_filter import get_open_roles
    
    duos_map = load_duos_map()
    roles_map = _load_roles_map()
    if not duos_map or not roles_map:
        return ""
    
    if state.is_pick_phase():
        our_picks = state.blue_picks if state.acting_side() == "BLUE" else state.red_picks
        if not our_picks:
            return ""
        
        best_partner = None
        best_role = None
        best_score = 0.0
        candidate_roles = roles_map.get(champion, set())
        open_roles = get_open_roles(state, roles_map, state.acting_side())
        if open_roles:
            candidate_roles = candidate_roles & open_roles
        
        for partner in our_picks:
            partner_roles = roles_map.get(partner, set())
            for partner_role in partner_roles:
                duo_data = duos_map.get(partner, {}).get(partner_role, {}).get(champion)
                if duo_data:
                    winrate, duo_role, matches = duo_data
                    if duo_role in candidate_roles:
                        score = ((winrate - WINRATE_BASELINE) / WINRATE_SCALING_FACTOR) * min(1.0, matches / MATCH_CONFIDENCE_THRESHOLD)
                        if score > best_score:
                            best_score = score
                            best_partner = partner
                            best_role = partner_role
        
        if best_partner:
            role_labels = {
                "ADC": "ADC",
                "BOTTOM": "ADC",
                "MID": "Mid",
                "JUNGLE": "Jungle",
                "TOP": "Top",
                "SUPPORT": "Support",
            }
            role_label = role_labels.get(best_role, "")
            
            if role_label:
                return f"{champion} is a powerful duo with {best_partner} {role_label}."
            else:
                return f"{champion} is a powerful duo with {best_partner}."
    else:
        enemy_picks = state.red_picks if state.acting_side() == "BLUE" else state.blue_picks
        if enemy_picks:
            for enemy_champ in enemy_picks:
                enemy_roles = roles_map.get(enemy_champ, set())
                for enemy_role in enemy_roles:
                    duo_data = duos_map.get(enemy_champ, {}).get(enemy_role, {}).get(champion)
                    if duo_data:
                        return f"{champion} synergizes with {enemy_champ}, denying a strong combo."
    return ""


def _generate_meta_message(champion: str, state: DraftState) -> str:
    """Generate a meta message for picks or bans."""
    if state.is_pick_phase():
        return f"{champion} is very strong this patch."
    else:
        return f"{champion} is very strong this patch, denying a meta pick."


def _get_best_player_for_champion(
    champion: str,
    state: DraftState,
    opponent_roster: dict
) -> str | None:
    """Find the player with the highest comfort for a champion.
    
    Returns the player name if found, None otherwise.
    """
    from inference.comfort_adapter import _load_player_champion_stats, _load_role_map
    from inference.role_filter import get_open_roles
    
    acting = state.acting_side()
    target_side = acting if state.is_pick_phase() else ("RED" if acting == "BLUE" else "BLUE")
    roster = opponent_roster.get(target_side, {})
    if not isinstance(roster, dict) or not roster:
        return None
    
    role_map = _load_role_map()
    stats_map = _load_player_champion_stats()
    open_roles = get_open_roles(state, role_map, target_side)
    
    roles = role_map.get(champion, set())
    if not roles:
        return None
    candidate_roles = roles & open_roles if open_roles else roles
    if not candidate_roles:
        return None
    
    best_player = None
    best_score = 0.0
    
    for role in candidate_roles:
        player = roster.get(role)
        if not player or not str(player).strip():
            continue
        stats = stats_map.get((str(player).strip(), str(role).upper(), champion))
        if stats:
            games, _ = stats
            score = games / COMFORT_GAMES_THRESHOLD
            if score > best_score:
                best_score = score
                best_player = str(player).strip()
    
    return best_player if best_score > 0.0 else None


def _generate_comfort_message(
    champion: str, 
    state: DraftState,
    opponent_roster: dict | None = None,
    opponent_team_name: str | None = None,
    our_team_name: str | None = None
) -> str:
    """Generate a comfort message for picks or bans.
    
    If ban-based comfort is strong, shows a specific message about that.
    - Pick phase: Shows message if champion is frequently banned against OUR team
    - Ban phase: Shows message if champion is frequently banned against OPPONENT team
    
    Otherwise, includes the player name if available.
    """
    from inference.comfort_adapter import _load_team_ban_stats
    ban_stats_map = _load_team_ban_stats()
    
    if state.is_pick_phase():
        if our_team_name:
            ban_rate = ban_stats_map.get((our_team_name, champion), 0.0)
            if ban_rate >= BAN_RATE_COMFORT_THRESHOLD:
                return f"Most teams ban {champion} against {our_team_name}."
        
        if opponent_roster:
            best_player = _get_best_player_for_champion(champion, state, opponent_roster)
            if best_player:
                return f"{champion} is a comfort pick for {best_player}."
        return f"{champion} is a strong comfort pick for your team."
    else:
        if opponent_team_name:
            ban_rate = ban_stats_map.get((opponent_team_name, champion), 0.0)
            if ban_rate >= BAN_RATE_COMFORT_THRESHOLD:
                return f"Most teams ban {champion} against {opponent_team_name}."
        
        if opponent_roster:
            best_player = _get_best_player_for_champion(champion, state, opponent_roster)
            if best_player:
                return f"{champion} is a comfort pick for {best_player}, denying his preferred champion."
        return f"{champion} is a strong comfort pick for the opponent, denying their preferred champion."


def _expand_step_label(step_label: str) -> str:
    """Expand step label abbreviations to full names.
    
    Examples:
        BB1 -> Blue Side Ban 1
        RB1 -> Red Side Ban 1
        BP1 -> Blue Side Pick 1
        RP1 -> Red Side Pick 1
    """
    if not step_label or len(step_label) < 3:
        return step_label
    
    side_letter = step_label[0]
    action_letter = step_label[1]
    number = step_label[2:]
    
    side_name = "Blue Side" if side_letter == "B" else "Red Side"
    action_name = "Ban" if action_letter == "B" else "Pick"
    
    return f"{side_name} {action_name} {number}"


def _generate_policy_message(
    champion: str, 
    state: DraftState, 
    top_factor: tuple[str, float] | None,
    opponent_roster: dict | None = None,
    opponent_team_name: str | None = None,
    our_team_name: str | None = None
) -> str:
    """Generate a policy-based message for picks or bans.
    
    Args:
        champion: The champion name
        state: Current draft state
        top_factor: Tuple of (factor_name, score) for the highest non-policy factor, or None
        opponent_team_name: Name of opponent team (for comfort messages)
        our_team_name: Name of our team (for comfort messages)
    """
    from draft.draft_order import DRAFT_SEQUENCE
    step_label_expanded = None
    if state.step_index < len(DRAFT_SEQUENCE):
        step = DRAFT_SEQUENCE[state.step_index]
        step_label = step[3]
        step_label_expanded = _expand_step_label(step_label)
    
    if step_label_expanded:
        if state.is_pick_phase():
            base_msg = f"{champion} is a common {step_label_expanded} pick among pros."
        else:
            base_msg = f"{champion} is commonly banned {step_label_expanded} among pros."
    else:
        if state.is_pick_phase():
            base_msg = f"{champion} is a strong pick based on professional draft patterns."
        else:
            base_msg = f"{champion} is a strong ban based on professional draft patterns."
    
    if top_factor and top_factor[1] >= FACTOR_SCORE_THRESHOLD:
        factor_name, factor_score = top_factor
        
        detailed_msg = ""
        if factor_name == "counter":
            detailed_msg = _generate_counter_message(champion, state)
        elif factor_name == "synergy":
            detailed_msg = _generate_synergy_message(champion, state)
        elif factor_name == "comfort":
            detailed_msg = _generate_comfort_message(
                champion, 
                state,
                opponent_roster=opponent_roster or {},
                opponent_team_name=opponent_team_name,
                our_team_name=our_team_name
            )
        elif factor_name == "meta":
            detailed_msg = _generate_meta_message(champion, state)
        
        if detailed_msg:
            return f"{base_msg} {detailed_msg}"
    
    return base_msg


def _contribution_percentages(policy_score: float, total_score: float) -> tuple[float, float]:
    """Compute contribution percentages with policy dominance."""
    if total_score <= 0.0:
        return 70.0, 30.0
    raw_policy_pct = (policy_score / total_score) * 100.0
    policy_pct = max(70.0, min(100.0, raw_policy_pct))
    modifier_pct = 100.0 - policy_pct
    return policy_pct, modifier_pct


def _is_second_ban_phase(state: DraftState) -> bool:
    """Check if we're in the second ban phase (bans 4 and 5, steps 13-16)."""
    return state.is_ban_phase() and 12 <= state.step_index <= 15


@lru_cache(maxsize=1)
def _load_roles_map() -> dict[str, set[str]]:
    """Load champion roles from disk once per process."""
    path = getenv("UGG_ROLES_PATH", "data/meta/u_gg_roles.csv")
    try:
        return load_roles(path)
    except Exception:
        return {}


def _get_filled_roles(state: DraftState, side: str) -> set[str]:
    """Get the set of roles that have been filled by the specified side.
    
    Uses role mapping to determine which roles each picked champion can play.
    Returns a set of role strings (e.g., {"TOP", "JUNGLE", "MID"}).
    """
    roles_map = _load_roles_map()
    if not roles_map:
        return set()
    
    picks = state.blue_picks if side == "BLUE" else state.red_picks
    filled_roles: set[str] = set()
    
    for champion in picks:
        champ_roles = roles_map.get(champion)
        if champ_roles:
            filled_roles.update(champ_roles)
        else:
            filled_roles.update(ALL_ROLES)
    
    return filled_roles


def _get_champion_roles(champion: str) -> set[str]:
    """Get the set of roles a champion can play."""
    roles_map = _load_roles_map()
    return roles_map.get(champion, set())


def _get_role_denial_bonus(state: DraftState, champion: str) -> float:
    """Calculate bonus for banning champions in roles we've filled but opponent hasn't.
    
    This bonus applies during the second ban phase (bans 4 and 5) to incentivize
    "pinching" the champion pool - banning multiple champions in the same role to
    deny the opponent's best options.
    
    The bonus scales with how many champions in that role have already been
    picked/banned (more champions gone = higher bonus for banning another).
    
    Returns:
        Bonus score to add to the total (typically 0.0-0.2 range).
    """
    if not _is_second_ban_phase(state):
        return 0.0
    
    acting_side = state.acting_side()
    our_side = acting_side
    opponent_side = "RED" if our_side == "BLUE" else "BLUE"
    
    # Get roles we've filled vs roles opponent has filled
    our_filled_roles = _get_filled_roles(state, our_side)
    opponent_filled_roles = _get_filled_roles(state, opponent_side)
    
    roles_to_deny = our_filled_roles - opponent_filled_roles
    
    if not roles_to_deny:
        return 0.0
    
    champion_roles = _get_champion_roles(champion)
    if not champion_roles:
        return 0.0
    
    matching_roles = champion_roles & roles_to_deny
    if not matching_roles:
        return 0.0
    
    max_bonus = 0.0
    roles_map = _load_roles_map()
    all_picked_banned = state.blue_picks | state.red_picks | state.bans
    
    for role in matching_roles:
        champions_in_role = {
            champ for champ, roles in roles_map.items() 
            if role in roles
        }
        
        champions_gone = len([champ for champ in champions_in_role if champ in all_picked_banned])
        
        if champions_gone == 0:
            bonus = ROLE_DENIAL_BONUS_BASE
        elif champions_gone == 1:
            bonus = ROLE_DENIAL_BONUS_ONE_GONE
        elif champions_gone == 2:
            bonus = ROLE_DENIAL_BONUS_TWO_GONE
        else:
            bonus = min(ROLE_DENIAL_BONUS_MAX, ROLE_DENIAL_BONUS_TWO_GONE + (champions_gone - 2) * ROLE_DENIAL_BONUS_INCREMENT)
        
        max_bonus = max(max_bonus, bonus)
    
    return max_bonus


@lru_cache(maxsize=1)
def _load_archetypes() -> dict[str, dict[str, bool]]:
    """Load champion archetypes from disk once per process."""
    archetypes_path = Path(__file__).parent.parent / "data" / "archetypes.json"
    try:
        with open(archetypes_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_majority_archetype(champions: set[str]) -> str | None:
    """Calculate the majority archetype from a set of champions.
    
    Returns the archetype that appears most frequently (Poke, Dive, or Counter-Engage).
    Returns None if no champions or no archetype data available.
    """
    archetypes = _load_archetypes()
    if not archetypes or not champions:
        return None
    
    archetype_counts = {"Poke": 0, "Dive": 0, "Counter-Engage": 0}
    
    for champion in champions:
        champ_archetypes = archetypes.get(champion, {})
        for archetype in ["Poke", "Dive", "Counter-Engage"]:
            if champ_archetypes.get(archetype, False):
                archetype_counts[archetype] += 1
    
    # Find the archetype with the highest count
    max_count = max(archetype_counts.values())
    if max_count == 0:
        return None
    
    # Return the first archetype that has the max count
    for archetype, count in archetype_counts.items():
        if count == max_count:
            return archetype
    
    return None


def _get_counter_archetype(archetype: str) -> str | None:
    """Get the archetype that counters the given archetype.
    
    Flow: Poke beats Counter-Engage, Counter-Engage beats Dive, Dive beats Poke
    So:
    - If opponent has Poke, we want Dive (Dive beats Poke)
    - If opponent has Counter-Engage, we want Poke (Poke beats Counter-Engage)
    - If opponent has Dive, we want Counter-Engage (Counter-Engage beats Dive)
    """
    counter_map = {
        "Poke": "Dive",
        "Counter-Engage": "Poke",
        "Dive": "Counter-Engage",
    }
    return counter_map.get(archetype)


def _champion_has_archetype(champion: str, archetype: str) -> bool:
    """Check if a champion has a specific archetype."""
    archetypes = _load_archetypes()
    champ_archetypes = archetypes.get(champion, {})
    return champ_archetypes.get(archetype, False)


def _get_archetype_bonuses(state: DraftState, champion: str) -> tuple[float, float]:
    """Calculate archetype-based synergy and counter bonuses.
    
    Synergy bonus: If champion matches our team's majority archetype
    Counter bonus: If champion counters opponent's majority archetype
    
    Returns:
        Tuple of (synergy_bonus, counter_bonus) in range [0.0, 0.15]
    """
    acting_side = state.acting_side()
    our_side = acting_side
    opponent_side = "RED" if our_side == "BLUE" else "BLUE"
    
    our_picks = state.blue_picks if our_side == "BLUE" else state.red_picks
    opponent_picks = state.red_picks if our_side == "BLUE" else state.blue_picks
    
    our_archetype = _get_majority_archetype(our_picks)
    opponent_archetype = _get_majority_archetype(opponent_picks)
    
    synergy_bonus = 0.0
    counter_bonus = 0.0
    
    if our_archetype and our_picks:
        if _champion_has_archetype(champion, our_archetype):
            matching_count = sum(
                1 for champ in our_picks 
                if _champion_has_archetype(champ, our_archetype)
            )
            synergy_bonus = min(ARCHETYPE_BONUS_MAX, ARCHETYPE_BONUS_BASE + (matching_count - 1) * ARCHETYPE_BONUS_INCREMENT)
    
    if opponent_archetype and opponent_picks:
        counter_archetype = _get_counter_archetype(opponent_archetype)
        if counter_archetype and _champion_has_archetype(champion, counter_archetype):
            opponent_matching_count = sum(
                1 for champ in opponent_picks 
                if _champion_has_archetype(champ, opponent_archetype)
            )
            counter_bonus = min(ARCHETYPE_BONUS_MAX, ARCHETYPE_BONUS_BASE + (opponent_matching_count - 1) * ARCHETYPE_BONUS_INCREMENT)
    
    return synergy_bonus, counter_bonus
