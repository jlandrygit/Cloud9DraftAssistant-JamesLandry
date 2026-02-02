"""Draft scoring engine that combines multiple signals."""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from os import getenv
import json
from pathlib import Path

from core.scoring_draft_state import DraftState
from data.meta.load_u_gg_roles import load_roles
from ml.inference.comfort_adapter import get_comfort_scores
from ml.inference.counter_adapter import get_counter_scores
from ml.inference.draft_recommender import get_policy_scores
from ml.inference.explanations import RecommendationExplanation
from ml.inference.meta_adapter import get_meta_scores
from ml.inference.role_filter import filter_by_filled_roles, ALL_ROLES
from ml.inference.synergy_adapter import get_synergy_scores


@dataclass
class DraftScoringEngine:
    """Combine policy, comfort, counter, synergy, and meta scores.

    The expert draft policy model is the primary decision driver.
    """

    policy_weight: float = .60
    comfort_weight: float = 0.05
    counter_weight: float = 0.05
    synergy_weight: float = 0.13
    meta_weight: float = 0.12



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
            
            # Apply archetype-based bonuses
            archetype_synergy_bonus, archetype_counter_bonus = _get_archetype_bonuses(state, champion)
            synergy_score += archetype_synergy_bonus
            counter_score += archetype_counter_bonus

            # Scoring math:
            # total = weighted sum of signals (no clamp).
            total = (
                self.policy_weight * policy_score
                + self.comfort_weight * comfort_score
                + self.counter_weight * counter_score
                + self.synergy_weight * synergy_score
                + self.meta_weight * meta_score
            )
            
            # Apply second ban phase role denial bonus
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
    
    Uses policy message when policy_score >= 0.9.
    Only includes other factors when their raw score >= 0.5.
    """
    # If policy score is very high, use policy message with top non-policy factor
    if policy_score >= 0.9:
        # Find the highest non-policy factor to include in the message
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
    
    # Filter factors by score threshold (0.5)
    SCORE_THRESHOLD = 0.5
    factors_with_scores = {
        "counter": counter_score,
        "synergy": synergy_score,
        "meta": meta_score,
        "comfort": comfort_score,
    }
    
    # Only consider factors that meet the threshold
    valid_factors = {
        factor: score 
        for factor, score in factors_with_scores.items() 
        if score >= SCORE_THRESHOLD
    }
    
    if not valid_factors:
        # Fallback to policy message if no factors meet threshold
        return _generate_policy_message(
            champion, 
            state, 
            None,
            opponent_roster=opponent_roster or {},
            opponent_team_name=opponent_team_name,
            our_team_name=our_team_name
        )
    
    # Calculate weighted contributions for valid factors only
    contributions = {
        factor: score * {
            "counter": counter_weight,
            "synergy": synergy_weight,
            "meta": meta_weight,
            "comfort": comfort_weight,
        }[factor]
        for factor, score in valid_factors.items()
    }
    
    # Find top 2 contributing factors
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
    
    # Generate messages for top factors
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
    
    # If we have messages, return them. Otherwise fall back to policy message.
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
    from ml.inference.counter_map import load_counter_map
    from ml.inference.role_filter import get_open_roles
    
    counter_map = load_counter_map()
    roles_map = _load_roles_map()
    if not counter_map or not roles_map:
        return ""
    
    if state.is_pick_phase():
        # Pick phase: counter enemy picks
        enemy_picks = state.red_picks if state.acting_side() == "BLUE" else state.blue_picks
        if not enemy_picks:
            return ""
        
        # Find the best counter target
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
                    score = (winrate - 45.0) / 10.0 * min(1.0, matches / 250.0)
                    if score > best_score:
                        best_score = score
                        best_target = enemy_champ
        
        if best_target:
            return f"{champion} is a strong counter to {best_target}."
    else:
        # Ban phase: counter our picks (deny counter to us)
        our_picks = state.blue_picks if state.acting_side() == "BLUE" else state.red_picks
        if our_picks:
            # Find if this counters any of our picks
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
    from ml.inference.duos_map import load_duos_map
    from ml.inference.role_filter import get_open_roles
    
    duos_map = load_duos_map()
    roles_map = _load_roles_map()
    if not duos_map or not roles_map:
        return ""
    
    if state.is_pick_phase():
        # Pick phase: synergy with our picks
        our_picks = state.blue_picks if state.acting_side() == "BLUE" else state.red_picks
        if not our_picks:
            return ""
        
        # Find the best synergy partner
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
                        score = ((winrate - 45.0) / 10.0) * min(1.0, matches / 250.0)
                        if score > best_score:
                            best_score = score
                            best_partner = partner
                            best_role = partner_role
        
        if best_partner:
            # Map role to readable label
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
        # Ban phase: deny synergy with enemy picks
        enemy_picks = state.red_picks if state.acting_side() == "BLUE" else state.blue_picks
        if enemy_picks:
            # Check if this synergizes with any enemy pick
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
    from ml.inference.comfort_adapter import _load_player_champion_stats, _load_role_map
    from ml.inference.role_filter import get_open_roles
    
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
    
    # Find the player with the highest comfort
    best_player = None
    best_score = 0.0
    
    for role in candidate_roles:
        player = roster.get(role)
        if not player or not str(player).strip():
            continue
        stats = stats_map.get((str(player).strip(), str(role).upper(), champion))
        if stats:
            games, _ = stats
            score = games / 20.0  # Same calculation as comfort adapter
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
    from ml.inference.comfort_adapter import _load_team_ban_stats
    ban_stats_map = _load_team_ban_stats()
    
    if state.is_pick_phase():
        # Pick phase: Check if champion is frequently banned against OUR team
        if our_team_name:
            ban_rate = ban_stats_map.get((our_team_name, champion), 0.0)
            if ban_rate >= 0.4:
                return f"Most teams ban {champion} against {our_team_name}."
        
        # Otherwise, show player-based comfort message
        if opponent_roster:
            best_player = _get_best_player_for_champion(champion, state, opponent_roster)
            if best_player:
                return f"{champion} is a comfort pick for {best_player}."
        return f"{champion} is a strong comfort pick for your team."
    else:
        # Ban phase: Check if champion is frequently banned against OPPONENT team
        if opponent_team_name:
            ban_rate = ban_stats_map.get((opponent_team_name, champion), 0.0)
            if ban_rate >= 0.4:
                return f"Most teams ban {champion} against {opponent_team_name}."
        
        # Otherwise, show player-based comfort message for opponent
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
    # Get draft step info to provide context
    from core.draft_order import DRAFT_SEQUENCE
    step_label_expanded = None
    if state.step_index < len(DRAFT_SEQUENCE):
        step = DRAFT_SEQUENCE[state.step_index]
        step_label = step[3]  # e.g., "BP1", "RB4", etc.
        step_label_expanded = _expand_step_label(step_label)
    
    # Build base message
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
    
    # Add top non-policy factor if provided and score is meaningful
    if top_factor and top_factor[1] >= 0.5:
        factor_name, factor_score = top_factor
        
        # Generate detailed message for the factor
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
        
        # If we have a detailed message, combine it with the policy message
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
    # Second ban phase is steps 13-16 (0-indexed: 12-15)
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
            # If we don't know the champion's roles, assume it could fill any role
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
    
    # Find roles we've filled that opponent hasn't
    roles_to_deny = our_filled_roles - opponent_filled_roles
    
    if not roles_to_deny:
        return 0.0
    
    # Check if this champion can play in any of the roles we want to deny
    champion_roles = _get_champion_roles(champion)
    if not champion_roles:
        return 0.0
    
    # Find which role(s) this champion shares with roles we want to deny
    matching_roles = champion_roles & roles_to_deny
    if not matching_roles:
        return 0.0
    
    # Calculate bonus based on how many champions in the target role are already gone
    # Use the role with the most champions already picked/banned for maximum bonus
    max_bonus = 0.0
    roles_map = _load_roles_map()
    all_picked_banned = state.blue_picks | state.red_picks | state.bans
    
    for role in matching_roles:
        # Get all champions that can play this role
        champions_in_role = {
            champ for champ, roles in roles_map.items() 
            if role in roles
        }
        
        # Count how many champions in this role have been picked/banned
        champions_gone = len([champ for champ in champions_in_role if champ in all_picked_banned])
        
        # Bonus scales with how many champions are gone
        # 0 champions gone: 0.05 bonus (small incentive to start pinching)
        # 1-2 champions gone: 0.10-0.15 bonus (moderate pinching)
        # 3+ champions gone: 0.20 bonus (strong pinching, pool is getting thin)
        if champions_gone == 0:
            bonus = 0.05
        elif champions_gone == 1:
            bonus = 0.10
        elif champions_gone == 2:
            bonus = 0.15
        else:
            # Cap at 0.20 for 3+ champions gone
            bonus = min(0.20, 0.15 + (champions_gone - 2) * 0.025)
        
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
    
    # Calculate majority archetypes
    our_archetype = _get_majority_archetype(our_picks)
    opponent_archetype = _get_majority_archetype(opponent_picks)
    
    synergy_bonus = 0.0
    counter_bonus = 0.0
    
    # Synergy bonus: match our team's archetype
    if our_archetype and our_picks:  # Only apply if we have picks
        if _champion_has_archetype(champion, our_archetype):
            # Bonus scales with how many of our picks share this archetype
            matching_count = sum(
                1 for champ in our_picks 
                if _champion_has_archetype(champ, our_archetype)
            )
            # Scale from 0.05 (1 match) to 0.15 (3+ matches)
            synergy_bonus = min(0.15, 0.05 + (matching_count - 1) * 0.05)
    
    # Counter bonus: counter opponent's archetype
    if opponent_archetype and opponent_picks:  # Only apply if opponent has picks
        counter_archetype = _get_counter_archetype(opponent_archetype)
        if counter_archetype and _champion_has_archetype(champion, counter_archetype):
            # Bonus scales with how many opponent picks share the archetype we're countering
            opponent_matching_count = sum(
                1 for champ in opponent_picks 
                if _champion_has_archetype(champ, opponent_archetype)
            )
            # Scale from 0.05 (1 match) to 0.15 (3+ matches)
            counter_bonus = min(0.15, 0.05 + (opponent_matching_count - 1) * 0.05)
    
    return synergy_bonus, counter_bonus
