import random
import time
from config import GameConfig, GestureConfig

class Player:
    def __init__(self, name, is_ai=False):
        self.name = name
        self.is_ai = is_ai
        self.health = GameConfig.MAX_HEALTH
        self.current_gesture = None
        self.current_action = None
        
    def take_damage(self, damage):
        self.health = max(0, self.health - damage)
        return self.health
    
    def heal(self, amount):
        self.health = min(GameConfig.MAX_HEALTH, self.health + amount)
        return self.health
    
    def is_alive(self):
        return self.health > 0
    
    def reset(self):
        self.health = GameConfig.MAX_HEALTH
        self.current_gesture = None
        self.current_action = None

class AIOpponent(Player):
    def __init__(self):
        super().__init__("AI", is_ai=True)
        self.strategy_mode = 'random'
        self.last_player_gestures = []
        
    def choose_gesture(self, player_history=None):
        if self.strategy_mode == 'random':
            return random.choice(list(GestureConfig.GESTURES.values()))
        elif self.strategy_mode == 'smart':
            return self._smart_choose(player_history)
        else:
            return random.choice(list(GestureConfig.GESTURES.values()))
    
    def _smart_choose(self, player_history):
        if not player_history or len(player_history) < 3:
            return random.choice(list(GestureConfig.GESTURES.values()))
        
        gesture_counts = {}
        for g in player_history[-10:]:
            if g in gesture_counts:
                gesture_counts[g] += 1
            else:
                gesture_counts[g] = 1
        
        if gesture_counts:
            most_common = max(gesture_counts, key=gesture_counts.get)
            counter_gesture = self._get_counter_gesture(most_common)
            return counter_gesture
        
        return random.choice(list(GestureConfig.GESTURES.values()))
    
    def _get_counter_gesture(self, gesture):
        counter_map = {
            'rock': 'paper',
            'paper': 'scissors',
            'scissors': 'rock',
            'open': 'fist',
            'fist': 'open'
        }
        return counter_map.get(gesture, random.choice(list(GestureConfig.GESTURES.values())))

class BattleSystem:
    def __init__(self):
        self.player = Player("玩家")
        self.ai = AIOpponent()
        self.round_number = 0
        self.battle_log = []
        self.current_state = 'waiting'
        self.last_attack_time = 0
        self.attack_cooldown = 2.0
        
        self.effects = {
            'player_attack': None,
            'ai_attack': None,
            'player_defend': None,
            'ai_defend': None,
            'damage': None
        }
        
    def start_battle(self):
        self.player.reset()
        self.ai.reset()
        self.round_number = 0
        self.battle_log = []
        self.current_state = 'waiting'
        self._add_log("战斗开始！")
        
    def _add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.battle_log.append(f"[{timestamp}] {message}")
        if len(self.battle_log) > 10:
            self.battle_log.pop(0)
    
    def can_attack(self):
        current_time = time.time()
        return (current_time - self.last_attack_time) >= self.attack_cooldown
    
    def execute_round(self, player_gesture):
        if not self.can_attack():
            return None
        
        self.last_attack_time = time.time()
        self.round_number += 1
        
        self.player.current_gesture = player_gesture
        self.player.current_action = GestureConfig.BATTLE_ACTIONS.get(player_gesture, 'attack')
        
        ai_gesture = self.ai.choose_gesture()
        self.ai.current_gesture = ai_gesture
        self.ai.current_action = GestureConfig.BATTLE_ACTIONS.get(ai_gesture, 'attack')
        
        result = self._calculate_damage()
        self._add_round_log(result)
        
        return result
    
    def _calculate_damage(self):
        player_action = self.player.current_action
        ai_action = self.ai.current_action
        
        result = {
            'round': self.round_number,
            'player_gesture': self.player.current_gesture,
            'ai_gesture': self.ai.current_gesture,
            'player_action': player_action,
            'ai_action': ai_action,
            'player_damage': 0,
            'ai_damage': 0,
            'message': ''
        }
        
        if player_action == 'attack' and ai_action == 'attack':
            result['player_damage'] = GameConfig.ATTACK_DAMAGE
            result['ai_damage'] = GameConfig.ATTACK_DAMAGE
            result['message'] = "双方互相攻击！"
        elif player_action == 'attack' and ai_action == 'defend':
            result['player_damage'] = 0
            result['ai_damage'] = GameConfig.ATTACK_DAMAGE - GameConfig.DEFENSE_BLOCK
            result['message'] = "AI 成功防御！"
        elif player_action == 'defend' and ai_action == 'attack':
            result['player_damage'] = GameConfig.ATTACK_DAMAGE - GameConfig.DEFENSE_BLOCK
            result['ai_damage'] = 0
            result['message'] = "玩家成功防御！"
        elif player_action == 'defend' and ai_action == 'defend':
            result['player_damage'] = 0
            result['ai_damage'] = 0
            result['message'] = "双方都在防御！"
        
        self.player.take_damage(result['player_damage'])
        self.ai.take_damage(result['ai_damage'])
        
        return result
    
    def _add_round_log(self, result):
        player_gesture_name = GestureConfig.GESTURE_NAMES.get(result['player_gesture'], result['player_gesture'])
        ai_gesture_name = GestureConfig.GESTURE_NAMES.get(result['ai_gesture'], result['ai_gesture'])
        
        log_msg = f"第{result['round']}回合: 玩家({player_gesture_name}) vs AI({ai_gesture_name}) - "
        log_msg += result['message']
        
        if result['player_damage'] > 0:
            log_msg += f" 玩家受到 {result['player_damage']} 伤害"
        if result['ai_damage'] > 0:
            log_msg += f" AI受到 {result['ai_damage']} 伤害"
            
        self._add_log(log_msg)
    
    def check_game_over(self):
        if not self.player.is_alive() and not self.ai.is_alive():
            return 'draw'
        elif not self.player.is_alive():
            return 'ai_win'
        elif not self.ai.is_alive():
            return 'player_win'
        return None
    
    def get_battle_state(self):
        return {
            'player': {
                'name': self.player.name,
                'health': self.player.health,
                'max_health': GameConfig.MAX_HEALTH,
                'gesture': self.player.current_gesture,
                'action': self.player.current_action
            },
            'ai': {
                'name': self.ai.name,
                'health': self.ai.health,
                'max_health': GameConfig.MAX_HEALTH,
                'gesture': self.ai.current_gesture,
                'action': self.ai.current_action
            },
            'round': self.round_number,
            'can_attack': self.can_attack(),
            'cooldown_remaining': max(0, self.attack_cooldown - (time.time() - self.last_attack_time)),
            'game_over': self.check_game_over(),
            'battle_log': self.battle_log
        }
