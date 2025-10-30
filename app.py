#!/usr/bin/env python3
"""
Kuhn Poker GTO Solver - Flask Web Application
Using your existing CFR implementation with a web interface

Combines:
- Your CFR implementation (KuhnNode, KuhnTrainer, KuhnGame, KuhnTest)
- My Linear Programming GTO solution  
- Flask web interface for interactive use

Run with: python app.py
"""

from flask import Flask, render_template, request, jsonify, session
import random
import pickle
import time
import os
from typing import Dict, List, Optional
import json

# Import your existing CFR classes
import random
from typing import *

PASS = 0
BET = 1
NUM_ACTIONS = 2

class KuhnNode:
    def __init__(self):
        self.infoSet = ''
        self.regretSum = [0] * NUM_ACTIONS
        self.strategy = [0] * NUM_ACTIONS
        self.strategySum = [0] * NUM_ACTIONS
        self.promising_branches = [0, 1]

    def __str__(self):
        return self.infoSet + ' ' + ', '.join(str(e) for e in self.getAverageStrategy())

    def getStrategy(self, realization_weight: float) -> List[float]:
        normalizingSum = 0
        for a in range(NUM_ACTIONS):
            if self.regretSum[a] > 0:
                self.strategy[a] = self.regretSum[a]
            else:
                self.strategy[a] = 0
            normalizingSum += self.strategy[a]

        for a in range(NUM_ACTIONS):
            if normalizingSum > 0:
                self.strategy[a] /= normalizingSum
            else:
                self.strategy[a] = 1 / NUM_ACTIONS
            self.strategySum[a] += realization_weight * self.strategy[a]

        return self.strategy

    def getAverageStrategy(self) -> list:
        avgStrategy = [0] * NUM_ACTIONS
        normalizingSum = sum(self.strategySum)

        for a in range(NUM_ACTIONS):
            if normalizingSum > 0:
                avgStrategy[a] = self.strategySum[a] / normalizingSum
            else:
                avgStrategy[a] = 1 / NUM_ACTIONS

        # Clean up very small probabilities
        for a in range(NUM_ACTIONS):
            if avgStrategy[a] < 0.01:
                avgStrategy[a] = 0

        normalizingSum = sum(avgStrategy)
        if normalizingSum > 0:
            for a in range(NUM_ACTIONS):
                avgStrategy[a] /= normalizingSum

        return avgStrategy

    def returnPayoff(self, cards: List[int]) -> Optional[int]:
        history = self.infoSet[1:len(self.infoSet)]
        plays = len(history)
        curr_player = plays % 2
        opponent = 1 - curr_player

        if plays > 1:
            terminalPass = history[plays - 1] == 'p'
            doubleBet = history[plays - 2: plays] == 'bb'
            isPlayerCardHigher = cards[curr_player] > cards[opponent]

            if terminalPass:
                if history == 'pp':
                    return 1 if isPlayerCardHigher else -1
                else:
                    return 1
            elif doubleBet:
                return 2 if isPlayerCardHigher else -2

        return None

class KuhnTrainer:
    def __init__(self):
        self.nodeMap = {}
        self.training_active = False
        self.current_iteration = 0
        self.current_game_value = 0.0

    def train_iterations(self, iterations: int):
        """Train for specified iterations"""
        cards = [1, 2, 3]
        util = 0

        for i in range(iterations):
            random.shuffle(cards)
            util += self.cfr(cards, '', 1, 1)
            self.current_iteration += 1

            if i % 1000 == 0:
                # Calculate current game value
                tester = KuhnTester()
                tester.nodeMap = self.nodeMap
                self.current_game_value = tester.gameValue()

        return self.current_game_value

    def cfr(self, cards: List[int], history: str, p0: float, p1: float) -> float:
        plays = len(history)
        curr_player = plays % 2
        infoSet = str(cards[curr_player]) + history

        temp_node = KuhnNode()
        temp_node.infoSet = infoSet
        payoff = temp_node.returnPayoff(cards)

        if payoff is not None:
            return payoff

        if infoSet not in self.nodeMap:
            self.nodeMap[infoSet] = KuhnNode()
            self.nodeMap[infoSet].infoSet = infoSet

        curr_node = self.nodeMap[infoSet]

        realization_weight = p1 if curr_player == 0 else p0
        strategy = curr_node.getStrategy(realization_weight)

        util = [0] * NUM_ACTIONS
        nodeUtil = 0

        for a in range(NUM_ACTIONS):
            nextHistory = history + ('p' if a == 0 else 'b')

            if curr_player == 0:
                util[a] = -self.cfr(cards, nextHistory, p0 * strategy[a], p1)
            else:
                util[a] = -self.cfr(cards, nextHistory, p0, p1 * strategy[a])

            nodeUtil += strategy[a] * util[a]

        for a in range(NUM_ACTIONS):
            regret = util[a] - nodeUtil
            curr_node.regretSum[a] += (p1 if curr_player == 0 else p0) * regret

        return nodeUtil

    def get_current_strategies(self):
        """Get current strategies in readable format"""
        strategies = {}
        card_names = {1: 'Jack', 2: 'Queen', 3: 'King'}
        
        for info_set in ['1', '2', '3', '1p', '1b', '2p', '2b', '3p', '3b']:
            if info_set in self.nodeMap:
                node = self.nodeMap[info_set]
                avg_strategy = node.getAverageStrategy()
                
                card_num = int(info_set[0])
                card_name = card_names[card_num]
                
                if len(info_set) == 1:  # Initial action
                    strategies[f"{card_name}_initial"] = {
                        'check_prob': avg_strategy[0],
                        'bet_prob': avg_strategy[1]
                    }
                else:  # Response action
                    action_type = 'after_check' if info_set[1] == 'p' else 'after_bet'
                    strategies[f"{card_name}_{action_type}"] = {
                        'pass_prob': avg_strategy[0],
                        'bet_call_prob': avg_strategy[1]
                    }
        
        return strategies

class KuhnTester:
    def __init__(self):
        self.nodeMap = {}

    def gameValue(self) -> float:
        if not self.nodeMap:
            return 0.0
            
        value = 0.0
        cardList = [[1,2],[1,3],[2,1],[2,3],[3,1],[3,2]]

        def valueRecursive(infoSet: str, cards: List[int]) -> float:
            if infoSet not in self.nodeMap:
                node = KuhnNode()
                node.infoSet = infoSet
                payoff = node.returnPayoff(cards)
                return payoff if payoff is not None else 0

            curr_player = (len(infoSet) - 1) % 2
            other = 1 - curr_player
            otherInfo = str(cards[other]) + infoSet[1:]

            strategy = self.nodeMap[infoSet].getAverageStrategy()
            value = 0

            for a in range(NUM_ACTIONS):
                action = 'p' if a == 0 else 'b'
                value += -valueRecursive(otherInfo + action, cards) * strategy[a]

            return value

        for cards in cardList:
            value += valueRecursive(str(cards[0]), cards) / 6

        return value

class KuhnGame:
    def __init__(self):
        self.AI = {}
        self.player_card = None
        self.ai_card = None
        self.history = ''
        self.pot = 2
        self.bankroll = 0
        self.game_over = False
        self.game_log = []

    def load_ai(self, nodeMap):
        """Load AI from trained model"""
        self.AI = nodeMap

    def start_new_game(self):
        """Start a new game"""
        cards = [1, 2, 3]
        random.shuffle(cards)
        self.player_card = cards[0]
        self.ai_card = cards[1]
        self.history = ''
        self.pot = 2
        self.game_over = False
        self.game_log = []
        
        return {
            'player_card': self.player_card,
            'ai_card_hidden': True,
            'pot': self.pot,
            'message': f"New game started! Your card: {self.get_card_name(self.player_card)}"
        }

    def player_action(self, action):
        """Process player action"""
        if self.game_over:
            return {'error': 'Game is over'}

        action_name = 'check' if action == 'p' else 'bet'
        self.history += action
        self.game_log.append(f"You {action_name}")

        if action == 'b':
            self.pot += 1

        # Check if game is terminal
        result = self.check_terminal()
        if result:
            return result

        # AI's turn
        return self.ai_action()

    def ai_action(self):
        """AI makes a decision"""
        if self.game_over:
            return {'error': 'Game is over'}

        info_set = str(self.ai_card) + self.history

        if info_set in self.AI:
            strategy = self.AI[info_set].getAverageStrategy()
            r = random.random()
            ai_action = 'p' if r < strategy[0] else 'b'
        else:
            # Default strategy if not trained
            ai_action = 'p' if random.random() < 0.5 else 'b'

        action_name = 'checks' if ai_action == 'p' else 'bets'
        self.history += ai_action
        self.game_log.append(f"AI {action_name}")

        if ai_action == 'b':
            self.pot += 1

        # Check if game is terminal
        return self.check_terminal()

    def check_terminal(self):
        """Check if game is in terminal state"""
        plays = len(self.history)
        
        if plays < 2:
            return None

        last_action = self.history[-1]
        
        if last_action == 'p':  # Someone passed
            if self.history == 'pp':  # Both passed
                winner = 'player' if self.player_card > self.ai_card else 'ai'
                winnings = 1 if winner == 'player' else -1
                self.bankroll += winnings
                self.game_over = True
                
                return {
                    'game_over': True,
                    'winner': winner,
                    'player_card': self.player_card,
                    'ai_card': self.ai_card,
                    'pot': self.pot,
                    'winnings': winnings,
                    'bankroll': self.bankroll,
                    'message': f"Showdown! AI had {self.get_card_name(self.ai_card)}. {winner.title()} wins ${abs(winnings)}!",
                    'log': self.game_log
                }
            else:  # Someone folded
                winner = 'ai' if plays % 2 == 1 else 'player'  # Last to act wins
                winnings = 1 if winner == 'player' else -1
                self.bankroll += winnings
                self.game_over = True
                
                return {
                    'game_over': True,
                    'winner': winner,
                    'player_card': self.player_card,
                    'ai_card': self.ai_card,
                    'pot': self.pot,
                    'winnings': winnings,
                    'bankroll': self.bankroll,
                    'message': f"Fold! AI had {self.get_card_name(self.ai_card)}. {winner.title()} wins ${abs(winnings)}!",
                    'log': self.game_log
                }
        
        elif plays >= 2 and self.history[-2:] == 'bb':  # Both bet
            winner = 'player' if self.player_card > self.ai_card else 'ai'
            winnings = 2 if winner == 'player' else -2
            self.bankroll += winnings
            self.game_over = True
            
            return {
                'game_over': True,
                'winner': winner,
                'player_card': self.player_card,
                'ai_card': self.ai_card,
                'pot': self.pot,
                'winnings': winnings,
                'bankroll': self.bankroll,
                'message': f"Showdown! AI had {self.get_card_name(self.ai_card)}. {winner.title()} wins ${abs(winnings)}!",
                'log': self.game_log
            }

        return None

    def get_card_name(self, card):
        """Get card name"""
        return {1: 'Jack', 2: 'Queen', 3: 'King'}[card]

# Linear Programming GTO Solution
class LinearProgrammingGTO:
    def __init__(self):
        self.game_value = -1/18  # Exact optimal value
        
        self.p1_behavioral = {
            'Jack': {'bet_prob': 1/9, 'call_check_bet_prob': 0.0},
            'Queen': {'bet_prob': 0.0, 'call_check_bet_prob': 4/9},
            'King': {'bet_prob': 1/3, 'call_check_bet_prob': 2/3}
        }
        
        self.p2_behavioral = {
            'Jack': {'call_prob': 0.0, 'bet_when_checked_prob': 1/3},
            'Queen': {'call_prob': 1/3, 'bet_when_checked_prob': 0.0},
            'King': {'call_prob': 1.0, 'bet_when_checked_prob': 1.0}
        }

    def get_gto_advice(self, player, card, situation):
        """Get GTO advice"""
        card_names = {1: 'Jack', 2: 'Queen', 3: 'King'}
        card_name = card_names[card]
        
        if player == 1:
            if situation == 'initial':
                bet_prob = self.p1_behavioral[card_name]['bet_prob']
                if bet_prob >= 0.5:
                    return f"BET ({bet_prob:.1%})", f"Optimal to bet with {card_name} {bet_prob:.1%} of time"
                else:
                    return f"CHECK ({1-bet_prob:.1%})", f"Optimal to check with {card_name} {1-bet_prob:.1%} of time"
            elif situation == 'check_bet':
                call_prob = self.p1_behavioral[card_name]['call_check_bet_prob']
                if call_prob >= 0.5:
                    return f"CALL ({call_prob:.1%})", f"Optimal to call with {card_name} {call_prob:.1%} vs check-bet"
                else:
                    return f"FOLD ({1-call_prob:.1%})", f"Optimal to fold with {card_name} {1-call_prob:.1%} vs check-bet"
        
        elif player == 2:
            if situation == 'vs_bet':
                call_prob = self.p2_behavioral[card_name]['call_prob']
                if call_prob >= 0.5:
                    return f"CALL ({call_prob:.1%})", f"Optimal to call with {card_name} {call_prob:.1%} vs bet"
                else:
                    return f"FOLD ({1-call_prob:.1%})", f"Optimal to fold with {card_name} {1-call_prob:.1%} vs bet"
            elif situation == 'checked_to':
                bet_prob = self.p2_behavioral[card_name]['bet_when_checked_prob']
                if bet_prob >= 0.5:
                    return f"BET ({bet_prob:.1%})", f"Optimal to bet with {card_name} {bet_prob:.1%} when checked to"
                else:
                    return f"CHECK ({1-bet_prob:.1%})", f"Optimal to check with {card_name} {1-bet_prob:.1%} when checked to"
        
        return "Unknown situation", "Please check inputs"

# Flask Application
app = Flask(__name__)
app.secret_key = 'kuhn_poker_gto_secret_key'

# Global instances
trainer = KuhnTrainer()
game = KuhnGame()
lp_gto = LinearProgrammingGTO()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start CFR training"""
    data = request.json
    iterations = data.get('iterations', 10000)
    
    try:
        # Train for specified iterations
        game_value = trainer.train_iterations(iterations)
        strategies = trainer.get_current_strategies()
        
        return jsonify({
            'success': True,
            'iterations': trainer.current_iteration,
            'game_value': game_value,
            'strategies': strategies,
            'message': f'Training complete: {iterations} iterations'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_strategies', methods=['GET'])
def get_strategies():
    """Get current CFR strategies"""
    try:
        strategies = trainer.get_current_strategies()
        tester = KuhnTester()
        tester.nodeMap = trainer.nodeMap
        game_value = tester.gameValue()
        
        return jsonify({
            'success': True,
            'strategies': strategies,
            'game_value': game_value,
            'iterations': trainer.current_iteration
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_game', methods=['POST'])
def start_game():
    """Start new game"""
    try:
        game.load_ai(trainer.nodeMap)
        result = game.start_new_game()
        session['bankroll'] = game.bankroll
        
        return jsonify({
            'success': True,
            **result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/player_action', methods=['POST'])
def player_action():
    """Process player action"""
    data = request.json
    action = data.get('action')  # 'p' or 'b'
    
    try:
        result = game.player_action(action)
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']})
        
        return jsonify({
            'success': True,
            **result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_advice', methods=['POST'])
def get_advice():
    """Get GTO strategy advice"""
    data = request.json
    player = data.get('player')
    card = data.get('card') 
    situation = data.get('situation')
    
    try:
        # Get LP advice
        lp_advice, lp_reasoning = lp_gto.get_gto_advice(player, card, situation)
        
        # Get CFR advice if available
        cfr_advice = "No CFR data available"
        if trainer.nodeMap:
            # Simplified CFR advice lookup
            card_names = {1: 'Jack', 2: 'Queen', 3: 'King'}
            card_name = card_names[card]
            
            if situation == 'initial':
                info_set = str(card)
            elif situation == 'check_bet':
                info_set = str(card) + 'p'
            elif situation == 'vs_bet':
                info_set = str(card) + 'b'
            elif situation == 'checked_to':
                info_set = str(card) + 'p'
            else:
                info_set = str(card)
            
            if info_set in trainer.nodeMap:
                strategy = trainer.nodeMap[info_set].getAverageStrategy()
                if situation in ['initial', 'checked_to']:
                    if strategy[1] > 0.5:
                        cfr_advice = f"BET ({strategy[1]:.1%})"
                    else:
                        cfr_advice = f"CHECK ({strategy[0]:.1%})"
                else:
                    if strategy[1] > 0.5:
                        cfr_advice = f"CALL ({strategy[1]:.1%})"
                    else:
                        cfr_advice = f"FOLD ({strategy[0]:.1%})"
        
        return jsonify({
            'success': True,
            'lp_advice': lp_advice,
            'lp_reasoning': lp_reasoning,
            'cfr_advice': cfr_advice,
            'card_name': {1: 'Jack', 2: 'Queen', 3: 'King'}[card]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/lp_solution', methods=['GET'])
def lp_solution():
    """Get Linear Programming solution"""
    try:
        return jsonify({
            'success': True,
            'game_value': lp_gto.game_value,
            'p1_behavioral': lp_gto.p1_behavioral,
            'p2_behavioral': lp_gto.p2_behavioral
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)