# rl_trading_agent.py - Agente de Reinforcement Learning para Trading
import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pickle
import os

# Importar modelos existentes
from ai_models import AdvancedTradingAI

class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

@dataclass
class TradeAction:
    action: ActionType
    amount: float  # Porcentaje del capital (0.0 - 1.0)
    confidence: float
    reasoning: str

@dataclass
class TradingState:
    price_history: np.ndarray  # Últimos N precios
    technical_indicators: np.ndarray  # RSI, MACD, etc.
    portfolio_state: np.ndarray  # Cash, positions, etc.
    market_features: np.ndarray  # Volume, volatility, etc.
    timestamp: datetime

class TradingEnvironment(gym.Env):
    """Ambiente de trading para Reinforcement Learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 lookback_window: int = 50, transaction_cost: float = 0.001):
        super(TradingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        
        # Estado del ambiente
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.trades_made = 0
        self.successful_trades = 0
        
        # Historial para análisis
        self.balance_history = []
        self.trades_history = []
        self.action_history = []
        
        # Definir espacios de acción y observación
        # Acciones: [HOLD, BUY_small, BUY_medium, BUY_large, SELL_small, SELL_medium, SELL_large]
        self.action_space = spaces.Discrete(7)
        
        # Observaciones: [precios, indicadores técnicos, estado del portafolio]
        obs_dim = (
            lookback_window +  # Precios históricos
            10 +  # Indicadores técnicos
            5 +   # Estado del portafolio
            5     # Features del mercado
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Calcular indicadores técnicos
        self._calculate_technical_indicators()
        
    def _calculate_technical_indicators(self):
        """Calcula indicadores técnicos para todo el dataset"""
        # RSI
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12).mean()
        exp2 = self.data['Close'].ewm(span=26).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
        
        # Bollinger Bands
        rolling_mean = self.data['Close'].rolling(20).mean()
        rolling_std = self.data['Close'].rolling(20).std()
        self.data['BB_upper'] = rolling_mean + (rolling_std * 2)
        self.data['BB_lower'] = rolling_mean - (rolling_std * 2)
        self.data['BB_width'] = (self.data['BB_upper'] - self.data['BB_lower']) / rolling_mean
        
        # Moving Averages
        self.data['SMA_10'] = self.data['Close'].rolling(10).mean()
        self.data['SMA_30'] = self.data['Close'].rolling(30).mean()
        self.data['EMA_12'] = self.data['Close'].ewm(span=12).mean()
        
        # Volume indicators
        self.data['Volume_SMA'] = self.data['Volume'].rolling(20).mean()
        self.data['Volume_ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
        # Volatility
        self.data['Volatility'] = self.data['Close'].pct_change().rolling(20).std()
        
        # Price momentum
        self.data['Momentum'] = self.data['Close'].pct_change(periods=10)
        
        # Fill NaN values
        self.data = self.data.fillna(method='bfill').fillna(0)
    
    def reset(self):
        """Reinicia el ambiente"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades_made = 0
        self.successful_trades = 0
        
        self.balance_history = [self.initial_balance]
        self.trades_history = []
        self.action_history = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """Obtiene la observación actual del estado"""
        if self.current_step < self.lookback_window:
            return np.zeros(self.observation_space.shape[0])
        
        # Precios históricos normalizados
        prices = self.data['Close'].iloc[
            self.current_step - self.lookback_window:self.current_step
        ].values
        prices_normalized = (prices - prices.mean()) / (prices.std() + 1e-8)
        
        # Indicadores técnicos actuales
        current_data = self.data.iloc[self.current_step]
        technical_indicators = np.array([
            current_data['RSI'] / 100.0,
            np.tanh(current_data['MACD'] / current_data['Close']),
            np.tanh(current_data['MACD_histogram'] / current_data['Close']),
            (current_data['Close'] - current_data['BB_lower']) / 
            (current_data['BB_upper'] - current_data['BB_lower'] + 1e-8),
            current_data['BB_width'],
            (current_data['Close'] - current_data['SMA_10']) / current_data['SMA_10'],
            (current_data['Close'] - current_data['SMA_30']) / current_data['SMA_30'],
            current_data['Volume_ratio'],
            current_data['Volatility'],
            current_data['Momentum']
        ])
        
        # Estado del portafolio
        current_price = current_data['Close']
        portfolio_value = self.balance + self.shares_held * current_price
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.shares_held * current_price / self.initial_balance,
            portfolio_value / self.max_net_worth,
            self.shares_held / 1000.0,  # Normalizado
            (portfolio_value - self.initial_balance) / self.initial_balance
        ])
        
        # Features del mercado
        market_features = np.array([
            current_data['Volume'] / self.data['Volume'].mean(),
            current_data['High'] / current_data['Low'] - 1,
            (current_data['Close'] - current_data['Open']) / current_data['Open'],
            self.current_step / len(self.data),  # Posición temporal
            np.sin(2 * np.pi * self.current_step / 252)  # Estacionalidad anual
        ])
        
        # Combinar todas las observaciones
        observation = np.concatenate([
            prices_normalized,
            technical_indicators,
            portfolio_state,
            market_features
        ])
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """Ejecuta una acción en el ambiente"""
        current_price = self.data['Close'].iloc[self.current_step]
        prev_net_worth = self.net_worth
        
        # Mapear acción a decisión de trading
        action_taken = self._execute_action(action, current_price)
        
        # Avanzar al siguiente paso
        self.current_step += 1
        
        # Calcular nueva net worth
        if self.current_step < len(self.data):
            new_price = self.data['Close'].iloc[self.current_step]
            self.net_worth = self.balance + self.shares_held * new_price
        else:
            self.net_worth = self.balance + self.shares_held * current_price
        
        # Actualizar máximo net worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        # Calcular recompensa
        reward = self._calculate_reward(prev_net_worth, action_taken)
        
        # Verificar si el episodio terminó
        done = (
            self.current_step >= len(self.data) - 1 or
            self.net_worth <= self.initial_balance * 0.5 or  # Stop loss del 50%
            self.balance < 0
        )
        
        # Información adicional
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'trades_made': self.trades_made,
            'action_taken': action_taken,
            'current_price': current_price
        }
        
        # Guardar historial
        self.balance_history.append(self.net_worth)
        self.action_history.append(action)
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action, current_price):
        """Ejecuta la acción de trading"""
        action_mapping = {
            0: ('HOLD', 0.0),
            1: ('BUY', 0.33),   # Comprar 33%
            2: ('BUY', 0.66),   # Comprar 66%
            3: ('BUY', 1.0),    # Comprar 100%
            4: ('SELL', 0.33),  # Vender 33%
            5: ('SELL', 0.66),  # Vender 66%
            6: ('SELL', 1.0)    # Vender 100%
        }
        
        action_type, amount = action_mapping[action]
        
        if action_type == 'BUY' and amount > 0:
            available_cash = self.balance * amount
            shares_to_buy = available_cash / (current_price * (1 + self.transaction_cost))
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    self.trades_made += 1
                    
                    trade_info = {
                        'type': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost,
                        'timestamp': self.current_step
                    }
                    self.trades_history.append(trade_info)
                    
                    return f"BUY_{amount:.0%}"
        
        elif action_type == 'SELL' and amount > 0:
            shares_to_sell = self.shares_held * amount
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares_held -= shares_to_sell
                self.trades_made += 1
                
                trade_info = {
                    'type': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue,
                    'timestamp': self.current_step
                }
                self.trades_history.append(trade_info)
                
                return f"SELL_{amount:.0%}"
        
        return "HOLD"
    
    def _calculate_reward(self, prev_net_worth, action_taken):
        """Calcula la recompensa para el agente"""
        # Recompensa base: cambio en net worth
        net_worth_change = self.net_worth - prev_net_worth
        base_reward = net_worth_change / self.initial_balance
        
        # Penalizar pérdidas excesivas
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        drawdown_penalty = -drawdown * 2
        
        # Bonificar consistencia (Sharpe ratio implícito)
        if len(self.balance_history) > 10:
            returns = np.diff(self.balance_history[-10:]) / self.balance_history[-11:-1]
            if np.std(returns) > 0:
                sharpe_bonus = np.mean(returns) / np.std(returns) * 0.1
            else:
                sharpe_bonus = 0
        else:
            sharpe_bonus = 0
        
        # Penalizar trading excesivo
        if self.trades_made > 0:
            trade_frequency_penalty = -min(0.1, self.trades_made / self.current_step) * 0.5
        else:
            trade_frequency_penalty = 0
        
        # Bonificar holding periods largos y exitosos
        holding_bonus = 0
        if action_taken == "HOLD" and self.shares_held > 0:
            if net_worth_change > 0:
                holding_bonus = 0.1
        
        # Recompensa total
        total_reward = (
            base_reward +
            drawdown_penalty +
            sharpe_bonus +
            trade_frequency_penalty +
            holding_bonus
        )
        
        return np.clip(total_reward, -1.0, 1.0)  # Limitar recompensa
    
    def render(self, mode='human'):
        """Visualiza el estado actual"""
        if mode == 'human':
            profit = self.net_worth - self.initial_balance
            profit_pct = (profit / self.initial_balance) * 100
            
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares: {self.shares_held:.2f}")
            print(f"Net Worth: ${self.net_worth:.2f}")
            print(f"Profit: ${profit:.2f} ({profit_pct:.2f}%)")
            print(f"Trades: {self.trades_made}")
            print("-" * 40)

class DQNAgent:
    """Deep Q-Network Agent para trading"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Networks
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
    def _build_network(self):
        """Construye la red neuronal"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena experiencia en memoria"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Selecciona acción usando epsilon-greedy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Entrena el agente con experiencias pasadas"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Actualiza la red objetivo"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Guarda el modelo"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Carga el modelo"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_size, action_size, learning_rate=3e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_network = self._build_policy_network().to(self.device)
        self.value_network = self._build_value_network().to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # PPO parameters
        self.clip_ratio = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        
    def _build_policy_network(self):
        """Red de política"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
            nn.Softmax(dim=-1)
        )
    
    def _build_value_network(self):
        """Red de valor"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def get_action(self, state):
        """Obtiene acción usando la política"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_network(state_tensor)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[0, action])
        return action, log_prob.item()
    
    def get_value(self, state):
        """Obtiene valor del estado"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.value_network(state_tensor).item()
    
    def update(self, states, actions, rewards, log_probs, next_states, dones):
        """Actualiza las redes usando PPO"""
        # Calcular advanteges y returns
        values = []
        for state in states:
            values.append(self.get_value(state))
        
        advantages, returns = self._compute_gae(rewards, values, next_states, dones)
        
        # Convertir a tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        
        # PPO update
        for _ in range(4):  # Multiple epochs
            # Policy loss
            new_probs = self.policy_network(states_tensor)
            new_log_probs = torch.log(new_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze())
            
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values_pred = self.value_network(states_tensor).squeeze()
            value_loss = F.mse_loss(values_pred, returns_tensor)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
    
    def _compute_gae(self, rewards, values, next_states, dones):
        """Calcula Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else self.get_value(next_states[i])
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lam * gae * (1 - dones[i])
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Normalizar advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages.tolist(), returns

class RLTradingSystem:
    """Sistema principal de RL para Trading"""
    
    def __init__(self, data_source, agent_type='DQN', model_save_path='models/rl_agent.pth'):
        self.data_source = data_source
        self.agent_type = agent_type
        self.model_save_path = model_save_path
        self.env = None
        self.agent = None
        self.training_history = []
        self.performance_metrics = {}
        
        # Integración con sistema existente
        self.traditional_ai = AdvancedTradingAI()
        
    def initialize_environment(self, data):
        """Inicializa el ambiente de trading"""
        self.env = TradingEnvironment(data)
        
        # Crear agente según el tipo especificado
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        if self.agent_type == 'DQN':
            self.agent = DQNAgent(state_size, action_size)
        elif self.agent_type == 'PPO':
            self.agent = PPOAgent(state_size, action_size)
        else:
            raise ValueError(f"Tipo de agente no soportado: {self.agent_type}")
    
    def train_agent(self, episodes=1000, save_every=100):
        """Entrena el agente de RL"""
        if not self.env or not self.agent:
            raise ValueError("Ambiente y agente deben estar inicializados")
        
        episode_rewards = []
        episode_profits = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            step_count = 0
            
            # Para PPO, almacenar experiencias del episodio
            if self.agent_type == 'PPO':
                states, actions, rewards, log_probs, next_states, dones = [], [], [], [], [], []
            
            while not done and step_count < 1000:  # Limitar pasos por episodio
                # Obtener acción del agente
                if self.agent_type == 'DQN':
                    action = self.agent.act(state)
                elif self.agent_type == 'PPO':
                    action, log_prob = self.agent.get_action(state)
                    log_probs.append(log_prob)
                
                # Ejecutar acción
                next_state, reward, done, info = self.env.step(action)
                
                # Almacenar experiencia
                if self.agent_type == 'DQN':
                    self.agent.remember(state, action, reward, next_state, done)
                elif self.agent_type == 'PPO':
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    dones.append(done)
                
                state = next_state
                total_reward += reward
                step_count += 1
            
            # Entrenar agente
            if self.agent_type == 'DQN':
                if len(self.agent.memory) > 1000:
                    self.agent.replay(32)
                    
                # Actualizar red objetivo cada 100 episodios
                if episode % 100 == 0:
                    self.agent.update_target_network()
                    
            elif self.agent_type == 'PPO':
                if len(states) > 0:
                    self.agent.update(states, actions, rewards, log_probs, next_states, dones)
            
            # Guardar métricas
            episode_rewards.append(total_reward)
            final_profit = (self.env.net_worth - self.env.initial_balance) / self.env.initial_balance
            episode_profits.append(final_profit)
            
            # Log progreso
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_profit = np.mean(episode_profits[-50:])
                epsilon = getattr(self.agent, 'epsilon', 0)
                
                print(f"Episode {episode}")
                print(f"  Avg Reward (50): {avg_reward:.4f}")
                print(f"  Avg Profit (50): {avg_profit:.4f}")
                print(f"  Final Net Worth: ${self.env.net_worth:.2f}")
                print(f"  Trades Made: {self.env.trades_made}")
                if hasattr(self.agent, 'epsilon'):
                    print(f"  Epsilon: {epsilon:.4f}")
                print("-" * 50)
            
            # Guardar modelo
            if episode % save_every == 0 and episode > 0:
                self.save_agent(f"{self.model_save_path}_episode_{episode}")
        
        # Guardar historial de entrenamiento
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episode_profits': episode_profits,
            'training_episodes': episodes
        }
        
        # Calcular métricas de rendimiento
        self._calculate_performance_metrics()
        
        print(f"✅ Entrenamiento completado: {episodes} episodios")
        print(f"📈 Profit promedio final: {np.mean(episode_profits[-100:]):.2%}")
        
    def _calculate_performance_metrics(self):
        """Calcula métricas de rendimiento del agente"""
        if not self.training_history:
            return
        
        profits = self.training_history['episode_profits']
        rewards = self.training_history['episode_rewards']
        
        self.performance_metrics = {
            'avg_profit': np.mean(profits),
            'max_profit': np.max(profits),
            'min_profit': np.min(profits),
            'profit_std': np.std(profits),
            'avg_reward': np.mean(rewards),
            'sharpe_ratio': np.mean(profits) / (np.std(profits) + 1e-8),
            'max_drawdown': self._calculate_max_drawdown(profits),
            'win_rate': len([p for p in profits if p > 0]) / len(profits),
            'profit_factor': self._calculate_profit_factor(profits)
        }
    
    def _calculate_max_drawdown(self, profits):
        """Calcula el máximo drawdown"""
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / (running_max + 1e-8)
        return np.max(drawdowns)
    
    def _calculate_profit_factor(self, profits):
        """Calcula el factor de profit"""
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        if not losses:
            return float('inf')
        
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        
        return total_wins / total_losses if total_losses > 0 else float('inf')
    
    def predict_action(self, current_state):
        """Predice acción usando el agente entrenado"""
        if not self.agent:
            raise ValueError("Agente no inicializado")
        
        if self.agent_type == 'DQN':
            action = self.agent.act(current_state, training=False)
        elif self.agent_type == 'PPO':
            action, _ = self.agent.get_action(current_state)
        
        # Mapear acción numérica a decisión de trading
        action_mapping = {
            0: TradeAction(ActionType.HOLD, 0.0, 0.5, "RL: Hold position"),
            1: TradeAction(ActionType.BUY, 0.33, 0.7, "RL: Buy 33%"),
            2: TradeAction(ActionType.BUY, 0.66, 0.8, "RL: Buy 66%"),
            3: TradeAction(ActionType.BUY, 1.0, 0.9, "RL: Buy 100%"),
            4: TradeAction(ActionType.SELL, 0.33, 0.7, "RL: Sell 33%"),
            5: TradeAction(ActionType.SELL, 0.66, 0.8, "RL: Sell 66%"),
            6: TradeAction(ActionType.SELL, 1.0, 0.9, "RL: Sell 100%")
        }
        
        return action_mapping.get(action, action_mapping[0])
    
    def combine_with_traditional_ai(self, market_data):
        """Combina RL con IA tradicional para decisión final"""
        # Obtener predicción de IA tradicional
        traditional_prediction = self.traditional_ai.predict_signal(market_data)
        
        # Obtener acción de RL
        rl_state = self._prepare_rl_state(market_data)
        rl_action = self.predict_action(rl_state)
        
        # Combinar ambas predicciones
        combined_decision = self._ensemble_predictions(traditional_prediction, rl_action)
        
        return combined_decision
    
    def _prepare_rl_state(self, market_data):
        """Prepara estado para el agente RL"""
        # Convertir datos de mercado al formato esperado por RL
        # (implementación simplificada)
        if isinstance(market_data, pd.DataFrame):
            return self.env._get_observation() if self.env else np.zeros(70)
        else:
            # Convertir dict a estado observable
            return np.array([
                market_data.get('rsi', 50) / 100,
                market_data.get('macd', 0),
                market_data.get('price', 100),
                # ... más features
            ] + [0] * 67)  # Padding para completar dimensiones
    
    def _ensemble_predictions(self, traditional_pred, rl_action):
        """Combina predicciones tradicionales y RL"""
        # Pesos para cada método
        traditional_weight = 0.4
        rl_weight = 0.6
        
        # Convertir predicción tradicional a formato estándar
        trad_signal = traditional_pred.get('signal', 'HOLD')
        trad_confidence = traditional_pred.get('confidence', 50) / 100
        
        # Combinar señales
        if trad_signal == 'BUY' and rl_action.action == ActionType.BUY:
            # Ambos dicen comprar
            final_action = ActionType.BUY
            final_amount = min(1.0, rl_action.amount * 1.2)  # Aumentar cantidad
            final_confidence = (trad_confidence * traditional_weight + 
                              rl_action.confidence * rl_weight)
            reasoning = f"Traditional: {trad_signal} + RL: {rl_action.reasoning}"
            
        elif trad_signal == 'SELL' and rl_action.action == ActionType.SELL:
            # Ambos dicen vender
            final_action = ActionType.SELL
            final_amount = min(1.0, rl_action.amount * 1.2)
            final_confidence = (trad_confidence * traditional_weight + 
                              rl_action.confidence * rl_weight)
            reasoning = f"Traditional: {trad_signal} + RL: {rl_action.reasoning}"
            
        elif (trad_signal in ['BUY', 'SELL'] and rl_action.action == ActionType.HOLD) or \
             (trad_signal == 'HOLD' and rl_action.action in [ActionType.BUY, ActionType.SELL]):
            # Uno dice actuar, otro dice hold - decisión conservadora
            if trad_confidence > rl_action.confidence:
                final_action = ActionType.BUY if trad_signal == 'BUY' else (
                    ActionType.SELL if trad_signal == 'SELL' else ActionType.HOLD)
                final_amount = 0.5 if final_action != ActionType.HOLD else 0.0
            else:
                final_action = rl_action.action
                final_amount = rl_action.amount * 0.5
            
            final_confidence = max(trad_confidence, rl_action.confidence) * 0.7
            reasoning = f"Ensemble: Traditional={trad_signal}, RL={rl_action.action.name}"
            
        else:
            # Señales conflictivas - hold
            final_action = ActionType.HOLD
            final_amount = 0.0
            final_confidence = 0.3
            reasoning = "Conflicting signals - Hold"
        
        return TradeAction(
            action=final_action,
            amount=final_amount,
            confidence=final_confidence,
            reasoning=reasoning
        )
    
    def save_agent(self, filepath=None):
        """Guarda el agente entrenado"""
        if not filepath:
            filepath = self.model_save_path
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.agent_type == 'DQN':
            self.agent.save(filepath)
        elif self.agent_type == 'PPO':
            torch.save({
                'policy_network_state_dict': self.agent.policy_network.state_dict(),
                'value_network_state_dict': self.agent.value_network.state_dict(),
                'policy_optimizer_state_dict': self.agent.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': self.agent.value_optimizer.state_dict(),
            }, filepath)
        
        # Guardar métricas y historial
        metadata = {
            'agent_type': self.agent_type,
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath + '_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Agente guardado en: {filepath}")
    
    def load_agent(self, filepath=None):
        """Carga un agente entrenado"""
        if not filepath:
            filepath = self.model_save_path
        
        if not os.path.exists(filepath):
            print(f"⚠️ Archivo no encontrado: {filepath}")
            return False
        
        try:
            if self.agent_type == 'DQN':
                self.agent.load(filepath)
            elif self.agent_type == 'PPO':
                checkpoint = torch.load(filepath, map_location=self.agent.device)
                self.agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
                self.agent.value_network.load_state_dict(checkpoint['value_network_state_dict'])
                self.agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
                self.agent.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            
            # Cargar metadata
            metadata_path = filepath + '_metadata.pkl'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.training_history = metadata.get('training_history', {})
                    self.performance_metrics = metadata.get('performance_metrics', {})
            
            print(f"✅ Agente cargado desde: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando agente: {e}")
            return False
    
    def evaluate_agent(self, test_data, episodes=10):
        """Evalúa el rendimiento del agente en datos de prueba"""
        if not self.agent:
            print("⚠️ Agente no inicializado")
            return None
        
        # Crear ambiente de prueba
        test_env = TradingEnvironment(test_data)
        
        results = []
        
        for episode in range(episodes):
            state = test_env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Obtener acción sin exploración
                if self.agent_type == 'DQN':
                    action = self.agent.act(state, training=False)
                elif self.agent_type == 'PPO':
                    action, _ = self.agent.get_action(state)
                
                state, reward, done, info = test_env.step(action)
                total_reward += reward
            
            # Guardar resultados del episodio
            profit = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance
            results.append({
                'episode': episode,
                'total_reward': total_reward,
                'profit': profit,
                'net_worth': test_env.net_worth,
                'trades_made': test_env.trades_made,
                'final_balance': test_env.balance,
                'shares_held': test_env.shares_held
            })
        
        # Calcular estadísticas
        profits = [r['profit'] for r in results]
        rewards = [r['total_reward'] for r in results]
        
        evaluation_metrics = {
            'avg_profit': np.mean(profits),
            'std_profit': np.std(profits),
            'max_profit': np.max(profits),
            'min_profit': np.min(profits),
            'avg_reward': np.mean(rewards),
            'win_rate': len([p for p in profits if p > 0]) / len(profits),
            'sharpe_ratio': np.mean(profits) / (np.std(profits) + 1e-8),
            'results': results
        }
        
        print("📊 Evaluación del Agente:")
        print(f"  Profit Promedio: {evaluation_metrics['avg_profit']:.2%}")
        print(f"  Profit Máximo: {evaluation_metrics['max_profit']:.2%}")
        print(f"  Profit Mínimo: {evaluation_metrics['min_profit']:.2%}")
        print(f"  Win Rate: {evaluation_metrics['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {evaluation_metrics['sharpe_ratio']:.3f}")
        
        return evaluation_metrics
    
    def plot_training_progress(self):
        """Visualiza el progreso del entrenamiento"""
        if not self.training_history:
            print("⚠️ No hay historial de entrenamiento")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Recompensas por episodio
        episodes = range(len(self.training_history['episode_rewards']))
        axes[0, 0].plot(episodes, self.training_history['episode_rewards'])
        axes[0, 0].set_title('Recompensas por Episodio')
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Recompensa')
        
        # Profits por episodio
        axes[0, 1].plot(episodes, self.training_history['episode_profits'])
        axes[0, 1].set_title('Profit por Episodio')
        axes[0, 1].set_xlabel('Episodio')
        axes[0, 1].set_ylabel('Profit (%)')
        
        # Media móvil de recompensas
        window = 50
        if len(self.training_history['episode_rewards']) > window:
            moving_avg_rewards = pd.Series(self.training_history['episode_rewards']).rolling(window).mean()
            axes[1, 0].plot(episodes, moving_avg_rewards)
            axes[1, 0].set_title(f'Media Móvil Recompensas ({window} episodios)')
            axes[1, 0].set_xlabel('Episodio')
            axes[1, 0].set_ylabel('Recompensa Promedio')
        
        # Media móvil de profits
        if len(self.training_history['episode_profits']) > window:
            moving_avg_profits = pd.Series(self.training_history['episode_profits']).rolling(window).mean()
            axes[1, 1].plot(episodes, moving_avg_profits)
            axes[1, 1].set_title(f'Media Móvil Profits ({window} episodios)')
            axes[1, 1].set_xlabel('Episodio')
            axes[1, 1].set_ylabel('Profit Promedio (%)')
        
        plt.tight_layout()
        plt.show()

# rl_integration_main.py - Integración principal con el sistema existente
import asyncio
from main import app, training_manager, data_collector
from fastapi import BackgroundTasks

# Variables globales para RL
rl_system = None
rl_trained = False

async def initialize_rl_system():
    """Inicializa el sistema de RL"""
    global rl_system
    
    try:
        print("🤖 Inicializando sistema de Reinforcement Learning...")
        
        # Crear sistema RL
        rl_system = RLTradingSystem(
            data_source=data_collector,
            agent_type='DQN',  # Cambiar a 'PPO' si prefieres
            model_save_path='models/rl_trading_agent.pth'
        )
        
        # Intentar cargar modelo pre-entrenado
        if rl_system.load_agent():
            global rl_trained
            rl_trained = True
            print("✅ Modelo RL pre-entrenado cargado exitosamente")
        else:
            print("⚠️ No se encontró modelo pre-entrenado. Será necesario entrenar.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inicializando sistema RL: {e}")
        return False

@app.on_event("startup")
async def startup_with_rl():
    """Startup incluyendo RL"""
    # Tu inicialización existente
    global training_manager
    training_manager, _ = await setup_auto_training()
    
    # Añadir inicialización de RL
    await initialize_rl_system()
    
    print("🚀 Sistema completo iniciado (Tradicional + RL)")

# Nuevos endpoints para RL
@app.get("/api/rl/status")
async def get_rl_status():
    """Obtiene estado del sistema RL"""
    global rl_system, rl_trained
    
    if not rl_system:
        return {"status": "not_initialized"}
    
    return {
        "status": "initialized",
        "trained": rl_trained,
        "agent_type": rl_system.agent_type,
        "performance_metrics": rl_system.performance_metrics
    }

@app.post("/api/rl/train")
async def train_rl_agent(background_tasks: BackgroundTasks, episodes: int = 1000):
    """Inicia entrenamiento del agente RL"""
    if not rl_system:
        return {"error": "Sistema RL no inicializado"}
    
    background_tasks.add_task(train_rl_agent_background, episodes)
    
    return {
        "message": f"Entrenamiento RL iniciado con {episodes} episodios",
        "status": "training_started"
    }

async def train_rl_agent_background(episodes: int):
    """Tarea de entrenamiento en background"""
    global rl_trained
    
    try:
        print(f"🎯 Iniciando entrenamiento RL con {episodes} episodios...")
        
        # Obtener datos históricos para entrenamiento
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        training_data = []
        
        for symbol in symbols:
            data = data_collector.get_market_data(symbol, "2y")
            if not data.empty:
                training_data.append(data)
        
        if training_data:
            # Usar el primer dataset para entrenamiento (en producción, combinar todos)
            combined_data = pd.concat(training_data, ignore_index=True).sort_values('Date')
            
            # Inicializar ambiente con datos
            rl_system.initialize_environment(combined_data)
            
            # Entrenar agente
            rl_system.train_agent(episodes=episodes)
            
            # Guardar modelo entrenado
            rl_system.save_agent()
            
            rl_trained = True
            print("✅ Entrenamiento RL completado exitosamente")
            
        else:
            print("❌ No se pudieron obtener datos para entrenamiento")
            
    except Exception as e:
        print(f"❌ Error en entrenamiento RL: {e}")

@app.post("/api/rl/predict")
async def get_rl_prediction(market_data: dict):
    """Obtiene predicción del agente RL"""
    if not rl_system or not rl_trained:
        return {"error": "Agente RL no entrenado"}
    
    try:
        # Combinar RL con IA tradicional
        combined_decision = rl_system.combine_with_traditional_ai(market_data)
        
        return {
            "action": combined_decision.action.name,
            "amount": combined_decision.amount,
            "confidence": combined_decision.confidence,
            "reasoning": combined_decision.reasoning,
            "source": "RL + Traditional AI Ensemble"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/rl/performance")
async def get_rl_performance():
    """Obtiene métricas de rendimiento del RL"""
    if not rl_system:
        return {"error": "Sistema RL no inicializado"}
    
    return {
        "performance_metrics": rl_system.performance_metrics,
        "training_history": rl_system.training_history
    }

@app.post("/api/rl/evaluate")
async def evaluate_rl_agent(symbol: str = "AAPL", episodes: int = 10):
    """Evalúa el agente RL en datos de prueba"""
    if not rl_system or not rl_trained:
        return {"error": "Agente RL no entrenado"}
    
    try:
        # Obtener datos de prueba
        test_data = data_collector.get_market_data(symbol, "6mo")
        
        if test_data.empty:
            return {"error": "No se pudieron obtener datos de prueba"}
        
        # Evaluar agente
        evaluation_results = rl_system.evaluate_agent(test_data, episodes)
        
        return {
            "evaluation_results": evaluation_results,
            "test_symbol": symbol,
            "test_episodes": episodes
        }
        
    except Exception as e:
        return {"error": str(e)}

# Modificar endpoint existente para incluir RL
@app.get("/api/assets")
async def get_recommended_assets_with_rl():
    """Obtiene activos recomendados combinando IA tradicional + RL"""
    try:
        assets = []
        symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'GOOGL']
        
        for symbol in symbols:
            try:
                # Obtener datos de mercado
                df = data_collector.get_market_data(symbol, "6mo")
                if df.empty:
                    continue
                
                # Análisis IA tradicional (tu código existente)
                df = technical_analyzer.calculate_indicators(df)
                traditional_signals = technical_analyzer.generate_signals(df)
                
                # Si RL está disponible, combinar predicciones
                if rl_system and rl_trained:
                    market_data = {
                        'symbol': symbol,
                        'price': df['Close'].iloc[-1],
                        'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else 50,
                        'macd': df['MACD'].iloc[-1] if 'MACD' in df.columns else 0,
                        'volume': df['Volume'].iloc[-1],
                        'volatility': df['Close'].pct_change().std() * 100
                    }
                    
                    # Obtener decisión combinada
                    rl_decision = rl_system.combine_with_traditional_ai(market_data)
                    
                    # Usar decisión de RL como principal
                    final_signal = rl_decision.action.name
                    final_confidence = int(rl_decision.confidence * 100)
                    final_reasoning = rl_decision.reasoning
                    
                else:
                    # Usar solo IA tradicional
                    final_signal = traditional_signals['signal']
                    final_confidence = traditional_signals['confidence']
                    final_reasoning = traditional_signals['reasoning']
                
                # Crear asset con datos combinados
                current_price = df['Close'].iloc[-1]
                previous_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                # Calcular precio objetivo
                if final_signal == 'BUY':
                    target_price = current_price * 1.05
                elif final_signal == 'SELL':
                    target_price = current_price * 0.95
                else:
                    target_price = current_price
                
                asset = {
                    'symbol': symbol,
                    'name': f"{symbol} Inc.",
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'changePercent': round(change_percent, 2),
                    'signal': final_signal,
                    'confidence': final_confidence,
                    'targetPrice': round(target_price, 2),
                    'timeframe': '3-5 días',
                    'reasoning': final_reasoning,
                    'volume': int(df['Volume'].iloc[-1]),
                    'marketCap': 'N/A',
                    'pe': 0,
                    'volatility': round(df['Close'].pct_change().std() * 100, 1),
                    'ai_type': 'RL + Traditional' if (rl_system and rl_trained) else 'Traditional'
                }
                
                assets.append(asset)
                
            except Exception as e:
                print(f"Error procesando {symbol}: {e}")
                continue
        
        return assets
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    # Script para entrenar agente RL desde línea de comandos
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar agente de Reinforcement Learning')
    parser.add_argument('--episodes', type=int, default=1000, help='Número de episodios')
    parser.add_argument('--agent', type=str, default='DQN', choices=['DQN', 'PPO'], help='Tipo de agente')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Símbolo para entrenamiento')
    
    args = parser.parse_args()
    
    # Inicializar sistema
    from main import DataCollector
    collector = DataCollector()
    
    # Crear sistema RL
    rl_system = RLTradingSystem(
        data_source=collector,
        agent_type=args.agent
    )
    
    # Obtener datos
    data = collector.get_market_data(args.symbol, "2y")
    
    if not data.empty:
        # Inicializar y entrenar
        rl_system.initialize_environment(data)
        rl_system.train_agent(episodes=args.episodes)
        
        # Guardar modelo
        rl_system.save_agent()
        
        # Mostrar resultados
        rl_system.plot_training_progress()
        
        print("🎉 Entrenamiento completado!")
        print(f"📊 Métricas finales: {rl_system.performance_metrics}")
    else:
        print("❌ No se pudieron obtener datos de mercado")