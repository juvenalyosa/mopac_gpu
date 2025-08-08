import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PortfolioStrategy:
    """
    Gestiona una estrategia de trading sobre un portafolio de pares de divisas.
    Analiza múltiples mercados para encontrar las mejores oportunidades y ejecuta
    un número limitado de operaciones simultáneamente.
    """

    def __init__(self, currency_pairs, start_date, end_date, short_window, long_window,
                 stop_loss_pct, take_profit_pct, max_open_positions=5, initial_capital=10000):
        """
        Inicializa la estrategia de portafolio.

        Args:
            currency_pairs (list): Lista de pares de divisas a analizar (ej. ['EURUSD=X', 'USDJPY=X']).
            start_date (str): Fecha de inicio para los datos históricos.
            end_date (str): Fecha de fin para los datos históricos.
            short_window (int): Ventana corta para la media móvil.
            long_window (int): Ventana larga para la media móvil.
            stop_loss_pct (float): Porcentaje para el stop-loss.
            take_profit_pct (float): Porcentaje para el take-profit.
            max_open_positions (int): Número máximo de operaciones abiertas simultáneamente.
            initial_capital (float): Capital inicial para la simulación.
        """
        self.currency_pairs = currency_pairs
        self.start_date = start_date
        self.end_date = end_date
        self.short_window = short_window
        self.long_window = long_window
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_open_positions = max_open_positions
        self.initial_capital = initial_capital
        self.capital_per_position = initial_capital / max_open_positions
        
        self.market_data = {}
        self.portfolio_results = None

    def analyze_markets(self):
        """
        Paso 1: Descarga datos y genera señales para todos los pares en el universo.
        """
        print("1. Analizando todos los mercados...")
        for pair in self.currency_pairs:
            print(f"  - Procesando {pair}")
            try:
                # Descargar datos
                data = yf.download(pair, start=self.start_date, end=self.end_date, interval='1h')
                if data.empty:
                    print(f"    ADVERTENCIA: No se encontraron datos para {pair}. Se omitirá.")
                    continue
                
                # Ingeniería de características y señales
                data['SMA_short'] = data['Close'].rolling(window=self.short_window).mean()
                data['SMA_long'] = data['Close'].rolling(window=self.long_window).mean()
                
                # Métrica para ranking de señales (momentum)
                data['momentum_strength'] = (data['SMA_short'] - data['SMA_long']) / data['SMA_long']
                
                data.dropna(inplace=True)
                
                signal = np.where(data['SMA_short'] > data['SMA_long'], 1, -1)
                data['position_signal'] = pd.Series(signal, index=data.index).diff()

                self.market_data[pair] = data
            except Exception as e:
                print(f"    ERROR procesando {pair}: {e}")
        
        print("Análisis de mercados completado.")

    def run_portfolio_backtest(self):
        """
        Paso 2: Ejecuta el backtesting sobre el portafolio, seleccionando las mejores
        operaciones y respetando el límite de posiciones abiertas.
        """
        if not self.market_data:
            print("No hay datos de mercado para el backtest. Ejecuta analyze_markets primero.")
            return

        print("\n2. Ejecutando backtesting de portafolio...")
        
        # Unificar todos los timestamps de todos los mercados
        all_dates = sorted(list(set.union(*[set(df.index) for df in self.market_data.values()])))
        
        self.portfolio_results = pd.DataFrame(index=all_dates, columns=['capital', 'pnl'])
        self.portfolio_results['capital'] = self.initial_capital
        
        open_positions = {} # Diccionario para rastrear posiciones: {pair: entry_price}

        for i in range(1, len(all_dates)):
            current_date = all_dates[i]
            previous_date = all_dates[i-1]
            
            # Propagar capital del día anterior
            self.portfolio_results.loc[current_date, 'capital'] = self.portfolio_results.loc[previous_date, 'capital']
            
            # --- GESTIÓN DE POSICIONES ABIERTAS ---
            positions_to_close = []
            for pair, entry_price in open_positions.items():
                if current_date in self.market_data[pair].index:
                    current_price = self.market_data[pair].loc[current_date, 'Close']
                    pnl_ratio = (current_price - entry_price) / entry_price
                    
                    # Chequear Take Profit o Stop Loss
                    if pnl_ratio >= self.take_profit_pct or pnl_ratio <= -self.stop_loss_pct:
                        capital_change = self.capital_per_position * pnl_ratio
                        self.portfolio_results.loc[current_date, 'capital'] += capital_change
                        self.portfolio_results.loc[current_date, 'pnl'] = self.portfolio_results.loc[current_date, 'pnl'] + capital_change
                        positions_to_close.append(pair)
                        
                        outcome = "Take Profit" if pnl_ratio > 0 else "Stop Loss"
                        print(f"{current_date} | CIERRE {pair} | {outcome} | Ganancia/Pérdida: {capital_change:.2f}")

            for pair in positions_to_close:
                del open_positions[pair]

            # --- BÚSQUEDA DE NUEVAS OPORTUNIDADES ---
            if len(open_positions) < self.max_open_positions:
                potential_trades = []
                for pair, data in self.market_data.items():
                    if pair not in open_positions and current_date in data.index:
                        # Señal de compra es 2.0 (cruce de -1 a 1)
                        if data.loc[current_date, 'position_signal'] == 2.0:
                            strength = data.loc[current_date, 'momentum_strength']
                            potential_trades.append({'pair': pair, 'strength': strength})
                
                # Ordenar por la fuerza de la señal (mayor momentum primero)
                if potential_trades:
                    ranked_trades = sorted(potential_trades, key=lambda x: x['strength'], reverse=True)
                    
                    # Abrir las mejores posiciones hasta alcanzar el límite
                    num_can_open = self.max_open_positions - len(open_positions)
                    for trade in ranked_trades[:num_can_open]:
                        pair_to_open = trade['pair']
                        entry_price = self.market_data[pair_to_open].loc[current_date, 'Close']
                        open_positions[pair_to_open] = entry_price
                        print(f"{current_date} | APERTURA {pair_to_open} | Precio: {entry_price:.5f}")

        self.portfolio_results['returns'] = self.portfolio_results['capital'].pct_change()
        print("Backtesting de portafolio completado.")

    def evaluate_performance(self):
        """
        Paso 3: Calcula y muestra las métricas de rendimiento del portafolio.
        """
        if self.portfolio_results is None:
            print("No hay resultados para evaluar.")
            return

        print("\n--- 3. Evaluación de Rendimiento del Portafolio ---")

        total_return = (self.portfolio_results['capital'].iloc[-1] / self.initial_capital) - 1
        print(f"Capital Inicial: ${self.initial_capital:,.2f}")
        print(f"Capital Final: ${self.portfolio_results['capital'].iloc[-1]:,.2f}")
        print(f"Ganancia/Pérdida Neta: ${self.portfolio_results['capital'].iloc[-1] - self.initial_capital:,.2f}")
        print(f"Retorno Total: {total_return:.2%}")

        # Calcular Sharpe Ratio
        hourly_risk_free_rate = (1.02)**(1/(252*24)) - 1
        excess_returns = self.portfolio_results['returns'].fillna(0) - hourly_risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252 * 24)
        print(f"Ratio de Sharpe (Anualizado): {sharpe_ratio:.2f}")

        # Calcular Max Drawdown
        self.portfolio_results['peak'] = self.portfolio_results['capital'].cummax()
        self.portfolio_results['drawdown'] = (self.portfolio_results['capital'] - self.portfolio_results['peak']) / self.portfolio_results['peak']
        max_drawdown = self.portfolio_results['drawdown'].min()
        print(f"Máximo Drawdown: {max_drawdown:.2%}")

        # Plotting
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Gráfico de Evolución del Capital y Drawdown
        ax1.plot(self.portfolio_results.index, self.portfolio_results['capital'], label='Evolución del Capital del Portafolio', color='green')
        ax1.set_ylabel('Capital ($)', fontsize=12, color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        ax1.set_title('Rendimiento de la Estrategia de Portafolio Multi-Divisa', fontsize=16)

        ax2 = ax1.twinx()
        ax2.fill_between(self.portfolio_results.index, self.portfolio_results['drawdown']*100, 0,
                         alpha=0.3, color='red', label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.xlabel('Fecha', fontsize=12)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # --- Parámetros de la Estrategia ---
    # Universo de pares de divisas para analizar
    CURRENCY_UNIVERSE = [
        'EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 
        'USDCAD=X', 'USDCHF=X', 'NZDUSD=X', 'EURJPY=X',
        'GBPJPY=X', 'EURGBP=X'
    ]
    
    START_DATE = '2023-01-01'
    END_DATE = '2024-01-01'
    SHORT_WINDOW = 20
    LONG_WINDOW = 50
    STOP_LOSS = 0.02
    TAKE_PROFIT = 0.04
    MAX_POSITIONS = 5 # Límite de transacciones
    CAPITAL = 10000

    # --- Ejecución del Algoritmo ---
    portfolio_strategy = PortfolioStrategy(
        currency_pairs=CURRENCY_UNIVERSE,
        start_date=START_DATE,
        end_date=END_DATE,
        short_window=SHORT_WINDOW,
        long_window=LONG_WINDOW,
        stop_loss_pct=STOP_LOSS,
        take_profit_pct=TAKE_PROFIT,
        max_open_positions=MAX_POSITIONS,
        initial_capital=CAPITAL
    )

    portfolio_strategy.analyze_markets()
    portfolio_strategy.run_portfolio_backtest()
    portfolio_strategy.evaluate_performance()
