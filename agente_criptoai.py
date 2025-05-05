# Corrigido e Aprimorado: Código Python completo para Análise de Cripto AI
# Versão: 2025-05-04 (v3 - Correções Finais e Melhorias)
# Correções:
# - Erros de sintaxe Pylance corrigidos (try/except, parênteses, statements).
# - Geração de PDF robustecida: Uso consistente de multi_cell, fonte UTF-8, tratamento de erros.
# - Aprimoramento do prompt da IA para justificativas mais detalhadas.
# - Refatorações menores e comentários adicionais para clareza.

import os
import json
import requests
import pandas as pd
import numpy as np
import tweepy
import ta
from datetime import datetime, timedelta
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from openai import OpenAI
from fpdf import FPDF
import streamlit as st
from binance.client import Client
from binance.exceptions import BinanceAPIException
from textblob import TextBlob
import plotly.graph_objects as go
from cachetools import TTLCache
import asyncio
import nest_asyncio
import shutil
import logging

# Configuração inicial
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constante para o caminho da fonte UTF-8
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_NAME = "NotoSansCJK"

# --------------------------------------------------
# 1. Definição de Tipos e Estado
# --------------------------------------------------
class TradingState(TypedDict):
    symbol: str
    interval: str
    lookback: str
    price_data: list
    twitter_sentiment: dict
    news_data: list
    indicators: dict
    sentiment_score: float
    decision: Literal["BUY", "SELL", "HOLD"]
    decision_reason: str
    report_path: Optional[str]
    timestamp: str
    price_chart: Optional[str]

# --------------------------------------------------
# 2. Cliente Binance
# --------------------------------------------------
class BinanceTradingClient:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        self._init_client()
        self.pairs_cache = TTLCache(maxsize=1, ttl=3600) # Cache por 1 hora

    def _init_client(self):
        try:
            # Garante um event loop asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if self.api_key and self.api_secret:
                self.client = Client(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    tld='com',
                    testnet=False,
                    requests_params={'timeout': 15}
                )
            else:
                self.client = Client(
                    tld='com',
                    requests_params={'timeout': 15}
                )
            self.client.ping() # Testa a conexão
            logging.info("Cliente Binance conectado com sucesso.")
            # Usar st.sidebar apenas se Streamlit estiver ativo
            if 'st' in globals() and hasattr(st, 'sidebar'):
                 st.sidebar.success("Cliente Binance conectado.")
        except BinanceAPIException as e:
            # CORRIGIDO: Adicionado except clause
            error_msg = f"Erro API Binance ao inicializar: {e.message}"
            logging.error(error_msg)
            if 'st' in globals() and hasattr(st, 'sidebar'):
                 st.sidebar.error(error_msg)
            self.client = None
        except Exception as e:
            # CORRIGIDO: Adicionado except clause
            error_msg = f"Erro inesperado na inicialização do Binance: {str(e)}"
            logging.error(error_msg)
            if 'st' in globals() and hasattr(st, 'sidebar'):
                 st.sidebar.error(error_msg)
            self.client = None

    def get_all_usdt_pairs(self, force_refresh=False):
        cache_key = "usdt_pairs"
        if not force_refresh and cache_key in self.pairs_cache:
            return self.pairs_cache[cache_key]

        if not self.client:
            msg = "Cliente Binance não inicializado. Não é possível buscar pares."
            logging.warning(msg)
            if 'st' in globals() and hasattr(st, 'error'):
                 st.error(msg)
            return []

        try:
            info = self.client.get_exchange_info()
            pairs = sorted(
                [s['symbol'] for s in info['symbols']
                 if s['symbol'].endswith('USDT')
                 and s['status'] == 'TRADING']
            )
            self.pairs_cache[cache_key] = pairs
            logging.info(f"{len(pairs)} pares USDT encontrados.")
            return pairs
        except BinanceAPIException as e:
            msg = f"Erro API Binance ao buscar pares: {e.message}"
            logging.error(msg)
            if 'st' in globals() and hasattr(st, 'error'):
                 st.error(msg)
            return []
        except Exception as e:
            msg = f"Erro inesperado ao buscar pares USDT: {str(e)}"
            logging.error(msg)
            if 'st' in globals() and hasattr(st, 'error'):
                 st.error(msg)
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"] # Fallback

    def get_historical_data(self, symbol: str, interval: str, lookback: str):
        if not self.client:
            msg = f"Cliente Binance não disponível para buscar dados de {symbol}. Análise interrompida."
            logging.error(msg)
            if 'st' in globals() and hasattr(st, 'error'): st.error(msg)
            return []

        try:
            logging.info(f"Buscando dados históricos para {symbol}, intervalo {interval}, período {lookback}...")
            if lookback.endswith('d'):
                days = int(lookback.replace('d', ''))
                start_dt = datetime.now() - timedelta(days=days)
                start_str = start_dt.strftime('%d %b, %Y %H:%M:%S')
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_str
                )
            else:
                # Para lookbacks não baseados em dias (ex: '1000'), assume limite
                try:
                    limit = int(lookback)
                except ValueError:
                    limit = 1000 # Default limit
                    logging.warning(f"Lookback '{lookback}' inválido, usando limite padrão de {limit}.")
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )

            if not klines:
                msg = f"Nenhum dado histórico encontrado para {symbol} com intervalo {interval} e período {lookback}."
                logging.warning(msg)
                if 'st' in globals() and hasattr(st, 'warning'): st.warning(msg)
                return []

            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True) # Remove linhas onde a conversão falhou

            # Calcula SMAs básicas aqui para garantir que existam
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            # Não remove NaNs aqui ainda, pois a análise técnica pode precisar dos dados iniciais

            logging.info(f"{len(df)} registros de preço obtidos para {symbol}.")
            return df.to_dict('records')

        except BinanceAPIException as e:
            msg = f"Erro API Binance ao buscar klines para {symbol}: {e.status_code} - {e.message}"
            logging.error(msg)
            if 'st' in globals() and hasattr(st, 'error'): st.error(msg)
            return []
        except Exception as e:
            msg = f"Erro inesperado ao obter dados históricos para {symbol}: {str(e)}"
            logging.error(msg)
            if 'st' in globals() and hasattr(st, 'error'): st.error(msg)
            return []

# --------------------------------------------------
# 3. Interface Streamlit para Input de APIs (se aplicável)
# --------------------------------------------------
def get_api_keys():
    if 'st' not in globals():
        logging.info("Streamlit não detectado. Carregando chaves de variáveis de ambiente (se definidas).")
        # Exemplo: Carregar de variáveis de ambiente
        keys = {
            'BINANCE_API_KEY': os.environ.get('BINANCE_API_KEY'),
            'BINANCE_SECRET_KEY': os.environ.get('BINANCE_SECRET_KEY'),
            'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
            'TWITTER_BEARER_TOKEN': os.environ.get('TWITTER_BEARER_TOKEN'),
            'NEWS_API_KEY': os.environ.get('NEWS_API_KEY')
        }
        if not keys['BINANCE_API_KEY'] or not keys['BINANCE_SECRET_KEY'] or not keys['OPENAI_API_KEY']:
            logging.warning("Chaves obrigatórias (Binance, OpenAI) não encontradas nas variáveis de ambiente.")
            # Poderia retornar None ou um dict parcial, dependendo da lógica desejada
            # Retornando parcial para permitir fallback para regras
        return keys

    # Lógica original com Streamlit
    st.sidebar.title("🔑 Configuração de APIs")

    # Inicializa session_state se necessário
    default_keys = {'binance_key': '', 'binance_secret': '', 'openai_key': '', 'twitter_token': '', 'newsapi_key': ''}
    for key, default_value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    with st.sidebar.expander("Credenciais (Obrigatório: Binance, OpenAI)", expanded=not st.session_state.get('api_keys_saved', False)):
        binance_key = st.text_input("Binance API Key", type="password", value=st.session_state.binance_key, key="bkey")
        binance_secret = st.text_input("Binance Secret Key", type="password", value=st.session_state.binance_secret, key="bsec")
        openai_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_key, key="okey")
        twitter_token = st.text_input("Twitter Bearer Token (Opcional)", type="password", value=st.session_state.twitter_token, key="tkey")
        newsapi_key = st.text_input("NewsAPI Key (Opcional)", type="password", value=st.session_state.newsapi_key, key="nkey")

        if st.button("Salvar Configurações", key="save_api_keys"):
            if not binance_key or not binance_secret or not openai_key:
                st.error("As chaves da Binance e OpenAI são obrigatórias!")
                st.session_state.api_keys_saved = False
                # Não retorna None aqui, permite que a interface continue
            else:
                st.session_state.binance_key = binance_key
                st.session_state.binance_secret = binance_secret
                st.session_state.openai_key = openai_key
                st.session_state.twitter_token = twitter_token
                st.session_state.newsapi_key = newsapi_key
                st.session_state.api_keys_saved = True
                st.success("Configurações de API salvas! Recarregando...")
                st.rerun() # Força o recarregamento para usar as novas chaves

    if st.session_state.get('api_keys_saved', False):
        return {
            'BINANCE_API_KEY': st.session_state.binance_key,
            'BINANCE_SECRET_KEY': st.session_state.binance_secret,
            'OPENAI_API_KEY': st.session_state.openai_key,
            'TWITTER_BEARER_TOKEN': st.session_state.twitter_token,
            'NEWS_API_KEY': st.session_state.newsapi_key
        }
    else:
        # Se as chaves não foram salvas (ou ainda não foram inseridas), retorna None
        # A lógica principal precisará lidar com isso (ex: desabilitar botão de análise)
        st.sidebar.warning("Insira e salve as chaves de API obrigatórias para continuar.")
        return None

# --------------------------------------------------
# 4. Agentes de Análise
# --------------------------------------------------
class DataCollector:
    def __init__(self, api_keys):
        self.api_keys = api_keys if api_keys else {}
        self.binance = BinanceTradingClient(
            api_key=self.api_keys.get('BINANCE_API_KEY'),
            api_secret=self.api_keys.get('BINANCE_SECRET_KEY')
        )
        self.twitter_client = None
        twitter_token = self.api_keys.get('TWITTER_BEARER_TOKEN')
        if twitter_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=twitter_token)
                self.twitter_client.get_me() # Testa a conexão
                logging.info("Cliente Twitter conectado.")
                if 'st' in globals() and hasattr(st, 'sidebar'): st.sidebar.success("Cliente Twitter conectado.")
            except Exception as e:
                msg = f"Erro ao conectar ao Twitter: {str(e)}. Análise de tweets desativada."
                logging.warning(msg)
                if 'st' in globals() and hasattr(st, 'sidebar'): st.sidebar.warning(msg)
                self.twitter_client = None

        self.news_api_key = self.api_keys.get('NEWS_API_KEY')
        if self.news_api_key:
             logging.info("NewsAPI configurada.")
             if 'st' in globals() and hasattr(st, 'sidebar'): st.sidebar.success("NewsAPI configurada.")
        else:
             logging.info("NewsAPI não configurada. Análise de notícias desativada.")
             if 'st' in globals() and hasattr(st, 'sidebar'): st.sidebar.info("NewsAPI não configurada. Análise de notícias desativada.")

    def fetch_market_data(self, state_input: dict):
        symbol = state_input['symbol']
        interval = state_input['interval']
        lookback = state_input['lookback']

        # Inicializa o estado com valores padrão
        state = TradingState(
            symbol=symbol,
            interval=interval,
            lookback=lookback,
            timestamp=datetime.utcnow().isoformat(),
            price_data=[],
            twitter_sentiment={'tweets': [], 'total_tweets': 0, 'query': '', 'error': None},
            news_data=[],
            indicators={},
            sentiment_score=0.0,
            decision='HOLD', # Default decision
            decision_reason='Aguardando análise completa.',
            report_path=None,
            price_chart=None
        )

        # 1. Buscar dados de preço
        state['price_data'] = self.binance.get_historical_data(symbol, interval, lookback)
        if not state['price_data']:
            msg = f"Falha crítica ao buscar dados de preço para {symbol}. Análise interrompida."
            logging.error(msg)
            if 'st' in globals() and hasattr(st, 'error'): st.error(msg)
            # Define uma razão de erro e retorna o estado parcial
            state['decision_reason'] = f"Erro Crítico: Falha ao buscar dados de preço para {symbol}."
            # Poderia lançar uma exceção ou ter um estado de erro dedicado
            return state # Retorna estado com erro

        # 2. Buscar dados do Twitter (se configurado e funcionando)
        if self.twitter_client:
            state['twitter_sentiment'] = self._get_twitter_sentiment(symbol)
        else:
            state['twitter_sentiment']['error'] = "Cliente Twitter não configurado ou falhou na inicialização."
            logging.warning(state['twitter_sentiment']['error'])

        # 3. Buscar notícias (se configurado)
        if self.news_api_key:
            state['news_data'] = self._get_news(symbol)

        return state

    def _get_twitter_sentiment(self, symbol: str):
        result = {'tweets': [], 'total_tweets': 0, 'query': '', 'error': None}
        # Tenta extrair um termo de busca mais relevante do símbolo
        query_term = symbol.replace('USDT', '').replace('BUSD', '').replace('BTC', '').replace('ETH', '')
        if len(query_term) < 2: query_term = symbol.replace('USDT', '') # Se sobrou pouco, usa a base (ex: BTC)
        query = f"#{query_term} OR ${query_term} lang:en -is:retweet"
        result['query'] = query
        logging.info(f"Buscando tweets com query: {query}")

        try:
            response = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=50, # Limita a 50 para não exceder limites rapidamente
                tweet_fields=['created_at', 'public_metrics', 'lang'],
                expansions=['author_id'],
                user_fields=['username']
            )
            tweet_list = []
            if response.data:
                users = {u.id: u.username for u in response.includes.get('users', [])} if response.includes else {}
                for tweet in response.data:
                    tweet_list.append({
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': str(tweet.created_at),
                        'likes': tweet.public_metrics.get('like_count', 0),
                        'retweets': tweet.public_metrics.get('retweet_count', 0),
                        'replies': tweet.public_metrics.get('reply_count', 0),
                        'author_id': tweet.author_id,
                        'author_username': users.get(tweet.author_id, 'N/A')
                    })
            result['tweets'] = tweet_list
            result['total_tweets'] = len(tweet_list)
            logging.info(f"{result['total_tweets']} tweets encontrados para {symbol}.")
        except tweepy.errors.TweepyException as e:
            # Erros comuns: Rate limit, autenticação, query inválida
            error_msg = f"Erro na API do Twitter ao buscar tweets: {str(e)}"
            result['error'] = error_msg
            logging.warning(error_msg)
            # Não mostrar erro no Streamlit aqui, será mostrado no relatório/UI depois
        except Exception as e:
            error_msg = f"Erro inesperado ao processar tweets ({symbol}): {str(e)}"
            result['error'] = error_msg
            logging.error(error_msg)
        return result

    def _get_news(self, symbol: str):
        news_list = []
        query_term = symbol.replace('USDT', '') # Busca pelo nome da moeda base
        url = ('https://newsapi.org/v2/everything?'
               f'q={query_term}&'
               'language=en&'
               'sortBy=publishedAt&' # Mais recentes primeiro
               'pageSize=10&' # Limita a 10 notícias
               f'apiKey={self.news_api_key}')
        logging.info(f"Buscando notícias para {query_term}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Levanta erro para status HTTP ruins (4xx, 5xx)
            data = response.json()
            if data.get('status') == 'ok':
                for article in data.get('articles', []):
                    news_list.append({
                        'title': article.get('title'),
                        'description': article.get('description'),
                        'url': article.get('url'),
                        'publishedAt': article.get('publishedAt'),
                        'source': article.get('source', {}).get('name')
                    })
                logging.info(f"{len(news_list)} notícias encontradas para {query_term}.")
            else:
                error_msg = f"Erro da NewsAPI: {data.get('code')} - {data.get('message')}"
                logging.warning(error_msg)
                # Poderia adicionar um erro ao estado aqui se necessário
        except requests.exceptions.RequestException as e:
            msg = f"Erro de rede ao buscar notícias ({symbol}): {str(e)}"
            logging.warning(msg)
            # Não mostrar erro no Streamlit aqui, será mostrado depois
        except Exception as e:
            msg = f"Erro inesperado ao processar notícias ({symbol}): {str(e)}"
            logging.error(msg)
        return news_list

class TechnicalAnalyst:
    def analyze(self, state: TradingState):
        price_data = state.get('price_data')
        if not price_data:
            msg = "Dados de preço ausentes para análise técnica."
            logging.warning(msg)
            if 'st' in globals() and hasattr(st, 'warning'): st.warning(msg)
            state['indicators'] = {}
            state['decision_reason'] = "Erro: Dados de preço indisponíveis para análise técnica."
            return state

        try:
            df = pd.DataFrame(price_data)
            logging.info(f"Iniciando análise técnica para {state['symbol']} com {len(df)} pontos.")

            # Garante que as colunas necessárias existem e são numéricas
            required_cols = ['high', 'low', 'close', 'volume', 'open']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Coluna técnica necessária '{col}' ausente nos dados.")
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove linhas com NaNs nas colunas essenciais APÓS a conversão
            df.dropna(subset=required_cols, inplace=True)
            if df.empty:
                 raise ValueError("DataFrame vazio após conversão numérica ou remoção de NaNs essenciais.")

            # Calcula indicadores usando a biblioteca 'ta'
            # Adiciona verificações de comprimento mínimo do DataFrame para evitar erros
            indicators_calculated = {}
            if len(df) >= 14:
                indicators_calculated['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            if len(df) >= 26: # MACD precisa de mais dados (12, 26, 9)
                macd = ta.trend.MACD(close=df['close'])
                indicators_calculated['macd'] = macd.macd()
                indicators_calculated['macd_signal'] = macd.macd_signal()
                indicators_calculated['macd_diff'] = macd.macd_diff()
            if len(df) >= 14:
                stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
                indicators_calculated['stoch'] = stoch.stoch()
                indicators_calculated['stoch_signal'] = stoch.stoch_signal()
            if len(df) >= 20:
                bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
                indicators_calculated['bollinger_hband'] = bollinger.bollinger_hband()
                indicators_calculated['bollinger_lband'] = bollinger.bollinger_lband()
                indicators_calculated['bollinger_mavg'] = bollinger.bollinger_mavg()
                indicators_calculated['bollinger_pband'] = bollinger.bollinger_pband() # Percent Band
                indicators_calculated['bollinger_wband'] = bollinger.bollinger_wband() # Width Band
                indicators_calculated['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20).chaikin_money_flow()

            # Adiciona os indicadores calculados ao DataFrame
            for key, series in indicators_calculated.items():
                df[key] = series

            # Adiciona SMAs (já calculadas no fetch, mas garante que estejam no df final)
            if 'sma_20' not in df.columns and len(df) >= 20:
                df['sma_20'] = df['close'].rolling(window=20).mean()
            if 'sma_50' not in df.columns and len(df) >= 50:
                df['sma_50'] = df['close'].rolling(window=50).mean()

            # Coleta os últimos valores dos indicadores (se existirem no df)
            last_row = df.iloc[-1] if not df.empty else pd.Series()
            state['indicators'] = {
                'rsi': round(last_row.get('rsi', np.nan), 2),
                'macd': round(last_row.get('macd', np.nan), 5),
                'macd_signal': round(last_row.get('macd_signal', np.nan), 5),
                'macd_diff': round(last_row.get('macd_diff', np.nan), 5),
                'stoch': round(last_row.get('stoch', np.nan), 2),
                'stoch_signal': round(last_row.get('stoch_signal', np.nan), 2),
                'bollinger_hband': round(last_row.get('bollinger_hband', np.nan), 5),
                'bollinger_lband': round(last_row.get('bollinger_lband', np.nan), 5),
                'bollinger_mavg': round(last_row.get('bollinger_mavg', np.nan), 5),
                'bollinger_pband': round(last_row.get('bollinger_pband', np.nan) * 100, 2) if pd.notna(last_row.get('bollinger_pband')) else np.nan, # Converte para %
                'bollinger_wband': round(last_row.get('bollinger_wband', np.nan), 2),
                'cmf': round(last_row.get('cmf', np.nan), 3),
                'sma_20': round(last_row.get('sma_20', np.nan), 5),
                'sma_50': round(last_row.get('sma_50', np.nan), 5),
                'last_close': round(last_row.get('close', np.nan), 5)
            }
            logging.info(f"Indicadores técnicos calculados para {state['symbol']}: {state['indicators']}")

            # Atualiza price_data no estado para incluir indicadores calculados (para gráfico)
            # Converte NaN para None para serialização JSON se necessário
            df_dict = df.replace({np.nan: None}).to_dict('records')
            state['price_data'] = df_dict

        except ValueError as ve:
             msg = f"Erro nos dados para análise técnica ({state['symbol']}): {str(ve)}"
             logging.warning(msg)
             if 'st' in globals() and hasattr(st, 'warning'): st.warning(msg)
             state['indicators'] = {}
             state['decision_reason'] = f"Erro: {msg}"
        except Exception as e:
            msg = f"Erro inesperado na análise técnica ({state['symbol']}): {str(e)}"
            logging.error(msg, exc_info=True) # Log com traceback
            if 'st' in globals() and hasattr(st, 'error'): st.error(msg)
            state['indicators'] = {}
            state['decision_reason'] = f"Erro: {msg}"

        return state

class SentimentAnalyst:
    def analyze(self, state: TradingState):
        news = state.get('news_data', [])
        tweets_data = state.get('twitter_sentiment', {})
        tweets = tweets_data.get('tweets', [])

        texts_to_analyze = []
        # Coleta textos de notícias (título e descrição)
        for article in news:
            title = article.get('title', '') or ''
            desc = article.get('description', '') or ''
            if title or desc:
                texts_to_analyze.append(f"{title}. {desc}".strip())
        # Coleta textos de tweets
        for tweet in tweets:
            text = tweet.get('text', '')
            if text:
                texts_to_analyze.append(text)

        if not texts_to_analyze:
            logging.info(f"Nenhum texto (notícias/tweets) encontrado para análise de sentimento de {state['symbol']}.")
            state['sentiment_score'] = 0.0
            return state

        sentiment_scores = []
        logging.info(f"Analisando sentimento de {len(texts_to_analyze)} textos para {state['symbol']}...")
        try:
            for i, text in enumerate(texts_to_analyze):
                if text and isinstance(text, str):
                    # Limita o tamanho do texto para evitar sobrecarga (opcional)
                    # text = text[:500]
                    analysis = TextBlob(text)
                    # Polarity: -1 (neg) a 1 (pos)
                    # Subjectivity: 0 (obj) a 1 (subj)
                    sentiment_scores.append(analysis.sentiment.polarity)
                    # Log a cada N análises para feedback
                    # if (i + 1) % 10 == 0:
                    #     logging.debug(f"Análise de sentimento {i+1}/{len(texts_to_analyze)} concluída.")

            if sentiment_scores:
                avg_score = np.mean(sentiment_scores)
                state['sentiment_score'] = round(avg_score, 3)
                logging.info(f"Score médio de sentimento para {state['symbol']}: {state['sentiment_score']}")
            else:
                state['sentiment_score'] = 0.0
                logging.info(f"Nenhum score de sentimento calculado para {state['symbol']}.")

        except Exception as e:
            msg = f"Erro durante a análise de sentimento com TextBlob ({state['symbol']}): {str(e)}"
            logging.error(msg)
            if 'st' in globals() and hasattr(st, 'warning'): st.warning(msg)
            state['sentiment_score'] = 0.0 # Reseta em caso de erro
        return state

class DecisionMaker:
    def __init__(self, openai_key):
        self.openai_client = None
        if openai_key:
            try:
                self.openai_client = OpenAI(api_key=openai_key)
                self.openai_client.models.list() # Testa a conexão/chave
                logging.info("Cliente OpenAI conectado e chave válida.")
                if 'st' in globals() and hasattr(st, 'sidebar'): st.sidebar.success("Cliente OpenAI conectado.")
            except Exception as e:
                msg = f"Erro ao inicializar ou validar OpenAI: {str(e)}. Decisões usarão regras fallback."
                logging.error(msg)
                if 'st' in globals() and hasattr(st, 'sidebar'): st.sidebar.error(msg)
                self.openai_client = None
        else:
            msg = "Chave OpenAI não fornecida. Decisões usarão regras fallback."
            logging.warning(msg)
            if 'st' in globals() and hasattr(st, 'sidebar'): st.sidebar.warning(msg)

    def make_decision(self, state: TradingState):
        # Verifica se houve erro em etapas anteriores que impeçam a decisão
        if "Erro" in state.get('decision_reason', ''):
            logging.warning(f"Decisão não será tomada para {state['symbol']} devido a erro anterior: {state['decision_reason']}")
            # Mantém a razão do erro e a decisão padrão (HOLD)
            return state
        # Verifica se há indicadores para tomar a decisão
        if not state.get('indicators'):
             logging.warning(f"Indicadores técnicos ausentes para {state['symbol']}. Usando regras fallback (limitadas).")
             # Tenta usar regras mesmo sem indicadores completos, se possível
             return self._make_decision_rules(state)

        # Tenta usar LLM se configurado e sem erros anteriores
        if self.openai_client:
            logging.info(f"Tentando obter decisão da IA (LLM) para {state['symbol']}...")
            context = self._prepare_context_for_llm(state)
            prompt = self._build_llm_prompt(context)

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview", # ou gpt-3.5-turbo se preferir custo menor
                    messages=[
                        {"role": "system", "content": "Você é um experiente analista de trading de criptomoedas. Sua análise é objetiva, baseada em dados e considera múltiplos fatores. Forneça recomendações claras e justificativas detalhadas."}, 
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3, # Um pouco mais de criatividade controlada
                    max_tokens=250, # Aumenta um pouco para permitir justificativa mais longa
                    response_format={"type": "json_object"}
                )
                decision_data_str = response.choices[0].message.content
                logging.debug(f"Resposta crua do LLM: {decision_data_str}")
                decision_data = json.loads(decision_data_str)
                
                decision = decision_data.get('decision', 'HOLD').upper()
                if decision not in ["BUY", "SELL", "HOLD"]:
                    logging.warning(f"Decisão inválida recebida do LLM ('{decision}'), usando HOLD.")
                    decision = 'HOLD'
                
                reason = decision_data.get('reason', 'Análise via LLM não forneceu justificativa detalhada.')
                
                state['decision'] = decision
                state['decision_reason'] = reason
                logging.info(f"Decisão da IA para {state['symbol']}: {decision}. Razão: {reason}")
                return state # Retorna após sucesso com LLM

            except json.JSONDecodeError as json_e:
                msg = f"Erro ao decodificar JSON da resposta do GPT-4: {json_e}. Resposta: '{decision_data_str}'. Usando regras fallback."
                logging.error(msg)
                if 'st' in globals() and hasattr(st, 'error'): st.error("Erro na resposta da IA. Usando análise baseada em regras.")
                # Cai para regras fallback
            except Exception as e:
                msg = f"Erro na consulta ao GPT-4: {str(e)}. Usando regras fallback."
                logging.error(msg)
                if 'st' in globals() and hasattr(st, 'error'): st.error(f"Erro na IA: {str(e)}. Usando análise baseada em regras.")
                # Cai para regras fallback
        
        # Se LLM não está disponível, falhou, ou houve erro anterior, usa regras
        logging.info(f"Usando regras fallback para decisão de {state['symbol']}...")
        return self._make_decision_rules(state)

    def _make_decision_rules(self, state: TradingState):
        indicators = state.get('indicators', {})
        sentiment = state.get('sentiment_score', 0.0)
        
        # Pega os valores dos indicadores, tratando NaN/None
        rsi = indicators.get('rsi')
        macd_diff = indicators.get('macd_diff')
        stoch = indicators.get('stoch')
        bollinger_pband = indicators.get('bollinger_pband') # %B
        sma_20 = indicators.get('sma_20')
        sma_50 = indicators.get('sma_50')
        last_close = indicators.get('last_close')

        buy_signals = 0
        sell_signals = 0
        reasons = []

        # --- Lógica de Sinais --- 
        # RSI
        if rsi is not None and not np.isnan(rsi):
            if rsi < 30: buy_signals += 1.5; reasons.append("RSI < 30 (Forte Sobrevenda)")
            elif rsi < 40: buy_signals += 0.5; reasons.append("RSI < 40 (Sobrevenda Moderada)")
            elif rsi > 70: sell_signals += 1.5; reasons.append("RSI > 70 (Forte Sobrecompra)")
            elif rsi > 60: sell_signals += 0.5; reasons.append("RSI > 60 (Sobrecompra Moderada)")
        else: reasons.append("RSI Indisponível")

        # MACD
        if macd_diff is not None and not np.isnan(macd_diff):
            if macd_diff > 0: buy_signals += 1; reasons.append("MACD Acima da Linha de Sinal (Momentum Positivo)")
            elif macd_diff < 0: sell_signals += 1; reasons.append("MACD Abaixo da Linha de Sinal (Momentum Negativo)")
        else: reasons.append("MACD Indisponível")

        # Stochastic
        if stoch is not None and not np.isnan(stoch):
            if stoch < 20: buy_signals += 1; reasons.append("Stochastic < 20 (Sobrevenda)")
            elif stoch > 80: sell_signals += 1; reasons.append("Stochastic > 80 (Sobrecompra)")
        else: reasons.append("Stochastic Indisponível")

        # Bollinger Bands %B
        if bollinger_pband is not None and not np.isnan(bollinger_pband):
            if bollinger_pband < 5: buy_signals += 0.5; reasons.append("Preço Próximo à Banda Inferior Bollinger (%B < 5)")
            elif bollinger_pband > 95: sell_signals += 0.5; reasons.append("Preço Próximo à Banda Superior Bollinger (%B > 95)")
        else: reasons.append("Bandas Bollinger Indisponíveis")

        # SMAs (Médias Móveis Simples)
        valid_smas = all(v is not None and not np.isnan(v) for v in [sma_20, sma_50, last_close])
        if valid_smas:
            if last_close > sma_20 and sma_20 > sma_50: buy_signals += 1.5; reasons.append("Preço > SMA20 > SMA50 (Forte Tendência de Alta)")
            elif last_close > sma_20: buy_signals += 0.5; reasons.append("Preço > SMA20 (Tendência de Curto Prazo Alta)")
            elif last_close < sma_20 and sma_20 < sma_50: sell_signals += 1.5; reasons.append("Preço < SMA20 < SMA50 (Forte Tendência de Baixa)")
            elif last_close < sma_20: sell_signals += 0.5; reasons.append("Preço < SMA20 (Tendência de Curto Prazo Baixa)")
        else: reasons.append("SMAs Indisponíveis ou Preço Inválido")

        # Sentimento
        if sentiment > 0.2: buy_signals += 1; reasons.append("Sentimento Geral Positivo (> 0.2)")
        elif sentiment < -0.2: sell_signals += 1; reasons.append("Sentimento Geral Negativo (< -0.2)")
        elif sentiment > 0.05: buy_signals += 0.5; reasons.append("Sentimento Geral Levemente Positivo (> 0.05)")
        elif sentiment < -0.05: sell_signals += 0.5; reasons.append("Sentimento Geral Levemente Negativo (< -0.05)")
        else: reasons.append("Sentimento Geral Neutro (-0.05 a 0.05)")

        # --- Decisão Final Baseada em Sinais --- 
        decision = 'HOLD'
        if buy_signals > sell_signals + 1.5: # Requer uma margem maior para comprar
            decision = 'BUY'
        elif sell_signals > buy_signals + 1.5: # Requer uma margem maior para vender
            decision = 'SELL'
        # Se não houver margem clara, mantém HOLD

        state['decision'] = decision
        state['decision_reason'] = f"Decisão por Regras: {decision}. Pontos Compra={buy_signals:.1f}, Pontos Venda={sell_signals:.1f}. Fatores: {'; '.join(reasons) if reasons else 'Nenhum fator determinante.'}"
        logging.info(f"Decisão por regras para {state['symbol']}: {state['decision_reason']}")
        return state

    def _prepare_context_for_llm(self, state: TradingState):
        price_data = state.get('price_data', [])
        indicators = state.get('indicators', {})
        last_price = indicators.get('last_close', 'N/A')
        price_change_24h = 'N/A'
        price_change_7d = 'N/A'

        # Tenta calcular variações (requer dados suficientes e corretos)
        if len(price_data) > 1:
            try:
                # Usa o DataFrame que já tem os dados convertidos e limpos
                df_context = pd.DataFrame(price_data)
                df_context['close'] = pd.to_numeric(df_context['close'], errors='coerce')
                df_context.dropna(subset=['close'], inplace=True)
                
                if not df_context.empty:
                    current_close = df_context['close'].iloc[-1]
                    # Calcula variação 24h (aproximado baseado no intervalo)
                    # Assume que 'interval' pode ser '1h', '4h', etc.
                    candles_per_day = 24
                    if state['interval'].endswith('h'):
                        try: candles_per_day = 24 // int(state['interval'].replace('h',''))
                        except: pass
                    elif state['interval'].endswith('m'):
                         try: candles_per_day = (24 * 60) // int(state['interval'].replace('m',''))
                         except: pass
                    
                    if len(df_context) > candles_per_day:
                        past_24h_close = df_context['close'].iloc[-candles_per_day - 1]
                        if pd.notna(past_24h_close) and past_24h_close != 0:
                             price_change_24h = round(((current_close - past_24h_close) / past_24h_close) * 100, 2)
                    
                    candles_per_week = candles_per_day * 7
                    if len(df_context) > candles_per_week:
                         past_7d_close = df_context['close'].iloc[-candles_per_week - 1]
                         if pd.notna(past_7d_close) and past_7d_close != 0:
                              price_change_7d = round(((current_close - past_7d_close) / past_7d_close) * 100, 2)
            except Exception as e:
                logging.warning(f"Erro ao calcular variações de preço para contexto LLM: {e}")
                pass # Mantém 'N/A'

        # Formata indicadores para o prompt, tratando NaN/None
        formatted_indicators = {}
        for key, value in indicators.items():
            if value is None or np.isnan(value):
                formatted_indicators[key.upper()] = 'N/A'
            elif isinstance(value, float):
                # Formatação específica para diferentes indicadores
                if 'price' in key or 'sma' in key or 'bollinger' in key and 'pband' not in key and 'wband' not in key:
                    formatted_indicators[key.upper()] = f"{value:.5f}"
                elif key in ['rsi', 'stoch', 'bollinger_pband', 'bollinger_wband']:
                    formatted_indicators[key.upper()] = f"{value:.2f}"
                elif key in ['macd', 'macd_signal', 'macd_diff']:
                     formatted_indicators[key.upper()] = f"{value:.5f}"
                elif key == 'cmf':
                     formatted_indicators[key.upper()] = f"{value:.3f}"
                else:
                    formatted_indicators[key.upper()] = f"{value:.2f}" # Default float format
            else:
                formatted_indicators[key.upper()] = str(value)

        return {
            "symbol": state.get('symbol', 'N/A'),
            "interval": state.get('interval', 'N/A'),
            "current_price": formatted_indicators.get('LAST_CLOSE', 'N/A'),
            "price_change_24h": price_change_24h,
            "price_change_7d": price_change_7d,
            "indicators": formatted_indicators,
            "sentiment_score": f"{state.get('sentiment_score', 0.0):.3f}",
            "news_count": len(state.get('news_data', [])),
            "tweets_count": state.get('twitter_sentiment', {}).get('total_tweets', 'N/A')
        }

    def _build_llm_prompt(self, context: dict):
        # APERFEIÇOADO: Prompt mais detalhado e pedindo justificativa mais elaborada.
        prompt = f"""
        **Análise de Trading para {context['symbol']} (Intervalo: {context['interval']})**

        **Contexto de Mercado:**
        - Preço Atual (Close): {context['current_price']}
        - Variação (aprox.): 24h: {context['price_change_24h']}% | 7d: {context['price_change_7d']}% 

        **Indicadores Técnicos Principais:**
        - RSI(14): {context['indicators'].get('RSI', 'N/A')} (Ideal: 30-70. <30 Sobrevenda, >70 Sobrecompra)
        - MACD Diff: {context['indicators'].get('MACD_DIFF', 'N/A')} (Positivo = Momentum Alta, Negativo = Momentum Baixa)
        - Stoch(14,3): {context['indicators'].get('STOCH', 'N/A')} (Ideal: 20-80. <20 Sobrevenda, >80 Sobrecompra)
        - Bollinger %B: {context['indicators'].get('BOLLINGER_PBAND', 'N/A')}% (>100 Próx. Sup., <0 Próx. Inf.)
        - SMA 20 / 50: {context['indicators'].get('SMA_20', 'N/A')} / {context['indicators'].get('SMA_50', 'N/A')} (Cruzamentos e posição do preço indicam tendência)
        - CMF(20): {context['indicators'].get('CMF', 'N/A')} (Positivo = Fluxo de Compra, Negativo = Fluxo de Venda)

        **Sentimento do Mercado:**
        - Score Agregado (Notícias/Tweets): {context['sentiment_score']} (Range: -1 a 1. >0.1 Positivo, <-0.1 Negativo)
        - Volume: {context['news_count']} Notícias | {context['tweets_count']} Tweets recentes analisados.

        **Sua Tarefa:**
        Como um analista de trading experiente, avalie a **convergência e divergência** dos sinais técnicos e de sentimento. Considere a força da tendência atual e potenciais pontos de reversão ou continuação. 
        
        **Instruções de Resposta:**
        1. Forneça uma decisão clara: **BUY**, **SELL** ou **HOLD**.
        2. Elabore uma **justificativa detalhada (3-4 frases)**, explicando os principais fatores (técnicos e de sentimento) que suportam sua decisão. Mencione sinais conflitantes, se houver, e como eles foram ponderados.

        **Responda ESTRITAMENTE em formato JSON com as chaves "decision" (string: "BUY", "SELL" ou "HOLD") e "reason" (string: sua justificativa detalhada).**

        Exemplo de formato de resposta:
        ```json
        {{
            "decision": "BUY",
            "reason": "A forte condição de sobrevenda indicada pelo RSI abaixo de 30 e Stochastics abaixo de 20 sugere um potencial ponto de reversão. Embora o MACD ainda esteja ligeiramente negativo, a proximidade do preço à banda inferior de Bollinger reforça a possibilidade de um repique. O sentimento neutro não contradiz a análise técnica, tornando a compra de curto prazo atrativa."
        }}
        ```
        """
        return prompt

class ReportGenerator:
    def __init__(self):
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        self.font_added = False
        self.pdf_instance = None # Para reutilizar a instância FPDF e a fonte

    def _get_pdf_instance(self) -> FPDF:
        """Retorna uma instância FPDF, adicionando a fonte UTF-8 na primeira vez."""
        if self.pdf_instance is None:
            self.pdf_instance = FPDF()
            self._add_utf8_font(self.pdf_instance)
        return self.pdf_instance

    def _add_utf8_font(self, pdf: FPDF):
        """Adiciona a fonte UTF-8 ao objeto FPDF se ainda não foi adicionada."""
        if not self.font_added:
            try:
                if os.path.exists(FONT_PATH):
                    pdf.add_font(FONT_NAME, fname=FONT_PATH)
                    self.font_added = True
                    logging.info(f"Fonte {FONT_NAME} ({FONT_PATH}) adicionada com sucesso ao FPDF.")
                else:
                    logging.error(f"Arquivo da fonte {FONT_NAME} não encontrado em {FONT_PATH}. Usando fontes padrão (Arial). Relatório pode ter problemas com caracteres especiais.")
            except Exception as e:
                logging.error(f"Erro ao adicionar fonte {FONT_NAME} ao FPDF: {e}. Usando fontes padrão (Arial).", exc_info=True)
                self.font_added = False # Garante que tentará Arial

    def _set_font(self, pdf: FPDF, style='', size=10):
        """Define a fonte, usando a fonte UTF-8 se disponível, senão a padrão Arial."""
        target_font = FONT_NAME if self.font_added else 'Arial'
        try:
            # FPDF lida com estilos (B, I, U) internamente para fontes TTF adicionadas
            pdf.set_font(target_font, style=style, size=size)
        except RuntimeError as e:
             # Fallback se houver problema com a fonte/estilo (raro)
             logging.warning(f"Erro ao definir fonte {target_font} com estilo '{style}': {e}. Tentando Arial básico.")
             try:
                 pdf.set_font('Arial', style=style, size=size)
             except Exception as e2:
                 logging.error(f"Falha crítica ao definir fonte fallback Arial: {e2}")
                 # Em último caso, tenta sem estilo
                 pdf.set_font('Arial', style='', size=size)

    def _write_utf8(self, pdf: FPDF, text: str, h=5, align='L', border=0):
        """Escreve texto usando multi_cell para suportar UTF-8 e quebra de linha automática."""
        try:
            # Garante que o texto é string
            text_str = str(text) if text is not None else ""
            pdf.multi_cell(0, h, text_str, border=border, align=align, ln=1) # ln=1 para mover para próxima linha
        except UnicodeEncodeError as ue_err:
            # Este erro é menos provável com a fonte TTF correta, mas mantém fallback
            logging.warning(f"Erro de encoding ao escrever texto (apesar da fonte UTF-8?): {ue_err}. Texto: '{text_str[:50]}...'. Tentando fallback Latin-1.")
            try:
                fallback_text = text_str.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, h, fallback_text, border=border, align=align, ln=1)
            except Exception as e:
                 logging.error(f"Erro crítico ao escrever texto (fallback latin-1 falhou): {e}")
                 pdf.multi_cell(0, h, "[Erro ao renderizar texto]", border=border, align=align, ln=1)
        except Exception as e:
            logging.error(f"Erro inesperado em _write_utf8: {e}. Texto: '{text_str[:50]}...'", exc_info=True)
            pdf.multi_cell(0, h, "[Erro ao renderizar texto]", border=border, align=align, ln=1)

    def generate(self, state: TradingState):
        symbol = state['symbol']
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = os.path.join(self.reports_dir, f"{symbol}_report_{timestamp_str}.pdf")
        chart_filename = os.path.join(self.reports_dir, f"{symbol}_chart_{timestamp_str}.png")
        logging.info(f"Iniciando geração de relatório para {symbol} em {pdf_filename}")

        try:
            # 1. Gerar Gráfico (se possível)
            chart_generated = self._generate_price_chart_image(state, chart_filename)
            state['price_chart'] = chart_filename if chart_generated else None

            # 2. Gerar PDF
            pdf = self._get_pdf_instance() # Obtém instância com fonte
            pdf.add_page()

            # --- Cabeçalho --- 
            self._set_font(pdf, 'B', 16)
            self._write_utf8(pdf, f"Relatório de Análise - {symbol}", h=7, align='C')
            self._set_font(pdf, '', 9)
            self._write_utf8(pdf, f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", h=5, align='C')
            pdf.ln(8)

            # --- Decisão --- 
            self._set_font(pdf, 'B', 14)
            self._write_utf8(pdf, "Decisão Recomendada:", h=6)
            pdf.ln(1)

            decision = state.get('decision', 'HOLD')
            reason = state.get('decision_reason', 'N/A')
            decision_color = (0, 100, 0) if decision == 'BUY' else (139, 0, 0) if decision == 'SELL' else (255, 140, 0) # Darker colors
            
            pdf.set_fill_color(*decision_color) # Define a cor de fundo (ou texto)
            pdf.set_text_color(255, 255, 255) # Texto branco para contraste
            self._set_font(pdf, 'B', 13)
            pdf.multi_cell(0, 8, f"   {decision}   ", border=0, align='C', fill=True) # Usa fill=True
            pdf.ln(3)
            pdf.set_text_color(0, 0, 0) # Reseta cor do texto

            self._set_font(pdf, 'B', 11)
            self._write_utf8(pdf, "Justificativa:", h=5)
            self._set_font(pdf, '', 10)
            self._write_utf8(pdf, reason, h=5)
            pdf.ln(6)

            # --- Análise Técnica --- 
            self._add_technical_section(pdf, state)
            pdf.ln(6)

            # --- Análise de Sentimento --- 
            self._add_sentiment_section(pdf, state)
            pdf.ln(6)

            # --- Gráfico (em nova página se gerado) --- 
            if chart_generated:
                try:
                    pdf.add_page()
                    self._set_font(pdf, 'B', 14)
                    self._write_utf8(pdf, "Gráfico de Preço (Candlestick com SMAs):", h=6)
                    pdf.ln(2)

                    # Calcula dimensões e posição para centralizar
                    page_width = pdf.w - pdf.l_margin - pdf.r_margin
                    available_height = pdf.h - pdf.t_margin - pdf.b_margin - 20 # Subtrai margens e espaço para título
                    
                    # Tenta manter a proporção da imagem (assumindo 1000x500 original)
                    img_aspect_ratio = 1000 / 500
                    img_width = page_width * 0.95 # Usa 95% da largura
                    img_height = img_width / img_aspect_ratio

                    # Se a altura calculada exceder o espaço, recalcula baseado na altura
                    if img_height > available_height * 0.95:
                        img_height = available_height * 0.95
                        img_width = img_height * img_aspect_ratio
                        # Recalcula x_pos se a largura mudou
                        x_pos = pdf.l_margin + (page_width - img_width) / 2
                    else:
                         x_pos = pdf.l_margin + (page_width - img_width) / 2
                    
                    y_pos = pdf.get_y()
                    pdf.image(chart_filename, x=x_pos, y=y_pos, w=img_width, h=img_height)
                    logging.info(f"Imagem do gráfico {chart_filename} adicionada ao PDF.")
                except FileNotFoundError:
                     msg = f"Erro: Arquivo de imagem do gráfico não encontrado em {chart_filename}."
                     logging.error(msg)
                     self._add_placeholder_text(pdf, "(Erro: Arquivo do gráfico não encontrado)")
                except Exception as img_err:
                     msg = f"Erro ao adicionar imagem do gráfico ao PDF: {img_err}"
                     logging.error(msg, exc_info=True)
                     self._add_placeholder_text(pdf, "(Erro ao incluir gráfico no PDF)")
            else:
                 # Adiciona nova página mesmo se o gráfico não foi gerado, para consistência ou futuras seções
                 # pdf.add_page()
                 self._add_placeholder_text(pdf, "(Gráfico de preço não pôde ser gerado ou 'kaleido' não instalado)")

            # --- Fim do PDF --- 
            pdf.output(pdf_filename)
            state['report_path'] = pdf_filename
            msg = f"Relatório PDF gerado com sucesso: {pdf_filename}"
            logging.info(msg)
            if 'st' in globals() and hasattr(st, 'success'): st.success(msg)

            # Limpa arquivos antigos (relatórios e gráficos)
            self._clean_old_files(symbol, "_report_", ".pdf")
            self._clean_old_files(symbol, "_chart_", ".png")

        except Exception as e:
            error_detail = f"Erro crítico ao gerar relatório PDF para {symbol}: {str(e)}"
            logging.error(error_detail, exc_info=True)
            if 'st' in globals() and hasattr(st, 'error'): st.error(error_detail)
            
            state['report_path'] = None
            # Tenta remover gráfico se foi gerado mas PDF falhou
            if state.get('price_chart') and os.path.exists(state['price_chart']):
                 try: 
                     os.remove(state['price_chart'])
                     logging.info(f"Removido arquivo de gráfico {state['price_chart']} devido a erro no PDF.")
                 except Exception as rm_err:
                     logging.warning(f"Não foi possível remover o arquivo de gráfico {state['price_chart']}: {rm_err}")
            state['price_chart'] = None
        return state

    def _add_placeholder_text(self, pdf: FPDF, text: str):
        """Adiciona um texto de aviso no PDF quando algo falha."""
        pdf.ln(5)
        self._set_font(pdf, 'I', 10)
        pdf.set_text_color(150, 150, 150) # Cinza
        self._write_utf8(pdf, text, h=5, align='C')
        pdf.set_text_color(0, 0, 0) # Reseta cor

    def _generate_price_chart_image(self, state: TradingState, filename: str):
        price_data = state.get('price_data')
        if not price_data:
            msg = f"Dados de preço ausentes para gerar gráfico de {state['symbol']}."
            logging.warning(msg)
            if 'st' in globals() and hasattr(st, 'warning'): st.warning(msg)
            return False

        try:
            df = pd.DataFrame(price_data)
            # Validação crucial: open_time e colunas de preço
            if 'open_time' not in df.columns or df['open_time'].isnull().all():
                 msg = f"Coluna 'open_time' ausente ou vazia nos dados para gráfico de {state['symbol']}."
                 logging.error(msg)
                 if 'st' in globals() and hasattr(st, 'error'): st.error(msg)
                 return False
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                 if col not in df.columns:
                     msg = f"Coluna de preço '{col}' ausente para gráfico de {state['symbol']}."
                     logging.error(msg)
                     if 'st' in globals() and hasattr(st, 'error'): st.error(msg)
                     return False
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
            df.dropna(subset=['open_time'] + price_cols, inplace=True) # Remove linhas onde conversão falhou
            
            if df.empty:
                msg = f"DataFrame vazio após tratamento de dados para gráfico de {state['symbol']}."
                logging.warning(msg)
                if 'st' in globals() and hasattr(st, 'warning'): st.warning(msg)
                return False

            logging.info(f"Gerando gráfico Plotly para {state['symbol']}...")
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df['open_time'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=state['symbol']
                )
            ])

            # Adiciona SMAs se disponíveis e válidas
            if 'sma_20' in df.columns and df['sma_20'].notna().any():
                 fig.add_trace(go.Scatter(x=df['open_time'], y=df['sma_20'], mode='lines', name='SMA 20', line=dict(color='orange', width=1)))
            if 'sma_50' in df.columns and df['sma_50'].notna().any():
                 fig.add_trace(go.Scatter(x=df['open_time'], y=df['sma_50'], mode='lines', name='SMA 50', line=dict(color='purple', width=1)))

            fig.update_layout(
                title=f"Gráfico: {state['symbol']} - Intervalo: {state['interval']} (Período: {state['lookback']})",
                xaxis_title="Data",
                yaxis_title="Preço (USDT)",
                xaxis_rangeslider_visible=False,
                width=1000, # Largura para a imagem salva
                height=500, # Altura para a imagem salva
                margin=dict(l=50, r=50, t=60, b=50) # Margens para títulos não cortarem
            )

            # Tenta salvar a imagem (requer kaleido instalado)
            # Verifica se a pasta existe antes de salvar
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.write_image(filename, engine='kaleido', scale=2) # scale=2 para melhor resolução
            logging.info(f"Imagem do gráfico salva com sucesso em {filename}")
            return True

        except ImportError:
            # Erro específico se kaleido não estiver instalado
            msg = "Biblioteca 'kaleido' não instalada. Gráfico não será incluído no PDF. Instale com: pip install kaleido"
            logging.error(msg)
            if 'st' in globals() and hasattr(st, 'warning'): st.warning(msg)
            return False
        except Exception as e:
            msg = f"Erro inesperado ao gerar imagem do gráfico para {state['symbol']}: {str(e)}"
            logging.error(msg, exc_info=True)
            if 'st' in globals() and hasattr(st, 'error'): st.error(msg)
            return False

    def _add_technical_section(self, pdf: FPDF, state: TradingState):
        self._set_font(pdf, 'B', 14)
        self._write_utf8(pdf, "Análise Técnica:", h=6)
        pdf.ln(1)
        self._set_font(pdf, '', 10)

        indicators = state.get('indicators', {})
        if not indicators or all(v is None or (isinstance(v, float) and np.isnan(v)) for v in indicators.values()):
            self._add_placeholder_text(pdf, "Indicadores técnicos não disponíveis ou inválidos.")
            return

        # Organiza indicadores em duas colunas para melhor layout
        items_col1 = []
        items_col2 = []
        
        # Função helper para formatar indicador
        def format_indicator(key, label, unit=""):
            val = indicators.get(key, 'N/A')
            if val == 'N/A' or val is None or (isinstance(val, float) and np.isnan(val)):
                return f"{label}: N/A"
            # Formatação específica baseada na chave
            if key in ['last_close', 'sma_20', 'sma_50', 'bollinger_hband', 'bollinger_lband', 'bollinger_mavg', 'macd', 'macd_signal', 'macd_diff']:
                return f"{label}: {val:.5f}{unit}"
            elif key in ['rsi', 'stoch', 'stoch_signal', 'bollinger_pband', 'bollinger_wband']:
                 return f"{label}: {val:.2f}{unit}"
            elif key == 'cmf':
                 return f"{label}: {val:.3f}{unit}"
            else:
                 return f"{label}: {val}{unit}"

        items_col1.append(format_indicator('last_close', "Preço Atual"))
        items_col1.append(format_indicator('rsi', "RSI (14)"))
        items_col1.append(format_indicator('macd_diff', "MACD Diff"))
        items_col1.append(format_indicator('stoch', "Stoch (14,3)"))

        items_col2.append(format_indicator('sma_20', "SMA 20"))
        items_col2.append(format_indicator('sma_50', "SMA 50"))
        items_col2.append(format_indicator('bollinger_pband', "Bollinger %B", unit="%"))
        items_col2.append(format_indicator('cmf', "CMF (20)"))

        col_width = (pdf.w - pdf.l_margin - pdf.r_margin - 10) / 2 # Largura da coluna (com 10 de espaço entre elas)
        line_height = 6
        max_len = max(len(items_col1), len(items_col2))
        initial_y = pdf.get_y()

        for i in range(max_len):
            current_y = initial_y + i * line_height
            pdf.set_y(current_y)
            
            if i < len(items_col1):
                pdf.set_x(pdf.l_margin)
                # Usa multi_cell para cada item para garantir quebra se necessário
                pdf.multi_cell(col_width, line_height, items_col1[i], border=0, align='L')
            
            if i < len(items_col2):
                # Define a posição X para a segunda coluna
                pdf.set_xy(pdf.l_margin + col_width + 10, current_y)
                pdf.multi_cell(col_width, line_height, items_col2[i], border=0, align='L')

        # Move para baixo após a altura máxima das colunas
        pdf.set_y(initial_y + max_len * line_height + 2)

    def _add_sentiment_section(self, pdf: FPDF, state: TradingState):
        self._set_font(pdf, 'B', 14)
        self._write_utf8(pdf, "Análise de Sentimento:", h=6)
        pdf.ln(1)
        self._set_font(pdf, '', 10)

        sentiment_score = state.get('sentiment_score', 0.0)
        news_data = state.get('news_data', [])
        twitter_data = state.get('twitter_sentiment', {})
        tweets = twitter_data.get('tweets', [])
        total_tweets = twitter_data.get('total_tweets', 0)

        sentiment_label = "Positivo" if sentiment_score > 0.1 else "Negativo" if sentiment_score < -0.1 else "Neutro"
        self._write_utf8(pdf, f"Score Geral Agregado: {sentiment_score:.3f} ({sentiment_label})")
        self._write_utf8(pdf, f"Notícias Recentes Analisadas: {len(news_data)}")
        self._write_utf8(pdf, f"Tweets Recentes Analisados: {total_tweets}")

        # Mostra erro do Twitter se houver
        if twitter_data.get('error'):
             pdf.ln(1)
             self._set_font(pdf, 'I', 9)
             pdf.set_text_color(200, 0, 0) # Vermelho escuro
             self._write_utf8(pdf, f"Aviso (Twitter): {twitter_data['error']}", h=4)
             pdf.set_text_color(0, 0, 0)
             self._set_font(pdf, '', 10)

        # Mostra resumo de notícias
        if news_data:
            pdf.ln(4)
            self._set_font(pdf, 'B', 11)
            self._write_utf8(pdf, "Principais Notícias Recentes (Máx. 3):", h=5)
            self._set_font(pdf, '', 9)
            for i, news in enumerate(news_data[:3], 1):
                title = news.get('title', 'Sem título')
                source = news.get('source', 'N/A')
                # Limita o tamanho para evitar overflow, mesmo com multi_cell
                display_text = f"{i}. [{source}] {title}"
                self._write_utf8(pdf, display_text[:200] + ('...' if len(display_text) > 200 else ''), h=4)
            pdf.ln(1)

        # Mostra resumo de tweets
        if tweets:
            pdf.ln(4)
            self._set_font(pdf, 'B', 11)
            self._write_utf8(pdf, "Amostra de Tweets Recentes (Máx. 3):", h=5)
            self._set_font(pdf, '', 9)
            for i, tweet in enumerate(tweets[:3], 1):
                text = tweet.get('text', '')
                author = tweet.get('author_username', 'N/A')
                # Limita o tamanho
                display_text = f"{i}. (@{author}) {text}"
                self._write_utf8(pdf, display_text[:220] + ('...' if len(display_text) > 220 else ''), h=4)
            pdf.ln(1)

    def _clean_old_files(self, symbol: str, file_type: str, extension: str, keep: int = 5):
        """Remove arquivos antigos de relatório/gráfico, mantendo os 'keep' mais recentes."""
        try:
            # Lista arquivos que correspondem ao padrão no diretório de relatórios
            files = sorted(
                [os.path.join(self.reports_dir, f) for f in os.listdir(self.reports_dir)
                 if f.startswith(symbol + file_type) and f.endswith(extension)],
                key=os.path.getmtime,
                reverse=True # Mais recentes primeiro
            )
            # Remove arquivos excedentes
            if len(files) > keep:
                for old_file in files[keep:]:
                    try:
                        os.remove(old_file)
                        logging.info(f"Arquivo antigo removido: {old_file}")
                    except OSError as e:
                        logging.warning(f"Não foi possível remover o arquivo antigo {old_file}: {e}")
        except Exception as e:
            # Captura erros como problemas de permissão ao listar diretório
            logging.error(f"Erro ao limpar arquivos antigos ({symbol}{file_type}*{extension}): {e}")

# --------------------------------------------------
# 5. Grafo de Trabalho LangGraph
# --------------------------------------------------
def create_trading_workflow(api_keys):
    workflow = StateGraph(TradingState)

    # Instancia os agentes com as chaves
    collector = DataCollector(api_keys)
    tech_analyst = TechnicalAnalyst()
    sentiment_analyst = SentimentAnalyst()
    # Passa a chave OpenAI corretamente para o DecisionMaker
    decision_maker = DecisionMaker(api_keys.get('OPENAI_API_KEY')) 
    report_generator = ReportGenerator()

    # Adiciona nós ao grafo
    workflow.add_node("collect_data", collector.fetch_market_data)
    workflow.add_node("technical_analysis", tech_analyst.analyze)
    workflow.add_node("sentiment_analysis", sentiment_analyst.analyze)
    workflow.add_node("make_decision", decision_maker.make_decision)
    workflow.add_node("generate_report", report_generator.generate)

    # Define as transições (fluxo de trabalho)
    workflow.set_entry_point("collect_data")
    workflow.add_edge("collect_data", "technical_analysis")
    workflow.add_edge("technical_analysis", "sentiment_analysis")
    workflow.add_edge("sentiment_analysis", "make_decision")
    workflow.add_edge("make_decision", "generate_report")
    workflow.add_edge("generate_report", END) # Fim do fluxo

    # Compila o grafo
    app = workflow.compile()
    logging.info("Workflow de trading compilado com sucesso.")
    return app

# --------------------------------------------------
# 6. Interface Streamlit Completa (ou execução simples)
# --------------------------------------------------

def display_results_streamlit(result: TradingState):
    """Exibe os resultados formatados na interface Streamlit."""
    symbol = result.get('symbol', 'N/A')
    decision = result.get('decision', 'N/A')
    reason = result.get('decision_reason', 'N/A')
    report_path = result.get('report_path')
    chart_path = result.get('price_chart')
    indicators = result.get('indicators', {})
    sentiment_score = result.get('sentiment_score', 0.0)
    news_data = result.get('news_data', [])
    twitter_data = result.get('twitter_sentiment', {})
    tweets = twitter_data.get('tweets', [])

    st.header(f"Análise Cripto AI para {symbol}")
    st.caption(f"Análise gerada em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Seção de Recomendação --- 
    st.subheader(f"🎯 Recomendação")
    cols_rec = st.columns([1, 4])
    with cols_rec[0]:
        if decision == 'BUY':
            st.success(f"## {decision}")
        elif decision == 'SELL':
            st.error(f"## {decision}")
        else:
            st.warning(f"## {decision}")
    with cols_rec[1]:
        st.markdown("**Justificativa da IA:**")
        # Verifica se a razão indica erro
        if "Erro" in reason:
            st.error(reason)
        else:
            st.info(reason)
    st.divider()

    # --- Abas com Detalhes --- 
    tab_titles = ["📊 Gráfico", "📈 Indicadores Técnicos", "📰 Sentimento", "📄 Relatório PDF"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    # --- Aba Gráfico --- 
    with tab1:
        st.subheader("Gráfico de Preços e SMAs")
        if chart_path and os.path.exists(chart_path):
            try:
                st.image(chart_path, caption=f"Gráfico Candlestick para {symbol}")
            except Exception as e:
                st.warning(f"Não foi possível exibir a imagem do gráfico: {e}")
        elif result.get('price_data'):
            # Tenta gerar gráfico Plotly interativo se a imagem falhou mas os dados existem
            st.info("Gerando gráfico interativo (pode levar um momento)...")
            try:
                df_chart = pd.DataFrame(result['price_data'])
                # Revalida e converte dados para Plotly
                df_chart['open_time'] = pd.to_datetime(df_chart['open_time'], errors='coerce')
                price_cols = ['open', 'high', 'low', 'close']
                for col in price_cols: df_chart[col] = pd.to_numeric(df_chart[col], errors='coerce')
                df_chart.dropna(subset=['open_time'] + price_cols, inplace=True)
                
                if not df_chart.empty:
                    fig_live = go.Figure(data=[
                        go.Candlestick(
                            x=df_chart['open_time'], open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'], name=symbol
                        )
                    ])
                    # Adiciona SMAs se existirem e forem válidas
                    if 'sma_20' in df_chart.columns and df_chart['sma_20'].notna().any():
                         fig_live.add_trace(go.Scatter(x=df_chart['open_time'], y=df_chart['sma_20'], mode='lines', name='SMA 20', line=dict(color='orange', width=1)))
                    if 'sma_50' in df_chart.columns and df_chart['sma_50'].notna().any():
                         fig_live.add_trace(go.Scatter(x=df_chart['open_time'], y=df_chart['sma_50'], mode='lines', name='SMA 50', line=dict(color='purple', width=1)))
                    
                    fig_live.update_layout(
                        xaxis_rangeslider_visible=False, 
                        height=450, 
                        margin=dict(l=20, r=20, t=40, b=20),
                        title=f"Gráfico Interativo: {symbol}"
                    )
                    st.plotly_chart(fig_live, use_container_width=True)
                else:
                    st.warning("Dados de preço insuficientes ou inválidos para gerar gráfico Plotly interativo.")
            except Exception as plot_err:
                st.warning(f"Não foi possível gerar o gráfico Plotly interativo: {plot_err}")
        else:
            st.warning("Gráfico não disponível (sem dados de preço ou erro na geração).")

    # --- Aba Indicadores --- 
    with tab2:
        st.subheader("Indicadores Técnicos Chave")
        if indicators and not all(v is None or (isinstance(v, float) and np.isnan(v)) for v in indicators.values()):
            # Helper para exibir métrica com tratamento de N/A
            def display_metric(col, label, key, format_str="{:.2f}"):
                val = indicators.get(key)
                if val is None or np.isnan(val):
                    col.metric(label, "N/A")
                else:
                    # Aplica formatação específica
                    if key in ['last_close', 'sma_20', 'sma_50', 'bollinger_hband', 'bollinger_lband', 'bollinger_mavg', 'macd', 'macd_signal', 'macd_diff']:
                         format_str = "{:.5f}"
                    elif key == 'cmf':
                         format_str = "{:.3f}"
                    elif key == 'bollinger_pband':
                         format_str = "{:.2f}%" # Adiciona %
                    col.metric(label, format_str.format(val))

            cols_ind1 = st.columns(3)
            display_metric(cols_ind1[0], "Preço Atual", 'last_close')
            display_metric(cols_ind1[1], "RSI (14)", 'rsi')
            display_metric(cols_ind1[2], "MACD Diff", 'macd_diff')
            
            cols_ind2 = st.columns(3)
            display_metric(cols_ind2[0], "Stoch (14,3)", 'stoch')
            display_metric(cols_ind2[1], "Bollinger %B", 'bollinger_pband')
            display_metric(cols_ind2[2], "CMF (20)", 'cmf')

            cols_ind3 = st.columns(3)
            display_metric(cols_ind3[0], "SMA 20", 'sma_20')
            display_metric(cols_ind3[1], "SMA 50", 'sma_50')
            # Espaço reservado para outro indicador se necessário
            # cols_ind3[2].metric("Outro Indicador", "Valor")

        else:
            st.warning("Indicadores técnicos não disponíveis ou inválidos.")

    # --- Aba Sentimento --- 
    with tab3:
        st.subheader("Análise de Sentimento (Notícias e Tweets)")
        sentiment_label = "Positivo" if sentiment_score > 0.1 else "Negativo" if sentiment_score < -0.1 else "Neutro"
        st.metric("Score Geral de Sentimento", f"{sentiment_score:.3f}", delta=sentiment_label)
        
        col_sent1, col_sent2 = st.columns(2)
        col_sent1.info(f"**Notícias Analisadas:** {len(news_data)}")
        col_sent2.info(f"**Tweets Analisados:** {twitter_data.get('total_tweets', 0)}")

        if twitter_data.get('error'):
            st.warning(f"Aviso (Twitter): {twitter_data['error']}")

        if news_data:
            with st.expander("Principais Notícias Recentes (Máx. 5)", expanded=False):
                for i, news in enumerate(news_data[:5], 1):
                    st.markdown(f"**{i}. {news.get('title', 'Sem título')}** ({news.get('source', 'N/A')}) - *{news.get('publishedAt', '')[:10]}*" )
                    st.caption(news.get('description', '')[:200] + "...")
                    st.link_button("Ler mais", news.get('url', '#'), disabled=(not news.get('url')))
                    st.divider()
        else:
            st.markdown("*Nenhuma notícia recente encontrada ou NewsAPI não configurada.*")

        if tweets:
            with st.expander("Amostra de Tweets Recentes (Máx. 5)", expanded=False):
                for i, tweet in enumerate(tweets[:5], 1):
                    st.markdown(f"**{i}. @{tweet.get('author_username', 'N/A')}** - *{tweet.get('created_at', '')[:16]}*" )
                    st.caption(tweet.get('text', ''))
                    st.divider()
        else:
            st.markdown("*Nenhum tweet recente encontrado ou Twitter API não configurada/com erro.*")

    # --- Aba Relatório PDF --- 
    with tab4:
        st.subheader("Relatório Completo em PDF")
        if report_path and os.path.exists(report_path):
            try:
                with open(report_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                st.download_button(
                    label="Baixar Relatório PDF",
                    data=pdf_bytes,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf"
                )
                st.success(f"Relatório disponível para download: {os.path.basename(report_path)}")
            except Exception as e:
                st.error(f"Erro ao ler o arquivo PDF para download: {e}")
        else:
            st.warning("Relatório PDF não foi gerado ou ocorreu um erro durante a geração.")

def display_results_terminal(result: TradingState):
    """Exibe os resultados formatados no terminal."""
    print("\n" + "="*50)
    print(f"=== Análise Cripto AI para {result.get('symbol', 'N/A')} ===")
    print("="*50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*50)
    print(f"DECISÃO RECOMENDADA: {result.get('decision', 'N/A')}")
    print("-"*50)
    print("Justificativa:")
    print(result.get('decision_reason', 'N/A'))
    print("-"*50)
    print("Indicadores Técnicos:")
    indicators = result.get('indicators', {})
    if indicators:
        for key, value in indicators.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                 print(f"  - {key.replace('_', ' ').title()}: N/A")
            else:
                 # Formatação simples para terminal
                 if isinstance(value, float):
                     print(f"  - {key.replace('_', ' ').title()}: {value:.4f}")
                 else:
                     print(f"  - {key.replace('_', ' ').title()}: {value}")
    else:
        print("  (Não disponíveis)")
    print("-"*50)
    print("Sentimento do Mercado:")
    print(f"  - Score Geral: {result.get('sentiment_score', 0.0):.3f}")
    print(f"  - Notícias Analisadas: {len(result.get('news_data', []))}")
    twitter_data = result.get('twitter_sentiment', {})
    print(f"  - Tweets Analisados: {twitter_data.get('total_tweets', 0)}")
    if twitter_data.get('error'):
        print(f"  - Aviso Twitter: {twitter_data['error']}")
    print("-"*50)
    print(f"Relatório PDF: {result.get('report_path', 'Não gerado')}")
    print(f"Gráfico PNG: {result.get('price_chart', 'Não gerado')}")
    print("="*50 + "\n")

def main_streamlit():
    st.set_page_config(layout="wide", page_title="Cripto AI Analyzer")
    st.title("🤖 Cripto AI Analyzer")
    st.caption("Análise de criptomoedas usando dados técnicos, sentimento e IA.")

    # 1. Obter chaves de API
    api_keys = get_api_keys()

    # 2. Seleção de Parâmetros (só habilita se as chaves estiverem OK)
    st.sidebar.title("⚙️ Parâmetros de Análise")
    analysis_enabled = api_keys is not None
    
    # Inicializa o cliente Binance aqui para buscar os pares
    # Mesmo que as chaves não estejam salvas, tenta inicializar sem chaves para buscar pares
    temp_binance_client = BinanceTradingClient(
        api_key=api_keys.get('BINANCE_API_KEY') if api_keys else None,
        api_secret=api_keys.get('BINANCE_SECRET_KEY') if api_keys else None
    )
    available_pairs = temp_binance_client.get_all_usdt_pairs()
    if not available_pairs:
        st.sidebar.warning("Não foi possível buscar a lista de pares da Binance. Usando lista padrão.")
        available_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"] # Fallback

    # Define um valor padrão ou o primeiro da lista
    default_pair_index = 0
    if "BTCUSDT" in available_pairs:
        default_pair_index = available_pairs.index("BTCUSDT")
        
    selected_symbol = st.sidebar.selectbox(
        "Selecione o Par USDT:",
        options=available_pairs,
        index=default_pair_index,
        disabled=not analysis_enabled
    )
    selected_interval = st.sidebar.selectbox(
        "Selecione o Intervalo:",
        options=["15m", "1h", "4h", "1d"],
        index=1, # Default '1h'
        disabled=not analysis_enabled
    )
    selected_lookback = st.sidebar.selectbox(
        "Selecione o Período/Limite:",
        options=["100", "500", "1000", "7d", "30d"], # Opções comuns
        index=2, # Default '1000' candles
        disabled=not analysis_enabled
    )

    # 3. Botão de Análise
    if st.sidebar.button("🚀 Analisar Agora", disabled=not analysis_enabled, use_container_width=True):
        if not selected_symbol or not selected_interval or not selected_lookback:
            st.error("Por favor, selecione Símbolo, Intervalo e Período.")
        else:
            # Cria o workflow COM as chaves obtidas
            trading_app = create_trading_workflow(api_keys)
            
            inputs = {
                "symbol": selected_symbol,
                "interval": selected_interval,
                "lookback": selected_lookback
            }
            
            st.info(f"Iniciando análise para {selected_symbol} ({selected_interval}, {selected_lookback})... Isso pode levar alguns minutos.")
            progress_bar = st.progress(0, text="Coletando dados...")
            
            try:
                # Executa o workflow passo a passo para atualizar a barra de progresso (simulado)
                # Nota: LangGraph executa tudo de uma vez, a barra aqui é ilustrativa
                # Idealmente, usaríamos callbacks ou streaming se LangGraph suportasse nativamente
                final_result = None
                step_counter = 0
                total_steps = 5 # collect, tech, sentiment, decision, report
                
                # Simulação de progresso enquanto o app.invoke roda
                # Em um cenário real, isso seria mais complexo
                with st.spinner('Executando análise...'):
                    final_result = trading_app.invoke(inputs)
                    # Atualiza a barra após a conclusão (simulado)
                    progress_bar.progress(100, text="Análise concluída!")

                # 4. Exibir Resultados
                if final_result:
                    display_results_streamlit(final_result)
                else:
                    st.error("A análise não retornou resultados.")
                    
            except Exception as e:
                st.error(f"Ocorreu um erro durante a execução do workflow: {str(e)}")
                logging.error("Erro no workflow Streamlit:", exc_info=True)
                progress_bar.progress(100, text="Erro na análise.")
    elif not analysis_enabled:
        st.warning("Por favor, insira e salve as credenciais de API necessárias na barra lateral para habilitar a análise.")

# --------------------------------------------------
# 7. Ponto de Entrada Principal
# --------------------------------------------------
if __name__ == "__main__":
    # Verifica se o script está sendo executado com Streamlit
    # CORRIGIDO: Erro de sintaxe aqui (estava faltando :) e identação
    try:
        # Tenta importar Streamlit e verificar se está rodando
        import sys
        if "streamlit" in sys.modules and hasattr(st, 'spinner'): # Uma verificação simples
             logging.info("Executando em modo Streamlit.")
             main_streamlit()
        else:
             raise ImportError # Força a execução no modo terminal
    except ImportError:
        logging.info("Executando em modo Terminal (Streamlit não detectado ou não é o runner principal).")
        # Execução simples no terminal (exemplo)
        print("Execução em modo terminal iniciada.")
        print("Carregando chaves de API das variáveis de ambiente...")
        api_keys = get_api_keys() # Tenta carregar do ambiente
        
        if not api_keys or not api_keys.get('BINANCE_API_KEY') or not api_keys.get('OPENAI_API_KEY'):
            print("\nERRO: Chaves de API obrigatórias (BINANCE_*, OPENAI_API_KEY) não encontradas nas variáveis de ambiente.")
            print("Defina as variáveis de ambiente antes de executar.")
        else:
            print("Chaves carregadas. Criando workflow...")
            trading_app = create_trading_workflow(api_keys)
            
            # Exemplo de input para teste no terminal
            test_inputs = {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "lookback": "7d" # Usa 7 dias de dados
            }
            print(f"\nIniciando análise de teste para: {test_inputs['symbol']} ({test_inputs['interval']}, {test_inputs['lookback']})...")
            
            try:
                result = trading_app.invoke(test_inputs)
                if result:
                    display_results_terminal(result)
                else:
                    print("\nERRO: A análise não retornou resultados.")
            except Exception as e:
                print(f"\nERRO durante a execução do workflow no terminal: {str(e)}")
                logging.error("Erro no workflow Terminal:", exc_info=True)

        print("\nExecução em modo terminal concluída.")

# CORRIGIDO: Removido código solto após o if __name__ == "__main__":

