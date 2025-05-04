# 🤖 Agente CriptoAI

Este repositório apresenta o **Agente CriptoAI**, um agente financeiro de IA desenvolvido com Streamlit, focado na análise de criptomoedas. A solução simula um especialista que orienta o investidor a partir de dados técnicos e fundamentalistas do mercado cripto.

---

## 🔍 Visão Geral

O **Agente CriptoAI** é uma aplicação de IA projetada para realizar análises financeiras e recomendar estratégias com base em criptomoedas selecionadas. Utiliza técnicas de _prompt engineering_, dados em tempo real e agentes de linguagem para simular um consultor financeiro cripto.

O projeto combina:

- Interface interativa via **Streamlit**
- APIs de dados de criptomoedas
- Agente IA baseado em **LangGraph** e **OpenAI GPT-4** 
- Geração de relatórios técnicos e fundamentalistas
- Visualizações dinâmicas com Plotly

---

## ❓ Problema e Solução

### 🔧 Problema

Investidores iniciantes e intermediários enfrentam dificuldades em tomar decisões no volátil mercado de criptomoedas, seja por falta de tempo, conhecimento técnico ou fontes confiáveis de dados integrados.

### ✅ Solução

O agente automatiza a análise cripto ao:

- Coletar dados de mercado em tempo real
- Realizar análise técnica e fundamentalista
- Gerar gráficos interativos (candlestick, volume, RSI)
- Elaborar relatórios com recomendações de compra, venda ou manutenção

---

## ⚙️ Processo

A aplicação segue os seguintes passos:

1. **Input do Usuário**:
   - O usuário insere o símbolo da criptomoeda desejada (ex: `BTC-USD`) e um período de análise (ex: `1mo`).

2. **Coleta e Processamento de Dados**:
   - Coleta dados históricos via `yfinance`
   - Calcula indicadores técnicos como RSI
   - Gera visualizações com **Plotly**

3. **Agente IA com LangGraph**:
   - Criação de um grafo de estados com nós:
     - `analisador_tecnico`: interpreta os gráficos e dados técnicos
     - `analisador_fundamentalista`: considera contexto de mercado e notícias
     - `gerador_relatorio`: resume a análise em um relatório completo
   - A IA responde com insights personalizados em linguagem natural

4. **Exibição do Resultado**:
   - Interface em **Streamlit** exibe gráficos, relatórios e diagnósticos interativos

5. **Fluxo de Decisão dos Agentes**:
---

## 📊 Resultados

Ao rodar a aplicação, o usuário obtém:

- Gráficos candlestick com RSI e volume
- Relatórios de IA simulando um analista profissional
- Diagnóstico final da ação recomendada: 📈 Comprar, 🔻 Vender ou 🤝 Manter

Exemplo de output da IA:

> _"Analisando o RSI, candlestick e volumes recentes, a tendência indica uma pressão compradora com suporte em $27.000. Considerando a dominância do BTC no mercado, a recomendação é: **Manter posição** até sinais mais claros de reversão."_ 

---

## 🧠 Conclusões

O **Agente CriptoAI** se destaca como uma ferramenta de apoio à decisão, que:

- Democratiza o acesso a análises profissionais
- Integra IA com finanças e dados em tempo real
- Cria uma experiência interativa e personalizada para investidores

Este projeto é ideal para estudos, protótipos de aplicações financeiras com IA e demonstração de técnicas de integração entre Streamlit, LangGraph e agentes de IA.

---

## 🚀 Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
