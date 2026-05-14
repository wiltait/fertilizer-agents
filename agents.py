# ============================================================
# agents.py
# Define os 3 agentes do sistema de corretagem de fertilizantes
# Cada agente tem um papel, objetivo e histórico bem definidos
#
# COMPATÍVEL COM: CrewAI >= 0.80 | Python 3.11+
# ============================================================

import os
from crewai import Agent, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# --------------------------------------------------------------
# CONFIGURAÇÃO DO LLM (Large Language Model)
#
# MUDANÇA IMPORTANTE no CrewAI >= 0.80:
#   Não use mais ChatGoogleGenerativeAI do LangChain diretamente.
#   O CrewAI agora tem seu próprio wrapper `LLM` que usa LiteLLM
#   por baixo dos panos. O formato do modelo para Gemini é:
#   "gemini/gemini-1.5-flash"  (prefixo "gemini/" é obrigatório)
#
# SOLUÇÃO para erro 404 NOT_FOUND:
#   O LiteLLM (usado pelo CrewAI LLM) já aponta para o endpoint
#   correto do Gemini automaticamente. Não precisa de transport
#   nem client_options.
# --------------------------------------------------------------
def get_llm():
    """
    Inicializa e retorna o modelo Gemini usando o wrapper nativo do CrewAI.
    O prefixo 'gemini/' instrui o LiteLLM a usar a API correta do Google.
    """
    return LLM(
        model="gemini/gemini-2.5-flash",   # Formato obrigatório no CrewAI >= 0.80
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,                    # Baixo = mais factual (ideal para análises financeiras)
    )


# --------------------------------------------------------------
# FERRAMENTA DE BUSCA NA INTERNET
#
# MUDANÇA IMPORTANTE no CrewAI >= 0.80:
#   Ferramentas LangChain (DuckDuckGoSearchRun) não são mais
#   compatíveis diretamente com o CrewAI. A solução é criar uma
#   ferramenta nativa usando o decorator @tool do crewai_tools.
#   A biblioteca duckduckgo-search é usada diretamente.
# --------------------------------------------------------------
@tool("Busca na Internet DuckDuckGo")
def search_internet(query: str) -> str:
    """
    Busca informações na internet usando DuckDuckGo.
    Use para pesquisar preços de fertilizantes, fretes marítimos,
    câmbio USD/BRL, notícias de mercado e dados logísticos portuários.
    Recebe uma query em texto e retorna os resultados relevantes.
    """
    try:
        with DDGS() as ddgs:
            # Busca os 5 primeiros resultados e formata como texto
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "Nenhum resultado encontrado para a busca."

            # Formata os resultados de forma legível para o agente
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"[Resultado {i}]\n"
                    f"Título: {r.get('title', 'N/A')}\n"
                    f"Fonte: {r.get('href', 'N/A')}\n"
                    f"Resumo: {r.get('body', 'N/A')}\n"
                )
            return "\n".join(formatted)

    except Exception as e:
        return f"Erro na busca: {str(e)}. Tente reformular a query."


# ==============================================================
# AGENTE 1: ANALISTA DE MERCADO (News Scout)
# Responsabilidade: Rastrear preços e tendências de MAP e Ureia
# ==============================================================
def create_market_analyst():
    """
    Cria o agente Analista de Mercado.

    Este agente age como um analista especializado que monitora:
    - Preços FOB (Free On Board) nos mercados internacionais
    - Preços CFR/Spot no mercado brasileiro
    - Sentimento de mercado e notícias relevantes
    - Condições de oferta e demanda global
    """
    return Agent(
        role="Analista de Mercado de Fertilizantes",
        goal=(
            "Coletar e analisar dados atualizados de preços de MAP (Monoamônio Fosfato) e Ureia "
            "nos mercados internacionais (FOB) e no mercado interno brasileiro (CFR Santos/Paranaguá). "
            "Identificar tendências, sazonalidade agrícola e fatores que impactam os preços."
        ),
        backstory=(
            "Você é um analista sênior com 15 anos de experiência em commodities agrícolas, "
            "ex-trader da Yara International e Mosaic. Domina os mercados de Tampa (MAP), "
            "Yuzhne (Ureia) e os corredores de importação brasileiros. "
            "Você tem uma rede de contatos em tradings como Bunge, Cargill e Louis Dreyfus, "
            "e acompanha diariamente publicações como Argus Media e Green Markets."
        ),
        tools=[search_internet],           # Ferramenta nativa CrewAI para buscar preços na internet
        llm=get_llm(),
        verbose=True,                       # Mostra o raciocínio passo a passo no terminal
        allow_delegation=False,             # Este agente não delega tarefas para outros
        max_iter=5,                         # Máximo de iterações para evitar loops infinitos
    )


# ==============================================================
# AGENTE 2: ESTRATEGISTA DE ARBITRAGEM (Math Genius)
# Responsabilidade: Calcular paridade e janelas de arbitragem
# ==============================================================
def create_arbitrage_strategist():
    """
    Cria o agente Estrategista de Arbitragem.

    Este agente realiza os cálculos financeiros críticos:
    - Fórmula: Preço FOB + Frete Marítimo + Seguro + Tarifas + Margem Portuária
    - Compara o custo total de importação vs. preço spot no Brasil
    - Calcula a janela de arbitragem em USD/tonelada e percentual
    - Avalia viabilidade considerando tempo de trânsito e sazonalidade
    """
    return Agent(
        role="Estrategista de Arbitragem de Fertilizantes",
        goal=(
            "Calcular com precisão a paridade de importação de MAP e Ureia, "
            "determinando se existe uma janela de arbitragem lucrativa entre o mercado FOB internacional "
            "e o mercado spot/CFR brasileiro. Considerar todos os custos logísticos: "
            "frete marítimo, seguro (0.3%), tarifa de importação (0% para fertilizantes), "
            "THC (Terminal Handling Charge), fila nos portos de Santos e Paranaguá, e spread de câmbio USD/BRL."
        ),
        backstory=(
            "Você é um ex-gerente de risco da Fertipar e Heringer, com MBA em Finanças Quantitativas. "
            "Você construiu modelos de precificação usados por grandes tradings no Brasil. "
            "Conhece profundamente a estrutura de custos CIF (Cost, Insurance and Freight) brasileira, "
            "os períodos de safra (soja em Outubro-Dezembro, milho em Janeiro-Março) "
            "e como as filas em Santos e Paranaguá afetam o custo efetivo de importação. "
            "Seu lema é: 'Lucro está nos detalhes que outros ignoram.'"
        ),
        tools=[search_internet],           # Pode buscar fretes atualizados, câmbio, etc.
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


# ==============================================================
# AGENTE 3: AGENTE DE COMUNICAÇÃO (Ghostwriter)
# Responsabilidade: Redigir comunicações profissionais
# ==============================================================
def create_communication_agent():
    """
    Cria o agente de Comunicação.

    Este agente é o 'porta-voz' do negócio:
    - Escreve e-mails em Inglês para fornecedores internacionais (produtores, tradings)
    - Redige propostas em Português para compradores brasileiros (distribuidores, cooperativas)
    - Adapta o tom e linguagem para cada audiência
    - Garante terminologia técnica correta do mercado de fertilizantes
    """
    return Agent(
        role="Especialista em Comunicação Comercial de Fertilizantes",
        goal=(
            "Redigir comunicações comerciais profissionais e persuasivas: "
            "e-mails de prospecção e negociação em Inglês para fornecedores internacionais (Rússia, Marrocos, China, EUA), "
            "e propostas comerciais detalhadas em Português para compradores brasileiros "
            "(distribuidores regionais, cooperativas agrícolas, indústrias). "
            "Cada comunicação deve refletir o contexto de mercado e os cálculos de arbitragem."
        ),
        backstory=(
            "Você é um profissional bilíngue com experiência em trade finance e vendas internacionais de commodities. "
            "Já negociou contratos de fertilizantes com fornecedores da Rússia (EuroChem, PhosAgro), "
            "Marrocos (OCP Group) e China (Sinofert). "
            "No Brasil, desenvolveu relacionamento com cooperativas como Coamo e C.Vale, "
            "e distribuidores como Nutrien e Raízen. "
            "Você sabe que no mercado de fertilizantes, credibilidade e timing são tudo: "
            "uma proposta bem escrita no momento certo pode fechar negócios de milhões de dólares."
        ),
        tools=[],                           # Este agente foca em escrita; não precisa de busca
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )